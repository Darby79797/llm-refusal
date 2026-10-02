"""Weight orthogonalisation (Arditi et al. §4): directional ablation as a rank-one weight edit.

Also takes a [k, d_model] stack of directions and projects out their span (a rank-k
edit, W ← W − Q Qᵀ W for an orthonormal basis Q), for behaviors one direction
doesn't fully carry.

Every matrix that writes into the residual stream is replaced by its projection
onto the orthogonal complement of the direction r̂:

    token embedding E (rows are residual vectors):  E ← E − (E r̂) r̂ᵀ
    attention o_proj, MLP down_proj (out = W x + b): W ← W − r̂ (r̂ᵀ W),  b ← b − (b·r̂) r̂

The residual stream is a sum of those writes, so it then has no r̂ component
anywhere, which is what the 3-hook ablation in interventions.py enforces at run
time. In exact arithmetic the two are the same model; in bf16 they differ by
rounding (the edit is computed in fp32 and rounded once into the weights, the
hooks round the projection at every step).

Two architecture conditions for that equivalence, checked rather than assumed:
  - Tied embeddings (Qwen2.5 ≤ 3B tie lm_head to embed_tokens). Editing the
    embedding in place would also edit the unembedding, which ablation never
    touches. `orthogonalized` unties them for the duration (lm_head gets its own
    copy of the original matrix) and re-ties them on exit.
  - No norm between a sublayer's output and the residual add. Gemma-2/3 apply
    RMSNorm with an elementwise gain to each sublayer's output, which does not
    preserve orthogonality to r̂, so those models are refused.

Memory. The edit is applied in place and undone by reloading the edited tensors
from the model's own checkpoint (bf16 arithmetic can't invert it), with an exact
checksum of every restored tensor. Swapping in an edited copy instead holds both
versions, ~6 GB extra on an 8B model, which left too little room for lm-eval and
LoRA training beside it. The copy path remains for models with no checkpoint on
disk and for `prepare_edit` (the equivalence check re-enters the edit per batch).
"""
import contextlib
import json
import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple

import torch as t
import torch.nn as nn

logger = logging.getLogger(__name__)

# Rows of the embedding edited per step, bounding the fp32 working copy
# (Qwen's 152k-row embedding would be 2 GB at once in fp32 on the 7B).
_EMBED_CHUNK_ROWS = 16384
# Columns of an o_proj/down_proj edited per step in place (fp32 working copy of
# d_model x 4096: 64 MB on an 8B model).
_LINEAR_CHUNK_COLS = 4096

# Residual-stream writers: (module, kind) where kind is "embedding" or "linear".
Writers = List[Tuple[nn.Module, str]]


def residual_writers(model) -> Writers:
    """The embedding and each layer's attention and MLP output projections."""
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "embed_tokens") or not hasattr(inner, "layers"):
        raise NotImplementedError(
            f"Weight orthogonalisation supports Llama/Qwen2-style models "
            f"(model.model.embed_tokens/layers), not {model.__class__.__name__}.")
    writers: Writers = [(inner.embed_tokens, "embedding")]
    for i, block in enumerate(inner.layers):
        if hasattr(block, "post_feedforward_layernorm"):
            raise NotImplementedError(
                f"Layer {i} normalises sublayer outputs before the residual add "
                f"({block.__class__.__name__}); a rank-one edit of o_proj/down_proj does not "
                f"remove the direction from the residual stream there.")
        writers.append((block.self_attn.o_proj, "linear"))
        writers.append((block.mlp.down_proj, "linear"))
    return writers


def orthonormal_basis(directions: t.Tensor) -> t.Tensor:
    """[k, d] orthonormal rows spanning `directions` ([d] or [k, d]), in fp32."""
    d = directions.detach().float()
    if d.dim() == 1:
        return (d / d.norm()).unsqueeze(0)
    q, r = t.linalg.qr(d.T)  # d.T is [d_model, k]
    rank = int((r.diagonal().abs() > 1e-6 * r.diagonal().abs().max()).sum())
    if rank < d.shape[0]:
        raise ValueError(f"Directions are linearly dependent: rank {rank} < {d.shape[0]}")
    return q.T


def edit_bytes(model) -> int:
    """Memory `orthogonalized` holds beside the weights while active, for batch-size
    planning (batching.reserve_bytes). In place: the untied lm_head copy on a model
    with tied embeddings, plus the fp32 working chunk. Copy path: every edited tensor."""
    writers = residual_writers(model)
    if checkpoint_tensor_files(model, writers) is None:
        total = 0
        for module, kind in writers:
            total += module.weight.numel() * module.weight.element_size()
            if kind == "linear" and getattr(module, "bias", None) is not None:
                total += module.bias.numel() * module.bias.element_size()
        return total
    embed = writers[0][0].weight
    tied = _is_tied(model, embed)
    chunk = max(_EMBED_CHUNK_ROWS, _LINEAR_CHUNK_COLS) * embed.shape[1] * 4
    return (embed.numel() * embed.element_size() if tied else 0) + chunk


def _is_tied(model, embed_weight) -> bool:
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    return head is not None and head.weight is embed_weight


def _param_names(model) -> Dict[int, List[str]]:
    """Every name each Parameter is registered under (a tied embedding has two)."""
    names: Dict[int, List[str]] = {}
    for n, p in model.named_parameters(remove_duplicate=False):
        names.setdefault(id(p), []).append(n)
    return names


def checkpoint_tensor_files(model, writers: Optional[Writers] = None) -> Optional[Dict[str, str]]:
    """{parameter name: safetensors file} for every edited tensor, from the model's
    local checkpoint (a directory, or a cached Hub snapshot; never downloads). None if
    there is no such checkpoint or it lacks any of the tensors."""
    name = getattr(model, "name_or_path", None) or getattr(model.config, "_name_or_path", None)
    if not name:
        return None
    if os.path.isdir(name):
        root = name
    else:
        try:
            from huggingface_hub import snapshot_download
            root = snapshot_download(name, local_files_only=True,
                                     allow_patterns=["*.safetensors", "*.safetensors.index.json"])
        except Exception:
            return None
    index = os.path.join(root, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = {k: os.path.join(root, v) for k, v in json.load(f)["weight_map"].items()}
    elif os.path.exists(os.path.join(root, "model.safetensors")):
        from safetensors import safe_open
        path = os.path.join(root, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            files = {k: path for k in f.keys()}
    else:
        return None
    names = _param_names(model)
    wanted = {}
    for module, kind in writers or residual_writers(model):
        params = [module.weight] + ([module.bias] if kind == "linear" and getattr(module, "bias", None) is not None else [])
        for p in params:
            n = next((x for x in names.get(id(p), []) if x in files), None)
            if n is None:
                return None
            wanted[n] = files[n]
    return wanted


def _checkpoint_name(names: Dict[int, List[str]], files: Dict[str, str], p) -> str:
    return next(x for x in names[id(p)] if x in files)


def _bits_checksum(x: t.Tensor) -> int:
    """Exact fingerprint of a tensor's bits (sum of its raw integers)."""
    ints = {2: t.int16, 4: t.int32, 8: t.int64}[x.element_size()]
    return int(x.detach().contiguous().view(ints).to(t.int64).sum())


def _synchronize() -> None:
    if t.backends.mps.is_available():
        t.mps.synchronize()
    if t.cuda.is_available():
        t.cuda.synchronize()


def _release_cached_memory() -> None:
    if t.backends.mps.is_available():
        t.mps.empty_cache()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def _orthogonalize_(weight: t.Tensor, basis: t.Tensor, kind: str) -> None:
    """In place: project span(basis) out of `weight`'s residual-stream side, in chunks.
    Same fp32 arithmetic per element as _orthogonalized_weight, so the same result."""
    q = basis.to(device=weight.device)
    if kind == "embedding":  # [vocab, d_model]: rows
        for start in range(0, weight.shape[0], _EMBED_CHUNK_ROWS):
            w = weight[start:start + _EMBED_CHUNK_ROWS].float()
            weight[start:start + _EMBED_CHUNK_ROWS] = (w - (w @ q.T) @ q).to(weight.dtype)
    else:  # [d_model (out), d_in]: columns are independent
        for start in range(0, weight.shape[1], _LINEAR_CHUNK_COLS):
            w = weight[:, start:start + _LINEAR_CHUNK_COLS].float()
            weight[:, start:start + _LINEAR_CHUNK_COLS] = (w - q.T @ (q @ w)).to(weight.dtype)


def _orthogonalized_weight(weight: t.Tensor, basis: t.Tensor, kind: str) -> t.Tensor:
    """A new tensor: `weight` with span(basis) projected out of its residual-stream side.
    The same chunked routine as the in-place edit, so both paths give identical weights
    (MPS's fp32 matmuls round differently for different shapes)."""
    out = weight.detach().clone()
    _orthogonalize_(out, basis, kind)
    return out


def _orthogonalized_bias(bias: t.Tensor, basis: t.Tensor) -> t.Tensor:
    b = bias.float()
    q = basis.to(device=bias.device)
    return (b - (q @ b) @ q).to(bias.dtype)


def prepare_edit(model, direction: t.Tensor) -> List[Tuple[nn.Module, str, nn.Parameter]]:
    """The edited Parameters for `orthogonalized`, built once: (module, attr, new Parameter).

    Pass the result as `prepared=` to re-enter the edit without rebuilding it (1.9 s a
    time on Llama-3-8B; scripts/ortho_equivalence.py enters it once per batch). Holding
    it keeps the edited copy (edit_bytes) allocated between uses.
    """
    basis = orthonormal_basis(direction)
    edits = []
    with t.no_grad():
        for module, kind in residual_writers(model):
            edits.append((module, "weight", nn.Parameter(_orthogonalized_weight(module.weight, basis, kind),
                                                         requires_grad=False)))
            if kind == "linear" and getattr(module, "bias", None) is not None:
                edits.append((module, "bias", nn.Parameter(_orthogonalized_bias(module.bias, basis),
                                                           requires_grad=False)))
    logger.info(f"Orthogonalised {len(edits)} residual-stream tensors against a rank-{basis.shape[0]} subspace")
    return edits


@contextlib.contextmanager
def orthogonalized(model, direction: t.Tensor, prepared=None) -> Iterator[None]:
    """Run the model with `direction` ([d], or [k, d] for a subspace) orthogonalised
    out of every residual writer, restoring the original weights exactly on exit.

    In place, restored from the model's checkpoint, when it has one on disk (see the
    module docstring); otherwise, or with `prepared` (from prepare_edit), by swapping
    in edited copies, which costs one extra copy of the edited matrices while active.
    """
    if prepared is None:
        writers = residual_writers(model)
        files = checkpoint_tensor_files(model, writers)
        if files is not None:
            with _orthogonalized_in_place(model, direction, writers, files):
                yield
            return
        logger.info("No local checkpoint to restore from: orthogonalising a copy of the weights")
    edits = prepared if prepared is not None else prepare_edit(model, direction)
    saved = []  # (module, name, original Parameter)
    try:
        for module, name, param in edits:
            saved.append((module, name, getattr(module, name)))
            setattr(module, name, param)
        yield
    finally:
        for module, name, param in reversed(saved):
            setattr(module, name, param)


@contextlib.contextmanager
def _orthogonalized_in_place(model, direction, writers: Writers, files: Dict[str, str]) -> Iterator[None]:
    basis = orthonormal_basis(direction)
    names = _param_names(model)
    targets = []  # (parameter, name, kind)
    for module, kind in writers:
        targets.append((module.weight, _checkpoint_name(names, files, module.weight), kind))
        if kind == "linear" and getattr(module, "bias", None) is not None:
            targets.append((module.bias, _checkpoint_name(names, files, module.bias), "bias"))
    before = {n: _bits_checksum(p) for p, n, _ in targets}
    embed_module = writers[0][0]
    head = model.get_output_embeddings()
    tied = _is_tied(model, embed_module.weight)
    try:
        with t.no_grad():
            if tied:  # lm_head keeps the unedited matrix (ablation never touches it)
                head.weight = nn.Parameter(embed_module.weight.detach().clone(), requires_grad=False)
            for p, _, kind in targets:
                if kind == "bias":
                    p.copy_(_orthogonalized_bias(p, basis))
                else:
                    _orthogonalize_(p.data, basis, kind)
        _release_cached_memory()   # the fp32 working chunks, so the edit holds nothing extra
        logger.info(f"Orthogonalised {len(targets)} residual-stream tensors in place "
                    f"against a rank-{basis.shape[0]} subspace")
        yield
    finally:
        # Every edit kernel must have finished before restored data lands.
        _synchronize()
        with t.no_grad():
            _restore_from_checkpoint([(p, n) for p, n, _ in targets], files)
            if tied:
                head.weight = embed_module.weight
        bad = [(p, n) for p, n, _ in targets if _bits_checksum(p) != before[n]]
        if bad:
            # Restoring is idempotent: re-read the mismatched tensors once before giving up.
            logger.warning(f"{len(bad)} restored tensors didn't match their pre-edit checksum "
                           f"(e.g. {bad[0][1]}); re-reading them from the checkpoint")
            with t.no_grad():
                _restore_from_checkpoint(bad, files)
            bad = [(p, n) for p, n in bad if _bits_checksum(p) != before[n]]
        if bad:
            raise RuntimeError(f"Restoring the orthogonalised weights from the checkpoint did not reproduce "
                               f"them exactly ({len(bad)} tensors differ, e.g. {bad[0][1]}); the in-memory model "
                               f"no longer matches its checkpoint")


def _restore_from_checkpoint(items: List[Tuple[nn.Parameter, str]], files: Dict[str, str]) -> None:
    from safetensors import safe_open
    by_file: Dict[str, List[Tuple[nn.Parameter, str]]] = {}
    for p, n in items:
        by_file.setdefault(files[n], []).append((p, n))
    for path, group in by_file.items():
        with safe_open(path, framework="pt") as f:
            for p, n in group:
                # The same cast the model got at load time (e.g. fp16 -> bf16).
                p.copy_(f.get_tensor(n).to(p.dtype))
            # The source is memory-mapped from this file: let any in-flight
            # host-to-device copy finish before it closes.
            _synchronize()
