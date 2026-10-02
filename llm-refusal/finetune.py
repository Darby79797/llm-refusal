"""Fine-tuning experiments around the refusal direction r̂ (PROPOSED_PLANS items 2 and 5).

`--mode rank1`: does gradient descent rediscover r̂? A rank-one adapter
ΔW = u vᵀ (by default on the down_proj of the layer just before r̂'s layer, i.e.
writing straight into the residual r̂ is read from) is trained by
self-distillation to reproduce directional ablation: harmful prompts → the
ablated model's own completions, harmless prompts → the clean model's own
completions (retain set, so the adapter can't just learn "change everything").
`--objective induce` trains the reverse (harmless → the addition-steered
model's refusals). Then: cos(u, r̂), and the behavioural effect of the adapter
as trained, with r̂ projected out of u, and with u replaced by its r̂ component.
u starts at zero and v at random (LoRA init), so u is not biased toward r̂.

`--mode regrow`: does refusal come back after the edit? The whole run is inside
`orthogonalized(model, r̂)`. A rank-r LoRA is trained on benign Alpaca data,
optionally mixed with harmful → refusal examples (the clean model's own
refusals, generated before the edit), on either the residual *writers*
(o_proj/down_proj, which can re-insert r̂) or only the *readers*
(q/k/v/gate/up, where the residual stays orthogonal to r̂ by construction).
Refusal is tracked during training; at the end the direction is re-extracted
from the fine-tuned model and compared with r̂.

Both write results/finetune/<model>-<concept>-<mode>-<tag>.json and the adapter
weights beside it.
"""
import collections
import contextlib
import json
import logging
import os
import random
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch as t
import torch.nn as nn

from batching import _empty_cache, is_oom_error
from capability import load_alpaca
from datatypes import PromptData
from orthogonalize import edit_bytes, orthogonalized

logger = logging.getLogger(__name__)

WRITERS = ("o_proj", "down_proj")
READERS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
TARGET_TOKENS = 64
# Padded tokens per micro-batch row: a micro-batch is also split when rows x padded
# width exceeds micro_batch_size x this. Alpaca rows are 64 tokens median, 151 at the
# 99th percentile, 314 at most; at 2 rows of 314 the regrow runs ran out of memory on
# 8B (every run at step 144, which holds that row).
MICRO_TOKENS_PER_ROW = 128


class LowRankAdapter(nn.Module):
    """y = base(x) + U (Vᵀ x): a trainable rank-r update on a frozen nn.Linear.

    U [out, r] starts at zero and V [in, r] at small random values (LoRA's init),
    so training starts from the unmodified model. The adapter is kept in fp32 and
    its output cast to the activation dtype.
    """

    def __init__(self, base: nn.Linear, rank: int, generator: t.Generator):
        super().__init__()
        self.base = base
        self.U = nn.Parameter(t.zeros(base.out_features, rank, device=base.weight.device))
        v = t.randn(base.in_features, rank, generator=generator) / base.in_features ** 0.5
        self.V = nn.Parameter(v.to(base.weight.device))
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if not self.enabled:
            return out
        return out + ((x.float() @ self.V) @ self.U.T).to(out.dtype)


def adapter_sites(model, layers: Sequence[int], names: Sequence[str]) -> List[Tuple[nn.Module, str]]:
    """(parent module, attribute) for each named projection in each layer."""
    sites = []
    for i in layers:
        block = model.model.layers[i]
        for name in names:
            parent = block.self_attn if name in ("q_proj", "k_proj", "v_proj", "o_proj") else block.mlp
            if not isinstance(getattr(parent, name, None), nn.Linear):
                raise ValueError(f"layer {i} has no nn.Linear {name}")
            sites.append((parent, name))
    return sites


@contextlib.contextmanager
def adapted(model, sites: List[Tuple[nn.Module, str]], rank: int, seed: int) -> Iterator[List[LowRankAdapter]]:
    """Install adapters at `sites`; restore the original modules on exit."""
    for p in model.parameters():
        p.requires_grad_(False)
    g = t.Generator().manual_seed(seed)
    installed = []
    try:
        for parent, name in sites:
            adapter = LowRankAdapter(getattr(parent, name), rank, g)
            setattr(parent, name, adapter)
            installed.append((parent, name, adapter))
        yield [a for _, _, a in installed]
    finally:
        for parent, name, adapter in reversed(installed):
            setattr(parent, name, adapter.base)


def train_adapters(model, prompt_formatter, adapters: List[LowRankAdapter], examples: List[Tuple[str, str]],
                   steps: int, lr: float, batch_size: int, seed: int, eval_fn=None, eval_every: int = 0,
                   micro_batch_size: Optional[int] = None) -> List[Dict]:
    """Adam on completion-token CE over `examples` (prompt, completion). Returns a log.

    Each step's batch is run in micro-batches of `micro_batch_size` rows with gradients
    accumulated, weighting each micro-batch by its share of the batch's scored tokens,
    so the update is the full batch's token-mean CE whatever the split. Activations
    for the backward pass scale with the micro-batch: on Llama-3-8B a batch of 8 needs
    ~6.7 GB, which with orthogonalized()'s 5.9 GB weight copy exceeds MPS's cap.
    Micro-batches are also capped at micro x MICRO_TOKENS_PER_ROW padded tokens
    (always at least one row), so one long example can't double the activations.

    Out of memory: only the failing micro-batch is retried, split in half; finished
    ones are kept, since a micro-batch's gradient is committed only after its backward
    completes. A single row that still doesn't fit is skipped for that step, with a
    warning, and the step's update is the token-mean over the rows used. The history
    records `oom_splits` and `skipped_rows` for any step where this happened.
    """
    params = [p for a in adapters for p in (a.U, a.V)]
    opt = t.optim.Adam(params, lr=lr)
    rng = random.Random(seed)
    order: List[int] = []
    history = []
    micro = micro_batch_size or batch_size
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    head = model.get_output_embeddings()
    dev = model.device
    model.train(False)  # no dropout; gradients still flow to the adapters
    if eval_fn is not None and eval_every:
        history.append({"step": 0, **eval_fn()})
    start = time.time()
    for step in range(1, steps + 1):
        if len(order) < batch_size:
            order += rng.sample(range(len(examples)), len(examples))
        batch = [examples[i] for i in order[:batch_size]]
        del order[:batch_size]
        enc = prompt_formatter.format_with_completions([p for p, _ in batch], [c for _, c in batch],
                                                       max_completion_tokens=TARGET_TOKENS)
        lengths = enc['attention_mask'].sum(-1).tolist()
        # Gradients are committed per micro-batch, only once its backward has finished:
        # an OOM part-way through leaves the committed sum untouched, so each row counts
        # exactly once, and only the failed micro-batch is retried (split in half). A row
        # that doesn't fit on its own is skipped with a warning.
        committed = [t.zeros_like(p) for p in params]
        pending = collections.deque(_micro_batches(lengths, micro, micro * MICRO_TOKENS_PER_ROW))
        total, used_tokens, splits, skipped = 0.0, 0, 0, []
        while pending:
            rows = pending.popleft()
            for p in params:
                p.grad = None
            try:
                nll, n_tok = _forward_backward(base, head, enc, rows, dev)
            except Exception as e:
                if not is_oom_error(e):
                    raise
                nll = None
            if nll is None:
                # Outside the except block, so the failed pass's tensors (kept alive by
                # the traceback until then) are actually released.
                for p in params:
                    p.grad = None
                _empty_cache()
                if rows.stop - rows.start == 1:
                    skipped.append(rows.start)
                    logger.warning(f"  step {step}: row {rows.start} ({lengths[rows.start]} tokens) runs out of "
                                   f"memory on its own; skipping it in this step")
                else:
                    mid = (rows.start + rows.stop) // 2
                    pending.extendleft([slice(mid, rows.stop), slice(rows.start, mid)])
                    splits += 1
                    logger.warning(f"  step {step}: out of memory on rows {rows.start}-{rows.stop - 1}; "
                                   f"splitting them in half (finished micro-batches are kept)")
                continue
            for c, p in zip(committed, params):
                if p.grad is not None:
                    c += p.grad
            total += nll
            used_tokens += n_tok
        if used_tokens == 0:
            logger.warning(f"  step {step}: every row was skipped; no update this step")
            loss = float('nan')
        else:
            # Token-mean over the rows actually used.
            for c, p in zip(committed, params):
                p.grad = c / used_tokens
            opt.step()
            loss = total / used_tokens
        entry = {"step": step, "loss": loss}
        if splits:
            entry["oom_splits"] = splits
        if skipped:
            entry["skipped_rows"] = len(skipped)
        if step % 10 == 0 or step == steps:
            logger.info(f"  step {step}/{steps}  loss {loss:.4f}  ({time.time() - start:.0f}s)")
        if eval_fn is not None and eval_every and (step % eval_every == 0 or step == steps):
            entry.update(eval_fn())
            logger.info(f"  step {step}: {entry}")
        history.append(entry)
    return history


def _forward_backward(base, head, enc, rows: slice, dev) -> Tuple[float, int]:
    """Forward + backward of the summed completion NLL for `rows`; returns (NLL sum,
    scored tokens). Its own function so a failed pass's tensors die with its frame."""
    width = int(enc['attention_mask'][rows].sum(-1).max())  # trim this micro-batch's padding
    hidden = base(input_ids=enc['input_ids'][rows, :width].to(dev),
                  attention_mask=enc['attention_mask'][rows, :width].to(dev),
                  position_ids=enc['position_ids'][rows, :width].to(dev), use_cache=False).last_hidden_state
    targets = enc['labels'][rows, 1:width].to(dev)
    scored = targets != -100
    # Unembed only the scored (completion) positions.
    nll_sum = t.nn.functional.cross_entropy(head(hidden[:, :-1][scored]).float(), targets[scored], reduction='sum')
    nll_sum.backward()
    return nll_sum.item(), int(scored.sum())


def _micro_batches(lengths: List[int], max_rows: int, max_tokens: int) -> List[slice]:
    """Consecutive row slices of at most `max_rows` rows whose padded size (rows x
    longest row) stays within `max_tokens`; a single row is always allowed."""
    out, start = [], 0
    while start < len(lengths):
        end = start + 1
        while (end < len(lengths) and end - start < max_rows
               and (end + 1 - start) * max(lengths[start:end + 1]) <= max_tokens):
            end += 1
        out.append(slice(start, end))
        start = end
    return out


def _generate_with(framework, prompts: List[str], intervention: Optional[Tuple[str, List[int]]], direction):
    applier = framework.intervention_applier
    ctx = (applier.intervened(direction, intervention[0], 1.0, layers=intervention[1])
           if intervention is not None else contextlib.nullcontext())
    with ctx:
        return framework.evaluator.generate_responses(prompts, max_new_tokens=TARGET_TOKENS)


def _behaviour(framework, prompts: List[str], key: str) -> Dict[str, float]:
    """Detection rate and log-odds on `prompts` under the model as it currently is."""
    ev = framework.evaluator
    texts = ev.generate_responses(prompts, max_new_tokens=TARGET_TOKENS)
    return {f"{key}_rate": sum(map(ev._check_for_detection, texts)) / len(texts),
            f"{key}_log_odds": ev._log_odds_metric(prompts)}


def _save(framework, mode: str, tag: str, results: Dict, adapters: List[LowRankAdapter]) -> str:
    out_dir = os.path.join("results", "finetune")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{framework.model_short}-{framework.concept.name}-{mode}"
                                 + (f"-{tag}" if tag else ""))
    with open(stem + ".json", "w") as f:
        json.dump(results, f, indent=1)
    t.save([{"U": a.U.detach().cpu(), "V": a.V.detach().cpu()} for a in adapters], stem + "-adapters.pt")
    logger.info(f"Saved {stem}.json (+ -adapters.pt)")
    return stem + ".json"


def _cos(a: t.Tensor, b: t.Tensor) -> float:
    return float(t.nn.functional.cosine_similarity(a.float().cpu().flatten(), b.float().cpu().flatten(), dim=0))


def run_rank1(framework, direction, train_pos: List[str], train_neg: List[str],
              eval_pos: List[str], eval_neg: List[str], config: Dict) -> Dict:
    model, n_layers = framework.model, len(framework.intervention_applier.transformer_layers)
    objective = config.get('objective', 'remove')
    layers = config.get('adapter_layers') or [direction.layer - 1]
    if layers == ["all"] or layers == "all":
        layers = list(range(direction.layer))
    names = config.get('adapter_modules') or ["down_proj"]
    seed = config.get('seed', 0)
    r_hat = direction.unit.float().cpu()

    # Targets: the model's own text under the intervention being distilled.
    if objective == "remove":
        steered = _generate_with(framework, train_pos, ("ablate", list(range(n_layers))), direction)
        retain = _generate_with(framework, train_neg, None, direction)
        examples = list(zip(train_pos, steered)) + list(zip(train_neg, retain))
    elif objective == "induce":
        steered = _generate_with(framework, train_neg, ("add", [direction.layer]), direction)
        retain = _generate_with(framework, train_pos, None, direction)
        examples = list(zip(train_neg, steered)) + list(zip(train_pos, retain))
    else:
        raise ValueError(f"objective must be 'remove' or 'induce', not {objective!r}")
    ev = framework.evaluator
    target_rate = sum(map(ev._check_for_detection, steered)) / len(steered)
    logger.info(f"rank1/{objective}: {len(examples)} examples; steered targets detected {target_rate:.0%}")

    def evaluate():
        return {**_behaviour(framework, eval_pos, "pos"), **_behaviour(framework, eval_neg, "neg")}

    results = {"model": framework.model_name, "concept": framework.concept.name, "objective": objective,
               "direction": {"layer": direction.layer, "position_index": direction.position_index},
               "adapter_layers": layers, "adapter_modules": names, "rank": 1, "seed": seed,
               "steps": config['train_steps'], "lr": config['lr'], "train_batch_size": config['train_batch_size'],
               "n_examples": len(examples), "steered_target_detection_rate": target_rate}
    results["before"] = evaluate()
    sites = adapter_sites(model, layers, names)
    with adapted(model, sites, rank=1, seed=seed) as adapters:
        results["history"] = train_adapters(model, framework.prompt_formatter, adapters, examples,
                                            config['train_steps'], config['lr'], config['train_batch_size'], seed,
                                            micro_batch_size=config.get('train_micro_batch_size'))
        results["after"] = evaluate()
        results["adapters"] = [{"layer": layers[i // len(names)], "module": names[i % len(names)],
                                "u_norm": float(a.U.norm()), "v_norm": float(a.V.norm()),
                                "cos_u_rhat": _cos(a.U[:, 0], r_hat)} for i, a in enumerate(adapters)]
        logger.info(f"cos(u, r̂): {[round(x['cos_u_rhat'], 3) for x in results['adapters']]}")
        # Causal test of the learned write direction: remove r̂ from u, then keep only r̂.
        trained = [a.U.detach().clone() for a in adapters]
        try:
            for a, u in zip(adapters, trained):
                r = r_hat.to(u.device)
                a.U.data = u - t.outer(r, r @ u)
            results["after_u_minus_rhat"] = evaluate()
            for a, u in zip(adapters, trained):
                r = r_hat.to(u.device)
                a.U.data = t.outer(r, r @ u)
            results["after_u_rhat_only"] = evaluate()
        finally:
            for a, u in zip(adapters, trained):
                a.U.data = u
        path = _save(framework, "rank1", config.get('run_tag') or objective, results, adapters)
    results["path"] = path
    return results


def run_regrow(framework, direction, train_pos: List[str], train_neg: List[str],
               eval_pos: List[str], eval_neg: List[str], config: Dict) -> Dict:
    model = framework.model
    n_layers = len(framework.intervention_applier.transformer_layers)
    arm = config.get('regrow_targets', 'writers')
    names = WRITERS if arm == "writers" else READERS if arm == "readers" else None
    if names is None:
        raise ValueError(f"regrow_targets must be 'writers' or 'readers', not {arm!r}")
    n_refusal = config.get('n_refusal_examples', 0)
    seed, rank = config.get('seed', 0), config.get('lora_rank', 8)
    r_hat = direction.unit.float().cpu()
    eval_subset = eval_pos[:config.get('regrow_eval_n', 50)]
    # The whole run holds the edited weight copy; plan generation batches around it.
    framework.evaluator.reserve_bytes = edit_bytes(model)

    # Refusal examples come from the clean model, before the edit.
    refusal_prompts = random.Random(seed).sample(train_pos, min(n_refusal, len(train_pos)))
    refusals = _generate_with(framework, refusal_prompts, None, direction) if refusal_prompts else []
    benign = [(a["instruction"], a["output"]) for a in load_alpaca("train")]
    examples = benign + list(zip(refusal_prompts, refusals))
    logger.info(f"regrow/{arm}: {len(benign)} benign + {len(refusals)} refusal examples, LoRA rank {rank}")

    def evaluate():
        return _behaviour(framework, eval_subset, "pos")

    results = {"model": framework.model_name, "concept": framework.concept.name,
               "direction": {"layer": direction.layer, "position_index": direction.position_index},
               "targets": arm, "modules": list(names), "rank": rank, "seed": seed,
               "n_benign": len(benign), "n_refusal_examples": len(refusals),
               "steps": config['train_steps'], "lr": config['lr'], "train_batch_size": config['train_batch_size'],
               "eval_n": len(eval_subset)}
    results["clean"] = evaluate()
    with orthogonalized(model, direction.vector):
        sites = adapter_sites(model, range(n_layers), names)
        with adapted(model, sites, rank=rank, seed=seed) as adapters:
            results["history"] = train_adapters(
                model, framework.prompt_formatter, adapters, examples, config['train_steps'], config['lr'],
                config['train_batch_size'], seed, eval_fn=evaluate, eval_every=config.get('eval_every', 50),
                micro_batch_size=config.get('train_micro_batch_size'))
            results["final_negative"] = _behaviour(framework, eval_neg[:len(eval_subset)], "neg")
            # Where does any regrown refusal live? Re-extract the direction at the same
            # coordinates, from the same train prompts, with the fine-tuned (still
            # orthogonalised) model.
            finder = framework.finder
            data = PromptData(train_pos + train_neg, [True] * len(train_pos) + [False] * len(train_neg))
            vecs = finder.direction_finder_method.compute_difference_vectors(data, max_positions=-direction.position_index)
            new = vecs[(direction.layer, direction.position_index)]
            results["regrown_direction"] = {"norm": float(new.norm()), "cos_with_rhat": _cos(new, r_hat),
                                            "clean_norm": float(direction.vector.norm())}
            logger.info(f"Re-extracted direction: {results['regrown_direction']}")
            path = _save(framework, "regrow", config.get('run_tag') or f"{arm}-r{n_refusal}", results, adapters)
    results["path"] = path
    return results
