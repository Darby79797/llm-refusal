"""Weight orthogonalisation (orthogonalize.py) against the hook ablation it replaces.

Tiny randomly initialised models in fp32 on CPU: no checkpoint download, so these
run in the lightweight suite.
"""
import pytest
import torch as t
from transformers import (Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM)

from datatypes import DirectionVector
from interventions import ModelInterventionApplier
from orthogonalize import orthogonalized

_SIZES = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)


def _tiny(kind):
    t.manual_seed(0)
    if kind == "llama":
        model = LlamaForCausalLM(LlamaConfig(**_SIZES))
    else:  # Qwen2.5 <= 3B: q/k/v biases and a tied lm_head
        model = Qwen2ForCausalLM(Qwen2Config(**_SIZES, tie_word_embeddings=True))
    return model.float().eval()


def _logits(model, ids):
    with t.no_grad():
        return model(input_ids=ids).logits


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_orthogonalized_matches_hook_ablation(kind):
    model = _tiny(kind)
    ids = t.randint(0, _SIZES["vocab_size"], (2, 12))
    direction = DirectionVector(vector=t.randn(_SIZES["hidden_size"]), layer=1, position_index=-1, score=0)

    applier = ModelInterventionApplier(model)
    applier.apply_direction_intervention(direction, "ablate", layers=None)
    try:
        hooked = _logits(model, ids)
    finally:
        applier.clear_interventions()
    with orthogonalized(model, direction.vector):
        edited = _logits(model, ids)

    assert t.allclose(edited, hooked, atol=1e-4), (edited - hooked).abs().max()
    # ...and the edit is not a no-op
    assert (edited - _logits(model, ids)).abs().max() > 1e-2


def test_orthogonalized_residual_has_no_direction_component():
    model = _tiny("llama")
    u = t.randn(_SIZES["hidden_size"])
    u = u / u.norm()
    with orthogonalized(model, u):
        with t.no_grad():
            out = model(input_ids=t.randint(0, _SIZES["vocab_size"], (1, 8)), output_hidden_states=True)
    # Every residual-stream state except the last (which has passed the final norm).
    for h in out.hidden_states[:-1]:
        assert (h @ u).abs().max() < 1e-4


def test_orthogonalized_restores_weights_exactly_and_keeps_lm_head():
    model = _tiny("qwen2")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    lm_head = model.lm_head.weight.clone()
    with orthogonalized(model, t.randn(_SIZES["hidden_size"])):
        # Tied embeddings: the edit must not reach the unembedding.
        assert t.equal(model.lm_head.weight, lm_head)
        assert not t.equal(model.model.embed_tokens.weight, lm_head)
    after = model.state_dict()
    assert all(t.equal(before[k], after[k]) for k in before)
    assert model.lm_head.weight is model.model.embed_tokens.weight  # still tied


def test_orthogonalized_restores_after_an_exception():
    model = _tiny("llama")
    before = model.model.layers[0].mlp.down_proj.weight
    with pytest.raises(RuntimeError):
        with orthogonalized(model, t.randn(_SIZES["hidden_size"])):
            raise RuntimeError("boom")
    assert model.model.layers[0].mlp.down_proj.weight is before


def test_orthogonalized_refuses_post_sublayer_norms():
    model = Gemma2ForCausalLM(Gemma2Config(**_SIZES, head_dim=16))
    with pytest.raises(NotImplementedError, match="normalises sublayer outputs"):
        with orthogonalized(model, t.randn(_SIZES["hidden_size"])):
            pass


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_rank_k_orthogonalized_matches_stacked_hook_ablations(kind):
    """Projecting out a k-dim span equals ablating each vector of an orthonormal basis of it."""
    from orthogonalize import orthonormal_basis
    model = _tiny(kind)
    ids = t.randint(0, _SIZES["vocab_size"], (2, 12))
    directions = t.randn(3, _SIZES["hidden_size"])
    basis = orthonormal_basis(directions)
    assert t.allclose(basis @ basis.T, t.eye(3), atol=1e-5)

    applier = ModelInterventionApplier(model)
    for q in basis:
        applier.apply_direction_intervention(
            DirectionVector(vector=q, layer=0, position_index=-1, score=0), "ablate", layers=None)
    try:
        hooked = _logits(model, ids)
    finally:
        applier.clear_interventions()
    with orthogonalized(model, directions):  # the un-orthonormalised stack
        edited = _logits(model, ids)
    assert t.allclose(edited, hooked, atol=1e-4), (edited - hooked).abs().max()


def test_rank_k_rejects_dependent_directions():
    v = t.randn(_SIZES["hidden_size"])
    with pytest.raises(ValueError, match="linearly dependent"):
        with orthogonalized(_tiny("llama"), t.stack([v, 2 * v])):
            pass


def test_prepared_edit_is_reusable_and_restores():
    from orthogonalize import prepare_edit
    model = _tiny("qwen2")
    ids = t.randint(0, _SIZES["vocab_size"], (2, 12))
    v = t.randn(_SIZES["hidden_size"])
    clean = _logits(model, ids)
    with orthogonalized(model, v):
        fresh = _logits(model, ids)
    edit = prepare_edit(model, v)
    for _ in range(2):
        with orthogonalized(model, v, prepared=edit):
            assert t.equal(_logits(model, ids), fresh)
        assert t.equal(_logits(model, ids), clean)


# ---- In place, restored from the checkpoint (the default when one is on disk) ----

def _saved(kind, tmp_path, save_dtype=t.float32, load_dtype=t.float32):
    """A tiny model written to disk as safetensors and loaded back, so it has a checkpoint."""
    src = _tiny(kind).to(save_dtype)
    src.save_pretrained(tmp_path / kind, safe_serialization=True)
    cls = LlamaForCausalLM if kind == "llama" else Qwen2ForCausalLM
    model = cls.from_pretrained(tmp_path / kind, dtype=load_dtype).eval()
    from orthogonalize import checkpoint_tensor_files
    assert checkpoint_tensor_files(model) is not None, "would silently take the copy path"
    return model


@pytest.mark.parametrize("kind", ["llama", "qwen2"])
def test_in_place_matches_copy_and_hooks_and_restores_exactly(kind, tmp_path):
    from orthogonalize import prepare_edit
    model = _saved(kind, tmp_path)
    ids = t.randint(0, _SIZES["vocab_size"], (2, 12))
    v = t.randn(_SIZES["hidden_size"])
    before = {k: x.clone() for k, x in model.state_dict().items()}
    params_before = {n: p for n, p in model.named_parameters(remove_duplicate=False)}
    with orthogonalized(model, v, prepared=prepare_edit(model, v)):    # copy path
        copy_logits = _logits(model, ids)
    with orthogonalized(model, v):                                      # in place
        # No new tensors: the same Parameter objects, edited.
        assert all(p is params_before[n] for n, p in model.named_parameters(remove_duplicate=False)
                   if n != "lm_head.weight")
        in_place = _logits(model, ids)
    assert t.equal(in_place, copy_logits)
    applier = ModelInterventionApplier(model)
    applier.apply_direction_intervention(DirectionVector(vector=v, layer=0, position_index=-1, score=0),
                                         "ablate", layers=None)
    try:
        assert t.allclose(in_place, _logits(model, ids), atol=1e-4)
    finally:
        applier.clear_interventions()
    after = model.state_dict()
    assert all(t.equal(before[k], after[k]) for k in before)


def test_in_place_unties_and_reties_embeddings(tmp_path):
    model = _saved("qwen2", tmp_path)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    head = model.lm_head.weight.clone()
    with orthogonalized(model, t.randn(_SIZES["hidden_size"])):
        assert t.equal(model.lm_head.weight, head)                       # unembedding untouched
        assert not t.equal(model.model.embed_tokens.weight, head)        # embedding edited
    assert model.lm_head.weight is model.model.embed_tokens.weight     # tied again


def test_in_place_restores_an_upcast_checkpoint_exactly(tmp_path):
    """Llama-2 ships fp16 and runs in bf16 on MPS: the restore must apply the same cast."""
    model = _saved("llama", tmp_path, save_dtype=t.float16, load_dtype=t.bfloat16)
    before = {k: x.clone() for k, x in model.state_dict().items()}
    with orthogonalized(model, t.randn(_SIZES["hidden_size"])):
        pass
    assert all(t.equal(before[k], model.state_dict()[k]) for k in before)


def test_in_place_restores_after_an_exception(tmp_path):
    model = _saved("llama", tmp_path)
    before = {k: x.clone() for k, x in model.state_dict().items()}
    with pytest.raises(KeyError):
        with orthogonalized(model, t.randn(_SIZES["hidden_size"])):
            raise KeyError("boom")
    assert all(t.equal(before[k], model.state_dict()[k]) for k in before)


def test_in_place_refuses_to_pass_off_a_mismatched_checkpoint(tmp_path):
    """If memory no longer matches the checkpoint, restoring can't reproduce it: say so."""
    model = _saved("llama", tmp_path)
    with t.no_grad():
        model.model.layers[0].mlp.down_proj.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        with orthogonalized(model, t.randn(_SIZES["hidden_size"])):
            pass


def test_edit_bytes_reserves_only_the_working_chunk_in_place(tmp_path):
    import orthogonalize
    model = _saved("llama", tmp_path)
    chunk = max(orthogonalize._EMBED_CHUNK_ROWS, orthogonalize._LINEAR_CHUNK_COLS) * _SIZES["hidden_size"] * 4
    assert orthogonalize.edit_bytes(model) == chunk                    # untied: no copy at all
    tied = _saved("qwen2", tmp_path)
    embed = tied.model.embed_tokens.weight
    assert orthogonalize.edit_bytes(tied) == chunk + embed.numel() * embed.element_size()
