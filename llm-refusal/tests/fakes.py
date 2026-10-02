"""Shared stand-ins for the lightweight tests: a hookable residual stack, a token-map
tokenizer, and a tiny random Qwen2 built on the real (cached) Qwen2.5 tokenizer."""
from types import SimpleNamespace

import pytest
import torch as t
import torch.nn as nn


class SimpleTokenizer:
    """Maps known tokens to fixed single ids; anything else hashes to one id."""
    def __init__(self, token_map=None):
        self._map = token_map or {}

    def encode(self, token, add_special_tokens=False):
        return [self._map.get(token, hash(token) % 50000)]


class ResidualBlock(nn.Module):
    """Block with real self_attn/mlp submodules whose output is x + attn(x) + mlp(x),
    so an ablation's block pre-hook and both sublayer post-hooks fire and all matter."""
    def __init__(self, dim):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.self_attn(x) + self.mlp(x)


class ResidualModel:
    """Exposes the `.model.layers` path the appliers/extractors expect. Called with a
    hidden-state tensor, or with input_ids (then the stream starts at zeros).
    fail_at_layer raises just before that block runs (0: before any block)."""
    def __init__(self, dim, n_layers=2, dtype=t.float32, seed=0, fail_at_layer=None):
        t.manual_seed(seed)
        self.model = SimpleNamespace(layers=nn.ModuleList([ResidualBlock(dim) for _ in range(n_layers)]).to(dtype))
        self.dim, self.dtype, self.device = dim, dtype, t.device("cpu")
        self.fail_at_layer = fail_at_layer

    def __call__(self, x=None, input_ids=None, **kwargs):
        if x is None:
            x = t.zeros(*input_ids.shape, self.dim, dtype=self.dtype)
        for i, layer in enumerate(self.model.layers):
            if i == self.fail_at_layer:
                raise RuntimeError("simulated forward failure")
            x = layer(x)
        return x


def no_hooks(layers):
    """True when no block pre-hook or sublayer post-hook is left registered."""
    return all(not layer._forward_pre_hooks and not layer.self_attn._forward_hooks
               and not layer.mlp._forward_hooks for layer in layers)


def cached_tokenizer(name):
    """A real tokenizer from the local HF cache; skips the test if it isn't there."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as e:
        pytest.skip(f"{name} tokenizer not in local HF cache: {e}")


def tiny_qwen2(tok, hidden_size=32, num_hidden_layers=2, **overrides):
    """Randomly initialised Qwen2 (fp32, CPU, eval) over tok's vocabulary."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    t.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=hidden_size, intermediate_size=2 * hidden_size,
                      num_hidden_layers=num_hidden_layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, **overrides)
    return Qwen2ForCausalLM(cfg).float().eval()
