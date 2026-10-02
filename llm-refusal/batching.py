"""Batch-size selection and OOM-safe batching for generation and scoring.

Two constraints pull against each other here:

  - Throughput wants big batches. On MPS in bf16 there is a cliff: at bs <= 8
    step time grows linearly with batch size (no gain at all), while at bs >= 16
    a faster matmul path kicks in. Measured on Llama-3-8B (M4 Pro), 128 new
    tokens: 14 tok/s at bs=2, 16 at bs=8, 104 at bs=16, 151 at bs=32. At 512 new
    tokens: ~10 at bs=2, 69 at bs=16, 90 at bs=32, 99 at bs=64.
  - Reproducibility wants a FIXED batch size. In bf16 on MPS, batch shape
    changes reduction order (and, per the cliff above, the kernel), so rates
    shift by a few pp across batch sizes.

So "auto" is dynamic in what it adapts to, but deterministic in outcome. It picks
the largest power of two whose ESTIMATED peak memory fits a budget derived from
the device's TOTAL memory and the model's weights. That budget does not depend
on current free memory, so the same machine, model, prompt set and generation
length always resolve to the same batch size. OOM is handled by a backstop,
not by the choice: `map_batched` splits a failing batch in half and records
that it did, because those rows then ran at a different batch shape.
"""
import logging
from typing import Callable, List, Optional, Sequence, TypeVar, Union

import torch as t

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

AUTO = "auto"
MAX_AUTO_BATCH_SIZE = 64
# Share of (device memory - weights) that "auto" may plan to use. The rest is
# headroom for the allocator's cache, the OS and other processes (e.g. Ollama
# serving LlamaGuard alongside the study model).
AUTO_MEMORY_FRACTION = 0.5
# Headroom over the analytic estimate. Calibrated on Llama-3-8B bf16 (MPS), 512
# new tokens: the analytic estimate is ~203 MB/row against a measured peak of
# 187-227 MB/row, provided the MPS allocator watermarks are set (see
# MPS_WATERMARKS). Without them, the caching allocator keeps every freed KV-cache
# buffer (the cache grows by concatenation each step), and bs=16 peaked at 38 GB
# above the weights and swapped.
ESTIMATE_SAFETY_FACTOR = 1.25

# PyTorch's MPS defaults let the caching allocator grow to 1.4x (soft) and 1.7x
# (hard) the device's recommended working set, i.e. 54-66 GB on a 48 GB machine.
# So instead of freeing its cache or raising OOM, it swaps. With these values it
# garbage-collects the cache past 0.6x, and past 0.8x it raises a real OOM, which
# map_batched catches. Set before the first MPS allocation (run_experiment.py
# and conftest.py); an explicit environment setting wins.
MPS_WATERMARKS = {"PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8"}


def parse_batch_size(value: Union[str, int]) -> Union[str, int]:
    """CLI/config parser: a positive int, or 'auto'."""
    if isinstance(value, int):
        if value < 1:
            raise ValueError(f"batch size must be >= 1, got {value}")
        return value
    if str(value).strip().lower() == AUTO:
        return AUTO
    n = int(value)
    if n < 1:
        raise ValueError(f"batch size must be >= 1, got {n}")
    return n


def _dtype_bytes(dtype: t.dtype) -> int:
    return t.tensor([], dtype=dtype).element_size()


def weight_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters()) + \
        sum(b.numel() * b.element_size() for b in model.buffers())


def device_total_bytes(device: t.device) -> int:
    """Memory the device can use in total. Deliberately NOT current free memory,
    so that 'auto' resolves identically across runs on the same machine."""
    if device.type == "mps":
        return int(t.mps.recommended_max_memory())
    if device.type == "cuda":
        return int(t.cuda.get_device_properties(device).total_memory)
    import psutil
    return int(psutil.virtual_memory().total)


def estimate_bytes_per_row(config, dtype: t.dtype, prompt_tokens: int, max_new_tokens: int) -> int:
    """Estimated peak bytes one batch row adds, over generation and coherence scoring.

    Generation: KV cache over prompt+response (x2: the cache is rebuilt by
    concatenation each step, so old and new copies coexist) plus prefill
    logits/activations. Scoring (coherence.response_nll): one forward over the
    full prompt+response returning logits for every position, which at 512
    tokens and a 128-152k vocabulary is the larger term.
    """
    b = _dtype_bytes(dtype)
    layers = config.num_hidden_layers
    heads = config.num_attention_heads
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    vocab = config.vocab_size
    hidden = config.hidden_size
    inter = getattr(config, "intermediate_size", None) or 4 * hidden
    P, T = prompt_tokens, prompt_tokens + max_new_tokens

    kv = 2 * layers * kv_heads * head_dim * T * b
    act = lambda n: n * (4 * hidden + 3 * inter) * b
    generation = 2 * kv + P * vocab * b + act(P)
    scoring = T * vocab * b + act(T)
    return int(ESTIMATE_SAFETY_FACTOR * max(generation, scoring))


def estimate_forward_bytes_per_row(config, dtype: t.dtype, tokens: int) -> int:
    """Peak bytes one row adds to a single no-cache forward that reads hidden states
    (no full-sequence logits): layer activations plus one layer's attention scores."""
    b = _dtype_bytes(dtype)
    hidden = config.hidden_size
    inter = getattr(config, "intermediate_size", None) or 4 * hidden
    heads = config.num_attention_heads
    return int(ESTIMATE_SAFETY_FACTOR * (tokens * (4 * hidden + 3 * inter) * b + heads * tokens * tokens * 4))


def forward_token_budget(model, memory_fraction: float = AUTO_MEMORY_FRACTION) -> int:
    """Padded tokens one cached forward batch may hold: the KV cache it keeps plus one
    layer's transient activations, within `memory_fraction` of (device - weights)."""
    c, b = model.config, _dtype_bytes(model.dtype)
    heads = c.num_attention_heads
    kv_heads = getattr(c, "num_key_value_heads", None) or heads
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // heads
    inter = getattr(c, "intermediate_size", None) or 4 * c.hidden_size
    per_token = ESTIMATE_SAFETY_FACTOR * (2 * c.num_hidden_layers * kv_heads * head_dim * b
                                          + (4 * c.hidden_size + 3 * inter) * b)
    budget = memory_fraction * (device_total_bytes(model.device) - weight_bytes(model))
    return int(budget / per_token)


def resolve_forward_batch_size(requested: Union[str, int], model, max_tokens: int,
                               cap: int = MAX_AUTO_BATCH_SIZE, memory_fraction: float = AUTO_MEMORY_FRACTION,
                               reserve_bytes: int = 0) -> int:
    """resolve_batch_size for forward-only passes (CAA vectors / A/B probabilities)."""
    requested = parse_batch_size(requested)
    if requested != AUTO:
        return requested
    per_row = estimate_forward_bytes_per_row(model.config, model.dtype, max_tokens)
    budget = memory_fraction * (device_total_bytes(model.device) - weight_bytes(model) - reserve_bytes)
    bs = 1
    while bs * 2 <= cap and bs * 2 * per_row <= budget:
        bs *= 2
    logger.info(f"Auto forward batch size: {bs} (longest sequence {max_tokens} tokens; "
                f"~{per_row / 1e6:.0f} MB/row est. vs {budget / 1e9:.1f} GB budget, cap {cap})")
    return bs


def resolve_batch_size(requested: Union[str, int], model, prompt_formatter, prompts: Sequence[str],
                       max_new_tokens: int, cap: int = MAX_AUTO_BATCH_SIZE,
                       memory_fraction: float = AUTO_MEMORY_FRACTION, reserve_bytes: int = 0) -> int:
    """An explicit int passes through; 'auto' picks the largest power of two <= cap
    whose estimated peak fits `memory_fraction` of (device total - weights - reserve_bytes).

    `reserve_bytes` is memory held beside the weights while the batches run, e.g. the
    edited weight copies of orthogonalize.orthogonalized (~36% of an 8B model): without
    it, an 8B capability run planned batches as if that copy weren't there and ran
    out of memory."""
    requested = parse_batch_size(requested)
    if requested != AUTO:
        return requested
    if not prompts:
        return 1
    prompt_tokens = max(int(prompt_formatter.format_batch([p])['attention_mask'].sum()) for p in prompts)
    per_row = estimate_bytes_per_row(model.config, model.dtype, prompt_tokens, max_new_tokens)
    budget = memory_fraction * (device_total_bytes(model.device) - weight_bytes(model) - reserve_bytes)
    bs = 1
    while bs * 2 <= cap and bs * 2 * per_row <= budget:
        bs *= 2
    logger.info(f"Auto batch size: {bs} (longest prompt {prompt_tokens} tokens + {max_new_tokens} new; "
                f"~{per_row / 1e6:.0f} MB/row est. vs {budget / 1e9:.1f} GB budget, cap {cap})")
    return bs


def is_oom_error(e: BaseException) -> bool:
    if isinstance(e, getattr(t, "OutOfMemoryError", ())):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _empty_cache():
    if t.backends.mps.is_available():
        t.mps.empty_cache()
    if t.cuda.is_available():
        t.cuda.empty_cache()


def map_batched(fn: Callable[[List[T]], List[R]], items: Sequence[T], batch_size: int,
                on_split: Optional[Callable[[int], None]] = None) -> List[R]:
    """Apply `fn` to consecutive chunks of `items` and concatenate the results.

    If a chunk runs out of memory, it is split in half and retried, recursively,
    and `on_split(chunk_size)` is called for each split. A single item that still
    runs out of memory re-raises. Non-OOM errors always propagate.
    """
    out: List[R] = []

    def run(chunk: List[T]):
        try:
            res = fn(chunk)
        except Exception as e:
            if not is_oom_error(e) or len(chunk) == 1:
                raise
            _empty_cache()
            logger.warning(f"Out of memory on a batch of {len(chunk)}; splitting in half and retrying. "
                           "These rows now run at a different batch shape (bf16 numerics may differ).")
            if on_split:
                on_split(len(chunk))
            mid = len(chunk) // 2
            run(chunk[:mid])
            run(chunk[mid:])
            return
        if len(res) != len(chunk):
            raise ValueError(f"batched fn returned {len(res)} results for {len(chunk)} items")
        out.extend(res)

    for i in range(0, len(items), batch_size):
        run(list(items[i:i + batch_size]))
    return out
