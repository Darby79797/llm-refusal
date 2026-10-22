"""Process environment for every entry point. Torch-free: import and call
setup_process_env() before torch/transformers are imported, because these
variables are read once, at import or first allocation.

    sys.path.insert(0, <llm-refusal dir>)
    from env import setup_process_env; setup_process_env()
"""
import os

# PyTorch's MPS defaults let the caching allocator grow to 1.4x (soft) and 1.7x
# (hard) the device's recommended working set, i.e. 54-66 GB on a 48 GB machine.
# So instead of freeing its cache or raising OOM, it swaps. With these values it
# garbage-collects the cache past 0.6x, and past 0.8x it raises a real OOM, which
# batching.map_batched catches. An explicit environment setting wins.
MPS_WATERMARKS = {"PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8"}

# Memory (GB) that another long-running process holds for the whole run, e.g. a training job sharing the
# machine (the queue runner sets it from results/queue/reserve_gb). batching's "auto" plans its batches as
# if the device were that much smaller, and the MPS allocator's cap shrinks by the same amount, so a job
# that would overrun raises an OOM (and splits its batch) instead of swapping the machine to a standstill
# (2026-10-21: an 8B job loading beside a 12 GB trainer locked the Mac up; it needed a hard reset).
MEMORY_RESERVE_ENV = "LLM_REFUSAL_MEMORY_RESERVE_GB"
# MPS's recommended working set (torch.mps.recommended_max_memory) is this share of physical memory;
# the watermark ratios are relative to it.
MPS_RECOMMENDED_SHARE = 0.75


def memory_reserve_bytes() -> int:
    """Bytes reserved for other processes (MEMORY_RESERVE_ENV, GB; 0 when unset)."""
    v = os.environ.get(MEMORY_RESERVE_ENV, "").strip()
    if not v:
        return 0
    gb = float(v)
    if gb < 0:
        raise ValueError(f"{MEMORY_RESERVE_ENV} must be >= 0 GB, got {v!r}")
    return int(gb * 10**9)


def physical_memory_bytes() -> int:
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def mps_watermarks(reserve_bytes: int = 0, phys_bytes: int = None) -> dict:
    """The allocator watermarks for a reserve: the default pair, or with the reserve taken off the
    recommended working set (HIGH = 1 - reserve/recommended, LOW = 0.75 HIGH), never above the defaults."""
    high = float(MPS_WATERMARKS["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
    low = float(MPS_WATERMARKS["PYTORCH_MPS_LOW_WATERMARK_RATIO"])
    if reserve_bytes > 0:
        recommended = MPS_RECOMMENDED_SHARE * (phys_bytes if phys_bytes is not None else physical_memory_bytes())
        high = min(high, max(0.1, 1.0 - reserve_bytes / recommended))
        low = min(low, 0.75 * high)
    return {"PYTORCH_MPS_LOW_WATERMARK_RATIO": f"{low:.3f}", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": f"{high:.3f}"}


def setup_process_env() -> None:
    """Set tokenizer/MPS env vars. Must run before torch is imported."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Unsupported MPS ops must fail loudly, not silently run on CPU.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    # Bound the MPS caching allocator so it GCs / raises OOM instead of swapping.
    # LOW must not exceed HIGH (PyTorch refuses to start), so derive it unless set.
    marks = mps_watermarks(memory_reserve_bytes())
    high = os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", marks["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO",
                          str(min(float(marks["PYTORCH_MPS_LOW_WATERMARK_RATIO"]), 0.75 * float(high))))
