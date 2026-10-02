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


def setup_process_env() -> None:
    """Set tokenizer/MPS env vars. Must run before torch is imported."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Unsupported MPS ops must fail loudly, not silently run on CPU.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    # Bound the MPS caching allocator so it GCs / raises OOM instead of swapping.
    # LOW must not exceed HIGH (PyTorch refuses to start), so derive it unless set.
    high = os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                                 MPS_WATERMARKS["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO",
                          str(min(float(MPS_WATERMARKS["PYTORCH_MPS_LOW_WATERMARK_RATIO"]), 0.75 * float(high))))
