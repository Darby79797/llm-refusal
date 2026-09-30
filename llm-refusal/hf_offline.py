"""Run fully cached models without touching the network.

transformers' tokenizer loader calls the Hub API on every load, even for a cached model
(`_patch_mistral_regex` -> `model_info()`), so a network blip kills a run whose files
are all on disk (2026-09-24: three queued reruns failed on a DNS error). Offline mode
skips that call. It must be set before transformers / huggingface_hub are imported,
so this module imports neither.
"""
import glob
import os
from typing import Optional


def hub_cache_dir() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    return os.path.join(home, "hub")


def is_cached(model_name: str, cache_dir: Optional[str] = None) -> bool:
    """A snapshot with config, tokenizer config and weights on disk."""
    repo = os.path.join(cache_dir or hub_cache_dir(), "models--" + model_name.replace("/", "--"), "snapshots")
    for snap in glob.glob(os.path.join(repo, "*")):
        has = lambda pat: bool(glob.glob(os.path.join(snap, pat)))
        if has("config.json") and has("tokenizer_config.json") and (has("*.safetensors") or has("*.bin")):
            return True
    return False


def use_offline_if_cached(model_name: Optional[str], needs_network: bool = False) -> bool:
    """Set HF_HUB_OFFLINE=1 if `model_name` is fully cached, unless the user set it
    explicitly or the run needs the Hub for something else (datasets, lm-eval)."""
    if not model_name or needs_network or "HF_HUB_OFFLINE" in os.environ:
        return False
    if is_cached(model_name):
        os.environ["HF_HUB_OFFLINE"] = "1"
        return True
    return False
