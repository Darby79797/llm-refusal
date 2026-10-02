"""Run fully cached models without touching the network.

transformers' tokenizer loader calls the Hub API on every load, even for a cached model
(`_patch_mistral_regex` -> `model_info()`), so a network blip kills a run whose files
are all on disk (2026-09-24: three queued reruns failed on a DNS error). Offline mode
skips that call. It must be set before transformers / huggingface_hub are imported,
so this module imports neither.
"""
import glob
import os
import subprocess
import sys
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


OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}

_TASKS_PROBE = """
import sys
from lm_eval.tasks import TaskManager, get_task_dict
for task in get_task_dict(sys.argv[1:], TaskManager()).values():
    docs = task.test_docs() if task.has_test_docs() else task.validation_docs()
    next(iter(docs))
"""


def lm_eval_tasks_cached(tasks) -> bool:
    """True if every lm-eval task's dataset loads with the network off.

    Checked in a subprocess: the offline switches only take effect if set before
    huggingface_hub/datasets are imported, and lm_eval imports both. Takes ~10 s.
    """
    if not tasks:
        return True
    env = {**os.environ, **OFFLINE_ENV}
    try:
        r = subprocess.run([sys.executable, "-c", _TASKS_PROBE, *tasks], env=env,
                           capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def use_offline_for_run(model_name: Optional[str], eval_tasks=()) -> bool:
    """Go fully offline (Hub and datasets) when the model and any lm-eval tasks are
    cached; otherwise leave the network on so they can be downloaded."""
    if not model_name or "HF_HUB_OFFLINE" in os.environ:
        return False
    if is_cached(model_name) and lm_eval_tasks_cached(list(eval_tasks)):
        os.environ.update(OFFLINE_ENV)
        return True
    return False


def hub_reachable(timeout: float = 2.0) -> bool:
    """Can we open a TCP connection to the Hub? (No HTTP request is made.)"""
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def offline_for_script(argv=None) -> bool:
    """Offline switch for scripts and tests, whose model names aren't known up front.

    Call before transformers is imported. Goes offline (Hub and datasets) when a model
    ID in the arguments is cached, or when the Hub can't be reached at all, so the
    cache is the only option anyway. Otherwise stays online, so first-time downloads
    work. An explicit HF_HUB_OFFLINE setting always wins.
    """
    if "HF_HUB_OFFLINE" in os.environ:
        return os.environ["HF_HUB_OFFLINE"] == "1"
    argv = sys.argv[1:] if argv is None else argv
    ids = [a for a in argv if a.count("/") == 1 and not a.startswith(("-", "/", ".")) and not os.path.exists(a)]
    if any(is_cached(m) for m in ids) or not hub_reachable():
        os.environ.update(OFFLINE_ENV)
        return True
    return False
