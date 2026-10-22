"""env.py: the MPS allocator watermarks, with and without a declared memory reserve."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import env  # noqa: E402

GB = 10**9
PHYS_48 = 51_539_607_552  # a 48 GB Mac; recommended working set = 0.75 x = 38.65 GB


def test_default_watermarks_without_reserve():
    assert env.mps_watermarks(0, PHYS_48) == {"PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.600",
                                              "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.800"}


def test_reserve_takes_its_share_off_the_recommended_working_set():
    m = env.mps_watermarks(16 * GB, PHYS_48)
    high = float(m["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
    assert abs(high - (1 - 16 * GB / (0.75 * PHYS_48))) < 1e-3   # 0.586: a 22.6 GB cap beside a 16 GB neighbour
    assert float(m["PYTORCH_MPS_LOW_WATERMARK_RATIO"]) == round(0.75 * high, 3)


def test_small_reserve_never_raises_the_defaults():
    assert env.mps_watermarks(1 * GB, PHYS_48) == env.mps_watermarks(0, PHYS_48)


def test_memory_reserve_bytes_reads_env(monkeypatch):
    monkeypatch.delenv(env.MEMORY_RESERVE_ENV, raising=False)
    assert env.memory_reserve_bytes() == 0
    monkeypatch.setenv(env.MEMORY_RESERVE_ENV, "12.5")
    assert env.memory_reserve_bytes() == int(12.5 * GB)
