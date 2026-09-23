"""95% Wilson intervals for every condition's detection rate in evaluate generations files.

Usage:
  python llm-refusal/scripts/confidence_intervals.py results/sweep4/*-generations.json

Reads the per-response `detected` labels saved by --mode evaluate. No model needed.
Wilson rather than the normal approximation because most rates here sit at or
near 0% / 100%, where the normal interval collapses to zero width.
"""
import sys
import json
import math

Z = 1.959964  # 95%


def wilson(k: int, n: int):
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    denom = 1 + Z**2 / n
    centre = (p + Z**2 / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main(paths):
    for path in paths:
        with open(path) as f:
            generations = json.load(f)
        print(f"== {path.split('/')[-1]}")
        for condition, entries in generations.items():
            n = len(entries)
            k = sum(e['detected'] for e in entries)
            lo, hi = wilson(k, n)
            print(f"  {condition:<28} {100*k/n:5.1f}%  [{100*lo:5.1f}, {100*hi:5.1f}]  (k={k}, n={n})")


if __name__ == "__main__":
    main(sys.argv[1:])
