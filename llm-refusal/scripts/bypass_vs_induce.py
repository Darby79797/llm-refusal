"""Do bypass and induce prefer the same coordinates? (limb 3.4; no model)

From every saved search CSV: the best-bypass candidate (lowest bypass among layers
under the concept's cutoff, KL under its threshold), the best-induce candidate, the
selected direction, and how far apart they are in the search grid.

  .venv/bin/python3 llm-refusal/scripts/bypass_vs_induce.py [--concept refusal] [--root results]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd  # noqa: E402

from concept import get_concept  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--root", default="results")
    a = ap.parse_args()
    cfg = get_concept(a.concept).search_config
    rows = []
    for csv in sorted(glob.glob(f"{a.root}/*-{a.concept}-search-scores.csv")):
        m = re.match(rf"^(?:[a-z][a-z0-9]*-(?=[A-Z]))?(.+?)-{a.concept}-search-scores\.csv$", os.path.basename(csv))
        if not m or os.path.basename(csv)[0].islower():
            continue
        model = m.group(1)
        df = pd.read_csv(csv).dropna(subset=["bypass_score"])
        n_layers = int(df.layer.max()) + 1   # the CSV already stops at the concept's layer cutoff
        d = df[df.layer > 0]
        ok = d[(d.kl_score < cfg["kl_threshold"]) & (d.induce_score > cfg["induce_threshold"])]
        pool = ok if len(ok) else d
        bb = pool.loc[pool.bypass_score.idxmin()]
        bi = pool.loc[pool.induce_score.idxmax()]
        sel = None
        p = f"{a.root}/{model}-{a.concept}-direction.json"
        if os.path.exists(p):
            sel = json.load(open(p))
        rows.append({"model": model, "n_layers": n_layers, "strict_passers": len(ok),
                     "best_bypass": (int(bb.layer), int(bb.position), round(float(bb.bypass_score), 2), round(float(bb.induce_score), 2)),
                     "best_induce": (int(bi.layer), int(bi.position), round(float(bi.bypass_score), 2), round(float(bi.induce_score), 2)),
                     "selected": (sel["layer"], sel["position_index"]) if sel else None,
                     "same_layer": int(bb.layer) == int(bi.layer), "layer_gap": int(abs(bb.layer - bi.layer))})
    print(f"{'model':26s} {'Lmax':>4s} {'pass':>4s}  best-bypass (L,P,byp,ind)       best-induce (L,P,byp,ind)       selected   gap")
    for r in rows:
        print(f"{r['model']:26s} {r['n_layers']:4d} {r['strict_passers']:4d}  {str(r['best_bypass']):30s}  "
              f"{str(r['best_induce']):30s}  {str(r['selected']):9s}  {r['layer_gap']}")
    os.makedirs(f"{a.root}/analysis", exist_ok=True)
    with open(f"{a.root}/analysis/bypass_vs_induce-{a.concept}.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
