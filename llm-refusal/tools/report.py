"""Build a self-contained HTML report of saved results (no server, no CDN, no model).

  report.py [--root results] [--root results/sweep4] [--out results/report/index.html]

Views: overview matrix (model x concept) and digest table; per-run detail (rates with
95% Wilson CIs, log-odds, degeneracy, search heatmap, side-by-side generations browser
with flip/label/degenerate/text filters); CAA A/B curves; cross-concept interference;
per-token projection heatmaps (from tools/project.py). Labels are re-scored with the
current detector at build time; responses whose label changed since the run are marked.
"""
import argparse
import json
import os
import sys
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.runs import SHORT, detector, load_index  # noqa: E402

MAX_CHARS = 1500


def build_data(roots):
    ix = load_index(roots)
    runs, seen = [], set()
    for r in sorted(ix.evals, key=lambda r: (r.model, r.concept, r.variant, r.tag)):
        if (r.id, r.tag) in seen:
            continue
        seen.add((r.id, r.tag))
        detect = detector(r.concept)
        gens, changed = {}, 0
        for cond, entries in r.generations.items():
            rows = []
            for e in entries:
                now = detect(e["response"])
                changed += now != e["detected"]
                rows.append({"p": e["prompt"], "r": e["response"][:MAX_CHARS], "d": now,
                             **({"was": e["detected"]} if now != e["detected"] else {}),
                             **({"x": 1} if e.get("degenerate") else {}),
                             **({"nll": round(e["nll"], 2)} if isinstance(e.get("nll"), float) else {}),
                             **({"lg": e["llamaguard"]} if e.get("llamaguard") is not None else {})})
            gens[cond] = rows
        s = ix.find_search(r.model, r.concept, r.variant)
        runs.append({
            "id": r.id, "model": r.model, "concept": r.concept, "variant": r.variant, "tag": r.tag,
            "root": r.root, "layer": r.layer, "pos": r.pos, "header": r.header, "filter": r.filter,
            "flags": r.flags(), "relabeled": changed,
            "conditions": [{"name": c.name, "short": SHORT.get(c.name, c.name), "side": c.side, "n": c.n,
                            "k": sum(1 for g in gens[c.name] if g["d"]), "k_saved": c.k,
                            "log_odds": c.log_odds, "degenerate": c.degenerate, "nll": c.nll, "unsafe": c.unsafe,
                            "effect": r.effect(c.name)} for c in r.conditions.values()],
            "search": ({"selected": s.selected, "tier": s.tier,
                        "rows": [[int(x["layer"]), int(x["position"]), x["bypass_score"], x["induce_score"],
                                  x["induce_global_score"], x["kl_score"]] for x in s.rows]} if s else None),
            "gens": gens,
        })
    caa = {}
    for model, path in ix.caa.items():
        with open(path) as f:
            d = json.load(f)
        strip = lambda row: {m: {"p": v["p_match"], "pc": v.get("p_match_caa")} for m, v in row.items()
                             if isinstance(v, dict) and "p_match" in v}
        caa[model] = {
            "behaviors": d["behaviors"], "dtype": d.get("dtype"), "batch_size": d.get("batch_size"),
            "baseline": {b: {"p": v["p_match"], "pc": v.get("p_match_caa")} for b, v in d["baseline"].items()},
            "sweep": {b: {l: strip(row) for l, row in rows.items()} for b, rows in d["sweep"].items()},
            "cross": {name: {b: {"layer": v["layer"], **strip(v)} for b, v in per.items()}
                      for name, per in d.get("cross", {}).items()},
            "cosine": {c: {b: {k: v for k, v in row.items() if k != "by_layer"} | {"by_layer": row["by_layer"]}
                           for b, row in rows.items()} for c, rows in d.get("cosine", {}).items()},
        }
    cross = {}
    for model, paths in ix.cross.items():
        for p in paths:
            with open(p) as f:
                d = json.load(f)
            cross[f"{model} ({'+'.join(d['concepts'])})"] = {
                k: d[k] for k in ("concepts", "directions", "gen_batch_size", "oom_splits",
                                  "similarity_matrix", "interference_matrix", "multi_ablation", "cells")}
    proj = {}
    for root in roots:
        pdir = os.path.join(root, "proj")
        if os.path.isdir(pdir):
            for fn in sorted(os.listdir(pdir)):
                if fn.endswith(".json"):
                    with open(os.path.join(pdir, fn)) as f:
                        proj[fn[:-5]] = json.load(f)
    return {"built": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "roots": roots,
            "runs": runs, "caa": caa, "cross": cross, "proj": proj}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append")
    ap.add_argument("--out", default="results/report/index.html")
    a = ap.parse_args(argv)
    data = build_data(a.root or ["results"])
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html")) as f:
        html = f.read()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(html.replace("/*__DATA__*/null", payload))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB): {len(data['runs'])} runs, "
          f"{len(data['caa'])} CAA, {len(data['cross'])} cross-concept, {len(data['proj'])} projection files")


if __name__ == "__main__":
    main()
