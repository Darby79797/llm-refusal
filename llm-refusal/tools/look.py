"""Text views of saved results, sized for reading in a terminal or by an agent.
Nothing here loads a model.

  look.py digest [--concept C] [--model M] [--variants]   one row per evaluate run + flags
  look.py run MODEL CONCEPT [--tag T]                      every condition of one run
  look.py gens MODEL CONCEPT [--cond a,b] [--flip a:b] [--detected y|n] [--degenerate]
                             [--grep RE] [-n N] [--chars C] [--seed S] [--rescore]
  look.py search MODEL CONCEPT [--metric bypass|induce|induce_global|kl] [--top K]
  look.py caa [MODEL]                                      CAA A/B sweep summary
  look.py cross [MODEL]                                    cross-concept cells
  look.py proj FILE [--layer L] [--item I]                 per-token projections (tools/project.py)

MODEL is a case-insensitive substring ("3B", "llama-3.1"). --root (repeatable) picks
result directories; default results/. Rates are detection rates with 95% Wilson CIs.
"""
import argparse
import json
import math
import os
import random
import re
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.runs import SHORT, load_index, wilson  # noqa: E402


def pct(x):
    return "  –  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:4.0f}%"


def fmt_cond(c):
    lo, hi = c.ci
    return f"{100 * c.rate:3.0f}% [{100 * lo:.0f},{100 * hi:.0f}]"


def table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    line = lambda r: "  ".join(str(v).ljust(w) for v, w in zip(r, widths))
    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows])


def pick_run(ix, model, concept, tag="", variant=None):
    runs = ix.find_evals(model, concept, tag, variant="" if variant is None else variant)
    if not runs:
        sys.exit(f"no evaluate run matches model~{model!r} concept={concept!r} tag~{tag!r}")
    if len({r.model for r in runs}) > 1 or len({r.tag for r in runs}) > 1:
        print("several runs match; using the first. Candidates:\n  " + "\n  ".join(r.id + "  @" + r.root for r in runs), file=sys.stderr)
    return runs[0]


# ── digest ────────────────────────────────────────────────────────
def cmd_digest(ix, a):
    runs = [r for r in ix.evals if (a.variants or not r.variant)
            and a.model.lower() in r.model.lower() and (not a.concept or r.concept == a.concept)]
    seen, rows = set(), []
    for r in sorted(runs, key=lambda r: (r.concept, r.model, r.variant, r.tag)):
        if (r.id, r.tag) in seen:      # same run archived under several roots
            continue
        seen.add((r.id, r.tag))
        c = r.conditions
        get = lambda n: fmt_cond(c[n]) if n in c else "–"
        lo = lambda n: f"{c[n].log_odds:+.1f}" if n in c and c[n].log_odds is not None else "–"
        filt = ("off" if r.filter == {} else f"{r.filter['pos_out']}/{r.filter['pos_in']}" if r.filter else "?")
        rows.append([r.concept, (r.variant + ":" if r.variant else "") + r.model.replace("-Instruct", ""), r.tag,
                     filt, c["baseline"].n if "baseline" in c else "–",
                     get("baseline"), get("global_ablation"), get("layer_specific_ablation"),
                     get("baseline_negative"), get("layer_specific_addition"),
                     f"{lo('baseline')}→{lo('global_ablation')}", f"{lo('baseline_negative')}→{lo('layer_specific_addition')}",
                     r.header.get("dtype", "?")[:4] + "/" + r.header.get("gen_batch_size", "?"),
                     "; ".join(f for f in r.flags() if not f.startswith("degenerate G-add"))])
    print(table(rows, ["concept", "model", "coords", "filter", "n", "baseline", "global abl", "layer abl",
                       "neg base", "addition", "lo abl", "lo add", "dt/bs", "flags"]))
    print("\nfilter = positives kept by behavioral filtering (off = prompt contrast). lo = log-odds base→condition.")


# ── run ───────────────────────────────────────────────────────────
def cmd_run(ix, a):
    r = pick_run(ix, a.model, a.concept, a.tag, a.variant)
    print(f"{r.id}  ({r.gen_path})")
    print(f"header {r.header}  filter {r.filter}  log {r.log_path}")
    d = ix.directions.get((r.model, r.concept))
    if d:
        print(f"saved direction: layer {d['layer']} pos {d['position_index']} score {d['score']:.3f}")
    s = ix.find_search(r.model, r.concept, r.variant)
    if s:
        print(f"search: selected {s.selected} via {s.tier}")
    rows = []
    for c in r.conditions.values():
        rows.append([c.name, c.side, c.n, fmt_cond(c), r.effect(c.name) or "",
                     f"{c.log_odds:+.2f}" if c.log_odds is not None else "–",
                     pct(c.degenerate), f"{c.nll:.2f}" if c.nll is not None else "–", pct(c.unsafe)])
    print(table(rows, ["condition", "side", "n", "rate [95% CI]", "vs base", "log-odds", "degen", "nll", "unsafe"]))
    for f in r.flags():
        print("flag:", f)


# ── gens ──────────────────────────────────────────────────────────
def cmd_gens(ix, a):
    r = pick_run(ix, a.model, a.concept, a.tag, a.variant)
    gens = r.generations
    if a.rescore:
        from tools.runs import detector
        detect = detector(r.concept)
        changed = 0
        for entries in gens.values():
            for e in entries:
                new = detect(e["response"])
                changed += new != e["detected"]
                e["detected"] = new
        print(f"rescored with the current {r.concept} detector: {changed} labels changed", file=sys.stderr)
    conds = a.cond.split(",") if a.cond else None
    if a.flip:
        x, y = a.flip.split(":")
        conds = conds or [x, y]
        if gens[x][0]["prompt"] != gens[y][0]["prompt"]:
            sys.exit(f"{x} and {y} run on different prompt sets")
    conds = conds or ["baseline", "global_ablation"]
    missing = [c for c in conds if c not in gens]
    if missing:
        sys.exit(f"no condition {missing}; have {list(gens)}")
    n_items = len(gens[conds[0]])
    idx = list(range(n_items))
    if a.flip:
        idx = [i for i in idx if gens[x][i]["detected"] != gens[y][i]["detected"]]
    if a.detected:
        want = a.detected.startswith("y")
        idx = [i for i in idx if gens[conds[-1]][i]["detected"] == want]
    if a.degenerate:
        idx = [i for i in idx if any(gens[c][i].get("degenerate") for c in conds)]
    if a.grep:
        rx = re.compile(a.grep, re.I)
        idx = [i for i in idx if any(rx.search(gens[c][i]["response"]) or rx.search(gens[c][i]["prompt"]) for c in conds)]
    total = len(idx)
    if a.seed is not None:
        random.Random(a.seed).shuffle(idx)
    idx = idx[:a.n]
    print(f"{r.id}: {total} of {n_items} prompts match; showing {len(idx)}.  [D]=detected [X]=degenerate\n")
    for i in idx:
        print(f"#{i}  {gens[conds[0]][i]['prompt'][:a.chars]}")
        for c in conds:
            e = gens[c][i]
            mark = ("D" if e["detected"] else " ") + ("X" if e.get("degenerate") else " ")
            body = " ".join(e["response"].split())[:a.chars]
            print(textwrap.fill(body, width=110, initial_indent=f"  [{mark}] {SHORT.get(c, c):8} ",
                                subsequent_indent=" " * 16))
        print()


# ── search ────────────────────────────────────────────────────────
def cmd_search(ix, a):
    s = ix.find_search(a.model, a.concept, a.variant or "")
    if not s:
        sys.exit("no search scores match")
    rows = s.rows
    layers = sorted({int(r["layer"]) for r in rows})
    positions = sorted({int(r["position"]) for r in rows}, reverse=True)
    grid = {(int(r["layer"]), int(r["position"])): r for r in rows}
    key = f"{a.metric}_score"
    print(f"{s.id}: {len(rows)} candidates, selected {s.selected} via {s.tier}")
    print(f"{a.metric} by layer (rows) x position (cols); * = selected\n")
    print("layer " + "".join(f"{p:>9}" for p in positions))
    for l in layers:
        cells = []
        for p in positions:
            v = grid.get((l, p), {}).get(key, float("nan"))
            star = "*" if s.selected == (l, p) else " "
            cells.append(f"{'':>1}{'nan' if math.isnan(v) else f'{v:+.2f}':>7}{star}")
        print(f"{l:>5} " + "".join(cells))
    finite = [r for r in rows if not math.isnan(r.get(key, float("nan")))]
    best = sorted(finite, key=lambda r: r[key], reverse=a.metric in ("induce", "induce_global"))[:a.top]
    print(f"\ntop {a.top} by {a.metric}:")
    print(table([[int(r["layer"]), int(r["position"])] + [f"{r[k]:+.3f}" for k in
                 ("bypass_score", "induce_score", "induce_global_score", "kl_score")] for r in best],
                ["layer", "pos", "bypass", "induce", "induce_g", "kl"]))


# ── caa ───────────────────────────────────────────────────────────
def cmd_caa(ix, a):
    models = [m for m in ix.caa if a.model.lower() in m.lower()]
    if not models:
        sys.exit("no CAA results (results/caa/<model>-ab.json)")
    for m in models:
        with open(ix.caa[m]) as f:
            d = json.load(f)
        print(f"== {m}  (bs {d.get('batch_size')}, {d.get('dtype')})")
        if "1.0" not in next(iter(d["sweep"].values()), {}).get(str(d["layers"][0]), {}):
            # multiplier sweep: p(match) per multiplier, and mean A/B probability mass
            # (a mass far below 1 means the model stopped answering with a letter)
            mults = [str(m) for m in d["multipliers"]]
            rows = []
            for b in d["behaviors"]:
                for l in d["layers"]:
                    row = d["sweep"][b][str(l)]
                    mass = lambda m: sum(x["ab_mass"] for x in row[m]["items"]) / len(row[m]["items"])
                    rows.append([b, l, f"{d['baseline'][b]['p_match']:.2f}"] +
                                [f"{row[m]['p_match']:.2f} ({mass(m):.2f})" for m in mults])
            print(table(rows, ["behavior", "layer", "base"] + [f"x{m} (A/B mass)" for m in mults]))
            print()
            continue
        rows = []
        for b in d["behaviors"]:
            sweep = d["sweep"].get(b, {})
            if not sweep:
                continue
            spread = {int(l): row["1.0"]["p_match"] - row["-1.0"]["p_match"] for l, row in sweep.items()
                      if "1.0" in row and "-1.0" in row}
            best = max(spread, key=spread.get)
            rows.append([b, f"{d['baseline'][b]['p_match']:.2f}", best,
                         f"{sweep[str(best)]['-1.0']['p_match']:.2f}", f"{sweep[str(best)]['1.0']['p_match']:.2f}",
                         f"{spread[best]:+.2f}", f"{d['baseline'][b].get('p_match_caa', float('nan')):.2f}"])
        print(table(rows, ["behavior", "base p", "best L", "x-1", "x+1", "spread", "base p (CAA scoring)"]))
        for name, per_b in d.get("cross", {}).items():
            print(f"\ncross-applied {name} (at CAA layer {next(iter(per_b.values()))['layer']}):")
            print(table([[b, f"{v['-1.0']['p_match']:.2f}", f"{v['1.0']['p_match']:.2f}",
                          f"{v['1.0']['p_match'] - v['-1.0']['p_match']:+.2f}",
                          f"{d['sweep'][b][str(v['layer'])]['1.0']['p_match'] - d['sweep'][b][str(v['layer'])]['-1.0']['p_match']:+.2f}"
                          if str(v['layer']) in d['sweep'].get(b, {}) else "–"]
                         for b, v in per_b.items()],
                        ["behavior", "x-1", "x+1", "spread", "CAA's own vector, same layer"]))
        for concept, row in d.get("cosine", {}).items():
            print(f"\ncos(our {concept} direction, CAA vectors):")
            print(table([[b, c["matched_caa_layer"], f"{c['cos_matched']:+.3f}", c["best_caa_layer"], f"{c['cos_best']:+.3f}"]
                         for b, c in row.items()], ["CAA behavior", "matched L", "cos", "best L", "cos"]))
        print()


# ── cross ─────────────────────────────────────────────────────────
def cmd_cross(ix, a):
    for m, paths in ix.cross.items():
        if a.model.lower() not in m.lower():
            continue
        for p in paths:
            with open(p) as f:
                d = json.load(f)
            names = d["concepts"]
            print(f"== {m}  {names}  (bs {d['gen_batch_size']}, oom splits {d['oom_splits']})")
            print("directions: " + ", ".join(f"{n} L{v['layer']}/P{v['position_index']}" for n, v in d["directions"].items()))
            sim = d["similarity_matrix"]
            print(table([[n] + [f"{sim[i][j]:+.2f}" for j in range(len(names))] for i, n in enumerate(names)],
                        ["cos"] + names))
            print()
            rows = []
            for key, c in d["cells"].items():
                k = round(c["rate"] * c["n"])
                lo, hi = wilson(k, c["n"])
                base = d["cells"].get("baseline->" + key.split("->")[1], {})
                note = []
                if c["degenerate_rate"] > 0.05:
                    note.append("model broken")
                if (base and base.get("log_odds") is not None and c.get("log_odds") is not None
                        and base["rate"] - c["rate"] >= 0.3 and base["log_odds"] > 0 and c["log_odds"] > 0):
                    note.append("rate fell, log-odds still > 0: rewording?")
                rows.append([key, f"{100 * c['rate']:3.0f}% [{100 * lo:.0f},{100 * hi:.0f}]",
                             f"{c['log_odds']:+.2f}" if c.get("log_odds") is not None else "–",
                             pct(c["degenerate_rate"]), c["n"], "; ".join(note)])
            print(table(rows, ["cell", "rate [95% CI]", "log-odds", "degen", "n", "check"]))
            print()


# ── proj ──────────────────────────────────────────────────────────
def cmd_proj(ix, a):
    with open(a.file) as f:
        d = json.load(f)
    layer = a.layer if a.layer is not None else d["direction_layer"]
    ref = d.get("reference", {})
    refs = ", ".join(f"{side} {v[layer]:+.2f}" for side, v in ref.items())
    print(f"{d['model']} {d['concept']} direction L{d['direction_layer']}{' (replayed interventions)' if d.get('replay') else ''}; "
          f"projections at layer {layer} onto the unit direction.\n"
          f"Scale: mean at the generation boundary over baseline prompts: {refs}\n")
    items = d["items"] if a.item is None else [d["items"][a.item]]
    for it in items:
        vals = it["proj"][layer]
        b = it["response_start"]
        print(f"[{it['condition']} #{it['index']}{' D' if it['detected'] else ''}] {' '.join(it['prompt'].split())[:100]}")
        def render(lo, hi):
            return " ".join(f"{tok.strip() or repr(tok)}({v:+.1f})" for tok, v in zip(it["tokens"][lo:hi], vals[lo:hi]))
        print(textwrap.fill("prompt tail: " + render(max(0, b - 8), b), 110, subsequent_indent="  "))
        print(textwrap.fill("response:    " + render(b, min(len(vals), b + a.tokens)), 110, subsequent_indent="  "))
        by_layer = [(sum(p[b:]) / max(1, len(p) - b)) for p in it["proj"]]
        print("mean response projection by layer: " + " ".join(f"{l}:{v:+.1f}" for l, v in enumerate(by_layer)))
        print()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help="results directory (repeatable; default results/)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("digest"); p.add_argument("--concept", default=""); p.add_argument("--model", default="")
    p.add_argument("--variants", action="store_true", help="include archived variants (filtered-, rerun-, ...)")
    for name in ("run", "gens"):
        p = sub.add_parser(name); p.add_argument("model"); p.add_argument("concept")
        p.add_argument("--tag", default=""); p.add_argument("--variant", default=None)
        if name == "gens":
            p.add_argument("--cond"); p.add_argument("--flip", help="a:b — prompts whose label differs")
            p.add_argument("--detected", help="y|n, on the last listed condition"); p.add_argument("--degenerate", action="store_true")
            p.add_argument("--grep"); p.add_argument("-n", type=int, default=8); p.add_argument("--chars", type=int, default=300)
            p.add_argument("--seed", type=int)
            p.add_argument("--rescore", action="store_true", help="relabel with the current detector (not the saved label)")
    p = sub.add_parser("search"); p.add_argument("model"); p.add_argument("concept")
    p.add_argument("--metric", default="bypass", choices=["bypass", "induce", "induce_global", "kl"])
    p.add_argument("--top", type=int, default=8); p.add_argument("--variant", default="")
    p = sub.add_parser("caa"); p.add_argument("model", nargs="?", default="")
    p = sub.add_parser("cross"); p.add_argument("model", nargs="?", default="")
    p = sub.add_parser("proj"); p.add_argument("file"); p.add_argument("--layer", type=int)
    p.add_argument("--item", type=int); p.add_argument("--tokens", type=int, default=24)
    a = ap.parse_args(argv)
    ix = load_index(a.root)
    globals()[f"cmd_{a.cmd}"](ix, a)


if __name__ == "__main__":
    main()
