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
  look.py limbs [MODEL]                                    side experiments: edit-cost, trajectory,
                                                           jailbreak, categories, finetune (identity,
                                                           regrow dose, rank-1 adapters)

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
    if a.flip:
        # Pair summary, e.g. global_ablation:orthogonalized should agree up to bf16 rounding.
        same = sum(gens[x][i]["response"] == gens[y][i]["response"] for i in range(n_items))
        flips = sum(gens[x][i]["detected"] != gens[y][i]["detected"] for i in range(n_items))
        lg = [i for i in range(n_items) if "llamaguard" in gens[x][i] and "llamaguard" in gens[y][i]]
        lg_note = (f", LlamaGuard verdict differs on {sum(gens[x][i]['llamaguard'] != gens[y][i]['llamaguard'] for i in lg)}/{len(lg)}"
                   if lg else "")
        print(f"{x} vs {y}: identical text {same}/{n_items}, label differs {flips}/{n_items}{lg_note}")
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


# ── limbs ─────────────────────────────────────────────────────────
def _num(x, f="+.2f"):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, f)


def _rate(x):
    """Rates are saved either bare or as {rate, log_odds, n}."""
    return pct(x["rate"] if isinstance(x, dict) else x)


def _short_model(m):
    return m.split("/")[-1].replace("-Instruct", "")


def _dir(d):
    return f"L{d['direction']['layer']}/P{d['direction']['position_index']}" if "direction" in d else "?"


def limbs_edit_cost(ix, a):
    for x in ix.find_limbs("edit_cost", a.model):
        d = x.load()
        none = d["specs"].get("none", {})
        n = d.get("n", {})
        print(f"== edit-cost  {x.model} {x.parts['concept']}{' ' + x.parts['tag'] if x.parts['tag'] else ''}  dir {_dir(d)}  "
              f"n harmful {n.get('harmful')} harmless {n.get('harmless')} alpaca {n.get('alpaca')} pile {n.get('pile')} "
              f"on-dist {n.get('on_distribution')}")
        prefix = f"results/finetune/{x.model}-{x.parts['concept']}-"
        rows = []
        for name, sp in d["specs"].items():
            dce = lambda k: _num(sp[k] - none[k], "+.3f") if k in sp and k in none else "–"
            rows.append([name.replace(prefix, ""), sp.get("kind", ""), sp.get("n_blocks_edited", "–"),
                         _rate(sp["harmful"]), _rate(sp["harmless"]), dce("ce_alpaca"), dce("ce_pile"), dce("ce_on_distribution")])
        print(table(rows, ["spec", "kind", "blocks", "refusal", "false ref", "dCE alp", "dCE pile", "dCE on-dist"]))
        print()


def limbs_trajectory(ix, a):
    rows = []
    for x in ix.find_limbs("trajectory", a.model):
        d = x.load()
        for cond, c in d["conditions"].items():
            p = c["proj_r_hat"]
            hm, hl, au = p["harmful_mean"], p["harmless_mean"], p["auroc"]
            peak = max(range(len(hm)), key=lambda i: hm[i])
            first = next((i for i, v in enumerate(au) if v is not None and v >= 0.95), None)
            rows.append([_short_model(x.model), cond, d["direction"]["layer"], len(hm), peak,
                         f"{hm[peak]:+.2f}/{hl[peak]:+.2f}", f"{hm[-1]:+.2f}/{hl[-1]:+.2f}", "–" if first is None else first])
    if rows:
        print("== trajectory  projection onto r̂ by layer (harmful/harmless means)")
        print(table(rows, ["model", "cond", "dir L", "layers", "peak L", "proj @peak", "proj @last", "AUROC≥.95 from L"]))
        print()


def limbs_jailbreak(ix, a):
    for x in ix.find_limbs("jailbreak", a.model):
        d = x.load()
        has_last = any("proj_last_mean" in t for t in d["templates"].values())
        rows = []
        for name, t in d["templates"].items():
            row = [name, pct(t["refusal_rate"]), _num(t["proj_mean"]), _num(t["auroc_proj_predicts_refusal"], ".2f")]
            if has_last:
                row += [_num(t.get("proj_last_mean")), _num(t.get("auroc_projlast_predicts_refusal"), ".2f")]
            rows.append(row)
        h = d.get("harmless", {})
        rows.append(["(harmless)", pct(h.get("refusal_rate")), _num(h.get("proj_mean")), ""] + (["", ""] if has_last else []))
        print(f"== jailbreak  {x.model} {x.parts['concept']}  dir {_dir(d)}  n {d.get('n')}  "
              f"pooled AUROC(proj→refusal) {_num(d.get('pooled_auroc_proj_predicts_refusal'), '.3f')}")
        print(table(rows, ["template", "refusal", "proj", "AUROC"] + (["proj last", "AUROC last"] if has_last else [])))
        print()


def limbs_categories(ix, a):
    for x in ix.find_limbs("categories", a.model):
        d = x.load()
        causal = d.get("causal", {})
        rows = []
        for cat, cos in d["cos_with_r_hat"].items():
            row = [cat, d.get("categories", {}).get(cat, "–"), f"{cos:+.3f}"]
            if causal:
                c = causal.get(cat, {})
                row += [_rate(c[k]) if k in c else "–" for k in ("ablate_dk_own", "ablate_loo_own", "add_dk_harmless")]
            rows.append(row)
        print(f"== categories  {x.model} {x.parts['concept']}  dir {_dir(d)}  "
              f"offdiag cos mean {_num(d.get('cos_offdiag_mean'), '.3f')} min {_num(d.get('cos_offdiag_min'), '.3f')}")
        print(table(rows, ["category", "n", "cos r̂"] + (["abl d_k own", "abl LOO own", "+d_k harmless"] if causal else [])))
        print()


def limbs_identity(ix, a):
    for x in ix.find_limbs("identity", a.model):
        d = x.load()
        ref_keys = list(dict.fromkeys(k for arm in d["arms"].values() for k in arm.get("refusal", {})))
        rows = [[name, _num(arm.get("cos_r_hat"), "+.3f"), _num(arm.get("cos_u_perp"), "+.3f"),
                 _num(arm.get("regrown_feature_auroc"), ".2f")] +
                [_rate(arm["refusal"][k]) if k in arm.get("refusal", {}) else "–" for k in ref_keys]
                for name, arm in d["arms"].items()]
        short = lambda k: k.replace("ablate_", "abl ").replace("harmless_", "h:").replace("add_", "+")
        print(f"== direction identity  {x.model}  dir {_dir(d)}  rank1 {d.get('rank1_tags')}  "
              f"cos(û, r̂) {_num(d.get('cos_u_hat_r_hat'), '+.3f')}")
        print(table(rows, ["arm", "cos r̂", "cos u⊥", "regrown AUROC"] + [short(k) for k in ref_keys]))
        print()


def limbs_regrow(ix, a):
    by_model = {}
    for x in ix.find_limbs("regrow", a.model):
        by_model.setdefault((x.model, x.parts["concept"]), []).append(x)
    for (model, concept), xs in by_model.items():
        rows = []
        for x in xs:
            d = x.load()
            h = [e for e in d.get("history", []) if "pos_rate" in e]
            last = h[-1] if h else {}
            rows.append((x.parts["arm"], int(x.parts["n"]), int(d.get("seed", x.parts["seed"] or 0)), x.parts["tag"] or "",
                         last.get("step", "–"), pct(d.get("clean", {}).get("pos_rate")), pct(last.get("pos_rate")),
                         _num(last.get("pos_log_odds")), pct(d.get("final_negative", {}).get("neg_rate"))))
        rows.sort(key=lambda r: r[:4])
        print(f"== regrow dose  {model} {concept}  (pos = refusal on harmful at the last eval step, neg = on harmless)")
        print(table([list(r) for r in rows], ["arm", "n ref", "seed", "tag", "step", "clean", "pos", "log-odds", "neg"]))
        print()


def limbs_rank1(ix, a):
    by_model = {}
    for x in ix.find_limbs("rank1", a.model):
        by_model.setdefault((x.model, x.parts["concept"]), []).append(x)
    for (model, concept), xs in by_model.items():
        rows = []
        for x in xs:
            d = x.load()
            g = lambda k: pct(d[k]["pos_rate"]) if k in d else "–"
            cos = [abs(ad["cos_u_rhat"]) for ad in d.get("adapters", []) if "cos_u_rhat" in ad]
            rows.append([x.parts["tag"], d.get("objective", ""), ",".join(map(str, d.get("adapter_layers", []))),
                         g("before"), g("after"), ",".join(f"{c:.2f}" for c in cos) or "–",
                         g("after_u_minus_rhat"), g("after_u_rhat_only")])
        print(f"== rank-1 adapters  {model} {concept}  (pos = refusal on harmful)")
        print(table(rows, ["tag", "objective", "layer", "before", "after", "|cos u,r̂|", "u−r̂", "r̂ only"]))
        print()


def cmd_limbs(ix, a):
    if not any(a.model.lower() in x.model.lower() for x in ix.limbs):
        sys.exit("no side-experiment results (edit-cost/, trajectory/, jailbreak/, categories/, finetune/) match")
    for f in (limbs_edit_cost, limbs_trajectory, limbs_jailbreak, limbs_categories, limbs_identity, limbs_regrow, limbs_rank1):
        f(ix, a)


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
    p = sub.add_parser("limbs"); p.add_argument("model", nargs="?", default="")
    p = sub.add_parser("proj"); p.add_argument("file"); p.add_argument("--layer", type=int)
    p.add_argument("--item", type=int); p.add_argument("--tokens", type=int, default=24)
    a = ap.parse_args(argv)
    ix = load_index(a.root)
    globals()[f"cmd_{a.cmd}"](ix, a)


if __name__ == "__main__":
    main()
