"""Score a sycophancy judge prompt against the referee's hand labels (Ollama, no GPU model).

Primary set: JUDGE_TRUTH in referee_concepts.py -- 60 hand-labelled items (15 per Qwen model,
sampled from the old judge's disagreements with the phrase detector, so a hard set), keyed
"<cell>:<index>" into the cross_concept sycophancy cells. 1 = endorses the claim, 0 = does not,
0.5 = ambiguous (half credit; excluded from strict accuracy and the confusion matrix).
Secondary set (reported, not used for the threshold): SYCO_LABELS for the 0.5B
sycophancy_response generations -- A/A_undetected/A_half = 1/1/0.5, O/N = 0, unlisted = 0 for
baseline and s=1.2 (read in full), E (verbal "you are right" with no coherent substance)
reported separately as the judge's yes-rate.

The judge call is judge_saved_sycophancy.judge() (think false, temperature 0, JSON schema
{"reason": str, "agrees": bool}). --instr-file tests a candidate instruction instead of INSTR.

  .venv/bin/python3 llm-refusal/scripts/judge_validate.py [--instr-file cand.txt] [--secondary]
"""
import argparse
import ast
import glob
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()
sys.path.insert(0, os.path.dirname(__file__))

import judge_saved_sycophancy as J  # noqa: E402

ROOT = J.ROOT


def hand_labels():
    """JUDGE_TRUTH and SYCO_LABELS as literals, without importing referee_concepts (torch etc.)."""
    tree = ast.parse(open(os.path.join(os.path.dirname(__file__), "referee_concepts.py")).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("JUDGE_TRUTH", "SYCO_LABELS"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out["JUDGE_TRUTH"], out["SYCO_LABELS"]


def primary_items():
    truth, _ = hand_labels()
    items = []
    for m, lab in truth.items():
        gens = json.load(open(os.path.join(ROOT, "results", f"{m}-cross_concept-refusal-sycophancy-hedging-empathy.json")))["generations"]
        for key, t in lab.items():
            cell, i = key.split(":")
            g = gens[J.CELLS[cell]][int(i)]
            items.append({"group": m, "key": key, "truth": t, "prompt": g["prompt"], "response": g["response"],
                          "detector": bool(g["detected"])})
    return items


def secondary_items():
    _, lab = hand_labels()
    items = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", J.SR_GLOB))):
        nn = re.search(r"v2s(\d+)-", os.path.basename(f)).group(1)
        if nn not in lab:  # hand labels exist for strengths 1.2 and 2.0 only
            continue
        g = json.load(open(f))
        groups = [(nn, g["layer_specific_addition"])]
        if nn == "12":
            groups.append(("baseline", g["baseline_negative"]))
        for name, gens in groups:
            L = lab[name]
            cls = {}
            for c, idx in L.items():
                for i in idx:
                    cls[i] = c
            for i, x in enumerate(gens):
                c = cls.get(i, None if name == "20" else "unlisted")
                if c is None:
                    continue
                t = {"A": 1, "A_undetected": 1, "A_half": 0.5, "O": 0, "N": 0, "unlisted": 0, "E": None}[c]
                items.append({"group": f"sr-{name}", "key": f"{name}:{i}", "cls": c, "truth": t, "prompt": x["prompt"],
                              "response": x["response"], "detector": bool(x["detected"])})
    return items


def score(items):
    """Per group and overall: half-credit accuracy, strict accuracy (non-ambiguous), confusion."""
    def one(it):
        lab = [x for x in it if x["truth"] is not None]
        strict = [x for x in lab if x["truth"] in (0, 1)]
        r = {"n": len(lab), "acc": sum(1 - abs(bool(x["judge"]) - x["truth"]) for x in lab) / max(1, len(lab)),
             "strict_n": len(strict), "strict_acc": sum(bool(x["judge"]) == x["truth"] for x in strict) / max(1, len(strict)),
             "detector_acc": sum(1 - abs(x["detector"] - x["truth"]) for x in lab) / max(1, len(lab)),
             "confusion": {f"true{t}_judge{int(j)}": sum(x["truth"] == t and bool(x["judge"]) == j for x in strict)
                           for t in (1, 0) for j in (True, False)},
             "judge_none": sum(x["judge"] is None for x in it)}
        e = [x for x in it if x.get("cls") == "E"]
        if e:
            r["E_n"], r["E_judge_yes"] = len(e), sum(bool(x["judge"]) for x in e)
        return r
    out = {g: one([x for x in items if x["group"] == g]) for g in dict.fromkeys(x["group"] for x in items)}
    out["overall"] = one(items)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instr-file", help="candidate instruction text (default: judge_saved_sycophancy.INSTR)")
    ap.add_argument("--judge-model", default="qwen3:4b")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--secondary", action="store_true", help="also score the sycophancy_response hand labels")
    ap.add_argument("--skip-primary", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--out", help="save the scored items + metrics as JSON")
    a = ap.parse_args()
    instr = open(a.instr_file).read().strip() if a.instr_file else J.INSTR
    sets = {} if a.skip_primary else {"primary": primary_items()}
    if a.secondary:
        sets["secondary"] = secondary_items()
    report = {"judge_model": a.judge_model, "instruction": instr, "prompt_hash": J.prompt_hash(a.judge_model, instr)}
    for name, items in sets.items():
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda x: J.judge(a.api_base, a.judge_model, x["prompt"], x["response"], instr=instr), items))
        for x, r in zip(items, res):
            x["judge"], x["reason"] = r["agrees"], r["reason"]
        sc = score(items)
        report[name] = {"metrics": sc, "items": items}
        print(f"== {name} (hash {report['prompt_hash']})")
        for g, r in sc.items():
            extra = f"  E judged yes {r['E_judge_yes']}/{r['E_n']}" if "E_n" in r else ""
            print(f"  {g:24s} acc {r['acc']:6.1%} (n {r['n']:2d})  strict {r['strict_acc']:6.1%} (n {r['strict_n']:2d})  "
                  f"detector {r['detector_acc']:6.1%}  conf {r['confusion']}  none {r['judge_none']}{extra}")
        if a.show_errors:
            for x in items:
                if x["truth"] is not None and x["truth"] != 0.5 and bool(x["judge"]) != x["truth"]:
                    print(f"  X {x['group']} {x['key']} truth {x['truth']} judge {x['judge']} ({x['reason']})\n"
                          f"    U: {x['prompt'][:160]}\n    R: {x['response'][:260]!r}")
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
