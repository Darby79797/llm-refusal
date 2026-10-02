"""Is the refusal direction one direction or an average over harm categories? (limb 3.2)

The 90 harmful train prompts come in nine labelled categories of ten (the comments in
prompts.create_refusal_train_data, read here with the tokenizer module so the data
file stays as it is). From one activation pass at r̂'s coordinates:

  per category k:   d_k = mean(category k) - mean(harmless);  the leave-one-out d_-k
  geometry:         cos(d_k, d_j), cos(d_k, r̂), cos(d_-k, r̂)
  causal transfer:  ablating d_k (every layer) on its own 10 prompts, on the other 80,
                    and on the harmful eval set; ablating d_-k on category k (held out);
                    adding d_k at r̂'s layer to harmless eval prompts (induction)

  .venv/bin/python3 llm-refusal/scripts/category_directions.py --model Qwen/Qwen2.5-0.5B-Instruct

Writes results/categories/<model>-refusal.json.
"""
import argparse
import io
import inspect
import json
import os
import sys
import tokenize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

import prompts  # noqa: E402
from probe import behaviour, cos, load_run, residuals_at, save_json, unit  # noqa: E402


def categorised_train_prompts():
    """[(category, prompt)] for the positive train prompts, from the source comments."""
    src = inspect.getsource(prompts.create_refusal_train_data)
    positives = set(prompts.create_refusal_train_data()[0])
    out, current = [], None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            current = tok.string.lstrip("# ").strip()
        elif tok.type == tokenize.STRING and current is not None:
            try:
                s = eval(tok.string)
            except Exception:
                continue
            if s in positives:
                out.append((current, s))
    return out


EVAL_CATEGORIES = os.path.join(os.path.dirname(__file__), "..", "data", "refusal_eval_categories.json")


def eval_prompts_by_category(cats):
    """{category: [eval prompts]} from data/refusal_eval_categories.json, or None if absent."""
    if not os.path.exists(EVAL_CATEGORIES):
        return None
    labels = json.load(open(EVAL_CATEGORIES))["eval"]
    return {c: [p for p, k in labels.items() if k == c] for c in cats}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-eval", type=int, default=40, help="harmful eval prompts per ablation test")
    ap.add_argument("--n-harmless", type=int, default=40, help="harmless eval prompts per induction test")
    ap.add_argument("--no-causal", action="store_true", help="geometry only (no generation)")
    a = ap.parse_args()
    run = load_run(a.model)
    fw, model, fmt, blocks, r_hat = run.fw, run.model, run.fmt, run.blocks, run.r_hat
    L, P = run.r.layer, run.r.position_index

    labelled = categorised_train_prompts()
    cats = list(dict.fromkeys(c for c, _ in labelled))
    by_cat = {c: [p for cc, p in labelled if cc == c] for c in cats}
    _, train_neg = fw.concept.train_data_fn()
    eval_pos, eval_neg = fw.concept.eval_data_fn()
    eval_pos, eval_neg = eval_pos[:a.n_eval], eval_neg[:a.n_harmless]
    eval_by_cat = eval_prompts_by_category(cats)
    if eval_by_cat:
        print("eval prompts by category:", {c: len(v) for c, v in eval_by_cat.items()}, flush=True)
    print("categories:", {c: len(v) for c, v in by_cat.items()}, flush=True)

    h_pos = {c: residuals_at(model, fmt, blocks, by_cat[c], P)[:, L] for c in cats}      # [n_c, d]
    h_neg = residuals_at(model, fmt, blocks, train_neg, P)[:, L]
    neg_mean = h_neg.mean(0)
    d = {c: h_pos[c].mean(0) - neg_mean for c in cats}
    all_pos = t.cat([h_pos[c] for c in cats])
    d_all = all_pos.mean(0) - neg_mean
    loo = {c: t.cat([h_pos[k] for k in cats if k != c]).mean(0) - neg_mean for c in cats}

    out = {"model": a.model, "direction": run.coords, "categories": {c: len(by_cat[c]) for c in cats},
           "cos_r_hat_vs_unfiltered_all": cos(d_all, r_hat),
           "norm": {c: float(d[c].norm()) for c in cats}, "norm_all": float(d_all.norm()),
           "cos_with_r_hat": {c: cos(d[c], r_hat) for c in cats},
           "cos_loo_with_r_hat": {c: cos(loo[c], r_hat) for c in cats},
           "cos_matrix": {c: {k: cos(d[c], d[k]) for k in cats} for c in cats}}
    # Within-category spread vs between: how much of each prompt's r̂ projection is category?
    out["proj_r_hat_by_category"] = {c: float((h_pos[c] @ r_hat).mean()) for c in cats}
    out["proj_r_hat_harmless"] = float((h_neg @ r_hat).mean())
    print("cos(d_k, r̂):", {c[:12]: round(v, 2) for c, v in out["cos_with_r_hat"].items()}, flush=True)
    print("cos(d_-k, r̂):", {c[:12]: round(v, 2) for c, v in out["cos_loo_with_r_hat"].items()}, flush=True)
    offdiag = [out["cos_matrix"][c][k] for c in cats for k in cats if c != k]
    out["cos_offdiag_mean"], out["cos_offdiag_min"] = sum(offdiag) / len(offdiag), min(offdiag)
    print(f"pairwise cos: mean {out['cos_offdiag_mean']:.2f} min {out['cos_offdiag_min']:.2f}", flush=True)

    if not a.no_causal:
        abl = lambda v: run.ablate(unit(v))  # noqa: E731
        out["baseline"] = {"train_by_category": {c: behaviour(fw, by_cat[c])["rate"] for c in cats},
                           "eval": behaviour(fw, eval_pos),
                           "harmless": behaviour(fw, eval_neg),
                           "eval_ablate_r_hat": behaviour(fw, eval_pos, abl(r_hat))}
        out["causal"] = {}
        for c in cats:
            others = [p for k in cats if k != c for p in by_cat[k]]
            res = {"ablate_dk_own": behaviour(fw, by_cat[c], abl(d[c]))["rate"],
                   "ablate_dk_others": behaviour(fw, others, abl(d[c]))["rate"],
                   "ablate_dk_eval": behaviour(fw, eval_pos, abl(d[c])),
                   "ablate_loo_own": behaviour(fw, by_cat[c], abl(loo[c]))["rate"],
                   "ablate_r_hat_own": behaviour(fw, by_cat[c], abl(r_hat))["rate"],
                   "add_dk_harmless": behaviour(fw, eval_neg, run.add(d[c]))}
            line_extra = ""
            if eval_by_cat and eval_by_cat[c]:
                ev = eval_by_cat[c]
                res["baseline_eval_own"] = behaviour(fw, ev)
                res["ablate_dk_eval_own"] = behaviour(fw, ev, abl(d[c]))
                res["ablate_loo_eval_own"] = behaviour(fw, ev, abl(loo[c]))
                line_extra = (f" | eval-own (n={len(ev)}): base {res['baseline_eval_own']['rate']:4.0%} "
                              f"d_k {res['ablate_dk_eval_own']['rate']:4.0%} LOO {res['ablate_loo_eval_own']['rate']:4.0%}")
            out["causal"][c] = res
            print(f"{c[:28]:28s} base {out['baseline']['train_by_category'][c]:4.0%} | ablate own-dir: own {res['ablate_dk_own']:4.0%} "
                  f"others {res['ablate_dk_others']:4.0%} eval {res['ablate_dk_eval']['rate']:4.0%} | LOO on own {res['ablate_loo_own']:4.0%} "
                  f"| r̂ on own {res['ablate_r_hat_own']:4.0%} | induce {res['add_dk_harmless']['rate']:4.0%}" + line_extra, flush=True)
    print("saved", save_json(run.path("categories", "refusal"), out))


if __name__ == "__main__":
    main()
