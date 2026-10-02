"""Statistics referee for RESULTS.md's fine-tuning and Limb sections. No model, no GPU.

Reads saved JSON (results/finetune, results/edit-cost, results/capability, results/jailbreak,
results/categories, results/analysis, regrown evaluate generations) and computes every interval
the saved data supports: Wilson CIs, Newcombe differences, exact McNemar on paired items,
prompt-level bootstraps of AUROCs, seed heterogeneity and seed-level tests for the regrow dose
table, a cluster (run) bootstrap of the exposure logistic, and an estimate of the row-level SD
of ΔCE from the edit-cost (200/100 rows) vs capability (500/200 rows) prefix subsets.

Usage:
  .venv/bin/python3 llm-refusal/scripts/referee_stats.py   # -> results/analysis/referee-stats.json
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "analysis", "referee-stats.json")
Z = 1.959964
RNG = np.random.default_rng(0)
B = 4000


def load(p):
    with open(os.path.join(R, p)) as f:
        return json.load(f)


def wilson(k, n):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + Z**2 / n
    c = (p + Z**2 / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 4), "wilson95": wilson(k, n)}


def newcombe(k1, n1, k2, n2):
    """95% CI for p2 - p1 (unpaired, Newcombe hybrid score)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    fisher_p = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue
    return {"diff": round(d, 4), "newcombe95": [round(lo, 4), round(hi, 4)], "fisher_p": round(float(fisher_p), 4)}


def k_of(r, n):
    return int(round(r * n))


def mcnemar(a, b):
    """Exact McNemar on paired booleans a (before), b (after)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    p = stats.binomtest(min(n10, n01), n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return {"lost": n10, "gained": n01, "exact_p": round(float(p), 5)}


def paired_boot_diff(a, b, b_iter=B):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = RNG.integers(0, n, (b_iter, n))
    d = b[idx].mean(1) - a[idx].mean(1)
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def best_case_mcnemar_p(k_extra):
    """Smallest attainable two-sided exact McNemar p when the after-condition has k_extra more
    positives than before (all discordant pairs in one direction)."""
    return round(min(1.0, 2 * 0.5**k_extra), 4) if k_extra > 0 else 1.0


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = stats.rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auroc(score, label, b_iter=B):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n = len(score)
    vals = []
    for _ in range(b_iter):
        i = RNG.integers(0, n, n)
        v = auroc(score[i], label[i])
        if not math.isnan(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


# ---------------------------------------------------------------- regrow dose
def regrow_dose():
    out = {"per_run": {}, "pooled": {}, "heterogeneity": {}, "log_odds_tests": {}}
    by_n = {}
    for f in sorted(glob.glob(os.path.join(R, "finetune", "Qwen2.5-0.5B-Instruct-refusal-regrow-readers-r*.json"))):
        name = os.path.basename(f)
        if "1000steps" in name:
            continue
        d = json.load(open(f))
        n_ex, en = d["n_refusal_examples"], d["eval_n"]
        k = k_of(d["history"][-1]["pos_rate"] if "pos_rate" in d["history"][-1] else
                 [h for h in d["history"] if "pos_rate" in h][-1]["pos_rate"], en)
        lo = [h for h in d["history"] if "pos_rate" in h][-1]["pos_log_odds"]
        neg = d["final_negative"]
        out["per_run"][name] = {"n_examples": n_ex, "seed": d["seed"], **rate(k, en), "log_odds": round(lo, 3),
                                "harmless": rate(k_of(neg["neg_rate"], en), en)}
        by_n.setdefault(n_ex, []).append((k, en, lo))
    for n_ex, rs in sorted(by_n.items()):
        K, N = sum(r[0] for r in rs), sum(r[1] for r in rs)
        out["pooled"][str(n_ex)] = {"seeds": len(rs), **rate(K, N),
                                    "seed_log_odds": [round(r[2], 3) for r in rs],
                                    "seed_log_odds_mean": round(float(np.mean([r[2] for r in rs])), 3),
                                    "seed_log_odds_sd": round(float(np.std([r[2] for r in rs], ddof=1)), 3) if len(rs) > 1 else None}
        if len(rs) > 1 and K > 0:
            tab = np.array([[r[0], r[1] - r[0]] for r in rs])
            chi2, p, _, _ = stats.chi2_contingency(tab)
            # binomial SD of a single 50-prompt reading at the pooled rate vs the observed seed SD
            pbar = K / N
            out["heterogeneity"][str(n_ex)] = {
                "chi2": round(float(chi2), 2), "p": round(float(p), 5),
                "binomial_sd_pp": round(100 * math.sqrt(pbar * (1 - pbar) / rs[0][1]), 1),
                "seed_sd_pp": round(100 * float(np.std([r[0] / r[1] for r in rs], ddof=1)), 1)}
    lo = {n: [r[2] for r in rs] for n, rs in by_n.items()}
    t = out["log_odds_tests"]
    for a, b in [(1, 2), (2, 4), (1, 4), (4, 8)]:
        res = stats.ttest_ind(lo[a], lo[b], equal_var=False)
        t[f"{a}_vs_{b}"] = {"welch_t": round(float(res.statistic), 2), "p": round(float(res.pvalue), 4)}
    pooled_sd = float(np.sqrt(np.mean([np.var(lo[n], ddof=1) for n in (1, 2, 4)])))
    t["pooled_seed_sd_doses_1_2_4"] = round(pooled_sd, 3)
    t["dose1_mean_minus_dose0_in_seed_sd"] = round((np.mean(lo[1]) - lo[0][0]) / pooled_sd, 2)
    pts = [(n, v) for n in (0, 1, 2, 4) for v in lo[n]]
    rho = stats.spearmanr([p[0] for p in pts], [p[1] for p in pts])
    t["spearman_dose_0_1_2_4"] = {"rho": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 4), "points": len(pts)}
    # ED50 on log2(n), n>=1, binomial logistic on each seed's final reading; seed-resampling bootstrap
    def fit_ed50(rows):
        x = np.array([math.log2(r[0]) for r in rows]); k = np.array([r[1] for r in rows]); m = np.array([r[2] for r in rows])
        beta = logistic_irls(np.column_stack([np.ones_like(x), x]), k, m)
        return 2 ** (-beta[0] / beta[1]) if beta[1] > 0 else float("nan")
    rows = [(n, r[0], r[1]) for n, rs in by_n.items() if n >= 1 for r in rs]
    ed = fit_ed50(rows)
    boots = []
    for _ in range(2000):
        rr = []
        for n, rs in by_n.items():
            if n < 1:
                continue
            if len(rs) > 1:
                for j in RNG.integers(0, len(rs), len(rs)):
                    rr.append((n, rs[j][0], rs[j][1]))
            else:  # single seed: binomial resample only (understates seed variance)
                rr.append((n, int(RNG.binomial(rs[0][1], rs[0][0] / rs[0][1])), rs[0][1]))
        v = fit_ed50(rr)
        if not math.isnan(v) and v < 1e4:
            boots.append(v)
    out["ed50_log2n"] = {"ed50": round(ed, 2), "seed_bootstrap95": [round(float(np.percentile(boots, 2.5)), 1),
                                                                     round(float(np.percentile(boots, 97.5)), 1)],
                         "note": "doses 1-16 have 3 seeds (resampled), 32 has one (binomial resample only)"}
    # writers arm and Llama points against readers
    w = {}
    for tag, f in [("writers_r4", "Qwen2.5-0.5B-Instruct-refusal-regrow-writers-r4-s0.json"),
                   ("writers_r8", "Qwen2.5-0.5B-Instruct-refusal-regrow-writers-r8-s0.json"),
                   ("llama8b_readers_r8", "Meta-Llama-3-8B-Instruct-refusal-regrow-readers-r8-s0.json")]:
        d = load(os.path.join("finetune", f))
        hist = [h for h in d["history"] if "pos_rate" in h]
        w[tag] = {**rate(k_of(hist[-1]["pos_rate"], 50), 50),
                  "vs_0.5B_readers_r8_pooled": newcombe(sum(r[0] for r in by_n[8]), 150, k_of(hist[-1]["pos_rate"], 50), 50),
                  "inside_readers_seed_range_6_22pct": 0.06 <= hist[-1]["pos_rate"] <= 0.22}
    out["cross_arm"] = w
    return out


def logistic_irls(X, k, m, iters=100):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = m * p * (1 - p) + 1e-9
        z = eta + (k - m * p) / W
        new = np.linalg.solve(X.T @ (W[:, None] * X) + 1e-8 * np.eye(X.shape[1]), X.T @ (W * z))
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            break
        beta = new
    return beta


def binom_dev(X, k, m):
    beta = logistic_irls(X, k, m)
    p = np.clip(1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))), 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(k / (m * p)), 0.0)
        t2 = np.where(m - k > 0, (m - k) * np.log((m - k) / (m * (1 - p))), 0.0)
    dev = 2 * float(np.sum(t1 + t2))
    pearson = float(np.sum((k - m * p) ** 2 / (m * p * (1 - p))))
    return beta, dev, pearson


# ---------------------------------------------------------------- exposure logistic
def exposure():
    rows = [r for r in load("analysis/regrow_exposure.json")["rows"]
            if r["arm"] == "readers" and r["n"] >= 1 and 50 <= r["step"] <= 200]
    k = np.array([round(r["pos_rate"] * r["eval_n"]) for r in rows], float)
    m = np.array([r["eval_n"] for r in rows], float)
    e = np.log2(1 + np.array([r["exposures"] for r in rows], float))
    l50 = np.log2(1 + np.array([r["exposures_last50"] for r in rows], float))
    nn = np.log2(np.array([r["n"] for r in rows], float))
    st = np.log2(np.array([r["step"] for r in rows], float))
    one = np.ones_like(e)
    models = {"exposures": np.column_stack([one, e]), "n": np.column_stack([one, nn]),
              "n+step": np.column_stack([one, nn, st]), "exposures+last50": np.column_stack([one, e, l50]),
              "n*step": np.column_stack([one, nn, st, nn * st])}
    res = {}
    for name, X in models.items():
        beta, dev, pear = binom_dev(X, k, m)
        df = len(k) - X.shape[1]
        res[name] = {"deviance": round(dev, 1), "df": df, "dispersion_pearson": round(pear / df, 2)}
    phi = res["exposures+last50"]["dispersion_pearson"]
    dD = res["exposures"]["deviance"] - res["exposures+last50"]["deviance"]
    res["recency_quasiF"] = {"delta_dev": round(dD, 1), "phi": phi, "F": round(dD / phi, 2),
                             "p": round(float(stats.f.sf(dD / phi, 1, res["exposures+last50"]["df"])), 4)}
    runs = sorted(set(r["file"] for r in rows))
    rid = np.array([runs.index(r["file"]) for r in rows])
    d_nstep, d_rec, ed50 = [], [], []
    for _ in range(2000):
        pick = RNG.integers(0, len(runs), len(runs))
        idx = np.concatenate([np.where(rid == j)[0] for j in pick])
        _, de, _ = binom_dev(models["exposures"][idx], k[idx], m[idx])
        _, dn, _ = binom_dev(models["n+step"][idx], k[idx], m[idx])
        b2, dr, _ = binom_dev(models["exposures+last50"][idx], k[idx], m[idx])
        d_nstep.append(dn - de)
        d_rec.append(de - dr)
        b1 = logistic_irls(models["exposures"][idx], k[idx], m[idx])
        if b1[1] > 0:
            ed50.append(2 ** (-b1[0] / b1[1]) - 1)
    b1 = logistic_irls(models["exposures"], k, m)
    pct = lambda a: [round(float(np.percentile(a, 2.5)), 1), round(float(np.percentile(a, 97.5)), 1)]
    res["run_cluster_bootstrap"] = {
        "runs": len(runs), "points": len(rows),
        "dev(n+step) - dev(exposures)": {"point": round(res["n+step"]["deviance"] - res["exposures"]["deviance"], 1),
                                         "ci95": pct(d_nstep), "frac_gt_0": round(float(np.mean(np.array(d_nstep) > 0)), 3)},
        "dev(exposures) - dev(exposures+last50)": {"point": round(dD, 1), "ci95": pct(d_rec)},
        "ed50_exposures": {"point": round(2 ** (-b1[0] / b1[1]) - 1, 1), "ci95": pct(ed50)}}
    return res


# ---------------------------------------------------------------- rank-one
def rank1():
    out = {}
    base_k, n = 89, 99
    for f in sorted(glob.glob(os.path.join(R, "finetune", "*-refusal-rank1-*.json"))):
        d = json.load(open(f))
        if "before" not in d:
            continue
        name = os.path.basename(f)[:-5]
        row = {"layer": d["adapter_layers"], "seed": d["seed"], "cos_u_rhat": [round(a["cos_u_rhat"], 3) for a in d["adapters"]]}
        bk = k_of(d["before"]["pos_rate"], 99)
        for v in ("before", "after", "after_u_minus_rhat", "after_u_rhat_only"):
            if v in d:
                row[v] = rate(k_of(d[v]["pos_rate"], 99), 99)
        if "after_u_rhat_only" in d:
            row["u_rhat_only_vs_before"] = newcombe(bk, 99, row["after_u_rhat_only"]["k"], 99)
        out[name] = row
    # seed spread of the r̂-only variant where seeds exist; same-seed rerun
    out["_seed_spread"] = {
        "Qwen2.5-7B_u_rhat_only_s0_s1_s2": [out[f"Qwen2.5-7B-Instruct-refusal-rank1-remove-s{s}"]["after_u_rhat_only"]["rate"] for s in range(3)],
        "Llama-3-8B_u_rhat_only_s0_s1_s2": [out[f"Meta-Llama-3-8B-Instruct-refusal-rank1-remove-s{s}"]["after_u_rhat_only"]["rate"] for s in range(3)],
        "Qwen2.5-0.5B_L13_seed0_run1_vs_run2": [out["Qwen2.5-0.5B-Instruct-refusal-rank1-remove"]["after_u_rhat_only"]["rate"],
                                                out["Qwen2.5-0.5B-Instruct-refusal-rank1-remove2"]["after_u_rhat_only"]["rate"]]}
    return out


# ---------------------------------------------------------------- edit cost
def edit_cost():
    out = {"rates": {}, "ce_row_sd_estimate": {}}
    pairs = {"Qwen2.5-0.5B": ("Qwen2.5-0.5B-Instruct-refusal.json", "Qwen2.5-0.5B-Instruct-refusal-L14-P-4-ceonly.json"),
             "Qwen2.5-3B": ("Qwen2.5-3B-Instruct-refusal.json", "Qwen2.5-3B-Instruct-refusal-L21-P-4.json"),
             "Llama-3-8B": ("Meta-Llama-3-8B-Instruct-refusal.json", "Meta-Llama-3-8B-Instruct-refusal-L12-P-3.json")}
    zs = {"alpaca": [], "pile": []}
    detail = []
    for model, (ef, cf) in pairs.items():
        e = load(os.path.join("edit-cost", ef))
        c = load(os.path.join("capability", cf))
        sp = e["specs"]
        nh, nn = e["n"]["harmful"], e["n"]["harmless"]
        base_h, base_n = k_of(sp["none"]["harmful"]["rate"], nh), k_of(sp["none"]["harmless"]["rate"], nn)
        rr = {}
        for name, s in sp.items():
            kh, kn = k_of(s["harmful"]["rate"], nh), k_of(s["harmless"]["rate"], nn)
            short = name.split("/")[-1] if name.startswith("adapter") else name
            rr[short] = {"refusal": rate(kh, nh), "false_refusal": rate(kn, nn),
                         "refusal_vs_none": newcombe(base_h, nh, kh, nh),
                         "false_refusal_vs_none": {**newcombe(base_n, nn, kn, nn),
                                                   "best_case_paired_mcnemar_p": best_case_mcnemar_p(kn - base_n)},
                         "dce": {s2: round(s[f"ce_{s2}"] - sp["none"][f"ce_{s2}"], 4) for s2 in ("alpaca", "pile", "on_distribution")}}
        out["rates"][model] = rr
        cv = c["variants"]
        N = {"alpaca": c["n_alpaca"], "pile": c["n_pile"]}
        for var_e, var_c in (("full", "orthogonalized"), ("random", "random_orthogonalized")):
            if var_e not in sp:
                continue
            for s2 in ("alpaca", "pile"):
                n1 = e["n"][s2]
                d1 = sp[var_e][f"ce_{s2}"] - sp["none"][f"ce_{s2}"]
                dall = cv[var_c][f"ce_{s2}"] - cv["baseline"][f"ce_{s2}"]
                n2 = N[s2] - n1
                drest = (N[s2] * dall - n1 * d1) / n2  # row-weighted approximation of token weighting
                z = d1 - drest
                zs[s2].append((z, n1, n2))
                detail.append({"model": model, "variant": var_e, "set": s2, "delta_subset": round(d1, 4),
                               "delta_full_set": round(dall, 4), "delta_rest": round(drest, 4)})
        # on-distribution set is identical in both runs: numerical reproducibility of CE
        out.setdefault("on_dist_reproducibility", {})[model] = {
            "baseline_editcost": sp["none"]["ce_on_distribution"], "baseline_capability": cv["baseline"]["ce_on_distribution"]}
    out["ce_subset_detail"] = detail
    for s2, lst in zs.items():
        sig2 = np.mean([z**2 / (1 / a + 1 / b) for z, a, b in lst])
        sig = math.sqrt(sig2)
        n_used = 200 if s2 == "alpaca" else 100
        out["ce_row_sd_estimate"][s2] = {"pairs": len(lst), "row_sd_nats": round(sig, 3),
                                         "se_of_delta_at_edit_cost_n": round(sig / math.sqrt(n_used), 4),
                                         "ci95_halfwidth": round(Z * sig / math.sqrt(n_used), 4),
                                         "note": "crude: from |delta(first n rows) - delta(remaining rows)| across "
                                                 f"{len(lst)} model x variant pairs, rows treated as equal-weight"}
    se_a = out["ce_row_sd_estimate"]["alpaca"]["se_of_delta_at_edit_cost_n"]
    se_p = out["ce_row_sd_estimate"]["pile"]["se_of_delta_at_edit_cost_n"]
    r3 = out["rates"]["Qwen2.5-3B"]
    r05 = out["rates"]["Qwen2.5-0.5B"]
    out["additivity_3B_pile"] = {"lt+ge": round(r3["lt:D"]["dce"]["pile"] + r3["ge:D"]["dce"]["pile"], 4),
                                 "full": r3["full"]["dce"]["pile"],
                                 "residual": round(r3["full"]["dce"]["pile"] - r3["lt:D"]["dce"]["pile"] - r3["ge:D"]["dce"]["pile"], 4),
                                 "residual_in_single_delta_SE": round((r3["full"]["dce"]["pile"] - r3["lt:D"]["dce"]["pile"] - r3["ge:D"]["dce"]["pile"]) / se_p, 1),
                                 "note": "SE of a single ΔCE; a 3-term contrast on the same rows has smaller SE if positively correlated, up to sqrt(3)x larger if not"}
    out["late_vs_random_0.5B"] = {"ge:D": r05["ge:D"]["dce"], "random": r05["random"]["dce"],
                                  "se_alpaca": se_a, "se_pile": se_p,
                                  "note": "one random direction (seed 0); spread over random directions unmeasured"}
    return out


# ---------------------------------------------------------------- jailbreak
def jailbreak():
    out = {}
    for m in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct", "Meta-Llama-3-8B-Instruct"):
        d = load(f"jailbreak/{m}-refusal.json")
        T = d["templates"]
        mo = {"templates": {}}
        harm = np.array(d["harmless"]["proj"], float)
        for name, t in T.items():
            ref = np.array(t["refused"], bool)
            row = {"refusal": rate(int(ref.sum()), len(ref)), "n_minority": int(min(ref.sum(), (~ref).sum())),
                   "mcnemar_vs_plain": mcnemar(T["plain"]["refused"], ref) if name != "plain" else None}
            for key, lab in (("proj", "auroc_proj"), ("proj_last", "auroc_last"), ("proj_max", "auroc_max")):
                if key in t:
                    a = auroc(t[key], ref)
                    if not math.isnan(a):
                        row[lab] = {"point": round(a, 3), "boot95": boot_auroc(t[key], ref)}
            p = np.array(t["proj"], float)
            bm = [float(RNG.choice(p, len(p)).mean() - RNG.choice(harm, len(harm)).mean()) for _ in range(B)]
            row["proj_mean"] = round(float(p.mean()), 3)
            row["proj_mean_minus_harmless"] = {"point": round(float(p.mean() - harm.mean()), 3),
                                               "boot95": [round(float(np.percentile(bm, 2.5)), 3), round(float(np.percentile(bm, 97.5)), 3)]}
            mo["templates"][name] = row
        # pooled AUROC, cluster bootstrap over prompts (all templates of a resampled prompt kept)
        names = list(T)
        P = np.array([T[n]["proj"] for n in names], float)
        Y = np.array([T[n]["refused"] for n in names], bool)
        nP = P.shape[1]
        pooled = auroc(P.ravel(), Y.ravel())
        vals = [auroc(P[:, i].ravel(), Y[:, i].ravel()) for i in (RNG.integers(0, nP, nP) for _ in range(2000))]
        mo["pooled_auroc"] = {"point": round(pooled, 3), "saved": round(d["pooled_auroc_proj_predicts_refusal"], 3),
                              "cluster_boot95": [round(float(np.nanpercentile(vals, 2.5)), 3), round(float(np.nanpercentile(vals, 97.5)), 3)]}
        # within-template-centred pooled AUROC: does projection predict refusal beyond template identity?
        Pc = P - P.mean(1, keepdims=True)
        mo["pooled_auroc_template_centred"] = round(auroc(Pc.ravel(), Y.ravel()), 3)
        mp = P.mean(1)
        rr = Y.mean(1)
        r = float(np.corrcoef(mp, rr)[0, 1])
        zf = math.atanh(r)
        se = 1 / math.sqrt(len(names) - 3)
        mo["template_level_corr"] = {"r": round(r, 3), "n_templates": len(names),
                                     "fisher95": [round(math.tanh(zf - Z * se), 2), round(math.tanh(zf + Z * se), 2)]}
        out[m] = mo
    return out


# ---------------------------------------------------------------- categories
def categories():
    d = load("categories/Qwen2.5-0.5B-Instruct-refusal.json")
    out = {"per_category": {}, "pooled": {}}
    own = oth = loo = add = ev = 0
    for c, v in d["causal"].items():
        k_own = k_of(v["ablate_dk_own"], 10)
        out["per_category"][c] = {"baseline_own": rate(k_of(d["baseline"]["train_by_category"][c], 10), 10),
                                  "ablate_own": rate(k_own, 10), "ablate_others": rate(k_of(v["ablate_dk_others"], 80), 80),
                                  "ablate_eval": rate(k_of(v["ablate_dk_eval"]["rate"], 40), 40),
                                  "ablate_loo_own": rate(k_of(v["ablate_loo_own"], 10), 10),
                                  "add_harmless": rate(k_of(v["add_dk_harmless"]["rate"], 40), 40)}
        own += k_own; oth += k_of(v["ablate_dk_others"], 80); loo += k_of(v["ablate_loo_own"], 10)
        add += k_of(v["add_dk_harmless"]["rate"], 40); ev += k_of(v["ablate_dk_eval"]["rate"], 40)
    nc = len(d["causal"])
    out["pooled"] = {"ablate_own": rate(own, 10 * nc), "ablate_loo_own": rate(loo, 10 * nc),
                     "ablate_eval_sum_over_directions": rate(ev, 40 * nc), "add_harmless": rate(add, 40 * nc),
                     "baseline_own": rate(sum(k_of(x, 10) for x in d["baseline"]["train_by_category"].values()), 10 * nc)}
    out["note"] = ("'own' prompts are the 10 train prompts each direction was computed from (in-sample); "
                   "no random-10-prompt-subset null for the cosines is saved")
    return out


# ---------------------------------------------------------------- identity / regrown evaluates / inhibitor safety
def identity():
    out = {}
    for m in ("Qwen2.5-0.5B-Instruct", "Meta-Llama-3-8B-Instruct"):
        d = load(f"finetune/{m}-direction-identity.json")
        for arm, v in d["arms"].items():
            if "refusal" not in v:
                continue
            rf = v["refusal"]
            base = k_of(rf["none"]["rate"], 50)
            row = {}
            for c, x in rf.items():
                k = k_of(x["rate"], 50)
                row[c] = rate(k, 50)
                if c.startswith("ablate"):
                    row[c]["vs_none"] = newcombe(base, 50, k, 50)
            out[f"{m}:{arm}"] = row
    return out


def regrown_evaluates():
    out = {}
    files = {"0.5B regrown readers-r32 (search pick L12/P-6)": "regrown-Qwen2.5-0.5B-Instruct-refusal-evaluate-regrown-Qwen2.5-0.5B-Instruct-refusal-direction-generations.json",
             "Llama-3-8B regrown readers-r32 (new L16/P-1 axis)": "regrown-Meta-Llama-3-8B-Instruct-refusal-evaluate-regrown-Meta-Llama-3-8B-Instruct-refusal-direction-generations.json",
             "Llama-3-8B clean model, regrown axis": "regrownonclean-Meta-Llama-3-8B-Instruct-refusal-evaluate-regrown-Meta-Llama-3-8B-Instruct-refusal-direction-generations.json"}
    for tag, f in files.items():
        g = load(f)
        b = [x["detected"] for x in g["baseline"]]
        row = {}
        for c, v in g.items():
            det = [x["detected"] for x in v]
            row[c] = {**rate(sum(det), len(det)), "degenerate": round(float(np.mean([bool(x.get("degenerate")) for x in v])), 3)}
            if c in ("global_ablation", "layer_specific_ablation"):
                row[c]["mcnemar_vs_baseline"] = mcnemar(b, det)
                row[c]["paired_boot95_diff"] = paired_boot_diff(b, det)
        out[tag] = row
    return out


def inhibitor_safety():
    d = load("finetune/Qwen2.5-0.5B-Instruct-inhibitor-safety.json")["conditions"]
    out = {}
    for c, v in d.items():
        j = v["jbb"]
        out[c] = {"refusal": rate(k_of(j["rate"], 100), 100), "unsafe": rate(k_of(j["llamaguard_unsafe_rate"], 100), 100),
                  "degenerate": rate(k_of(j["degenerate_rate"], 100), 100)}
    out["inhibitor_vs_hook_unsafe"] = newcombe(k_of(d["global_ablation"]["jbb"]["llamaguard_unsafe_rate"], 100), 100,
                                               k_of(d["inhibitor"]["jbb"]["llamaguard_unsafe_rate"], 100), 100)
    out["inhibitor_vs_hook_refusal"] = newcombe(0, 100, k_of(d["inhibitor"]["jbb"]["rate"], 100), 100)
    out["note"] = "per-item labels not saved: unpaired tests only (paired would be tighter)"
    return out


# ---------------------------------------------------------------- judged sycophancy
def sycophancy():
    out = {}
    deltas = {}
    for m in ("0.5B", "1.5B", "3B", "7B"):
        d = load(f"analysis/judged-sycophancy-Qwen2.5-{m}-Instruct.json")
        row = {}
        for lab in ("judge", "detector"):
            a = [bool(x[lab]) for x in d["baseline"]]
            b = [bool(x[lab]) for x in d["ablated"]]
            row[lab] = {"baseline": rate(sum(a), 80), "ablated": rate(sum(b), 80),
                        "delta_pp": round(100 * (sum(b) - sum(a)) / 80, 1),
                        "paired_boot95_pp": [round(100 * v, 1) for v in paired_boot_diff(a, b)],
                        "mcnemar": mcnemar(a, b)}
            deltas[(m, lab)] = (np.array(a, float), np.array(b, float))
        # detector - judge gap in the delta, paired by prompt
        ja, jb = deltas[(m, "judge")]
        da, db = deltas[(m, "detector")]
        idx = RNG.integers(0, 80, (B, 80))
        g = ((db - da)[idx].mean(1) - (jb - ja)[idx].mean(1)) * 100
        row["detector_minus_judge_delta_pp"] = {"point": round(float(((db - da) - (jb - ja)).mean() * 100), 1),
                                                "boot95": [round(float(np.percentile(g, 2.5)), 1), round(float(np.percentile(g, 97.5)), 1)]}
        row["judge_detector_agreement_ablated"] = round(float(np.mean(jb == db)), 3)
        out[m] = row
    # is 1.5B's judge delta the largest? paired over the same 80 prompts
    ja, jb = deltas[("1.5B", "judge")]
    for other in ("0.5B", "3B", "7B"):
        oa, ob = deltas[(other, "judge")]
        idx = RNG.integers(0, 80, (B, 80))
        g = ((jb - ja)[idx].mean(1) - (ob - oa)[idx].mean(1)) * 100
        out[f"judge_delta_1.5B_minus_{other}_pp"] = {"point": round(float(((jb - ja) - (ob - oa)).mean() * 100), 1),
                                                     "boot95": [round(float(np.percentile(g, 2.5)), 1), round(float(np.percentile(g, 97.5)), 1)]}
    out["note"] = "single judge run (qwen3:4b), no human labels: judge validity is not estimable here"
    return out


def sycophancy_response_strengths():
    pts = {"baseline": 0.1125, "0.4": 0.25, "0.8": 0.40, "1.2": 0.525, "1.6": 0.6875, "2.0": 0.825, "2.5": 0.7625}
    out = {k: rate(k_of(v, 80), 80) for k, v in pts.items()}
    out["global_ablation_42.5_to_36"] = newcombe(34, 80, 29, 80)
    out["note"] = "phrase-detector scored; the same detector reads 2.2x the judge's rate on 0.5B ablated sycophancy"
    return out


def main():
    res = {"regrow_dose": regrow_dose(), "regrow_exposure": exposure(), "rank1": rank1(), "edit_cost": edit_cost(),
           "jailbreak": jailbreak(), "categories": categories(), "identity_n50": identity(),
           "regrown_evaluates": regrown_evaluates(), "inhibitor_safety_jbb": inhibitor_safety(),
           "judged_sycophancy": sycophancy(), "sycophancy_response_strengths": sycophancy_response_strengths()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
