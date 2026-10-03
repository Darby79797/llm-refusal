"""Regrow dose-response: distinct refusal examples (n) vs exposures (example x times seen).

Replays train_adapters' batch order (finetune.py) from each run's seed, with no model:

    rng = random.Random(seed); order = []
    for step in 1..steps:
        if len(order) < batch_size: order += rng.sample(range(N), N)
        batch = order[:batch_size]; del order[:batch_size]

`rng` is local to train_adapters and nothing else draws from it (the refusal prompts
are picked by a separate random.Random(seed) in run_regrow; the LoRA init uses a torch
Generator). examples = benign + (refusal pairs) * refusal_repeat (default 1 when
absent from the JSON), so N = n_benign + n*repeat and the refusal rows are indices >= n_benign;
distinct example id = (i - n_benign) % n. Exposures = cumulative refusal rows seen; distinct =
distinct ids seen. Eval at step k runs after step k's update, so exposures are
counted through step k inclusive. No run in scope skipped a row on OOM (no
'skipped_rows' in any history), so every scheduled row was trained on.

Fits: binomial logistic of pos_rate (x eval_n prompts) on log2(1+exposures), log2(n),
log2(1+distinct) and two-predictor variants, on the readers-arm runs with n >= 1 at
steps 50/100/150/200; plus OLS of pos_log_odds on the same predictors.

Usage: analyze_regrow_exposure.py [--model Qwen2.5-0.5B-Instruct | Meta-Llama-3-8B-Instruct]
Also prints a matched-exposure table of repeat runs (e.g. r4x4, r2x8) vs n=16 x1 runs.

json/numpy/matplotlib only. Writes results/analysis/regrow_exposure.json and
plots/limbs/regrow-exposure.png (for 0.5B; other models get the model name in both paths).
"""
import argparse, json, random, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FT = ROOT / "results" / "finetune"
DEFAULT_MODEL = "Qwen2.5-0.5B-Instruct"
MODELS = (DEFAULT_MODEL, "Meta-Llama-3-8B-Instruct")
OUT_JSON = ROOT / "results" / "analysis" / "regrow_exposure.json"
OUT_PNG = ROOT / "plots" / "limbs" / "regrow-exposure.png"
PREFIX = f"{DEFAULT_MODEL}-refusal-regrow-"
MODEL = DEFAULT_MODEL


def set_model(model):
    global MODEL, PREFIX, OUT_JSON, OUT_PNG
    MODEL, PREFIX = model, f"{model}-refusal-regrow-"
    if model != DEFAULT_MODEL:
        OUT_JSON = OUT_JSON.with_name(f"regrow_exposure-{model}.json")
        OUT_PNG = OUT_PNG.with_name(f"regrow-exposure-{model}.png")
CHECKPOINTS = (50, 100, 150, 200)


def batch_order(n_examples, steps, batch_size, seed):
    """Exact replay of train_adapters' sampling: list of per-step index batches."""
    rng = random.Random(seed)
    order, batches = [], []
    for _ in range(steps):
        if len(order) < batch_size:
            order += rng.sample(range(n_examples), n_examples)
        batches.append(order[:batch_size])
        del order[:batch_size]
    return batches


def exposure_trace(d):
    nb, n, bs, steps = d["n_benign"], d["n_refusal_examples"], d["train_batch_size"], d["steps"]
    rep = max(1, int(d.get("refusal_repeat", 1)))
    batches = batch_order(nb + n * rep, steps, bs, d["seed"])
    refusal = range(n)                                             # distinct example ids
    per_step = [sum(1 for i in b if i >= nb) for b in batches]     # refusal rows in each step's batch
    first, last, count = {}, {}, {i: 0 for i in refusal}
    for s, b in enumerate(batches, 1):
        for i in b:
            if i >= nb:
                e = (i - nb) % n
                first.setdefault(e, s); last[e] = s; count[e] += 1
    rsteps = [s for s, c in enumerate(per_step, 1) for _ in range(c)]   # step of every refusal row
    cum = np.cumsum(per_step)
    ckpts = {}
    for e in d["history"]:
        if "pos_rate" not in e:
            continue
        k = e["step"]
        seen = [i for i in refusal if first.get(i, 10**9) <= k]
        ckpts[k] = {"step": k, "pos_rate": e["pos_rate"], "pos_log_odds": e["pos_log_odds"],
                    "exposures": int(cum[k - 1]) if k > 0 else 0,
                    "distinct_seen": len(seen),
                    "exposures_last50": int(sum(per_step[max(0, k - 50):k])) if k > 0 else 0,
                    "steps_since_last_exposure": (k - max(s for s in rsteps if s <= k)
                                                  if any(s <= k for s in rsteps) else None)}
    return {"first_seen": [first.get(i) for i in refusal],
            "times_seen_total": [count[i] for i in refusal],
            "refusal_steps": rsteps,
            "checkpoints": ckpts}


def load_runs():
    runs = []
    for f in sorted(FT.glob(PREFIX + "*.json")):
        tag = f.stem[len(PREFIX):]
        m = re.fullmatch(r"(readers|writers)-r(\d+)(?:x(\d+))?(?:-s(\d+)|-1000steps)?", tag)
        if not m or f.stem.endswith("-adapters"):
            continue
        d = json.loads(f.read_text())
        tr = exposure_trace(d)
        runs.append({"file": f.name, "arm": m.group(1), "n": d["n_refusal_examples"], "repeat": max(1, int(d.get("refusal_repeat", 1))),
                     "seed": d["seed"],
                     "steps": d["steps"], "eval_n": d.get("eval_n", 50), **tr})
    return runs


# ---------- fitting ----------
def logistic_fit(X, k, m, iters=100):
    """Binomial logistic (k successes of m) by Newton-Raphson with a tiny ridge; X includes intercept."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (k - m * p) - 1e-6 * b
        H = (X * (m * p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ b)), 1e-12, 1 - 1e-12)
    ll = float(np.sum(k * np.log(p) + (m - k) * np.log(1 - p)))
    phat = np.clip(k / m, 1e-12, 1 - 1e-12)
    ll_sat = float(np.sum(k * np.log(phat) + (m - k) * np.log(1 - phat)))
    return {"coef": b.tolist(), "loglik": ll, "deviance": 2 * (ll_sat - ll), "aic": 2 * len(b) - 2 * ll,
            "brier": float(np.mean((p - k / m) ** 2)), "pred": p}


def ols_fit(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return {"coef": b.tolist(), "r2": float(1 - r @ r / ((y - y.mean()) @ (y - y.mean()))),
            "rmse": float(np.sqrt(np.mean(r ** 2)))}


def repeat_comparison(rows):
    """For each repeat-run checkpoint, the n=16 x1 readers checkpoint (any seed/step 50-200) whose
    cumulative exposures are closest. Distinct examples differ (n vs 16), exposures ~match."""
    base = [x for x in rows if x["arm"] == "readers" and x["n"] == 16 and x["repeat"] == 1
            and "1000steps" not in x["file"]]
    out = []
    for x in rows:
        if x["arm"] != "readers" or x["repeat"] == 1 or not base:
            continue
        b = min(base, key=lambda y: (abs(y["exposures"] - x["exposures"]), abs(y["step"] - x["step"]), y["file"]))
        out.append({"repeat_file": x["file"], "n": x["n"], "repeat": x["repeat"], "step": x["step"],
                    "exposures": x["exposures"], "distinct_seen": x["distinct_seen"], "pos_rate": x["pos_rate"],
                    "pos_log_odds": x["pos_log_odds"], "match_file": b["file"], "match_step": b["step"],
                    "match_exposures": b["exposures"], "match_distinct_seen": b["distinct_seen"],
                    "match_pos_rate": b["pos_rate"], "match_pos_log_odds": b["pos_log_odds"]})
    return out


def fit_all(rows, label):
    if len(rows) < 3:
        return None
    k = np.array([r["pos_rate"] * r["eval_n"] for r in rows]).round()
    m = np.array([r["eval_n"] for r in rows], float)
    lo = np.array([r["pos_log_odds"] for r in rows])
    E = np.log2(1 + np.array([r["exposures"] for r in rows], float))
    N = np.log2(np.array([r["n"] for r in rows], float))
    D = np.log2(1 + np.array([r["distinct_seen"] for r in rows], float))
    S = np.log2(np.array([r["step"] for r in rows], float))
    R = np.log2(1 + np.array([r["exposures_last50"] for r in rows], float))
    one = np.ones(len(rows))
    designs = {"log2(1+exposures)": [E], "log2(n)": [N], "log2(1+distinct_seen)": [D],
               "log2(n) + log2(step)": [N, S], "log2(1+exposures) + log2(n)": [E, N],
               "log2(1+exposures_last50)": [R], "log2(1+exposures) + log2(1+exposures_last50)": [E, R]}
    out = {"label": label, "n_points": len(rows), "logistic": {}, "ols_log_odds": {}}
    for name, cols in designs.items():
        X = np.column_stack([one] + cols)
        f = logistic_fit(X, k, m)
        f.pop("pred")
        out["logistic"][name] = f
        out["ols_log_odds"][name] = ols_fit(X, lo)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    set_model(ap.parse_args().model)
    runs = load_runs()
    rows = [{"file": r["file"], "arm": r["arm"], "n": r["n"], "repeat": r["repeat"], "seed": r["seed"], "eval_n": r["eval_n"], **c}
            for r in runs for s, c in r["checkpoints"].items() if s in CHECKPOINTS]
    fit_rows = [x for x in rows if x["arm"] == "readers" and x["n"] >= 1 and x["repeat"] == 1
                and "1000steps" not in x["file"]]
    fits = {"readers_n_ge_1": fit_all(fit_rows, "readers arm, n>=1, steps 50-200"),
            "readers_n_ge_4": fit_all([x for x in fit_rows if x["n"] >= 4], "readers arm, n>=4, steps 50-200"),
            "readers_step200": fit_all([x for x in fit_rows if x["step"] == 200], "readers arm, n>=1, step 200 only"),
            "both_arms_n_ge_1": fit_all([x for x in rows if x["n"] >= 1 and x["repeat"] == 1],
                                        "readers+writers, n>=1, steps 50-200")}
    if any(x["repeat"] > 1 for x in rows):
        fits["readers_with_repeats"] = fit_all(
            [x for x in rows if x["arm"] == "readers" and x["n"] >= 1 and "1000steps" not in x["file"]],
            "readers arm incl. repeat runs (n = distinct), steps 50-200")
    fits = {k: v for k, v in fits.items() if v is not None}
    cmp_rows = repeat_comparison(rows)

    # Q2: within-n seed comparison
    seeds = {}
    for r in runs:
        if r["arm"] == "readers" and r["n"] in (8, 16) and r["repeat"] == 1:
            seeds.setdefault(str(r["n"]), []).append({
                "seed": r["seed"], "first_seen": r["first_seen"], "times_seen_total": r["times_seen_total"],
                "mean_first_seen": float(np.mean(r["first_seen"])),
                "refusal_steps": r["refusal_steps"],
                "by_step": {s: {kk: r["checkpoints"][s][kk] for kk in
                                ("pos_rate", "pos_log_odds", "exposures", "distinct_seen", "exposures_last50",
                                 "steps_since_last_exposure")} for s in CHECKPOINTS}})

    # print tables
    print(f"model: {MODEL}  ({len(runs)} runs)")
    print(f"{'run':<34}{'step':>5}{'rate':>6}{'logodds':>8}{'exp':>5}{'dist':>5}{'last50':>7}{'since':>6}")
    for x in sorted(rows, key=lambda x: (x["arm"], x["n"], x["seed"], x["file"], x["step"])):
        print(f"{x['file'][len(PREFIX):-5]:<34}{x['step']:>5}{x['pos_rate']:>6.2f}{x['pos_log_odds']:>8.2f}"
              f"{x['exposures']:>5}{x['distinct_seen']:>5}{x['exposures_last50']:>7}"
              f"{str(x['steps_since_last_exposure']):>6}")
    print("\nfirst-seen steps (readers n=8,16):")
    for n, ss in seeds.items():
        for s in ss:
            print(f"  n={n} s{s['seed']}: mean first {s['mean_first_seen']:.1f}; first {sorted(s['first_seen'])}  refusal-row steps {s['refusal_steps']}")
    print("\nrepeat runs vs n=16 x1 at nearest cumulative exposures:")
    if not cmp_rows:
        print("  (no repeat runs and/or no n=16 x1 runs present)")
    else:
        print(f"  {'repeat run':<22}{'step':>5}{'exp':>5}{'dist':>5}{'rate':>6}{'lo':>7}   {'n=16 run':<12}{'step':>5}{'exp':>5}{'dist':>5}{'rate':>6}{'lo':>7}")
        for c in cmp_rows:
            print(f"  {c['repeat_file'][len(PREFIX):-5]:<22}{c['step']:>5}{c['exposures']:>5}{c['distinct_seen']:>5}"
                  f"{c['pos_rate']:>6.2f}{c['pos_log_odds']:>7.2f}   {c['match_file'][len(PREFIX):-5]:<12}"
                  f"{c['match_step']:>5}{c['match_exposures']:>5}{c['match_distinct_seen']:>5}"
                  f"{c['match_pos_rate']:>6.2f}{c['match_pos_log_odds']:>7.2f}")
    for key, f in fits.items():
        print(f"\n{f['label']}  ({f['n_points']} points)")
        print(f"  {'predictor':<46}{'logLik':>9}{'AIC':>8}{'dev':>8}{'brier':>8}{'OLS R2(lo)':>11}{'rmse':>7}")
        for name, lf in f["logistic"].items():
            o = f["ols_log_odds"][name]
            print(f"  {name:<46}{lf['loglik']:>9.1f}{lf['aic']:>8.1f}{lf['deviance']:>8.1f}{lf['brier']:>8.4f}"
                  f"{o['r2']:>11.3f}{o['rmse']:>7.2f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "sampling_replay": "exact replay of finetune.train_adapters: random.Random(seed); full permutation "
                           "rng.sample(range(N), N) appended whenever < batch_size indices remain; refusal "
                           "examples are indices n_benign..N-1; exposures counted through the eval step",
        "caveat": "runs written 2026-09-30 (readers/writers r32, r0) predate the commit that holds this code "
                  "(c7e1110, 2026-10-02); the sampling lines are unchanged since that first commit, but the "
                  "exact code those runs used is not in git",
        "model": MODEL, "rows": rows, "fits": fits, "seeds_n8_n16": seeds,
        "repeat_vs_n16_matched_exposures": cmp_rows}, indent=1, default=str))

    # ---------- plot ----------
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"lines.linewidth": 1.0, "axes.linewidth": 0.7, "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    ns = sorted({x["n"] for x in rows if x["n"] >= 1}) or [1]
    cmap = plt.get_cmap("viridis")
    col = {n: cmap(i / max(len(ns) - 1, 1) * 0.9) for i, n in enumerate(ns)}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    have_fit = "readers_n_ge_1" in fits
    if have_fit:
        f_e = fits["readers_n_ge_1"]["logistic"]["log2(1+exposures)"]["coef"]
        f_n = fits["readers_n_ge_1"]["logistic"]["log2(n)"]["coef"]
    jit = np.random.default_rng(0)
    for x in rows:
        if x["n"] < 1:
            continue
        mk = "o" if x["arm"] == "readers" else "^"
        fc = col[x["n"]] if x["arm"] == "readers" else "none"
        kw = dict(marker=mk, s=10 + x["step"] / 8, facecolors=fc, edgecolors=col[x["n"]], linewidths=0.8, alpha=0.85)
        axes[0].scatter(x["exposures"] + jit.uniform(-0.15, 0.15), x["pos_rate"], **kw)
        axes[1].scatter(x["exposures"] + jit.uniform(-0.15, 0.15), x["pos_log_odds"], **kw)
        axes[2].scatter(x["n"] * 2 ** jit.uniform(-0.08, 0.08), x["pos_rate"], **kw)
    if have_fit:
        xx = np.linspace(0, max(36, max(x["exposures"] for x in rows) + 1), 200)
        axes[0].plot(xx, 1 / (1 + np.exp(-(f_e[0] + f_e[1] * np.log2(1 + xx)))), color="#555555", lw=1.2,
                     label="logistic on log2(1+exposures)")
        nn = np.logspace(0, np.log2(max(ns[-1], 2)), 200, base=2)
        axes[2].plot(nn, 1 / (1 + np.exp(-(f_n[0] + f_n[1] * np.log2(nn)))), color="#555555", lw=1.2,
                     label="logistic on log2(n)")
    for ax in axes[:2]:
        ax.set_xlabel("cumulative refusal-example exposures by checkpoint")
    axes[0].set_ylabel("harmful-prompt refusal rate"); axes[1].set_ylabel("refusal log-odds")
    axes[2].set_xscale("log", base=2); axes[2].set_xlabel("distinct refusal examples n"); axes[2].set_ylabel("refusal rate")
    axes[0].set_title("rate vs exposures (all checkpoints)"); axes[1].set_title("log-odds vs exposures")
    axes[2].set_title("rate vs n (all checkpoints)")
    for n in ns:
        axes[1].scatter([], [], color=col[n], s=14, label=f"n={n}")
    axes[1].scatter([], [], marker="o", facecolors="none", edgecolors="#555555", s=14, label="readers (filled)")
    axes[1].scatter([], [], marker="^", facecolors="none", edgecolors="#555555", s=14, label="writers (open)")
    axes[1].legend(fontsize=6.5, loc="lower right", ncol=2)
    axes[0].legend(fontsize=6.5, loc="upper left"); axes[2].legend(fontsize=6.5, loc="upper left")
    fig.text(0.5, 0.005, f"{MODEL} regrow; marker size grows with step (50-200). Exposures replayed exactly from the seeded batch order.",
             ha="center", fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(OUT_PNG, dpi=130); plt.close(fig)
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)} and {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
