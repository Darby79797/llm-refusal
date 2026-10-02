"""Figures for the limbs analysis (regrow dose, edit cost, trajectory, jailbreak). No torch."""
import json, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "plots" / "limbs"
OUT.mkdir(parents=True, exist_ok=True)
STATS = {}
HARM, HLESS = "#C44E52", "#4C72B0"
GREY, GREEN, DGREY = "#8C8C8C", "#6A9F58", "#555555"
plt.rcParams.update({"lines.linewidth": 1.0, "axes.linewidth": 0.7, "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

def J(p): return json.load(open(ROOT / p))
def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=130); plt.close(fig)

# ---------- 1. regrow dose ----------
FT = "results/finetune/Qwen2.5-0.5B-Instruct-refusal-regrow-"
def endpoint(path, step=200):
    h = J(path)["history"]
    ev = [e for e in h if "pos_rate" in e]
    e = next((e for e in ev if e["step"] == step), ev[-1])
    return e["step"], e["pos_rate"], e["pos_log_odds"], ev[-1]["pos_log_odds"], ev[-1]["step"]

def collect(arm, ns):
    pts = []
    for n in ns:
        files = [f"{FT}{arm}-r{n}-s{s}.json" for s in (0, 1, 2)] if n not in (0, 32) else [f"{FT}{arm}-r{n}.json"]
        for f in files:
            if (ROOT / f).exists():
                st, r, _, lo, _ = endpoint(f)
                pts.append((n, r, lo, st))
    return np.array(pts)
rd = collect("readers", [0, 1, 2, 4, 8, 16, 32])
wr = collect("writers", [4, 8, 32])
STATS["regrow"] = {"readers": rd.tolist(), "writers": wr.tolist(), "columns": ["n", "rate_step200", "final_log_odds", "step_used"]}
xpos = lambda n: np.where(n == 0, -1.0, np.log2(np.maximum(n, 1)))  # n=0 drawn at x=-1
fit_pts = rd[rd[:, 0] > 0]
X, Y = np.log2(fit_pts[:, 0]), fit_pts[:, 1]
best = None
for k in np.linspace(0.05, 6, 600):
    for x0 in np.linspace(-2, 7, 900):
        sse = np.sum((1 / (1 + np.exp(-k * (X - x0))) - Y) ** 2)
        if best is None or sse < best[0]: best = (sse, k, x0)
_, k, x0 = best
STATS["regrow"]["fit"] = {"k": float(k), "x0_log2n": float(x0), "n50": float(2 ** x0), "sse": float(best[0]),
                          "note": "readers arm, n>=1 only (n=0 not on log axis), all seeds, step-200 rate"}
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for j, (col, ylab) in enumerate([(1, "refusal rate at step 200"), (2, "final log-odds (pos)")]):
    a = ax[j]
    a.scatter(xpos(rd[:, 0]), rd[:, col], s=14, color=HARM, label="readers (per seed)", zorder=3)
    m = [(n, rd[rd[:, 0] == n, col].mean()) for n in np.unique(rd[:, 0])]
    a.plot(xpos(np.array([q[0] for q in m])), [q[1] for q in m], color=HARM, lw=0.8, label="readers mean")
    a.scatter(xpos(wr[:, 0]), wr[:, col], s=22, marker="^", facecolors="none", edgecolors=DGREY, label="writers", zorder=3)
    if j == 0:
        xs = np.linspace(0, 5.2, 200)
        a.plot(xs, 1 / (1 + np.exp(-k * (xs - x0))), color=GREY, ls="--", lw=0.8,
               label=f"logistic fit: 50% at n={2**x0:.1f}")
    ticks = [-1, 0, 1, 2, 3, 4, 5]
    a.set_xticks(ticks); a.set_xticklabels(["0"] + [str(2 ** t) for t in ticks[1:]])
    a.set_xlabel("refusal examples n in the fine-tuning data (log2)"); a.set_ylabel(ylab)
    a.legend(fontsize=7)
ax[0].set_title("(a) rate vs n"); ax[1].set_title("(b) log-odds vs n")
save(fig, "regrow-dose-Qwen2.5-0.5B.png")

# ---------- 2. benign 1000 ----------
h = [e for e in J(FT + "readers-r0-1000steps.json")["history"] if "pos_rate" in e]
st = [e["step"] for e in h]
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
ax[0].plot(st, [e["pos_rate"] for e in h], color=HARM, marker="o", ms=2.5, label="refusal rate (harmful)")
ax[0].set_ylabel("refusal rate"); ax[0].set_ylim(-0.02, 1.02)
ax[1].plot(st, [e["pos_log_odds"] for e in h], color=GREEN, marker="o", ms=2.5, label="log-odds (harmful)")
ax[1].set_ylabel("log-odds")
for a in ax: a.set_xlabel("step"); a.legend(fontsize=7)
fig.suptitle("Qwen2.5-0.5B, benign-only fine-tune (r0), 1000 steps", fontsize=9)
STATS["benign_1000"] = {"steps": st, "rate": [e["pos_rate"] for e in h], "log_odds": [e["pos_log_odds"] for e in h]}
save(fig, "regrow-benign-1000.png")

# ---------- 3. edit cost ----------
models = ["Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct", "Meta-Llama-3-8B-Instruct"]
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
STATS["edit_cost"] = {}
for a, m in zip(axs, models):
    p = f"results/edit-cost/{m}-refusal.json"
    if not (ROOT / p).exists(): continue
    sp = J(p)["specs"]; base = sp["none"]["ce_alpaca"]
    rows = {s: (v["ce_alpaca"] - base, v["harmful"]["rate"]) for s, v in sp.items()}
    STATS["edit_cost"][m] = {s: {"dCE_alpaca": x, "harmful_rate": y} for s, (x, y) in rows.items()}
    for s, (x, y) in rows.items():
        if s in ("random", "full"): continue
        a.scatter(x, y, s=14, color=GREY, zorder=2)
        a.annotate(s.replace("results/finetune/", "").replace("Qwen2.5-0.5B-Instruct-refusal-", ""), (x, y), fontsize=5.5,
                   xytext=(3, 2), textcoords="offset points", color=DGREY)
    for s, c, lab in (("random", GREEN, "random"), ("full", HARM, "full")):
        if s in rows:
            a.scatter(*rows[s], s=40, color=c, zorder=3, label=lab, edgecolors="k", linewidths=0.4)
            a.annotate(s, rows[s], fontsize=7, xytext=(4, 4), textcoords="offset points")
    a.set_title(m); a.set_xlabel("ΔCE Alpaca (vs none)"); a.set_ylabel("harmful refusal rate"); a.legend(fontsize=7)
save(fig, "edit-cost.png")

# ---------- 4. trajectory ----------
tm = sorted(Path(ROOT / "results/trajectory").glob("*-refusal.json"))
n = len(tm); cols = min(3, n); rows_ = (n + cols - 1) // cols
fig, axs = plt.subplots(2 * rows_, cols, figsize=(4.2 * cols, 2.6 * 2 * rows_), squeeze=False,
                        gridspec_kw={"height_ratios": [2, 1] * rows_})
STATS["trajectory"] = {}
for i, f in enumerate(tm):
    d = json.load(open(f)); r, c = divmod(i, cols)
    a, b = axs[2 * r][c], axs[2 * r + 1][c]
    pr = d["conditions"]["plain"]["proj_r_hat"]
    hm, lm = np.array(pr["harmful_mean"], float), np.array(pr["harmless_mean"], float)
    pk = np.nanmax(hm); L = np.arange(len(hm)); dl = d["direction"]["layer"]
    a.plot(L, hm / pk, color=HARM, label="harmful"); a.plot(L, lm / pk, color=HLESS, label="harmless")
    a.axvline(dl, color=GREY, ls=":", lw=0.8, label=f"direction layer {dl}")
    a.set_title(d["model"].split("/")[-1], fontsize=8); a.set_ylabel("mean proj / harmful peak"); a.legend(fontsize=6)
    au = np.array(pr["auroc"], float)
    b.plot(L, au, color=GREEN, label="AUROC"); b.axvline(dl, color=GREY, ls=":", lw=0.8)
    b.set_ylim(0.4, 1.02); b.set_xlabel("layer"); b.set_ylabel("AUROC"); b.legend(fontsize=6)
    STATS["trajectory"][d["model"]] = {"direction_layer": dl, "n_layers": d["n_layers"], "harmful_peak": float(pk),
        "peak_layer": int(np.nanargmax(hm)), "auroc_at_direction_layer": float(au[dl]) if dl < len(au) else None}
for j in range(n, rows_ * cols):
    axs[2 * (j // cols)][j % cols].axis("off"); axs[2 * (j // cols) + 1][j % cols].axis("off")
save(fig, "trajectory-all.png")

# ---------- 5. jailbreak ----------
jm = sorted(Path(ROOT / "results/jailbreak").glob("*-refusal.json"))
fig, axs = plt.subplots(1, len(jm), figsize=(5 * len(jm), 4), squeeze=False)
STATS["jailbreak"] = {}
for a, f in zip(axs[0], jm):
    d = json.load(open(f)); T = d["templates"]; p0 = T["plain"]["proj_mean"]
    STATS["jailbreak"][d["model"]] = {k: {"proj_norm": v["proj_mean"] / p0, "refusal_rate": v["refusal_rate"]} for k, v in T.items()}
    STATS["jailbreak"][d["model"]]["harmless"] = {"proj_norm": d["harmless"]["proj_mean"] / p0, "refusal_rate": d["harmless"]["refusal_rate"]}
    for k, v in T.items():
        col = HARM if k == "plain" else GREY
        a.scatter(v["proj_mean"] / p0, v["refusal_rate"], s=18, color=col, zorder=3)
        a.annotate(k, (v["proj_mean"] / p0, v["refusal_rate"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    h = d["harmless"]
    a.scatter(h["proj_mean"] / p0, h["refusal_rate"], s=26, color=HLESS, marker="s", label="harmless prompts")
    a.scatter([], [], s=18, color=HARM, label="plain (harmful)"); a.scatter([], [], s=18, color=GREY, label="jailbreak templates")
    a.set_title(d["model"].split("/")[-1]); a.set_xlabel("mean projection / plain projection"); a.set_ylabel("refusal rate")
    a.legend(fontsize=7)
save(fig, "jailbreak.png")

(ROOT / "results/analysis").mkdir(parents=True, exist_ok=True)
json.dump(STATS, open(ROOT / "results/analysis/limbs-figures.json", "w"), indent=1)
print(json.dumps(STATS["regrow"]["fit"], indent=1)); print(json.dumps(STATS["trajectory"], indent=1))
print(np.array2string(rd, precision=3)); print(np.array2string(wr, precision=3))
