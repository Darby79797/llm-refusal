"""Per-class circuit attributions vs per-class r-hat trajectory (Qwen2.5-0.5B). No torch.
  .venv/bin/python3 llm-refusal/scripts/analyze_perclass_circuit.py
Writes results/analysis/perclass_circuit_Qwen2.5-0.5B.json.

Trajectory proj[l] is read at the INPUT of block l, so block l's write = proj[l+1]-proj[l] (l=0..L-2);
block L-1's write is not observable in the trajectory. Circuit 'total'[l] = mean over 30 prompts of (attn_l+mlp_l) . r-hat.
Predicted trajectory: pred[l] = proj[0] + sum_{k<l} total[k].
"""
import json, os
import numpy as np, pandas as pd

R = "results"
M = "Qwen2.5-0.5B-Instruct"
tr = json.load(open(f"{R}/trajectory/{M}-refusal.json"))
cc = json.load(open(f"{R}/circuit/{M}-circuit-perclass.json"))
pl = tr["conditions"]["plain"]["proj_r_hat"]
traj = {"harmful": np.array(pl["harmful_mean"]), "harmless": np.array(pl["harmless_mean"])}
L = len(traj["harmful"])
att = {c: pd.DataFrame(cc[f"{c}_layer_attributions"]).set_index("layer") for c in traj}
heads = {c: {e["layer"]: e for e in cc[f"{c}_head_attributions"]} for c in traj}
SHARED_TOL, LATE, CONTRAST_MIN = 0.30, 14, 0.5

OUT = {"setup": {"trajectory_direction": tr["direction"],
                 "circuit_direction": {"layer": cc["direction_layer"], "pos": cc["direction_pos"]},
                 "shared_rule": f"both totals>0 and |h-b|/max(h,b)<={SHARED_TOL}; late = layer>={LATE}",
                 "contrastive_rule": f"|harmful-harmless|>={CONTRAST_MIN}",
                 "head_decomposed_layers": sorted(heads["harmful"])}}
pd.set_option("display.width", 250)

# 1. cumulative vs trajectory, per class
q1 = {}
for c in traj:
    p = traj[c]; tot = att[c].total.values
    inc = np.diff(p); t = tot[:L - 1]
    pred = p[0] + np.concatenate([[0], np.cumsum(t)])
    dif = inc - t
    q1[c] = {"corr_levels": float(np.corrcoef(p, pred)[0, 1]),
             "corr_increments": float(np.corrcoef(inc, t)[0, 1]),
             "spearman_increments": float(pd.Series(inc).corr(pd.Series(t), method="spearman")),
             "max_abs_level_gap": [int(np.abs(p - pred).argmax()), float(np.abs(p - pred).max())],
             "final_traj_L23": float(p[-1]), "final_pred_L23": float(pred[-1]),
             "max_abs_incr_gap": [int(np.abs(dif).argmax()), float(np.abs(dif).max())],
             "diverging_layers_abs_incr_gap_gt_0.3": [{"layer": int(l), "traj_inc": round(float(inc[l]), 2), "circuit": round(float(t[l]), 2)}
                                                       for l in np.where(np.abs(dif) > 0.3)[0]],
             "traj_peak": [int(p.argmax()), float(p.max())], "pred_peak": [int(pred.argmax()), float(pred.max())],
             "table": pd.DataFrame({"layer": range(L), "traj": p.round(2), "pred": pred.round(2)}).to_dict("records")}
    q1[c]["cum_through_L23_incl_last_block"] = float(p[0] + tot.sum())
OUT["q1_cumulative_vs_trajectory"] = q1

# 2. shared late writes
h, b = att["harmful"], att["harmless"]
rows = []
for l in range(L):
    th, tb = h.total[l], b.total[l]
    if th > 0 and tb > 0 and abs(th - tb) / max(th, tb) <= SHARED_TOL:
        r = {"layer": l, "late": l >= LATE, "harm_total": th, "harmless_total": tb,
             "harm_attn": h.attn[l], "harm_mlp": h.mlp[l], "harmless_attn": b.attn[l], "harmless_mlp": b.mlp[l],
             "dominant": {"harmful": "attn" if abs(h.attn[l]) > abs(h.mlp[l]) else "mlp",
                          "harmless": "attn" if abs(b.attn[l]) > abs(b.mlp[l]) else "mlp"}}
        if l in heads["harmful"]:
            def top(c):
                e = heads[c][l]; items = {f"H{k}": v for k, v in e["heads"].items()}; items["MLP"] = e["mlp"]
                return {k: round(v, 3) for k, v in sorted(items.items(), key=lambda kv: -abs(kv[1]))[:4]}
            r["top_components_harmful"], r["top_components_harmless"] = top("harmful"), top("harmless")
        rows.append(r)
OUT["q2_shared_writes"] = rows
OUT["q2_head_tables"] = {c: {str(l): {**{f"H{k}": round(v, 3) for k, v in e["heads"].items()}, "MLP": round(e["mlp"], 3)}
                             for l, e in heads[c].items()} for c in traj}

# 3. contrastive and post-peak negatives
contrast = (h.total - b.total)
OUT["q3_contrastive_writes"] = [{"layer": int(l), "diff": round(float(contrast[l]), 2), "harm": round(float(h.total[l]), 2),
                                 "harmless": round(float(b.total[l]), 2),
                                 "attn_diff": round(float(h.attn[l] - b.attn[l]), 2), "mlp_diff": round(float(h.mlp[l] - b.mlp[l]), 2)}
                                for l in range(L) if abs(contrast[l]) >= CONTRAST_MIN]
post = {}
for c in traj:
    pk = int(traj[c].argmax()); a = att[c]
    neg = a[(a.index >= pk) & (a.total < 0)]
    post[c] = {"traj_peak_layer": pk, "negative_writes_at_or_after_peak": [
        {"layer": int(l), "total": round(float(r.total), 2), "attn": round(float(r.attn), 2), "mlp": round(float(r.mlp), 2)} for l, r in neg.iterrows()],
        "sum_after_peak_block_writes": round(float(a.total[a.index >= pk].sum()), 2)}
OUT["q3_post_peak_negative"] = post

# 4. shared vs contrastive fraction of final harmful projection
fr = {}
for name, idx, cumf in [("L23_input_(traj_comparable)", L - 1, lambda c: att[c].total.values[:L - 1].sum()),
                        ("after_block_23_(full_circuit)", L, lambda c: att[c].total.values.sum())]:
    ch, cb = cumf("harmful"), cumf("harmless")
    sh = min(ch, cb); con = ch - cb
    fr[name] = {"cum_writes_harmful": float(ch), "cum_writes_harmless": float(cb), "shared_min": float(sh),
                "contrastive": float(con), "shared_frac_of_cum_writes": float(sh / ch) if ch else None,
                "contrastive_frac": float(con / ch) if ch else None}
t = {c: float(traj[c][-1] - traj[c][0]) for c in traj}
fr["trajectory_net_L0_to_L23"] = {"harmful": t["harmful"], "harmless": t["harmless"], "shared_min": min(t.values()),
                                  "contrastive": t["harmful"] - t["harmless"],
                                  "shared_frac": min(t.values()) / t["harmful"] if t["harmful"] else None}
fr["trajectory_absolute_L23"] = {"harmful": float(traj["harmful"][-1]), "harmless": float(traj["harmless"][-1])}
OUT["q4_shared_vs_contrastive"] = fr

os.makedirs(f"{R}/analysis", exist_ok=True)
json.dump(OUT, open(f"{R}/analysis/perclass_circuit_Qwen2.5-0.5B.json", "w"), indent=1)

# print
for c in traj:
    q = q1[c]
    print(f"\n== {c}: corr levels {q['corr_levels']:.3f}, corr increments {q['corr_increments']:.3f} (spearman {q['spearman_increments']:.3f}), "
          f"max level gap L{q['max_abs_level_gap'][0]}={q['max_abs_level_gap'][1]:.2f}, traj L23 {q['final_traj_L23']:.2f} vs pred {q['final_pred_L23']:.2f}, "
          f"peak traj {q['traj_peak']} pred {q['pred_peak']}")
    print("diverging layers:", q["diverging_layers_abs_incr_gap_gt_0.3"])
print("\n-- trajectory vs predicted (harm / harmless)")
print(pd.DataFrame({"L": range(L), "h_traj": traj["harmful"].round(2), "h_pred": [r["pred"] for r in q1["harmful"]["table"]],
                    "b_traj": traj["harmless"].round(2), "b_pred": [r["pred"] for r in q1["harmless"]["table"]]}).T.to_string(header=False))
print("\n-- per-layer totals"); print(pd.DataFrame({"harm": h.total, "harmless": b.total, "diff": contrast, "h_attn": h.attn, "h_mlp": h.mlp,
                                                    "b_attn": b.attn, "b_mlp": b.mlp}).round(2).T.to_string())
print("\n-- shared writes"); print(pd.DataFrame(rows).drop(columns=["dominant"]).round(2).to_string() if rows else "none")
for r in rows:
    if "top_components_harmful" in r: print(r["layer"], "H:", r["top_components_harmful"], "B:", r["top_components_harmless"])
print("\n-- contrastive"); print(pd.DataFrame(OUT["q3_contrastive_writes"]).to_string())
print("\n-- post-peak negatives"); print(json.dumps(post, indent=1))
print("\n-- shared vs contrastive"); print(json.dumps(fr, indent=1))
