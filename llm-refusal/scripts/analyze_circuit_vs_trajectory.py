"""Compare saved per-layer contrastive circuit attributions with the saved r-hat trajectory (0.5B).

No torch. Writes results/analysis/circuit_vs_trajectory.json.
  .venv/bin/python3 llm-refusal/scripts/analyze_circuit_vs_trajectory.py

Conventions: trajectory proj[l] is read at the INPUT of block l, so
  proj[l+1] - proj[l] = (attn_l + mlp_l write onto r-hat) (exactly, class-mean-wise, same prompts/position/direction).
Circuit 'total'[l] = mean over 30 eval prompts (harmful) minus mean over 30 (benign) of attn_l+mlp_l output . r-hat at pos direction_pos.
"""
import json, os
import numpy as np, pandas as pd

R = "results"
OUT = {}
tr = json.load(open(f"{R}/trajectory/Qwen2.5-0.5B-Instruct-refusal.json"))
cc = json.load(open(f"{R}/circuit/Qwen2.5-0.5B-Instruct-circuit.json"))
pl = tr["conditions"]["plain"]["proj_r_hat"]
h, b = np.array(pl["harmful_mean"]), np.array(pl["harmless_mean"])
d = h - b
L = len(h)
att = pd.DataFrame(cc["contrastive_layer_attributions"]).set_index("layer")
OUT["setup"] = {"trajectory_direction": tr["direction"], "circuit_direction": {"layer": cc["direction_layer"], "position_index": cc["direction_pos"]},
                "note": "circuit file was produced with an older direction (L13) than the trajectory (L14, re-searched 2026-09-30); circuit uses first 30 harmful/30 benign eval prompts, trajectory all eval prompts"}

# Q1
q1 = att.loc[13:18].round(3).reset_index().to_dict("records")
neg = att[(att.total < 0)].round(3).reset_index().to_dict("records")
OUT["q1_layer_table_13_18"] = q1
OUT["q1_negative_total_layers"] = neg
heads = []
for e in cc["contrastive_head_attributions"]:
    for k, v in e["heads"].items():
        heads.append({"layer": e["layer"], "comp": f"H{k}", "val": v})
    heads.append({"layer": e["layer"], "comp": "MLP", "val": e["mlp"]})
hd = pd.DataFrame(heads)
OUT["q1_decomposed_layers"] = sorted(hd.layer.unique().tolist())
OUT["q1_head_mlp_negatives_sorted"] = hd.sort_values("val").head(8).round(3).to_dict("records")
OUT["q1_head_mlp_positives_sorted"] = hd.sort_values("val", ascending=False).head(6).round(3).to_dict("records")
OUT["q1_verified_components"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()} for c in cc["verified_components"]]

# Q2
inc = np.diff(d)                       # inc[l] = d[l+1]-d[l] = what layer l wrote (harm-harmless)
tot = att.total.values[:L - 1]
cum = np.cumsum(att.total.values)      # cum[l] = writes of layers 0..l
pred = d[0] + np.concatenate([[0], cum[:L - 1]])  # predicted d[l] (l=0..L-1), anchored at traj d[0]
OUT["q2"] = {
    "traj_diff_by_layer": d.round(2).tolist(),
    "cum_circuit_pred_anchored_L0": pred.round(2).tolist(),
    "corr_levels": float(np.corrcoef(d, pred)[0, 1]),
    "corr_increments_vs_total": float(np.corrcoef(inc, tot)[0, 1]),
    "spearman_increments": float(pd.Series(inc).corr(pd.Series(tot), method="spearman")),
    "max_abs_level_gap": float(np.abs(d - pred).max()), "argmax_gap_layer": int(np.abs(d - pred).argmax()),
    "traj_diff_peak": [int(d.argmax()), float(d.max())], "circuit_pred_peak": [int(pred.argmax()), float(pred.max())],
    "traj_diff_L14_L17_L23": [float(d[14]), float(d[17]), float(d[23])],
    "pred_L14_L17_L23": [float(pred[14]), float(pred[17]), float(pred[23])],
    "increments_table": pd.DataFrame({"layer": range(L - 1), "traj_inc": inc.round(2), "circuit_total": tot.round(2)}).to_dict("records"),
    "harmful_mean": np.round(h, 2).tolist(), "harmless_mean": np.round(b, 2).tolist(),
}

# Q3
rows = []
for f in sorted(os.listdir(f"{R}/circuit")):
    c = json.load(open(f"{R}/circuit/{f}"))
    a = pd.DataFrame(c["contrastive_layer_attributions"]).set_index("layer")
    dl = c["direction_layer"]
    post = a.loc[dl:dl + 4].total
    rows.append({"model": c["model_name"].split("/")[-1], "dir_layer_circuit": dl, "pos": c["direction_pos"], "n_layers": c["num_layers"],
                 "top_pos": {int(k): round(v, 2) for k, v in a.total.nlargest(3).items()},
                 "top_neg": {int(k): round(v, 2) for k, v in a.total.nsmallest(3).items()},
                 f"sum_last3": round(float(a.total.iloc[-3:].sum()), 2),
                 "sum_dirlayer_to_+4": round(float(post.sum()), 2), "min_in_dir..dir+4": [int(post.idxmin()), round(float(post.min()), 2)],
                 "sum_all": round(float(a.total.sum()), 2),
                 "n_neg_layers_after_dir": int((a.loc[dl:].total < 0).sum()), "n_layers_after_dir": int(len(a.loc[dl:]))})
OUT["q3"] = rows
os.makedirs(f"{R}/analysis", exist_ok=True)
json.dump(OUT, open(f"{R}/analysis/circuit_vs_trajectory.json", "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(att.round(2).T.to_string())
print("q1 neg heads/mlp", OUT["q1_head_mlp_negatives_sorted"]); print("pos", OUT["q1_head_mlp_positives_sorted"])
print("verified", OUT["q1_verified_components"])
q = OUT["q2"]; print({k: v for k, v in q.items() if k not in ("increments_table",)})
print(pd.DataFrame(q["increments_table"]).T.to_string())
print(pd.DataFrame(rows).to_string())
