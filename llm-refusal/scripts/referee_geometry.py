"""Referee checks on the refusal-geometry claims, from saved data only (no forward pass).

  1. Trajectory in cosine terms: mean projection onto r̂ / mean residual norm, per layer
     and class, in units of a random direction's typical |cos| (1/sqrt(d)); the late
     "shared rise" in raw and normalised terms; whether harmless std grows with its mean
     (a constant offset raises the mean, not the spread).
  2. Coordinate concentration of each saved r̂ (top-1/5/20 share of ||r̂||², excess
     kurtosis, participation ratio) against random Gaussian unit vectors; and r̂'s mass on
     candidate outlier ("massive activation") dimensions read from the checkpoint's own
     weights: RMSNorm gains (all layers) and early down_proj row norms. These are single
     tensors read with safetensors on the CPU; no model is built and nothing runs.
  3. Per-class circuit (0.5B): the late MLP writes, class ratio, and the last-block cancel.
  4. Edit-cost decomposition: interaction full − (before + after) and two-player Shapley
     shares per CE set.
  5. Category directions: the category-vs-pooled-harmful contrasts c_k = d_k − mean_j d_j
     reconstructed exactly from the saved Gram matrix (norms × cosines).
  6. Jailbreaks: AUROC of r̂ projection for wrapped prompts the model *complied* with vs
     harmless prompts (does r̂ at its layer read harmful content or the refusal decision?).

  .venv/bin/python3 llm-refusal/scripts/referee_geometry.py

Writes results/analysis/referee-geometry.json.
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported

import torch as t  # noqa: E402

from capability import random_direction  # noqa: E402
from probe import auroc, save_json  # noqa: E402

R = "results"
HUB = os.path.expanduser("~/.cache/huggingface/hub")
ORG = {"Qwen2.5": "Qwen", "Meta-Llama-3": "meta-llama", "Llama-3.1": "meta-llama", "Llama-2": "meta-llama"}


def r3(x):
    return None if x is None else (round(x, 4) if isinstance(x, float) else x)


# --- 1. trajectory in cosine terms ------------------------------------------------

def trajectory_cosine():
    out = {}
    for f in sorted(glob.glob(f"{R}/trajectory/*-refusal.json")):
        d = json.load(open(f))
        m, D, L = d["model"].split("/")[-1], d["direction"]["layer"], d["n_layers"]
        dim = t.load(f"{R}/{m}-refusal-direction.pt", map_location="cpu").numel()
        p = d["conditions"]["plain"]
        pr = p["proj_r_hat"]
        rows = []
        for l in range(L):
            row = {"layer": l, "norm_h": p["norm_harmful"][l], "norm_b": p["norm_harmless"][l]}
            for c, k in (("h", "harmful"), ("b", "harmless")):
                mu, sd, nm = pr[f"{k}_mean"][l], pr[f"{k}_std"][l], p[f"norm_{k}"][l]
                row[f"proj_{c}"], row[f"std_{c}"] = mu, sd
                row[f"cos_{c}"] = mu / nm                      # mean projection / mean norm
                row[f"cos_sqrt_d_{c}"] = mu / nm * math.sqrt(dim)  # in units of a random |cos|
            row["auroc"] = pr["auroc"][l]
            row["contrast_cos"] = row["cos_h"] - row["cos_b"]
            rows.append(row)
        post = rows[D:]
        i_min = D + min(range(len(post)), key=lambda i: post[i]["proj_h"])
        last = rows[-1]
        def rise(key, a, b):
            return rows[b][key] - rows[a][key]
        b_trough = D + min(range(len(post)), key=lambda i: post[i]["proj_b"])
        summ = {
            "d_model": dim, "direction_layer": D, "last_layer_input": L - 1,
            "harmful_post_peak_trough_layer": i_min,
            "raw": {"harmful_at_D": rows[D]["proj_h"], "harmful_trough": rows[i_min]["proj_h"], "harmful_last": last["proj_h"],
                    "harmless_at_D": rows[D]["proj_b"], "harmless_trough_after_D": rows[b_trough]["proj_b"],
                    "harmless_last": last["proj_b"]},
            "cos": {"harmful_at_D": rows[D]["cos_h"], "harmful_trough": rows[i_min]["cos_h"], "harmful_last": last["cos_h"],
                    "harmless_at_D": rows[D]["cos_b"], "harmless_trough_after_D": rows[b_trough]["cos_b"],
                    "harmless_last": last["cos_b"], "harmless_max_cos_any_layer": max(r["cos_b"] for r in rows),
                    "random_direction_typical_abs_cos": 1 / math.sqrt(dim)},
            "norm": {"at_D": rows[D]["norm_b"], "trough": rows[b_trough]["norm_b"], "last": last["norm_b"]},
            "harmless_late_rise_raw": rise("proj_b", b_trough, L - 1),
            "harmless_late_rise_cos": rise("cos_b", b_trough, L - 1),
            "harmless_late_rise_over_norm_growth": (rise("proj_b", b_trough, L - 1) / rise("norm_b", b_trough, L - 1)
                                                    if rise("norm_b", b_trough, L - 1) else None),
            "harmless_std_trough_to_last": [rows[b_trough]["std_b"], last["std_b"]],
            "harmless_late_rise_in_trough_stds": rise("proj_b", b_trough, L - 1) / max(rows[b_trough]["std_b"], 1e-9),
            "harmful_late_rise_raw": rise("proj_h", i_min, L - 1),
            "harmful_late_rise_cos": rise("cos_h", i_min, L - 1),
            "first_layer_auroc_ge_0.95": next((r["layer"] for r in rows if r["auroc"] >= 0.95), None),
            "first_layer_auroc_ge_0.99": next((r["layer"] for r in rows if r["auroc"] >= 0.99), None),
            "auroc_L1": rows[1]["auroc"], "cos_local_diff_r_hat_L1": p["cos_local_diff_r_hat"][1],
        }
        out[m] = {"summary": summ, "per_layer": [{k: r3(v) for k, v in r.items()} for r in rows]}
    return out


# --- 2. coordinate concentration + outlier dims ----------------------------------

def concentration(u):
    u = u.float() / u.float().norm()
    s = (u ** 2).sort(descending=True).values
    c = u - u.mean()
    return {"top1": float(s[0]), "top5": float(s[:5].sum()), "top20": float(s[:20].sum()),
            "excess_kurtosis": float((c ** 4).mean() / (c ** 2).mean() ** 2 - 3),
            "participation_ratio_frac_d": float(1 / (u ** 4).sum() / len(u))}


def gaussian_baseline(dim, n=2000, seed=0):
    g = t.Generator().manual_seed(seed)
    stats = [concentration(t.randn(dim, generator=g)) for _ in range(n)]
    out = {}
    for k in stats[0]:
        xs = sorted(s[k] for s in stats)
        out[k] = {"mean": sum(xs) / n, "p99": xs[int(0.99 * n)]}
    return out


def snapshot(m):
    org = next((o for p, o in ORG.items() if m.startswith(p)), None)
    if org is None:
        return None
    snaps = glob.glob(f"{HUB}/models--{org}--{m}/snapshots/*/")
    return snaps[0] if snaps else None


def read_tensors(snap, want):
    """{name: tensor} for the names in `want` (a predicate), via safetensors on the CPU."""
    from safetensors import safe_open
    out = {}
    for f in sorted(glob.glob(snap + "*.safetensors")):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if want(k):
                    out[k] = fh.get_tensor(k).float()
    return out


def outlier_dims(m, u, n_early=4, k=5):
    snap = snapshot(m)
    if snap is None:
        return None
    norms = read_tensors(snap, lambda k_: k_.endswith("layernorm.weight") or k_ == "model.norm.weight")
    downs = read_tensors(snap, lambda k_: k_.endswith("mlp.down_proj.weight") and int(k_.split(".")[2]) < n_early)
    u = u.float() / u.float().norm()
    dim = len(u)
    # Gain-based: dims whose median |gain| over every input_layernorm is the most extreme
    # (|log| of gain / median gain): massive-activation dims are typically gained down.
    ln = t.stack([v.abs() for kk, v in norms.items() if "input_layernorm" in kk])     # [L, d]
    rel = (ln / ln.median(dim=1, keepdim=True).values).median(dim=0).values          # [d]
    gain_low = rel.argsort()[:k]
    # Write-based: rows of the early down_proj with the largest norm (the early MLP that
    # creates massive activations writes them through a few huge rows).
    rown = t.stack([v.norm(dim=1) for v in downs.values()]).max(dim=0).values         # [d]
    write_hi = rown.argsort(descending=True)[:k]
    top_u = (u ** 2).argsort(descending=True)[:20]
    def mass(idx):
        return float((u[idx] ** 2).sum())
    # Massive-activation candidates: input-norm gain < 0.1x the layer median (this picks out
    # the published massive dims of Llama-2-7B, 1415 and 2533, among others).
    massive = [int(i) for i in rel.argsort() if rel[i] < 0.1]
    final = norms["model.norm.weight"].abs()
    final_rel = final / final.median()
    rand = random_direction(dim)
    rank = lambda i: int((u.abs() > u[i].abs()).sum())  # noqa: E731
    return {"snapshot": snap, "k": k,
            "massive_dims_gain_lt_0.1": massive,
            "r_hat_on_massive": [round(float(u[i]), 4) for i in massive],
            "rank_in_abs_r_hat": [rank(i) for i in massive],
            "final_norm_rel_gain": [round(float(final_rel[i]), 4) for i in massive],
            "r_hat_mass_on_massive": float((u[massive] ** 2).sum()) if massive else 0.0,
            "seeded_random_direction_mass_on_massive": float((rand[massive] ** 2).sum()) if massive else 0.0,
            "low_gain_dims": gain_low.tolist(), "low_gain_rel_median": [round(float(rel[i]), 3) for i in gain_low],
            "r_hat_mass_on_low_gain_dims": mass(gain_low),
            "big_early_down_proj_rows": write_hi.tolist(),
            "early_down_proj_row_norm_over_median": [round(float(rown[i] / rown.median()), 1) for i in write_hi],
            "r_hat_mass_on_big_write_dims": mass(write_hi),
            "chance_mass_k_dims": k / dim,
            "r_hat_top20_dims": top_u.tolist(),
            "overlap_top20_with_low_gain": sorted(set(top_u.tolist()) & set(gain_low.tolist())),
            "overlap_top20_with_big_write": sorted(set(top_u.tolist()) & set(write_hi.tolist()))}


def direction_coordinates():
    out = {}
    base = {}
    for f in sorted(glob.glob(f"{R}/*-refusal-direction.pt")):
        m = os.path.basename(f)[:-len("-refusal-direction.pt")]
        if m.startswith(("pretiebreak-", "regrown-")):
            continue
        u = t.load(f, map_location="cpu").float()
        meta = json.load(open(f[:-3] + ".json"))
        dim = len(u)
        if dim not in base:
            base[dim] = gaussian_baseline(dim)
        rec = {"layer": meta["layer"], "position_index": meta["position_index"], "d_model": dim,
               "raw_norm": float(u.norm()), **concentration(u), "gaussian_baseline": base[dim]}
        try:
            rec["outlier_dims"] = outlier_dims(m, u)
        except Exception as e:  # missing checkpoint shard etc.
            rec["outlier_dims"] = {"error": repr(e)}
        out[m] = rec
        od = rec["outlier_dims"] or {}
        print(f"{m:28s} top1 {rec['top1']:.3f} top5 {rec['top5']:.3f} top20 {rec['top20']:.3f} "
              f"(gauss top20 {base[dim]['top20']['mean']:.3f}, p99 {base[dim]['top20']['p99']:.3f}) "
              f"kurt {rec['excess_kurtosis']:.1f} (p99 {base[dim]['excess_kurtosis']['p99']:.2f}) | "
              f"mass on 5 low-gain dims {od.get('r_hat_mass_on_low_gain_dims', float('nan')):.3f}, "
              f"5 big-write dims {od.get('r_hat_mass_on_big_write_dims', float('nan')):.3f} "
              f"(chance {od.get('chance_mass_k_dims', float('nan')):.4f})", flush=True)
    return out


# --- 3. per-class circuit ----------------------------------------------------------

def perclass_circuit():
    c = json.load(open(f"{R}/circuit/Qwen2.5-0.5B-Instruct-circuit-perclass.json"))
    D = c["direction_layer"]
    rows = []
    for h, b in zip(c["harmful_layer_attributions"], c["harmless_layer_attributions"]):
        rows.append({"layer": h["layer"], "harm_attn": r3(h["attn"]), "harm_mlp": r3(h["mlp"]),
                     "harmless_attn": r3(b["attn"]), "harmless_mlp": r3(b["mlp"]),
                     "mlp_ratio_harmless_over_harmful": r3(b["mlp"] / h["mlp"]) if abs(h["mlp"]) > 0.1 else None})
    late = [r for r in rows if D + 2 < r["layer"] < c["num_layers"] - 1]
    last = rows[-1]
    return {"late_blocks": f"L{D + 3}-L{c['num_layers'] - 2}",
            "late_mlp_sum_harmful": r3(sum(r["harm_mlp"] for r in late)),
            "late_mlp_sum_harmless": r3(sum(r["harmless_mlp"] for r in late)),
            "late_attn_sum_harmful": r3(sum(r["harm_attn"] for r in late)),
            "late_attn_sum_harmless": r3(sum(r["harmless_attn"] for r in late)),
            "last_block_mlp_harmful_harmless": [last["harm_mlp"], last["harmless_mlp"]],
            "embedding_and_L0_L2_mlp_shared": [rows[i]["harm_mlp"] for i in range(3)],
            "per_layer": rows,
            "cannot_tell": "Attributions are scalars (each component's write · r̂, averaged over prompts). "
                           "Which residual coordinates carry those writes, whether they coincide with r̂'s "
                           "top coordinates or with outlier dims, and their per-prompt spread are not saved."}


# --- 4. edit-cost interaction ----------------------------------------------------

def edit_cost():
    out = {}
    for f in sorted(glob.glob(f"{R}/edit-cost/*-refusal.json")):
        d = json.load(open(f))
        m, S = d["model"].split("/")[-1], d["specs"]
        full_key = "noemb" if "noemb" in S else "full"  # lt:D / ge:D exclude the embedding
        rec = {"full_spec": full_key}
        for ce in ("ce_alpaca", "ce_pile", "ce_on_distribution"):
            z = S["none"][ce]
            early, late, full = S["lt:D"][ce] - z, S["ge:D"][ce] - z, S[full_key][ce] - z
            inter = full - early - late
            rec[ce] = {"early": r3(early), "late": r3(late), "full": r3(full), "sum_parts": r3(early + late),
                       "interaction": r3(inter), "interaction_over_full": r3(inter / full) if full else None,
                       "shapley_early": r3(early + inter / 2), "shapley_late": r3(late + inter / 2),
                       "shapley_late_share": r3((late + inter / 2) / full) if full else None,
                       "random": r3(S["random"][ce] - z) if "random" in S else None}
        out[m] = rec
    return out


# --- 5. category contrasts ---------------------------------------------------------

def categories():
    d = json.load(open(f"{R}/categories/Qwen2.5-0.5B-Instruct-refusal.json"))
    cats = list(d["categories"])
    n = t.tensor([d["norm"][c] for c in cats], dtype=t.float64)
    C = t.tensor([[d["cos_matrix"][a][b] for b in cats] for a in cats], dtype=t.float64)
    G = n[:, None] * C * n[None, :]                       # Gram of d_k = mu_k - mu_harmless
    K = len(cats)
    J = t.eye(K, dtype=t.float64) - 1.0 / K               # centring: c_k = d_k - mean_j d_j (equal n per category)
    Gc = J @ G @ J                                        # Gram of the category-vs-pooled contrasts
    nc = Gc.diag().clamp_min(0).sqrt()
    Cc = Gc / (nc[:, None] * nc[None, :])
    norm_dbar = float(G.mean().sqrt())                    # |mean_k d_k|, must equal the saved norm_all
    rd = n * t.tensor([d["cos_with_r_hat"][c] for c in cats], dtype=t.float64)   # r̂·d_k
    rc = rd - rd.mean()                                                          # r̂·c_k
    off = [float(Cc[i, j]) for i in range(K) for j in range(K) if i != j]
    return {
        "reconstruction_check": {"norm_mean_dk_from_gram": norm_dbar, "saved_norm_all": d["norm_all"]},
        "contrast_norm": {c: r3(float(nc[i])) for i, c in enumerate(cats)},
        "contrast_norm_over_dk_norm": {c: r3(float(nc[i] / n[i])) for i, c in enumerate(cats)},
        "between_category_var_over_shared_sq": r3(float(Gc.trace() / K) / norm_dbar ** 2),
        "contrast_cos_matrix": {a: {b: r3(float(Cc[i, j])) for j, b in enumerate(cats)} for i, a in enumerate(cats)},
        "contrast_cos_offdiag_mean": r3(sum(off) / len(off)), "contrast_cos_offdiag_min": r3(min(off)),
        "contrast_cos_offdiag_max": r3(max(off)),
        "contrast_cos_offdiag_expected_if_unstructured": r3(-1 / (K - 1)),
        "contrast_cos_with_r_hat": {c: r3(float(rc[i] / nc[i])) for i, c in enumerate(cats)},
        "note": "c_k sum to zero, so their mean pairwise cosine is forced near -1/(K-1) = -0.125 when norms are "
                "similar; structure shows up as pairs well above that. cos(c_k, r̂) measures how much a category's "
                "excess lies along r̂ (r̂ ≈ mean d_k, cos 0.996), i.e. whether categories differ in r̂ strength. "
                "No noise floor: with 10 prompts per category, ||c_k|| includes sampling noise the saved data can't size.",
    }


# --- 6. jailbreak: complied-but-harmful vs harmless -------------------------------

def jailbreaks():
    out = {}
    for f in sorted(glob.glob(f"{R}/jailbreak/*-refusal.json")):
        d = json.load(open(f))
        m = d["model"].split("/")[-1]
        hb = d["harmless"]["proj"]
        rec = {}
        for name, tp in d["templates"].items():
            comp = [p for p, r in zip(tp["proj"], tp["refused"]) if not r]
            ref = [p for p, r in zip(tp["proj"], tp["refused"]) if r]
            rec[name] = {"n_complied": len(comp), "n_refused": len(ref),
                         "auroc_complied_vs_harmless": r3(auroc(comp, hb)) if comp else None,
                         "auroc_refused_vs_harmless": r3(auroc(ref, hb)) if ref else None,
                         "mean_complied": r3(sum(comp) / len(comp)) if comp else None,
                         "mean_harmless": r3(sum(hb) / len(hb))}
        out[m] = rec
    return out


# --- 7. lexical baseline for the early-layer "probe" -----------------------------

def lexical_baseline(models=("Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct", "Meta-Llama-3-8B-Instruct")):
    """AUROC on the eval prompts of a bag-of-embeddings probe: each prompt is the mean of its
    instruction's input-embedding rows (no forward pass, no position, no template); the probe
    direction is the train-set difference-in-means. If this is ~1, early-layer separation along
    r̂ needs no more than which words the prompt contains."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    import prompts
    tr_p, tr_n = prompts.create_refusal_train_data()
    ev_p, ev_n = prompts.create_refusal_eval_data()
    out = {}
    for m in models:
        snap = snapshot(m)
        if snap is None:
            continue
        emb = read_tensors(snap, lambda k: k == "model.embed_tokens.weight")["model.embed_tokens.weight"]
        tok = AutoTokenizer.from_pretrained(snap)
        def bag(ps):
            return t.stack([emb[tok(p, add_special_tokens=False)["input_ids"]].mean(0) for p in ps])
        Xp, Xn, Ep, En = bag(tr_p), bag(tr_n), bag(ev_p), bag(ev_n)
        w = Xp.mean(0) - Xn.mean(0)
        u = t.load(f"{R}/{m}-refusal-direction.pt", map_location="cpu").float()
        out[m] = {"auroc_bag_of_embeddings_diffmeans": r3(auroc((Ep @ w).tolist(), (En @ w).tolist())),
                  "auroc_bag_of_embeddings_along_r_hat": r3(auroc((Ep @ u).tolist(), (En @ u).tolist())),
                  "cos_bag_diff_r_hat": r3(float(w @ u / (w.norm() * u.norm())))}
        print(m, out[m], flush=True)
        del emb
    return out


def main():
    res = {"trajectory_cosine": trajectory_cosine()}
    for m, r in res["trajectory_cosine"].items():
        s = r["summary"]
        print(f"{m:28s} harmless raw {s['raw']['harmless_trough_after_D']:+.2f} -> {s['raw']['harmless_last']:+.2f} "
              f"| cos {s['cos']['harmless_trough_after_D']:+.4f} -> {s['cos']['harmless_last']:+.4f} "
              f"(random |cos| {s['cos']['random_direction_typical_abs_cos']:.4f}) | norm {s['norm']['trough']:.0f} -> {s['norm']['last']:.0f} "
              f"| harmful cos at D {s['cos']['harmful_at_D']:+.4f} trough {s['cos']['harmful_trough']:+.4f} last {s['cos']['harmful_last']:+.4f}")
    res["direction_coordinates"] = direction_coordinates()
    res["perclass_circuit_0.5B"] = perclass_circuit()
    res["edit_cost_interaction"] = edit_cost()
    res["category_contrasts_0.5B"] = categories()
    res["jailbreak_complied_vs_harmless"] = jailbreaks()
    res["lexical_baseline"] = lexical_baseline()
    print("saved", save_json(f"{R}/analysis/referee-geometry.json", res))


if __name__ == "__main__":
    main()
