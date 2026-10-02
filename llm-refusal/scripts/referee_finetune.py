"""Referee checks on the fine-tuning limb, from saved tensors only (no model is loaded).

  1. rank-one adapters: cosines of writes u (with and without r̂) and gates v across seeds,
     objectives (remove / induce / null), layers (0.5B sweep), and the regrown-model
     off-switch vs the clean one. v is compared only between adapters on the same
     down_proj (same d_ff neuron basis); the product sign(cos v)·cos u is the sign-free
     comparison of the full rank-one maps u vᵀ.
  2. how much of the regrown off-switch u lies in the span of known directions.
  3. regrow LoRAs: where each arm's ΔW = U Vᵀ writes (writers: r̂ / û⊥ share of the
     output map) or reads (readers: share of the input map along the regrown axis / û⊥),
     r32 vs r0, and per-layer ‖ΔW‖_F.
  4. logit lens of the directions through the final RMSNorm gain and the unembedding,
     read straight from the checkpoint's safetensors (two tensors; no model built).
  5. norms and cosines of other saved directions at the regrown axis's coordinates (Q5 controls).

    .venv/bin/python3 llm-refusal/scripts/referee_finetune.py

Writes results/analysis/referee-finetune.json.
"""
import glob
import json
import math
import os

import torch as t

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
FT = os.path.join(ROOT, "results", "finetune")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {"0.5B": "Qwen2.5-0.5B-Instruct", "7B": "Qwen2.5-7B-Instruct", "L3": "Meta-Llama-3-8B-Instruct"}
HF = {"0.5B": "Qwen--Qwen2.5-0.5B-Instruct", "L3": "meta-llama--Meta-Llama-3-8B-Instruct"}


def unit(x):
    x = x.float().flatten()
    return x / x.norm()


def cos(a, b):
    return float(unit(a) @ unit(b))


def perp(u, r):
    u = u.float().flatten()
    return u - (u @ r) * r


def load_dir(stem):
    v = t.load(os.path.join(ROOT, stem + ".pt"), weights_only=False)
    v = v.vector if hasattr(v, "vector") else v
    meta = json.load(open(os.path.join(ROOT, stem + ".json")))
    return v.float().flatten(), meta


def rank1(short, tag):
    w = t.load(os.path.join(FT, f"{short}-refusal-rank1-{tag}-adapters.pt"), weights_only=True)[0]
    meta = json.load(open(os.path.join(FT, f"{short}-refusal-rank1-{tag}.json")))
    return w["U"][:, 0].float(), w["V"][:, 0].float(), meta["adapter_layers"][0]


def r3(x):
    return round(float(x), 3)


def rank1_block(key, tags, r_hat):
    short = MODELS[key]
    ad = {tg: rank1(short, tg) for tg in tags if os.path.exists(os.path.join(FT, f"{short}-refusal-rank1-{tg}-adapters.pt"))}
    out = {"cos_u_rhat": {tg: r3(cos(u, r_hat)) for tg, (u, v, L) in ad.items()},
           "layer": {tg: L for tg, (u, v, L) in ad.items()}, "pairs": {}}
    names = list(ad)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ua, va, la = ad[a]
            ub, vb, lb = ad[b]
            p = {"cos_u": r3(cos(ua, ub)), "cos_u_perp": r3(cos(perp(ua, r_hat), perp(ub, r_hat)))}
            if la == lb:
                cv = cos(va, vb)
                p["cos_v"] = r3(cv)
                p["signed_cos_u"] = r3(math.copysign(1, cv) * cos(ua, ub))
            out["pairs"][f"{a}|{b}"] = p
    return ad, out


def span_fraction(x, basis):
    """‖P x‖² / ‖x‖² for the span of `basis` (list of vectors)."""
    Q, _ = t.linalg.qr(t.stack([unit(b) for b in basis], 1))
    x = unit(x)
    return float(((Q.T @ x) ** 2).sum())


def regrow_maps(short, tag):
    meta = json.load(open(os.path.join(FT, f"{short}-refusal-regrow-{tag}.json")))
    ws = t.load(os.path.join(FT, f"{short}-refusal-regrow-{tag}-adapters.pt"), weights_only=True)
    mods = meta["modules"]
    out = []
    for i, w in enumerate(ws):
        out.append((i // len(mods), mods[i % len(mods)], w["U"].float(), w["V"].float()))
    return out


def out_share(U, V, d):
    """‖dᵀ U Vᵀ‖ / ‖U Vᵀ‖_F: the share of the update's output along unit d."""
    G = V.T @ V
    a = U.T @ d
    num = float(a @ G @ a)
    den = float(t.trace(U.T @ U @ G))
    return math.sqrt(max(num, 0) / den) if den > 0 else float("nan")


def in_share(U, V, d):
    """‖U Vᵀ d‖ / ‖U Vᵀ‖_F: the share of the update's input read along unit d."""
    b = V.T @ d
    num = float(b @ (U.T @ U) @ b)
    den = float(t.trace(U.T @ U @ (V.T @ V)))
    return math.sqrt(max(num, 0) / den) if den > 0 else float("nan")


def fro(U, V):
    return math.sqrt(float(t.trace(U.T @ U @ (V.T @ V))))


def regrow_block(short, r_hat, probes):
    """Per arm: per-layer ‖ΔW‖_F and output/input shares along the probe directions."""
    res = {}
    for tag in ["writers-r0", "writers-r32", "readers-r0", "readers-r32"]:
        if not os.path.exists(os.path.join(FT, f"{short}-refusal-regrow-{tag}-adapters.pt")):
            continue
        maps = regrow_maps(short, tag)
        d_model = r_hat.numel()
        rows = []
        for L, mod, U, V in maps:
            row = {"layer": L, "module": mod, "fro": r3(fro(U, V))}
            if U.shape[0] == d_model:   # writes the residual
                for k, d in probes.items():
                    row[f"out_{k}"] = r3(out_share(U, V, unit(d)))
            if V.shape[0] == d_model:   # reads the residual
                for k, d in probes.items():
                    row[f"in_{k}"] = r3(in_share(U, V, unit(d)))
            rows.append(row)
        res[tag] = rows
    return res


def summarise_regrow(res, d_model, rank=8):
    """Max over layers of each share, and the chance level sqrt(1/d)."""
    s = {"chance_share": r3(1 / math.sqrt(d_model))}
    for tag, rows in res.items():
        keys = sorted({k for r in rows for k in r if k.startswith(("out_", "in_"))})
        s[tag] = {}
        for k in keys:
            vals = [(r[k], r["layer"], r["module"]) for r in rows if k in r]
            vals.sort(reverse=True)
            s[tag][k] = {"max": vals[0][0], "at": f"L{vals[0][1]} {vals[0][2]}",
                         "mean": r3(sum(v[0] for v in vals) / len(vals))}
        by_layer = {}
        for r in rows:
            by_layer.setdefault(r["layer"], 0.0)
            by_layer[r["layer"]] += r["fro"] ** 2
        s[tag]["fro_by_layer"] = [r3(math.sqrt(by_layer[L])) for L in sorted(by_layer)]
    return s


def logit_lens(key, dirs, n_top=12):
    from safetensors import safe_open
    from tokenizers import Tokenizer
    snap = glob.glob(os.path.join(HUB, f"models--{HF[key]}", "snapshots", "*"))[0]
    tok = Tokenizer.from_file(os.path.join(snap, "tokenizer.json"))
    files = glob.glob(os.path.join(snap, "*.safetensors"))
    W = gamma = None
    for f in files:
        with safe_open(f, framework="pt") as sf:
            names = set(sf.keys())
            if "lm_head.weight" in names:
                W = sf.get_tensor("lm_head.weight")
            elif W is None and "model.embed_tokens.weight" in names and key == "0.5B":
                W = sf.get_tensor("model.embed_tokens.weight")   # tied
            if "model.norm.weight" in names:
                gamma = sf.get_tensor("model.norm.weight").float()
    refusal = ["I", "As", " I", " As"]
    comply = ["Sure", "Here", "To", "Certainly", "**", "1", "The", "Title", "Step", "Creating", "Hello"]
    ids = {s: tok.encode(s, add_special_tokens=False).ids for s in refusal + comply}
    ids = {s: v[0] for s, v in ids.items() if len(v) == 1}
    out = {}
    for name, d in dirs.items():
        g = (gamma * unit(d)).to(W.dtype)
        logits = t.cat([(W[i:i + 32768] @ g).float() for i in range(0, W.shape[0], 32768)])
        z = (logits - logits.mean()) / logits.std()
        order = t.argsort(logits)
        rank = t.empty_like(order)
        rank[order] = t.arange(len(order))
        pct = lambda i: r3(100 * float(rank[i]) / (len(order) - 1))
        out[name] = {
            "z_refusal_tokens": {s: r3(z[i]) for s, i in ids.items() if s in refusal},
            "pct_refusal_tokens": {s: pct(i) for s, i in ids.items() if s in refusal},
            "z_comply_tokens": {s: r3(z[i]) for s, i in ids.items() if s in comply},
            "top": [tok.decode([int(i)]) for i in order[-n_top:].flip(0)],
            "bottom": [tok.decode([int(i)]) for i in order[:n_top]],
        }
    return out


def main():
    res = {}
    rh = {k: unit(load_dir(f"results/{m}-refusal-direction")[0]) for k, m in MODELS.items()}

    # 1-2. rank-one adapters
    ad05, res["rank1_0.5B"] = rank1_block("0.5B", ["remove", "remove2", "null", "regrownremove", "regrownremove2",
                                                   "remove-L3", "remove-L6", "remove-L9", "remove-L11",
                                                   "remove-L16", "remove-L19"], rh["0.5B"])
    ad7, res["rank1_7B"] = rank1_block("7B", ["remove-s0", "remove-s1", "remove-s2", "induce-s0"], rh["7B"])
    adL, res["rank1_L3"] = rank1_block("L3", ["remove-s0", "remove-s1", "remove-s2", "remove3", "induce-s0",
                                              "regrownremove"], rh["L3"])

    # Clean seed-mean û⊥ (signs aligned by the gate v), Llama-3 and 0.5B.
    def seed_mean_perp(ad, tags, r):
        ref_v = ad[tags[0]][1]
        acc = 0
        for tg in tags:
            u, v, _ = ad[tg]
            acc = acc + math.copysign(1, cos(v, ref_v)) * unit(perp(u, r))
        return unit(acc), ref_v

    uL, vL = seed_mean_perp(adL, ["remove-s0", "remove-s1", "remove-s2", "remove3"], rh["L3"])
    u05, v05 = seed_mean_perp(ad05, ["remove"], rh["0.5B"])
    axisL, axisL_meta = load_dir("results/regrown-Meta-Llama-3-8B-Instruct-refusal-direction")
    ax05, ax05_meta = load_dir("results/regrown-Qwen2.5-0.5B-Instruct-refusal-direction")

    span = {}
    for key, ad, u_clean, v_clean, r, axis, induce in [
            ("L3", adL, uL, vL, rh["L3"], axisL, adL["induce-s0"][0]),
            ("0.5B", ad05, u05, v05, rh["0.5B"], ax05, None)]:
        u_rg, v_rg, _ = ad["regrownremove" if key == "L3" else "regrownremove2"]
        s = math.copysign(1, cos(v_rg, v_clean))
        basis = {"clean_u_perp": [u_clean], "+r_hat": [u_clean, r], "+regrown_axis": [u_clean, r, axis]}
        if induce is not None:
            basis["+induce_u"] = [u_clean, r, axis, induce]
        if key == "0.5B":
            basis["+layer_sweep_u_perp"] = [u_clean, r, axis] + [perp(ad[f"remove-L{L}"][0], r) for L in (3, 6, 9, 11, 16, 19)]
            basis["+null_u"] = basis["+layer_sweep_u_perp"] + [ad["null"][0]]
        span[key] = {"cos_v_regrown_clean": r3(cos(v_rg, v_clean)),
                     "signed_cos_u_regrown_clean_perp": r3(s * cos(u_rg, u_clean)),
                     "cos_u_regrown_axis": r3(cos(u_rg, axis)),
                     "v_norms_regrown_clean": [r3(v_rg.norm()), r3(v_clean.norm())],
                     "u_norms_regrown_clean": [r3(u_rg.norm()), r3(ad["remove-s0" if key == "L3" else "remove"][0].norm())],
                     "span_fraction": {k: r3(span_fraction(u_rg, b)) for k, b in basis.items()},
                     "chance_span_fraction_per_vector": r3(1 / r.numel())}
        # top neuron overlap of the gates: how many of v's 50 largest |entries| are shared
        k = 50
        top_rg = set(t.topk(v_rg.abs(), k).indices.tolist())
        top_cl = set(t.topk(v_clean.abs(), k).indices.tolist())
        span[key]["gate_top50_neuron_overlap"] = len(top_rg & top_cl)
        span[key]["gate_top50_chance"] = r3(k * k / v_rg.numel())
        # Gate concentration: share of ‖v‖² in its top 50 neurons
        span[key]["gate_top50_mass_regrown_clean"] = [r3((t.topk(v_rg.abs(), k).values ** 2).sum() / (v_rg ** 2).sum()),
                                                     r3((t.topk(v_clean.abs(), k).values ** 2).sum() / (v_clean ** 2).sum())]
    res["regrown_vs_clean"] = span

    # 3. regrow LoRAs
    res["regrow_L3"] = summarise_regrow(regrow_block(MODELS["L3"], rh["L3"],
                                                     {"rhat": rh["L3"], "u_perp": uL, "axis": axisL}), 4096)
    res["regrow_0.5B"] = summarise_regrow(regrow_block(MODELS["0.5B"], rh["0.5B"],
                                                       {"rhat": rh["0.5B"], "u_perp": u05, "axis": ax05}), 896)

    # 4. logit lens
    res["logit_lens_L3"] = logit_lens("L3", {
        "r_hat": rh["L3"], "clean_u_perp(signed by gate on harmful: -)": -uL, "clean_u_perp(+)": uL,
        "induce_u": adL["induce-s0"][0], "regrown_offswitch_u": adL["regrownremove"][0], "regrown_axis": axisL})
    res["logit_lens_0.5B"] = logit_lens("0.5B", {
        "r_hat": rh["0.5B"], "clean_u_perp(+)": u05, "clean_u_perp(-)": -u05,
        "regrown_offswitch_u": ad05["regrownremove2"][0], "regrown_axis_L12P-6": ax05})

    # 5. Q5 controls available on disk: other Llama-3 directions vs the regrown axis
    ctrl = {"regrown_axis": {"layer": axisL_meta["layer"], "pos": axisL_meta["position_index"], "norm": r3(axisL.norm())}}
    for c in ["empathy", "sycophancy", "hedging", "refusal_arditi_exact", "refusal"]:
        p = f"results/Meta-Llama-3-8B-Instruct-{c}-direction"
        if os.path.exists(os.path.join(ROOT, p + ".pt")):
            v, m = load_dir(p)
            ctrl[c] = {"layer": m["layer"], "pos": m["position_index"], "norm": r3(v.norm()),
                       "cos_regrown_axis": r3(cos(v, axisL)), "cos_clean_u_perp": r3(cos(v, uL))}
    res["q5_llama_directions"] = ctrl

    path = os.path.join(ROOT, "results", "analysis", "referee-finetune.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
