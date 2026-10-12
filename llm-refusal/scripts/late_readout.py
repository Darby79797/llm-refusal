"""Is the late refusal stage downstream of r̂? Activation readout, no generation (plan 26).

The ablations say the late axis ⊥ r̂ is needed for the refusal template (Qwens) and r̂ for harm recognition; this reads
the causal chain off the residual stream instead. At P-1, for every layer: the late axis a_l is the layer's own
harmful-minus-harmless difference-in-means with r̂ projected out (from the train prompts, unit), and the readout is the
eval prompts' mean projection onto it, under:
  plain             harmful and harmless, no intervention (the gap = harmful - harmless)
  push              harmless + r̂ ×S added at r̂'s layer (all positions, as ablate_in_variants --rhat-push)
  ablate_all        harmful with r̂ ablated at every layer
  ablate_at         harmful with r̂ ablated only at r̂'s layer
  random_push / random_ablate   the same with a norm-matched random direction (controls)
  ablate_from<K>    harmful with r̂ ablated at layers >= K only (`--ablate-from K ...`: where do late layers read r̂?)
Reports per layer the fraction of the plain gap the push recreates ((push - harmless) / gap) and r̂ ablation removes
((harmful - ablated) / gap). `--axis A+B+...` also reads a fixed axis at every layer: the first principal axis of those
saved directions ⊥ r̂ (e.g. 0.5B's plan-22 pool, whose PC1 removes the template).

  .venv/bin/python3 llm-refusal/scripts/late_readout.py --model Qwen/Qwen2.5-0.5B-Instruct

Writes results/analysis/<short>-late-readout[-<tag>].json.
"""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from datatypes import DirectionVector  # noqa: E402
from probe import load_run, residuals_at, save_json, unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pos", type=int, default=-1, help="template position read (default P-1)")
    ap.add_argument("--push", type=float, default=1.0, help="r̂ multiplier for the push condition")
    ap.add_argument("--axis", default=None, help="'+'-joined saved direction stems: also read their PC1 ⊥ r̂")
    ap.add_argument("--n-train", type=int, default=128, help="train prompts per side for the per-layer axes")
    ap.add_argument("--ablate-from", type=int, nargs="*", default=[], help="also ablate r̂ only at layers >= each K")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    run = load_run(a.model)
    model, fmt, blocks, r, u = run.model, run.fmt, run.blocks, run.r, run.r_hat
    L = len(blocks)
    harmful, harmless = run.fw.concept.eval_data_fn()
    tr_h, tr_n = (x[:a.n_train] for x in run.fw.concept.train_data_fn())
    res = lambda prompts: residuals_at(model, fmt, blocks, prompts, a.pos)  # noqa: E731  [n, L, d] fp32 CPU

    # Per-layer axes from the train prompts (held out from the eval prompts read below), ⊥ r̂, unit.
    diff = res(tr_h).double().mean(0) - res(tr_n).double().mean(0)          # [L, d]
    diff = diff.float()
    diff = diff - (diff @ u)[:, None] * u
    axes = {"own": diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-8)}
    if a.axis:
        V = t.stack([unit(DirectionVector.load(s).vector.float().cpu()) for s in a.axis.split("+")])
        V = V - (V @ u)[:, None] * u
        pc1 = t.linalg.svd(V, full_matrices=False).Vh[0]
        pc1 = pc1 * t.sign(diff[L - 1] @ pc1)                                # sign: harmful side positive
        axes["fixed"] = pc1.expand(L, -1)

    g = t.Generator().manual_seed(a.seed)
    rnd = unit(t.randn(u.shape[-1], generator=g)) * r.vector.float().norm()
    rnd_dv = DirectionVector(vector=rnd.to(r.vector.dtype), layer=r.layer, position_index=r.position_index, score=0)
    at = [r.layer]
    ap_ = run.applier
    conds = {
        "harmful": (harmful, contextlib.nullcontext),
        "harmless": (harmless, contextlib.nullcontext),
        "push": (harmless, lambda: ap_.intervened(r, "add", a.push, layers=at)),
        "ablate_all": (harmful, lambda: ap_.intervened(r, "ablate")),
        "ablate_at": (harmful, lambda: ap_.intervened(r, "ablate", layers=at)),
        "random_push": (harmless, lambda: ap_.intervened(rnd_dv, "add", a.push, layers=at)),
        "random_ablate": (harmful, lambda: ap_.intervened(rnd_dv, "ablate")),
        **{f"ablate_from{k}": (harmful, lambda k=k: ap_.intervened(r, "ablate", layers=list(range(k, L))))
           for k in a.ablate_from},
    }
    proj = {}
    for name, (prompts, ctx) in conds.items():
        with ctx():
            h = res(prompts)                                                   # [n, L, d]
        proj[name] = {k: (h * ax[None]).sum(-1).mean(0) for k, ax in axes.items()}   # [L]
        proj[name]["rhat"] = (h @ u).mean(0)

    out = {"model": a.model, "pos": a.pos, "push": a.push, "axis": a.axis, "n_train": a.n_train, "ablate_from": a.ablate_from,
           "rhat": run.coords, "n_layers": L, "readout": {}}
    for k in [*axes, "rhat"]:
        gap = proj["harmful"][k] - proj["harmless"][k]
        frac = lambda x: (x / gap).tolist()  # noqa: E731
        out["readout"][k] = {
            "mean": {c: proj[c][k].tolist() for c in conds}, "gap": gap.tolist(),
            "push_recreates": frac(proj["push"][k] - proj["harmless"][k]),
            "random_push_recreates": frac(proj["random_push"][k] - proj["harmless"][k]),
            "ablate_all_removes": frac(proj["harmful"][k] - proj["ablate_all"][k]),
            "ablate_at_removes": frac(proj["harmful"][k] - proj["ablate_at"][k]),
            "random_ablate_removes": frac(proj["harmful"][k] - proj["random_ablate"][k]),
            **{f"ablate_from{f}_removes": frac(proj["harmful"][k] - proj[f"ablate_from{f}"][k]) for f in a.ablate_from}}
    print(f"{a.model}  r̂ L{r.layer} P{r.position_index}  readout at P{a.pos}; fractions of the harmful-harmless gap")
    for k, ro in out["readout"].items():
        print(f"\naxis {k}:  {'L':>3} {'gap':>7} {'push':>6} {'rpush':>6} {'abl_all':>8} {'abl_at':>7} {'rabl':>6}"
              + "".join(f" {'from' + str(f):>7}" for f in a.ablate_from))
        for l in range(r.layer, L):
            print(f"{'':10}{l:>3} {ro['gap'][l]:>7.2f} {ro['push_recreates'][l]:>6.0%} {ro['random_push_recreates'][l]:>6.0%} "
                  f"{ro['ablate_all_removes'][l]:>8.0%} {ro['ablate_at_removes'][l]:>7.0%} {ro['random_ablate_removes'][l]:>6.0%}"
                  + "".join(f" {ro[f'ablate_from{f}_removes'][l]:>7.0%}" for f in a.ablate_from))
    print("saved", save_json(run.path("analysis", "late-readout", a.tag), out))


if __name__ == "__main__":
    main()
