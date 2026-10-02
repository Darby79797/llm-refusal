"""The refusal direction through the layers (limb 3.1): where the harmful/harmless split opens.

One forward pass per prompt set. At r̂'s template position, for every layer: the mean
projection of harmful and harmless eval prompts onto r̂, the AUROC of that projection as
a harmful-vs-harmless feature, the residual norm, and the cosine between r̂ and that
layer's own harmful-harmless mean difference (is the same direction used throughout,
or does r̂ only match at its layer?). Under the plain model, r̂ hook-ablated, and (when a
rank1 remove adapter exists) the inhibitor. Also the same projections onto û⊥.

  .venv/bin/python3 llm-refusal/scripts/trajectory.py --model Qwen/Qwen2.5-0.5B-Instruct

Writes results/trajectory/<model>-refusal.json and plots/trajectory/<model>-refusal.png.
"""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

from finetune import installed  # noqa: E402
from probe import auroc, cos, inhibitor_direction, load_run, residuals_at, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--adapter", help="rank1 remove adapter stem (default: the model's remove-s0 or remove run, if any)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--direction", help="direction stem to use instead of the saved one (e.g. a masked copy)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    run = load_run(a.model, a.concept, direction=a.direction or True)
    model, fmt, blocks, r, r_hat, short = run.model, run.fmt, run.blocks, run.r, run.r_hat, run.short
    pos, neg = run.fw.concept.eval_data_fn()

    adapter = a.adapter
    if adapter is None:
        for tag in ("remove-s0", "remove"):
            if os.path.exists(f"results/finetune/{short}-{a.concept}-rank1-{tag}.json"):
                adapter = f"results/finetune/{short}-{a.concept}-rank1-{tag}"
                break
    dirs = {"r_hat": r_hat}
    conditions = {"plain": contextlib.nullcontext, "ablate_r_hat": run.ablate(r)}
    if adapter:
        dirs["u_perp"] = inhibitor_direction([adapter], r_hat)[1]
        conditions["inhibitor"] = lambda: installed(model, adapter, "u_perp", r_hat)

    out = {"model": a.model, "concept": a.concept, "direction": run.coords,
           "n_layers": len(blocks), "adapter": adapter, "conditions": {}}
    for name, ctx in conditions.items():
        with ctx():
            hp = residuals_at(model, fmt, blocks, pos, r.position_index)   # [n, L, d]
            hn = residuals_at(model, fmt, blocks, neg, r.position_index)
        res = {"norm_harmful": hp.norm(dim=-1).mean(0).tolist(), "norm_harmless": hn.norm(dim=-1).mean(0).tolist()}
        diff = hp.mean(0) - hn.mean(0)                                       # [L, d] local difference-in-means
        res["cos_local_diff_r_hat"] = [cos(diff[l], r_hat) for l in range(len(blocks))]
        res["local_diff_norm"] = diff.norm(dim=-1).tolist()
        for dname, d in dirs.items():
            pp, pn = hp @ d, hn @ d                                          # [n, L]
            res[f"proj_{dname}"] = {"harmful_mean": pp.mean(0).tolist(), "harmless_mean": pn.mean(0).tolist(),
                                    "harmful_std": pp.std(0).tolist(), "harmless_std": pn.std(0).tolist(),
                                    "auroc": [auroc(pp[:, l], pn[:, l]) for l in range(len(blocks))]}
        out["conditions"][name] = res
        pr = res["proj_r_hat"]
        print(f"{name:12s} r̂ proj harmful/harmless by layer:")
        print("  " + " ".join(f"L{l}:{pr['harmful_mean'][l]:+.1f}/{pr['harmless_mean'][l]:+.1f}" for l in range(len(blocks))))
        print("  AUROC: " + " ".join(f"{x:.2f}" for x in pr["auroc"]))
        print("  cos(local diff, r̂): " + " ".join(f"{x:+.2f}" for x in res["cos_local_diff_r_hat"]), flush=True)

    print("saved", save_json(run.path("trajectory", a.concept), out))
    if not a.no_plot:
        plot(out, f"plots/trajectory/{short}-{a.concept}.png")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = out["n_layers"]
    x = list(range(L))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    colours = {"plain": "#4C72B0", "ablate_r_hat": "#DD8452", "inhibitor": "#55A868"}
    for name, res in out["conditions"].items():
        c = colours.get(name, "grey")
        pr = res["proj_r_hat"]
        axes[0].plot(x, pr["harmful_mean"], color=c, label=f"{name} harmful")
        axes[0].plot(x, pr["harmless_mean"], color=c, linestyle="--", label=f"{name} harmless")
        axes[1].plot(x, pr["auroc"], color=c, label=name)
        axes[2].plot(x, res["cos_local_diff_r_hat"], color=c, label=name)
    for ax in axes:
        ax.axvline(out["direction"]["layer"], color="k", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("layer (block input)")
        ax.grid(alpha=0.3)
    axes[0].set_title("mean projection onto r̂"); axes[0].legend(fontsize=7)
    axes[1].set_title("AUROC harmful vs harmless along r̂"); axes[1].set_ylim(0.4, 1.02); axes[1].legend(fontsize=7)
    axes[2].set_title("cos(layer's own diff-in-means, r̂)"); axes[2].legend(fontsize=7)
    fig.suptitle(f"{out['model']} — r̂ at L{out['direction']['layer']} P{out['direction']['position_index']}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
