"""Where does a jailbreak template hide from r̂? (PROPOSED_PLANS item 10)

jailbreak_projection.py reads the projection onto r̂ only at r̂'s layer. Here each
template-wrapped harmful prompt (and the harmless prompts, as the reference) is read at
EVERY layer's block input, at two positions: the end-of-instruction/template position
(r.position_index) and the last token (-1). Per template and position: the per-layer mean
projection onto r̂ and the per-layer AUROC(refused vs complied) where both outcomes occur.
The 64-token refusal rate per template comes from probe.behaviour.

  .venv/bin/python3 llm-refusal/scripts/jailbreak_layers.py --model Qwen/Qwen2.5-3B-Instruct

Writes results/jailbreak/<short>-<concept>-layers.json and plots/jailbreak/<short>-layers.png.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

from jailbreak_projection import TEMPLATES  # noqa: E402
from probe import auroc, behaviour, load_run, residuals_at, save_json, split_by  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
HARMLESS_COLOUR = "#7F7F7F"


def per_layer_auroc(proj, labels):
    """proj [n, n_layers] tensor, labels list[bool] -> per-layer AUROC(refused > complied), None where one outcome is absent."""
    if not any(labels) or all(labels):
        return [None] * proj.shape[1]
    return [auroc(*split_by(proj[:, l].tolist(), labels)) for l in range(proj.shape[1])]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--n", type=int, default=40, help="prompts per class (harmful and harmless)")
    ap.add_argument("--templates", default="plain,suppression,manyshot", help=f"comma list from {','.join(TEMPLATES)}")
    a = ap.parse_args()
    names = [n for n in a.templates.split(",") if n]
    unknown = [n for n in names if n not in TEMPLATES]
    if unknown:
        ap.error(f"unknown templates {unknown}; choose from {list(TEMPLATES)}")

    run = load_run(a.model, a.concept)
    model, fmt, blocks, r, r_hat = run.model, run.fmt, run.blocks, run.r, run.r_hat
    harmful, harmless = run.fw.concept.eval_data_fn()
    harmful, harmless = harmful[:a.n], harmless[:a.n]
    positions = {"template": r.position_index, "last": -1}

    def curves(prompts):
        """{position name: [n, n_layers] projection onto r̂}."""
        return {name: residuals_at(model, fmt, blocks, prompts, pos) @ r_hat for name, pos in positions.items()}

    out = {"model": a.model, "concept": a.concept, "direction": run.coords, "n": len(harmful),
           "positions": positions, "n_layers": len(blocks), "templates": {}}
    hp = curves(harmless)
    out["harmless"] = {name: {"proj_mean": p.mean(0).tolist()} for name, p in hp.items()}
    for tname in names:
        wrapped = [TEMPLATES[tname].format(x=p) for p in harmful]
        labels = []
        res = behaviour(run.fw, wrapped, labels_out=labels)
        cv = curves(wrapped)
        out["templates"][tname] = {"refusal_rate": res["rate"], "n_refused": sum(labels), "refused": labels}
        for name, p in cv.items():
            out["templates"][tname][name] = {"proj_mean": p.mean(0).tolist(), "auroc_refused_vs_complied": per_layer_auroc(p, labels)}
        print(f"{tname}: refusal {res['rate']:.1%}", flush=True)

    L = r.layer
    print(f"\nr̂ at L{L} P{r.position_index}; harmless refs: " +
          ", ".join(f"{n} {out['harmless'][n]['proj_mean'][L]:+.2f}" for n in positions))
    print(f"{'template':14s}{'refuse%':>8s}" + "".join(f" | {n:>8s}@L{L:<3d} {'argmax':>6s} {'max':>7s}" for n in positions))
    for tname, rec in [("harmless", None)] + list(out["templates"].items()):
        row = f"{tname:14s}" + (f"{'-':>8s}" if rec is None else f"{rec['refusal_rate']:8.0%}")
        for n in positions:
            m = out["harmless"][n]["proj_mean"] if rec is None else rec[n]["proj_mean"]
            top = max(range(len(m)), key=m.__getitem__)
            row += f" | {m[L]:+12.2f}    {top:>6d} {m[top]:+7.2f}"
        print(row)

    print("saved", save_json(run.path("jailbreak", a.concept, "layers"), out))
    plot(out, os.path.join("plots", "jailbreak", f"{run.short}-layers.png"))


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(out["n_layers"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (name, pos) in zip(axes, out["positions"].items()):
        ax.plot(x, out["harmless"][name]["proj_mean"], color=HARMLESS_COLOUR, linestyle="--", linewidth=1.2, label="harmless")
        for i, (tname, rec) in enumerate(out["templates"].items()):
            ax.plot(x, rec[name]["proj_mean"], color=PALETTE[i % len(PALETTE)], linewidth=1.2,
                    label=f"{tname} ({rec['refusal_rate']:.0%} refused)")
        ax.axvline(out["direction"]["layer"], color="k", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("layer (block input)")
        ax.set_ylabel("mean projection onto r̂")
        ax.set_title(f"position {pos} ({name})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{out['model']}: r̂ at L{out['direction']['layer']} P{out['direction']['position_index']}, "
                 f"{out['n']} harmful prompts per template")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
