"""Does r̂ read the same on ordinary text as at the template token? Per-token projection onto r̂ at every layer.

trajectory.py reads the projection at one template position; the edit's CE cost is measured
on ordinary text tokens. This captures, at every block input, the projection onto r̂ and the
residual norm of every real token of N raw Pile documents (truncated to 256 tokens, BOS where
the model uses one) and of the harmless eval prompts (chat template). Dropped: position 0 and
attention-sink tokens (norm > 4x the row's median, as probe.token_projections does); for the
chat prompts the same rule is the only template filter, so template tokens that are not
sink-like stay in. Per layer: mean projection, mean cosine (projection / norm per token, then
averaged), std of the projection across tokens, fraction of tokens with projection > 0.

  .venv/bin/python3 llm-refusal/scripts/pile_trajectory.py --model Qwen/Qwen2.5-0.5B-Instruct
  .venv/bin/python3 llm-refusal/scripts/pile_trajectory.py --model Qwen/Qwen2.5-0.5B-Instruct \\
      --direction results/Qwen2.5-0.5B-Instruct-refusal-nomassive-direction --tag nomassive

Writes results/trajectory/<short>-<concept>-pile[-tag].json and plots/trajectory/<short>-pile[-tag].png.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from capability import load_pile  # noqa: E402
from probe import _forward, load_run, save_json  # noqa: E402

MAX_TOKENS = 256


@t.no_grad()
def token_stats(model, blocks, r_hat, batches):
    """batches: iterable of encodings. Returns per-layer lists of kept-token (proj, cos) tensors."""
    L = len(blocks)
    projs: list = [[] for _ in range(L)]
    coss: list = [[] for _ in range(L)]
    n_kept = n_real = 0
    for enc in batches:
        mask = enc["attention_mask"]
        cap = {}

        def make(l):
            def hook(module, args):
                x = args[0].float()
                cap[l] = ((x @ r_hat.to(x.device)).cpu(), x.norm(dim=-1).cpu())
            return hook
        handles = [b.register_forward_pre_hook(make(l)) for l, b in enumerate(blocks)]
        try:
            _forward(model, enc)
        finally:
            for h in handles:
                h.remove()
        for l in range(L):
            p, n = cap[l]
            med = t.nanmedian(n.masked_fill(mask == 0, float("nan")), dim=1).values[:, None]
            keep = (mask == 1) & (n <= 4 * med)
            keep[:, 0] = False
            projs[l].append(p[keep])
            coss[l].append(p[keep] / n[keep])
            if l == 0:
                n_kept += int(keep.sum()); n_real += int(mask.sum())
    return [t.cat(x) for x in projs], [t.cat(x) for x in coss], n_kept, n_real


def summarise(projs, coss, n_kept, n_real):
    return {"n_tokens_kept": n_kept, "n_tokens_real": n_real,
            "mean_proj": [float(p.mean()) for p in projs],
            "mean_cos": [float(c.mean()) for c in coss],
            "std_proj": [float(p.std()) for p in projs],
            "frac_pos": [float((p > 0).float().mean()) for p in projs],
            "n_kept_per_layer": [len(p) for p in projs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--n", type=int, default=100, help="number of Pile documents")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--direction", help="direction stem to use instead of the saved one (e.g. a masked copy)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    run = load_run(a.model, a.concept, direction=a.direction or True)
    model, fmt, blocks, r_hat, short = run.model, run.fmt, run.blocks, run.r_hat, run.short
    bs = a.batch_size

    docs = load_pile()[:a.n]
    pile_batches = (fmt.format_with_completions(None, docs[i:i + bs], max_completion_tokens=MAX_TOKENS)
                    for i in range(0, len(docs), bs))
    _, harmless = run.fw.concept.eval_data_fn()
    chat_batches = (fmt.format_batch(harmless[i:i + bs]) for i in range(0, len(harmless), bs))

    out = {"model": a.model, "concept": a.concept, "direction": run.coords, "direction_stem": a.direction,
           "n_layers": len(blocks), "n_docs": len(docs), "n_harmless": len(harmless),
           "max_tokens": MAX_TOKENS, "sets": {}}
    for name, batches in (("pile", pile_batches), ("harmless_chat", chat_batches)):
        res = summarise(*token_stats(model, blocks, r_hat, batches))
        out["sets"][name] = res
        print(f"{name}: {res['n_tokens_kept']}/{res['n_tokens_real']} tokens kept")
        for k in ("mean_proj", "mean_cos", "std_proj", "frac_pos"):
            print(f"  {k}: " + " ".join(f"{x:+.2f}" for x in res[k]), flush=True)

    print("saved", save_json(run.path("trajectory", a.concept, "pile" + (f"-{a.tag}" if a.tag else "")), out))
    if not a.no_plot:
        plot(out, f"plots/trajectory/{short}-pile{'-' + a.tag if a.tag else ''}.png")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = list(range(out["n_layers"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, c in (("pile", "#4C72B0"), ("harmless_chat", "#DD8452")):
        s = out["sets"][name]
        axes[0].plot(x, s["mean_proj"], color=c, linewidth=1, label=f"{name} (n={s['n_tokens_kept']} tok)")
        axes[1].plot(x, s["mean_cos"], color=c, linewidth=1, label=name)
    for ax, title in zip(axes, ("mean projection onto r̂ per token", "mean cosine(token, r̂)")):
        ax.axvline(out["direction"]["layer"], color="k", linewidth=0.8, alpha=0.5, label="r̂ layer")
        ax.axhline(0, color="grey", linewidth=0.5)
        ax.set_xlabel("layer (block input)"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"{out['model']}: r̂ at L{out['direction']['layer']}, sink tokens excluded")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
