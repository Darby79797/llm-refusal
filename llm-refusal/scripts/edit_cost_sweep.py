"""The cheapest weight edit that removes refusal (limb 1: PROPOSED_PLANS item 3 + the inhibitor).

One model load, then for each edit *spec*: refusal on the harmful eval prompts
(64-token rate + log-odds), false refusal on the harmless ones, and CE (nats/token)
on Alpaca reference completions, raw Pile text and the clean model's own completions
(capability.py's three sets, subsampled). Specs:

  none | full | random          nothing / r̂ out of every writer / a random direction out of every writer
  noemb                         every block, embedding untouched
  embonly                       only the embedding
  lt:E | ge:E | only:E          blocks < E / >= E / == E (no embedding)
  win:E:k                       blocks E-k..E+k (no embedding)
  +emb suffix                   add the embedding to any of the above, e.g. lt:D+emb
  adapter:STEM[:VARIANT]        a saved rank1/regrow adapter set (finetune.installed), VARIANT full/u_perp/u_rhat
E is an integer expression in D, the direction's layer (D, D-1, D+2 ...).

  .venv/bin/python3 llm-refusal/scripts/edit_cost_sweep.py --model Qwen/Qwen2.5-0.5B-Instruct \\
      --specs none random full noemb lt:D lt:D+1+emb ge:D win:D:2 adapter:results/finetune/Qwen2.5-0.5B-Instruct-refusal-rank1-remove:u_perp

Writes results/edit-cost/<model>-<concept>[-<tag>].json, one entry per spec, saved after each.
"""
import argparse
import contextlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from batching import resolve_forward_batch_size  # noqa: E402
from capability import MAX_TOKENS, completion_ce, load_alpaca, load_pile, random_direction  # noqa: E402
from datatypes import DirectionVector  # noqa: E402
from finetune import installed  # noqa: E402
from framework import DirectionTestFramework  # noqa: E402
from orthogonalize import edit_bytes, orthogonalized  # noqa: E402
from probe import behaviour  # noqa: E402


def parse_spec(spec: str, D: int, n_layers: int):
    """-> (kind, layers or None, embedding, extra). kind in {none, edit, random, adapter}."""
    if spec == "none":
        return "none", None, False, None
    if spec == "random":
        return "random", None, True, None
    if spec.startswith("adapter:"):
        _, stem, *rest = spec.split(":")
        return "adapter", None, False, (stem, rest[0] if rest else "full")
    emb = spec.endswith("+emb")
    core = spec[:-4] if emb else spec
    if core == "full":
        return "edit", None, True, None
    if core == "noemb":
        return "edit", None, False, None
    if core == "embonly":
        return "edit", [], True, None
    ev = lambda e: int(eval(e, {"__builtins__": {}}, {"D": D}))  # noqa: E731 - tiny integer expressions in D
    parts = core.split(":")
    if parts[0] == "lt":
        layers = list(range(0, ev(parts[1])))
    elif parts[0] == "ge":
        layers = list(range(ev(parts[1]), n_layers))
    elif parts[0] == "only":
        layers = [ev(parts[1])]
    elif parts[0] == "win":
        c, k = ev(parts[1]), ev(parts[2])
        layers = list(range(max(0, c - k), min(n_layers, c + k + 1)))
    else:
        raise ValueError(f"unknown spec {spec!r}")
    return "edit", layers, emb, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--direction", help="direction stem (default results/<model>-<concept>-direction)")
    ap.add_argument("--specs", nargs="+", default=["none", "random", "full", "noemb", "embonly",
                                                   "lt:D", "lt:D+emb", "ge:D", "ge:D-2", "win:D:1", "win:D:2", "win:D:4", "only:D-1"])
    ap.add_argument("--n-alpaca", type=int, default=200)
    ap.add_argument("--n-pile", type=int, default=100)
    ap.add_argument("--n-od", type=int, default=100)
    ap.add_argument("--n-harmful", type=int, default=99)
    ap.add_argument("--n-harmless", type=int, default=80)
    ap.add_argument("--tag", default="")
    ap.add_argument("--torch-dtype", default="auto")
    a = ap.parse_args()
    short = a.model.split("/")[-1]

    fw = DirectionTestFramework(model_name=a.model, concept=a.concept, torch_dtype=a.torch_dtype)
    model, fmt, ev = fw.model, fw.prompt_formatter, fw.evaluator
    n_layers = len(fw.intervention_applier.transformer_layers)
    r = DirectionVector.load(a.direction or f"results/{short}-{a.concept}-direction")
    r_hat = r.unit.float().cpu()
    harmful, harmless = fw.concept.eval_data_fn()
    harmful, harmless = harmful[:a.n_harmful], harmless[:a.n_harmless]

    ev.reserve_bytes = edit_bytes(model)
    bs = resolve_forward_batch_size("auto", model, 2 * MAX_TOKENS, reserve_bytes=ev.reserve_bytes)
    alpaca = load_alpaca("eval")[:a.n_alpaca]
    pile = load_pile()[:a.n_pile]
    od_prompts = [x["instruction"] for x in load_alpaca("eval")[:a.n_od]]
    od_completions = ev.generate_responses(od_prompts, max_new_tokens=MAX_TOKENS)
    rand = random_direction(r.vector.shape[-1])

    out_dir = "results/edit-cost"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{short}-{a.concept}" + (f"-{a.tag}" if a.tag else "") + ".json")
    out = {"model": a.model, "concept": a.concept, "direction": {"layer": r.layer, "position_index": r.position_index},
           "n_layers": n_layers, "ce_batch_size": bs, "dtype": str(model.dtype),
           "n": {"alpaca": len(alpaca), "pile": len(pile), "on_distribution": len(od_prompts),
                 "harmful": len(harmful), "harmless": len(harmless)}, "specs": {}}

    def ctx_for(spec):
        kind, layers, emb, extra = parse_spec(spec, r.layer, n_layers)
        if kind == "none":
            return contextlib.nullcontext, {"kind": kind}
        if kind == "random":
            return (lambda: orthogonalized(model, rand)), {"kind": kind}
        if kind == "adapter":
            stem, variant = extra
            return (lambda: installed(model, stem, variant, r_hat)), {"kind": kind, "stem": stem, "variant": variant}
        n_edited = (n_layers if layers is None else len(layers))
        return (lambda: orthogonalized(model, r.vector, layers=layers, embedding=emb)), \
            {"kind": kind, "layers": layers, "embedding": emb, "n_blocks_edited": n_edited}

    for spec in a.specs:
        ctx, meta = ctx_for(spec)
        t0 = time.time()
        with ctx():
            res = dict(meta)
            res["harmful"] = behaviour(fw, harmful, contextlib.nullcontext)
            res["harmless"] = behaviour(fw, harmless, contextlib.nullcontext)
            res["ce_alpaca"] = completion_ce(model, fmt, [x["instruction"] for x in alpaca], [x["output"] for x in alpaca],
                                             bs, on_split=ev._record_split)
            res["ce_pile"] = completion_ce(model, fmt, None, pile, bs, on_split=ev._record_split)
            res["ce_on_distribution"] = completion_ce(model, fmt, od_prompts, od_completions, bs, on_split=ev._record_split)
        res["seconds"] = round(time.time() - t0, 1)
        out["specs"][spec] = res
        print(f"{spec:40s} refusal {res['harmful']['rate']:6.1%} (lo {res['harmful']['log_odds']:+6.2f})  "
              f"false-refusal {res['harmless']['rate']:5.1%}  CE alpaca {res['ce_alpaca']:.4f} pile {res['ce_pile']:.4f} "
              f"on-dist {res['ce_on_distribution']:.4f}  [{res['seconds']}s]", flush=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
