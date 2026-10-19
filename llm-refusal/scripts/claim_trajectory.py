"""Does a pushed claim's direction stay read through the answer, or decay back? (no generation)

A push on the prompt alone makes Llama-3-8B, Qwen2.5-7B and Llama-2 open "I don't know" about a famous fact and then
answer it anyway, while Llama-3.1 keeps the claim (RESULTS §59); the push has to be held for tens of answer tokens to
keep it (§60). This reads the model's own state along the answer: each saved response (familiarity_dial.py's file) is
teacher-forced after its templated prompt under the condition it was generated with, but with the push on the prompt
tokens only (so every answer token's residual is the model's own), and the residual's projection on the axis (unit
vector) is read at --layers on every answer token. Reported per condition and answer-token bucket (1-4, 5-16, 17-64,
65+), with the baseline condition's famous-question answers as the reference.

  .venv/bin/python3 llm-refusal/scripts/claim_trajectory.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --axis results/Meta-Llama-3-8B-Instruct-unknown-direction \\
      --source results/analysis/Meta-Llama-3-8B-Instruct-familiarity-dial-pos.json --conditions baseline,addprompt,add

Writes results/analysis/<short>-claim-trajectory[-tag].json.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from probe import save_json  # noqa: E402

BUCKETS = [(1, 4), (5, 16), (17, 64), (65, 10_000)]


def bucket_means(proj: t.Tensor, start: int, end: int) -> List[float]:
    """Mean of proj[start + k - 1] over each bucket's answer tokens k (1-based); NaN for an empty bucket."""
    out = []
    for lo, hi in BUCKETS:
        a, b = start + lo - 1, min(start + hi, end)
        out.append(proj[a:b].mean().item() if b > a else float("nan"))
    return out


def prompt_only_push(spec: str, stem_strength):
    """(stem, strength) of an add:/addpos:/sub:/subpos: spec, its sign folded into the strength; None for baseline."""
    kind, _, rest = spec.partition(":")
    if kind == "baseline":
        return None
    if kind in ("addpos", "subpos"):
        rest = rest.partition(":")[2]
    stem, s = stem_strength(rest)
    return stem, (-s if kind.startswith("sub") else s)


@t.no_grad()
def project(run, prompts: List[str], responses: List[str], push, layers: List[int], u: t.Tensor,
            batch_size: int = 8) -> List[Dict[int, List[float]]]:
    """Per response: {layer: bucket means of the projection on u over its answer tokens}, the push (vector, layer)
    added at its layer's input on the prompt tokens only."""
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = run.fmt.format_with_completions(prompts[i:i + batch_size], responses[i:i + batch_size])
        answer = enc["labels"] != -100                                          # [B, T] the answer tokens
        prompt_mask = enc["attention_mask"].bool() & ~answer
        captured = {}

        def reader(l):
            def hook(module, args):
                captured[l] = (args[0].float() @ u.to(args[0].device)).cpu()
            return hook

        handles = [run.blocks[l].register_forward_pre_hook(reader(l)) for l in layers]
        if push is not None:
            vec, pl = push

            def add(module, args):
                h = args[0]
                return (h + prompt_mask.to(h.device)[..., None].to(h.dtype) * vec.to(h.device, h.dtype),) + args[1:]
            handles.insert(0, run.blocks[pl].register_forward_pre_hook(add))
        try:
            dev = run.model.device
            run.model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                      position_ids=enc["position_ids"].to(dev))
        finally:
            for h in handles:
                h.remove()
        for j in range(answer.shape[0]):
            idx = answer[j].nonzero().flatten()
            if len(idx) == 0:
                out.append({l: [float("nan")] * len(BUCKETS) for l in layers})
                continue
            start, end = idx[0].item(), idx[-1].item() + 1
            out.append({l: bucket_means(captured[l][j], start, end) for l in layers})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--axis", required=True, help="the claim's direction stem (read on every answer token)")
    ap.add_argument("--source", required=True, help="familiarity_dial.py output with the conditions' responses")
    ap.add_argument("--conditions", required=True, help="comma-separated condition names in --source")
    ap.add_argument("--set", default="famous")
    ap.add_argument("--layers", default="", help="comma-separated layers to read (default: axis layer +2, +6, last)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    from datatypes import DirectionVector
    from familiarity_dial import prompt_sets
    from probe import load_run
    from push_positions import split_strength
    run = load_run(a.model, **({"force_cpu": True} if a.cpu else {}))
    d = DirectionVector.load(a.axis)
    u = d.vector.float().cpu()
    u = u / u.norm()
    n_layers = len(run.blocks)
    layers = ([int(x) for x in a.layers.split(",")] if a.layers else
              sorted({min(d.layer + 2, n_layers - 1), min(d.layer + 6, n_layers - 1), n_layers - 1}))
    prompts = prompt_sets()[a.set][:a.n]
    src = json.load(open(a.source))["generations"]
    out = {"model": a.model, "axis": a.axis, "axis_layer": d.layer, "set": a.set, "layers": layers,
           "buckets": BUCKETS, "conditions": {}}
    for name in a.conditions.split(","):
        spec = src[name]["spec"]
        responses = src[name]["responses"][a.set][:len(prompts)]
        p = prompt_only_push(spec, split_strength)
        push = None
        if p is not None:
            pd = DirectionVector.load(p[0])
            push = (pd.vector.float().cpu() * p[1], pd.layer)
        per = project(run, prompts, responses, push, layers, u)
        mean = {l: [sum(r[l][b] for r in per if r[l][b] == r[l][b]) / max(1, sum(r[l][b] == r[l][b] for r in per))
                    for b in range(len(BUCKETS))] for l in layers}
        out["conditions"][name] = {"spec": spec, "per_response": per, "mean": mean}
        for l in layers:
            print(f"{name:10s} L{l:<3d} " + "  ".join(f"{lo}-{hi if hi < 10_000 else '+'}: {m:+7.2f}"
                                                   for (lo, hi), m in zip(BUCKETS, mean[l])), flush=True)
    path = f"results/analysis/{run.short}-claim-trajectory{'-' + a.tag if a.tag else ''}.json"
    save_json(path, out)
    print("saved", path)


if __name__ == "__main__":
    main()
