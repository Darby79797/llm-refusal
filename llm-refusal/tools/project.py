"""Per-token projections of the residual stream onto a saved direction, at every layer.

For each selected (prompt, response) pair from an evaluate run's generations file, runs
one teacher-forced forward pass over the formatted prompt + response and records, at
the input of every block (the point our directions live at), the scalar projection
of each token's residual onto the unit direction. The result shows *where* the
direction lights up: which prompt tokens, where in the response, and from which layer.

By default the forward pass is on the clean model: "how does the unmodified model
represent this text". --replay applies the intervention that produced the response
(e.g. global ablation zeroes the projection everywhere, by construction).

  project.py --model Qwen/Qwen2.5-3B-Instruct --concept refusal \\
      --conditions baseline,global_ablation,layer_specific_addition --n 8
  → results/proj/<model>-<concept>-<tag>.json   (view: look.py proj FILE, or report.py)

Loads a model: don't run it next to another model-loading job on a small machine.
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported
import torch as t  # noqa: E402

from datatypes import DirectionVector  # noqa: E402
from formatting import last_real_token_indices, pad_rows  # noqa: E402
from tools.runs import SIDE, load_index  # noqa: E402

REPLAY = {  # condition -> (intervention, all layers?)
    "global_ablation": ("ablate", True), "layer_specific_ablation": ("ablate", False),
    "layer_specific_addition": ("add", False), "global_addition": ("add", True),
    "layer_specific_subtraction": ("subtract", False),
}


def encode(fw, prompt: str, response: str):
    """Token ids of formatted prompt + response, and the index where the response starts."""
    fmt, tok = fw.prompt_formatter, fw.tokenizer
    text = fmt.format_text(prompt)
    bos = [tok.bos_token_id] if fmt.prepend_bos else []
    p_ids = bos + tok.encode(text, add_special_tokens=False)
    ids = bos + tok.encode(text + response, add_special_tokens=False)
    if ids[:len(p_ids)] != p_ids:   # merge across the boundary: fall back to separate encodes
        ids = p_ids + tok.encode(response, add_special_tokens=False)
    return ids, len(p_ids)


@t.no_grad()
def project(fw, unit: t.Tensor, seqs, batch_size: int):
    """seqs: list of id lists. Returns per sequence a (n_layers, len) list of projections."""
    blocks = fw.intervention_applier.transformer_layers
    out = []
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        enc = pad_rows(chunk, fw.tokenizer.pad_token_id)
        ids, mask, pos = enc['input_ids'], enc['attention_mask'], enc['position_ids']
        last_real_token_indices(mask)                   # asserts right padding
        dev = fw.model.device
        captured = [None] * len(blocks)

        def make(l):
            def hook(module, args):
                captured[l] = (args[0].float() @ unit.to(dev)).cpu()   # (B, T)
            return hook
        handles = [b.register_forward_pre_hook(make(l)) for l, b in enumerate(blocks)]
        try:
            fw.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev))
        finally:
            for h in handles:
                h.remove()
        stacked = t.stack(captured, dim=1)              # (B, n_layers, T)
        for j, s in enumerate(chunk):
            out.append([[round(v, 2) for v in row[:len(s)].tolist()] for row in stacked[j]])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", default="refusal")
    ap.add_argument("--tag", default="", help="evaluate run tag, e.g. L21-P-4 (default: the only/first match)")
    ap.add_argument("--direction", help="direction path without .pt/.json (default: the run's saved direction)")
    ap.add_argument("--conditions", default="baseline,global_ablation,baseline_negative,layer_specific_addition")
    ap.add_argument("--n", type=int, default=8, help="prompts per condition (same prompt indices across a side)")
    ap.add_argument("--replay", action="store_true", help="apply each condition's intervention during the pass")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--root", default="results")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    short = a.model.split("/")[-1]
    ix = load_index([a.root])
    runs = [r for r in ix.find_evals(short, a.concept, a.tag, variant="") if r.model == short]
    if not runs:
        sys.exit(f"no evaluate run for {short}/{a.concept} tag~{a.tag!r} under {a.root}")
    run = runs[0]
    dpath = a.direction or os.path.join(a.root, f"{short}-{a.concept}-direction")
    direction = DirectionVector.load(dpath)
    if run.layer is not None and direction.layer != run.layer:
        print(f"warning: saved direction is L{direction.layer}, the run is {run.tag}; "
              f"recomputing is not supported here, pass --direction", file=sys.stderr)

    from framework import DirectionTestFramework
    fw = DirectionTestFramework(model_name=a.model, torch_dtype=a.torch_dtype, concept=a.concept)
    unit = direction.unit.float()
    gens = run.generations
    items, seqs = [], []
    for cond in a.conditions.split(","):
        if cond not in gens:
            print(f"skipping {cond}: not in {run.gen_path} (have {list(gens)})", file=sys.stderr)
            continue
        for i, e in enumerate(gens[cond][:a.n]):
            ids, start = encode(fw, e["prompt"], e["response"])
            seqs.append(ids)
            items.append({"condition": cond, "index": i, "prompt": e["prompt"], "response": e["response"],
                          "detected": e["detected"], "response_start": start,
                          "tokens": [fw.tokenizer.decode([x]) for x in ids]})

    projections = []
    for cond in dict.fromkeys(it["condition"] for it in items):
        sel = [k for k, it in enumerate(items) if it["condition"] == cond]
        if a.replay and cond in REPLAY:
            kind, everywhere = REPLAY[cond]
            layers = list(range(len(fw.intervention_applier.transformer_layers))) if everywhere else [direction.layer]
            ctx = fw.intervention_applier.intervened(direction, kind, 1.0, layers=layers)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            for k, p in zip(sel, project(fw, unit, [seqs[k] for k in sel], a.batch_size)):
                items[k]["proj"] = p

    # Scale reference: mean projection at the generation boundary (last prompt token)
    # over the positive and negative baseline prompts, per layer.
    reference = {}
    for side, cond in (("positive", "baseline"), ("negative", "baseline_negative")):
        if cond in gens:
            prompts = [e["prompt"] for e in gens[cond][:32]]
            ps = project(fw, unit, [encode(fw, p, "")[0] for p in prompts], a.batch_size)
            n_layers = len(ps[0])
            reference[side] = [round(sum(p[l][-1] for p in ps) / len(ps), 2) for l in range(n_layers)]

    out = a.out or os.path.join(a.root, "proj", f"{short}-{a.concept}-{run.tag}{'-replay' if a.replay else ''}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"model": short, "concept": a.concept, "run": run.id, "direction_layer": direction.layer,
                   "direction_pos": direction.position_index, "replay": a.replay,
                   "sides": {it["condition"]: SIDE.get(it["condition"], "?") for it in items},
                   "reference": reference, "items": items}, f)
    print(f"wrote {out} ({len(items)} items)")


if __name__ == "__main__":
    main()
