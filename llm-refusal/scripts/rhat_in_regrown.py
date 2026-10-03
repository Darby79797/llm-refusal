"""Does the regrown model still obey the clean r̂? (PROPOSED_PLANS item 12)

After the weight edit removes r̂ and a LoRA regrows refusal along a new axis, add the
*clean* r̂ to harmless prompts and see whether the model still refuses. The edit only
removes r̂ from what layers write; what they read is untouched, so the response to an
injected r̂ should survive the edit itself. The question is whether fine-tuning changes it.

Model variants (one model load):
  clean                        reference
  edited                       the weight edit alone
  edited + <arm>-r0            fine-tuned on benign data only
  edited + <arm>-r32           fine-tuned with 32 refusal examples (refusal regrown)

Conditions on harmless eval prompts in every variant: nothing; r̂ added at its own layer
at each --strengths; r̂ added at every layer (strength 1); a norm-matched random
direction at r̂'s layer (strength 1 and the largest strength). With --mediator (the
regrown model's own direction, e.g. results/regrownvar-...-L18-P-1-direction): the
mediator added at its own layer (positive control), and r̂ added with the mediator
ablated at every layer (does r̂ act through the new axis?). Also the mean projection
onto the mediator at its layer/position, with and without r̂ added. Harmful-prompt
refusal with nothing is recorded per variant to confirm the edit / regrowth.

  .venv/bin/python3 llm-refusal/scripts/rhat_in_regrown.py --model Qwen/Qwen2.5-0.5B-Instruct \
      --mediator results/regrownvar-Qwen2.5-0.5B-Instruct-L18-P-1-direction

Writes results/finetune/<model>-rhat-in-regrown.json.
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
from finetune import installed  # noqa: E402
from orthogonalize import edit_bytes, orthogonalized  # noqa: E402
from coherence import is_degenerate  # noqa: E402
from probe import cos, load_run, residuals_at, save_json, unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="readers-r0,readers-r32,writers-r0,writers-r32")
    ap.add_argument("--strengths", default="0.5,1,2,4")
    ap.add_argument("--mediator", default=None, help="stem of the regrown model's own direction")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-harmful", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = load_run(a.model)
    fw, model, r, r_hat = run.fw, run.model, run.r, run.r_hat
    strengths = [float(s) for s in a.strengths.split(",")]
    eval_pos, eval_neg = fw.concept.eval_data_fn()
    harmless, harmful = eval_neg[:a.n_eval], eval_pos[:a.n_harmful]
    fw.evaluator.reserve_bytes = edit_bytes(model)

    rnorm = r.vector.float().norm()
    rand = unit(t.randn(len(r_hat), generator=t.Generator().manual_seed(a.seed))) * rnorm
    rand = rand.to(r.vector.dtype)
    med = DirectionVector.load(a.mediator) if a.mediator else None

    tok = fw.tokenizer

    def behaviour(fw_, prompts, ctx=contextlib.nullcontext):
        ev = fw_.evaluator
        with ctx():
            texts = ev.generate_responses(prompts, max_new_tokens=64)
            scores = ev.log_odds_scores(prompts)
        labels = [bool(ev._check_for_detection(x)) for x in texts]
        finite = [x for x in scores if x == x]
        res = {"rate": sum(labels) / len(labels), "log_odds": sum(finite) / len(finite), "n": len(labels),
               "labels": labels, "log_odds_per_prompt": scores}
        res["degenerate"] = sum(is_degenerate(tok.encode(x, add_special_tokens=False)) for x in texts) / len(texts)
        res["samples"] = texts[:3]
        return res

    def dv(v, layer, pos):
        return DirectionVector(vector=v, layer=layer, position_index=pos, score=0)

    def add_at(v, layer, pos, s=1.0, every_layer=False):
        return lambda: run.applier.intervened(dv(v, layer, pos), "add", strength=s,
                                              layers=None if every_layer else [layer])

    @contextlib.contextmanager
    def ablate_med_add_rhat(s):
        app = run.applier
        app.apply_direction_intervention(med, "ablate", 1.0, layers=None)
        app.apply_direction_intervention(r, "add", s, layers=[r.layer])
        try:
            yield
        finally:
            app.clear_interventions()

    def med_projection(ctx):
        with ctx():
            h = residuals_at(model, run.fmt, run.blocks, harmless, med.position_index)[:, med.layer]
        return float((h @ unit(med.vector)).mean())

    def conditions():
        res = {"harmful_none": behaviour(fw, harmful), "harmless_none": behaviour(fw, harmless)}
        for s in strengths:
            res[f"add_rhat_s{s:g}"] = behaviour(fw, harmless, add_at(r.vector, r.layer, r.position_index, s))
        res["add_rhat_every_layer_s1"] = behaviour(fw, harmless, add_at(r.vector, r.layer, r.position_index, 1.0, True))
        for s in sorted({1.0, max(strengths)}):
            res[f"add_random_s{s:g}"] = behaviour(fw, harmless, add_at(rand, r.layer, r.position_index, s))
        if med is not None:
            res["add_mediator_s1"] = behaviour(fw, harmless, add_at(med.vector, med.layer, med.position_index))
            for s in sorted({1.0, max(strengths)}):
                res[f"ablate_mediator_add_rhat_s{s:g}"] = behaviour(fw, harmless, lambda s=s: ablate_med_add_rhat(s))
            res["mediator_projection"] = {
                "none": med_projection(contextlib.nullcontext),
                **{f"add_rhat_s{s:g}": med_projection(add_at(r.vector, r.layer, r.position_index, s)) for s in strengths}}
        return res

    out = {"model": a.model, "direction": run.coords, "rhat_norm": float(rnorm), "n_harmless": len(harmless),
           "n_harmful": len(harmful), "strengths": strengths, "random_seed": a.seed,
           "mediator": a.mediator and {"stem": a.mediator, "layer": med.layer, "position_index": med.position_index,
                                       "cos_r_hat": cos(med.vector, r_hat)},
           "variants": {}}

    out_path = run.path("finetune", "rhat-in-regrown")
    if os.path.exists(out_path):
        import json
        prev = json.load(open(out_path))
        if prev.get("strengths") == strengths and prev.get("mediator") == out["mediator"]:
            out["variants"] = prev.get("variants", {})
            print("resuming; done:", list(out["variants"]), flush=True)

    def report(name, res):
        out["variants"][name] = res
        rates = " ".join(f"{k}={v['rate']:.0%}/{v['degenerate']:.0%}d" for k, v in res.items() if isinstance(v, dict) and "rate" in v)
        print(f"{name:22s} {rates}", flush=True)
        if "mediator_projection" in res:
            print(f"{'':22s} mediator projection: " + " ".join(f"{k}={v:+.2f}" for k, v in res["mediator_projection"].items()),
                  flush=True)
        save_json(run.path("finetune", "rhat-in-regrown"), out)

    def todo(name):
        return name not in out["variants"]

    if todo("clean"):
        report("clean", conditions())
    with orthogonalized(model, r.vector):
        if todo("edited"):
            report("edited", conditions())
        for arm in a.arms.split(","):
            stem = f"results/finetune/{run.short}-refusal-regrow-{arm}"
            if not todo(f"edited+{arm}"):
                continue
            if not os.path.exists(stem + ".json"):
                print(f"skip {arm}: no {stem}.json", flush=True)
                continue
            with installed(model, stem, "full"):
                report(f"edited+{arm}", conditions())
    print("saved", run.path("finetune", "rhat-in-regrown"))


if __name__ == "__main__":
    main()
