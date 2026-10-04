"""Ablate each of several directions inside each of several model variants (one model load).

Built for the 2026-10-04 checks of the regrown-mediator claims: does ablating seed A's regrown direction remove seed
B's regrown refusal (cross-seed transfer), and does the clean model need the regrown axis (and only downstream of
r̂'s layer)? Variants: `clean`, `edited`, or a regrow tag (edited + results/finetune/<short>-refusal-regrow-<tag>).
Directions: a saved stem, `rhat` (the clean refusal direction), or `random@<stem>` (a norm-matched random direction
at that stem's layer/position; ablation ignores the norm). `--from-layer K` ablates only at layers >= K (default
every layer). Reports refusal on harmful prompts (64-token generation, phrase detector) and first-token log-odds,
plus harmless log-odds as a disruption check.

  .venv/bin/python3 llm-refusal/scripts/ablate_in_variants.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --variants readers-r32,readers-r32-s1 --directions rhat,results/regrown100-Meta-Llama-3-8B-Instruct-refusal-direction \\
      --tag crossseed

Writes results/analysis/<short>-ablate-in-variants-<tag>.json; resumes per (variant, direction).
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from coherence import is_degenerate  # noqa: E402
from datatypes import DirectionVector  # noqa: E402
from finetune import installed  # noqa: E402
from orthogonalize import edit_bytes, orthogonalized  # noqa: E402
from probe import load_run, save_json, unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--from-layer", type=int, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-harmful", type=int, default=99)
    ap.add_argument("--n-harmless", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    run = load_run(a.model)
    fw, model, r = run.fw, run.model, run.r
    ev, tok = fw.evaluator, fw.tokenizer
    ev.reserve_bytes = edit_bytes(model)
    harmful, harmless = fw.concept.eval_data_fn()
    harmful, harmless = harmful[:a.n_harmful], harmless[:a.n_harmless]
    n_layers = len(run.blocks)
    layers = None if a.from_layer is None else list(range(a.from_layer, n_layers))

    def load_dir(spec):
        if spec == "rhat":
            return r
        if spec.startswith("random@"):
            ref = DirectionVector.load(spec[7:])
            g = t.Generator().manual_seed(a.seed)
            v = unit(t.randn(ref.vector.shape[-1], generator=g)) * ref.vector.float().norm()
            return DirectionVector(vector=v.to(ref.vector.dtype), layer=ref.layer, position_index=ref.position_index, score=0)
        return DirectionVector.load(spec)

    dirs = {"none": None, **{d: load_dir(d) for d in a.directions.split(",")}}

    @contextlib.contextmanager
    def ablated(d):
        if d is None:
            yield
            return
        run.applier.apply_direction_intervention(d, "ablate", 1.0, layers=layers)
        try:
            yield
        finally:
            run.applier.clear_interventions()

    def measure(d):
        with ablated(d):
            texts = ev.generate_responses(harmful, max_new_tokens=64)
            lo_h = ev.log_odds_scores(harmful)
            lo_n = ev.log_odds_scores(harmless)
        labels = [bool(ev._check_for_detection(x)) for x in texts]
        fin = lambda xs: [x for x in xs if x == x]  # noqa: E731
        return {"rate": sum(labels) / len(labels), "log_odds": sum(fin(lo_h)) / len(fin(lo_h)),
                "harmless_log_odds": sum(fin(lo_n)) / len(fin(lo_n)),
                "degenerate": sum(is_degenerate(tok.encode(x, add_special_tokens=False)) for x in texts) / len(texts),
                "n": len(labels), "samples": texts[:3]}

    path = run.path("analysis", "ablate-in-variants", a.tag)
    out = {"model": a.model, "from_layer": a.from_layer, "directions": {k: (None if v is None else
           {"layer": v.layer, "position_index": v.position_index}) for k, v in dirs.items()}, "results": {}}
    if os.path.exists(path):
        prev = json.load(open(path))
        if prev.get("from_layer") == a.from_layer:
            out["results"] = prev.get("results", {})
            print("resuming; done:", {k: list(v) for k, v in out["results"].items()}, flush=True)

    def do_variant(name):
        res = out["results"].setdefault(name, {})
        for dname, d in dirs.items():
            if dname in res:
                continue
            res[dname] = m = measure(d)
            print(f"{name:18s} ablate {dname[-60:]:60s} refusal {m['rate']:5.0%}  lo {m['log_odds']:+6.2f}  "
                  f"harmless lo {m['harmless_log_odds']:+6.2f}  degenerate {m['degenerate']:.0%}", flush=True)
            save_json(path, out)

    variants = a.variants.split(",")
    if "clean" in variants:
        do_variant("clean")
    rest = [v for v in variants if v != "clean"]
    if rest:
        with orthogonalized(model, r.vector):
            for v in rest:
                if v == "edited":
                    do_variant("edited")
                    continue
                stem = f"results/finetune/{run.short}-refusal-regrow-{v}"
                with installed(model, stem, "full"):
                    do_variant(v)
    print("saved", path)


if __name__ == "__main__":
    main()
