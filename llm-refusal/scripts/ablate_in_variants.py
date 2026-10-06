"""Ablate each of several directions inside each of several model variants (one model load).

Built for the 2026-10-04 checks of the regrown-mediator claims: does ablating seed A's regrown direction remove seed
B's regrown refusal (cross-seed transfer), and does the clean model need the regrown axis (and only downstream of
r̂'s layer)? Variants: `clean`, `edited`, `edited@S1+S2` (the edit with saved directions S1, S2 also orthogonalised out), or a
regrow tag (results/finetune/<short>-refusal-regrow-<tag>, installed inside the edit it was trained under: r̂ plus
its recorded extra_edit_directions). Directions: a saved stem, `rhat` (the clean refusal direction), `random@<stem>`
(a norm-matched random direction at that stem's layer/position; ablation ignores the norm), `span:A+B+...` (the
whole span of several of those removed in one projection), `randspan:K@<stem>` (K random directions' span),
`pcs:K@A+B+...` (the span of the top K principal axes of those directions, unit-normed and uncentred, so the first
is close to their mean) or `perp:<spec>` (that direction with r̂ projected out). `--from-layer K` ablates only at layers >= K (default
every layer). Reports refusal on harmful prompts (64-token generation, phrase detector) and first-token log-odds,
plus harmless log-odds as a disruption check (`--harmless-gen` also generates on the harmless prompts: false
refusal and degeneracy, for when ablation moves harmless log-odds). `--rhat-push S` also adds r̂ ×S at its own layer
to the harmless prompts under each ablation (refusal, degeneracy and log-odds): does r̂ still act with that direction
gone?

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
from finetune import installed, load_adapters  # noqa: E402
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
    ap.add_argument("--harmless-gen", action="store_true", help="also generate on harmless prompts (false refusal)")
    ap.add_argument("--rhat-push", type=float, default=None, help="also add r̂ at this strength on harmless prompts")
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
        if spec.startswith("span:"):
            return t.stack([load_dir(x).vector.float().cpu() for x in spec[5:].split("+")])
        if spec.startswith("pcs:"):
            k, stems = spec[4:].split("@", 1)
            V = t.stack([unit(load_dir(x).vector.float().cpu()) for x in stems.split("+")])
            return t.linalg.svd(V, full_matrices=False).Vh[:int(k)]
        if spec.startswith("perp:"):
            d = load_dir(spec[5:])
            u = unit(r.vector.float().cpu())
            if isinstance(d, t.Tensor):
                return d.float() - (d.float() @ u)[:, None] * u
            v = d.vector.float().cpu()
            return DirectionVector(vector=(v - (v @ u) * u).to(d.vector.dtype), layer=d.layer,
                                   position_index=d.position_index, score=0)
        if spec.startswith("randspan:"):
            k, ref = spec[9:].split("@", 1)
            g = t.Generator().manual_seed(a.seed)
            return t.randn(int(k), DirectionVector.load(ref).vector.shape[-1], generator=g)
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
            texts_n = ev.generate_responses(harmless, max_new_tokens=64) if a.harmless_gen else None
            lo_h = ev.log_odds_scores(harmful)
            lo_n = ev.log_odds_scores(harmless)
        labels = [bool(ev._check_for_detection(x)) for x in texts]
        degen = lambda xs: sum(is_degenerate(tok.encode(x, add_special_tokens=False)) for x in xs) / len(xs)  # noqa: E731
        fin = lambda xs: [x for x in xs if x == x]  # noqa: E731
        m = {"rate": sum(labels) / len(labels), "log_odds": sum(fin(lo_h)) / len(fin(lo_h)),
             "harmless_log_odds": sum(fin(lo_n)) / len(fin(lo_n)), "degenerate": degen(texts),
             "n": len(labels), "samples": texts[:3]}
        if a.rhat_push is not None:
            with ablated(d):
                run.applier.apply_direction_intervention(r, "add", a.rhat_push, layers=[r.layer])
                try:
                    texts_p = ev.generate_responses(harmless, max_new_tokens=64)
                    lo_p = ev.log_odds_scores(harmless)
                finally:
                    run.applier.clear_interventions()
            m.update(push_rate=sum(bool(ev._check_for_detection(x)) for x in texts_p) / len(texts_p),
                     push_degenerate=degen(texts_p), push_log_odds=sum(fin(lo_p)) / len(fin(lo_p)), push_samples=texts_p[:3])
        if texts_n is not None:
            m.update(harmless_rate=sum(bool(ev._check_for_detection(x)) for x in texts_n) / len(texts_n),
                     harmless_degenerate=degen(texts_n), harmless_samples=texts_n[:3])
        return m

    path = run.path("analysis", "ablate-in-variants", a.tag)
    out = {"model": a.model, "from_layer": a.from_layer, "rhat_push": a.rhat_push, "directions": {k: (
           None if v is None else {"span": v.shape[0]} if isinstance(v, t.Tensor) else
           {"layer": v.layer, "position_index": v.position_index}) for k, v in dirs.items()}, "results": {}}
    if os.path.exists(path):
        prev = json.load(open(path))
        if prev.get("from_layer") == a.from_layer and prev.get("rhat_push") == a.rhat_push:
            out["results"] = prev.get("results", {})
            print("resuming; done:", {k: list(v) for k, v in out["results"].items()}, flush=True)

    def do_variant(name):
        res = out["results"].setdefault(name, {})
        for dname, d in dirs.items():
            if dname in res:
                continue
            res[dname] = m = measure(d)
            print(f"{name:18s} ablate {dname[-60:]:60s} refusal {m['rate']:5.0%}  lo {m['log_odds']:+6.2f}  "
                  f"harmless lo {m['harmless_log_odds']:+6.2f}  degenerate {m['degenerate']:.0%}"
                  + (f"  harmless refusal {m['harmless_rate']:.0%} degenerate {m['harmless_degenerate']:.0%}"
                     if "harmless_rate" in m else "")
                  + (f"  r̂ push refusal {m['push_rate']:.0%} lo {m['push_log_odds']:+6.2f}" if "push_rate" in m else ""), flush=True)
            save_json(path, out)

    def edit_of(v):
        """(extra edit directions, adapter stem or None) for a non-clean variant."""
        if v == "edited" or v.startswith("edited@"):
            return tuple(v[7:].split("+")) if "@" in v else (), None
        stem = f"results/finetune/{run.short}-refusal-regrow-{v}"
        return tuple(load_adapters(stem)[0].get("extra_edit_directions") or ()), stem

    variants = a.variants.split(",")
    if "clean" in variants:
        do_variant("clean")
    groups = {}  # one weight edit per distinct extra-direction set, in first-seen order
    for v in variants:
        if v != "clean":
            extra, stem = edit_of(v)
            groups.setdefault(extra, []).append((v, stem))
    for extra, members in groups.items():
        vec = r.vector if not extra else t.stack([r.vector.float().cpu()] + [
            DirectionVector.load(x).vector.float().cpu() for x in extra]).to(r.vector.dtype)
        with orthogonalized(model, vec):
            for v, stem in members:
                if stem is None:
                    do_variant(v)
                    continue
                with installed(model, stem, "full"):
                    do_variant(v)
    print("saved", path)


if __name__ == "__main__":
    main()
