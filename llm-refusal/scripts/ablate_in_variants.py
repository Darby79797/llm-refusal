"""Ablate each of several directions inside each of several model variants (one model load).

Built for the 2026-10-04 checks of the regrown-mediator claims: does ablating seed A's regrown direction remove seed
B's regrown refusal (cross-seed transfer), and does the clean model need the regrown axis (and only downstream of
r̂'s layer)? Variants: `clean`, `edited`, `edited@S1+S2` (the edit with saved directions S1, S2 also orthogonalised out), or a
regrow tag (results/finetune/<short>-refusal-regrow-<tag>, installed inside the edit it was trained under: r̂ plus
its recorded extra_edit_directions). Directions (`dirspecs.py`, the shared grammar): a saved stem, `rhat` (the clean refusal direction), `random@<stem>`
(a norm-matched random direction at that stem's layer/position; ablation ignores the norm), `span:A+B+...` (the
whole span of several of those removed in one projection), `randspan:K@<stem>` (K random directions' span),
`pcs:K@A+B+...` (the span of the top K principal axes of those directions, unit-normed and uncentred, so the first
is close to their mean), `pc:K1/K2/...@A+B+...` (only those principal axes, 1-based), `tilt:C:N@<spec>` (draw N of a
random unit direction ⊥ r̂ with cosine exactly C to that spec's first axis: is a direction's effect just its overlap
with the axis?), `A&B&...` (several specs' axes ablated together) or `perp:<spec>` (that direction with r̂ projected out). `--from-layer K` ablates only at layers >= K (`--to-layer K2`: and < K2; default
every layer). Reports refusal on harmful prompts (64-token generation, phrase detector) and first-token log-odds,
plus harmless log-odds as a disruption check (`--harmless-gen` also generates on the harmless prompts: false
refusal and degeneracy, for when ablation moves harmless log-odds). `--rhat-push S` also adds r̂ ×S at its own layer
to the harmless prompts under each ablation (refusal, degeneracy and log-odds): does r̂ still act with that direction
gone? `--push-spec SPEC@L:S` does the same with any spec's axes (`A&B`: both), each scaled to S x the clean gap along it (induction
by a late axis; `random~SPEC@L:S` is the norm-matched control); every push response is saved for `label_deflection.py`.

  .venv/bin/python3 llm-refusal/scripts/ablate_in_variants.py --model meta-llama/Meta-Llama-3-8B-Instruct \\
      --variants readers-r32,readers-r32-s1 --directions rhat,results/regrown100-Meta-Llama-3-8B-Instruct-refusal-direction \\
      --tag crossseed

Writes results/analysis/<short>-ablate-in-variants-<tag>.json (every harmful-prompt response and its detector label, for
re-scoring: the phrase detector misses moralising deflections); resumes per (variant, direction).
"""
import argparse
import contextlib
import json
import os
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from coherence import is_degenerate  # noqa: E402
import dirspecs  # noqa: E402
from datatypes import DirectionVector  # noqa: E402
from finetune import installed, load_adapters  # noqa: E402
from orthogonalize import edit_bytes, orthogonalized  # noqa: E402
from probe import load_run, residuals_at, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--directions", default="", help="comma-separated specs (dirspecs.py); 'none' is always run")
    ap.add_argument("--from-layer", type=int, default=None)
    ap.add_argument("--to-layer", type=int, default=None, help="ablate only at layers < this (with --from-layer: a window)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the unablated 'none' condition (already measured)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-harmful", type=int, default=99)
    ap.add_argument("--n-harmless", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--harmless-gen", action="store_true", help="also generate on harmless prompts (false refusal)")
    ap.add_argument("--rhat-push", type=float, default=None, help="also add r̂ at this strength on harmless prompts")
    ap.add_argument("--push-spec", nargs="*", default=[], metavar="SPEC@L:S",
                    help="also add SPEC's axes at layer L to the harmless prompts under each ablation, each scaled to S x "
                         "the clean harmful-minus-harmless gap along it at L (P-1); random~SPEC@L:S adds a random "
                         "direction of the same norm (does an axis induce refusal, as r̂ does?)")
    ap.add_argument("--max-new-tokens", type=int, default=64,
                    help="generation length (64 cuts some compliance off after its preamble; 160 resolves most)")
    a = ap.parse_args()

    run = load_run(a.model)
    fw, model, r = run.fw, run.model, run.r
    ev, tok = fw.evaluator, fw.tokenizer
    ev.reserve_bytes = edit_bytes(model)
    harmful, harmless = fw.concept.eval_data_fn()
    harmful, harmless = harmful[:a.n_harmful], harmless[:a.n_harmless]
    n_layers = len(run.blocks)
    layers = None if a.from_layer is None and a.to_layer is None else list(range(a.from_layer or 0, a.to_layer or n_layers))

    def spec_seed(spec):
        """--seed for the first random spec of a run (as before 2026-10-10); others mix in the spec, so two random specs
        in one run are not the same direction."""
        return a.seed if spec == first_random else a.seed * 1_000_003 + zlib.crc32(spec.encode())

    specs = [x for x in a.directions.split(",") if x]
    first_random = next((x for x in specs if x.startswith(("random@", "randspan:"))), None)

    def load_dir(spec):
        return dirspecs.load_spec(spec, r, spec_seed)

    dirs = {**({} if a.no_baseline else {"none": None}), **{d: load_dir(d) for d in specs}}

    # Push vectors, fixed before any ablation: the clean model's gap along each axis sets the scale.
    pushes, diffs = {}, {}  # diffs: layer -> clean harmful-minus-harmless mean residual at P-1
    for ps in a.push_spec:
        spec, ls = ps.rsplit("@", 1)
        layer, scale = int(ls.split(":")[0]), float(ls.split(":")[1])
        rand = spec.startswith("random~")
        if layer not in diffs:
            diffs[layer] = (residuals_at(model, run.fmt, run.blocks, harmful, -1)[:, layer].mean(0)
                            - residuals_at(model, run.fmt, run.blocks, harmless, -1)[:, layer].mean(0))
        axes = [dirspecs.unit(x) for x in dirspecs.rows(load_dir(spec[7:] if rand else spec))]
        gaps = [float(diffs[layer] @ x) for x in axes]
        v = sum(dirspecs.push_vector(x, gap, scale) for x, gap in zip(axes, gaps))
        if rand:
            g = t.Generator().manual_seed(spec_seed(ps))
            v = dirspecs.unit(t.randn(v.shape[-1], generator=g)) * v.norm()
        pushes[ps] = (DirectionVector(vector=v.to(r.vector.dtype), layer=layer, position_index=-1, score=0), gaps)

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
            texts = ev.generate_responses(harmful, max_new_tokens=a.max_new_tokens)
            texts_n = ev.generate_responses(harmless, max_new_tokens=a.max_new_tokens) if a.harmless_gen else None
            lo_h = ev.log_odds_scores(harmful)
            lo_n = ev.log_odds_scores(harmless)
        labels = [bool(ev._check_for_detection(x)) for x in texts]
        degen = lambda xs: sum(is_degenerate(tok.encode(x, add_special_tokens=False)) for x in xs) / len(xs)  # noqa: E731
        fin = lambda xs: [x for x in xs if x == x]  # noqa: E731
        m = {"rate": sum(labels) / len(labels), "log_odds": sum(fin(lo_h)) / len(fin(lo_h)),
             "harmless_log_odds": sum(fin(lo_n)) / len(fin(lo_n)), "degenerate": degen(texts),
             "n": len(labels), "samples": texts[:3], "texts": texts, "labels": labels}
        if a.rhat_push is not None:
            with ablated(d):
                run.applier.apply_direction_intervention(r, "add", a.rhat_push, layers=[r.layer])
                try:
                    texts_p = ev.generate_responses(harmless, max_new_tokens=a.max_new_tokens)
                    lo_p = ev.log_odds_scores(harmless)
                finally:
                    run.applier.clear_interventions()
            m.update(push_rate=sum(bool(ev._check_for_detection(x)) for x in texts_p) / len(texts_p),
                     push_degenerate=degen(texts_p), push_log_odds=sum(fin(lo_p)) / len(fin(lo_p)), push_samples=texts_p[:3])
        for ps, (pv, gap) in pushes.items():  # gap: one per axis
            with ablated(d):
                run.applier.apply_direction_intervention(pv, "add", 1.0, layers=[pv.layer])
                try:
                    texts_q = ev.generate_responses(harmless, max_new_tokens=a.max_new_tokens)
                    lo_q = ev.log_odds_scores(harmless)
                finally:
                    run.applier.clear_interventions()
            labels_q = [bool(ev._check_for_detection(x)) for x in texts_q]
            m.setdefault("pushes", {})[ps] = {
                "gaps": gap, "norm": float(pv.vector.float().norm()), "rate": sum(labels_q) / len(labels_q),
                "log_odds": sum(fin(lo_q)) / len(fin(lo_q)), "degenerate": degen(texts_q), "texts": texts_q, "labels": labels_q}
        if texts_n is not None:
            m.update(harmless_rate=sum(bool(ev._check_for_detection(x)) for x in texts_n) / len(texts_n),
                     harmless_degenerate=degen(texts_n), harmless_samples=texts_n[:3])
        return m

    path = run.path("analysis", "ablate-in-variants", a.tag)
    settings = {"from_layer": a.from_layer, "to_layer": a.to_layer, "rhat_push": a.rhat_push, "seed": a.seed, "n_harmful": a.n_harmful,
                "n_harmless": a.n_harmless, "harmless_gen": a.harmless_gen, "max_new_tokens": a.max_new_tokens,
                "push_specs": a.push_spec}
    out = {"model": a.model, **settings, "directions": {k: (
           None if v is None else {"span": v.shape[0]} if isinstance(v, t.Tensor) else
           {"layer": v.layer, "position_index": v.position_index}) for k, v in dirs.items()}, "results": {}}
    if os.path.exists(path):
        prev = json.load(open(path))
        # Files from before 2026-10-10 lack the newer keys: treat a missing key as matching.
        diff = {k: (prev[k], v) for k, v in settings.items() if k in prev and prev[k] != v}
        if diff:
            sys.exit(f"{path} was written with other settings {diff}: use another --tag")
        out["results"] = prev.get("results", {})
        print("resuming; done:", {k: list(v) for k, v in out["results"].items()}, flush=True)

    def do_variant(name):
        res = out["results"].setdefault(name, {})
        for dname, d in dirs.items():
            if dname in res:
                continue
            res[dname] = m = measure(d)
            short = dname if len(dname) <= 60 else dname[:24] + "…" + dname[-35:]
            print(f"{name:18s} ablate {short:60s} refusal {m['rate']:5.0%}  lo {m['log_odds']:+6.2f}  "
                  f"harmless lo {m['harmless_log_odds']:+6.2f}  degenerate {m['degenerate']:.0%}"
                  + (f"  harmless refusal {m['harmless_rate']:.0%} degenerate {m['harmless_degenerate']:.0%}"
                     if "harmless_rate" in m else "")
                  + (f"  r̂ push refusal {m['push_rate']:.0%} lo {m['push_log_odds']:+6.2f}" if "push_rate" in m else ""), flush=True)
            for ps, q in m.get("pushes", {}).items():
                print(f"{'':18s}   push {ps[:56]:56s} (gap {'/'.join(f'{x:+.2f}' for x in q['gaps'])}, norm {q['norm']:.1f}) harmless refusal "
                      f"{q['rate']:5.0%}  lo {q['log_odds']:+6.2f}  degenerate {q['degenerate']:.0%}", flush=True)
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
