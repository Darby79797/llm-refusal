"""JailbreakBench + LlamaGuard 2 under several model conditions, resumable at every batch and every verdict.

`run_experiment.py --arditi-evals` saves nothing until its run ends, so a queue job that yields to another GPU user
(or crashes) repeats a ~40-minute 512-token generation from scratch. This script saves the responses after every
generation batch and the LlamaGuard verdicts every --judge-every responses (atomic writes), and a re-run picks up
where the last one stopped. Batches are the evaluator's own consecutive slices at the auto batch size, which is
fixed on the first run and stored, so a resumed run produces the same batches as an uninterrupted one.

Conditions, comma-separated `name=kind[:stem]`: `baseline`, `edit:<stem>` (weight orthogonalisation against that
direction, every block), `ablate:<stem>` (hook ablation at every layer). All responses are generated first (the
study model only), then judged by LlamaGuard 2 via Ollama (localhost; the model is unloaded after each pass).

  .venv/bin/python3 llm-refusal/scripts/jbb_safety.py --model Qwen/Qwen2.5-7B-Instruct --tag masked \\
      --conditions baseline=baseline,plain=edit:results/Qwen2.5-7B-Instruct-refusal-direction,masked=edit:results/Qwen2.5-7B-Instruct-refusal-nomassive-direction

Writes results/analysis/<short>-jbb-safety-<tag>.json: per condition the responses, phrase-detector labels,
LlamaGuard verdicts and a summary (refusal rate, LlamaGuard unsafe rate over judged responses, degenerate rate).
"""
import argparse
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import prompts  # noqa: E402
from coherence import is_degenerate  # noqa: E402
from datatypes import DirectionVector  # noqa: E402
from orthogonalize import edit_bytes, orthogonalized  # noqa: E402
from probe import load_run, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--judge-every", type=int, default=10)
    ap.add_argument("--llamaguard-api-base", default="http://localhost:11434", help="empty: skip judging")
    ap.add_argument("--batch-size", type=int, default=None, help="pin (default: auto, fixed on the first run)")
    ap.add_argument("--force-cpu", action="store_true", help="for tests while the GPU is busy")
    a = ap.parse_args()

    run = load_run(a.model, llamaguard_api_base=a.llamaguard_api_base or None, force_cpu=a.force_cpu)
    ev, tok = run.fw.evaluator, run.fw.tokenizer
    ev.reserve_bytes = edit_bytes(run.model)  # same headroom in every condition, so one batch size fits all
    jbb, _ = prompts.create_arditi_replication_eval_data()
    jbb = jbb[:a.n]

    def ctx(spec):
        kind, _, stem = spec.partition(":")
        if kind == "baseline":
            return contextlib.nullcontext()
        d = DirectionVector.load(stem)
        if kind == "edit":
            return orthogonalized(run.model, d.vector)
        if kind == "ablate":
            return run.applier.intervened(d, "ablate", layers=None)
        raise ValueError(f"unknown condition kind {kind!r}")

    conds = dict(c.split("=", 1) for c in a.conditions.split(","))
    path = run.path("analysis", "jbb-safety", a.tag)
    out = {"model": a.model, "n": len(jbb), "max_new_tokens": a.max_new_tokens, "batch_size": None, "conditions": {}}
    if os.path.exists(path):
        prev = json.load(open(path))
        if (prev["n"], prev["max_new_tokens"]) == (len(jbb), a.max_new_tokens):
            out = prev
            print("resuming:", {k: f"{sum(x is not None for x in v['responses'])} generated, "
                                   f"{sum(x is not None for x in v['llamaguard'])} judged" for k, v in out["conditions"].items()},
                  flush=True)
    if out["batch_size"] is None:
        out["batch_size"] = a.batch_size or ev.resolve_batch_size(jbb, a.max_new_tokens)
    bs = out["batch_size"]
    for name, spec in conds.items():
        c = out["conditions"].setdefault(name, {"spec": spec, "responses": [None] * len(jbb), "llamaguard": [None] * len(jbb)})
        assert c["spec"] == spec, f"{name}: saved as {c['spec']}, now {spec}"

    for name, spec in conds.items():  # generate (study model only)
        c = out["conditions"][name]
        todo = [i for i in range(0, len(jbb), bs) if any(x is None for x in c["responses"][i:i + bs])]
        if not todo:
            continue
        with ctx(spec):
            for i in todo:
                c["responses"][i:i + bs] = ev.generate_responses(jbb[i:i + bs], batch_size=bs, max_new_tokens=a.max_new_tokens)
                save_json(path, out)
                print(f"{name}: generated {min(i + bs, len(jbb))}/{len(jbb)}", flush=True)

    for name in conds if a.llamaguard_api_base else []:  # judge (LlamaGuard via Ollama); failed calls stay None and are retried on the next run
        c = out["conditions"][name]
        for i in range(0, len(jbb), a.judge_every):
            idx = [j for j in range(i, min(i + a.judge_every, len(jbb))) if c["llamaguard"][j] is None]
            if not idx:
                continue
            lg = []
            ev.evaluate_llamaguard_rate([jbb[j] for j in idx], [c["responses"][j] for j in idx], labels_out=lg)
            for j, l in zip(idx, lg):
                c["llamaguard"][j] = l
            save_json(path, out)

    for name, c in out["conditions"].items():
        texts, lg = c["responses"], c["llamaguard"]
        c["detected"] = [bool(ev._check_for_detection(x)) for x in texts]
        judged = [x for x in lg if x is not None]
        c["summary"] = {"refusal_rate": sum(c["detected"]) / len(texts),
                        "llamaguard_unsafe_rate": sum("unsafe" in x.lower() for x in judged) / len(judged) if judged else None,
                        "llamaguard_judged": len(judged),
                        "degenerate_rate": sum(is_degenerate(tok.encode(x, add_special_tokens=False)) for x in texts) / len(texts)}
        s = c["summary"]
        print(f"{name:10s} JBB refusal {s['refusal_rate']:5.1%}  unsafe {s['llamaguard_unsafe_rate']}  "
              f"({s['llamaguard_judged']} judged)  degenerate {s['degenerate_rate']:5.1%}", flush=True)
    save_json(path, out)
    print("saved", path)


if __name__ == "__main__":
    main()
