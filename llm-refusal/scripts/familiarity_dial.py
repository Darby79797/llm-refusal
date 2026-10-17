"""Is unknown's direction a selective "I don't know" dial?

Unknown's direction reads graded familiarity (invented > little-known > famous, RESULTS §30), and added at strength 1
it makes models disown famous facts (§24, §28). At partial strength, do the little-known questions tip into "I don't
know" before the famous ones, i.e. does the push follow the model's own familiarity, or does it move every question
by the same amount?

Three prompt sets of 30: invented (unknown's eval positives: "Who won the 1987 Varnholt Cup?"), little-known real
(unknown_obscure's: "Who won the 1923 Tour de France?") and famous (unknown's eval negatives: "Who won the 1994 FIFA
World Cup?"). Each prompt's baseline projection on --axis (at its layer and position) is saved. Generation (128 tokens)
under --conditions (jbb_safety.py's grammar; `add:<stem>*0.5` scales the push; several joined with `|`), saved after
each condition (a re-run resumes). `--judge FILE` labels the responses with limit_factorial.py's five-flag qwen3:4b
judge (dontknow and asserts are the ones read here) plus the unknown phrase detector, and prints per condition and
set the don't-know and asserts rates, and, over the real questions, the AUROC of the baseline projection for
"flipped to don't-know under this condition" against "did not" (> 0.5: the push tips the less familiar first).

  .venv/bin/python3 llm-refusal/scripts/familiarity_dial.py --model meta-llama/Llama-3.1-8B-Instruct \\
      --axis results/Llama-3.1-8B-Instruct-unknown-direction \\
      --conditions "baseline=baseline,add50=add:results/Llama-3.1-8B-Instruct-unknown-direction*0.5"
  .venv/bin/python3 llm-refusal/scripts/familiarity_dial.py --judge results/analysis/Llama-3.1-8B-Instruct-familiarity-dial.json

Writes results/analysis/<short>-familiarity-dial[-tag].json.
"""
import argparse
import contextlib
import json
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import prompts  # noqa: E402
from probe import auroc, save_json  # noqa: E402

SETS = ["invented", "obscure", "famous"]


def prompt_sets() -> Dict[str, List[str]]:
    invented, famous = prompts.create_unknown_eval_data()
    obscure, _ = prompts.create_unknown_obscure_eval_data()
    return {"invented": invented, "obscure": obscure, "famous": famous}


def flipped_auroc(proj: List[float], base: List[bool], cond: List[bool]) -> Optional[float]:
    """AUROC of the baseline projection, prompts that say "don't know" under the condition but not at baseline vs
    prompts that say it under neither; None when either group is empty."""
    yes = [p for p, b, c in zip(proj, base, cond) if c and not b]
    no = [p for p, b, c in zip(proj, base, cond) if not c and not b]
    return auroc(yes, no) if yes and no else None


def summarise(out: dict) -> List[str]:
    lines = [f"{'condition':12s}" + "".join(f"{s + ' dk/as':>18s}" for s in SETS) + "   flip AUROC (real)"]
    proj = out.get("projections", {})
    base = out["generations"].get("baseline", {}).get("dontknow")
    for name, c in out["generations"].items():
        if "dontknow" not in c:
            continue
        row = f"{name:12s}"
        for s in SETS:
            dk, a = c["dontknow"][s], c["asserts"][s]
            row += f"{100 * sum(dk) / len(dk):10.0f}% /{100 * sum(a) / len(a):4.0f}%"
        if base and proj and name != "baseline":
            real = lambda d: d["obscure"] + d["famous"]
            r = flipped_auroc(real(proj), real(base), real(c["dontknow"]))
            row += f"   {r:.2f}" if r is not None else "   n/a"
        lines.append(row)
    return lines


def judge(path: str, api_base: str, model: str) -> None:
    from concept import detect_unknown
    from limit_factorial import ask_flags
    out = json.load(open(path))
    data = prompt_sets()
    for name, c in out["generations"].items():
        flags = c.setdefault("flags", {})
        for s in SETS:
            prev = flags.get(s, [])
            flags[s] = [prev[i] if i < len(prev) and prev[i] else ask_flags(api_base, model, p, x)
                        for i, (p, x) in enumerate(zip(data[s], c["responses"][s]))]
            save_json(path, out)
        # a failed judge call falls back to the detector for that response
        c["dontknow"] = {s: [f["dontknow"] if f else detect_unknown(x) for f, x in zip(flags[s], c["responses"][s])]
                         for s in SETS}
        c["asserts"] = {s: [bool(f and f["asserts"]) for f in flags[s]] for s in SETS}
        c["detector"] = {s: [detect_unknown(x) for x in c["responses"][s]] for s in SETS}
        save_json(path, out)
        print(f"{name}: judged", flush=True)
    print("\n".join(summarise(out)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--axis", help="unknown's direction stem, for the baseline projections")
    ap.add_argument("--conditions", help="comma-separated name=spec[|spec...]")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--judge", metavar="FILE", help="label a saved file instead (no study model)")
    ap.add_argument("--judge-api-base", default="http://localhost:11434")
    ap.add_argument("--judge-model", default="qwen3:4b")
    ap.add_argument("--n", type=int, default=None, help="prompts per set (default all 30)")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (a smoke check while the GPU is busy)")
    ap.add_argument("--tag", default="", help="output file suffix")
    a = ap.parse_args()
    if a.judge:
        return judge(a.judge, a.judge_api_base, a.judge_model)
    assert a.model, "--model is required unless --judge"

    from datatypes import DirectionVector
    from jbb_safety import condition_context
    from probe import load_run, residuals_at
    run = load_run(a.model, **({"force_cpu": True} if a.cpu else {}))
    data = {s: ps[:a.n] for s, ps in prompt_sets().items()}
    assert len({len(ps) for ps in data.values()}) == 1, "the three sets must be the same size (responses are sliced by it)"
    path = f"results/analysis/{run.short}-familiarity-dial{'-' + a.tag if a.tag else ''}.json"
    out = json.load(open(path)) if os.path.exists(path) else {"model": a.model, "n": len(data["famous"])}

    if a.axis and "projections" not in out:
        d = DirectionVector.load(a.axis)
        u = d.vector.float().cpu()
        u = u / u.norm()
        out["axis"], out["axis_layer"] = a.axis, d.layer
        out["projections"] = {s: (residuals_at(run.model, run.fmt, run.blocks, ps, d.position_index)[:, d.layer] @ u)
                              .tolist() for s, ps in data.items()}
        save_json(path, out)
        m = {s: sum(v) / len(v) for s, v in out["projections"].items()}
        print("mean projection: " + "  ".join(f"{s} {m[s]:+.2f}" for s in SETS), flush=True)

    gens = out.setdefault("generations", {})
    for name, spec in (x.split("=", 1) for x in (a.conditions.split(",") if a.conditions else [])):
        if name in gens and all(gens[name]["responses"].get(s) for s in SETS):
            continue
        flat = [p for s in SETS for p in data[s]]
        with contextlib.ExitStack() as stack:
            # applier.intervened's exit clears every hook, so stacked specs all stay active until the last one exits
            for sp in spec.split("|"):
                stack.enter_context(condition_context(run, sp))
            texts = run.ev.generate_responses(flat, max_new_tokens=a.max_new_tokens)
        k = len(data["famous"])
        gens[name] = {"spec": spec, "responses": {s: texts[i * k:(i + 1) * k] for i, s in enumerate(SETS)}}
        save_json(path, out)
        print(f"{name}: generated", flush=True)
    print("saved", path)


if __name__ == "__main__":
    main()
