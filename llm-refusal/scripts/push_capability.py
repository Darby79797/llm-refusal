"""What does a jailbreak push cost the model's general ability? Benchmarks under jbb_safety.py's conditions.

Every JailbreakBench result should come with what the same intervention does to ordinary tasks, measured on the
same chat-formatted model it was found on. lm-eval (capability.py) scores untemplated few-shot prompts and takes
~35 min per condition on an 8B model. This scores a fixed sample (data/benchmark_items.json,
scripts/fetch_benchmark_items.py) in the chat template under each --conditions entry (jbb_safety.py's grammar:
`baseline`, `sub:STEM[*s]`, `subpos:<subset>[+gen]:STEM`, `edit:`, `ablate:`, ...; several joined with `|`):

  mmlu, arc  0-shot multiple choice. Accuracy = the answer letter with the highest logit at the generation boundary
             among the options' letters; `format` = how often the unrestricted top token is one of them (does the
             push still let the model follow the instruction?)
  gsm8k      0-shot, step by step, greedy, --max-new-tokens; the number after "Answer:" (else the last number) against
             the reference; also the degenerate rate

  .venv/bin/python3 llm-refusal/scripts/push_capability.py --model meta-llama/Llama-3.1-8B-Instruct --tag lim \\
      --conditions "baseline=baseline,lim=sub:results/limits4-Llama-3.1-8B-Instruct-direction"
  .venv/bin/python3 llm-refusal/scripts/push_capability.py --model meta-llama/Llama-3.1-8B-Instruct --tag jbb \\
      --from-jbb limsub,limbest,limpos     # every condition of those jailbreak runs (each distinct spec once)

Writes results/analysis/<short>-push-capability-<tag>.json, saved after each benchmark of each condition (a re-run
resumes).
"""
import argparse
import contextlib
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402

from formatting import last_real_token_indices  # noqa: E402
from probe import save_json  # noqa: E402

ITEMS = os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_items.json")
LETTERS = "ABCDE"
BENCHMARKS = ["mmlu", "arc", "gsm8k"]


def mc_prompt(item: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(item["choices"]))
    return f"{item['question']}\n\n{opts}\n\nAnswer with the letter of the correct option only."


def gsm8k_prompt(item: dict) -> str:
    return f"{item['question']}\n\nSolve it step by step, then give the final answer on its own line as \"Answer: <number>\"."


def gsm8k_extract(text: str) -> Optional[str]:
    """The number after the last "Answer:", else the last number in the text; commas and a trailing period removed."""
    m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
    nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def gsm8k_correct(pred: Optional[str], gold: str) -> bool:
    try:
        return pred is not None and math.isclose(float(pred), float(gold), abs_tol=1e-6)
    except ValueError:
        return False


def letter_ids(tok) -> List[List[int]]:
    """Per letter, the token ids that start with it at the beginning of a reply ("A" and " A")."""
    out = []
    for L in LETTERS:
        ids = {tok.encode(s, add_special_tokens=False)[0] for s in (L, " " + L)}
        out.append(sorted(ids))
    return out


def mc_scores(logits: t.Tensor, ids: List[List[int]], n_choices: int) -> List[float]:
    """Each option's score: its letter's best logit over its token variants."""
    return [float(logits[ids[i]].max()) for i in range(n_choices)]


@t.no_grad()
def multiple_choice(run, items: List[dict], batch_size: int = 16) -> Dict[str, object]:
    ids = letter_ids(run.fw.tokenizer)
    correct, fmt_ok = [], []
    dev = run.model.device
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        enc = run.fmt.format_batch([mc_prompt(x) for x in batch])
        last = last_real_token_indices(enc["attention_mask"])
        keep = t.unique(last)
        logits = run.model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev),
                           position_ids=enc["position_ids"].to(dev), logits_to_keep=keep.to(dev)).logits
        for j, x in enumerate(batch):
            row = logits[j, int((keep == last[j]).nonzero())].float().cpu()
            s = mc_scores(row, ids, len(x["choices"]))
            correct.append(int(max(range(len(s)), key=s.__getitem__) == x["answer"]))
            allowed = {k for L in ids[:len(x["choices"])] for k in L}
            fmt_ok.append(int(int(row.argmax()) in allowed))
    return {"accuracy": sum(correct) / len(correct), "format": sum(fmt_ok) / len(fmt_ok), "correct": correct}


def gsm8k(run, items: List[dict], max_new_tokens: int) -> Dict[str, object]:
    from coherence import is_degenerate
    texts = run.ev.generate_responses([gsm8k_prompt(x) for x in items], max_new_tokens=max_new_tokens)
    preds = [gsm8k_extract(x) for x in texts]
    correct = [int(gsm8k_correct(p, x["answer"])) for p, x in zip(preds, items)]
    tok = run.fw.tokenizer
    deg = [int(is_degenerate(tok.encode(x, add_special_tokens=False))) for x in texts]
    return {"accuracy": sum(correct) / len(correct), "degenerate": sum(deg) / len(deg), "correct": correct,
            "responses": texts}


def jbb_conditions(short: str, tags: List[str]) -> List[tuple]:
    """(name, spec) of every condition in results/analysis/<short>-jbb-safety-<tag>.json, in order, each distinct spec
    once; a name already used for another spec becomes name-tag. Missing files are skipped."""
    out, seen = [], {}
    for tag in tags:
        path = f"results/analysis/{short}-jbb-safety-{tag}.json"
        if not os.path.exists(path):
            print("no jailbreak run", path, flush=True)
            continue
        for name, c in json.load(open(path))["conditions"].items():
            if c["spec"] in seen.values():
                continue
            if name in seen:
                name = f"{name}-{tag}"
            seen[name] = c["spec"]
            out.append((name, c["spec"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conditions", default="", help="comma-separated name=spec[|spec...]")
    ap.add_argument("--from-jbb", default="", help="comma-separated jbb_safety.py tags: also run each of their conditions")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--n", type=int, default=None, help="items per benchmark (default all)")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (a smoke check while the GPU is busy)")
    a = ap.parse_args()

    from jbb_safety import condition_context
    from probe import load_run
    items = {k: v[:a.n] for k, v in json.load(open(ITEMS)).items()}
    from orthogonalize import edit_bytes
    run = load_run(a.model, **({"force_cpu": True} if a.cpu else {}))
    run.ev.reserve_bytes = edit_bytes(run.model)  # room for an edit: condition's weight copy, so one batch size fits all
    path = f"results/analysis/{run.short}-push-capability-{a.tag}.json"
    out = json.load(open(path)) if os.path.exists(path) else {"model": a.model, "n": {k: len(v) for k, v in items.items()},
                                                               "conditions": {}}
    conds = [tuple(c.split("=", 1)) for c in a.conditions.split(",") if c]
    for name, spec in jbb_conditions(run.short, [x for x in a.from_jbb.split(",") if x]):
        if spec not in {s for _, s in conds}:
            conds.append((name if name not in {n for n, _ in conds} else f"{name}-jbb", spec))
    for name, spec in conds:
        if name in out["conditions"] and out["conditions"][name]["spec"] != spec:
            name = f"{name}-2"
        c = next((x for x in out["conditions"].values() if x["spec"] == spec), None) or \
            out["conditions"].setdefault(name, {"spec": spec})
        for b in a.benchmarks.split(","):
            if b in c:
                continue
            with contextlib.ExitStack() as stack:
                # applier.intervened's exit clears every hook, so stacked specs all stay active until the last one exits
                for sp in spec.split("|"):
                    stack.enter_context(condition_context(run, sp))
                c[b] = gsm8k(run, items[b], a.max_new_tokens) if b == "gsm8k" else multiple_choice(run, items[b])
            save_json(path, out)
            extra = f"  degenerate {c[b]['degenerate']:.0%}" if b == "gsm8k" else f"  format {c[b]['format']:.0%}"
            print(f"{name:10s} {b:6s} accuracy {c[b]['accuracy']:.1%}{extra}", flush=True)
    print("saved", path)


if __name__ == "__main__":
    main()
