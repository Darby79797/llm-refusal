"""Why does lm-eval GSM8K depend on batch size? (PROPOSED_PLANS item 9)

Llama-3-8B scored 0.53 at lm-eval batch 1 and 0.73 at batch 16. This runs the same
GSM8K questions at both batch sizes with per-sample logging and diffs them: the
prompt lm-eval built, the generation, the extracted answer and correctness.

    .venv/bin/python3 llm-refusal/scripts/gsm8k_batch_diff.py --model meta-llama/Meta-Llama-3-8B-Instruct --limit 30

Writes results/gsm8k-batch-diff-<model>.json.
"""
import argparse
import json
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
_high = os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", str(min(0.6, 0.75 * float(_high))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t  # noqa: E402
import lm_eval  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from formatting import ChatPromptFormatter  # noqa: E402


def run(model, tok, batch_size, limit):
    out = lm_eval.simple_evaluate(model=HFLM(pretrained=model, tokenizer=tok, device=str(model.device),
                                             batch_size=batch_size),
                                  tasks=["gsm8k"], limit=limit, log_samples=True)
    rows = {}
    for s in out["samples"]["gsm8k"]:
        # lm-eval versions store request arguments as [[context, gen_kwargs]] or as
        # {"gen_args_0": {"arg_0": context, ...}}.
        a = s["arguments"]
        a0 = a[0] if isinstance(a, (list, tuple)) else next(iter(a.values()))
        prompt = a0[0] if isinstance(a0, (list, tuple)) else a0.get("arg_0")
        rows[s["doc_id"]] = {"prompt": prompt, "generation": s["resps"][0][0], "extracted": s["filtered_resps"][0],
                             "target": s["target"],
                             "correct": next((v for k, v in s.items() if k.startswith("exact_match")), None)}
    return out["results"]["gsm8k"], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    device = "mps" if t.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(device).eval()
    if model.dtype == t.float16 and device == "mps":
        model = model.to(t.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    ChatPromptFormatter(tok)   # the pipeline's tokenizer setup (pad token, right padding)

    res1, rows1 = run(model, tok, 1, args.limit)
    res16, rows16 = run(model, tok, 16, args.limit)
    print("batch 1 :", {k: v for k, v in res1.items() if "exact_match" in k})
    print("batch 16:", {k: v for k, v in res16.items() if "exact_match" in k})
    diffs = []
    for i in sorted(rows1):
        a, b = rows1[i], rows16[i]
        same_prompt, same_gen = a["prompt"] == b["prompt"], a["generation"] == b["generation"]
        if not same_gen or a["extracted"] != b["extracted"]:
            k = next((j for j, (x, y) in enumerate(zip(a["generation"], b["generation"])) if x != y),
                     min(len(a["generation"]), len(b["generation"])))
            diffs.append({"doc_id": i, "same_prompt": same_prompt, "first_diff_char": k,
                          "bs1": a, "bs16": b})
    print(f"{len(diffs)}/{len(rows1)} questions differ; prompts identical in all: "
          f"{all(rows1[i]['prompt'] == rows16[i]['prompt'] for i in rows1)}")
    for d in diffs[:6]:
        print(f"\n#{d['doc_id']}: extracted bs1={d['bs1']['extracted']!r} bs16={d['bs16']['extracted']!r} "
              f"target={d['bs1']['target']!r}; first difference at char {d['first_diff_char']}")
        print("  bs1 :", repr(d["bs1"]["generation"][:300]))
        print("  bs16:", repr(d["bs16"]["generation"][:300]))
    os.makedirs("results", exist_ok=True)
    path = f"results/gsm8k-batch-diff-{args.model.split('/')[-1]}.json"
    with open(path, "w") as f:
        json.dump({"batch1": res1, "batch16": res16, "diffs": diffs}, f, indent=1, default=str)
    print("saved", path)


if __name__ == "__main__":
    main()
