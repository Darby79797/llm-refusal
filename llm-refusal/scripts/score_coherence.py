"""Backfill coherence scores into an evaluate run's generations JSON.

Usage:
  python llm-refusal/scripts/score_coherence.py MODEL GENERATIONS_JSON [--dtype float32|bfloat16] [--batch-size 4]

Adds per-response `nll` and `degenerate` fields in place and prints a per-condition
table (refusal rate, mean response NLL under the clean model, degenerate rate).
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from formatting import ChatPromptFormatter
from coherence import score_condition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("generations_json")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    device = "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
    dtype = args.dtype if args.dtype == "auto" else getattr(t, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    formatter = ChatPromptFormatter(tokenizer)

    with open(args.generations_json) as f:
        generations = json.load(f)

    print(f"{'condition':<28} {'rate':>6} {'nll':>7} {'degen':>6}")
    for condition, entries in generations.items():
        summary = score_condition(model, tokenizer, formatter, entries, batch_size=args.batch_size)
        rate = sum(e['detected'] for e in entries) / len(entries)
        print(f"{condition:<28} {rate:6.3f} {summary['response_nll']:7.3f} {summary['degenerate_rate']:6.3f}", flush=True)

    with open(args.generations_json, "w") as f:
        json.dump(generations, f, indent=1)


if __name__ == "__main__":
    main()
