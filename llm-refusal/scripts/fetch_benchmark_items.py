"""Fetch a fixed sample of benchmark items for push_capability.py, so its queue runs need no network or `datasets`.
Run once; writes llm-refusal/data/benchmark_items.json:

  mmlu   5 test questions per subject (57 subjects, 285), seeded, from cais/mmlu
  arc    200 ARC-Challenge test questions, seeded, from allenai/ai2_arc
  gsm8k  100 test problems, seeded, from openai/gsm8k (answer = the number after ####)

Each multiple-choice item is {"question", "choices": [...], "answer": index}; each GSM8K item {"question", "answer"}.

    .venv/bin/python3 llm-refusal/scripts/fetch_benchmark_items.py
"""
import json
import os
import random
from collections import defaultdict

from datasets import load_dataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
N_MMLU_PER_SUBJECT, N_ARC, N_GSM8K = 5, 200, 100


def mmlu():
    by_subject = defaultdict(list)
    for r in load_dataset("cais/mmlu", "all", split="test"):
        by_subject[r["subject"]].append({"question": r["question"], "choices": list(r["choices"]), "answer": int(r["answer"]),
                                         "subject": r["subject"]})
    rng = random.Random(0)
    return [x for s in sorted(by_subject) for x in rng.sample(by_subject[s], N_MMLU_PER_SUBJECT)]


def arc():
    rows = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    random.Random(0).shuffle(rows)
    return [{"question": r["question"], "choices": list(r["choices"]["text"]),
             "answer": list(r["choices"]["label"]).index(r["answerKey"])} for r in rows[:N_ARC]]


def gsm8k():
    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    random.Random(0).shuffle(rows)
    return [{"question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
            for r in rows[:N_GSM8K]]


if __name__ == "__main__":
    out = {"mmlu": mmlu(), "arc": arc(), "gsm8k": gsm8k()}
    with open(os.path.join(DATA, "benchmark_items.json"), "w") as f:
        json.dump(out, f, indent=1)
    print({k: len(v) for k, v in out.items()})
