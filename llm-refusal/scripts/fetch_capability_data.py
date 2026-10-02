"""Fetch the text used to measure what a weight edit costs (capability.py) and the
benign fine-tuning data (finetune.py). Run once; writes into llm-refusal/data/.

  alpaca_completions.json  {"eval": [...500], "train": [...1500]} of
                           {"instruction", "output"} from tatsu-lab/alpaca, seeded
                           sample, excluding every instruction in
                           arditi_harmless_train.json (the direction's training data)
  pile_sample.json         200 documents from monology/pile-uncopyrighted (the
                           first 200 of >= 1000 chars, cut to 4000 chars)

    .venv/bin/python3 llm-refusal/scripts/fetch_capability_data.py
"""
import json
import os
import random

from datasets import load_dataset

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
N_EVAL, N_TRAIN, N_PILE = 500, 1500, 200


def fetch_alpaca():
    with open(os.path.join(DATA, "arditi_harmless_train.json")) as f:
        used = {e["instruction"] if isinstance(e, dict) else e for e in json.load(f)}
    rows = [r for r in load_dataset("tatsu-lab/alpaca", split="train")
            if r["output"].strip() and r["instruction"] not in used]
    random.Random(0).shuffle(rows)
    def item(r):
        instruction = r["instruction"] + (f"\n\n{r['input']}" if r["input"].strip() else "")
        return {"instruction": instruction, "output": r["output"]}
    out = {"eval": [item(r) for r in rows[:N_EVAL]],
           "train": [item(r) for r in rows[N_EVAL:N_EVAL + N_TRAIN]]}
    with open(os.path.join(DATA, "alpaca_completions.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"alpaca: {len(out['eval'])} eval + {len(out['train'])} train (excluded {len(used)} used instructions)")


def fetch_pile():
    docs = []
    for r in load_dataset("monology/pile-uncopyrighted", split="train", streaming=True):
        if len(r["text"]) >= 1000:
            docs.append(r["text"][:4000])
        if len(docs) == N_PILE:
            break
    with open(os.path.join(DATA, "pile_sample.json"), "w") as f:
        json.dump(docs, f, indent=1)
    print(f"pile: {len(docs)} documents")


if __name__ == "__main__":
    fetch_alpaca()
    fetch_pile()
