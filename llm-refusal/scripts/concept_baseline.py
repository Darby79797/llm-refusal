"""Does the model show a concept at all on its eval prompts? (limb 4.1 gate)

Generates 64 tokens for the concept's positive and negative eval prompts on the plain
model and reports the detector's rate on each side, with a few examples. A concept
whose positive rate is ~0% has nothing for ablation to remove, so run this before
searching (hedging_v2 was searched in April and never checked this way).

  .venv/bin/python3 llm-refusal/scripts/concept_baseline.py --model Qwen/Qwen2.5-0.5B-Instruct --concept hedging_v2

Writes results/analysis/<model>-<concept>-baseline.json.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from env import setup_process_env; setup_process_env()  # before torch is imported
from hf_offline import offline_for_script; offline_for_script()  # before transformers is imported

from probe import behaviour, load_run, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--train", action="store_true", help="also the train prompts")
    a = ap.parse_args()
    run = load_run(a.model, a.concept, direction=False)
    fw = run.fw
    out = {"model": a.model, "concept": a.concept, "max_new_tokens": a.max_new_tokens}
    sets = {"eval": fw.concept.eval_data_fn()}
    if a.train:
        sets["train"] = fw.concept.train_data_fn()
    for split, (pos, neg) in sets.items():
        for side, ps in (("positive", pos), ("negative", neg)):
            texts, labels = [], []
            res = behaviour(fw, ps, max_new_tokens=a.max_new_tokens, texts_out=texts, labels_out=labels)
            res["examples"] = [{"prompt": p, "response": x, "detected": l} for p, x, l in zip(ps, texts, labels)]
            out[f"{split}_{side}"] = res
            print(f"{split} {side}: {res['rate']:.1%} detected (n={res['n']}), log-odds {res['log_odds']}", flush=True)
            for e in res["examples"][:3]:
                print(f"   [{'Y' if e['detected'] else 'n'}] {e['prompt'][:70]!r} -> {e['response'][:160]!r}", flush=True)
    print("saved", save_json(run.path("analysis", a.concept, "baseline"), out))


if __name__ == "__main__":
    main()
