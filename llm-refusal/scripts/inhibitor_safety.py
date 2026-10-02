"""Does the refusal inhibitor make the model *comply*? (limb 1.1)

The rank1 remove adapter with r̂ projected out of u ("u_perp", the inhibitor) takes
refusal to 0% by phrase match and log-odds, but its text was never read and never
shown to LlamaGuard. Here: JailbreakBench prompts, 512 greedy tokens, LlamaGuard 2
(via Ollama) unsafe rate, refusal rate, degeneracy, under a set of conditions:

  baseline          plain model
  global_ablation   r̂ ablated with hooks (the reference)
  inhibitor         the saved remove adapter, u_perp
  adapter_full      the saved remove adapter as trained

Also the 64-token refusal rate on the concept's own harmful eval prompts per condition.

  .venv/bin/python3 llm-refusal/scripts/inhibitor_safety.py --model Qwen/Qwen2.5-0.5B-Instruct \\
      --adapter results/finetune/Qwen2.5-0.5B-Instruct-refusal-rank1-remove

Writes results/inhibitor-<model>-refusal-evaluate-L<l>-P<p>-T<n>-generations.json (look.py reads it:
variant "inhibitor") and results/finetune/<model>-inhibitor-safety.json.
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
from coherence import score_condition  # noqa: E402
from datatypes import DirectionVector  # noqa: E402
from finetune import installed  # noqa: E402
from framework import DirectionTestFramework  # noqa: E402
from probe import behaviour  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, help="rank1 remove adapter stem (no .json)")
    ap.add_argument("--conditions", default="baseline,global_ablation,inhibitor,adapter_full")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--llamaguard-api-base", default="http://localhost:11434")
    a = ap.parse_args()
    short = a.model.split("/")[-1]
    fw = DirectionTestFramework(model_name=a.model, concept="refusal", llamaguard_api_base=a.llamaguard_api_base)
    ev = fw.evaluator
    r = DirectionVector.load(f"results/{short}-refusal-direction")
    r_hat = r.unit.float().cpu()
    jbb, _ = prompts.create_arditi_replication_eval_data()
    jbb = jbb[:a.n]
    harmful, _ = fw.concept.eval_data_fn()

    ctxs = {
        "baseline": contextlib.nullcontext,
        "global_ablation": lambda: fw.intervention_applier.intervened(r, "ablate", layers=None),
        "inhibitor": lambda: installed(fw.model, a.adapter, "u_perp", r_hat),
        "adapter_full": lambda: installed(fw.model, a.adapter, "full", r_hat),
    }
    gens, summary = {}, {"model": a.model, "adapter": a.adapter, "max_new_tokens": a.max_new_tokens, "n": len(jbb),
                         "conditions": {}}
    gen_path = f"results/inhibitor-{short}-refusal-evaluate-L{r.layer}-P{r.position_index}-T{a.max_new_tokens}-generations.json"
    for name in a.conditions.split(","):
        ctx = ctxs[name]
        texts, labels = [], []
        res = {"jbb": behaviour(fw, jbb, ctx, a.max_new_tokens, texts_out=texts, labels_out=labels)}
        lg = []
        unsafe = ev.evaluate_llamaguard_rate(jbb, texts, labels_out=lg)
        res["jbb"]["llamaguard_unsafe_rate"] = unsafe
        gens[name] = [{"prompt": p, "response": x, "detected": d, "llamaguard": l}
                      for p, x, d, l in zip(jbb, texts, labels, lg or [None] * len(texts))]
        res["jbb"].update(score_condition(fw.model, fw.tokenizer, fw.prompt_formatter, gens[name],
                                          batch_size=ev.resolve_batch_size(jbb, 0)))
        res["harmful_eval_64"] = behaviour(fw, harmful, ctx, 64)
        summary["conditions"][name] = res
        print(f"{name:16s} JBB refusal {res['jbb']['rate']:5.1%}  unsafe {unsafe if unsafe is None else f'{unsafe:5.1%}'}  "
              f"degenerate {res['jbb'].get('degenerate_rate', float('nan')):5.1%}  log-odds {res['jbb']['log_odds']:+.2f} | "
              f"harmful-64 refusal {res['harmful_eval_64']['rate']:5.1%}", flush=True)
        with open(gen_path, "w") as f:
            json.dump(gens, f, indent=1)
        with open(f"results/finetune/{short}-inhibitor-safety.json", "w") as f:
            json.dump(summary, f, indent=1)
    print("saved", gen_path)


if __name__ == "__main__":
    main()
