"""What does removing the direction cost? (Arditi et al.'s fine-tuning comparison table.)

`--mode capability` measures the unedited model, the model with the direction
orthogonalised out of its weights (orthogonalize.py), and a control edit against
a seeded random direction of the same rank. The paper omits that control, and
without it "CE rose by 0.05" has no scale. Each variant gets:

  ce_alpaca            CE on Alpaca's reference outputs given the chat-templated
                       instruction (the paper's Alpaca CE; our old `alpaca_ce_loss`
                       scored the instruction tokens instead)
  ce_pile              CE on raw Pile text, no template (off-chat language modelling)
  ce_on_distribution   CE on the unedited model's own greedy completions to Alpaca
                       instructions: how far the edit moves the model off its own
                       harmless behaviour
  lm-eval scores       for the requested --eval-tasks (e.g. mmlu arc_challenge
                       gsm8k truthfulqa_mc2), with --limit per task

Every CE is token-weighted over completion tokens, truncated to 256 tokens per
row. Data: scripts/fetch_capability_data.py.
"""
import contextlib
import gc
import json
import logging
import math
import os
from typing import Dict, List, Optional

import torch as t

from batching import _empty_cache, map_batched, resolve_forward_batch_size
from orthogonalize import edit_bytes, orthogonalized

logger = logging.getLogger(__name__)

DATA = os.path.join(os.path.dirname(__file__), "data")
MAX_TOKENS = 256
N_ON_DISTRIBUTION = 100
RANDOM_DIRECTION_SEED = 0
# Scored tokens per unembedding + CE step. All of a batch's scored tokens at once is
# up to 64 rows x 256 tokens x 128k vocab: ~12 GB of bf16 + fp32 logits on Llama-3,
# outside the forward-pass memory estimate (it OOMed an 8B run). 1024 caps it at ~0.8 GB.
CE_CHUNK_TOKENS = 1024


def load_alpaca(split: str = "eval") -> List[Dict[str, str]]:
    path = os.path.join(DATA, "alpaca_completions.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run scripts/fetch_capability_data.py")
    with open(path) as f:
        return json.load(f)[split]


def load_pile() -> List[str]:
    path = os.path.join(DATA, "pile_sample.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run scripts/fetch_capability_data.py")
    with open(path) as f:
        return json.load(f)


def random_direction(d_model: int, seed: int = RANDOM_DIRECTION_SEED) -> t.Tensor:
    g = t.Generator().manual_seed(seed)
    v = t.randn(d_model, generator=g)
    return v / v.norm()


def completion_ce(model, prompt_formatter, prompts: Optional[List[str]], completions: List[str],
                  batch_size: int, max_completion_tokens: int = MAX_TOKENS, on_split=None) -> float:
    """Token-weighted mean CE (nats) of `completions` given `prompts` (None = raw text).

    Runs the base model for hidden states and applies the unembedding only at the
    scored positions, CE_CHUNK_TOKENS at a time, so the logits never exceed that many
    rows of the vocabulary (the forward-pass batch estimate doesn't include logits).
    """
    device = model.device
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    head = model.get_output_embeddings()

    def score(chunk):
        idx = list(chunk)
        enc = prompt_formatter.format_with_completions(
            [prompts[i] for i in idx] if prompts is not None else None,
            [completions[i] for i in idx], max_completion_tokens=max_completion_tokens)
        with t.no_grad():
            hidden = base(input_ids=enc['input_ids'].to(device), attention_mask=enc['attention_mask'].to(device),
                          position_ids=enc['position_ids'].to(device), use_cache=False).last_hidden_state
            targets = enc['labels'][:, 1:].to(device)
            scored = targets != -100
            h, y = hidden[:, :-1][scored], targets[scored]
            del hidden
            nll = t.cat([t.nn.functional.cross_entropy(head(h[k:k + CE_CHUNK_TOKENS]).float(),
                                                       y[k:k + CE_CHUNK_TOKENS], reduction='none')
                         for k in range(0, len(y), CE_CHUNK_TOKENS)])
        # Per-row (sum, count) so the mean is over tokens, not rows.
        rows = scored.nonzero()[:, 0]
        return [(nll[rows == j].sum().item(), int((rows == j).sum())) for j in range(len(idx))]

    parts = map_batched(score, list(range(len(completions))), batch_size, on_split=on_split)
    total, count = sum(s for s, _ in parts), sum(n for _, n in parts)
    return total / count if count else float('nan')


def run_capability(framework, direction, config: Dict) -> Dict:
    model, fmt, evaluator = framework.model, framework.prompt_formatter, framework.evaluator
    alpaca = load_alpaca("eval")
    pile = load_pile()
    # Two of the three variants hold the edited weight copy: plan every batch around it.
    evaluator.reserve_bytes = edit_bytes(model)
    bs = resolve_forward_batch_size(config.get('gen_batch_size', 'auto'), model, 2 * MAX_TOKENS,
                                    reserve_bytes=evaluator.reserve_bytes)

    # The unedited model's own completions, generated once and scored under every variant.
    od_prompts = [a["instruction"] for a in alpaca[:N_ON_DISTRIBUTION]]
    od_completions = evaluator.generate_responses(od_prompts, max_new_tokens=MAX_TOKENS)

    rand = random_direction(direction.vector.shape[-1])
    cos = float(t.nn.functional.cosine_similarity(rand, direction.vector.float().cpu(), dim=0))
    variants = {
        "baseline": lambda: contextlib.nullcontext(),
        "orthogonalized": lambda: orthogonalized(model, direction.vector),
        "random_orthogonalized": lambda: orthogonalized(model, rand),
    }
    results = {"model": framework.model_name, "concept": framework.concept.name,
               "layer": direction.layer, "position_index": direction.position_index,
               "batch_size": bs, "dtype": str(model.dtype), "max_tokens": MAX_TOKENS,
               "n_alpaca": len(alpaca), "n_pile": len(pile), "n_on_distribution": len(od_prompts),
               "random_direction_seed": RANDOM_DIRECTION_SEED, "random_direction_cos": cos,
               "eval_tasks": config.get('eval_tasks', []), "limit": config.get('limit'),
               "variants": {}}
    for name, make in variants.items():
        logger.info(f"\n--- Capability: {name} ---")
        with make():
            r = {
                "ce_alpaca": completion_ce(model, fmt, [a["instruction"] for a in alpaca],
                                           [a["output"] for a in alpaca], bs, on_split=evaluator._record_split),
                "ce_pile": completion_ce(model, fmt, None, pile, bs, on_split=evaluator._record_split),
                "ce_on_distribution": completion_ce(model, fmt, od_prompts, od_completions, bs,
                                                    on_split=evaluator._record_split),
            }
            r["standard_eval_scores"] = evaluator.run_standard_evals(config.get('eval_tasks', []),
                                                                     limit=config.get('limit'))
        results["variants"][name] = r
        # lm-eval's model wrapper and batches are garbage by now; return them to the
        # allocator before the next variant allocates its edited weights.
        gc.collect()
        _empty_cache()
        logger.info(f"  {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in r.items() if isinstance(v, float))
                    + (f", {r['standard_eval_scores']}" if r["standard_eval_scores"] else ""))
    results["oom_splits"] = evaluator.oom_splits
    results["on_distribution_examples"] = [{"prompt": p, "response": c}
                                           for p, c in zip(od_prompts[:5], od_completions[:5])]

    out_dir = os.path.join("results", "capability")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{framework.model_short}-{framework.concept.name}"
                                 f"-L{direction.layer}-P{direction.position_index}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    logger.info(f"Saved capability results to {path}")
    return results
