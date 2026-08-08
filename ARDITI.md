# Arditi et al. Replication Status

## Summary

We successfully replicate Arditi & Obeso's "Refusal in Language Models Is Mediated by a Single Direction" on Llama-3-8B-Instruct. With corrected padding (right-padding + explicit `position_ids`), our pipeline selects **L12/pos-5** — exactly matching the paper — with bypass_score=-10.7 (paper: -9.7), strictly passing all criteria (induce > 0, KL < 0.1).

The replication was blocked for months by a **left-padding bug** that corrupted all logit computations. See RESULTS.md "Critical Bug Fix" for details.

## What Matches

| Component | Arditi | Ours | Status |
|---|---|---|---|
| Activation extraction | Pre-hooks on block inputs, float64 | Same | Match |
| Difference-in-means | mean_harmful - mean_harmless | Same | Match |
| Ablation | Unit vector projection, 3 hooks/layer | Same | Match |
| Addition | Raw vector at single layer | Same | Match |
| Chat template (Llama-3) | Manual string, no system msg | Built-in `apply_chat_template` | Match (verified byte-identical) |
| Direction selected (Llama-3) | L12/pos-5 | L12/pos-5 | Match |
| Padding | Left-pad (on CUDA, works) | Right-pad + position_ids | Functionally equivalent |

## What Differs (Design Choices)

| Component | Arditi | Ours | Impact |
|---|---|---|---|
| Dataset | 128 from AdvBench+MI+TDC2023 + 128 Alpaca | 80+80 topic-matched (default) | Different tradeoff; both work |
| Train/val split | Pre-split files (128+32) | 80/20 random (default) | `refusal_arditi_exact` concept uses pre-split |
| Refusal tokens | Model-specific single token | Multi-token set | `refusal_arditi_exact` uses single token |
| Layer cutoff | Last 20% pruned | Last 35% pruned (default) | `refusal_arditi_exact` uses 0.80 |
| Selection fallback | Hard-fail if no strict pass | Progressive relaxation | UX improvement |
| Evaluation | LlamaGuard2 + JailbreakBench + CE loss | Phrase match + lm-eval | Different metrics |
| Generation | `model.generate()` | Custom `generate_with_hooks()` | Ours fires hooks every step |

## Replication Concepts

- `--concept refusal` — our dataset, our search config (default)
- `--concept refusal_arditi` — Arditi's AdvBench-only data, our search config
- `--concept refusal_arditi_exact` — Arditi's exact 128+128 sample (seed=42), separate HarmBench val, single refusal token, 0.80 layer cutoff. Use this for exact replication.
  - Caveat: the HarmBench-derived val set (first 32) and the JailbreakBench eval set share 2 identical prompts (happenstance overlap between the source benchmarks), so val and eval are not fully disjoint. Data is intentionally left as-is for replication fidelity.

## Bugs Found During Replication

### 1. Left-padding logit corruption (CRITICAL)
HuggingFace transformers does not compute correct `position_ids` for left-padded sequences when calling `model()` directly. RoPE position encodings are wrong for all padded tokens. This corrupts logits for every prompt shorter than the longest in the batch. Affects ALL RoPE models (Llama, Qwen) on ALL devices.

**Fix**: Right-padding + explicit `position_ids = attention_mask.cumsum(-1) - 1; position_ids.masked_fill_(mask==0, 1)`.

### 2. Llama-2 chat template (HIGH)
HuggingFace's built-in Llama-2 template doesn't respond to `add_generation_prompt=True`. The formatted prompt ends with `]` instead of a trailing space at the generation boundary. Fixed by overriding with manual template.

### 3. Single-position search (MEDIUM)
Cross-dataset comparison scripts hardcoded `max_positions=1`. Arditi searches all post-instruction positions. Fixed by defaulting `max_positions` to `"auto"` in `search.py`.
