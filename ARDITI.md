# Arditi et al. Replication Status

## Summary

We replicate the **direction-selection** result of Arditi & Obeso's "Refusal in Language Models Is Mediated by a Single Direction" on Llama-3-8B-Instruct: our pipeline selects **L12/pos-5** — exactly matching the paper — with bypass_score=-10.7 (paper: -9.7), strictly passing all criteria (induce > 0, KL < 0.1).

Scope: this covers §2.3 (extracting the direction) and the refusal-score half of §3. It does **not** yet cover the paper's *safety* score (Llama Guard 2 over JailbreakBench, 512-token generations), so "ablation elicits unsafe completions" is untested here — only "ablation removes refusal phrasing". §4 (weight orthogonalisation) and §5 (adversarial suffixes) are out of scope.

The replication was blocked for months by an **indexing bug on padded batches**, and the behavioural numbers were subsequently wrong again for a different padding reason. Both are recorded under "Bugs Found" below and in RESULTS.md "Two Padding Bugs".

## What Matches

| Component | Arditi | Ours | Status |
|---|---|---|---|
| Activation extraction | Pre-hooks on block inputs, float64 | Same | Match |
| Difference-in-means | mean_harmful - mean_harmless | Same | Match |
| Ablation | Unit vector projection, 3 hooks/layer | Same | Match |
| Addition | Raw vector at single layer | Same | Match |
| Chat template (Llama-3) | Manual string, no system msg | Built-in `apply_chat_template` | Match (verified byte-identical) |
| Direction selected (Llama-3) | L12/pos-5 | L12/pos-5 | Match |
| Padding | Left-pad, read at index -1 | Right-pad, read at `last_real_token_indices()` | Equivalent — both are internally consistent. Their scheme was never broken; ours was broken by mixing left padding with right-padding index arithmetic (see Bugs below) |
| Decoding | Greedy | Greedy (`generate_with_hooks`, pure argmax) | Match — but only after disabling Qwen's shipped `repetition_penalty` (see Bug 1c) |

## What Differs (Design Choices)

| Component | Arditi | Ours | Impact |
|---|---|---|---|
| Dataset | 128 from AdvBench+MI+TDC2023 + 128 Alpaca | 90 harmful + 64 harmless, topic-matched (default) | Different tradeoff; both work. Note ours is unbalanced and unpaired, despite "topic-matched" |
| Train/val split | Pre-split files (128+32) | 80/20 random (default) | `refusal_arditi_exact` concept uses pre-split |
| Train filtering | By refusal score (harmful > 0, harmless < 0) | Default path: by generating 64 tokens and phrase-matching | `refusal_arditi_exact` filters train by score like the paper (since 2026-09; its earlier L12/pos-5 result used generation-based filtering) |
| Refusal tokens | Model-specific single token | Multi-token set | `refusal_arditi_exact` uses single token |
| Layer cutoff | Last 20% pruned | Last 35% pruned (default) | `refusal_arditi_exact` uses 0.80 |
| Selection fallback | Hard-fail if no strict pass | Progressive relaxation | UX improvement |
| Evaluation | LlamaGuard2 + JailbreakBench + CE loss | Phrase match + lm-eval | Different metrics |
| Generation | `model.generate()` | Custom `generate_with_hooks()` | Ours reads the last *real* token under right padding, and does pure greedy argmax with no logits processors |

## Replication Concepts

- `--concept refusal` — our dataset, our search config (default)
- `--concept refusal_arditi` — Arditi's AdvBench-only data, our search config
- `--concept refusal_arditi_exact` — Arditi's exact 128+128 sample (seed=42), separate HarmBench val, single refusal token, 0.80 layer cutoff. Use this for exact replication.
  - Caveat: the HarmBench-derived val set (first 32) and the JailbreakBench eval set share 2 identical prompts (happenstance overlap between the source benchmarks), so val and eval are not fully disjoint. Data is intentionally left as-is for replication fidelity.

## Bugs Found During Replication

### 1. Indexing left-padded batches as if right-padded (CRITICAL)
This was originally recorded here as "left-padding corrupts logits via wrong RoPE position encodings". **That diagnosis was wrong.** RoPE depends only on relative positions, so uniformly shifting every real token is a no-op: measured on Qwen2.5-0.5B, a left-padded batch with no `position_ids`, read at the true last token, matches the unpadded run's top-1 and correlates 0.99978 on logits.

The real bug was **indexing**. `scoring._get_logits` read `attention_mask.sum(-1)-1` and `activations` read `true_len + pos_idx` — expressions valid only under *right* padding. Under left padding they address a token in the middle of the prompt (top-1 `' Alibaba'` vs `'2'`; max |Δlogit| 23.7). Note this means Arditi's own left-padded pipeline was never affected: they read index `-1`, which is correct under left padding.

**Fix**: right-pad everywhere, and route every boundary read through `formatting.last_real_token_indices()`, which asserts the invariant instead of assuming it. The correct lesson is *never index a padded batch by true length without knowing the padding side* — not "always pass position_ids".

### 1b. Generating from a pad slot (CRITICAL, introduced by the fix above)
Evaluation called `model.generate()` on right-padded batches. `generate()` reads next-token logits from the **last column**, which is a pad slot for every row shorter than the longest in its batch, so those rows decoded from `<|endoftext|>` at position sentinel 1. The corrupted distribution has entropy 5.06 nats vs 2.58, with the refusal-onset token `'I'` falling from rank 1 (p=0.327) to rank 5 (p=0.041). Corruption is binary in pad count (logits identical for k=1..10) and confined to the first decoding step.

Impact on the Arditi comparison: every Qwen refusal rate was depressed by up to 24pp, understating the paper's effect. Llama was largely spared because `<|eot_id|>` is a turn boundary — the model re-opens a turn and refuses again. **Fix**: evaluation uses `generate_with_hooks()`. See RESULTS.md "Two Padding Bugs".

### 1c. Undeclared repetition penalty on Qwen (HIGH)
`model.generate()` applies the model's shipped `generation_config`, and Qwen2.5 ships `repetition_penalty` 1.05–1.1. It is a logits processor, so it applied even under `do_sample=False`. Every CLI Qwen result was decoded with a repetition penalty rather than the plain greedy decoding the paper specifies (§2.5), and was not comparable with the `scripts/` results, which used pure greedy. With `repetition_penalty=1.0` the two paths agree bit-for-bit.

### 2. Llama-2 chat template (HIGH)
HuggingFace's built-in Llama-2 template doesn't respond to `add_generation_prompt=True`. The formatted prompt ends with `]` instead of a trailing space at the generation boundary. Fixed by overriding with manual template.

### 3. Single-position search (MEDIUM)
Cross-dataset comparison scripts hardcoded `max_positions=1`. Arditi searches all post-instruction positions. Fixed by defaulting `max_positions` to `"auto"` in `search.py`.
