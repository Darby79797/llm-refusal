# Results

Current best results and key findings, post right-padding fix (April 2025). All numbers below are from the corrected pipeline; every evaluate number was reproduced bit-for-bit in the 2026-08-08 verification sweep (`results/repro-*.log` — greedy decoding makes runs deterministic). Full per-model detail: `results/results_summary.md` (gitignored, local only — **partially stale**: pre-fix numbers for some models, and its 3-way cross-concept section used superseded direction files); experiment log: `results/extended_results_list.md`.

## Best Layers (Refusal)

| Model | Best | Pos | Global Abl Δ | Layer Abl Δ | Induction | Notes |
|-------|------|-----|--------------|-------------|-----------|-------|
| Qwen2.5-0.5B (24L) | 13 | -4 | -66pp | -59pp | 93% | |
| Qwen2.5-1.5B (28L) | 16 | -1 | -96pp | -95pp | 100% | Near-perfect ablation |
| Qwen2.5-3B (36L) | 21 | -4 | -72pp | -69pp | 98% | |
| Qwen2.5-7B (28L) | 17 | -4 | -75pp | -75pp | 88% | Previously "ablation-resistant" — now works |
| Llama-3-8B (32L) | 12 | -3 | -100pp | -96pp | 99% | Matches Arditi L12; strongest result |
| Llama-3.1-8B (32L) | 12 | -2 | -96pp | -94pp | 73% | |
| Llama-2-7B (32L) | 12 | -1 | -84pp | -18pp | 89% | Global ablation now works (was 0pp) |

Sweet spot: Qwen ~54-61% depth, Llama ~38% depth. Multi-position search matters — 5/7 models selected non-pos-1 positions.

## Key Findings

- **Refusal direction works universally**: 73-100% induction and 66-100pp ablation across all 7 models tested (Qwen2.5 0.5B-7B, Llama-2-7B, Llama-3-8B, Llama-3.1-8B). The "single direction" hypothesis holds strongly.
- **Right-padding fix was critical**: Left-padding with RoPE models corrupts logits for all padded sequences. This was the root cause of all prior Llama "weak results" and the 7B "ablation-resistant" finding. See "Critical Bug Fix" below.
- **Arditi replication successful**: With correct padding, our pipeline selects L12/pos-5 on Llama-3-8B with Arditi's exact data (bypass=-10.7 vs paper's -9.7), strictly passing all criteria.
- **Multi-position search matters**: 5/7 models select non-pos-1 positions. The `max_positions=auto` setting (derived from `assistant_prefix_tokens`) correctly searches all post-instruction token positions.
- **Llama-2 template bug**: HuggingFace's Llama-2 chat template doesn't respond to `add_generation_prompt`. We override with `[INST] {x} [/INST] ` (trailing space). Without this, induction fails completely.
- **Our dataset produces strong directions**: Topic-matched 80+80 pairs work well. Arditi's 128+128 (from AdvBench + MaliciousInstruct + TDC2023) also works when scoring is correct.
- **Sycophancy is harder**: Low induction rates (2-12%), no clean single direction on small models. May need re-evaluation with corrected padding.
- **Hedging: negative result**: 0% behavioral detection across all conditions on all 4 Qwen2.5 models. Models don't hedge on factual questions at baseline.
- **Empathy direction works**: Layer subtraction drops empathy 80%→10% on 1.5B and 25%→0% on 3B. Layer addition induces empathy on 100% (1.5B) and 40% (3B) of neutral prompts.
- **Refusal–sycophancy entanglement**: Ablating the sycophancy direction collapses refusal (−63pp on Qwen2.5-3B) despite near-orthogonality (cos 0.23). See Cross-Concept below.

## Cross-Concept (Qwen2.5-3B, refusal × sycophancy × hedging)

From `results/repro-Qwen2.5-3B-cross_concept.log` (2026-08-08), using the saved post-fix direction files. Detection rates are over 30 prompts per condition (~3pp granularity). Baselines: refusal 66.7%, sycophancy 6.7%, hedging 0%.

**Cosine similarity**: refusal–sycophancy 0.23, refusal–hedging 0.09, sycophancy–hedging 0.07 — all near-orthogonal.

**Subspace**: uncentered SVD on the stacked unit directions gives explained variance [43%, 32%, 26%] — effective rank 3; the three directions span the full 3D space, fairly evenly. (Centered PCA cannot measure span from the origin: k centered vectors are bounded to rank k−1, so it is not used for this.)

**Interference** (single-direction ablation, detection-rate deltas):

| Ablated | Refusal Δ | Sycophancy Δ | Hedging Δ |
|---------|-----------|--------------|-----------|
| Refusal | −63pp | +3pp | 0 |
| Sycophancy | −63pp | +47pp | 0 |
| Hedging | −37pp | +3pp | 0 |

**Joint span ablation** (order-independent, ablating the other two concepts' directions simultaneously): refusal 66.7%→6.7%; sycophancy 6.7%→23.3%; hedging 0%→0%.

Notable: ablating the sycophancy direction is as destructive to refusal as ablating the refusal direction itself; ablating the sycophancy direction *raises* sycophancy detection by 47pp (unexplained — worth an eyeball pass); hedging detection is 0% in every condition, consistent with the hedging negative result.

## Critical Bug Fix: Right-Padding (April 2025)

**Bug**: Left-padding with HuggingFace transformers corrupts logits for all padded prompts in a batch. With left-padding, the model assigns wrong RoPE position encodings to real tokens (position 0 goes to a pad token instead of the first real token). The internal `create_causal_mask` does not correct for this. This affects ALL models with RoPE (Llama, Qwen, etc.) on ALL devices (MPS, CPU, CUDA).

**Impact**: Every prior experiment had corrupted scoring. The search selected wrong directions because bypass/induce/KL scores were computed from garbage logits for most prompts in every batch. Direction vectors from difference-in-means were less affected (activation extraction uses hooks, not logits), but the selected (layer, position) was often wrong.

**Fix**: Switched to right-padding + explicit `position_ids` in `formatting.py`. All `model()` calls in `activations.py`, `scoring.py`, `generation.py`, and `framework.py` now pass `position_ids`. `generate_with_hooks()` tracks per-sequence positions for autoregressive decoding.

**Verification**: `test_per_model.py::TestPaddedBatchConsistency` checks that batched logits match individual processing. `TestLeftPaddingRegression` confirms left-padding fails on all tested models.

**Result**: All 7 models re-searched and re-evaluated. Every model now finds a strictly-passing direction (no fallback). Ablation effectiveness improved dramatically across the board; Llama-3 went from 16% to 99% induction.
