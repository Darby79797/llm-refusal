# Future Plans

## Current State (Post Padding Fix)

The right-padding fix (April 2025) resolved the Arditi replication and dramatically improved all results. Refusal direction finding now works universally across 7 models with 73-100% induction and 66-100pp ablation. The "single direction" hypothesis is confirmed.

**What's solid:**
- Refusal: works on all 7 models (Qwen 0.5B-7B, Llama-2/3/3.1)
- Arditi replication: L12/pos-5 on Llama-3, bypass=-10.7 (paper: -9.7)
- Multi-position search: auto-derived from assistant_prefix_tokens
- Per-model tests: padding, template, position_ids, activation consistency

**What needs re-running** (prior results used corrupted scoring):
- Sycophancy search + evaluation on all models
- Hedging search + evaluation on all models
- Empathy search + evaluation on all models
- Cross-concept analysis (refusal × sycophancy × hedging)

## Priority 1: Re-run Non-Refusal Concepts

All sycophancy, hedging, and empathy experiments were conducted with left-padding (corrupted scoring). The directions found may be wrong, and all evaluations are unreliable.

**Action**: Re-run `--mode search` then `--mode evaluate` for each concept on each model. Start with sycophancy (most developed) on Qwen2.5-3B (best prior results).

The sycophancy "low induction rates (2-12%)" finding may be an artifact of corrupted scoring. With correct padding, we might find a much stronger sycophancy direction.

## Priority 2: Cross-Concept Re-analysis

The cross-concept findings (refusal × sycophancy interference, 2D subspace) were computed with corrupted directions. Re-run after Priority 1 produces new directions.

Key question: does ablating sycophancy still reduce refusal by 57pp? This was the most surprising finding and needs verification with correct directions.

## Priority 3: Arditi Full Evaluation

Run `--mode evaluate` with `--concept refusal_arditi_exact` on Llama-3-8B to get full detection rates, then compare to paper's numbers. Consider adding LlamaGuard2 evaluation and JailbreakBench for apples-to-apples comparison.

## Priority 4: New Research Directions

### Hedging on subjective prompts
Hedging on factual questions was a negative result (0% detection). But models may hedge more on opinion/subjective/ambiguous topics. Design a new hedging prompt set with opinion questions and re-test.

### Larger models
Test on 13B+ models for sycophancy (which is harder than refusal at small scale). The refusal direction is already clean at all sizes — sycophancy may need more capacity.

### Separate bypass vs induction directions
The current pipeline uses one direction for both. But bypass (ablation) and induction (addition) may have different optimal directions. Search separately and compare cosine similarity.

### Direction transfer across model families
Do directions transfer? Take the Qwen-3B refusal direction, project it into the Llama-3-8B residual stream (via alignment), and test if it still ablates/induces refusal. Would establish whether refusal is architecturally universal or model-specific.

### Strength curves
Now that scoring is correct, systematically map induction rate vs strength for each model. Find the minimum strength needed for >90% induction and the maximum strength before coherence degrades.
