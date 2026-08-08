# Future Plans

## Priority 0: Pipeline Throughput (biggest wall-clock wins for sweeps)

Serial evaluate sweeps are generation-bound, not load-bound (model load is ~2 min of a 1.5-2h run on 7-8B MPS). Two cheap structural wins, in order:

1. **Cache the prompt-filtering pass per (model, concept).** Every evaluate run regenerates ~160 baseline responses just to filter prompts (~15-20% of run time), and the result is identical across runs on the same model+concept. Persist the filtered prompt list (or the baseline generations) keyed by model+concept+prompt-set hash, and reuse.
2. **Better batch size selection.** `batch_size=2` is hardcoded and very conservative; with right-padding + explicit position_ids verified correct on all 7 models, larger batches (e.g. 8, or memory-adaptive per model size) should give 2-3× generation throughput on 48GB. Results should be batch-invariant now — verify once against a known run (greedy decoding makes this an exact comparison), then raise the default.

Related, larger refactor: a **per-model experiment queue** — load a model once, then run all pending concepts/modes against it, sharing baseline generations and filtering across concepts before unloading.

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

### Jailbreak direction analysis
Grab 50+ known jailbreak prompts (DAN, AIM, roleplay prefixes, multi-turn escalation). Compute the refusal direction's projection magnitude on these prompts and compare to normal harmful prompts. If successful jailbreaks suppress the direction below a threshold, we have a mechanistic jailbreak detector. Test whether: (a) jailbreaks work by suppressing the refusal direction or by a different mechanism, (b) the direction's projection predicts jailbreak success, (c) monitoring the direction at inference time can flag jailbreak attempts. Connects mechanistic interpretability to the safety/red-teaming literature.

### Category-specific refusal directions
Compute difference-in-means separately for each harm category (weapons, hacking, bioweapons, fraud, self-harm, violence). Measure pairwise cosine similarity between category-specific directions. If parallel → refusal is truly unified. If divergent → the "single direction" is an average, and targeted editing becomes possible. Test cross-category transfer (does the weapons direction induce refusal on hacking prompts?).

### Per-layer refusal direction trajectory
For each prompt, project every layer's residual stream onto the refusal direction → trajectory across depth. Compare harmful vs. benign prompts — divergence point localizes where the refusal decision happens. Nearly free (one forward pass), produces a key figure, validates circuit analysis findings.

### Separate bypass vs induction directions
The current pipeline uses one direction for both. But bypass (ablation) and induction (addition) may have different optimal directions. Search separately and compare cosine similarity.

### Direction transfer across model families
Do directions transfer? Take the Qwen-3B refusal direction, project it into the Llama-3-8B residual stream (via alignment), and test if it still ablates/induces refusal. Would establish whether refusal is architecturally universal or model-specific.

### Strength curves
Now that scoring is correct, systematically map induction rate vs strength for each model. Find the minimum strength needed for >90% induction and the maximum strength before coherence degrades.
