# Future Plans

## Priority 0: Pipeline Throughput (biggest wall-clock wins for sweeps)

Serial evaluate sweeps are generation-bound, not load-bound (model load is ~2 min of a 1.5-2h run on 7-8B MPS). Two cheap structural wins, in order:

1. ~~**Cache the prompt-filtering pass per (model, concept).**~~ Done (2026-09): `results/filter-cache/`, keyed on model/dtype/batch size/detector/prompt set. Default evaluate also dropped the two uninformative conditions (global addition, subtraction), for ~40% less evaluate time in total.
2. ~~**Better batch size selection.**~~ Done (2026-09): `--gen-batch-size auto` is the default (`batching.py`). Key measurement: on MPS bf16 there's a throughput cliff. bs 2/4/8 are equally slow (~15 tok/s on Llama-3-8B), bs 16/32/64 give 7-10×. The MPS caching allocator also had to be bounded with watermarks, or 512-token runs swapped (38 GB of retained KV-cache buffers at bs=16).

   **Correction to the original premise: results are _not_ batch-invariant, so "verify once against a known run as an exact comparison" does not work.** bf16 reduction order depends on batch shape and greedy argmax amplifies it — one flipped token forks the trajectory. On Qwen2.5-0.5B the divergence is small and flat (95% of texts identical to bs=1 at every bs from 2 to 32, rate unchanged), but on Qwen2.5-3B it is not: text identity falls 57.5% → 52.5% and the refusal rate moves 97.5% → 90% going from bs=2 to bs=8. Practical rule: raise it freely for ≤1.5B models, keep it fixed within any set of runs being compared, and re-run a whole comparison (baseline + all intervention conditions) if you change it.

   Further win, not yet done: sort prompts by length before batching. `generate_with_hooks` runs until every row in a batch finishes, so a batch costs as much as its longest generation.

Related, larger refactor: a **per-model experiment queue** — load a model once, then run all pending concepts/modes against it, sharing baseline generations and filtering across concepts before unloading.

## Current State (Post Generation-Path Fix, 2026-08-11)

Two padding bugs have now been fixed: the April 2026 indexing bug in the scoring path, and the August 2026 generation bug where evaluation decoded from a pad slot. All 7 refusal `evaluate` runs were regenerated (`results/rerun3-MANIFEST.md`). Refusal direction finding works universally: **83.8-100% induction and 87.9-100pp global ablation**. The "single direction" hypothesis is confirmed, and more strongly than pre-fix numbers showed — the artifact was suppressing the effect.

**What's solid:**
- Refusal `evaluate`: re-run post-fix on all 7 models (Qwen 0.5B-7B, Llama-2/3/3.1), with the continuous log-odds metric alongside every rate
- Arditi replication of *direction selection*: L12/pos-5 on Llama-3, bypass=-10.7 (paper: -9.7)
- Multi-position search: auto-derived from assistant_prefix_tokens
- Padding invariant is now asserted, not assumed (`formatting.last_real_token_indices`), with unit + per-model tests

**What needs re-running** (all used at least one of the two broken paths):
- ~~Refusal **search** on all models~~ Done 2026-09-23: all 7 pass strictly; 6/7 re-select the old coordinates (Qwen2.5-0.5B: L13→L14). Evaluate re-run at the selected coordinates reproduces 2026-08 exactly. See RESULTS.md.
- ~~Sycophancy, hedging, empathy: search + evaluation on all models~~ Done 2026-09-24 (RESULTS.md "Non-Refusal Concepts"): empathy works on 6/7, hedging is a real negative, sycophancy needs a response-contrast direction.
- ~~Cross-concept analysis~~ Done 2026-09-25 on all 7 models (4 concepts); the −63pp headline does not replicate.
- ~~The paper's **safety score**~~ Done 2026-09-25 on all 7 models (ablation → 66-89% LlamaGuard-unsafe).

## Next (from the 2026-09-25 sweep)

1. **Sycophancy from a response contrast.** Same prompts, activations when the model was vs wasn't sycophantic (CAA-style), instead of the prompt contrast (which encodes framing) or behavioral filtering (too few sycophantic responses on ≥1.5B models).
2. **Selection rule:** break bypass near-ties by induce (Llama-3.1: L11/P-1 gives 100% induction vs 85% at the selected L12/P-2).
3. **Detection:** widen the refusal phrase list for non-apologetic refusals ("I do not provide…", "I must strongly advise against…"), or lead with log-odds where phrasing can change (cross-concept).
4. **CAA open-ended half** (LLM judge via OpenRouter, pending a key); check "ablating refusal raises sycophancy" against the text.
5. `hedging_v2` (subjective questions); a larger empathy eval set (20 prompts now).

## Priority 1: Re-run Non-Refusal Concepts (DONE 2026-09-24, see RESULTS.md)

All sycophancy, hedging, and empathy results were produced through the broken generation path on Qwen2.5 models — the family that bug hit hardest. The directions may be wrong and every evaluation is unreliable. Note in particular that the hedging "0% detection" negative result is exactly what a corrupted generation path also produces, so it is currently uninterpretable in either direction.

**Action**: Re-run `--mode search` then `--mode evaluate` for each concept on each model. Start with sycophancy (most developed) on Qwen2.5-3B (best prior results).

The sycophancy "low induction rates (2-12%)" finding may be an artifact of the broken generation path (detection rates measured from text decoded off a pad slot). With both padding bugs fixed, we might find a much stronger sycophancy direction — the refusal numbers all moved in that direction.

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
