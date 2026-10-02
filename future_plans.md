# Future Plans

Open work lives in [PROPOSED_PLANS.md](PROPOSED_PLANS.md); this file only tracks what is done and what is not planned there.

## Where things stand (2026-10-02)

The refusal direction r̂ controls refusal on all 7 models, and the safety half of Arditi replicates. Weight orthogonalisation is the ablated model, baked in: nearly free on Llama, 5-20x a random edit on Qwen. Fine-tuning finds a separate rank-one off-switch rather than r̂, and 8-16 refusal examples (~1% of the data) regrow refusal after the edit, along a new axis. Editing only the blocks before r̂'s layer removes refusal; on Qwen the later blocks carry much of the edit's cost. r̂ is a probe from L1-L8, is one direction across harm categories, and jailbreaks on the small Qwens work by lowering it. Empathy works on 6/7 models, hedging on factual questions is a negative, sycophancy needs a response-contrast direction, and ablating refusal raises sycophancy on every Qwen. The Limb Experiments section of RESULTS.md is still in progress (3B/8B replications and the open-ended CAA half are queued or running). Numbers: the Summary at the top of [RESULTS.md](RESULTS.md).

## Done, see RESULTS.md

- Re-runs after the two padding bugs (search, evaluate, non-refusal concepts, cross-concept, safety score): Best Layers, Safety Score, Non-Refusal Concepts, Cross-Concept, "Two Padding Bugs".
- Sycophancy from a response contrast: Limb Experiments section 4 (`sycophancy_response`, up to 83% induced).
- Selection tie-break by induce; non-apologetic refusal phrases: Best Layers / Key Findings.
- Arditi weight orthogonalisation and capability cost: "Weight Orthogonalisation", "What the Edit Costs".
- Rank-one adapters and regrowth (including the dose-response threshold): "Does Fine-Tuning Rediscover r̂?", "Does Refusal Regrow After the Edit?", Limb section 2.
- Per-layer trajectory, category-specific directions, jailbreak projection, separate bypass/induce coordinates: Limb section 3.
- Cheapest edit that removes refusal (layer-restricted edits): Limb section 1.
- Hedging on subjective prompts (`hedging_v2`): Limb section 4 (0% on the hedging detector; models deflect instead, `opinion_avoidance` queued).
- Strength curves for sycophancy: Limb section 4 table. Refusal strength curves were not done as a systematic sweep.
- Pipeline throughput: filter cache and `--gen-batch-size auto` (see below).

## Still open and not in PROPOSED_PLANS.md

- Larger models (13B+) for sycophancy, which may need more capacity than refusal.
- Direction transfer across model families (e.g. Qwen-3B r̂ into Llama-3-8B via an alignment): is refusal architecturally universal?
- Arditi full evaluation with `refusal_arditi_exact` on Llama-3-8B, with JailbreakBench numbers compared to the paper's.
- A larger empathy eval set (20 prompts now).
- Part 4 current checkpoints (Qwen3, Gemma-3, Llama-3.2-3B) need a download; only Llama-3.2-1B is cached (PROPOSED_PLANS item 8).
- Sort prompts by length before batching: `generate_with_hooks` runs until every row finishes, so a batch costs its longest generation.
- Per-model experiment queue: load a model once and run all pending concepts/modes, sharing baseline generations and filtering. A file-based job queue exists (`results/queue/runner.sh`) but runs one job per load.

## Throughput notes

- On MPS bf16 there is a throughput cliff: bs 2/4/8 are equally slow (~15 tok/s on Llama-3-8B), bs 16/32/64 are 7-10x faster. `--gen-batch-size auto` picks from this; the MPS allocator needs watermarks (`env.py`) or 512-token runs swap.
- Results are not batch-invariant in bf16 (Qwen2.5-3B: refusal 97.5% to 90% from bs 2 to 8). Compare conditions within one run, or re-run both sides at auto.
