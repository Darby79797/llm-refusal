# Results

Current best results and key findings, after the **generation-path padding fix (2026-08-11)**. All evaluate numbers below come from `results/rerun3-*-evaluate.log` (index: `results/rerun3-MANIFEST.md`). They supersede the pre-fix numbers, which are preserved as `results/prefix-*.log` and `results/repro-*.log`.

Every prior evaluate number in this file was wrong, by up to 24pp, because generation ran through `model.generate()` on right-padded batches. See "Two Padding Bugs" below. Full per-model detail: `results/results_summary.md` (gitignored, local only — **stale**, pre-fix); experiment log: `results/extended_results_list.md`.

## Best Layers (Refusal)

Baseline / ablation / addition are phrase-match detection rates; `log-odds` is the continuous refusal metric (Arditi's proxy — one forward pass, no decoding) measured under the same intervention. Both are reported because a rate derived from greedy argmax flips on near-tied logits while the log-odds moves smoothly; a real effect moves both.

| Model | Best | Pos | Baseline | Global Abl Δ | Layer Abl Δ | Induction | log-odds base → global abl | dtype/bs |
|-------|------|-----|----------|--------------|-------------|-----------|-----------------------------|----------|
| Qwen2.5-0.5B (24L) | 13 | -4 | 88.9% | **-87.9pp** | -83.8pp | 98.8% | +0.60 → -4.21 | fp32 / 8 |
| Qwen2.5-1.5B (28L) | 16 | -1 | 92.9% | -92.9pp | -90.9pp | 100% | +1.18 → -4.66 | fp32 / 8 |
| Qwen2.5-3B (36L) | 21 | -4 | 96.0% | **-96.0pp** | -91.9pp | 100% | +4.07 → -8.15 | fp32 / 8 |
| Qwen2.5-7B (28L) | 17 | -4 | 88.9% | **-88.9pp** | -88.9pp | 100% | +4.20 → -14.56 | bf16 / 2 |
| Llama-3-8B (32L) | 12 | -3 | 100% | -100pp | -93.9pp | 97.5% | +9.79 → -10.60 | bf16 / 8 |
| Llama-3.1-8B (32L) | 12 | -2 | 95.0% | -95.0pp | -92.9pp | 83.8% | +7.88 → -11.53 | bf16 / 2 |
| Llama-2-7B (32L) | 12 | -1 | 100% | **-95.0pp** | -21.2pp | 97.5% | +8.92 → -6.91 | bf16 / 2 |

Bolded values moved by >10pp from the pre-fix numbers. Sweet spot: Qwen ~54-61% depth, Llama ~38% depth. Multi-position search matters — 5/7 models selected non-pos-1 positions.

**Reproducibility caveats.**
- *Batch size is part of the measurement.* Greedy decoding is not batch-shape invariant in bf16: batch shape changes reduction order, and argmax over near-tied logits is discontinuous. Measured on Qwen2.5-3B, bs=2 vs bs=8 moves a rate by up to 3pp. It is fixed *within* each run (all conditions share it) and recorded per row above; do not compare rows generated at different settings at single-point precision.
- *fp32 is exactly batch-invariant.* On Qwen2.5-0.5B, fp32 at bs=2 and bs=8 agree to every printed digit on all six conditions and both metrics. bf16 at bs=1 reproduces fp32 exactly on all six rates. Models ≤3B are therefore run in fp32; 7-8B stay in bf16 (fp32 would be 28-32GB of weights before activations) and carry the residual.
- *Layer/position not re-searched.* These are the coordinates chosen by the **pre-fix** search. The direction vector at each coordinate was recomputed post-fix, but the search itself used generation-based prompt filtering through the broken path — which discarded up to 42% of harmful training prompts on the affected Qwens. The selected coordinates may not be optimal. Re-running `--mode search` is outstanding.

## Key Findings

- **Refusal direction works universally**: 83.8-100% induction and 87.9-100pp global ablation across all 7 models tested (Qwen2.5 0.5B-7B, Llama-2-7B, Llama-3-8B, Llama-3.1-8B). The "single direction" hypothesis holds strongly, and holds *more* strongly than this file previously claimed — the padding artifact was suppressing the measured effect, not manufacturing it.
- **Layer-specific ablation is nearly as good as global, except on Llama-2**: six models lose 83.8-93.9pp from ablating at the source layer alone; Llama-2-7B loses only 21.2pp (vs 95.0pp globally) and its log-odds stays *positive* (+2.26) under single-layer ablation. Refusal is redundantly encoded across layers there. This survived the fix unchanged, so it is a property of the model, not an artifact.
- **The padding bugs suppressed the result**: every Qwen's ablation effect strengthened by 14-24pp after the fix, while Llama-3/3.1 moved ≤1pp. That split was *predicted* from the pad token's semantics before the re-runs (see below) and is the strongest evidence the diagnosis is right.
- **Arditi replication successful**: With correct padding, our pipeline selects L12/pos-5 on Llama-3-8B with Arditi's exact data (bypass=-10.7 vs paper's -9.7), strictly passing all criteria.
- **Multi-position search matters**: 5/7 models select non-pos-1 positions. The `max_positions=auto` setting (derived from `assistant_prefix_tokens`) correctly searches all post-instruction token positions.
- **Llama-2 template bug**: HuggingFace's Llama-2 chat template doesn't respond to `add_generation_prompt`. We override with `[INST] {x} [/INST] ` (trailing space). Without this, induction fails completely.
- **Our dataset produces strong directions**: Topic-matched pairs work well (90 harmful / 64 harmless train, 99/80 eval — not the "80+80" this file previously claimed). Arditi's 128+128 (from AdvBench + MaliciousInstruct + TDC2023) also works when scoring is correct.
- **Not yet replicated: the safety half of the paper.** Arditi pair the refusal score with a Llama Guard 2 *safety* score over JailbreakBench, and generate 512 tokens. Everything above is phrase-match refusal only, at 64 tokens. "Ablation removes refusal phrases" is established here; "ablation elicits unsafe completions" is not. The plumbing exists (`--arditi-evals`); no reported run has used it.
- **No confidence intervals anywhere.** At n=99 harmful / 80 harmless, a 3pp difference is ~3 prompts and is inside both sampling error and the bf16 decoding noise measured above. Differences of that size in the table should not be interpreted.

> **Everything below this line is pre-fix and unreliable.** All non-refusal findings are detection rates produced by the broken generation path (bug 2), on Qwen2.5 models — the family the bug hit hardest. None have been re-run. Treat them as hypotheses, not results.

- **Sycophancy is harder** *(pre-fix)*: Low induction rates (2-12%), no clean single direction on small models.
- **Hedging: negative result** *(pre-fix)*: 0% behavioral detection across all conditions on all 4 Qwen2.5 models. Note this "negative result" is exactly what a corrupted generation path also produces, so it needs re-running before it can be believed either way.
- **Empathy direction works** *(pre-fix)*: Layer subtraction drops empathy 80%→10% on 1.5B and 25%→0% on 3B. Layer addition induces empathy on 100% (1.5B) and 40% (3B) of neutral prompts.
- **Refusal–sycophancy entanglement** *(pre-fix)*: Ablating the sycophancy direction collapses refusal (−63pp on Qwen2.5-3B) despite near-orthogonality (cos 0.23). See Cross-Concept below.

## Cross-Concept (Qwen2.5-3B, refusal × sycophancy × hedging) — SUPERSEDED

> **Do not cite these numbers.** Every detection rate here came from the broken generation path, on Qwen2.5-3B — the model where that bug was worst. Its refusal baseline is quoted below as 66.7%; the corrected value is **96.0%**, a 29pp error, and every interference delta is measured against that wrong baseline. The cosine similarities and the SVD subspace analysis are unaffected (they operate on the saved direction vectors, not on generations), so those survive; the interference matrix and joint-ablation rows do not. Re-running `--mode cross_concept` is outstanding.

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

## Two Padding Bugs

There were two, in opposite directions, and the first was misdiagnosed for months.

### Bug 1: wrong index into left-padded batches (April 2026, scoring path)

**What this file used to say**: "left-padding corrupts logits because RoPE position encodings are wrong; affects ALL RoPE models on ALL devices." **That explanation is wrong.** RoPE attention depends only on *relative* positions, so a uniform shift of every real token is a no-op. Measured on Qwen2.5-0.5B: left-pad with no `position_ids`, reading the true last token, gives the same top-1 token as the unpadded run and a logit correlation of **0.99978** (the residual is bf16 noise).

**The actual bug** was indexing. Under left padding the last real token sits at index `seq_len-1`, but `scoring._get_logits` read `attention_mask.sum(-1)-1` and `activations` read `true_len + pos_idx` — both valid only under *right* padding. Under left padding they read a token from the middle of the prompt. Same measurement, reading that index: top-1 becomes `' Alibaba'` instead of `'2'`, max |Δlogit| **23.7**.

Switching to right padding fixed it by making those indices valid again; explicit `position_ids` was incidental (under right padding, `cumsum-1` equals the default `arange` on real tokens). The correct lesson is **never index a padded batch by true length without knowing the padding side** — not "always pass position_ids". Note `test_per_model.py::TestLeftPaddingRegression` encodes the old misdiagnosis: it indexes left-padded rows at `mask.sum()-1`, so it "confirms" the RoPE bug by exercising the indexing bug.

### Bug 2: generating from a pad slot (fixed 2026-08-11, evaluation path)

Fixing bug 1 introduced its mirror image. Evaluation generated via `model.generate()` on **right-padded** batches, and `generate()` reads next-token logits from the *last column* — a pad slot for every row shorter than the longest in its batch. `generate_with_hooks()` (correct: reads the last real token) existed but was only used by `scripts/`, never by the CLI.

Mechanism, measured at batch size 1 to exclude batch effects (Qwen2.5-0.5B):

- Two of three inputs at that slot are wrong: the token embedded there is `<|endoftext|>`, not the true final prompt token; and HF assigns every pad slot `position_id = 1` (`masked_fill(mask==0, 1)` — a sentinel designed for *left* padding, where pad outputs are discarded and never read).
- **Pad count is irrelevant**; the corruption is binary. All pads share the sentinel position and none are attendable keys, so the final pad's logits are provably independent of *k*: max |Δlogit| = **0.0000** for k ∈ {1,2,3,5,10}.
- The corrupted distribution is high-entropy: **5.06 nats vs 2.58** at the true slot, with the refusal-onset token `'I'` falling from p=0.327 (rank 1) to p=0.041 (rank 5), and the new top-1 a near-tie between generic openers. The first token becomes an arbitrary essay opener; greedy decoding then writes a fluent non-refusal from it.
- Only the first decoding step is causally corrupted — afterwards mask and positions are correct — so the rest is a faithful continuation of a wrong first token.

**Why the damage was model-dependent, and the prediction it made.** The pad token's *semantics* decide the outcome. Qwen pads with `<|endoftext|>` (a document boundary), so the model leaves chat mode and the refusal vanishes. Llama pads with `<|eot_id|>` (a turn boundary), so the model opens a fresh assistant turn and refuses again — all its padded rows' text is still wrong, but the phrase-match label survives. This predicted, before the re-runs, that the Qwen numbers would move a lot and the Llama numbers barely at all. Outcome: Qwens +14 to +24pp, Llama-3/3.1 ≤1pp.

**Third defect, same call site**: `model.generate()` applies the model's shipped `generation_config`, and Qwen2.5 ships `repetition_penalty` 1.05-1.1 — a *logits processor*, so it applies even with `do_sample=False`. Every CLI Qwen result was therefore decoded with a repetition penalty rather than the plain greedy decoding the paper specifies, and was not comparable with any `scripts/` result. With `repetition_penalty=1.0`, `model.generate()` and `generate_with_hooks()` agree bit-for-bit.

**Why it went unnoticed**: HF *does* warn ("right-padding was detected!"), but the check compares the last column against `generation_config.pad_token_id`. The code padded with `tokenizer.pad_token` (`<|endoftext|>`) while passing `pad_token_id=tokenizer.eos_token_id` (`<|im_end|>`) to `generate()` — different values on Qwen, so no warning. On Llama they coincide, and the warning is in the logs: ~296 occurrences per `repro-Llama-*-evaluate.log`, zero in the Qwen ones.

**Fix**: evaluation routes through `generate_with_hooks()`; it now also stops on any id in `generation_config.eos_token_id`. Generation batch size is a single `--gen-batch-size` shared by every condition and the filtering pass, logged in the report header. Every condition additionally reports `log_odds_metric`.

**Verification**: `test_generation.py::test_batched_generation_matches_individual` (batch-invariance) and `::test_generation_ignores_shipped_sampling_config` (pure greedy). `test_unit.py` guards that evaluation never calls `model.generate()`.
