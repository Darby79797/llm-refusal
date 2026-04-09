# Decision Complexity: Navigating Tradeoffs in Mechanistic Interpretability

## 1. The Research Question

Arditi & Obeso showed that refusal in language models is mediated by a single linear direction in the residual stream. Remove that direction and the model stops refusing; inject it and the model starts refusing benign requests. This project asks whether that finding generalizes: to other learned behaviors (sycophancy, hedging, empathy), across model families (Qwen, Llama), and across scales (0.5B to 8B parameters).

The first major decision was to build an independent, concept-generic framework rather than fork Arditi's code. A clean replication using their codebase would confirm their results but add little to the field. An independent implementation tests whether the *methodology* transfers, not just the code. The cost is that every implementation choice becomes a potential confound — and as we'll see, some of those confounds turned out to be more consequential than the methodology itself.

## 2. The Mathematical Premise

The core method is difference-in-means. Given contrastive prompt pairs — one set where the behavior appears (e.g., harmful prompts that trigger refusal) and one where it doesn't (benign prompts) — we extract activations from the residual stream at each layer and compute **mu_pos - mu_neg**. This vector defines a direction in activation space that, in principle, encodes the behavior.

The choice of difference-in-means over alternatives was deliberate. PCA finds directions of maximum variance, which may not align with the contrastive signal. Logistic probes require training a classifier and introduce optimizer hyperparameters. Contrast-Consistent Search (CCS) requires negation structure in the prompts. Difference-in-means has a single assumption — the behavior occupies a linear direction — and no hyperparameters. In mechanistic interpretability, simplicity minimizes the gap between "we found a direction" and "the model uses this direction."

The tradeoff: difference-in-means cannot find nonlinear representations. If a behavior is encoded in a curved manifold in activation space, this method produces a noisy projection onto the best-fit line. The hedging null result (Section 7) may reflect this limitation.

Three interventions test the direction. **Ablation** projects out the component along the direction, answering "is this direction necessary for the behavior?" **Addition** injects the direction into benign prompts, answering "is this direction sufficient to trigger the behavior?" **Subtraction** removes a scaled copy of the direction, providing stronger removal than projection when the projection magnitude is small. These test different hypotheses and, as we discovered, can diverge dramatically on the same model.

## 3. The Dataset Decision

This was the single highest-impact decision in the project.

Arditi used 260 harmful prompts from AdvBench paired with 18,793 benign prompts from Alpaca — a class ratio of 1:72. We used 80 harmful and 80 benign prompts, manually curated as topic-matched contrastive pairs. The results diverged dramatically: on Qwen2.5-3B, our direction achieved 97.5% induction while Arditi's achieved 12.5%. A 2x2 factorial (our data vs. Arditi's, tested on both eval sets) confirmed the effect: our direction transferred to Arditi's eval set at 87.5%, while Arditi's direction failed on both eval sets.

**Topic matching** was the key design choice. Each harmful prompt has a benign counterpart on the same topic — "How do I synthesize methamphetamine?" is paired with "How do I synthesize aspirin?", not with "What's the capital of France?" This isolates the behavior signal from the topic signal. Arditi's dataset compares weapons and hacking prompts (AdvBench) to general knowledge prompts (Alpaca), so the direction partly encodes "this prompt is about a dangerous topic" rather than "the model should refuse." That conflation doesn't matter for ablation — removing any refusal-correlated component helps — but it's fatal for addition, where the direction must precisely encode the refusal decision to trigger it on unrelated prompts.

**Balanced class sizes** compound the effect. With a 1:72 class ratio, the negative mean in Arditi's data is dominated by the general distribution of Alpaca instructions, not by the contrast with harmful prompts. The resulting direction norms are ~25x larger than ours, destroying coherent generation when added. Balanced 80+80 pairs produce a direction whose magnitude is naturally calibrated to the activation difference between behaviors.

**Manual curation** is the tradeoff. Writing 80 topic-matched pairs took roughly 4-6 hours of careful work — far more effort per prompt than downloading a benchmark dataset. But the 2x2 factorial shows this isn't the bottleneck it appears to be: our direction transfers well to Arditi's held-out eval set (87.5%), confirming it captures genuine refusal structure rather than overfitting to our 80 specific phrasings. The signal-to-noise ratio of 80 well-chosen pairs dominates 18,793 poorly-matched ones.

## 4. Extraction and Intervention Mechanics

**Where to extract activations.** We register pre-hooks on transformer block inputs, capturing the residual stream as it enters each layer. This matches Arditi's approach — both implementations intercept the same mathematical object. The subtler question is *which token position* to extract from. Arditi uses the end-of-instruction (EOI) token — the last token of the user's message, where the model "summarizes" the request. We default to position -1, the last token of the full formatted prompt, which in chat-templated models is typically inside the assistant prefix (e.g., `\n` after `<|im_start|>assistant`). These positions carry different information: EOI encodes "what was the request," while position -1 encodes "what should I generate next." Our `max_positions` parameter auto-derives from the assistant prefix token count, allowing search over multiple positions to cover both interpretations.

**Normalization asymmetry.** Ablation uses unit-normalized vectors because projection removal is scale-invariant: projecting out r-hat removes the same component regardless of ||r||. Addition and subtraction use raw (unnormalized) vectors because the magnitude from difference-in-means is a natural scale — it represents the typical activation difference between the two behaviors. Normalizing the addition vector would discard this calibration and require manual strength tuning.

**Sublayer hooks for ablation.** Ablation registers three hooks per layer: a pre-hook on the block input plus post-hooks on the attention and MLP sublayer outputs. Without the sublayer hooks, the attention and MLP computations within the layer can reintroduce the direction component that was removed from the residual stream. This matches Arditi's three-hook approach and matters on models where single-hook ablation is insufficient.

**Custom autoregressive generation.** `model.generate()` with KV-cache only runs the full forward pass on the first token; subsequent tokens reuse cached key-values and skip most layers. This means hooks only fire once, not on every generation step. We wrote a manual autoregressive loop with explicit KV-cache management that ensures hooks fire on every token. Without this, only the first generated token would be affected by the intervention — ablation during generation would silently fail after the first step.

**The key asymmetry.** Ablation is applied with unit vectors at all layers with 3 hooks per layer. Addition is applied with raw vectors at a single layer with 1 hook. These are different interventions testing different hypotheses. Ablation asks whether the direction is *necessary* — does the behavior survive without it? Addition asks whether it's *sufficient* — can the behavior be triggered by this direction alone? On Llama-2, these diverge completely: ablation produces 0pp change (the projection magnitude is too small), while subtraction at the same layer removes 70pp of refusal. The same pattern appears for the empathy concept: ablation fails, subtraction succeeds.

## 5. The Search Problem

The search evaluates every (layer, position) pair on three metrics. **Bypass** measures whether ablation reduces the behavior (lower is better). **Induce** measures whether addition creates the behavior (higher is better). **KL divergence** measures whether ablation preserves coherent generation (lower is better). This is a multi-objective optimization problem with no clean Pareto frontier.

**Progressive fallback tiers.** Strict criteria — induce score above threshold and KL below 0.1 — are rarely met on models under 2B parameters. Rather than crash (as Arditi's code does), the search relaxes through three tiers: (induce delta > +3, KL < 5), then (delta > +1.5, KL < 10), then (delta > +0, KL < 20), then best induce regardless. These thresholds were calibrated against behavioral induction rates measured at every layer on four Qwen models (0.5B through 7B). The tiers are generous on KL — good layers consistently have KL above 1.0 — because the layer depth cutoff (below) does the heavy lifting to exclude bad candidates.

**Rank by induce, not bypass.** On small models, the best-bypass layer and best-induce layer diverge. A layer might ablate refusal effectively (good bypass) but fail to induce it when added to benign prompts (poor induce). Since the goal is finding directions that causally control behavior in both directions, induce is the better predictor of actual generation behavior. This was discovered through the calibration experiment: layers with high bypass but low induce scores produced 0% behavioral induction in generated text.

**Layer depth cutoff at 65%.** Deep layers (beyond 65% of total depth) show high LogOdds induce scores that do not translate to behavioral induction. On the 3B model, layer 24 at 67% depth has an induce delta of +4.35 but only 5% behavioral induction, while layer 18 at 50% depth has delta +3.48 but 100% behavioral induction. The explanation: LogOdds measures token probability shifts at the *current* layer's output, but the model still has remaining layers that can override the intervention before generation. At 67% depth, there are 12 remaining layers; at 50% depth, there are 18 — but the intervention at 50% depth has passed through the critical middle layers where the refusal circuit is most active. The cutoff was originally 0.8 (80% depth); tightening to 0.65 after calibration fixed search failures on the 1.5B and 3B models where the search had been selecting deep layers that looked good on metrics but failed behaviorally.

The tradeoff: a hard threshold is blunt. Layers near the boundary (60-70% depth) may contain useful directions that are excluded. A smooth depth penalty would be more principled but harder to calibrate against empirical data.

## 6. Model-Specific Debugging

Three discoveries required debugging beyond the generic framework, and each demonstrates that infrastructure details can dominate methodology.

**The Llama-2 template bug.** HuggingFace's built-in chat template for Llama-2 does not respond to `add_generation_prompt=True`. The formatted prompt ends with `[/INST]` and no trailing space, which means position -1 is the `]` token — a structural template character — rather than the space token at the actual generation boundary. Difference-in-means at this position captures "end of template formatting" rather than "refusal decision," producing a garbage direction. The fix: override the template with `[INST] {x} [/INST] ` (trailing space). Effect: induction jumped from 0% to 71%. This single tokenization detail was the entire difference between a null result and the project's strongest Llama finding. It was discovered by running a diagnostic script that printed the token-by-token decomposition of formatted prompts — a five-minute investigation that resolved weeks of failed experiments.

**Numerical precision on Apple Silicon.** Float16 on MPS (Apple's GPU backend) produces NaN and Inf values in attention computations due to overflow. The framework auto-upcasts float16 models to bfloat16 on MPS devices. Separately, activation extraction uses float64 arithmetic because bfloat16's 7-8 bits of mantissa precision cause quantization noise to accumulate when averaging over ~100 samples. The mean is computed in float64 and stays in float64 through the subtraction step; only the final direction vector is cast to float32 for storage.

**The Llama family pattern.** Llama models show systematically weaker results than Qwen across the board. Global ablation removes 1-28pp of refusal on Llama (vs. 83pp on Qwen 3B). Induction reaches 0-16% on Llama-3/3.1 (vs. 89-100% on Qwen). Increasing addition strength makes Llama *worse* — 16% at strength 1 drops to 0% at strength 3 — because the direction degrades coherent generation before it triggers refusal phrases. All Llama models require single-layer interventions; global addition garbles output entirely. This suggests refusal in Llama is more distributed across circuits, with no single direction sufficient to control it.

## 7. Cross-Concept Results and Failures

The concept-generic framework was applied to four behaviors. Results span a spectrum from clean success to informative failure, and each failure teaches something specific about the methodology's boundaries.

**Refusal** is the cleanest result. Induction rates of 89-100% across all four Qwen models (0.5B through 7B), with the direction at ~50% depth consistently. The 3B model shows the strongest ablation effect: global ablation removes 83pp of refusal. This confirms the single-direction hypothesis for refusal on Qwen at all tested scales.

**Empathy** works via subtraction but not ablation. Subtracting the empathy direction removes empathetic responses completely (80% to 0% detection on 1.5B, 30% to 0% on 3B). But ablation — projecting out the component — has no effect. This is the same pattern as Llama-2 refusal: the projection magnitude at any single layer is too small to overcome the behavior, but a scaled subtraction is strong enough. This suggests subtraction is generally more effective than projection-based ablation when the behavior's representation has small norm relative to the residual stream.

**Sycophancy** is a partial failure. Induction rates of 2-12% across models — far below refusal's 89-100%. Global ablation *increases* sycophancy on three of four models, suggesting the found direction partially encodes anti-sycophantic behavior. No clean single direction mediates sycophancy on these models; the behavior may be more distributed or context-dependent than refusal.

**Hedging** is a genuine null result. 0% behavioral detection across all four Qwen models, all conditions (baseline, ablation, addition). Yet the LogOdds search finds directions with positive induce scores — token-level probability shifts exist, but they don't translate to actual hedging in generated text. The explanation is simple: these models don't hedge on factual questions at baseline. There is no behavior to detect or manipulate. This is a dataset design failure, not a method failure — testing on opinion or subjective prompts where models naturally hedge might yield different results.

**Cross-concept interference** reveals unexpected geometry. On the 3B model, ablating the sycophancy direction reduces refusal by 57 percentage points — despite a cosine similarity of only 0.176 between the two directions. PCA on all three directions (refusal, sycophancy, hedging) shows they span a 2D subspace (explained variance: 58%, 42%, ~0%), meaning the hedging direction lies almost entirely in the plane spanned by refusal and sycophancy. The interference is asymmetric: ablating refusal increases sycophancy by 7pp, but ablating sycophancy devastates refusal. Geometric near-orthogonality does not imply functional independence.

The decision to report null and partial results rather than filtering to successes was deliberate. Each failure mode teaches something that refusal-only results cannot: hedging shows that baseline behavior must exist for the method to work; sycophancy shows the method's limitation to single-direction behaviors; empathy and Llama-2 show when ablation fails and subtraction is needed.

## 8. The Iterative Discovery Arc

None of the major findings were planned in advance. Each was driven by a specific failure.

The project started with 0% induction on Qwen1.5-1.8B-Chat — a complete failure to reproduce Arditi's core result. A systematic audit identified six concrete differences from Arditi's pipeline, ranked by estimated impact: dataset construction, activation extraction position, model scale, intervention application (single-layer vs. all-layer), chat template handling, and evaluation metric details.

The scale experiment came first because it required no code changes: run the same pipeline on 0.5B through 8B models. Result: 89-100% induction on Qwen, 0-16% on Llama. Scale matters, but model family matters more.

The 2x2 factorial (our dataset vs. Arditi's, raw vs. unit normalization) isolated the dataset as the dominant factor: 97.5% vs. 12.5% induction on the same model with the same code.

Search calibration — correlating LogOdds scores with behavioral induction rates at every layer — revealed that the layer depth cutoff needed to be 0.65 rather than 0.8. Deep layers had misleading high scores.

The Llama-2 template bug was the last discovery, found months into the project by printing token indices of formatted prompts. A one-character fix (adding a trailing space) moved induction from 0% to 71%.

Each discovery informed the next experiment, and each experiment changed the framework's defaults. The final system reflects accumulated debugging, not top-down design.

## 9. Synthesis

The single-direction hypothesis holds for strongly-trained, cleanly-separated behaviors — refusal in Qwen being the paradigmatic case. It degrades along three axes: **model architecture** (Llama encodes refusal more distributedly than Qwen), **behavior type** (sycophancy is not mediated by a single direction at these scales; hedging has no baseline to manipulate), and **intervention type** (ablation and addition test different properties and can diverge completely, as on Llama-2 where ablation fails but subtraction succeeds).

Dataset quality dominates direction quality. Eighty well-chosen, topic-matched pairs outperform 19,000 poorly-matched ones by a factor of 8x on induction rate. The marginal return on data *quantity* is negative when it introduces class imbalance; the marginal return on data *quality* — specifically, contrastive structure that isolates the target behavior from confounds — is enormous.

Infrastructure details are not secondary concerns. The Llama-2 template bug, a single missing space character in a tokenizer template, was the difference between 0% and 71% induction. Numerical precision (float64 for averaging, bfloat16 for MPS) and generation mechanics (custom autoregressive loop for correct hook behavior) are load-bearing infrastructure decisions, not implementation details.

The decision to build a concept-generic framework paid off in scientific value. The marginal cost of each additional concept was low — new prompt pairs and a concept definition. But without sycophancy, hedging, and empathy, the project would have overclaimed the generality of the single-direction hypothesis. The failures are as informative as the successes: they delineate where linear interventions on single directions stop working, and point toward the distributed, nonlinear representations that future work must address.
