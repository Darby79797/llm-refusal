# Arditi et al. vs Our Implementation: Detailed Comparison

A structured comparison between [Arditi & Obeso's `refusal_direction` repo](https://github.com/andyrdt/refusal_direction) and our implementation, focused on identifying which differences might explain experimental results (e.g., ablation works perfectly but addition fails to induce refusal).

## Summary of Differences

| Component | Arditi et al. | Ours | Likely Impact |
|---|---|---|---|
| Token positions | All EOI positions | Last token only (`-1`) | **High** |
| Data filtering | Filter to correctly-handled examples | No filtering | **Medium-High** |
| Ablation hook points | 3 per layer (block input, attn out, MLP out) | Block input only | Low (95%→0% anyway) |
| Activation averaging | Incremental mean | Collect-then-average | None |
| Direction dtype | float64 throughout | float64 compute, float32 storage | Negligible |
| Train/val split | Pre-split files (128/32) | Random 80/20, seed=39 | Low |
| Search fallback | Crashes if no strict candidates | Lenient fallback formula | Low (UX only) |
| Final evaluation | LlamaGuard2 + JailbreakBench + CE loss | Phrase match / API judge + lm-eval | Different metrics, not comparable |
| Weight editing | Permanent orthogonalization | Not implemented | N/A (different feature) |
| Generation | `model.generate()` | Custom `generate_with_hooks()` | Different tradeoff (see below) |

## Detailed Analysis

### 1. Activation Extraction

**Arditi**: Registers `pre` hooks on each transformer block's forward pass, extracting the block input (i.e., the residual stream entering that layer). Uses incremental mean computation — updates a running mean as each example is processed, never storing all activations in memory.

**Ours** (`activations.py`): Also uses pre-hooks on block inputs (same interception point). Collects all activations into a list, then averages at the end. Casts to float64 before averaging for numerical stability.

**Impact**: None. Both extract from the same point. Incremental vs batch mean is mathematically identical (barring floating point ordering effects, which are negligible at float64).

### 2. Token Positions

**Arditi**: Extracts activations at **all positions corresponding to end-of-instruction tokens**. For chat-templated models, this means the last token of the user message (before the assistant turn begins). They identify these positions by tokenizing just the instruction, finding its length, and extracting at that position for each example. Since instructions vary in length, each example contributes activations at different absolute positions.

**Ours**: Extracts at position `-1` only (the very last token in the full formatted prompt). For chat-templated models, this is typically the last token of the assistant turn prefix (e.g., the newline or `<|im_start|>assistant` token), not the last token of the user's instruction.

**Impact**: **Potentially high**. The residual stream at the end-of-instruction position carries the model's "summary" of what the user asked. The last token of the assistant prefix may carry different information (more about "what comes next in generation" than "what was the request"). This could affect direction quality, especially for addition/steering where you need the direction to cleanly encode "refuse this" vs "comply with this". Our `max_positions` config supports multiple positions, but defaults to 1.

### 3. Direction Computation

**Arditi**: Computes `mean_positive - mean_negative` in float64. Stores the result in float64.

**Ours** (`direction_methods.py`): Same subtraction in float64. Casts result to float32 for storage in `DirectionVector`.

**Impact**: Negligible. The float32 cast loses ~7 decimal digits of precision, but direction vectors are used for projection (ablation) or addition where this precision loss is immaterial relative to the scale of activations.

### 4. Data Filtering

**Arditi**: Before computing the direction, filters training data:
- **Harmful prompts**: Only keeps examples where the model actually refuses (verified by generating a response and checking with a classifier). Removes harmful prompts that the model already complies with.
- **Harmless prompts**: Only keeps examples where the model actually complies. Removes harmless prompts that the model incorrectly refuses.

This ensures the direction captures "what's different when the model refuses vs complies" rather than being diluted by misclassified examples.

**Ours**: No filtering. All harmful prompts are assumed to produce refusal, all harmless prompts are assumed to produce compliance. Any mislabeled examples (harmful prompts the model already answers, harmless prompts the model refuses) add noise to the direction estimate.

**Impact**: **Medium-High**. For well-aligned models (Llama-2-7b-chat, which Arditi tested on), most harmful prompts do trigger refusal and most harmless prompts don't, so filtering removes only a small fraction. But for smaller or less well-aligned models (Qwen1.5-1.8B-Chat), the mismatch rate could be higher, degrading direction quality. This is especially relevant for the addition/steering case: if the direction is noisy, adding it may not cleanly induce refusal.

### 5. Ablation Hook Points

**Arditi**: For ablation, hooks **three points per layer**:
1. Block input (residual stream entering the layer)
2. Attention sublayer output (before it's added back to the residual stream)
3. MLP sublayer output (before it's added back to the residual stream)

At each hook point, they project out the direction component. This is more thorough — it removes the direction not just from the residual stream but also from each sublayer's contribution, preventing the direction from being "re-injected" by attention or MLP computations within the layer.

**Ours** (`interventions.py`): Hooks only the block input (residual stream). The direction component is removed from the residual stream as it enters each layer, but sublayer outputs within each layer can still inject direction-aligned components.

**Impact**: **Surprisingly low in practice**. Our ablation achieved 95%→0% refusal detection on Qwen1.5-1.8B-Chat despite only hooking block inputs. This suggests that for refusal, the direction component in the residual stream is the primary carrier, and sublayer re-injection is minimal. However, this difference might matter more for:
- Larger models with more complex internal representations
- Other concepts (sycophancy, hedging) where the direction is less clean
- Partial-layer interventions where you only ablate at specific layers

### 6. Addition / Steering

**Arditi**: Adds the **raw (unnormalized)** direction vector scaled by a coefficient (default 1.0) at a **single layer**. Applied via a hook on the block input at the chosen layer.

**Ours**: Same approach. Raw vector, strength parameter (default 1.0), single layer.

**Impact**: Same behavior. Both rely on the raw vector's magnitude being "naturally" scaled by the difference-in-means computation.

### 7. Direction Selection (Search)

**Arditi**: Evaluates directions across layers using the same three metrics (bypass rate, induce rate, KL divergence). Selects using strict thresholds. If no candidate passes all thresholds, the code raises an error / crashes.

**Ours** (`search.py`): Same three-metric evaluation. If strict criteria aren't met (common on models ≤1.8B), falls back to a lenient formula: minimize `(10 * bypass) + kl - induce`. This prevents crashes but may select a suboptimal direction.

**Impact**: Low for the direction quality itself (both would pick similar candidates if strict criteria are met). The fallback is a UX improvement, not a methodological difference. On larger models where strict criteria are met, results should be identical.

### 8. Normalization

**Arditi**: Uses unit vector for ablation (projection removal is scale-invariant, so this is just for cleanliness). Uses raw vector for addition.

**Ours**: Same convention. Unit vector for ablation, raw for addition.

**Impact**: None. Mathematically identical.

### 9. Train/Val Split

**Arditi**: Uses pre-split files: 128 harmful + 128 harmless for training, 32 + 32 for validation. The split is fixed and reproducible.

**Ours**: Loads all prompts, then does a random 80/20 split with `random_state=39`. Different total counts depending on the prompt dataset.

**Impact**: Low. The split methodology doesn't fundamentally change the direction — both approaches give enough examples for a stable mean estimate. The exact prompts differ, which introduces some variance, but the directions should be qualitatively similar.

### 10. Evaluation

**Arditi**:
- **Safety**: LlamaGuard2 classifier + JailbreakBench artifact for harmful prompt detection
- **Capability**: Cross-entropy loss and perplexity on Alpaca dataset and The Pile
- **Qualitative**: Manual inspection of generations

**Ours**:
- **Safety**: Three-tier detection (API judge → heuristic → phrase matching)
- **Capability**: lm-eval benchmarks (MMLU, ARC, GSM8K, TruthfulQA)
- **Qualitative**: Side-by-side generation comparison (eyeball mode)

**Impact**: Not directly comparable. Our phrase-matching baseline may have different sensitivity/specificity than LlamaGuard2. For reporting purposes, results aren't directly comparable across implementations, but the relative effects (before/after intervention) should tell the same story.

### 11. Weight Editing (Orthogonalization)

**Arditi**: Implements permanent weight orthogonalization — modifies the model's weight matrices to remove the direction component, making the intervention persistent without hooks.

**Ours**: Not implemented. All interventions are hook-based (temporary, removed when hooks are cleared).

**Impact**: None for research purposes. Hook-based ablation is mathematically equivalent to the weight edit: the hook computes `x' = (I - r̂r̂ᵀ)x`, which is the same linear projection that weight orthogonalization bakes into `W' = (I - r̂r̂ᵀ)W`. For addition, the hook adds a constant vector (a bias shift, not a rank-1 weight update), which is equally trivial to make permanent. Weight editing is a deployment technique — useful for shipping a modified model, not needed during experimentation.

### 12. Generation

**Arditi**: Uses `model.generate()` directly. This works because their hooks are registered globally and `model.generate()` calls forward passes that trigger the hooks.

**Ours**: Uses custom `generate_with_hooks()` (`generation.py`) that does manual autoregressive generation with KV-cache management. This was necessary because `model.generate()` with KV-cache skips the full forward pass on cached tokens, meaning hooks only fire on the first pass and not on subsequent tokens.

**Impact**: Different tradeoff. Arditi's approach is simpler but may have subtle issues with KV-cache and hooks (though for their use case — hooking all layers at block input — it may work fine since the first pass establishes the cache with the intervention applied). Our approach is more correct for interventions that need to apply on every generation step, at the cost of complexity and the known batched-generation padding issue.

## Differences Most Likely to Explain "Ablation Works, Addition Fails"

Ranked by likely impact:

1. **Token positions** (Section 2): If we're extracting at the wrong position (assistant prefix token vs end-of-instruction token), the direction may capture "what happens at this structural position" rather than "the model's decision to refuse/comply". Ablation is more forgiving because removing any refuse-correlated component helps, but addition requires the direction to precisely encode the refusal behavior to induce it.

2. **Data filtering** (Section 4): Unfiltered data adds noise to the direction. A noisy direction can still be "good enough" for ablation (removing an approximate direction still reduces refusal) but insufficient for addition (adding noise doesn't cleanly trigger refusal).

3. **Ablation hook points** (Section 5): Less likely to be the cause since our ablation already achieves near-perfect results. But worth noting as a methodological difference.

## Recommended Experiments

To isolate which difference matters:

1. **Token position experiment**: Run search with EOI token positions instead of `-1`. Compare direction quality metrics and addition effectiveness.
2. **Data filtering experiment**: Add a filtering step (generate on training prompts, keep only correctly-handled ones). Re-run search and compare.
3. **Both combined**: Apply both fixes simultaneously to see if they compose.
