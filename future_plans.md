# Future Plans: Improving Refusal Direction Induction

## Part 0: Current State

Our refusal direction search on **Qwen1.5-1.8B-Chat** found a direction at layer 12, pos -1 via fallback (strict criteria never jointly satisfied). The direction works well for **bypass** (ablation eliminates refusal tokens) but completely fails for **induction** (adding the direction to benign prompts does not make the model refuse).

**Key numbers from the 1.8B search:**
| Metric | Value |
|--------|-------|
| Best layer | 12 |
| Best pos | -1 |
| Bypass score | -10.90 |
| Induce score | -11.53 |
| KL divergence | 1.11 |
| Baseline pos log-odds | -9.32 |
| Baseline neg log-odds | -11.54 |
| Direction norm | 9.00 |

The induce score (-11.53) is essentially identical to the baseline negative log-odds (-11.54), meaning adding the direction to benign prompts has zero measurable effect on refusal token probability.

The probe script (`scripts/probe_layer_directions.py`) confirmed: even at strength=50, induce barely moved from baseline across layers 12, 13, and 15.

**This is unexpected.** Arditi et al. achieved positive induce scores on all 13 models they tested (1B to 70B). Something about our setup differs from theirs.

## Part 1: Differences Between Our Pipeline and Arditi et al.

We identified 6 concrete differences between our implementation and Arditi et al.'s setup. These are ordered by estimated impact on induction.

### Difference 1: Dataset Construction (HIGH impact)

**Arditi:** Used AdvBench harmful prompts paired with Alpaca benign instructions. Both are established benchmarks with hundreds of examples, carefully curated. AdvBench prompts are specifically designed to trigger strong refusal responses.

**Us:** Hand-written prompts (~100 positive, ~60 negative for training). While covering diverse categories, they may not trigger refusal as strongly or consistently as AdvBench. The contrastive signal may be weaker.

**Why it matters for induction:** If the positive prompts don't elicit maximally distinct refusal activations, the difference-in-means vector captures a noisier, weaker signal. Bypass works because ablation is a blunt instrument (remove any component correlated with refusal), but addition requires the vector to actually point in the "more refusal" direction with precision.

### Difference 2: Activation Extraction Position (HIGH impact)

**Arditi:** Extracted activations at the **last token position only** (the token where the model makes its "refuse or comply" decision).

**Us:** Search over last 5 positions (`max_positions=5`). While this is more thorough, the best direction for bypass (which dominates our multi-objective search via the lenient formula) may not be the best for induction. The information geometry may be different: the "refusal decision" is concentrated at the last token, but bypass-useful information might be spread across positions.

**Why it matters for induction:** Adding a direction that's optimal for bypass at pos=-2 or pos=-3 may not correspond to the induction-optimal direction at pos=-1 where the model actually decides what token to generate next.

### Difference 3: Model Scale (MEDIUM impact)

**Arditi:** Tested on models from 1B to 70B, with cleanest results on 7B+. Their smallest model (Qwen-1.8B-Chat, same as ours) likely showed the weakest induction too — but they still reported positive scores.

**Us:** Only tested on Qwen1.5-1.8B-Chat (1.8B params, 24 layers, hidden_dim=2048).

**Why it matters:** Smaller models may encode refusal in a more distributed, less linear way. The "single direction" hypothesis may hold more cleanly at scale, where the model has capacity to dedicate a clean subspace to refusal rather than overloading directions with multiple functions (polysemanticity).

### Difference 4: Intervention Application (MEDIUM impact)

**Arditi:** For induction, added the direction at **all layers** simultaneously (global addition).

**Us:** For induction, add at **the source layer only** (`_compute_induce_score` uses `[direction.layer]`). This is a design choice — we wanted to test whether the direction is sufficient at its source layer alone — but it makes induction harder because the model has subsequent layers to "recover" from the perturbation.

**Why it matters for induction:** Adding at one layer gives the model N-L remaining layers to counteract the added signal. Global addition is a stronger intervention that's harder for the model to route around.

### Difference 5: Chat Template / Tokenization (LOW-MEDIUM impact)

**Arditi:** Used specific tokenization settings for each model family. The exact details aren't fully documented but they likely used the standard HuggingFace tokenizer chat templates.

**Us:** `ChatPromptFormatter` applies chat templates with left-padding. We auto-detect instruction-tuned models and apply templates accordingly. This should be equivalent, but subtle differences in how the prompt is formatted (system message, BOS token, padding) could shift activations.

**Why it matters:** If the chat template doesn't perfectly match what the model expects, the model may be slightly "confused" and not activate its refusal circuits as cleanly, producing noisier directions.

### Difference 6: Evaluation Metric Details (LOW impact)

**Arditi:** Used a set of refusal-indicating tokens for log-odds, but the exact token set may differ from ours.

**Us:** `DEFAULT_REFUSAL_TOKENS = ["I", "I'm", "As", "I cannot", ...]` including space-prefixed variants. Multi-token entries like "I cannot" get filtered to single-token IDs during `LogOddsMetric.__init__()`.

**Why it matters:** The metric measures whether refusal tokens become more likely, but if the token set doesn't match what the model actually uses to refuse, the metric may be insensitive. This is more of a measurement issue than a real induction failure.

## Part 2: Scale Experiment

Before fixing the above differences (which requires code changes), we want to test whether **scaling alone** improves induction. This is the cheapest experiment: run the exact same pipeline on larger models.

**Rationale:** If the 1.8B model's induction failure is primarily due to model scale (Difference 3), then 3B and 7B models should show improvement without any pipeline changes. If induction still fails at 7B, then we know the pipeline differences (1, 2, 4) are the real culprits and must be addressed.

**Models to test:**
| Model | Layers | Hidden | Params | Memory (bf16) |
|-------|--------|--------|--------|---------------|
| Qwen2.5-3B-Instruct | 36 | 2048 | 3B | ~6 GB |
| Qwen2.5-7B-Instruct | 28 | 3584 | 7B | ~14 GB |

**Implementation:** `scripts/scale_experiment.py` — autonomous script, zero user input required. See that file for details.

## Part 3: Pipeline Fixes (If Scale Alone Doesn't Work)

Ordered by expected impact and implementation effort:

### Fix A: Global Addition for Induction
**Change:** In `scoring.py:_compute_induce_score()`, change `[direction.layer]` to `list(range(num_layers))`.
**Effort:** One line change.
**Risk:** May inflate KL divergence since global addition is a stronger perturbation.

### Fix B: Use AdvBench + Alpaca Datasets
**Change:** In `prompts.py`, add `create_refusal_train_data_advbench()` that loads AdvBench harmful behaviors and Alpaca instructions.
**Effort:** Medium — need to download/parse datasets, handle any formatting differences.
**Risk:** Low. These are the gold standard datasets for this task.

### Fix C: Restrict to Last Token Only
**Change:** Set `max_positions=1` in the search call, or add a config option.
**Effort:** Trivial.
**Risk:** May miss bypass-optimal directions at other positions, but should improve induction alignment.

### Fix D: Combined — Match Arditi Exactly
Apply A + B + C simultaneously to replicate Arditi's setup as closely as possible. If induction still fails after this, the issue is likely model-architecture-specific (Qwen vs Llama) or there's a subtle bug we haven't identified.

## Part 4: Novel Approaches (Beyond Replication)

### Idea 1: Separate Bypass and Induction Directions
The current pipeline uses a single direction for both bypass and induction. But these might be geometrically different. We could:
1. Find the direction that maximizes bypass (ablation eliminates refusal)
2. Separately find the direction that maximizes induction (addition creates refusal)
3. Compare their cosine similarity — if low, they're different mechanisms

**Implementation:** Add `mode="bypass_only"` and `mode="induce_only"` to the search, each optimizing for one score.

### Idea 2: Contrastive Activation Addition (CAA) vs Difference-in-Means
Difference-in-means is the simplest direction-finding method. Alternatives:
- **PCA on the difference:** Take the first principal component of (positive - negative) activations instead of the mean difference
- **Logistic probe:** Train a linear classifier on activations, use the weight vector as the direction
- **CCS (Contrast-Consistent Search):** Unsupervised method from Burns et al. that finds directions consistent across negations

These could find a direction that's better aligned with the causal mechanism rather than just the statistical mean.

### Idea 3: Layer-Specific Analysis
Instead of searching all (layer, pos) pairs with a single objective, analyze each layer independently:
- Which layers show the biggest positive-negative separation? (t-SNE / PCA visualization)
- Which layers' directions are most causally active? (causal tracing / activation patching)
- Is there a "refusal layer" where ablation has maximum effect?

### Idea 4: Strength Calibration
Our current pipeline uses strength=1.0 for both bypass and induction. But the optimal strength may differ:
- Bypass may need strength=1.0 (complete removal of the component)
- Induction may need strength >> 1.0 to overcome the model's "safe response" default on benign prompts

The probe script tested this (strengths 1-50) and found minimal effect, but on a larger model the strength-response curve might be different.

### Idea 5: Residual Stream vs Attention Output
We extract activations from the residual stream (full hidden state after each layer). Alternatively:
- Extract from attention output only (before MLP)
- Extract from MLP output only (before residual addition)
- This could isolate whether refusal is primarily an attention phenomenon or MLP phenomenon

## Part 5: Research Questions

1. **Does induction improve with scale?** → Scale experiment (Part 2)
2. **Is our induction failure due to single-layer addition?** → Fix A
3. **Is our induction failure due to dataset quality?** → Fix B
4. **Are bypass and induction geometrically aligned?** → Idea 1
5. **Is difference-in-means the right method?** → Idea 2
6. **Is there a clean "refusal layer" in Qwen architectures?** → Idea 3
7. **Does the three-way cross-concept analysis reveal a shared "alignment" subspace?** → Pending hedging search + cross-concept mode
