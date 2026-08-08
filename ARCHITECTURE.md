# Architecture

All code under `llm-refusal/`. Flat imports (`from formatting import ...`), set by `pythonpath = ["llm-refusal"]` in pyproject.toml. One-off experiment scripts in `scripts/`; tests in `tests/` (`test_unit`, `test_generation`, `test_per_model`, `test_smoke`).

| Module | Purpose |
|--------|---------|
| `datatypes.py` | `PromptData` (stratified `train_val_split`), `DirectionScores` (bypass/induce/induce_global/KL), `DirectionVector` (`.save()`/`.load()`) |
| `concept.py` | `ConceptDefinition` dataclass, registry (`register_concept`/`get_concept`), `DEFAULT_SEARCH_CONFIG`, all concept definitions |
| `prompts.py` | Train + eval prompt datasets per concept, returning `(positive, negative)` or pre-split `((train_pos, train_neg), (val_pos, val_neg))` |
| `formatting.py` | `ChatPromptFormatter` — chat templates, tokenization, right-padding, `position_ids`, `assistant_prefix_tokens` |
| `activations.py` | `ActivationExtractor` — pre-hooks on block inputs (residual stream entering each layer), float64 for stability |
| `direction_methods.py` | `DirectionMethod` ABC, `DifferenceInMeans` (mean_pos - mean_neg in float64) |
| `interventions.py` | `ModelInterventionApplier` — pre-hooks for add/subtract/ablate + sublayer post-hooks for ablation (3 hooks/layer, matches Arditi) |
| `scoring.py` | `LogOddsMetric`, `Three_Score_Evaluator` (bypass/induce/induce_global/KL) |
| `search.py` | `DirectionFinder` — multi-objective search with progressive fallback tiers, supports `induce_mode` |
| `evaluation.py` | `BigEvaluator` (detection rates, LlamaGuard2, JailbreakBench, Alpaca CE, lm-eval), `InterventionSuite` (eyeball) |
| `generation.py` | `generate_with_hooks()` — autoregressive gen with KV-cache and explicit `position_ids` (hooks fire per-step) |
| `cross_concept.py` | Cosine similarity, PCA, interference matrix, multi-ablation composition |
| `attribution.py` | Circuit analysis: per-head/MLP projection onto the direction, contrastive (harmful−benign) attribution. Used by `scripts/`, not the CLI |
| `framework.py` | `DirectionTestFramework` — orchestrator: model loading, mode dispatch, prompt filtering, pre-split val support |
| `run_experiment.py` | CLI (argparse or `--json`) |

## Adding a Concept

Train/eval data fns in `prompts.py` (topic-matched contrastive pairs, train/eval disjoint) → tokens/phrases/search config + optional `detect_<concept>()` heuristic in `concept.py` → `register_concept()` → detection tests in `test_unit.py`.

## Adding a Model Architecture

Branch in `_get_transformer_layers()`/`_get_sublayers()` in `interventions.py`; manual chat template fallback in `formatting.py` if needed; then run `test_per_model.py` on it (add to `PER_MODEL_TEST_MODELS`).

## Implementation Notes

- **Vector normalization**: Addition and subtraction use raw (unnormalized) vectors, matching Arditi; ablation uses unit vectors (projection removal is scale-invariant). All interventions are pre-hooks on block inputs; ablation additionally post-hooks `self_attn` and `mlp` to prevent re-injection (3 hooks/layer).
- **`generate_with_hooks()`** exists because `model.generate()` doesn't invoke forward hooks every step.
- **NaN in scoring**: `LogOddsMetric` returns nan for inf/nan logits; downstream uses `nanmean`. Typically only layer-0 candidates (near-zero norms). Not an error.
- **Search fallback**: Strict criteria rarely met on models <=1.8B. Progressive tiers: (Δinduce>+3, KL<5) → (Δinduce>+1.5, KL<10) → (Δinduce>+0, KL<20) → best induce. With `induce_mode=all_layers`, tiers use `induce_global`.
- **Prompt filtering on by default** but has zero measured effect (3-21% mislabeled prompts don't contaminate difference-in-means). `--no-filter-prompts` disables.
- **Pre-split val data**: `train_data_fn` may return `((train_pos, train_neg), (val_pos, val_neg))`; the framework then skips `train_val_split()` and uses all train data for the direction. Used by `refusal_arditi_exact`.
- **External API judge**: `JUDGE_API_BASE`, `JUDGE_API_KEY`, `JUDGE_MODEL` env vars (or `--judge-*` flags). `--arditi-evals` enables LlamaGuard2 + JailbreakBench + Alpaca CE; `--eval-tasks` runs lm-eval benchmarks.
