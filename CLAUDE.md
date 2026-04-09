# CLAUDE.md

Mechanistic interpretability research: finding directions in the residual stream that mediate learned behaviors (refusal, sycophancy, hedging). Extends Arditi & Obeso's "Refusal in Language Models is Mediated by a Single Direction". Concept-generic: register a `ConceptDefinition` to study new behaviors without touching core code.

## Commands

```bash
pip install -e .                          # setuptools, Python >=3.11
pytest llm-refusal/tests/                 # unit tests (excludes smoke)
pytest --run-smoke llm-refusal/tests/     # smoke tests (gemma-3-270m)
pytest llm-refusal/tests/test_per_model.py -v  # per-model correctness (padding, template, positions)

# Main CLI
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode search
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode evaluate --layer 21 --pos -4
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode eyeball --layer 21 --pos -4
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode cross_concept --concepts refusal,sycophancy
python llm-refusal/run_experiment.py --json '{"model_name": "...", "mode": "search"}'
```

## Key Config Options

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | required | HuggingFace model ID |
| `--mode` | required | `search`, `evaluate`, `eyeball`, `cross_concept` |
| `--concept` | `refusal` | Registry key: `refusal`, `refusal_arditi`, `refusal_arditi_exact`, `sycophancy`, `sycophancy_neutral`, `hedging`, `empathy` |
| `--layer`, `--pos` | — | Required for evaluate/eyeball |
| `--no-filter-prompts` | on by default | Disable filtering train prompts by actual model behavior |
| `--induce-mode` | `single_layer` | Search induce mode: `single_layer` or `all_layers` (Arditi-style) |
| `--force-cpu` | off | Override device (default: CUDA > MPS > CPU) |
| `--torch-dtype` | `auto` | Torch dtype for `from_pretrained` |
| `--concepts` | — | For cross_concept: comma-separated list |
| `--eval-tasks` | — | lm-eval tasks: `mmlu`, `arc_challenge`, `gsm8k`, `truthfulqa` |
| `--limit` | `100` | Sample limit for lm-eval benchmarks |
| `--judge-*` | env vars | External API judge: `JUDGE_API_BASE`, `JUDGE_API_KEY`, `JUDGE_MODEL` |
| `--arditi-evals` | off | Enable LlamaGuard2 + JailbreakBench + Alpaca CE loss |
| `--alpaca-max-prompts` | `500` | Max prompts for Alpaca CE loss |
| `--json` | — | JSON config string or file path (overrides all flags) |

Hardcoded: `max_positions=auto` (derived from assistant prefix tokens), val split 20% (`random_state=39`), `batch_size=2`, `max_new_tokens=64`, `layer_cutoff_frac=0.65`.

## Architecture (one-line per module)

All under `llm-refusal/`. Flat imports (`from formatting import ...`), set by `pythonpath = ["llm-refusal"]` in pyproject.toml.

| Module | Purpose |
|--------|---------|
| `datatypes.py` | `PromptData` (with stratified `train_val_split`), `DirectionScores` (bypass/induce/induce_global/KL), `DirectionVector` (with `.save()`/`.load()`) |
| `concept.py` | `ConceptDefinition` dataclass, registry (`register_concept`/`get_concept`), `DEFAULT_SEARCH_CONFIG`, all concept definitions |
| `prompts.py` | Train + eval prompt datasets per concept, all return `(positive, negative)` or `((train_pos, train_neg), (val_pos, val_neg))` for pre-split data |
| `formatting.py` | `ChatPromptFormatter` — chat templates, tokenization, right-padding, `position_ids`, `assistant_prefix_tokens` for auto max_positions |
| `activations.py` | `ActivationExtractor` — pre-hooks on block inputs (residual stream entering each layer), casts to float64 for stability |
| `direction_methods.py` | `DirectionMethod` ABC, `DifferenceInMeans` (mean_pos - mean_neg in float64) |
| `interventions.py` | `ModelInterventionApplier` — pre-hooks for add/subtract/ablate + sublayer post-hooks for ablation (3 hooks/layer, matches Arditi) |
| `scoring.py` | `LogOddsMetric`, `Three_Score_Evaluator` (bypass/induce/induce_global/KL) |
| `search.py` | `DirectionFinder` — multi-objective search with progressive fallback tiers, supports `induce_mode` |
| `evaluation.py` | `BigEvaluator` (detection rates, LlamaGuard2, JailbreakBench, Alpaca CE, lm-eval), `InterventionSuite` (eyeball) |
| `generation.py` | `generate_with_hooks()` — autoregressive gen with KV-cache and explicit `position_ids` (hooks fire per-step) |
| `cross_concept.py` | Cosine similarity, PCA, interference matrix, multi-ablation composition |
| `framework.py` | `DirectionTestFramework` — orchestrator, model loading, mode dispatch, prompt filtering, supports pre-split val data |
| `run_experiment.py` | CLI (argparse or `--json`) |

Scripts in `scripts/`: `search_calibration.py`, `arditi_comparison.py`, `cross_dataset_comparison.py`, `layer_sweep.py`, `llama2_strength_sweep.py`, `llama2_replication_diagnostic.py`, `compare_lg2_lg3.py`, `arditi_replication.py`, `arditi_raw_addition.py`, `arditi_evals_factorial.py`, `inspect_ablated_outputs.py`, `probe_layer_directions.py`, `scale_experiment.py`, `debug_batch_padding.py`, `inspect_chat_template.py`, `arditi_exact_replication.py`, `arditi_pos5_test.py`, `arditi_llama3_replication.py`.

Tests in `tests/`: `test_unit.py` (datatypes, scoring, detection), `test_generation.py` (per-model generation + stability), `test_per_model.py` (padding consistency, chat template, position_ids, activation extraction), `test_smoke.py` (full pipeline on gemma-3-270m).

Output dirs (gitignored): `results/` (logs, `.pt`/`.json` directions, calibration JSON), `plots/` (PNGs).

## Adding a New Concept

1. **`prompts.py`**: Add `create_<concept>_train_data()` and `create_<concept>_eval_data()` returning `(positive, negative)`. Use topic-matched contrastive pairs. Train/eval disjoint.
2. **`concept.py`**: Define tokens (include space-prefixed), phrases, search config. Optionally add `detect_<concept>()` heuristic and judge prompt. Create factory, call `register_concept()`.
3. Run `--mode search --concept <name>`.
4. Add detection heuristic tests in `test_unit.py`.
5. Add model to `PER_MODEL_TEST_MODELS` in `test_per_model.py` and verify padding/template tests pass.

## Adding a New Model Architecture

- `interventions.py`: Add branch to `_get_transformer_layers()` (currently: `.model.layers`, `.transformer.h`) and `_get_sublayers()` (currently: `.self_attn`+`.mlp`, `.attn`+`.mlp`).
- `formatting.py`: Add manual chat template fallback if model lacks `tokenizer.chat_template`.
- Run `test_per_model.py` on the new model to verify padding, template, and position_ids correctness.

## Gotchas

- **Right-padding is required**: Left-padding with RoPE models (Llama, Qwen) corrupts logits for all padded prompts due to incorrect position encoding. `formatting.py` uses right-padding + explicit `position_ids` (computed from attention_mask via `cumsum`). All `model()` calls must pass `position_ids`. The `generate_with_hooks()` function tracks per-sequence positions for autoregressive decoding. See "Critical Bug Fix" below.
- **Hook cleanup**: `clear_interventions()` after every intervention (use try/finally). Leaked hooks corrupt results.
- **Environment variables**: `TOKENIZERS_PARALLELISM=false` and `PYTORCH_ENABLE_MPS_FALLBACK=0` must be set before importing ML libs (already done in `run_experiment.py` and `conftest.py`).
- **MPS float16**: Framework auto-upcasts float16 to bfloat16 on MPS to prevent attention overflow.
- **Addition uses raw (unnormalized) vectors** (matching Arditi). Ablation uses unit vectors (projection removal is scale-invariant). Subtraction uses raw vectors (like addition but reversed). All interventions use pre-hooks on block inputs; ablation additionally hooks `self_attn` and `mlp` post-hooks to prevent re-injection (3 hooks/layer).
- **NaN in scoring**: `LogOddsMetric` returns nan for inf/nan logits; downstream uses `nanmean`. Typically occurs only for layer-0 candidate directions (near-zero norms cause unstable unit vector normalization). Not an error.
- **Search fallback**: Strict criteria rarely met on models <=1.8B. Progressive tiers: (Δinduce>+3, KL<5) → (Δinduce>+1.5, KL<10) → (Δinduce>+0, KL<20) → best induce. Tiers rank by induce score. When `induce_mode=all_layers`, tiers use `induce_global` instead of `induce`.
- **Detection hierarchy**: API judge → `detection_fn` heuristic → phrase matching. Study model never self-judges.
- **LlamaGuard 2 not 3**: LG3 flags by topic (80-93% FP on refusals). LG2 flags by compliance. Use LG2.
- **Llama-2 chat template**: HuggingFace's built-in template doesn't add a generation prompt. We override it with `[INST] {x} [/INST] ` (trailing space) so pos -1 is the generation boundary, not `]`. Without this, difference-in-means finds a garbage direction and induction fails completely.
- **`generate_with_hooks()`** exists because `model.generate()` doesn't invoke forward hooks every step.
- **Prompt filtering on by default** but has zero measured effect (3-21% mislabeled prompts don't contaminate difference-in-means). Use `--no-filter-prompts` to disable.
- **Pre-split val data**: `train_data_fn` can return `((train_pos, train_neg), (val_pos, val_neg))` to provide a separate validation set. The framework detects this and skips `train_val_split()`, using all train data for direction computation. Used by `refusal_arditi_exact` concept.

## Critical Bug Fix: Right-Padding (April 2025)

**Bug**: Left-padding with HuggingFace transformers corrupts logits for all padded prompts in a batch. With left-padding, the model assigns wrong RoPE position encodings to real tokens (position 0 goes to a pad token instead of the first real token). The internal `create_causal_mask` does not correct for this. This affects ALL models with RoPE (Llama, Qwen, etc.) on ALL devices (MPS, CPU, CUDA).

**Impact**: Every prior experiment had corrupted scoring. The search selected wrong directions because bypass/induce/KL scores were computed from garbage logits for most prompts in every batch. Direction vectors from difference-in-means were less affected (activation extraction uses hooks, not logits), but the selected (layer, position) was often wrong.

**Fix**: Switched to right-padding + explicit `position_ids` in `formatting.py`. All `model()` calls in `activations.py`, `scoring.py`, `generation.py`, and `framework.py` now pass `position_ids`. `generate_with_hooks()` tracks per-sequence positions for autoregressive decoding.

**Verification**: `test_per_model.py::TestPaddedBatchConsistency` checks that batched logits match individual processing. `TestLeftPaddingRegression` confirms left-padding fails on all tested models.

**Result**: All 7 models re-searched and re-evaluated. Every model now finds a strictly-passing direction (no fallback). Ablation effectiveness improved dramatically across the board; Llama-3 went from 16% to 99% induction.

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

Full results: `results/results_summary.md`, experiment log: `results/extended_results_list.md`.

## Key Findings

- **Refusal direction works universally**: 73-100% induction and 66-100pp ablation across all 7 models tested (Qwen2.5 0.5B-7B, Llama-2-7B, Llama-3-8B, Llama-3.1-8B). The "single direction" hypothesis holds strongly.
- **Right-padding fix was critical**: Left-padding with RoPE models corrupts logits for all padded sequences. This was the root cause of all prior Llama "weak results" and the 7B "ablation-resistant" finding. See "Critical Bug Fix" above.
- **Arditi replication successful**: With correct padding, our pipeline selects L12/pos-5 on Llama-3-8B with Arditi's exact data (bypass=-10.7 vs paper's -9.7), strictly passing all criteria.
- **Multi-position search matters**: 5/7 models select non-pos-1 positions. The `max_positions=auto` setting (derived from `assistant_prefix_tokens`) correctly searches all post-instruction token positions.
- **Llama-2 template bug**: HuggingFace's Llama-2 chat template doesn't respond to `add_generation_prompt`. We override with `[INST] {x} [/INST] ` (trailing space). Without this, induction fails completely.
- **Our dataset produces strong directions**: Topic-matched 80+80 pairs work well. Arditi's 128+128 (from AdvBench + MaliciousInstruct + TDC2023) also works when scoring is correct.
- **Sycophancy is harder**: Low induction rates (2-12%), no clean single direction on small models. May need re-evaluation with corrected padding.
- **Hedging: negative result**: 0% behavioral detection across all conditions on all 4 Qwen2.5 models. Models don't hedge on factual questions at baseline.
- **Empathy direction works**: Subtraction removes empathy completely (80%→0% on 1.5B, 30%→0% on 3B). Addition induces empathy on 30-50% of neutral prompts.

## Next Steps

- Re-run sycophancy, hedging, and empathy experiments with corrected padding — prior results may be invalid.
- Re-run cross-concept analysis (refusal × sycophancy × hedging) with corrected directions.
- Evaluate Llama-3/3.1 with Arditi's exact data (`--concept refusal_arditi_exact`) for full replication.
- Update `results_summary.md` and `extended_results_list.md` with corrected numbers.
- Hedging: try opinion/subjective prompts instead of factual questions.
- Larger models (13B+) for cleaner sycophancy directions.
