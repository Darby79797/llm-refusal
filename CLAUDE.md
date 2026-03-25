# CLAUDE.md

Mechanistic interpretability research: finding directions in the residual stream that mediate learned behaviors (refusal, sycophancy, hedging). Extends Arditi & Obeso's "Refusal in Language Models is Mediated by a Single Direction". Concept-generic: register a `ConceptDefinition` to study new behaviors without touching core code.

## Commands

```bash
pip install -e .                          # setuptools, Python >=3.11
pytest llm-refusal/tests/                 # unit tests (excludes smoke)
pytest --run-smoke llm-refusal/tests/     # smoke tests (gemma-3-270m)

# Main CLI
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode search
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode evaluate --layer 18 --pos -1
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode eyeball --layer 18 --pos -1
python llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode cross_concept --concepts refusal,sycophancy
python llm-refusal/run_experiment.py --json '{"model_name": "...", "mode": "search"}'
```

## Key Config Options

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | required | HuggingFace model ID |
| `--mode` | required | `search`, `evaluate`, `eyeball`, `cross_concept` |
| `--concept` | `refusal` | Registry key: `refusal`, `refusal_arditi`, `sycophancy`, `sycophancy_neutral`, `hedging`, `empathy` |
| `--layer`, `--pos` | — | Required for evaluate/eyeball (pos is typically -1) |
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
| `prompts.py` | Train + eval prompt datasets per concept, all return `(positive, negative)` |
| `formatting.py` | `ChatPromptFormatter` — chat templates, tokenization, left-padding, `assistant_prefix_tokens` for auto max_positions |
| `activations.py` | `ActivationExtractor` — pre-hooks on block inputs (residual stream entering each layer), casts to float64 for stability |
| `direction_methods.py` | `DirectionMethod` ABC, `DifferenceInMeans` (mean_pos - mean_neg in float64) |
| `interventions.py` | `ModelInterventionApplier` — pre-hooks for add/subtract/ablate + sublayer post-hooks for ablation (3 hooks/layer, matches Arditi) |
| `scoring.py` | `LogOddsMetric`, `Three_Score_Evaluator` (bypass/induce/induce_global/KL) |
| `search.py` | `DirectionFinder` — multi-objective search with progressive fallback tiers, supports `induce_mode` |
| `evaluation.py` | `BigEvaluator` (detection rates, LlamaGuard2, JailbreakBench, Alpaca CE, lm-eval), `InterventionSuite` (eyeball) |
| `generation.py` | `generate_with_hooks()` — autoregressive gen with KV-cache (hooks fire per-step) |
| `cross_concept.py` | Cosine similarity, PCA, interference matrix, multi-ablation composition |
| `framework.py` | `DirectionTestFramework` — orchestrator, model loading, mode dispatch, prompt filtering |
| `run_experiment.py` | CLI (argparse or `--json`) |

Scripts in `scripts/`: `search_calibration.py`, `arditi_comparison.py`, `cross_dataset_comparison.py`, `layer_sweep.py`, `llama2_strength_sweep.py`, `llama2_replication_diagnostic.py`, `compare_lg2_lg3.py`, `arditi_replication.py`, `arditi_raw_addition.py`, `arditi_evals_factorial.py`, `inspect_ablated_outputs.py`, `probe_layer_directions.py`, `scale_experiment.py`, `debug_batch_padding.py`, `inspect_chat_template.py`.

Output dirs (gitignored): `results/` (logs, `.pt`/`.json` directions, calibration JSON), `plots/` (PNGs).

## Adding a New Concept

1. **`prompts.py`**: Add `create_<concept>_train_data()` and `create_<concept>_eval_data()` returning `(positive, negative)`. Use topic-matched contrastive pairs. Train/eval disjoint.
2. **`concept.py`**: Define tokens (include space-prefixed), phrases, search config. Optionally add `detect_<concept>()` heuristic and judge prompt. Create factory, call `register_concept()`.
3. Run `--mode search --concept <name>`.
4. Add detection heuristic tests in `test_unit.py`.

## Adding a New Model Architecture

- `interventions.py`: Add branch to `_get_transformer_layers()` (currently: `.model.layers`, `.transformer.h`) and `_get_sublayers()` (currently: `.self_attn`+`.mlp`, `.attn`+`.mlp`).
- `formatting.py`: Add manual chat template fallback if model lacks `tokenizer.chat_template`.

## Gotchas

- **Hook cleanup**: `clear_interventions()` after every intervention (use try/finally). Leaked hooks corrupt results.
- **Environment variables**: `TOKENIZERS_PARALLELISM=false` and `PYTORCH_ENABLE_MPS_FALLBACK=0` must be set before importing ML libs (already done in `run_experiment.py` and `conftest.py`).
- **MPS float16**: Framework auto-upcasts float16 to bfloat16 on MPS to prevent attention overflow.
- **Addition uses raw (unnormalized) vectors** (matching Arditi). Ablation uses unit vectors (projection removal is scale-invariant). Subtraction uses raw vectors (like addition but reversed). All interventions use pre-hooks on block inputs; ablation additionally hooks `self_attn` and `mlp` post-hooks to prevent re-injection (3 hooks/layer). On Llama-2, ablation is ineffective but subtraction works — the projection magnitude is too small to overcome refusal at a single layer.
- **NaN in scoring**: `LogOddsMetric` returns nan for inf/nan logits; downstream uses `nanmean`. Not an error.
- **Search fallback**: Strict criteria rarely met on models <=1.8B. Progressive tiers: (Δinduce>+3, KL<5) → (Δinduce>+1.5, KL<10) → (Δinduce>+0, KL<20) → best induce. Tiers rank by induce score. When `induce_mode=all_layers`, tiers use `induce_global` instead of `induce`.
- **Detection hierarchy**: API judge → `detection_fn` heuristic → phrase matching. Study model never self-judges.
- **LlamaGuard 2 not 3**: LG3 flags by topic (80-93% FP on refusals). LG2 flags by compliance. Use LG2.
- **Llama-2 chat template**: HuggingFace's built-in template doesn't add a generation prompt. We override it with `[INST] {x} [/INST] ` (trailing space) so pos -1 is the generation boundary, not `]`. Without this, difference-in-means finds a garbage direction and induction fails completely.
- **`generate_with_hooks()`** exists because `model.generate()` doesn't invoke forward hooks every step.
- **Prompt filtering on by default** but has zero measured effect (3-21% mislabeled prompts don't contaminate difference-in-means). Use `--no-filter-prompts` to disable.

## Best Layers (Refusal)

| Model | Best | Pos | Ablation Δ | Induction | Notes |
|-------|------|-----|------------|-----------|-------|
| Qwen2.5-0.5B (24L) | 12 | -1 | -38pp | 89% | |
| Qwen2.5-1.5B (28L) | 13 | -1 | -20pp | 100% | Sweet spot L12-14 |
| Qwen2.5-3B (36L) | 18 | -1 | -83pp | 98% | Strongest result |
| Qwen2.5-7B (28L) | 14 | -1 | -1pp | 91% | Ablation-resistant |
| Llama-3-8B (32L) | 14 | -1 | -20pp | 16% (s=1), 0% (s=3) | Ablation works, induction fails |
| Llama-3.1-8B (32L) | 15 | -2 | -28pp | 4% (s=1), 0% (s=3) | Best Llama ablation |
| Llama-2-7B (32L) | 14 | -1 | -70pp (subtract s=2) | 71% (add s=2) | Template fix critical; single-layer only |

Sweet spot is ~35-60% depth. Full results: `results/results_summary.md`, experiment log: `results/extended_results_list.md`.

## Key Findings

- **Refusal direction works on Qwen**: 89-100% induction across all Qwen2.5 models at ~50% depth, raw strength=1.
- **Llama-2 template bug fixed**: HuggingFace's Llama-2 chat template doesn't respond to `add_generation_prompt`, causing pos -1 to be `]` instead of the generation-boundary space token. Fixing this (`[INST] {x} [/INST] ` with trailing space) moved induction from 0% to 71% at L14/s=2 (single-layer). Llama-3/3.1 templates are correct.
- **Llama-2 bidirectional result**: Same L14/-1 direction induces refusal (+71%) and removes it (-70pp). Ablation (projection) fails — subtraction (scaled vector removal) required. Single-layer only; all-layer garbles.
- **Llama-3/3.1: weaker results**: Ablation -1 to -28pp, induction 0-16%. May benefit from single-layer subtract approach (not yet tested).
- **Ablation is model-dependent**: 3B loses 83pp refusal; 7B loses 1pp (redundant circuits).
- **Our dataset >> Arditi's**: Topic-matched 80+80 pairs produce 20x stronger directions than Arditi's 260+18793 (class imbalance dilutes signal).
- **Sycophancy is harder**: Low induction rates (2-12%), no clean single direction on small models.
- **Asymmetric interference**: Ablating refusal increases sycophancy +17pp; ablating sycophancy reduces refusal -57pp on 3B (strong cross-interference despite cos=0.176).
- **Prompt filtering has zero effect**: 3-21% mislabeled prompts don't contaminate difference-in-means. Not worth compute.
- **Search calibration**: LogOdds induce scores mislead at deep layers (>60% depth). Fixed via `layer_cutoff_frac=0.65`.
- **Hedging: negative result**: Search finds directions with positive LogOdds induce scores, but 0% behavioral detection across all conditions on all 4 Qwen2.5 models. Models don't hedge on factual questions at baseline.
- **3-way cross-concept (3B)**: Refusal/sycophancy/hedging directions span a 2D subspace. Ablating sycophancy direction reduces refusal by 57pp (strong cross-interference).
- **Empathy direction works**: Subtraction removes empathy completely (80%→0% on 1.5B, 30%→0% on 3B). Addition induces empathy on 30-50% of neutral prompts. Ablation ineffective — same pattern as Llama-2 refusal, suggesting subtraction is generally superior to projection-based ablation.

## Next Steps

- Hedging: try opinion/subjective prompts instead of factual questions (models may hedge more on ambiguous topics)
- Larger models (7B+) for cleaner sycophancy directions
- Investigate refusal-sycophancy cross-interference: why does ablating sycophancy direction reduce refusal by 57pp?
- Re-evaluate Llama-3/3.1 with template awareness (their templates are correct but results may improve with single-layer addition + higher strength)
- Check if Qwen results are affected by the template change (they use built-in templates, should be unaffected)
