# Architecture

All code under `llm-refusal/`. Flat imports (`from formatting import ...`), set by `pythonpath = ["llm-refusal"]` in pyproject.toml. One-off experiment scripts in `scripts/`; tests in `tests/` (`test_unit`, `test_generation`, `test_per_model`, `test_smoke`).

| Module | Purpose |
|--------|---------|
| `datatypes.py` | `PromptData` (stratified `train_val_split`), `DirectionScores` (bypass/induce/induce_global/KL), `DirectionVector` (`.save()`/`.load()`) |
| `concept.py` | `ConceptDefinition` dataclass, registry (`register_concept`/`get_concept`), `DEFAULT_SEARCH_CONFIG`, all concept definitions |
| `prompts.py` | Train + eval prompt datasets per concept, returning `(positive, negative)` or pre-split `((train_pos, train_neg), (val_pos, val_neg))` |
| `formatting.py` | `ChatPromptFormatter` — chat templates, tokenization, right-padding, `position_ids`, `assistant_prefix_tokens`; `format_with_completions()` builds [prompt | completion] rows with completion-only `labels` for scoring and training |
| `activations.py` | `ActivationExtractor` — pre-hooks on block inputs (residual stream entering each layer), float64 for stability |
| `direction_methods.py` | `DirectionMethod` ABC, `DifferenceInMeans` (mean_pos - mean_neg in float64) |
| `interventions.py` | `ModelInterventionApplier` — pre-hooks for add/subtract/ablate + sublayer post-hooks for ablation (3 hooks/layer, matches Arditi). Ablate also takes a [k, d] stack and removes its span in one order-independent projection (`orthogonalize.orthonormal_basis`); `cross_concept` uses this for joint ablation. `intervened()` context manager clears hooks on exit |
| `orthogonalize.py` | Weight orthogonalisation (Arditi §4): `orthogonalized(model, direction, layers=None, embedding=True)` projects r̂ (or a rank-k span) out of the embedding and every `o_proj`/`down_proj` (and biases). Edits the weights in place and restores them from the model's local safetensors checkpoint on exit (same load-time cast, exact bit checksum per tensor, one re-read before failing), so it holds no second copy (~6 GB on 8B). Tied embeddings are untied for the duration. Falls back to swapping in edited copies when there's no checkpoint on disk; `prepare_edit` builds those copies for repeated re-entry (the equivalence script). Refuses models that normalise sublayer outputs before the residual add (Gemma-2/3) |
| `scoring.py` | `LogOddsMetric`, `Three_Score_Evaluator` (bypass/induce/induce_global/KL) |
| `search.py` | `DirectionFinder` — multi-objective search with progressive fallback tiers, supports `induce_mode` |
| `evaluation.py` | `BigEvaluator` (detection rates, LlamaGuard2, JailbreakBench, Alpaca CE, lm-eval), `InterventionSuite` (eyeball) |
| `coherence.py` | Per-response coherence under the clean model: `degenerate` (repetition-loop flag, the breakage signal) and response NLL (divergence from clean behaviour, not fluency). Scored for every evaluate condition (~4% of runtime) |
| `capability.py` | `--mode capability`: what the edit costs. CE on Alpaca reference completions, raw Pile text and the clean model's own completions (`completion_ce`, lm_head applied only at scored positions), plus `--eval-tasks` lm-eval scores, for the unedited, orthogonalised and random-direction-orthogonalised model → `results/capability/*.json`. Data from `scripts/fetch_capability_data.py` (`data/alpaca_completions.json`, `data/pile_sample.json`; Alpaca rows used by the direction's training data are excluded) |
| `finetune.py` | `LowRankAdapter` (y = Wx + U Vᵀx, U=0 init, fp32 on a frozen model), `adapted()` context manager, `train_adapters()`. `--mode rank1`: a rank-one adapter distils directional ablation (or addition, `--objective induce`); reports cos(u, r̂) and the effect with r̂ removed from u / only r̂ kept. `--mode regrow`: LoRA on writers or readers of the orthogonalised model, benign Alpaca ± refusal examples; tracks refusal and re-extracts the direction → `results/finetune/`. `installed(model, stem, variant, r_hat)` re-installs a saved adapter set (`framework.model_variant` wires it to `--adapter-file`) |
| `batching.py` | `--gen-batch-size auto` resolution (deterministic memory estimate → largest power of two), `map_batched` (splits a batch in half on OOM and records it) |
| `env.py` | Torch-free `setup_process_env()`: tokenizer/MPS env vars and the MPS allocator watermarks (`MPS_WATERMARKS`). Every entry point calls it before importing torch |
| `generation.py` | `generate_with_hooks()` — autoregressive gen with KV-cache and explicit `position_ids` (hooks fire per-step) |
| `cross_concept.py` | Cosine similarity, PCA, interference matrix, multi-ablation composition |
| `attribution.py` | Circuit analysis: per-head/MLP projection onto the direction, contrastive (harmful−benign) attribution. Used by `scripts/`, not the CLI |
| `framework.py` | `DirectionTestFramework` — orchestrator: model loading, mode dispatch, prompt filtering, pre-split val support |
| `run_experiment.py` | CLI (argparse or `--json`) |
| `caa.py` | CAA (Panickssery et al.) A/B replication: answer-letter contrast vectors at block *outputs* (CAA layer L = our layer L+1), per-layer normalization across behaviors, steering from the prompt boundary, p(matching) metric. `--mode caa`; data in `data/caa/` |
| `tools/runs.py` | Index of saved artifacts (generations, logs, search CSVs, directions, CAA, cross-concept), Wilson CIs, run flags, current-detector rescoring. No model |
| `tools/look.py` | Text views: `digest`, `run`, `gens` (flips/labels/grep/`--rescore`), `search` (layer×pos grid), `caa`, `cross`, `proj` |
| `tools/report.py` | Self-contained HTML report (`results/report/index.html`): overview matrix, run detail + search heatmap + generations browser, CAA curves, cross-concept, token projections |
| `probe.py` | Forward-pass helpers for the analysis scripts: `residuals_at` (per-prompt residual at a template position, every layer), `behaviour` (rate + log-odds under a context), `auroc`, `cos` |
| `tools/project.py` | Per-token projections onto a direction at every layer (loads a model) → `results/proj/*.json` |

## Adding a Concept

Train/eval data fns in `prompts.py` (topic-matched contrastive pairs, train/eval disjoint) → tokens/phrases/search config + optional `detect_<concept>()` heuristic in `concept.py` → `register_concept()` → detection tests in `test_unit.py`.

## Adding a Model Architecture

Branch in `_get_transformer_layers()`/`_get_sublayers()` in `interventions.py`; manual chat template fallback in `formatting.py` if needed; then run `test_per_model.py` on it (add to `PER_MODEL_TEST_MODELS`).

## Implementation Notes

- **Vector normalization**: Addition and subtraction use raw (unnormalized) vectors, matching Arditi; ablation uses unit vectors (projection removal is scale-invariant). All interventions are pre-hooks on block inputs; ablation additionally post-hooks `self_attn` and `mlp` to prevent re-injection (3 hooks/layer).
- **`generate_with_hooks()`** exists because `model.generate()` doesn't invoke forward hooks every step.
- **NaN in scoring**: `LogOddsMetric` returns nan for inf/nan logits; downstream uses `nanmean`. Search skips layer 0 (raw embeddings; its template-position difference vectors are exactly zero). Not an error.
- **Strict selection**: among candidates passing induce > threshold and KL < threshold, lowest bypass wins (Arditi's rule), except that passers within `bypass_tie_frac` (default 5%) of the best bypass count as tied and the highest induce breaks the tie. `refusal_arditi_exact` sets it to 0 (the paper's rule). At 5% it moves Llama-3.1 refusal L12/P-2 → L11/P-1 (85% → 100% induction) and Qwen2.5-7B L17/P-4 → L16/P-4 (a near-tie); every other saved selection is unchanged.
- **Search fallback**: Strict criteria rarely met on models <=1.8B. Progressive tiers: (Δinduce>+3, KL<5) → (Δinduce>+1.5, KL<10) → (Δinduce>+0, KL<20) → best induce. With `induce_mode=all_layers`, tiers use `induce_global`.
- **Evaluate conditions**: `--conditions` picks from `evaluation.CONDITIONS`. Default is global ablation, layer ablation, layer addition, plus whichever baselines (harmful/harmless) they need. `global_addition` (always degenerate repetition loops), `layer_specific_subtraction` (near-redundant with ablation) and `orthogonalized` (global ablation as a weight edit; run it beside `global_ablation` to compare the texts) are opt-in; `--conditions all` runs everything. Every run writes `results/<model>-<concept>-evaluate-L<l>-P<p>-generations.json` with each response, its detection label and coherence fields.
- **Filtering cache**: the prompt-filtering generation pass (~17% of an evaluate run) is cached in `results/filter-cache/`, keyed on model, dtype, gen batch size, detector and prompt set. `--no-filter-cache` regenerates; bump `framework.FILTER_CACHE_VERSION` when generation/detection logic changes.
- **Prompt filtering on by default** but has zero measured effect (3-21% mislabeled prompts don't contaminate difference-in-means). `--no-filter-prompts` disables.
- **Pre-split val data**: `train_data_fn` may return `((train_pos, train_neg), (val_pos, val_neg))`; the framework then skips `train_val_split()` and uses all train data for the direction. Used by `refusal_arditi_exact`.
- **External API judge**: `JUDGE_API_BASE`, `JUDGE_API_KEY`, `JUDGE_MODEL` env vars (or `--judge-*` flags). `--arditi-evals` enables LlamaGuard2 + JailbreakBench + Alpaca CE; `--eval-tasks` runs lm-eval benchmarks.
