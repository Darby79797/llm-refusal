# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic interpretability research investigating how LLMs implement learned behaviors (refusal, sycophancy, etc.) via directions in the residual stream activation space. Currently configured for refusal (reproducing and extending Arditi & Obeso's paper "Refusal in Language Models is Mediated by a Single Direction"). The framework is concept-generic: new behaviors can be studied by registering a `ConceptDefinition` without modifying core pipeline code.

## Repository Structure

```
llm-refusal/
├── CLAUDE.md
├── pyproject.toml              # setuptools, Python >=3.11
└── llm-refusal/
    ├── datatypes.py            # PromptData, DirectionScores, DirectionVector
    ├── concept.py              # ConceptDefinition, concept registry, refusal defaults
    ├── formatting.py           # ChatPromptFormatter
    ├── activations.py          # ActivationExtractor
    ├── interventions.py        # ModelInterventionApplier, InterventionStrategy + subclasses
    ├── direction_methods.py    # DirectionMethod ABC, DifferenceInMeans
    ├── scoring.py              # LogOddsMetric, DirectionEvaluator ABC, Three_Score_Evaluator
    ├── search.py               # DirectionFinder
    ├── evaluation.py           # InterventionSuite, BigEvaluator
    ├── generation.py           # generate_with_hooks
    ├── framework.py            # DirectionTestFramework, main()
    ├── prompts.py              # Train + eval prompt datasets (refusal-specific)
    ├── run_experiment.py       # CLI runner (argparse, --model/--mode flags or --json)
    ├── scripts/
    │   ├── debug_batch_padding.py    # Reproduces batched-generation padding bugs
    │   └── inspect_chat_template.py  # Prints the resolved chat template for a model
    └── tests/
        ├── conftest.py         # Shared fixtures, pytest hooks, model list
        ├── test_unit.py        # Unit tests with mocks
        ├── test_generation.py  # Integration tests for batched generation
        └── test_smoke.py       # Quick smoke tests on small models
```

Output directories (gitignored): `results/` (log files), `plots/` (PNG visualizations).

## Commands

```bash
# Install dependencies (pip/setuptools, not poetry)
pip install -e .

# Run all tests (excludes smoke tests by default)
pytest llm-refusal/tests/

# Run only smoke tests (quick validation, uses google/gemma-3-270m)
pytest --run-smoke llm-refusal/tests/

# Run a single test file
pytest llm-refusal/tests/test_unit.py

# Run an experiment via CLI
python llm-refusal/run_experiment.py --model Qwen/Qwen1.5-1.8B-Chat --mode search
python llm-refusal/run_experiment.py --model Qwen/Qwen1.5-1.8B-Chat --mode evaluate --layer 10 --pos -1

# Or pass config as JSON (inline or file path)
python llm-refusal/run_experiment.py --json '{"model_name": "Qwen/Qwen1.5-1.8B-Chat", "mode": "search"}'
```

## Configuration

`run_experiment.py` accepts the following config (via flags or `--json`):

```python
config = {
    "model_name": "Qwen/Qwen1.5-1.8B-Chat",  # HuggingFace model ID
    "torch_dtype": "auto",   # Passed to from_pretrained()
    "force_cpu": False,       # Override device selection to CPU
    "mode": "search",         # "search", "evaluate", or "eyeball"
    "layer": None,            # Required for evaluate/eyeball (int)
    "pos": None,              # Required for evaluate/eyeball (int, typically -1)
    "eval_tasks": [],         # lm-eval tasks: "mmlu", "arc_challenge", "gsm8k", "truthfulqa"
    "limit": 100,             # Sample limit for lm-eval benchmark runs
    "concept": "refusal"      # Concept to study (string key into concept registry)
}
```

**Modes:**
- **search** — Scans first 80% of layers to find the best direction via multi-objective optimization. Produces plots in `plots/`.
- **evaluate** — Requires `layer` and `pos`. Runs quantitative evaluation: detection rates (with/without intervention) and optional lm-eval benchmarks.
- **eyeball** — Requires `layer` and `pos`. Generates text side-by-side (baseline vs. intervened) for qualitative inspection.

## Architecture

The codebase is organized into focused modules with clean interfaces.

### Concepts (`concept.py`)
- **`ConceptDefinition`** — Dataclass bundling everything needed to study a behavior: `name`, `train_data_fn`, `eval_data_fn`, `target_tokens`, `detection_phrases`, `search_config`. Pure data — no behavior to override.
- **`make_refusal_concept()`** — Factory that builds the refusal `ConceptDefinition` using prompts from `prompts.py`.
- **`CONCEPT_REGISTRY`** / `register_concept()` / `get_concept()` — String-keyed registry mapping concept names to factory functions. `"refusal"` is registered by default. New concepts are added via `register_concept("sycophancy", make_sycophancy_concept)`.
- **`DEFAULT_REFUSAL_TOKENS`**, **`DEFAULT_REFUSAL_PHRASES`**, **`DEFAULT_SEARCH_CONFIG`** — Constants bundled into the refusal concept by `make_refusal_concept()`.

### Data & Formatting
- **`prompts.py`** — `create_refusal_train_data()` and `create_refusal_eval_data()` return `(harmful_prompts, safe_prompts)`. Train and eval sets are disjoint. Both accept optional subset sizes and `random_seed`. Future concepts would have their own prompt modules, consumed via `ConceptDefinition.train_data_fn` / `eval_data_fn`.
- **`PromptData`** (`datatypes.py`) — Dataclass for prompts + boolean labels (`True`=positive). Has `.train_val_split()`.
- **`ChatPromptFormatter`** (`formatting.py`) — Applies chat templates, tokenizes, and left-pads for batched generation. Auto-detects instruction-tuned models by checking for `-it`, `-instruct`, or `-chat` in the model name. Falls back to manual templates for Gemma, Qwen, Yi, Llama-2, Llama-3. Base models get pass-through with BOS prepending.

### Extraction & Direction Finding
- **`ActivationExtractor`** (`activations.py`) — Extracts residual stream activations from all layers at specified positions using forward hooks.
- **`DirectionMethod`** (`direction_methods.py`) — ABC defining `compute_direction_vectors(train_data, max_positions) -> Dict[(layer, pos), Tensor]`. New direction-finding methods implement this interface.
- **`DifferenceInMeans`** (`direction_methods.py`) — Computes direction vectors (mean_positive − mean_negative) for each (layer, position) pair. Implements `DirectionMethod`. Keeps `compute_difference_vectors` as a backward-compat alias.

### Intervention
- **`ModelInterventionApplier`** (`interventions.py`) — Registers forward hooks on transformer layers for `add`, `subtract`, or `ablate` (projection removal) interventions. Auto-detects layer structure (`.model.layers` or `.transformer.h`).
- **`InterventionStrategy`** (`interventions.py`) ABC with two subclasses:
  - **`GlobalInterventionStrategy`** — Applies to all layers.
  - **`LayerSpecificInterventionStrategy`** — Applies only to the layer where the direction was found.
- **`DirectionVector`** (`datatypes.py`) — Dataclass holding vector tensor, layer index, position index, and score. Has `.unit` property.

### Evaluation
- **`LogOddsMetric`** (`scoring.py`) — Log-odds ratio of target tokens vs. all other tokens from next-token logits.
- **`Three_Score_Evaluator`** (`scoring.py`) — Concept-parameterized via `target_tokens`. Evaluates directions on three metrics:
  - **bypass** (lower=better): Log-odds on positive prompts with global ablation.
  - **induce** (higher=better): Log-odds on negative prompts with layer-specific addition.
  - **kl** (lower=better): KL divergence on negative prompts with global ablation.
- **`DirectionFinder`** (`search.py`) — Multi-objective search with injected `DirectionMethod`, `Three_Score_Evaluator`, and `search_config` (from concept). Satisfices on induce > threshold and KL < threshold, then minimizes bypass. Produces score-vs-layer plots.

### Testing & Orchestration
- **`InterventionSuite`** (`evaluation.py`) — Qualitative: generates text with/without interventions at various strengths.
- **`BigEvaluator`** (`evaluation.py`) — Concept-parameterized via `detection_phrases`. Quantitative: detection rate (string-match) + lm-eval benchmarks (wraps model in `HFLM`). Methods `_check_for_detection` / `evaluate_detection_rate` have backward-compat aliases `_check_for_refusal` / `evaluate_refusal_rate`.
- **`DirectionTestFramework`** (`framework.py`) — Main orchestrator. Accepts `concept` string (default `"refusal"`), looks up `ConceptDefinition` from registry, loads model/tokenizer, wires concept through to all components, dispatches to mode.
- **`generate_with_hooks()`** (`generation.py`) — Standalone autoregressive generation with KV-cache. Needed because `model.generate()` doesn't invoke registered forward hooks on every step.

### Import DAG (no cycles)

```
datatypes
├── concept           (+ prompts)
├── formatting
├── activations       (+ formatting)
├── interventions     (+ datatypes)
├── direction_methods (+ activations, datatypes)
├── scoring           (+ interventions, formatting, datatypes, concept)
├── generation        (+ formatting)
├── search            (+ direction_methods, scoring, interventions, concept, datatypes)
├── evaluation        (+ interventions, formatting, datatypes, concept)
└── framework         (+ all above)
```

## Device Handling

`DirectionTestFramework` selects device: CUDA > MPS > CPU (overridable via `force_cpu: True`). If a model loads as `float16` on MPS, it is upcast to `bfloat16` to prevent attention overflow (NaN/Inf).

## Test Structure

- `test_unit.py` — Unit tests with mocks for data splitting, hooks, interventions, metrics. Patches `scoring.F.kl_div` for KL score tests.
- `test_generation.py` — Integration tests parametrized across: `google/gemma-3-1b-pt`, `google/gemma-3-1b-it` (expected to fail), `openai-community/gpt2-xl`.
- `test_smoke.py` — End-to-end pipeline smoke test on `google/gemma-3-270m`.
- `conftest.py` — Shared fixtures (`model_and_tokenizer` parametrized across test models, session-scoped), `--run-smoke` CLI option, mock factories.

## Output Artifacts

- `results/` — Log files: `{ModelShortName}-{mode}[-L{layer}][-P{pos}].log`
- `plots/` — PNGs: `{ModelShortName}-induce_score_vs_layer.png`, `{ModelShortName}-bypass_score_vs_layer.png`
