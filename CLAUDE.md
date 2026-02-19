# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mechanistic interpretability research reproducing and extending Arditi and Obeso's paper "Refusal in Language Models is Mediated by a Single Direction." The project investigates how LLMs implement refusal behaviors and whether they can be controlled via specific directions in the residual stream activation space.

## Repository Structure

```
llm-refusal/
├── CLAUDE.md
├── pyproject.toml              # setuptools, Python >=3.11
└── llm-refusal/
    ├── scratch.py              # Core implementation (~1060 lines)
    ├── prompts.py              # Train + eval prompt datasets
    ├── run_experiment.py       # CLI runner (JSON config via argv[1])
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

# Run the main experiment (edit config dict in scratch.py)
python llm-refusal/scratch.py

# Run via CLI runner (pass config as JSON, avoids editing scratch.py)
python llm-refusal/run_experiment.py '{"model_name": "Qwen/Qwen1.5-1.8B-Chat", "mode": "search"}'
```

## Configuration

Both `scratch.py` (inline dict) and `run_experiment.py` (JSON via argv[1]) use the same config schema:

```python
config = {
    "model_name": "Qwen/Qwen1.5-1.8B-Chat",  # HuggingFace model ID
    "torch_dtype": "auto",   # Passed to from_pretrained()
    "force_cpu": False,       # Override device selection to CPU
    "mode": "search",         # "search", "evaluate", or "eyeball"
    "layer": None,            # Required for evaluate/eyeball (int)
    "pos": None,              # Required for evaluate/eyeball (int, typically -1)
    "eval_tasks": [],         # lm-eval tasks: "mmlu", "arc_challenge", "gsm8k", "truthfulqa"
    "limit": 100              # Sample limit for lm-eval benchmark runs
}
```

**Modes:**
- **search** — Scans first 80% of layers to find the best refusal direction via multi-objective optimization. Produces plots in `plots/`.
- **evaluate** — Requires `layer` and `pos`. Runs quantitative evaluation: refusal rates (with/without intervention) and optional lm-eval benchmarks.
- **eyeball** — Requires `layer` and `pos`. Generates text side-by-side (baseline vs. intervened) for qualitative inspection.

## Architecture

The core implementation lives in `llm-refusal/scratch.py`.

### Data & Formatting
- **`prompts.py`** — `create_refusal_train_data()` and `create_refusal_eval_data()` return `(harmful_prompts, safe_prompts)`. Train and eval sets are disjoint. Both accept optional subset sizes and `random_seed`.
- **`PromptData`** — Dataclass for prompts + boolean labels (`True`=harmful). Has `.train_val_split()`.
- **`ChatPromptFormatter`** — Applies chat templates, tokenizes, and left-pads for batched generation. Auto-detects instruction-tuned models by checking for `-it`, `-instruct`, or `-chat` in the model name. Falls back to manual templates for Gemma, Qwen, Yi, Llama-2, Llama-3. Base models get pass-through with BOS prepending.

### Extraction
- **`ActivationExtractor`** — Extracts residual stream activations from all layers at specified positions using forward hooks.
- **`DifferenceInMeans`** — Computes direction vectors (mean_harmful − mean_safe) for each (layer, position) pair.

### Intervention
- **`ModelInterventionApplier`** — Registers forward hooks on transformer layers for `add`, `subtract`, or `ablate` (projection removal) interventions. Auto-detects layer structure (`.model.layers` or `.transformer.h`).
- **`InterventionStrategy`** (ABC) with two subclasses:
  - **`GlobalInterventionStrategy`** — Applies to all layers.
  - **`LayerSpecificInterventionStrategy`** — Applies only to the layer where the direction was found.
- **`DirectionVector`** — Dataclass holding vector tensor, layer index, position index, and score. Has `.unit` property.

### Evaluation
- **`LogOddsMetric`** — Log-odds ratio of refusal tokens vs. all other tokens from next-token logits.
- **`Three_Score_Evaluator`** — Evaluates directions on three metrics:
  - **bypass** (lower=better): Log-odds on harmful prompts with global ablation. Measures refusal elimination.
  - **induce** (higher=better): Log-odds on safe prompts with layer-specific addition. Measures if adding the direction induces refusal.
  - **kl** (lower=better): KL divergence on safe prompts with global ablation. Ensures ablation doesn't degrade performance.
- **`DirectionFinder`** — Multi-objective search: satisfices on induce > 0 and KL < 0.1, then minimizes bypass. Produces score-vs-layer plots.

### Testing & Orchestration
- **`InterventionSuite`** — Qualitative: generates text with/without interventions at various strengths.
- **`BigEvaluator`** — Quantitative: refusal rate (string-match against `DEFAULT_REFUSAL_PHRASES`) + lm-eval benchmarks (wraps model in `HFLM`).
- **`DirectionTestFramework`** — Main orchestrator. Loads model/tokenizer, initializes all components, dispatches to mode.
- **`generate_with_hooks()`** — Standalone autoregressive generation with KV-cache. Needed because `model.generate()` doesn't invoke registered forward hooks on every step.

### Module-Level Defaults
- `DEFAULT_REFUSAL_TOKENS` — 12 tokens checked by `LogOddsMetric`.
- `DEFAULT_REFUSAL_PHRASES` — 12 phrases checked by `BigEvaluator` for string-match refusal detection.
- `DEFAULT_SEARCH_CONFIG` — `layer_cutoff_frac: 0.8`, `induce_threshold: 0`, `kl_threshold: 0.1`.

## Device Handling

`DirectionTestFramework` selects device: CUDA > MPS > CPU (overridable via `force_cpu: True`). If a model loads as `float16` on MPS, it is upcast to `bfloat16` to prevent attention overflow (NaN/Inf).

## Test Structure

- `test_unit.py` — Unit tests with mocks for data splitting, hooks, interventions, metrics.
- `test_generation.py` — Integration tests parametrized across: `google/gemma-3-1b-pt`, `google/gemma-3-1b-it` (expected to fail), `openai-community/gpt2-xl`.
- `test_smoke.py` — End-to-end pipeline smoke test on `google/gemma-3-270m`.
- `conftest.py` — Shared fixtures (`model_and_tokenizer` parametrized across test models, session-scoped), `--run-smoke` CLI option, mock factories.

## Output Artifacts

- `results/` — Log files: `{ModelShortName}-{mode}[-L{layer}][-P{pos}].log`
- `plots/` — PNGs: `{ModelShortName}-induce_score_vs_layer.png`, `{ModelShortName}-bypass_score_vs_layer.png`
