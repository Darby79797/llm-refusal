# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mechanistic interpretability research project reproducing and extending Arditi and Obeso's paper "Refusal in Language Models is Mediated by a Single Direction." The project investigates how LLMs implement refusal behaviors and whether they can be controlled via specific directions in the activation space.

## Commands

```bash
# Install dependencies
poetry install

# Run all tests (excludes smoke tests by default)
pytest llm-refusal/tests/

# Run only smoke tests (quick validation, uses google/gemma-3-270m)
pytest --run-smoke llm-refusal/tests/

# Run a single test file
pytest llm-refusal/tests/test_unit.py

# Run the main experiment
python llm-refusal/scratch.py
```

## Architecture

The core implementation is in `llm-refusal/scratch.py`. Key components:

### Data Flow
1. **PromptData** - Container for prompts with labels (True=harmful, False=safe)
2. **ChatPromptFormatter** - Handles chat templates and tokenization for different model families (Qwen, Gemma, Llama, Yi, GPT-2). Uses left-padding for batched generation.
3. **ActivationExtractor** - Extracts residual stream activations from specified layers/positions
4. **DifferenceInMeans** - Computes direction vectors between harmful and safe prompt activations

### Intervention System
- **ModelInterventionApplier** - Applies hooks to transformer layers for interventions (add, subtract, ablate)
- **DirectionVector** - Represents a found direction with layer index, position, and scores

### Evaluation
- **LogOddsMetric** - Computes log-odds of refusal tokens from logits
- **Three_Score_Evaluator** - Evaluates directions using three metrics:
  - **bypass** (lower=better): How much the direction eliminates refusal on harmful prompts
  - **induce** (higher=better): How much adding the direction induces refusal on safe prompts
  - **kl** (lower=better): KL divergence to ensure ablation doesn't hurt general performance
- **DirectionFinder** - Multi-objective search for best direction vectors (searches first 80% of layers)

### Testing Framework
- **InterventionSuite** - Qualitative testing via text generation with/without interventions
- **BigEvaluator** - Quantitative evaluation (refusal rates + standard benchmarks via lm-eval: MMLU, ARC, GSM8K, TruthfulQA)
- **DirectionTestFramework** - Main orchestrator with three modes: "search", "evaluate", "eyeball"

### Configuration
The main entry point configures runs via a config dict in `scratch.py`:
```python
config = {
    "model_name": "Qwen/Qwen1.5-1.8B-Chat",
    "mode": "evaluate",  # search, evaluate, or eyeball
    "layer": 15,
    "pos": -1,
    "eval_tasks": ["mmlu", "arc_challenge", "gsm8k", "truthfulqa"],
}
```

## Test Structure

- `test_unit.py` - Unit tests with mocks for data splitting, hooks, interventions, metrics
- `test_generation.py` - Integration tests for batched generation across multiple models (parametrized)
- `test_smoke.py` - Quick smoke tests on small models
- `conftest.py` - Shared fixtures including `model_and_tokenizer` parametrized across test models

## Output Artifacts

- `results/` - Log files named by model and mode (e.g., `Qwen1.5-1.8B-Chat-search-L15-P-1.log`)
- `plots/` - Induce/bypass score visualizations vs layers
