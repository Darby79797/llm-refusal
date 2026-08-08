# CLAUDE.md

Mechanistic interpretability: finding residual-stream directions that mediate learned behaviors (refusal, sycophancy, hedging). Extends Arditi & Obeso's "Refusal in Language Models is Mediated by a Single Direction". Concept-generic via a `ConceptDefinition` registry.

- `ARCHITECTURE.md` — module map, adding concepts/models, implementation notes
- `RESULTS.md` — best layers per model, key findings, right-padding bug fix
- `future_plans.md` — roadmap
- `results/`, `plots/` — local outputs (gitignored)

## Commands

Always use the venv (`.venv/bin/python3`, `.venv/bin/pytest`), never system python.

```bash
# Lightweight unit tests — safe default for verification:
pytest llm-refusal/tests/ --ignore=llm-refusal/tests/test_per_model.py --ignore=llm-refusal/tests/test_generation.py --ignore=llm-refusal/tests/test_smoke.py

# HEAVY — loads six models incl. three 7-8B (~15GB+ each on MPS). Only run when the
# user explicitly asks, and NEVER from more than one process/agent at a time (OOMs RAM):
pytest llm-refusal/tests/test_per_model.py -v  # padding/template/position_ids correctness
pytest llm-refusal/tests/ --run-smoke          # gemma-3-270m pipeline test

# Modes: search, evaluate, eyeball, cross_concept. --help for all flags; --json '{...}' overrides flags.
.venv/bin/python3 llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode search
.venv/bin/python3 llm-refusal/run_experiment.py --model ... --mode evaluate --layer 21 --pos -4
```

- `--concept`: `refusal` (default), `refusal_arditi`, `refusal_arditi_exact`, `sycophancy`, `sycophancy_neutral`, `hedging`, `empathy`
- Hardcoded: `max_positions=auto` (from assistant prefix tokens), val split 20% (`random_state=39`), `batch_size=2`, `max_new_tokens=64`. `layer_cutoff_frac` is per-concept (0.65 refusal/hedging/empathy, 0.80 sycophancy/arditi_exact)

## Gotchas

- **Right-padding is required**: left-padding corrupts logits for all padded prompts on RoPE models (see `RESULTS.md`). Every `model()` call must pass explicit `position_ids` (from `formatting.py`). Guarded by `test_per_model.py`.
- **Hook cleanup**: `clear_interventions()` after every intervention (try/finally). Leaked hooks corrupt results.
- **Env vars**: `TOKENIZERS_PARALLELISM=false`, `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing ML libs (done in `run_experiment.py`/`conftest.py`). Framework auto-upcasts float16→bfloat16 on MPS.
- **Llama-2 chat template**: HF's ignores `add_generation_prompt`; we override with `[INST] {x} [/INST] ` (trailing space) so pos -1 is the generation boundary. Without it, induction fails completely.
- **LlamaGuard 2 not 3**: LG3 flags by topic (80-93% FP on refusals); LG2 flags by compliance.
- **Detection hierarchy**: API judge → `detection_fn` heuristic → phrase matching. Study model never self-judges.
- **Framework clobbers its own logs**: every run also writes `results/<model>-<mode>-L<layer>-P<pos>.log` (and `<model>-cross_concept.log`) internally, silently overwriting any previous run with the same model/mode/layer/pos. Historical logs you want to keep must be renamed (e.g. `rerun-*`) before re-running.
