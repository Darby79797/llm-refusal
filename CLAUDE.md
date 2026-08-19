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
- Hardcoded: `max_positions=auto` (from assistant prefix tokens), val split 20% (`random_state=39`), `max_new_tokens=64`. Generation batch size is `--gen-batch-size` (default 2) — greedy decoding is only bit-reproducible at a fixed batch size, so keep it constant across runs you compare (see `future_plans.md` for measured throughput/stability). `layer_cutoff_frac` is per-concept (0.65 refusal/hedging/empathy, 0.80 sycophancy/arditi_exact)

## Gotchas

- **Padding: right, and never hand-roll the boundary index.** The pipeline right-pads. Real tokens occupy `0..true_len-1`, so the generation boundary is `attention_mask.sum(-1)-1` and activations sit at `true_len+pos_idx` — expressions that are silently *wrong* under left padding (they address a mid-prompt token). Always use `formatting.last_real_token_indices()` / `assert_right_padded()`, which validate the invariant rather than assume it; don't reintroduce `attention_mask.sum(dim=1) - 1` inline. `format_batch()` re-asserts `padding_side='right'` on every call because the tokenizer is shared and `lm_eval` mutates it. (Left padding is not inherently broken — RoPE is shift-invariant, and Arditi left-pad and read index -1. The bug was mixing the two conventions. See `RESULTS.md` "Two Padding Bugs".)
- **Never generate with `model.generate()`**: it reads the next token from the batch's last column, which under right padding is a pad slot for every short row — the model then decodes from `<|endoftext|>` and drifts off-task, costing up to 24pp of measured refusal rate. It also applies the model's shipped `generation_config` (Qwen2.5 ships `repetition_penalty` 1.05–1.1, a logits processor that fires even with `do_sample=False`), so it isn't greedy decoding. Use `generation.generate_with_hooks()`. Guarded by `test_unit.py` and `test_generation.py`.
- **Generation batch size is part of the measurement**: `--gen-batch-size` (default 2) is shared by every condition and the filtering pass. In bf16, batch shape changes reduction order and greedy argmax is discontinuous, so rates shift by a few pp across batch sizes; fp32 is exactly batch-invariant. Keep it fixed across runs you compare.
- **Hook cleanup**: `clear_interventions()` after every intervention (try/finally). Leaked hooks corrupt results.
- **Env vars**: `TOKENIZERS_PARALLELISM=false`, `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing ML libs (done in `run_experiment.py`/`conftest.py`). Framework auto-upcasts float16→bfloat16 on MPS.
- **Llama-2 chat template**: HF's ignores `add_generation_prompt`; we override with `[INST] {x} [/INST] ` (trailing space) so pos -1 is the generation boundary. Without it, induction fails completely.
- **LlamaGuard 2 not 3**: LG3 flags by topic (80-93% FP on refusals); LG2 flags by compliance.
- **Detection hierarchy**: API judge → `detection_fn` heuristic → phrase matching. Study model never self-judges.
- **Framework clobbers its own logs**: every run also writes `results/<model>-<mode>-L<layer>-P<pos>.log` (and `<model>-cross_concept.log`) internally, silently overwriting any previous run with the same model/mode/layer/pos. Historical logs you want to keep must be renamed (e.g. `rerun-*`) before re-running.
