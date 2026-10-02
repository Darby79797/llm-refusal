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

# Modes: search, evaluate, eyeball, cross_concept, caa, capability, rank1, regrow. --help for all flags; --json '{...}' overrides flags.
.venv/bin/python3 llm-refusal/run_experiment.py --model Qwen/Qwen2.5-3B-Instruct --mode search
.venv/bin/python3 llm-refusal/run_experiment.py --model ... --mode evaluate --layer 21 --pos -4

# Looking at results (no model): start here instead of grepping logs
.venv/bin/python3 llm-refusal/tools/look.py digest                  # every run: rates+CIs, log-odds, flags
.venv/bin/python3 llm-refusal/tools/look.py gens 3B sycophancy --flip baseline:global_ablation --rescore
.venv/bin/python3 llm-refusal/tools/look.py search 3.1 refusal --metric induce
.venv/bin/python3 llm-refusal/tools/report.py --root results --root results/sweep4   # → results/report/index.html
```

- `--concept`: `refusal` (default), `refusal_arditi`, `refusal_arditi_exact`, `sycophancy`, `sycophancy_response` (response contrast; evaluate it with `--direction-file`), `sycophancy_neutral`, `hedging`, `hedging_v2`, `opinion_avoidance`, `empathy`
- Model variants for any mode: `--adapter-file STEM [--adapter-variant full|u_perp|u_rhat]` installs a saved rank1/regrow adapter set; `--orthogonalize-first` runs inside the weight edit of the saved direction (`--edit-layers`, `--no-edit-embedding` restrict it). `--run-tag TAG` prefixes search/evaluate outputs (`results/TAG-<model>-...`, lowercase) so they don't overwrite the untagged run.
- Limb scripts (2026-10-02, each loads one model; `--help`): `scripts/edit_cost_sweep.py` (refusal + CE per partial edit / adapter), `inhibitor_safety.py` (LlamaGuard on the inhibitor), `direction_identity.py` (regrown vs inhibitor directions), `trajectory.py` (r̂ projection per layer), `category_directions.py`, `jailbreak_projection.py`, `caa_open_ended.py` (qwen3:4b judge via Ollama), `concept_baseline.py`; `bypass_vs_induce.py` needs no model. Queue pattern: `results/QUEUE-*.sh` (serial, resumable via the progress file).
- Hardcoded: `max_positions=auto` (from assistant prefix tokens), val split 20% (`random_state=39`). Evaluate generation length is `--max-new-tokens` (default 64; the filtering pass always uses 64). Batch size is `--gen-batch-size` (default `auto`; see gotcha below). `layer_cutoff_frac` is per-concept (0.65 refusal/hedging/empathy, 0.80 sycophancy/arditi_exact)

## Gotchas

- **Padding: right, and never hand-roll the boundary index.** The pipeline right-pads. Real tokens occupy `0..true_len-1`, so the generation boundary is `attention_mask.sum(-1)-1` and activations sit at `true_len+pos_idx` — expressions that are silently *wrong* under left padding (they address a mid-prompt token). Always use `formatting.last_real_token_indices()` / `assert_right_padded()`, which validate the invariant rather than assume it; don't reintroduce `attention_mask.sum(dim=1) - 1` inline. `format_batch()` re-asserts `padding_side='right'` on every call because the tokenizer is shared and `lm_eval` mutates it. (Left padding is not inherently broken — RoPE is shift-invariant, and Arditi left-pad and read index -1. The bug was mixing the two conventions. See `RESULTS.md` "Two Padding Bugs".)
- **Never generate with `model.generate()`**: it reads the next token from the batch's last column, which under right padding is a pad slot for every short row — the model then decodes from `<|endoftext|>` and drifts off-task, costing up to 24pp of measured refusal rate. It also applies the model's shipped `generation_config` (Qwen2.5 ships `repetition_penalty` 1.05–1.1, a logits processor that fires even with `do_sample=False`), so it isn't greedy decoding. Use `generation.generate_with_hooks()`. Guarded by `test_unit.py` and `test_generation.py`.
- **Batch size is a few-pp noise source, not a setting to match; `auto` is the default**: in bf16 on MPS, batch shape changes the kernel and reduction order, so rates shift by a few pp across batch sizes; fp32 is exactly batch-invariant. A conclusion that flips with batch size isn't a result. Compare conditions within one run, or re-run *both* sides of a comparison at the same (auto) setting; don't pin a slow small batch to match an old log. `--gen-batch-size auto` (`batching.py`) picks the largest power of two (≤64) whose estimated memory fits half of (device memory − weights). It is deterministic per machine/model/prompt set/length and logged in the report header. (The 2026-08/09 sweeps used 2 or 8; an int pin exists only for exact reproduction.) On MPS bf16, bs ≤ 8 is no faster than bs=2; the speed-up starts at 16 (7-10× on 8B models). Entry points set MPS allocator watermarks (`env.py`); without them, 512-token runs swap.
- **Hook cleanup**: `clear_interventions()` after every intervention (try/finally). Leaked hooks corrupt results.
- **Compute dtype defaults to the checkpoint's own** (`--torch-dtype auto`): bf16 for Qwen2.5/Llama-3, fp16 for Llama-2 (upcast to bf16 on MPS, where fp16 attention overflows). That's how the models are served, and bf16 vs fp32 changed 7/1361 refusal labels, concentrated in borderline conditions like single-layer ablation on 0.5B. `float32` is opt-in for debugging (exactly batch-invariant on MPS; on CUDA the framework pins true fp32, no TF32). Loading refuses if the weights won't fit, instead of swapping. Metrics still upcast where it matters (logits → fp32, activation means in float64).
- **Env vars (`env.py`)**: `setup_process_env()` sets `TOKENIZERS_PARALLELISM=false`, `PYTORCH_ENABLE_MPS_FALLBACK=0` and the MPS allocator watermarks (`env.MPS_WATERMARKS`, the one source of truth). Every entry point calls it right after `sys.path.insert` and before importing torch/transformers: `from env import setup_process_env; setup_process_env()`. Framework auto-upcasts float16→bfloat16 on MPS.
- **Offline by default (`hf_offline.py`)**: online mode calls the Hub on every tokenizer load, so a network drop kills runs whose files are all cached. `run_experiment.py` goes offline when the model and any `--eval-tasks` datasets are cached (logged as "Hugging Face offline mode"); scripts and `conftest.py` call `offline_for_script()` (offline for a cached model ID in argv, or when the Hub is unreachable). New scripts must call it right after `sys.path.insert`, before importing transformers.
- **Llama-2 chat template**: HF's ignores `add_generation_prompt`; we override with `[INST] {x} [/INST] ` (trailing space) so pos -1 is the generation boundary. Without it, induction fails completely.
- **LlamaGuard 2 not 3**: LG3 flags by topic (80-93% FP on refusals); LG2 flags by compliance.
- **Detection hierarchy**: API judge → `detection_fn` heuristic → phrase matching. Study model never self-judges. Refusal phrases: `refusal`/`refusal_arditi` use `REFUSAL_PHRASES` (Arditi's list + non-apologetic refusals like "I do not provide"); `refusal_arditi_exact` keeps Arditi's exact list. Phrase matching is still a floor when an intervention rewords the refusal ("I should note that…"): read log-odds there.
- **Framework clobbers its own logs**: every run also writes `results/<model>-<concept>-<mode>-L<layer>-P<pos>[-T<tokens>].log` (and `<model>-cross_concept.log`) internally, silently overwriting any previous run with the same model/concept/mode/layer/pos/length (the `-T` suffix, for non-64-token runs, dates from 2026-09-30). (Before 2026-09, the name had no concept, so other-concept searches overwrote every refusal search log.) Historical logs you want to keep must be renamed (e.g. `rerun-*`) before re-running. Search also writes `results/<model>-<concept>-search-scores.csv` (every candidate) and overwrites `results/<model>-<concept>-direction.pt/.json`; evaluate writes `...-evaluate-L<l>-P<p>-generations.json` (every condition's responses).
