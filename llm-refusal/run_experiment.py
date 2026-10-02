"""CLI runner for direction experiments."""
import os
import re
import sys
import json
import logging
import argparse

from env import setup_process_env; setup_process_env()  # before torch is imported

# Cached models run offline (transformers otherwise calls the Hub API on every
# tokenizer load). Must precede the transformers import below.
from hf_offline import use_offline_for_run  # noqa: E402
if "--json" not in sys.argv:
    _model = next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a == "--model"), None)
    # lm-eval tasks need their datasets; offline too once those are cached (checked).
    # --arditi-evals is local: Alpaca prompts from data/, LlamaGuard via Ollama,
    # JailbreakBench only with an API key.
    _tasks = []
    if "--eval-tasks" in sys.argv:
        for a in sys.argv[sys.argv.index("--eval-tasks") + 1:]:
            if a.startswith("--"):
                break
            _tasks.append(a)
    _offline = use_offline_for_run(_model, _tasks)

from framework import main
from evaluation import CONDITIONS, DEFAULT_CONDITIONS
from batching import parse_batch_size

logger = logging.getLogger(__name__)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run a direction-finding experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --model Qwen/Qwen1.5-1.8B-Chat --mode search
  %(prog)s --model Qwen/Qwen1.5-1.8B-Chat --mode evaluate --layer 10 --pos -1
  %(prog)s --json '{"model_name": "Qwen/Qwen1.5-1.8B-Chat", "mode": "search"}'
  %(prog)s --json config.json
""",
    )

    parser.add_argument("--model", dest="model_name", help="HuggingFace model ID")
    parser.add_argument(
        "--mode",
        choices=["search", "evaluate", "eyeball", "cross_concept", "caa", "capability", "rank1", "regrow"],
        help="Experiment mode",
    )
    parser.add_argument("--concept", default="refusal", help="Concept to study (default: refusal)")
    parser.add_argument("--concepts", default=None, help="Comma-separated concept names for cross_concept mode")
    parser.add_argument("--layer", type=int, default=None, help="Layer index (required for evaluate/eyeball)")
    parser.add_argument("--pos", type=int, default=None, help="Position index (required for evaluate/eyeball)")
    parser.add_argument("--caa-behaviors", default=None,
                        help="caa mode: comma-separated CAA behaviors (default: all 7)")
    parser.add_argument("--caa-layers", type=int, nargs="+", default=None,
                        help="caa mode: layers to sweep, in CAA's block-output convention (default: all)")
    parser.add_argument("--caa-multipliers", type=float, nargs="+", default=[-1.0, 1.0],
                        help="caa mode: steering multipliers (default: -1 1; 0 is always run as the baseline)")
    parser.add_argument("--caa-reuse-vectors", action="store_true",
                        help="caa mode: load results/caa/<model>-ab-vectors.pt instead of recomputing vectors")
    parser.add_argument("--caa-tag", default="",
                        help="caa mode: suffix for the output (results/caa/<model>-ab-<tag>.json), e.g. 'mult'")
    parser.add_argument("--objective", choices=["remove", "induce", "null"], default="remove",
                        help="rank1 mode: distil directional ablation (remove), layer addition (induce), or the "
                             "clean model itself (null: the self-distillation control)")
    parser.add_argument("--rank1-examples-file", default=None,
                        help="rank1 mode: train on a saved (prompt, completion) set (<stem>-examples.json from an "
                             "earlier rank1 run) instead of generating targets from the current model")
    parser.add_argument("--adapter-layers", nargs="+", default=None,
                        help="rank1 mode: layers to adapt (default: the direction's layer - 1; 'all' = every "
                             "layer before it)")
    parser.add_argument("--adapter-modules", nargs="+", default=None,
                        help="rank1 mode: projections to adapt (default: down_proj)")
    parser.add_argument("--regrow-targets", choices=["writers", "readers"], default="writers",
                        help="regrow mode: LoRA on residual writers (o_proj/down_proj) or readers (q/k/v/gate/up)")
    parser.add_argument("--n-refusal-examples", type=int, default=0,
                        help="regrow mode: harmful->refusal examples mixed into the benign Alpaca data")
    parser.add_argument("--refusal-repeat", type=int, default=1,
                        help="regrow mode: duplicate each refusal example this many times (exposures vs distinct examples)")
    parser.add_argument("--lora-rank", type=int, default=8, help="regrow mode: adapter rank")
    parser.add_argument("--train-steps", type=int, default=200, help="rank1/regrow: optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="rank1/regrow: Adam learning rate")
    parser.add_argument("--train-batch-size", type=int, default=8, help="rank1/regrow: examples per step")
    parser.add_argument("--train-micro-batch-size", type=int, default=None,
                        help="rank1/regrow: rows per forward/backward, gradients accumulated to "
                             "--train-batch-size (same update; less activation memory). Default: no split")
    parser.add_argument("--eval-every", type=int, default=50, help="regrow mode: steps between refusal checks")
    parser.add_argument("--seed", type=int, default=0, help="rank1/regrow: adapter init and data order")
    parser.add_argument("--run-tag", default="", help="suffix for the output files (rank1/regrow tag; search/evaluate/"
                                                       "capability: keeps the run from overwriting the untagged one)")
    parser.add_argument("--adapter-file", default=None,
                        help="run inside a saved rank1/regrow adapter set: the stem of results/finetune/<...>.json "
                             "and -adapters.pt (every mode)")
    parser.add_argument("--adapter-variant", choices=["full", "u_perp", "u_rhat"], default="full",
                        help="with --adapter-file: as trained, with r̂ projected out of U (the inhibitor), "
                             "or only U's r̂ part. r̂ = the saved direction (--edit-direction-file)")
    parser.add_argument("--orthogonalize-first", action="store_true",
                        help="run the whole mode inside the weight edit of the saved direction "
                             "(--edit-direction-file, default results/<model>-<concept>-direction)")
    parser.add_argument("--edit-direction-file", default=None,
                        help="direction (path without .pt/.json) for --orthogonalize-first / --adapter-variant")
    parser.add_argument("--edit-layers", type=int, nargs="+", default=None,
                        help="restrict the weight edit (orthogonalized condition, --orthogonalize-first) to these "
                             "blocks' o_proj/down_proj (default: all)")
    parser.add_argument("--no-edit-embedding", dest="edit_embedding", action="store_false",
                        help="leave the token embedding out of the weight edit")
    parser.set_defaults(edit_embedding=True)
    parser.add_argument("--direction-file", default=None,
                        help="evaluate mode: evaluate a saved direction (path without .pt/.json) instead of "
                             "recomputing one at --layer/--pos, e.g. a CAA vector")
    parser.add_argument("--torch-dtype", default="auto",
                        help="Compute dtype: auto (default: the checkpoint's stored dtype, bf16 for "
                             "Qwen2.5/Llama-3, fp16 for Llama-2, upcast to bf16 on MPS), bfloat16, float16, or "
                             "float32 (debugging: exactly batch-invariant on MPS). Refuses to load if the "
                             "weights won't fit.")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU device")
    parser.add_argument("--eval-tasks", nargs="*", default=[], help="lm-eval tasks: mmlu, arc_challenge, gsm8k, truthfulqa")
    parser.add_argument("--limit", type=int, default=100, help="Sample limit for lm-eval benchmarks (default: 100)")
    parser.add_argument("--judge-api-base", default=None, help="OpenAI-compatible API base URL for judge (env: JUDGE_API_BASE)")
    parser.add_argument("--judge-api-key", default=None, help="API key for judge endpoint (env: JUDGE_API_KEY)")
    parser.add_argument("--judge-model", default=None, help="Model name at the judge API (env: JUDGE_MODEL)")
    parser.add_argument("--arditi-evals", action="store_true", help="Enable Arditi-style evals (LlamaGuard2, JailbreakBench, Alpaca CE loss)")
    parser.add_argument("--llamaguard-api-base", default=None, help="API base URL for LlamaGuard2 (env: LLAMAGUARD_API_BASE)")
    parser.add_argument("--llamaguard-api-key", default=None, help="API key for LlamaGuard2 (env: LLAMAGUARD_API_KEY)")
    parser.add_argument("--llamaguard-model", default=None, help="Model name for LlamaGuard2 (env: LLAMAGUARD_MODEL)")
    parser.add_argument("--jbb-api-key", default=None, help="Together AI API key for JailbreakBench (env: JBB_API_KEY)")
    parser.add_argument("--alpaca-max-prompts", type=int, default=500, help="Max prompts for Alpaca CE loss (default: 500)")
    parser.add_argument("--no-filter-prompts", dest="filter_prompts", action="store_false",
                        help="Disable filtering train prompts by actual model behavior (on by default)")
    parser.set_defaults(filter_prompts=True)
    parser.add_argument("--strength", type=float, default=1.0,
                        help="Intervention strength for evaluate mode (default: 1.0)")
    parser.add_argument("--induce-mode", choices=["single_layer", "all_layers"], default="single_layer",
                        help="Induce score mode for search selection: single_layer (default) or all_layers (Arditi-style)")
    parser.add_argument("--gen-batch-size", type=parse_batch_size, default="auto",
                        help="Batch size for generation and scoring: an int, or 'auto' (default). 'auto' picks "
                             "the largest power of two (<=64) whose estimated peak memory fits half of "
                             "(device memory - weights). It is deterministic for a given machine, model, prompt "
                             "set and generation length, and is logged. In bf16 on MPS, results shift by a few "
                             "pp across batch sizes (and bs<=8 is no faster than bs=2), so pin an int to "
                             "reproduce an earlier run.")
    parser.add_argument("--conditions", nargs="+", default=None,
                        choices=list(CONDITIONS) + ["all"],
                        help="Evaluate-mode intervention conditions (default: %s). 'all' adds "
                             "global_addition (always degenerate) and layer_specific_subtraction. "
                             "Each costs one generation pass." % ", ".join(DEFAULT_CONDITIONS))
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Tokens generated per response in evaluate mode (default: 64; Arditi's "
                             "safety evaluation uses 512). The prompt-filtering pass always uses 64.")
    parser.add_argument("--no-filter-cache", dest="filter_cache", action="store_false",
                        help="Regenerate the prompt-filtering pass instead of reusing the cached "
                             "result for this model/concept/dtype/batch size/prompt set")
    parser.set_defaults(filter_cache=True)
    parser.add_argument(
        "--json",
        dest="json_config",
        metavar="JSON",
        help="JSON config string or path to a JSON file (overrides all other flags)",
    )
    return parser


def _namespace_to_config(args):
    """Turn a parsed argparse.Namespace into the final config dict shape.

    Used both for real CLI invocations and (via an empty/synthetic argv) to
    derive the canonical set of config defaults, so the two never drift apart.
    """
    return {
        "model_name": args.model_name,
        "mode": args.mode,
        "concept": args.concept,
        "concepts": [s.strip() for s in args.concepts.split(",")] if args.concepts else [],
        "layer": args.layer,
        "pos": args.pos,
        "torch_dtype": args.torch_dtype,
        "force_cpu": args.force_cpu,
        "eval_tasks": args.eval_tasks,
        "limit": args.limit,
        "judge_api_base": args.judge_api_base or os.environ.get("JUDGE_API_BASE"),
        "judge_api_key": args.judge_api_key or os.environ.get("JUDGE_API_KEY"),
        "judge_model": args.judge_model or os.environ.get("JUDGE_MODEL"),
        "arditi_evals": args.arditi_evals,
        "llamaguard_api_base": args.llamaguard_api_base or os.environ.get("LLAMAGUARD_API_BASE"),
        "llamaguard_api_key": args.llamaguard_api_key or os.environ.get("LLAMAGUARD_API_KEY"),
        "llamaguard_model": args.llamaguard_model or os.environ.get("LLAMAGUARD_MODEL"),
        "jbb_api_key": args.jbb_api_key or os.environ.get("JBB_API_KEY"),
        "alpaca_max_prompts": args.alpaca_max_prompts,
        "filter_prompts": args.filter_prompts,
        "induce_mode": args.induce_mode,
        "strength": args.strength,
        "gen_batch_size": args.gen_batch_size,
        "conditions": (list(CONDITIONS) if args.conditions and "all" in args.conditions
                       else args.conditions),
        "filter_cache": args.filter_cache,
        "max_new_tokens": args.max_new_tokens,
        "caa_behaviors": [s.strip() for s in args.caa_behaviors.split(",")] if args.caa_behaviors else None,
        "caa_layers": args.caa_layers,
        "caa_multipliers": args.caa_multipliers,
        "direction_file": args.direction_file,
        "caa_reuse_vectors": args.caa_reuse_vectors,
        "caa_tag": args.caa_tag,
        "objective": args.objective,
        "adapter_layers": ([int(x) for x in args.adapter_layers] if args.adapter_layers and args.adapter_layers != ["all"]
                           else args.adapter_layers and "all"),
        "adapter_modules": args.adapter_modules,
        "rank1_examples_file": args.rank1_examples_file,
        "regrow_targets": args.regrow_targets,
        "n_refusal_examples": args.n_refusal_examples,
        "refusal_repeat": args.refusal_repeat,
        "lora_rank": args.lora_rank,
        "train_steps": args.train_steps,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "train_micro_batch_size": args.train_micro_batch_size,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "run_tag": args.run_tag,
        "adapter_file": args.adapter_file,
        "adapter_variant": args.adapter_variant,
        "orthogonalize_first": args.orthogonalize_first,
        "edit_direction_file": args.edit_direction_file,
        "edit_layers": args.edit_layers,
        "edit_embedding": args.edit_embedding,
    }


def build_default_config(parser=None):
    """Full config dict of argparse defaults, keyed exactly like the final config.

    This is the single source of truth for config defaults: derived by parsing
    an empty argv against the real parser, so adding/changing a CLI flag keeps
    this in sync automatically (no hand-duplicated defaults dict to maintain).
    """
    if parser is None:
        parser = _build_parser()
    return _namespace_to_config(parser.parse_args([]))


def apply_config_defaults(json_dict, parser=None):
    """Overlay a user-supplied (e.g. --json) config dict on top of the argparse defaults.

    Raises ValueError if `json_dict` contains any key that isn't a recognized
    config key (catches typos that would otherwise be silently ignored).
    """
    defaults = build_default_config(parser)
    unknown = sorted(set(json_dict) - set(defaults))
    if unknown:
        raise ValueError(
            f"Unknown config key(s): {unknown}. Valid keys: {sorted(defaults)}"
        )
    merged = dict(defaults)
    merged.update(json_dict)
    return merged


def parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Build config from --json or from individual flags
    if args.json_config is not None:
        text = args.json_config
        if os.path.isfile(text):
            with open(text) as f:
                json_dict = json.load(f)
        else:
            json_dict = json.loads(text)
        try:
            config = apply_config_defaults(json_dict, parser)
        except ValueError as e:
            parser.error(str(e))
        if config["model_name"] is None or config["mode"] is None:
            parser.error("'model_name' and 'mode' are required (via --model/--mode flags or in the --json config)")
    else:
        if args.model_name is None or args.mode is None:
            parser.error("--model and --mode are required unless --json is provided")
        config = _namespace_to_config(args)

    return config


def setup_logging(config):
    model_short_name = config["model_name"].split("/")[-1]
    # The concept is part of the name: without it, a sycophancy/hedging search
    # silently overwrote the refusal search log for the same model.
    log_filename_parts = [
        config.get("run_tag") or "",
        model_short_name,
        config["concept"] if config["mode"] not in ("cross_concept", "caa") else "",
        config["mode"],
        f"L{config['layer']}" if config.get("layer") is not None else "",
        f"P{config['pos']}" if config.get("pos") is not None else "",
        # Same suffix as the generations file, so a 512-token run doesn't
        # overwrite the 64-token log at the same coordinates.
        f"T{config['max_new_tokens']}" if config.get("max_new_tokens", 64) != 64 else "",
    ]
    log_filename = "-".join(filter(None, log_filename_parts)) + ".log"

    log_dir = "results"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)

    return log_path


def check_variant_tag(config):
    """A model-variant run (saved adapter, edit-first, restricted edit) must carry a run tag,
    or it overwrites the untagged results, including the direction file every variant
    reads r̂ from. Tags must be lowercase [a-z][a-z0-9]* so tools/runs.py indexes them."""
    variant = (config.get("adapter_file") or config.get("orthogonalize_first") or config.get("edit_layers")
               or not config.get("edit_embedding", True))
    tag = config.get("run_tag") or ""
    if variant and config["mode"] in ("search", "evaluate", "capability", "rank1", "regrow") and not tag:
        raise SystemExit("--run-tag is required with --adapter-file / --orthogonalize-first / --edit-layers / "
                         "--no-edit-embedding (the run would overwrite the untagged results)")
    if tag and not re.fullmatch(r"[a-z][a-z0-9]*", tag):
        raise SystemExit(f"--run-tag {tag!r} must match [a-z][a-z0-9]* (tools/runs.py reads it as a variant prefix)")


if __name__ == "__main__":
    config = parse_args()
    check_variant_tag(config)
    log_path = setup_logging(config)

    logger.info(f"Logging to: {log_path}")
    logger.info(f"Config: {config}")
    logger.info("Hugging Face offline mode: " + ", ".join(
        f"{v}={os.environ.get(v, 'unset')}" for v in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")))
    main(config)
