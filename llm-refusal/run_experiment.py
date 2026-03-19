"""CLI runner for direction experiments."""
import os
import sys
import json
import logging
import argparse

# Must be set before importing any ML libraries
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

from framework import main

logger = logging.getLogger(__name__)


def parse_args(argv=None):
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
        choices=["search", "evaluate", "eyeball", "cross_concept"],
        help="Experiment mode",
    )
    parser.add_argument("--concept", default="refusal", help="Concept to study (default: refusal)")
    parser.add_argument("--concepts", default=None, help="Comma-separated concept names for cross_concept mode")
    parser.add_argument("--layer", type=int, default=None, help="Layer index (required for evaluate/eyeball)")
    parser.add_argument("--pos", type=int, default=None, help="Position index (required for evaluate/eyeball)")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype for from_pretrained (default: auto)")
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
    parser.add_argument(
        "--json",
        dest="json_config",
        metavar="JSON",
        help="JSON config string or path to a JSON file (overrides all other flags)",
    )

    args = parser.parse_args(argv)

    # Build config from --json or from individual flags
    if args.json_config is not None:
        text = args.json_config
        if os.path.isfile(text):
            with open(text) as f:
                config = json.load(f)
        else:
            config = json.loads(text)
    else:
        if args.model_name is None or args.mode is None:
            parser.error("--model and --mode are required unless --json is provided")
        config = {
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
        }

    return config


def setup_logging(config):
    model_short_name = config["model_name"].split("/")[-1]
    log_filename_parts = [
        model_short_name,
        config["mode"],
        f"L{config['layer']}" if config.get("layer") is not None else "",
        f"P{config['pos']}" if config.get("pos") is not None else "",
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


if __name__ == "__main__":
    config = parse_args()
    log_path = setup_logging(config)

    logger.info(f"Logging to: {log_path}")
    logger.info(f"Config: {config}")
    main(config)
