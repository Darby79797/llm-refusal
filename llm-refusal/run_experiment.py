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
        choices=["search", "evaluate", "eyeball"],
        help="Experiment mode",
    )
    parser.add_argument("--concept", default="refusal", help="Concept to study (default: refusal)")
    parser.add_argument("--layer", type=int, default=None, help="Layer index (required for evaluate/eyeball)")
    parser.add_argument("--pos", type=int, default=None, help="Position index (required for evaluate/eyeball)")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype for from_pretrained (default: auto)")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU device")
    parser.add_argument("--eval-tasks", nargs="*", default=[], help="lm-eval tasks: mmlu, arc_challenge, gsm8k, truthfulqa")
    parser.add_argument("--limit", type=int, default=100, help="Sample limit for lm-eval benchmarks (default: 100)")
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
            "layer": args.layer,
            "pos": args.pos,
            "torch_dtype": args.torch_dtype,
            "force_cpu": args.force_cpu,
            "eval_tasks": args.eval_tasks,
            "limit": args.limit,
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
