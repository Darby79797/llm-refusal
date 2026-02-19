"""Lightweight experiment runner — pass config via CLI to avoid editing scratch.py."""
import sys
import os
import json
import logging

# Must be set before importing scratch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import scratch

if __name__ == "__main__":
    config = json.loads(sys.argv[1])

    # Setup logging to file
    model_short_name = config["model_name"].split('/')[-1]
    log_filename_parts = [
        model_short_name,
        config['mode'],
        f"L{config['layer']}" if config.get('layer') is not None else '',
        f"P{config['pos']}" if config.get('pos') is not None else ''
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
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    scratch.logger.info(f"Logging to: {log_path}")
    scratch.logger.info(f"Config: {config}")
    scratch.main(config)
