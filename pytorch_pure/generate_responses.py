"""Generate responses with angular steering using pure PyTorch.

This script replaces the vLLM-based generate_responses.py with a pure PyTorch
implementation using HuggingFace transformers and manual hooks.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import generate_completions, get_angular_steering_output_hook, get_input_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses with angular steering (pure PyTorch)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="./output",
        help="Directory containing steering configs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save generated responses",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "jp"],
        help="Language for datasets",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for generation (lower if OOM)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=10,
        help="Rotation angle step (10 for 36 angles, 30 for 12 angles)",
    )
    parser.add_argument(
        "--adaptive-mode",
        type=int,
        default=1,
        help="Adaptive steering mode (0: all, 1: conditional on harmful direction)",
    )
    parser.add_argument(
        "--strategy-filter",
        type=str,
        default=None,
        help="Filter configs by strategy (e.g., 'max_sim', 'max_norm')",
    )

    args = parser.parse_args()

    # Setup paths
    model_name = args.model.split("/")[-1]
    config_path = Path(args.config_dir) / model_name
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Get module dict for hook registration
    module_dict = dict(model.named_modules())

    # Load test data
    logger.info(f"Loading test data ({args.language})...")
    _, data_test = get_input_data("harmful", args.language)
    logger.info(f"Loaded {len(data_test)} test samples")

    # Generate baseline (no steering)
    baseline_file = output_path / f"harmful-{args.language}-baseline.json"
    if not baseline_file.exists():
        logger.info("Generating baseline responses (no steering)...")
        baseline_responses = generate_completions(
            model=model,
            instructions=data_test,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_tokens,
        )
        # Extract just the responses
        baseline_responses = [item["response"] for item in baseline_responses]

        with open(baseline_file, "w") as f:
            json.dump(baseline_responses, f, indent=4)
        logger.info(f"Saved baseline to {baseline_file}")
    else:
        logger.info(f"Baseline already exists: {baseline_file}")

    # Find all steering configs
    if not config_path.exists():
        logger.error(f"Config directory not found: {config_path}")
        logger.error("Please run extract_directions.py first!")
        return

    steering_configs = list(config_path.glob("steering_config-*.npy"))
    if not steering_configs:
        logger.error(f"No steering configs found in {config_path}")
        logger.error("Please run extract_directions.py first!")
        return

    logger.info(f"Found {len(steering_configs)} steering config(s)")

    # Process each config
    for config_file in steering_configs:
        # Parse filename
        stem = config_file.stem  # e.g., "steering_config-en-max_sim_15_mid-pca_0"
        parts = stem.split("-")

        if len(parts) < 3:
            logger.warning(f"Skipping {config_file}: unexpected filename format")
            continue

        lang_code = parts[1]
        direction_info = parts[2]  # e.g., "max_sim_15_mid"

        # Filter by language
        if lang_code != args.language:
            logger.info(f"Skipping {config_file.name}: language mismatch")
            continue

        # Filter by strategy if specified
        if args.strategy_filter and args.strategy_filter not in direction_info:
            logger.info(f"Skipping {config_file.name}: strategy filter")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {config_file.name}")
        logger.info(f"{'='*60}")

        # Load config
        config = np.load(config_file, allow_pickle=True).item()

        # Generate responses at different angles
        steered_responses = {}

        for degree in range(0, 360, args.angle_step):
            logger.info(f"  Generating at {degree}° rotation...")

            # Setup steering hooks
            output_hooks = [
                (
                    module_dict[module_name],
                    get_angular_steering_output_hook(
                        steering_config=steering_config,
                        target_degree=degree,
                        adaptive_mode=args.adaptive_mode,
                    ),
                )
                for module_name, steering_config in config.items()
            ]

            # Generate
            completions = generate_completions(
                model=model,
                instructions=data_test,
                tokenizer=tokenizer,
                fwd_hooks=output_hooks,
                batch_size=args.batch_size,
                max_new_tokens=args.max_tokens,
            )

            # Extract responses
            responses = [item["response"] for item in completions]
            steered_responses[str(degree)] = responses

        # Save responses with adaptive mode suffix to match parent's expected format
        adaptive_mode_label = (
            "rotated" if args.adaptive_mode == 0 else f"adaptive_{args.adaptive_mode}"
        )
        output_file = (
            output_path
            / f"harmful-{args.language}-{direction_info}-pca_0-{adaptive_mode_label}.json"
        )
        with open(output_file, "w") as f:
            json.dump(steered_responses, f, indent=4)

        logger.info(f"  Saved to: {output_file}")

    logger.info("\n✓ Generation complete!")
    logger.info(f"  Output directory: {output_path}")


if __name__ == "__main__":
    main()
