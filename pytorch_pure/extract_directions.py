"""Extract steering directions from model activations.

This script replaces the angular_steering.ipynb notebook with a pure PyTorch
implementation that extracts steering directions and saves them as .npy files.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    add_hooks,
    get_activations_hook,
    get_input_data,
    tokenize_instructions_fn,
)


def extract_activations(
    model,
    instructions: list[str],
    tokenizer,
    layers: list[int],
    positions: list[str],
    batch_size: int = 8,
):
    """Extract activations from specified layers and positions.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace model
    instructions : list[str]
        List of instructions
    tokenizer : PreTrainedTokenizer
        Tokenizer
    layers : list[int]
        Layer indices to extract from
    positions : list[str]
        Positions within layers: 'mid' (after attention) and/or 'post' (after MLP)
    batch_size : int
        Batch size for processing

    Returns
    -------
    activations : dict
        Dictionary mapping (layer, position) -> activations tensor
    """
    # Prepare cache
    cache = {}

    # Get module dict for hook registration
    module_dict = dict(model.named_modules())

    # Setup hooks for each layer and position
    hooks = []
    for layer_idx in layers:
        layer_name = f"model.layers.{layer_idx}"

        if "mid" in positions:
            # Hook input_layernorm OUTPUT (before self-attention)
            # This matches parent's "mid" position
            module_name = f"{layer_name}.input_layernorm"
            if module_name in module_dict:
                cache_key = f"layer_{layer_idx}_mid"
                hooks.append(
                    (
                        module_dict[module_name],
                        get_activations_hook(cache_key, cache, positions=[-1]),
                    )
                )

        if "post" in positions:
            # Hook post_attention_layernorm OUTPUT (after attention, before MLP)
            # This matches parent's "post" position
            module_name = f"{layer_name}.post_attention_layernorm"
            if module_name in module_dict:
                cache_key = f"layer_{layer_idx}_post"
                hooks.append(
                    (
                        module_dict[module_name],
                        get_activations_hook(cache_key, cache, positions=[-1]),
                    )
                )

    # Process in batches
    all_input_ids = []
    all_attention_masks = []

    for i in range(0, len(instructions), batch_size):
        batch_instructions = instructions[i : i + batch_size]
        tokenized = tokenize_instructions_fn(batch_instructions, tokenizer)
        all_input_ids.append(tokenized.input_ids)
        all_attention_masks.append(tokenized.attention_mask)

    # Run forward passes with hooks
    print(f"Extracting activations from {len(instructions)} samples...")
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            for input_ids, attention_mask in tqdm(
                zip(all_input_ids, all_attention_masks),
                total=len(all_input_ids),
                desc="Forward passes",
            ):
                _ = model(
                    input_ids=input_ids.to(model.device),
                    attention_mask=attention_mask.to(model.device),
                )

    # Organize activations
    # Convert cache to (layer, position, batch, hidden_dim) format
    activations = {}
    for key, value in cache.items():
        # key format: "layer_{idx}_{position}"
        activations[key] = value.squeeze(
            1
        )  # Remove token dim (we only kept last token)

    return activations


def compute_steering_directions(
    harmful_acts: dict, harmless_acts: dict, strategy: str = "both"
):
    """Compute steering directions from activations.

    Parameters
    ----------
    harmful_acts : dict
        Activations for harmful instructions, keyed by 'layer_{idx}_{position}'
    harmless_acts : dict
        Activations for harmless instructions, keyed by 'layer_{idx}_{position}'
    strategy : str
        'max_sim', 'max_norm', or 'both'

    Returns
    -------
    directions : dict
        Dictionary mapping strategy -> {'layer': int, 'position': str, 'first_direction': array, 'second_direction': array}
    """
    # Compute candidate directions for all layers/positions
    candidate_directions = {}
    norms = {}

    for key in harmful_acts.keys():
        harmful = harmful_acts[key].float()  # (batch, hidden_dim) - convert to float32
        harmless = harmless_acts[
            key
        ].float()  # (batch, hidden_dim) - convert to float32

        # Normalize each activation sample first (per-sample normalization)
        # This matches the parent implementation: harmful_acts / harmful_acts.norm(dim=-1, keepdim=True)
        harmful_normed = harmful / harmful.norm(dim=-1, keepdim=True)
        harmless_normed = harmless / harmless.norm(dim=-1, keepdim=True)

        # Compute mean of normalized activations
        harmful_mean = harmful_normed.mean(dim=0)
        harmless_mean = harmless_normed.mean(dim=0)

        # Normalize means again
        harmful_mean_norm = harmful_mean / harmful_mean.norm()
        harmless_mean_norm = harmless_mean / harmless_mean.norm()

        # Candidate direction (normalized difference)
        diff = harmful_mean_norm - harmless_mean_norm
        candidate_directions[key] = diff
        norms[key] = diff.norm()

    # Stack all candidate directions for PCA
    all_candidates = torch.stack(
        [candidate_directions[key] for key in sorted(candidate_directions.keys())]
    )

    # Get device from the first candidate
    device = all_candidates.device

    # Fit PCA on all candidate directions (already in float32)
    pca = PCA()
    pca.fit(all_candidates.cpu().numpy())
    second_direction_pca = torch.from_numpy(pca.components_[0]).to(device)

    # Select layer based on strategy
    directions = {}

    if strategy in ["max_sim", "both"]:
        # Max similarity: highest mean pairwise cosine similarity
        # Normalize all candidates
        candidates_normalized = {
            k: v / v.norm() for k, v in candidate_directions.items()
        }
        candidates_stack = torch.stack(
            [candidates_normalized[key] for key in sorted(candidates_normalized.keys())]
        )

        # Compute pairwise cosine similarities
        pairwise_cosine = candidates_stack @ candidates_stack.T
        mean_cosine = pairwise_cosine.mean(dim=-1)

        # Find layer with highest mean cosine similarity
        max_idx = mean_cosine.argmax().item()
        selected_key = sorted(candidate_directions.keys())[max_idx]

        # DEBUG: Print layer selection info
        print(f"\n  Max sim layer selection:")
        for i, key in enumerate(sorted(candidate_directions.keys())):
            layer_num = int(key.split("_")[1])
            marker = " ← SELECTED" if i == max_idx else ""
            print(f"    Layer {layer_num}: cosine={mean_cosine[i].item():.4f}{marker}")

        # Parse layer and position from key
        parts = selected_key.split("_")
        layer_idx = int(parts[1])
        position = parts[2]

        first_direction = candidate_directions[selected_key]
        first_direction = first_direction / first_direction.norm()

        # DO NOT orthogonalize second direction here - match parent behavior
        # Parent saves PCA component directly without orthogonalization
        # Orthogonalization happens at runtime in _get_rotation_args
        second_direction = second_direction_pca

        directions["max_sim"] = {
            "layer": layer_idx,
            "position": position,
            "first_direction": first_direction.cpu().numpy(),
            "second_direction": second_direction.cpu().numpy(),
        }

    if strategy in ["max_norm", "both"]:
        # Max norm: highest norm of candidate direction
        max_key = max(norms.keys(), key=lambda k: norms[k])

        # Parse layer and position from key
        parts = max_key.split("_")
        layer_idx = int(parts[1])
        position = parts[2]

        first_direction = candidate_directions[max_key]
        first_direction = first_direction / first_direction.norm()

        # DO NOT orthogonalize second direction here - match parent behavior
        # Parent saves PCA component directly without orthogonalization
        # Orthogonalization happens at runtime in _get_rotation_args
        second_direction = second_direction_pca

        directions["max_norm"] = {
            "layer": layer_idx,
            "position": position,
            "first_direction": first_direction.cpu().numpy(),
            "second_direction": second_direction.cpu().numpy(),
        }

    return directions


def main():
    parser = argparse.ArgumentParser(
        description="Extract steering directions from model activations"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save steering configs",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "jp"],
        help="Language for datasets",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=512,
        help="Number of samples to use for extraction",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for forward passes",
    )

    parser.add_argument(
        "--positions",
        type=str,
        nargs="+",
        default=["mid", "post"],
        choices=["mid", "post"],
        help="Positions to extract: mid (after attention) and/or post (after MLP)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="both",
        choices=["max_sim", "max_norm", "both"],
        help="Direction computation strategy",
    )

    args = parser.parse_args()

    # Create output directory
    model_name = args.model.split("/")[-1]
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Extract from all layers
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))

    print(f"Extracting from all {num_layers} layers")
    print(f"Positions: {args.positions}")

    # Load data
    print(f"\nLoading {args.language} datasets...")
    harmful_train, _ = get_input_data("harmful", args.language)
    harmless_train, _ = get_input_data("harmless", args.language)

    harmful_train = harmful_train[: args.n_samples]
    harmless_train = harmless_train[: args.n_samples]

    print(
        f"Using {len(harmful_train)} harmful and {len(harmless_train)} harmless samples"
    )

    # Extract activations
    print("\nExtracting harmful activations...")
    harmful_acts = extract_activations(
        model, harmful_train, tokenizer, layers, args.positions, args.batch_size
    )

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    print("\nExtracting harmless activations...")
    harmless_acts = extract_activations(
        model, harmless_train, tokenizer, layers, args.positions, args.batch_size
    )

    # Clear cache
    gc.collect()
    torch.cuda.empty_cache()

    # Compute directions
    print("\nComputing steering directions...")
    directions = compute_steering_directions(harmful_acts, harmless_acts, args.strategy)

    # Save steering configs for ALL layers
    print(f"\nSaving steering configs to {output_path}")
    for strategy, config in directions.items():
        best_layer_idx = config["layer"]
        position = config["position"]
        first_direction = config["first_direction"]
        second_direction = config["second_direction"]

        # Create config dict for ALL layers using the selected strategy's directions
        # Match parent structure: save entries for BOTH layernorm modules
        config_all_layers = {}
        layernorm_modules = ["input_layernorm", "post_attention_layernorm"]

        num_layers = len(layers)
        for layer_idx in layers:
            for module in layernorm_modules:
                if module != "input_layernorm":
                    # post_attention_layernorm: use same layer
                    module_name = f"model.layers.{layer_idx}.{module}"
                elif layer_idx < num_layers - 1:
                    # input_layernorm: use NEXT layer (parent's pattern)
                    module_name = f"model.layers.{layer_idx + 1}.{module}"
                else:
                    # Skip last layer's input_layernorm
                    continue

                config_all_layers[module_name] = {
                    "first_direction": first_direction,
                    "second_direction": second_direction,
                }

        filename = f"steering_config-{args.language}-{strategy}_{best_layer_idx}_{position}-pca_0.npy"
        filepath = output_path / filename

        np.save(filepath, config_all_layers, allow_pickle=True)
        print(
            f"  Saved: {filename} (best: layer {best_layer_idx}, {len(config_all_layers)} module entries)"
        )

    print("\n✓ Direction extraction complete!")
    print(f"  Configs saved to: {output_path}")
    print(f"  Total configs: {len(directions)}")


if __name__ == "__main__":
    main()
