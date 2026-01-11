"""Utilities for angular steering with pure PyTorch.

This module contains:
- Data loading for harmful/harmless instructions
- Hook utilities for capturing and modifying activations
- Angular steering implementation with rotation matrices
"""

import functools
import io
from contextlib import contextmanager
from functools import cache
from typing import Callable, Dict, List, Tuple

import pandas as pd
import requests
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# =============================================================================
# Data Loading
# =============================================================================


def get_harmful_instructions():
    """Load harmful instructions from AdvBench dataset."""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    """Load harmless instructions from Alpaca dataset."""
    dataset = load_dataset("tatsu-lab/alpaca")
    instructions = [
        item["instruction"] for item in dataset["train"] if item["input"].strip() == ""
    ]
    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train[:512], test[:128]


@cache
def get_input_data(
    data_type: str, language_id: str = "en"
) -> Tuple[List[str], List[str]]:
    """Get training and test data.

    Args:
        data_type: "harmful" or "harmless"
        language_id: "en" (only English supported)

    Returns:
        Tuple of (train_data, test_data)
    """
    if language_id != "en":
        raise ValueError(f"Only English (en) is supported, got: {language_id}")

    if data_type == "harmful":
        return get_harmful_instructions()
    elif data_type == "harmless":
        return get_harmless_instructions()
    else:
        raise ValueError(f"Unknown data_type: {data_type}. Use 'harmful' or 'harmless'")


# =============================================================================
# Hook Utilities for Activation Extraction
# =============================================================================


def get_activations_hook(cache_key: str, cache: dict, positions: list = None):
    """Create a forward hook that captures activations at specified token positions.

    Args:
        cache_key: Key to store activations in cache dict
        cache: Dictionary to store activations
        positions: List of token positions to extract (e.g., [-1] for last token)

    Returns:
        Hook function that can be registered on a module
    """
    positions = positions or [-1]

    def hook_fn(module, input, output):
        # output shape: (batch_size, seq_len, hidden_dim)
        if positions == [-1]:
            # Extract last token only
            acts = output[:, -1:, :].detach().cpu()
        else:
            # Extract specified positions
            acts = output[:, positions, :].detach().cpu()

        if cache_key in cache:
            cache[cache_key] = torch.cat([cache[cache_key], acts], dim=0)
        else:
            cache[cache_key] = acts

    return hook_fn


@contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[nn.Module, Callable]] = None,
    module_forward_hooks: List[Tuple[nn.Module, Callable]] = None,
    **kwargs,
):
    """Context manager for temporarily adding forward hooks to a model."""
    module_forward_pre_hooks = module_forward_pre_hooks or []
    module_forward_hooks = module_forward_hooks or []

    handles = []
    try:
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def tokenize_instructions_fn(instructions, tokenizer, system_prompt=None):
    """Tokenize instructions using chat template."""
    inputs = tokenizer.apply_chat_template(
        [
            (
                [{"role": "user", "content": instruction}]
                if system_prompt is None
                else [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction},
                ]
            )
            for instruction in instructions
        ],
        padding=True,
        truncation=False,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return inputs


def generate_completions(
    model,
    instructions: List[str],
    tokenizer,
    system_prompt: str = None,
    fwd_pre_hooks: list = None,
    fwd_hooks: list = None,
    batch_size: int = 8,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    """Generate completions with optional steering hooks.

    Args:
        temperature: Sampling temperature (0 = greedy/deterministic). Matches parent's SamplingParams(temperature=0)
        top_p: Nucleus sampling threshold. Matches parent's behavior.

    Returns:
        List of dicts with 'prompt' and 'response' keys
    """
    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []

    completions = []
    num_batches = (len(instructions) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(instructions), batch_size),
        total=num_batches,
        desc="Generating",
        disable=False,
    ):
        batch = instructions[i : i + batch_size]
        inputs = tokenize_instructions_fn(batch, tokenizer, system_prompt)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Decode prompts
        batch_prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        with add_hooks(
            module_forward_pre_hooks=fwd_pre_hooks,
            module_forward_hooks=fwd_hooks,
        ):
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        batch_responses = tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :], skip_special_tokens=True
        )

        # Return as list of dicts
        for prompt, response in zip(batch_prompts, batch_responses):
            completions.append({"prompt": prompt, "response": response})

    return completions


# =============================================================================
# Angular Steering Implementation
# =============================================================================


def _get_rotation_args(
    first_direction: torch.Tensor,
    second_direction: torch.Tensor,
    theta: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rotation matrix and projection for angular steering.

    Args:
        first_direction: Primary direction vector
        second_direction: Secondary direction vector (will be orthogonalized)
        theta: Rotation angle in degrees

    Returns:
        Tuple of (proj_matrix, steering_vector)
    """
    device = first_direction.device
    theta_rad = torch.tensor(theta * torch.pi / 180.0, device=device)

    # Orthonormalize directions (Gram-Schmidt)
    b1 = first_direction / first_direction.norm()
    b2 = second_direction - (second_direction @ b1) * b1
    b2 = b2 / b2.norm()

    # Projection matrix: P = b1⊗b1 + b2⊗b2
    proj_matrix = torch.outer(b1, b1) + torch.outer(b2, b2)

    # 2D rotation matrix in the plane spanned by b1, b2
    cos_theta = torch.cos(theta_rad)
    sin_theta = torch.sin(theta_rad)
    rotation_matrix = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta]),
            torch.stack([sin_theta, cos_theta]),
        ]
    )

    # Compute steering vector: rotate [1, 0] and convert back to original space
    unit_vector = torch.tensor([1.0, 0.0], device=device)
    rotated_2d = rotation_matrix @ unit_vector
    steering_vector = rotated_2d[0] * b1 + rotated_2d[1] * b2

    return proj_matrix, steering_vector


def get_angular_steering_output_hook(
    steering_config: dict,
    target_degree: float,
    adaptive_mode: int = 1,
):
    """Create a hook function that applies angular steering to layer outputs.

    Args:
        steering_config: Dict with 'first_direction' and 'second_direction' keys (numpy arrays)
        target_degree: Rotation angle in degrees (0-360)
        adaptive_mode: 0=always steer, 1=only when aligned with first_direction

    Returns:
        Hook function that can be registered with register_forward_hook()
    """
    # Convert numpy arrays to tensors
    first_direction = torch.from_numpy(steering_config["first_direction"])
    second_direction = torch.from_numpy(steering_config["second_direction"])

    proj_matrix, steering_vector = _get_rotation_args(
        first_direction, second_direction, target_degree
    )

    # Cache for device/dtype converted tensors (avoid repeated .to() calls)
    # This is safe because proj_matrix, steering_vector, and first_direction
    # are immutable closure variables - they never change for this hook instance
    _cache = {}

    def steering_hook(_module, _input, output):
        # Move tensors to output device and dtype (with caching)
        device = output.device
        dtype = output.dtype
        cache_key = (device, dtype)

        if cache_key not in _cache:
            _cache[cache_key] = (
                proj_matrix.to(device=device, dtype=dtype),
                steering_vector.to(device=device, dtype=dtype),
                first_direction.to(device=device, dtype=dtype),
            )

        proj, steer, first_dir = _cache[cache_key]

        # Project activation onto steering plane
        projected = output @ proj  # (batch, seq, hidden_dim)

        # Compute dynamic scale from projected activation norm (matches parent)
        scale = projected.norm(dim=-1, keepdim=True)  # (batch, seq, 1)

        if adaptive_mode == 0:
            # Always apply steering
            steered = output - projected + scale * steer
            return steered

        elif adaptive_mode == 1:
            # Only steer when activation aligns with first_direction
            proj_to_first = output @ first_dir  # (batch, seq)
            mask = (proj_to_first > 0).unsqueeze(-1)  # (batch, seq, 1)

            steered = output - projected + scale * steer
            return torch.where(mask, steered, output)

        else:
            raise ValueError(f"Unknown adaptive_mode: {adaptive_mode}")

    return steering_hook
