"""Phase Portrait Extraction for Angular Steering on SO(2).

Extracts activation angle (phi_k) and angular velocity (omega_k = phi_{k+1} - phi_k)
at each layer for harmful and harmless prompts, producing phase portraits that
characterize the dynamical behavior of the steering plane across depth.

This is Experiment 1 of the SO(2) research proposal.
"""

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    add_hooks,
    get_activations_hook,
    get_input_data,
    tokenize_instructions_fn,
)


def extract_all_layer_activations(
    model,
    instructions: list[str],
    tokenizer,
    positions: list[str] = ("mid",),
    batch_size: int = 8,
) -> dict:
    """Extract activations from ALL layers at specified positions.

    Returns dict mapping "layer_{idx}_{position}" -> tensor of shape (N, hidden_dim).
    """
    num_layers = model.config.num_hidden_layers
    layers = list(range(num_layers))

    cache = {}
    module_dict = dict(model.named_modules())
    hooks = []

    for layer_idx in layers:
        layer_name = f"model.layers.{layer_idx}"
        for pos in positions:
            if pos == "mid":
                module_name = f"{layer_name}.input_layernorm"
            elif pos == "post":
                module_name = f"{layer_name}.post_attention_layernorm"
            else:
                continue

            if module_name in module_dict:
                cache_key = f"layer_{layer_idx}_{pos}"
                hooks.append(
                    (
                        module_dict[module_name],
                        get_activations_hook(cache_key, cache, positions=[-1]),
                    )
                )

    # Tokenize all
    all_input_ids = []
    all_attention_masks = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i : i + batch_size]
        tokenized = tokenize_instructions_fn(batch, tokenizer)
        all_input_ids.append(tokenized.input_ids)
        all_attention_masks.append(tokenized.attention_mask)

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

    activations = {}
    for key, value in cache.items():
        activations[key] = value.squeeze(1)  # (N, hidden_dim)

    return activations


def compute_steering_plane(
    harmful_acts: dict, harmless_acts: dict
) -> dict:
    """Compute the steering plane basis {b1, b2} and per-layer candidate directions.

    Returns dict with keys:
        - "b1": unit feature direction (highest mean cosine similarity)
        - "b2": orthogonalized first PCA component
        - "selected_key": which layer/position was selected
        - "candidate_directions": dict of all per-layer candidate directions
        - "mean_cosines": mean cosine similarity per layer
    """
    candidate_directions = {}

    for key in sorted(harmful_acts.keys()):
        harmful = harmful_acts[key].float()
        harmless = harmless_acts[key].float()

        # Per-sample normalize
        harmful_normed = harmful / harmful.norm(dim=-1, keepdim=True)
        harmless_normed = harmless / harmless.norm(dim=-1, keepdim=True)

        # Mean of normalized
        harmful_mean = harmful_normed.mean(dim=0)
        harmless_mean = harmless_normed.mean(dim=0)

        # Normalize means
        harmful_mean = harmful_mean / harmful_mean.norm()
        harmless_mean = harmless_mean / harmless_mean.norm()

        diff = harmful_mean - harmless_mean
        candidate_directions[key] = diff

    # Drop degenerate candidates whose direction underflows to ~0. This happens
    # at layer 0, where the extracted last-token activation (pre first layernorm)
    # is identical across prompts because the chat-template suffix is shared, so
    # harmful_mean == harmless_mean and diff == 0. Normalizing such a vector
    # yields NaN (0/0), which then poisons selection, PCA, and b1/b2.
    eps = 1e-8
    degenerate = [k for k, v in candidate_directions.items() if v.norm().item() <= eps]
    for k in degenerate:
        del candidate_directions[k]
    if degenerate:
        print(f"[compute_steering_plane] dropped degenerate layers: {degenerate}")
    if not candidate_directions:
        raise ValueError("All candidate directions are degenerate (near-zero).")

    # PCA on all candidates
    sorted_keys = sorted(candidate_directions.keys())
    all_candidates = torch.stack([candidate_directions[k] for k in sorted_keys])

    pca = PCA()
    pca.fit(all_candidates.cpu().numpy())
    pca_components = torch.from_numpy(pca.components_).to(
        device=all_candidates.device, dtype=all_candidates.dtype
    )

    # Select by max mean cosine similarity
    candidates_normalized = {k: v / v.norm() for k, v in candidate_directions.items()}
    candidates_stack = torch.stack([candidates_normalized[k] for k in sorted_keys])
    pairwise_cosine = candidates_stack @ candidates_stack.T
    mean_cosine = pairwise_cosine.mean(dim=-1)

    max_idx = mean_cosine.argmax().item()
    selected_key = sorted_keys[max_idx]

    # Build orthonormal basis
    b1 = candidate_directions[selected_key]
    b1 = b1 / b1.norm()

    # Orient b1 toward the safe (harmless) equilibrium so that phi=0 corresponds
    # to harmless prompts and harmful prompts are displaced toward phi=pi (high
    # potential energy 1-cos(phi)). This matches the pendulum hypothesis where
    # "unsafe" = displaced from the stable equilibrium. b1 is built as
    # (harmful_mean - harmless_mean), so by default phi=0 points at harmful; we
    # flip it if the harmful mean projects more positively onto b1 than harmless.
    h_sel = harmful_acts[selected_key].float()
    l_sel = harmless_acts[selected_key].float()
    h_mean = (h_sel / h_sel.norm(dim=-1, keepdim=True)).mean(0)
    l_mean = (l_sel / l_sel.norm(dim=-1, keepdim=True)).mean(0)
    if (h_mean @ b1) > (l_mean @ b1):
        b1 = -b1

    # b2 = leading PCA component, orthogonalized against b1. The top component is
    # typically near-parallel to b1 (the most central candidate), so its residual
    # collapses to ~0 and b2/b2.norm() would be NaN. Fall back to the next
    # component that retains a usable orthogonal residual.
    b2 = None
    for comp in pca_components:
        residual = comp - (comp @ b1) * b1
        if residual.norm() > 1e-6:
            b2 = residual / residual.norm()
            break
    if b2 is None:
        raise ValueError(
            "Could not construct b2 orthogonal to b1: all PCA components are "
            "parallel to b1."
        )

    return {
        "b1": b1,
        "b2": b2,
        "selected_key": selected_key,
        "candidate_directions": candidate_directions,
        "mean_cosines": {k: mean_cosine[i].item() for i, k in enumerate(sorted_keys)},
    }


def compute_phase_trajectories(
    activations: dict,
    b1: torch.Tensor,
    b2: torch.Tensor,
    position: str = "mid",
) -> dict:
    """Compute phi_k and omega_k for each sample across all layers.

    Returns dict with:
        - "phi": array of shape (N, L) — activation angle at each layer
        - "omega": array of shape (N, L-1) — angular velocity between layers
        - "r": array of shape (N, L) — magnitude of projection onto steering plane
        - "c1": array of shape (N, L) — b1 coordinate
        - "c2": array of shape (N, L) — b2 coordinate
        - "layers": list of layer indices
    """
    # Collect layer keys in order
    layer_keys = sorted(
        [k for k in activations.keys() if k.endswith(f"_{position}")],
        key=lambda x: int(x.split("_")[1]),
    )

    if not layer_keys:
        raise ValueError(f"No activations found for position '{position}'")

    layer_indices = [int(k.split("_")[1]) for k in layer_keys]
    n_samples = activations[layer_keys[0]].shape[0]
    n_layers = len(layer_keys)

    device = b1.device
    dtype = b1.dtype

    phi = np.zeros((n_samples, n_layers))
    r = np.zeros((n_samples, n_layers))
    c1_arr = np.zeros((n_samples, n_layers))
    c2_arr = np.zeros((n_samples, n_layers))

    for j, key in enumerate(layer_keys):
        acts = activations[key].to(device=device, dtype=dtype)

        # Project onto steering plane
        coord1 = (acts @ b1).cpu().numpy()  # (N,)
        coord2 = (acts @ b2).cpu().numpy()  # (N,)

        c1_arr[:, j] = coord1
        c2_arr[:, j] = coord2
        phi[:, j] = np.arctan2(coord2, coord1)
        r[:, j] = np.sqrt(coord1**2 + coord2**2)

    # Angular velocity: omega_k = phi_{k+1} - phi_k, wrapped to [-pi, pi]
    dphi = np.diff(phi, axis=1)
    # Wrap to [-pi, pi]
    omega = np.arctan2(np.sin(dphi), np.cos(dphi))

    return {
        "phi": phi,
        "omega": omega,
        "r": r,
        "c1": c1_arr,
        "c2": c2_arr,
        "layers": layer_indices,
    }


def plot_phase_portrait(
    harmful_traj: dict,
    harmless_traj: dict,
    model_name: str,
    output_dir: Path,
):
    """Generate phase portrait plots.

    Creates:
    1. Phase portrait (phi vs omega), color-coded by layer depth
    2. Phase portrait, color-coded by prompt type (harmful vs harmless)
    3. Angle evolution across layers
    4. Angular velocity evolution across layers
    5. 2D trajectory in the steering plane (c1 vs c2)
    6. Energy landscape
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = harmful_traj["layers"]
    n_layers = len(layers)

    # --- Figure 1: Phase portrait colored by layer depth ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, traj, label in [
        (axes[0], harmful_traj, "Harmful"),
        (axes[1], harmless_traj, "Harmless"),
    ]:
        n_samples = traj["phi"].shape[0]
        # Use a subset for clarity
        n_show = min(n_samples, 100)

        for i in range(n_show):
            phi_vals = traj["phi"][i, :-1]
            omega_vals = traj["omega"][i, :]
            layer_vals = np.array(layers[:-1])

            scatter = ax.scatter(
                phi_vals,
                omega_vals,
                c=layer_vals,
                cmap="viridis",
                s=3,
                alpha=0.3,
                vmin=0,
                vmax=n_layers - 1,
            )
            # Connect with lines
            ax.plot(phi_vals, omega_vals, alpha=0.05, color="gray", linewidth=0.5)

        ax.set_xlabel("φ (activation angle)", fontsize=12)
        ax.set_ylabel("ω (angular velocity)", fontsize=12)
        ax.set_title(f"{label} Prompts", fontsize=14)
        ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

    plt.colorbar(scatter, ax=axes, label="Layer index", shrink=0.8)
    fig.suptitle(f"Phase Portrait — {model_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "phase_portrait_by_layer.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 2: Phase portrait colored by prompt type (overlay) ---
    fig, ax = plt.subplots(figsize=(10, 8))

    n_show = min(harmful_traj["phi"].shape[0], harmless_traj["phi"].shape[0], 100)

    for i in range(n_show):
        ax.plot(
            harmful_traj["phi"][i, :-1],
            harmful_traj["omega"][i, :],
            alpha=0.08,
            color="red",
            linewidth=0.5,
        )
        ax.plot(
            harmless_traj["phi"][i, :-1],
            harmless_traj["omega"][i, :],
            alpha=0.08,
            color="blue",
            linewidth=0.5,
        )

    # Plot means
    harmful_phi_mean = harmful_traj["phi"][:, :-1].mean(axis=0)
    harmful_omega_mean = harmful_traj["omega"].mean(axis=0)
    harmless_phi_mean = harmless_traj["phi"][:, :-1].mean(axis=0)
    harmless_omega_mean = harmless_traj["omega"].mean(axis=0)

    ax.plot(
        harmful_phi_mean,
        harmful_omega_mean,
        color="red",
        linewidth=2.5,
        label="Harmful (mean)",
        zorder=5,
    )
    ax.plot(
        harmless_phi_mean,
        harmless_omega_mean,
        color="blue",
        linewidth=2.5,
        label="Harmless (mean)",
        zorder=5,
    )

    # Mark start and end
    for phi_m, omega_m, color in [
        (harmful_phi_mean, harmful_omega_mean, "red"),
        (harmless_phi_mean, harmless_omega_mean, "blue"),
    ]:
        ax.scatter(phi_m[0], omega_m[0], color=color, s=80, marker="o", zorder=6)
        ax.scatter(phi_m[-1], omega_m[-1], color=color, s=80, marker="X", zorder=6)

    ax.set_xlabel("φ (activation angle)", fontsize=12)
    ax.set_ylabel("ω (angular velocity)", fontsize=12)
    ax.set_title(f"Phase Portrait — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "phase_portrait_overlay.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 3: Angle evolution across layers ---
    fig, ax = plt.subplots(figsize=(12, 5))

    n_show = min(harmful_traj["phi"].shape[0], 50)
    for i in range(n_show):
        ax.plot(layers, harmful_traj["phi"][i], alpha=0.1, color="red", linewidth=0.5)
    for i in range(n_show):
        ax.plot(layers, harmless_traj["phi"][i], alpha=0.1, color="blue", linewidth=0.5)

    ax.plot(layers, harmful_traj["phi"].mean(axis=0), color="red", linewidth=2, label="Harmful (mean)")
    ax.plot(layers, harmless_traj["phi"].mean(axis=0), color="blue", linewidth=2, label="Harmless (mean)")

    ax.fill_between(
        layers,
        harmful_traj["phi"].mean(axis=0) - harmful_traj["phi"].std(axis=0),
        harmful_traj["phi"].mean(axis=0) + harmful_traj["phi"].std(axis=0),
        alpha=0.15,
        color="red",
    )
    ax.fill_between(
        layers,
        harmless_traj["phi"].mean(axis=0) - harmless_traj["phi"].std(axis=0),
        harmless_traj["phi"].mean(axis=0) + harmless_traj["phi"].std(axis=0),
        alpha=0.15,
        color="blue",
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("φ (activation angle)", fontsize=12)
    ax.set_title(f"Angle Evolution Across Layers — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "angle_evolution.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 4: Angular velocity across layers ---
    fig, ax = plt.subplots(figsize=(12, 5))

    mid_layers = layers[:-1]
    ax.plot(mid_layers, harmful_traj["omega"].mean(axis=0), color="red", linewidth=2, label="Harmful (mean)")
    ax.plot(mid_layers, harmless_traj["omega"].mean(axis=0), color="blue", linewidth=2, label="Harmless (mean)")

    ax.fill_between(
        mid_layers,
        harmful_traj["omega"].mean(axis=0) - harmful_traj["omega"].std(axis=0),
        harmful_traj["omega"].mean(axis=0) + harmful_traj["omega"].std(axis=0),
        alpha=0.15,
        color="red",
    )
    ax.fill_between(
        mid_layers,
        harmless_traj["omega"].mean(axis=0) - harmless_traj["omega"].std(axis=0),
        harmless_traj["omega"].mean(axis=0) + harmless_traj["omega"].std(axis=0),
        alpha=0.15,
        color="blue",
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("ω (angular velocity)", fontsize=12)
    ax.set_title(f"Angular Velocity Across Layers — {model_name}", fontsize=14)
    ax.axhline(y=0, color="k", linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "angular_velocity.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 5: 2D trajectory in steering plane (c1 vs c2) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, traj, label, color in [
        (axes[0], harmful_traj, "Harmful", "red"),
        (axes[1], harmless_traj, "Harmless", "blue"),
    ]:
        n_show = min(traj["c1"].shape[0], 50)
        for i in range(n_show):
            ax.plot(traj["c1"][i], traj["c2"][i], alpha=0.1, color=color, linewidth=0.5)

        # Mean trajectory with arrows
        c1_mean = traj["c1"].mean(axis=0)
        c2_mean = traj["c2"].mean(axis=0)
        ax.plot(c1_mean, c2_mean, color=color, linewidth=2.5, label=f"{label} (mean)")

        # Color points by layer
        scatter = ax.scatter(
            c1_mean, c2_mean, c=layers, cmap="viridis", s=30, zorder=5,
            vmin=0, vmax=n_layers - 1,
        )
        ax.scatter(c1_mean[0], c2_mean[0], color="green", s=100, marker="o", zorder=6, label="Start")
        ax.scatter(c1_mean[-1], c2_mean[-1], color="black", s=100, marker="X", zorder=6, label="End")

        # Draw b1 and b2 directions
        max_coord = max(abs(c1_mean).max(), abs(c2_mean).max()) * 0.3
        ax.annotate("", xy=(max_coord, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
        ax.annotate("", xy=(0, max_coord), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
        ax.text(max_coord * 1.05, 0, "b₁ (refusal)", fontsize=9, color="gray")
        ax.text(0, max_coord * 1.05, "b₂ (PCA)", fontsize=9, color="gray")

        ax.set_xlabel("b₁ coordinate", fontsize=12)
        ax.set_ylabel("b₂ coordinate", fontsize=12)
        ax.set_title(f"{label} — Steering Plane Trajectory", fontsize=14)
        ax.legend(fontsize=10)
        ax.set_aspect("equal")

    plt.colorbar(scatter, ax=axes, label="Layer index", shrink=0.8)
    fig.suptitle(f"2D Trajectory in Steering Plane — {model_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "steering_plane_trajectory.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Figure 6: Behavioral energy landscape ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Define energy: V = (1/2)*J*omega^2 + kappa*(1 - cos(phi - phi*))
    # Use phi*=0 (b1 direction = refusal direction)
    # Fit J and kappa from data range
    kappa = 1.0
    J = 1.0

    for ax, traj, label, color in [
        (axes[0], harmful_traj, "Harmful", "red"),
        (axes[1], harmless_traj, "Harmless", "blue"),
    ]:
        phi_vals = traj["phi"][:, :-1]  # (N, L-1)
        omega_vals = traj["omega"]  # (N, L-1)

        kinetic = 0.5 * J * omega_vals**2
        potential = kappa * (1 - np.cos(phi_vals))
        energy = kinetic + potential

        # Mean energy across layers
        mid_layers = layers[:-1]
        energy_mean = energy.mean(axis=0)
        energy_std = energy.std(axis=0)

        ax.plot(mid_layers, energy_mean, color=color, linewidth=2, label=f"{label} (mean)")
        ax.fill_between(
            mid_layers,
            energy_mean - energy_std,
            energy_mean + energy_std,
            alpha=0.2,
            color=color,
        )

        # Separatrix energy level
        ax.axhline(y=kappa, color="orange", linewidth=1.5, linestyle="--", label=f"Separatrix (κ={kappa})")

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("V (behavioral energy)", fontsize=12)
        ax.set_title(f"{label} — Energy Evolution", fontsize=14)
        ax.legend(fontsize=10)

    fig.suptitle(f"Behavioral Energy — {model_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "energy_landscape.pdf", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"All plots saved to {output_dir}")


def compute_statistics(harmful_traj: dict, harmless_traj: dict) -> dict:
    """Compute summary statistics for the phase portrait analysis."""
    stats = {}

    # Mean angle separation at each layer
    phi_harmful_mean = harmful_traj["phi"].mean(axis=0)
    phi_harmless_mean = harmless_traj["phi"].mean(axis=0)
    angle_sep = np.arctan2(
        np.sin(phi_harmful_mean - phi_harmless_mean),
        np.cos(phi_harmful_mean - phi_harmless_mean),
    )
    stats["angle_separation_per_layer"] = angle_sep.tolist()
    stats["max_angle_separation"] = float(np.max(np.abs(angle_sep)))
    stats["max_separation_layer"] = int(np.argmax(np.abs(angle_sep)))

    # Angular velocity statistics
    stats["harmful_omega_mean"] = float(harmful_traj["omega"].mean())
    stats["harmful_omega_std"] = float(harmful_traj["omega"].std())
    stats["harmless_omega_mean"] = float(harmless_traj["omega"].mean())
    stats["harmless_omega_std"] = float(harmless_traj["omega"].std())

    # Energy statistics (with kappa=1, J=1)
    for label, traj in [("harmful", harmful_traj), ("harmless", harmless_traj)]:
        phi = traj["phi"][:, :-1]
        omega = traj["omega"]
        energy = 0.5 * omega**2 + (1 - np.cos(phi))

        stats[f"{label}_energy_mean"] = float(energy.mean())
        stats[f"{label}_energy_final_mean"] = float(energy[:, -1].mean())
        stats[f"{label}_energy_max_mean"] = float(energy.max(axis=1).mean())

    # Separatrix crossing (V > kappa = 1.0). The "any layer" reduction is
    # degenerate: over a long noisy trajectory essentially every prompt of either
    # class exceeds the threshold at some layer, so it saturates at 100%/100% and
    # carries no signal. Instead report layer-specific metrics:
    #   - *_separatrix_crossings: count above threshold at the FINAL layer (the
    #     model's settled state), which is what actually discriminates the classes.
    #   - *_separatrix_fraction: mean fraction of layers each prompt spends above
    #     the threshold (a smooth per-prompt measure of how "displaced" it is).
    # The old saturating value is kept as *_separatrix_crossings_any for reference.
    kappa = 1.0
    for label, traj in [("harmful", harmful_traj), ("harmless", harmless_traj)]:
        phi = traj["phi"][:, :-1]
        omega = traj["omega"]
        energy = 0.5 * omega**2 + (1 - np.cos(phi))
        above = energy > kappa
        stats[f"{label}_max_energy_any"] = float(energy.max())
        stats[f"{label}_separatrix_crossings"] = int(above[:, -1].sum())
        stats[f"{label}_separatrix_crossings_any"] = int(above.any(axis=1).sum())
        stats[f"{label}_separatrix_fraction"] = float(above.mean(axis=1).mean())

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract phase portraits from LLM activation dynamics"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID (e.g., 'Qwen/Qwen2.5-3B-Instruct')",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./phase_portrait_output",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--position", type=str, default="mid", choices=["mid", "post"],
        help="Extraction position: mid (after layernorm, before attn) or post (before MLP)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=512,
        help="Number of samples per category",
    )
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("Loading datasets...")
    harmful_train, _ = get_input_data("harmful", "en")
    harmless_train, _ = get_input_data("harmless", "en")
    harmful_train = harmful_train[: args.n_samples]
    harmless_train = harmless_train[: args.n_samples]
    print(f"Using {len(harmful_train)} harmful and {len(harmless_train)} harmless samples")

    # Extract activations
    print("\nExtracting harmful activations...")
    harmful_acts = extract_all_layer_activations(
        model, harmful_train, tokenizer, [args.position], args.batch_size,
    )
    gc.collect()
    torch.cuda.empty_cache()

    print("\nExtracting harmless activations...")
    harmless_acts = extract_all_layer_activations(
        model, harmless_train, tokenizer, [args.position], args.batch_size,
    )
    gc.collect()
    torch.cuda.empty_cache()

    # Compute steering plane
    print("\nComputing steering plane...")
    plane = compute_steering_plane(harmful_acts, harmless_acts)
    print(f"Selected direction from: {plane['selected_key']}")

    # Compute phase trajectories
    print("\nComputing phase trajectories...")
    harmful_traj = compute_phase_trajectories(
        harmful_acts, plane["b1"], plane["b2"], args.position,
    )
    harmless_traj = compute_phase_trajectories(
        harmless_acts, plane["b1"], plane["b2"], args.position,
    )

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(harmful_traj, harmless_traj)
    stats["model"] = args.model
    stats["position"] = args.position
    stats["n_harmful"] = len(harmful_train)
    stats["n_harmless"] = len(harmless_train)
    stats["selected_direction"] = plane["selected_key"]
    stats["n_layers"] = len(harmful_traj["layers"])

    # Print key results
    print(f"\n{'='*60}")
    print(f"PHASE PORTRAIT RESULTS — {model_name}")
    print(f"{'='*60}")
    print(f"Layers: {stats['n_layers']}")
    print(f"Selected direction: {stats['selected_direction']}")
    print(f"Max angle separation: {stats['max_angle_separation']:.4f} rad "
          f"({np.degrees(stats['max_angle_separation']):.1f}°) at layer {stats['max_separation_layer']}")
    print(f"\nAngular velocity (omega):")
    print(f"  Harmful:  mean={stats['harmful_omega_mean']:.4f}, std={stats['harmful_omega_std']:.4f}")
    print(f"  Harmless: mean={stats['harmless_omega_mean']:.4f}, std={stats['harmless_omega_std']:.4f}")
    print(f"\nBehavioral energy (V = 0.5*omega^2 + 1-cos(phi)):")
    print(f"  Harmful:  mean={stats['harmful_energy_mean']:.4f}, final={stats['harmful_energy_final_mean']:.4f}")
    print(f"  Harmless: mean={stats['harmless_energy_mean']:.4f}, final={stats['harmless_energy_final_mean']:.4f}")
    print(f"\nSeparatrix crossings at final layer (energy > 1.0):")
    print(f"  Harmful:  {stats['harmful_separatrix_crossings']}/{stats['n_harmful']} samples "
          f"({stats['harmful_separatrix_fraction']:.1%} of layers above, on average)")
    print(f"  Harmless: {stats['harmless_separatrix_crossings']}/{stats['n_harmless']} samples "
          f"({stats['harmless_separatrix_fraction']:.1%} of layers above, on average)")

    # Save statistics
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save raw trajectory data
    np.savez_compressed(
        output_dir / "trajectories.npz",
        harmful_phi=harmful_traj["phi"],
        harmful_omega=harmful_traj["omega"],
        harmful_r=harmful_traj["r"],
        harmful_c1=harmful_traj["c1"],
        harmful_c2=harmful_traj["c2"],
        harmless_phi=harmless_traj["phi"],
        harmless_omega=harmless_traj["omega"],
        harmless_r=harmless_traj["r"],
        harmless_c1=harmless_traj["c1"],
        harmless_c2=harmless_traj["c2"],
        layers=harmful_traj["layers"],
        b1=plane["b1"].cpu().numpy(),
        b2=plane["b2"].cpu().numpy(),
    )

    # Generate plots
    print("\nGenerating plots...")
    plot_phase_portrait(harmful_traj, harmless_traj, model_name, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
