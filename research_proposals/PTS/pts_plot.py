"""Plot PTS results.

Two figure sets:
  * pts_offline.png   — from <traj_dir>/pts_offline_results.npz  (controller maths)
  * pts_prototype.png — from pts_prototype_results.npz           (model in the loop)

Run:  python pts_plot.py            # both, if the npz files exist
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OFFLINE = "phase_portrait_output/Qwen2.5-3B-Instruct/pts_offline_results.npz"
PROTO = "pts_prototype_results.npz"


def plot_offline(path=OFFLINE, out="pts_offline.png"):
    d = np.load(path, allow_pickle=True)
    layers = d["layers"]; band = d["band"]; band_layers = d["band_layers"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (1) trajectory separation 2D vs 1D
    ax[0, 0].plot(layers, d["auc_1d"], "o-", color="gray", label="1D (b₁ projection)")
    ax[0, 0].plot(layers, d["auc_2d"], "o-", color="crimson", label="2D (steering plane)")
    ax[0, 0].axvspan(layers[band[0]], layers[band[1]], color="gold", alpha=0.15,
                     label="steering band")
    ax[0, 0].axhline(0.5, color="k", ls=":", alpha=0.4)
    ax[0, 0].set_xlabel("layer"); ax[0, 0].set_ylabel("harmful-vs-harmless AUC")
    ax[0, 0].set_title(f"Exp 1: 2D carries more info than 1D\n"
                       f"mean AUC {d['auc_1d'].mean():.3f} → {d['auc_2d'].mean():.3f}")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # (2) dynamics fit per-layer R^2
    ax[0, 1].plot(layers[:-1], d["per_layer_r2"], "o-", color="seagreen", label="affine A_k,b_k")
    ax[0, 1].plot(layers[:-1], d["per_layer_r2_lin"], "o-", color="orange", alpha=0.6,
                  label="linear A_k only")
    ax[0, 1].set_ylim(min(0.8, float(np.min(d["per_layer_r2"])) - 0.02), 1.005)
    ax[0, 1].set_xlabel("layer transition k→k+1"); ax[0, 1].set_ylabel("1-step R²")
    ax[0, 1].set_title(f"Exp 2: 2×2 dynamics fit\nR²(1)={float(d['r2_h1']):.3f} "
                       f"R²(5)={float(d['r2_h5']):.3f}")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # (3) reference trajectories in the plane
    hm, lm, ref = d["harmful_mean"], d["harmless_mean"], d["ref"]
    ax[0, 2].plot(hm[:, 0], hm[:, 1], "-", color="crimson", alpha=0.5, label="harmful mean")
    ax[0, 2].plot(lm[:, 0], lm[:, 1], "-", color="royalblue", alpha=0.5, label="harmless mean")
    sc = ax[0, 2].scatter(ref[band_layers, 0], ref[band_layers, 1],
                          c=band_layers, cmap="viridis", s=40, zorder=5, label="reference τ*")
    ax[0, 2].scatter([0], [0], marker="+", color="k", s=80)
    ax[0, 2].set_xlabel("b₁ coordinate"); ax[0, 2].set_ylabel("b₂ coordinate")
    ax[0, 2].set_title("2D reference trajectory (Option A)\nin the steering plane")
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=0.3); ax[0, 2].set_aspect("equal")
    plt.colorbar(sc, ax=ax[0, 2], label="layer", shrink=0.8)

    # (4) tracking vs perturbation Pareto
    names = [str(x) for x in d["sweep_names"]]
    colors = {"PTS-MPC": "crimson", "MPC-myopic(H=1)": "darkorange",
              "LQR-unconstrained": "purple", "FixedAngle": "royalblue", "No-steer": "gray"}
    for i, n in enumerate(names):
        umax = d["sweep_umax"][i]; angerr = d["sweep_ang"][i]
        if n in ("PTS-MPC", "MPC-myopic(H=1)"):
            ax[1, 0].plot(umax, angerr, "o-", color=colors[n], label=n)
        else:                                   # single high-perturbation points
            ax[1, 0].scatter(umax[0], angerr[0], color=colors[n], s=90, marker="X",
                             zorder=5, label=n)
    ax[1, 0].set_xlabel("realised perturbation  mean ‖u‖  (off-manifold push)")
    ax[1, 0].set_ylabel("angular tracking error (deg)")
    ax[1, 0].set_title("Exp 3: PTS is a tunable frontier\n"
                       "(ties LQR at high ‖u‖, beats FixedAngle below it)")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # (5) horizon ablation + rotation magnitude
    ax2 = ax[1, 1].twinx()
    ax[1, 1].plot(d["ablation_horizons"], d["ablation_ang"], "o-", color="crimson",
                  label="MPC ang-err vs H")
    ax[1, 1].set_xlabel("prediction horizon H"); ax[1, 1].set_ylabel("ang-err (deg)", color="crimson")
    ax[1, 1].set_title("Exp 3b: lookahead benign in the\nnear-identity late band")
    ax2.plot(layers[:-1], d["rot_mag"], "-", color="teal", alpha=0.6)
    ax2.axvspan(layers[band[0]], layers[band[1]], color="gold", alpha=0.12)
    ax2.set_ylabel("‖A_k − I‖₂  (teal)", color="teal")
    ax[1, 1].grid(alpha=0.3)

    # (6) constraint satisfaction
    tn = [str(x) for x in d["track_names"]]
    umaxr = d["track_umax"]; u_max = float(d["u_max"])
    bars = ax[1, 2].bar(range(len(tn)), umaxr,
                        color=[colors.get(n, "gray") for n in tn])
    ax[1, 2].axhline(u_max, color="red", ls="--", lw=1.5, label=f"budget u_max={u_max:.2f}")
    ax[1, 2].set_xticks(range(len(tn)))
    ax[1, 2].set_xticklabels([n[:10] for n in tn], rotation=30, ha="right", fontsize=7)
    ax[1, 2].set_ylabel("realised max ‖u‖")
    ax[1, 2].set_title("Exp 4: only PTS respects the\nperturbation budget")
    ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3, axis="y")

    fig.suptitle("Predictive Trajectory Steering (PTS) — offline controller validation "
                 "(Qwen2.5-3B trajectories)", fontsize=15)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def plot_prototype(path=PROTO, out="pts_prototype.png"):
    d = np.load(path, allow_pickle=True)
    band = d["band"]; layers = np.arange(int(d["L"]))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    # (1) real-model tracking: incoming angle vs reference, autonomous vs PTS
    ax[0].plot(band, np.degrees(d["ref_ang"]), "s-", color="seagreen", label="reference τ* angle")
    ax[0].plot(band, np.degrees(d["real_non_ang"]), "o-", color="gray", label="autonomous (no steer)")
    ax[0].plot(band, np.degrees(d["real_pts_ang"]), "o-", color="crimson", label="PTS-MPC (steered)")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("mean in-plane angle (deg)")
    ax[0].set_title(f"[B] PTS bends the REAL residual stream\n"
                    f"to-ref: {float(d['track_autonomous']):.0f}° → {float(d['track_pts']):.0f}°  "
                    f"(w_j={float(d['wj_mean']):.2f})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # (2) behavioural dial: margin vs lambda
    lam = d["lambdas"]; ml = d["margins_lam"]; pert = d["pert_lam"]
    ax[1].plot(lam, ml, "o-", color="crimson", label="PTS-MPC margin")
    ax[1].axhline(float(d["base_margin"].mean()), color="gray", ls="--", label="baseline (no steer)")
    ax[1].axhline(float(d["fixed_margin"].mean()), color="royalblue", ls=":", label="fixed-angle")
    ax[1].set_xlabel("reference interp λ  (0=harmful ref → 1=harmless ref)")
    ax[1].set_ylabel("refusal–compliance margin")
    ax[1].set_title("[C] 2D reference is a behavioural dial\n"
                    "(monotone λ 0→0.75, saturates by λ=1)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # (3) behavioural shift vs perturbation (PTS bounded vs fixed-angle)
    ax[2].plot(pert, ml, "o-", color="crimson", label="PTS-MPC (λ sweep)")
    ax[2].set_xlabel("realised perturbation  mean ‖u‖")
    ax[2].set_ylabel("refusal–compliance margin")
    ax[2].set_title("[C] PTS shifts behaviour at\nbounded, tunable perturbation")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle(f"PTS in the loop — Qwen2.5-3B, band layers {band[0]}–{band[-1]} "
                 f"(R²₁={float(d['r2_h1']):.3f})", fontsize=15)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    if Path(OFFLINE).exists():
        plot_offline()
    else:
        print(f"(skip offline: {OFFLINE} not found — run pts_offline.py)")
    if Path(PROTO).exists():
        plot_prototype()
    else:
        print(f"(skip prototype: {PROTO} not found — run pts_prototype.py)")
