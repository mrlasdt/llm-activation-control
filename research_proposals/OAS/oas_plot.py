"""Plot OAS results (mirror of pts_plot.py).

Two figure sets:
  * oas_offline.png   — from <traj_dir>/oas_offline_results.npz  (the controller maths,
                        model-FREE: enforcement-layer sweep, soft-vs-deadbeat frontier,
                        observer / LQG validation)
  * oas_prototype.png — from oas_prototype_results.npz           (model in the loop:
                        authority gate, behavioral enforcement curve, coherence Pareto)

OAS = Observer-Based, Soft-Landing Angular Steering (depth-domain LQG). All angle
errors in the offline npz are in RADIANS; we convert to degrees for the figures.

matplotlib Agg backend; both plots are GUARDED on the npz existing (the offline file
is written by oas_offline.py which runs in CI; the prototype file needs the model).

Run:  python oas_plot.py            # both, if the npz files exist
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# the offline driver writes alongside the trajectories it consumes (SO2 output dir)
OFFLINE = "../SO2/phase_portrait_output/Qwen2.5-3B-Instruct/oas_offline_results.npz"
PROTO = "oas_prototype_results.npz"


def plot_offline(path=OFFLINE, out="oas_offline.png"):
    """Six-panel offline figure (OAS_SPEC S6, offline half).

    Row 0: Exp-2 geometric enforcement-layer curve | Exp-3a effort-vs-terminal-error
           frontier (rho sweep, deadbeat = rho->0) | Exp-3b soft-vs-deadbeat push bar.
    Row 1: Exp-3d advantage vs ||A_k-I|| scatter | Exp-4b Kalman/single/EMA angle RMSE
           | Exp-4c LQG-vs-LQR terminal error vs injected measurement noise (var bands).
    """
    d = np.load(path, allow_pickle=True)
    band = d["band"]; band_layers = d["band_layers"]
    kT = int(d["terminal_layer"]); peak = int(d["peak_layer"])
    target_deg = np.degrees(float(d["target_angle"]))

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (1) Exp 2 — geometric enforcement-layer sweep: does early enforcement wash out?
    enf = d["exp2_enf_layers_global"]
    enf_err = np.degrees(d["exp2_terminal_ang_err"])
    best = int(d["exp2_best_enf_layer"])
    ax[0, 0].plot(enf, enf_err, "o-", color="crimson", label="terminal angle error")
    ax[0, 0].axvline(best, color="seagreen", ls="--", lw=1.5,
                     label=f"best enf. layer = {best} (= kT)")
    ax[0, 0].axvline(peak, color="gray", ls=":", alpha=0.6, label=f"peak layer {peak}")
    ax[0, 0].set_xlabel("single enforcement layer")
    ax[0, 0].set_ylabel("terminal angle error (deg)")
    ax[0, 0].set_title("Exp 2 (geometric): early enforcement washes out\n"
                       "under the natural dynamics -> land LATE (motivates kT)")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # (2) Exp 3a — rho frontier: effort vs terminal angle error (deadbeat = rho->0)
    rhos = d["exp3_rhos"]
    eff = d["exp3_total_effort"]; umax = d["exp3_max_u"]
    ferr = np.degrees(d["exp3_terminal_ang_err"])
    ax[0, 1].plot(ferr, eff, "o-", color="purple", label="total effort Σ‖u‖")
    ax[0, 1].plot(ferr, umax, "s-", color="darkorange", label="max ‖u‖ (per-layer push)")
    # annotate the rho->0 (deadbeat) endpoint and the soft-landing matched rho
    ax[0, 1].annotate(f"ρ→0\n(deadbeat)", xy=(ferr[0], eff[0]), fontsize=7,
                      xytext=(ferr[0] + 1.0, eff[0] - 0.6),
                      arrowprops=dict(arrowstyle="->", color="gray"))
    ax[0, 1].annotate(f"ρ→{rhos[-1]:.0f}", xy=(ferr[-1], eff[-1]), fontsize=7,
                      xytext=(ferr[-1] - 4.0, eff[-1] + 0.4),
                      arrowprops=dict(arrowstyle="->", color="gray"))
    ax[0, 1].set_xlabel("terminal angle error (deg)")
    ax[0, 1].set_ylabel("control effort")
    ax[0, 1].set_title("Exp 3a: soft-landing is a tunable frontier\n"
                       "(ρ↑ ⇒ less effort, more terminal error)")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    # (3) Exp 3b — soft-landing vs deadbeat at MATCHED terminal angle (push bar)
    db_eff = float(d["exp3_db_total_effort"]); db_umax = float(d["exp3_db_max_u"])
    sl_eff = float(d["exp3_sl_total_effort"]); sl_umax = float(d["exp3_sl_max_u"])
    sl_rho = float(d["exp3_sl_rho"]); gap = np.degrees(float(d["exp3_match_gap"]))
    xs = np.arange(2)
    w = 0.36
    ax[0, 2].bar(xs - w / 2, [db_eff, sl_eff], w, color=["crimson", "seagreen"],
                 label="total effort Σ‖u‖")
    ax[0, 2].bar(xs + w / 2, [db_umax, sl_umax], w, color=["crimson", "seagreen"],
                 alpha=0.5, hatch="//", label="max ‖u‖ (per-layer)")
    ax[0, 2].set_xticks(xs)
    ax[0, 2].set_xticklabels(["deadbeat\n(1 layer)", f"soft-landing\n(band, ρ={sl_rho:.2f})"])
    ax[0, 2].set_ylabel("control push")
    push_ratio = db_umax / (sl_umax + 1e-12)
    ax[0, 2].set_title(f"Exp 3b: matched terminal angle (gap={gap:.3f}°)\n"
                       f"soft-landing → {push_ratio:.1f}× smaller per-layer push")
    ax[0, 2].legend(fontsize=8); ax[0, 2].grid(alpha=0.3, axis="y")

    # (4) Exp 3d — soft-landing advantage (scale-free push delta) vs ||A_k - I||₂
    rot_band = d["exp3_adv_rot_band"]; adv = d["exp3_adv_advantage"]
    corr = float(d["exp3_adv_corr"]); bw = int(d["exp3_adv_band_width"])
    frac_pos = float(d["exp3_adv_frac_pos"])
    sc = ax[1, 0].scatter(rot_band, adv, c=d["exp3_adv_kT"], cmap="viridis",
                          s=40, zorder=5)
    ax[1, 0].axhline(0.0, color="k", ls=":", alpha=0.5)
    ax[1, 0].set_xlabel("band-mean ‖A_k − I‖₂  (rotation magnitude)")
    ax[1, 0].set_ylabel("advantage = deadbeat push − soft push")
    ax[1, 0].set_title(f"Exp 3d: advantage vs rotation (width-{bw} bands)\n"
                       f"adv>0 in {100*frac_pos:.0f}% of bands  (corr={corr:+.2f})")
    ax[1, 0].grid(alpha=0.3)
    plt.colorbar(sc, ax=ax[1, 0], label="terminal layer kT", shrink=0.8)

    # (5) Exp 4b — angle RMSE: Kalman-fused vs best-single-layer vs ad-hoc EMA
    est_layers = d["exp4_est_layers"]
    Vsc = float(d["exp4_est_V_scale"])
    ax[1, 1].plot(est_layers, d["exp4_kalman_rmse"], "o-", color="seagreen",
                  label="Kalman-fused")
    ax[1, 1].plot(est_layers, d["exp4_single_rmse"], "s-", color="crimson",
                  label="best single-layer")
    ax[1, 1].plot(est_layers, d["exp4_ema_rmse"], "^-", color="darkorange", alpha=0.8,
                  label="ad-hoc EMA")
    ax[1, 1].set_xlabel("band layer")
    ax[1, 1].set_ylabel("angle estimate RMSE (rad)")
    ax[1, 1].set_title(f"Exp 4b: Kalman fuses the depth sequence\n"
                       f"(measurement noise V={Vsc:.0%}; lower is better)")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    # (6) Exp 4c — LQG (control on x̂) vs LQR-on-raw vs injected measurement noise
    Vscales = d["exp4_V_scales"]
    lqg = np.degrees(d["exp4_lqg_err"]); lqr = np.degrees(d["exp4_lqr_err"])
    lqg_sd = np.degrees(np.sqrt(d["exp4_lqg_var"]))
    lqr_sd = np.degrees(np.sqrt(d["exp4_lqr_var"]))
    ax[1, 2].plot(Vscales, lqg, "o-", color="seagreen", label="LQG (filtered x̂)")
    ax[1, 2].fill_between(Vscales, lqg - lqg_sd, lqg + lqg_sd, color="seagreen", alpha=0.18)
    ax[1, 2].plot(Vscales, lqr, "s-", color="crimson", label="LQR on raw readout")
    ax[1, 2].fill_between(Vscales, lqr - lqr_sd, lqr + lqr_sd, color="crimson", alpha=0.18)
    ax[1, 2].set_xlabel("injected measurement-noise scale V")
    ax[1, 2].set_ylabel("terminal angle error (deg)")
    ax[1, 2].set_title("Exp 4c: LQG wins as noise grows\n"
                       "(ties LQR at V→0 — reported honestly)")
    ax[1, 2].legend(fontsize=8); ax[1, 2].grid(alpha=0.3)

    fig.suptitle(f"OAS — offline controller validation (Qwen2.5-3B trajectories): "
                 f"band layers {int(band[0])}–{int(band[1])}, kT={kT}, "
                 f"target {target_deg:.0f}°", fontsize=15)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


def plot_prototype(path=PROTO, out="oas_prototype.png"):
    """Three-panel prototype figure (OAS_SPEC S6, model-in-the-loop half).

    (1) Exp-1 AUTHORITY GATE: realized behavioral range vs steered band WIDTH, for
        refusal (multi-angle + soft-landing) and sentiment (multi-angle) — the lever.
    (2) Exp-2 behavioral enforcement-layer curve: deadbeat margin vs enforcement layer.
    (3) Exp-3 coherence-vs-effect Pareto: KL(steered‖unsteered) vs behavioral effect,
        soft-landing (ρ sweep) vs single-layer deadbeat.
    """
    d = np.load(path, allow_pickle=True)
    band = d["band"]; steer = int(d["steer_layer"]); kT = int(d["kT"])
    smoke = bool(d["smoke"])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

    # (1) Exp 1 — authority vs bandwidth (the gate): refusal + sentiment
    rw = d["exp1_refusal_widths"]
    ax[0].plot(rw, d["exp1_refusal_auth_multi"], "o-", color="crimson",
               label="refusal · multi-angle")
    ax[0].plot(rw, d["exp1_refusal_auth_soft"], "s--", color="seagreen",
               label="refusal · soft-landing LQR")
    sw = d["exp1_sent_widths"]
    ax2 = ax[0].twinx()
    ax2.plot(sw, d["exp1_sent_auth_multi"], "^-", color="royalblue", alpha=0.8,
             label="sentiment · multi-angle")
    ax2.set_ylabel("sentiment behavioral range", color="royalblue")
    ax[0].set_xlabel("steered band width (layers)")
    ax[0].set_ylabel("refusal behavioral range", color="crimson")
    ax[0].set_xticks(rw)
    ax[0].set_title("[Exp 1] AUTHORITY GATE: does a wider band\n"
                    "raise behavioral authority vs a single layer?")
    # merge the two legends
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
    ax[0].grid(alpha=0.3)

    # (2) Exp 2 — behavioral enforcement-layer curve (single-layer deadbeat)
    el = d["exp2_enf_layers"]; em = d["exp2_enf_margins"]
    baseline = float(d["exp2_baseline"]); best_enf = int(d["exp2_best_enf"])
    ax[1].plot(el, em, "o-", color="crimson", label="deadbeat margin")
    ax[1].axhline(baseline, color="gray", ls="--", label="baseline (no steer)")
    ax[1].axvline(best_enf, color="seagreen", ls=":", lw=1.5,
                  label=f"best enf. layer {best_enf}")
    ax[1].set_xlabel("single enforcement layer")
    ax[1].set_ylabel("refusal–compliance margin")
    ax[1].set_xticks(el)
    ax[1].set_title("[Exp 2] behavioral enforcement-layer curve\n"
                    "(cross-check of the offline geometric curve)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # (3) Exp 3 — coherence (KL) vs behavioral effect Pareto: soft-landing vs deadbeat
    sl_eff = d["exp3_sl_eff"]; sl_kl = d["exp3_sl_kl"]; rhos = d["exp3_rhos"]
    db_eff = float(d["exp3_db_eff"]); db_kl = float(d["exp3_db_kl"])
    mi = int(d["exp3_match_idx"])
    ax[2].plot(sl_eff, sl_kl, "o-", color="seagreen", label="soft-landing (ρ sweep)")
    for x, y, r in zip(sl_eff, sl_kl, rhos):       # annotate the rho values
        ax[2].annotate(f"ρ={r:.2g}", xy=(x, y), fontsize=6,
                       xytext=(3, 3), textcoords="offset points", color="seagreen")
    ax[2].scatter([db_eff], [db_kl], color="crimson", s=90, marker="X", zorder=5,
                  label="single-layer deadbeat")
    ax[2].scatter([sl_eff[mi]], [sl_kl[mi]], facecolors="none", edgecolors="k",
                  s=140, lw=1.5, zorder=6, label="matched-effect point")
    ax[2].set_xlabel("behavioral effect (steered margin, lower = stronger refusal-break)")
    ax[2].set_ylabel("coherence cost  KL(steered‖unsteered)")
    ax[2].set_title("[Exp 3] coherence vs effect Pareto\n"
                    "(soft-landing matches effect at lower KL)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    tag = "  [SMOKE run — tiny N]" if smoke else ""
    fig.suptitle(f"OAS in the loop — Qwen2.5-3B, band layers {int(band[0])}–{int(band[-1])}, "
                 f"steer={steer}, kT={kT}{tag}", fontsize=15)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    if Path(OFFLINE).exists():
        plot_offline()
    else:
        print(f"(skip offline: {OFFLINE} not found — run oas_offline.py)")
    if Path(PROTO).exists():
        plot_prototype()
    else:
        print(f"(skip prototype: {PROTO} not found — run oas_prototype.py)")
