"""PTS — offline validation of the load-bearing claims from research_proposal_PTS.md.

Runs WITHOUT the model, from the saved phase-portrait trajectories
(phase_portrait_output/<model>/trajectories.npz). Covers:

  Exp 1 (S7.4) Trajectory separation: 2D (c1,c2) vs 1D (b1-projection) class
               separability per layer -> "2D trajectories carry more info".
  Exp 2 (S7.4) Dynamics model validation: fit 2x2 affine A_k,b_k on a TRAIN split,
               report held-out 1-/5-step and full-rollout prediction accuracy.
  Exp 3 (S7.4) Controller comparison (exact-model simulation, common plant) under a
               per-layer control-magnitude budget: PTS-MPC vs myopic (H=1) vs
               fixed-angle (Angular Steering) vs unconstrained LQR. Tracking error
               to the harmless-mean reference + realised-vs-planned angle-only residual.
  Exp 4 (S7.4) Constraint satisfaction: ||u_k|| <= u_max enforced; bounded perturbation.
  Exp 7 (S7.4) Efficiency: QP solve time and storage (2x2 vs d x d).

Writes pts_offline_results.npz for pts_plot.py.

Run:  python pts_offline.py [--traj <path>]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from pts_dynamics import fit_layer_dynamics, validate_dynamics, rollout
from pts_mpc import (MPCController, reference_trajectory, build_condensed_qp,
                     assemble_q, solve_qp)
from pts_controller import lqr_gain_schedule


def circular_mean(angles: np.ndarray) -> float:
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def angle_to_unit(theta):
    return np.stack([np.cos(theta), np.sin(theta)], axis=-1)


# =============================================================================
# Exp 1 — trajectory separation (2D vs 1D)
# =============================================================================


def exp1_separation(hc1, hc2, lc1, lc2, layers, seed=0):
    """Per-layer 5-fold CV AUC of harmful-vs-harmless from 2D coords vs 1D (c1)."""
    L = hc1.shape[1]
    y = np.concatenate([np.ones(hc1.shape[0]), np.zeros(lc1.shape[0])])
    auc_2d, auc_1d = [], []
    for k in range(L):
        X1 = np.concatenate([hc1[:, k], lc1[:, k]])[:, None]               # 1D
        X2 = np.concatenate([np.stack([hc1[:, k], hc2[:, k]], 1),
                             np.stack([lc1[:, k], lc2[:, k]], 1)])         # 2D
        clf = LogisticRegression(max_iter=500)
        auc_1d.append(float(cross_val_score(clf, X1, y, cv=5, scoring="roc_auc").mean()))
        auc_2d.append(float(cross_val_score(clf, X2, y, cv=5, scoring="roc_auc").mean()))
    return np.array(auc_1d), np.array(auc_2d)


# =============================================================================
# Exp 3 — controller comparison in simulation (common exact-model plant)
# =============================================================================


def simulate_closed_loop(A, b, c0, start_layer, layers, policy_fn,
                         actuator="angle"):
    """Roll the fitted dynamics as the plant on the GLOBAL layer grid, starting the
    state at `start_layer` (where the first observation is c0). At each actuated
    layer apply the policy's control u.

    actuator='angle'    : the real device (S5.4) — preserve ||s||, set angle(s+u).
    actuator='additive' : the MPC's own internal model — commit s = s + u.
    The gap between the two on the same controller is the angle-only residual w_j.

    Returns (states (N,L,2) with rows < start_layer = NaN, applied {layer:(N,2)}).
    """
    A = np.asarray(A); b = np.asarray(b)
    N = c0.shape[0]
    L = A.shape[0] + 1
    states = np.full((N, L, 2), np.nan)
    applied = {}
    s = c0.copy()
    layerset = set(layers)
    for k in range(start_layer, L):
        if k in layerset:
            u, _ = policy_fn(k, s)                  # (N,2)
            applied[k] = u
            if actuator == "additive":
                s = s + u
            else:                                  # angle-only (real actuator)
                r = np.linalg.norm(s, axis=-1, keepdims=True)
                tgt = s + u
                ang = np.arctan2(tgt[:, 1], tgt[:, 0])
                s = r * np.stack([np.cos(ang), np.sin(ang)], -1)
        states[:, k, :] = s
        if k < L - 1:
            s = s @ A[k].T + b[k]                   # propagate plant
    return states, applied


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def tracking_error(states, ref, layers):
    """Return (mean angular err [rad], mean coord err, per-layer angular err)."""
    ang_errs, coord_errs = [], []
    for k in layers:
        s = states[:, k, :]
        ang = np.abs(_wrap(np.arctan2(s[:, 1], s[:, 0])
                           - np.arctan2(ref[k, 1], ref[k, 0]))).mean()
        coord = np.linalg.norm(s - ref[k], axis=-1).mean()
        ang_errs.append(ang); coord_errs.append(coord)
    return float(np.mean(ang_errs)), float(np.mean(coord_errs)), np.array(ang_errs)


def make_controllers(A, b, ref, band_layers, H, u_max, fixed_angle):
    """Build the {name: (policy_fn, actuator)} controller suite at a given budget."""
    mpc = MPCController(A, b, ref, layers=band_layers, H=H,
                        q_pos=1.0, r_ctrl=0.02, qf_scale=4.0, u_max=u_max)
    myopic = MPCController(A, b, ref, layers=band_layers, H=1,
                          q_pos=1.0, r_ctrl=0.02, qf_scale=4.0, u_max=u_max)
    lqr_gains = lqr_gain_schedule(A, b, ref, band_layers, q_pos=1.0, r_ctrl=0.02,
                                  qf_scale=4.0, horizon=H)

    def pol_lqr(k, s):
        if k not in lqr_gains:
            return np.zeros_like(s), None
        g = lqr_gains[k]
        return g["c0"] - (s - ref[k]) @ g["K"].T, None

    def pol_fixed(k, s):
        r = np.linalg.norm(s, axis=-1, keepdims=True)
        tgt = r * np.array([np.cos(fixed_angle), np.sin(fixed_angle)])
        return tgt - s, None

    return {
        "PTS-MPC": (lambda k, s: mpc.control(k, s), "angle"),
        f"MPC-myopic(H=1)": (lambda k, s: myopic.control(k, s), "angle"),
        "LQR-unconstrained": (pol_lqr, "angle"),
        "FixedAngle": (pol_fixed, "angle"),
        "No-steer": (lambda k, s: (np.zeros_like(s), None), "angle"),
    }


def run_suite(A, b, ref, c0, start, band_layers, controllers):
    """Run each controller; return {name: dict(ang, coord, umax, errcurve)}."""
    out = {}
    for name, (pol, act) in controllers.items():
        states, applied = simulate_closed_loop(A, b, c0, start, band_layers, pol,
                                               actuator=act)
        ang, coord, curve = tracking_error(states, ref, band_layers)
        umax = max((np.linalg.norm(u, axis=-1).max() for u in applied.values()
                    if u is not None), default=0.0)
        out[name] = {"ang": ang, "coord": coord, "umax": float(umax), "curve": curve}
    return out


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str,
                    default="phase_portrait_output/Qwen2.5-3B-Instruct/trajectories.npz")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--u-max", type=float, default=0.5,
                    help="control-magnitude budget as a fraction of mean ref scale")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.traj)
    hc1, hc2 = d["harmful_c1"], d["harmful_c2"]
    lc1, lc2 = d["harmless_c1"], d["harmless_c2"]
    layers = d["layers"]
    L = hc1.shape[1]
    b1, b2 = d["b1"], d["b2"]
    print(f"loaded {args.traj}: harmful {hc1.shape}, harmless {lc1.shape}, L={L}")

    rng = np.random.RandomState(args.seed)

    # ---------------------------------------------------------------- Exp 1
    print("\n[Exp 1] trajectory separation (5-fold CV AUC, harmful vs harmless)")
    auc_1d, auc_2d = exp1_separation(hc1, hc2, lc1, lc2, layers, args.seed)
    gain = auc_2d - auc_1d
    print(f"  mean AUC: 1D(b1 only)={auc_1d.mean():.3f}  2D(plane)={auc_2d.mean():.3f}  "
          f"(2D-1D)={gain.mean():+.3f}")
    best_layer = int(np.argmax(auc_2d))
    print(f"  best layer {layers[best_layer]}: 1D={auc_1d[best_layer]:.3f} "
          f"2D={auc_2d[best_layer]:.3f}")

    # steering band: anchored on the *refusal geometry* — the contiguous late-depth
    # run where the harmful and harmless mean trajectories are angularly separated
    # (|angle_sep| > 0.5 rad), which is where the steering plane is behaviourally
    # meaningful (it contains the CLAS steer layer 27 for Qwen2.5-3B). This is the
    # band the MPC slides its horizon through; we never actuate the final layer.
    harmful_mean = np.stack([hc1.mean(0), hc2.mean(0)], 1)    # (L,2)
    harmless_mean = np.stack([lc1.mean(0), lc2.mean(0)], 1)
    angle_sep = np.abs(_wrap(np.arctan2(harmful_mean[:, 1], harmful_mean[:, 0])
                             - np.arctan2(harmless_mean[:, 1], harmless_mean[:, 0])))
    peak = int(np.argmax(angle_sep))
    sep_hi = angle_sep > 0.5
    lo = peak
    while lo - 1 >= 0 and sep_hi[lo - 1]:
        lo -= 1
    himax = peak
    while himax + 1 < L - 1 and sep_hi[himax + 1]:
        himax += 1
    band = (lo, himax)
    print(f"  steering band (|angle_sep|>0.5 rad around peak layer {layers[peak]}): "
          f"layers {layers[band[0]]}..{layers[band[1]]}")

    # train/test split shared by Exp 2/3
    nH, nL = hc1.shape[0], lc1.shape[0]
    hi = rng.permutation(nH); li = rng.permutation(nL)
    htr, hte = hi[:nH // 2], hi[nH // 2:]
    ltr, lte = li[:nL // 2], li[nL // 2:]

    # ---------------------------------------------------------------- Exp 2
    print("\n[Exp 2] dynamics model validation (2x2 affine, held-out)")
    tr_c1 = np.concatenate([hc1[htr], lc1[ltr]]); tr_c2 = np.concatenate([hc2[htr], lc2[ltr]])
    te_c1 = np.concatenate([hc1[hte], lc1[lte]]); te_c2 = np.concatenate([hc2[hte], lc2[lte]])
    fit = fit_layer_dynamics(tr_c1, tr_c2, affine=True)
    fit_lin = fit_layer_dynamics(tr_c1, tr_c2, affine=False)
    val = validate_dynamics(te_c1, te_c2, fit, horizons=(1, 5))
    val_lin = validate_dynamics(te_c1, te_c2, fit_lin, horizons=(1, 5))
    print(f"  affine  : R2(1-step)={val['r2_h1']:.4f}  R2(5-step)={val['r2_h5']:.4f}  "
          f"NRMSE(1)={val['nrmse_h1']:.4f}  full-rollout RMSE={val['full_rollout_rmse']:.3f}")
    print(f"  linear  : R2(1-step)={val_lin['r2_h1']:.4f}  R2(5-step)={val_lin['r2_h5']:.4f}  "
          f"(affine term matters: dR2_1step={val['r2_h1']-val_lin['r2_h1']:+.4f})")

    # ---------------------------------------------------------------- Exp 3
    print("\n[Exp 3] controller comparison in simulation (common fitted-model plant)")
    A, b = fit["A"], fit["b"]
    ref = reference_trajectory(harmful_mean, harmless_mean, option="A")  # track harmless
    band_layers = list(range(band[0], band[1] + 1))
    ref_scale = np.linalg.norm(ref[band_layers], axis=-1).mean()
    fixed_angle = circular_mean(np.arctan2(ref[band_layers, 1], ref[band_layers, 0]))
    c0 = np.stack([hc1[hte, band[0]], hc2[hte, band[0]]], 1)   # held-out harmful ICs
    print(f"  reference = harmless mean; actuate layers {layers[band_layers[0]]}.."
          f"{layers[band_layers[-1]]} ({len(band_layers)} layers); ref scale~{ref_scale:.2f}")
    print(f"  best fixed angle over band = {np.degrees(fixed_angle):.0f}deg")

    # ---- 3a. tracking-vs-perturbation Pareto (S7.4 Exp 3) ----
    # The honest axis is the *realised* in-plane perturbation ||u|| (how far off the
    # reachable manifold the actuation pushes — cf. the non-surjectivity result,
    # arXiv:2604.09839), NOT the budget knob. FixedAngle and the unconstrained LQR
    # ignore the budget (they snap to the target at full magnitude), so they are
    # single high-perturbation points; PTS-MPC traces a tunable frontier whose u_max
    # we dial. The claim is: at MATCHED perturbation, PTS tracks no worse, and PTS can
    # reach low-perturbation operating points the baselines cannot.
    budgets = np.array([0.10, 0.15, 0.22, 0.35, 0.55, 1.0]) * ref_scale
    names = ["PTS-MPC", "MPC-myopic(H=1)", "LQR-unconstrained", "FixedAngle", "No-steer"]
    sweep_ang = {n: [] for n in names}
    sweep_umax = {n: [] for n in names}
    print("  3a. tracking-vs-perturbation sweep  [ang-err deg @ realised ||u||]:")
    header = "      u_max  " + "".join(f"{n[:12]:>17s}" for n in names)
    print(header)
    for um in budgets:
        ctrls = make_controllers(A, b, ref, band_layers, args.horizon, um, fixed_angle)
        res = run_suite(A, b, ref, c0, band[0], band_layers, ctrls)
        row = f"      {um:5.2f}  "
        for n in names:
            sweep_ang[n].append(np.degrees(res[n]["ang"]))
            sweep_umax[n].append(res[n]["umax"])
            row += f"{np.degrees(res[n]['ang']):7.1f}@{res[n]['umax']:<8.2f}"
        print(row)
    # matched-perturbation headline: PTS at the budget closest to FixedAngle's ||u||
    u_fixed = sweep_umax["FixedAngle"][0]
    j = int(np.argmin(np.abs(np.array(sweep_umax["PTS-MPC"]) - u_fixed)))
    print(f"  -> at FixedAngle's perturbation ||u||~{u_fixed:.2f}: PTS-MPC tracks "
          f"{sweep_ang['PTS-MPC'][j]:.1f}deg vs FixedAngle {sweep_ang['FixedAngle'][0]:.1f}deg; "
          f"and PTS can dial down to ||u||~{sweep_umax['PTS-MPC'][0]:.2f} "
          f"(FixedAngle/LQR cannot reduce perturbation at all).")

    # ---- 3b. horizon ablation at a tight budget (does lookahead help?) ----
    tight = 0.15 * ref_scale
    print(f"\n  3b. horizon ablation at tight budget u_max={tight:.2f}:")
    horizons = [1, 2, 3, 5, 8]
    h_ang = []
    for H in horizons:
        c = MPCController(A, b, ref, layers=band_layers, H=H, q_pos=1.0,
                          r_ctrl=0.02, qf_scale=4.0, u_max=tight)
        res = run_suite(A, b, ref, c0, band[0], band_layers,
                        {"mpc": (lambda k, s, c=c: c.control(k, s), "angle")})
        h_ang.append(np.degrees(res["mpc"]["ang"]))
        print(f"      H={H}: ang-err={h_ang[-1]:.1f}deg")

    # 3b explanation: characterise the per-layer dynamics. Lookahead can only help
    # where the plant rotates/expands meaningfully between actuations; if A_k ~ I in
    # the behavioural band, myopic feedback is already near-optimal.
    rot_mag = np.array([np.linalg.norm(A[k] - np.eye(2), 2) for k in range(L - 1)])
    print(f"      ||A_k - I||_2 over band = {rot_mag[band_layers].mean():.3f} "
          f"(near 0 -> benign, near-identity late dynamics; lookahead has little to "
          f"exploit here — the predictive value would surface in rotation-dominated bands)")

    # headline operating point (tight budget) for the saved table
    ctrls_h = make_controllers(A, b, ref, band_layers, args.horizon, tight, fixed_angle)
    results = run_suite(A, b, ref, c0, band[0], band_layers, ctrls_h)
    print(f"\n  headline @ u_max={tight:.2f}: "
          + "  ".join(f"{n}={np.degrees(results[n]['ang']):.1f}deg"
                      for n in ["PTS-MPC", "FixedAngle", "No-steer"]))

    # ---------------------------------------------------------------- Exp 4
    print(f"\n[Exp 4] constraint satisfaction at tight budget u_max={tight:.2f}")
    con_ok = results["PTS-MPC"]["umax"] <= tight + 1e-6
    lqr_viol = results["LQR-unconstrained"]["umax"] > tight
    print(f"  PTS-MPC   max||u||={results['PTS-MPC']['umax']:.3f} <= u_max : {con_ok}")
    print(f"  LQR(unc.) max||u||={results['LQR-unconstrained']['umax']:.3f} "
          f"-> exceeds budget by {results['LQR-unconstrained']['umax']/tight:.1f}x : {lqr_viol}")
    u_max = tight

    # ---------------------------------------------------------------- Exp 7
    print("\n[Exp 7] efficiency")
    cqp = build_condensed_qp(A, b, band_layers[0], args.horizon,
                             np.eye(2), 0.02 * np.eye(2), 4 * np.eye(2))
    xi_batch = c0
    q = assemble_q(cqp, xi_batch, ref[band_layers[0]:band_layers[0] + args.horizon + 1])
    # constrained (FISTA) and unconstrained (single linear solve / "explicit MPC") costs
    t0 = time.perf_counter()
    for _ in range(200):
        _ = solve_qp(cqp["P"], q, tight, args.horizon, L=cqp["L"], iters=60)
    dt = (time.perf_counter() - t0) / 200
    t0 = time.perf_counter()
    for _ in range(2000):
        _ = solve_qp(cqp["P"], q, None, args.horizon)
    dt_unc = (time.perf_counter() - t0) / 2000
    per_item = dt / xi_batch.shape[0] * 1e6
    d_model = b1.shape[0]
    storage_pts = (A.nbytes + b.nbytes)
    print(f"  QP solve (batch={xi_batch.shape[0]}, H={args.horizon}): "
          f"constrained-FISTA {dt*1e3:.3f} ms ({per_item:.1f} us/prompt); "
          f"unconstrained {dt_unc/xi_batch.shape[0]*1e6:.2f} us/prompt")
    print(f"  PTS storage (all A_k,b_k): {storage_pts} bytes "
          f"(~{storage_pts/(L-1):.0f} B/layer) vs A-LQR d x d K per layer "
          f"= {d_model*d_model*4/1e6:.1f} MB/layer ({d_model*d_model*4//(storage_pts//(L-1)):,}x larger)")

    # ---------------------------------------------------------------- save
    out = Path(args.traj).parent
    np.savez_compressed(
        out / "pts_offline_results.npz",
        layers=layers, auc_1d=auc_1d, auc_2d=auc_2d, band=np.array(band),
        per_layer_r2=np.array(val["per_layer_r2_1step"]),
        per_layer_r2_lin=np.array(val_lin["per_layer_r2_1step"]),
        full_rollout_rmse_per_layer=np.array(val["full_rollout_rmse_per_layer"]),
        r2_h1=val["r2_h1"], r2_h5=val["r2_h5"],
        harmful_mean=harmful_mean, harmless_mean=harmless_mean, ref=ref,
        band_layers=np.array(band_layers), u_max=u_max, fixed_angle=fixed_angle,
        angle_sep=angle_sep,
        track_names=np.array(list(results.keys())),
        track_ang=np.array([results[k]["ang"] for k in results]),
        track_coord=np.array([results[k]["coord"] for k in results]),
        track_umax=np.array([results[k]["umax"] for k in results]),
        track_curves=np.array([results[k]["curve"] for k in results]),
        rot_mag=rot_mag,
        sweep_budgets=budgets,
        sweep_names=np.array(names),
        sweep_ang=np.array([sweep_ang[n] for n in names]),
        sweep_umax=np.array([sweep_umax[n] for n in names]),
        ablation_horizons=np.array(horizons),
        ablation_ang=np.array(h_ang),
        qp_ms=dt * 1e3, qp_us_per_prompt=per_item,
    )
    print(f"\nsaved {out/'pts_offline_results.npz'}")


if __name__ == "__main__":
    main()
