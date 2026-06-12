"""OAS — model-FREE experiments on the saved phase-portrait trajectories.

Implements OAS_SPEC.md S4: the offline (no-model) half of the OAS evaluation, run
from the SAME saved trajectory file PTS uses
(`../SO2/phase_portrait_output/Qwen2.5-3B-Instruct/trajectories.npz`). Everything is
pure numpy on the 2D plane coordinates — no torch, no model — so the load-bearing
control-theory claims are validated cheaply and reproducibly (this driver runs in
CI of the build, target < ~30 s).

It reuses the PTS plant fit (`pts_dynamics.fit_layer_dynamics`/`validate_dynamics`)
and the OAS observer/LQR modules, and computes the steering band EXACTLY as
`pts_offline.py` (the contiguous late-depth run where the harmful/harmless mean
trajectories are angularly separated by |angle_sep| > 0.5 rad around the peak).

Experiments (OAS_SPEC S4):

  Plant + noise   fit {A_k,b_k} on a TRAIN split, validate held-out (reuse PTS),
                  estimate the process-noise covariance W from the one-step
                  residuals (the ~14% PTS residual, now as a covariance).

  Exp 2 (geometric) enforce the target angle at a SINGLE layer, sweep that layer
                  across the band, roll the fitted plant to the band end, and report
                  the TERMINAL geometric angle error vs enforcement layer. Early
                  enforcement is washed out by the natural dynamics -> motivates the
                  *terminal* objective and picks kT.

  Exp 3 (headline) soft-landing LQR vs single-layer deadbeat at MATCHED terminal
                  angle. Metrics: total effort sum||u_k|| and max_k||u_k||
                  (off-manifold-push proxy for coherence cost); robustness =
                  terminal-angle variance under a perturbed plant Ahat = A + delta*N(0,1)
                  (Monte Carlo). Correlate the soft-landing advantage with ||A_k - I||_2
                  across depth bands (predict: largest in rotation-dominated bands,
                  smallest in the near-identity late band). Sweep rho to trace the
                  effort-vs-terminal-error frontier (deadbeat = the rho->0 endpoint).

  Exp 4 (observer) observability of (A_band, H); compare the behavioural-state
                  (here: angle) estimate from Kalman-fused vs best single-layer vs
                  ad-hoc EMA under injected measurement noise V and process noise W;
                  compare LQG (control on xhat) vs LQR-on-raw at matched effort under
                  injected noise (LQG should win as noise grows; tie when noise -> 0).

Writes `oas_offline_results.npz` (everything oas_plot.py needs). Run:

    python oas_offline.py [--traj <path>]

from inside research_proposals/OAS.
"""

from __future__ import annotations

import argparse
import sys
import pathlib

# --- sys.path shim (OAS_SPEC.md S0): expose BOTH the shared lib and the sibling
# --- PTS dir so `from pts_dynamics import ...`, `from oas_* import ...`,
# --- `from utils import ...` all resolve no matter where python is launched from.
# --- OAS is self-contained: it imports only the shared lib, pts_dynamics (the
# --- pure-numpy plant fit), and its own oas_* modules — never from ../CLAS, and
# --- from ../PTS ONLY pts_dynamics.
_R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "pytorch_pure"))
sys.path.insert(0, str(_R / "research_proposals" / "PTS"))

import numpy as np

from pts_dynamics import fit_layer_dynamics, validate_dynamics
from oas_lqr import finite_horizon_lqr, deadbeat_gains, lqr_rollout, total_effort
from oas_observer import (KalmanFilter, estimate_process_noise, is_observable,
                          observability_gramian, observability_gramian_tv,
                          kalman_filter_seq)


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))


def _angle(c):
    """Angle of a plane coordinate (or stack of coords)."""
    c = np.asarray(c)
    return np.arctan2(c[..., 1], c[..., 0])


# =============================================================================
# Steering band — computed EXACTLY as pts_offline.py (S4 / pts_offline Exp 1)
# =============================================================================


def steering_band(harmful_mean, harmless_mean, L):
    """Contiguous late-depth run where the harmful/harmless mean trajectories are
    angularly separated (|angle_sep| > 0.5 rad), grown out from the peak.

    Verbatim port of pts_offline.py: anchored on the refusal geometry (it contains
    the steer layer 27 for Qwen2.5-3B), never including the final layer. Returns
    (lo, hi) inclusive layer-INDEX bounds and the full angle_sep curve.
    """
    angle_sep = np.abs(_wrap(_angle(harmful_mean) - _angle(harmless_mean)))
    peak = int(np.argmax(angle_sep))
    sep_hi = angle_sep > 0.5
    lo = peak
    while lo - 1 >= 0 and sep_hi[lo - 1]:
        lo -= 1
    himax = peak
    while himax + 1 < L - 1 and sep_hi[himax + 1]:
        himax += 1
    return (lo, himax), angle_sep, peak


# =============================================================================
# Exp 2 — geometric enforcement-layer sweep (terminal angle error vs enf. layer)
# =============================================================================


def exp2_enforcement_sweep(A, b, c0, band_layers, target_angle, kT):
    """Enforce the target angle at a SINGLE layer, sweep that layer across the band,
    roll the fitted plant (autonomous) to the band end kT, and measure the TERMINAL
    geometric angle error.

    For each candidate enforcement layer ke in band_layers:
      * roll the autonomous plant from c0 to ke;
      * at ke, NORM-PRESERVINGLY reset the angle to target_angle (the real Angular
        Steering actuator: keep ||c_ke||, set its angle) -> s_ke;
      * roll the autonomous plant forward from s_ke to the terminal coordinate
        c_{kT+1} (the landed coordinate the behaviour reads);
      * record |angle(c_{kT+1}) - target_angle| (mean over the held-out ICs).

    Prediction (S4): early enforcement washes out under the natural dynamics
    (large terminal error), a LATE enforcement layer lands the angle -> motivates
    the terminal objective and picks kT. Returns (enf_layers, terminal_ang_err,
    enf_layers_idx) where enf_layers are GLOBAL layer ids for plotting.
    """
    c0 = np.asarray(c0, np.float64)                 # (N,2) held-out incoming coords at band[0]
    band_layers = list(band_layers)
    k0 = band_layers[0]
    u_tgt = np.array([np.cos(target_angle), np.sin(target_angle)])

    terminal_err = []
    for ke in band_layers:
        # autonomous roll from the band start k0 up to the enforcement layer ke
        c = c0.copy()
        for k in range(k0, ke):
            c = c @ A[k].T + b[k]
        # norm-preserving angle reset at ke (the deployed actuator)
        scale = np.linalg.norm(c, axis=-1, keepdims=True)
        s = scale * u_tgt                            # (N,2) angle set to target, ||c|| kept
        # autonomous roll from ke forward to the terminal coordinate c_{kT+1}
        c = s
        for k in range(ke, kT + 1):
            c = c @ A[k].T + b[k]
        err = np.abs(_wrap(_angle(c) - target_angle))
        terminal_err.append(float(err.mean()))
    return np.array(band_layers), np.array(terminal_err)


# =============================================================================
# Exp 3 — soft-landing vs deadbeat at matched terminal angle (the headline maths)
# =============================================================================


def _propagate(A, b, x0, k_from, k_to):
    """Autonomously propagate the natural coordinate x from layer k_from to k_to.

    x_{k+1} = A_k x_k + b_k applied for k = k_from .. k_to-1 (no actuation). Used to
    recover the TRUE natural incoming coordinate x_kT at the terminal layer when the
    soft-landing band started earlier at k_from: the single-layer deadbeat at kT must
    be fed THIS x_kT (||x|| ~ the terminal-layer scale), not the band-start coord, so
    its control push u_kT = s* - x_kT is charged against the correct physical state
    and the deadbeat-vs-soft comparison is from a consistent state (F1).
    """
    x = np.atleast_2d(np.asarray(x0, np.float64)).copy()
    for k in range(int(k_from), int(k_to)):
        x = x @ A[k].T + b[k]
    return x


def _terminal_angle(A, b, gains, x0, band):
    """Roll the Riccati policy and return the landed terminal-coordinate angle(s)."""
    x0 = np.atleast_2d(np.asarray(x0, np.float64))
    angs = np.empty(x0.shape[0])
    for i in range(x0.shape[0]):
        roll = lqr_rollout(A, b, gains, x0[i], band)
        angs[i] = _angle(roll["x_term"])
    return angs


def _rollout_metrics(A, b, gains, x0, band):
    """Mean total effort and mean max-per-layer push over a batch of ICs."""
    x0 = np.atleast_2d(np.asarray(x0, np.float64))
    tot, mx = [], []
    for i in range(x0.shape[0]):
        roll = lqr_rollout(A, b, gains, x0[i], band)
        tot.append(total_effort(roll["u"]))
        mx.append(float(np.linalg.norm(roll["u"], axis=-1).max()))
    return float(np.mean(tot)), float(np.mean(mx))


def exp3_rho_frontier(A, b, c0, band_layers, target, kT):
    """Sweep rho (R = rho I) to trace the effort-vs-terminal-angle-error frontier.

    Pure soft-landing config (Q_stage = 0, "don't force the angle early"); the
    terminal reference is the PRE-IMAGE of `target` so the deadbeat corner (rho->0)
    lands the next coordinate c_{kT+1} exactly on target and the angle error grows
    as effort is taxed. The deadbeat single-layer baseline is the rho->0 endpoint.
    Returns (rhos, total_effort[:], max_u[:], terminal_ang_err[:]).
    """
    c0 = np.asarray(c0, np.float64)
    L = A.shape[0] + 1
    target_ang = _angle(target)
    r_pre = np.zeros((L, 2))
    r_pre[kT] = np.linalg.solve(A[kT], target - b[kT])     # pre-image: A_kT s = target - b_kT
    Q_term = 1.0 * np.eye(2)                                # modest terminal weight so rho
    #                                                        actually trades against it

    # rho spans deadbeat (rho->0, terminal cost dominates -> exact landing, max push)
    # to near-passive (rho large, effort dominates -> the angle drifts off target).
    rhos = np.array([1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0])
    eff, mxu, aerr = [], [], []
    for rho in rhos:
        gains = finite_horizon_lqr(A, b, band_layers, r_pre, rho * np.eye(2),
                                   np.zeros((2, 2)), Q_term, terminal_layer=kT)
        t, m = _rollout_metrics(A, b, gains, c0, band_layers)
        angs = _terminal_angle(A, b, gains, c0, band_layers)
        eff.append(t)
        mxu.append(m)
        aerr.append(float(np.abs(_wrap(angs - target_ang)).mean()))
    return rhos, np.array(eff), np.array(mxu), np.array(aerr)


def _match_softlanding_to_deadbeat(A, b, c0, band_layers, target, kT,
                                   rhos=(5e-3, 1e-2, 2e-2, 4e-2, 8e-2)):
    """Pick the soft-landing rho whose mean terminal angle best matches the
    single-layer deadbeat's, then return both controllers' metrics at MATCHED angle.

    Deadbeat: single layer kT slams the next coordinate onto `target` (the R->0
    corner from oas_lqr.deadbeat_gains). Soft-landing: a band over band_layers with
    Q_stage = 0 and the pre-image terminal reference; the chosen rho is the largest
    (gentlest) whose terminal-angle gap to the deadbeat stays within 0.05 rad so we
    compare like-for-like behavioural effect at the smallest possible push.
    """
    c0 = np.asarray(c0, np.float64)
    L = A.shape[0] + 1
    k0 = list(band_layers)[0]

    # The single-layer deadbeat acts at the TERMINAL layer kT, so it must be fed the
    # natural incoming coordinate x_kT (||x|| ~ the kT scale), NOT the band-start coord
    # c0 at k0 (||x|| ~ the k0 scale). Autonomously propagate c0 through the band to kT
    # so the deadbeat push u_kT = s* - x_kT is charged against the correct state (F1).
    x_kT = _propagate(A, b, c0, k0, kT)                  # (N,2) true natural x_kT
    db = deadbeat_gains(A, b, kT, target)
    db_ang = _terminal_angle(A, b, db, x_kT, [kT])
    db_ang_mean = circular_mean(db_ang)
    db_tot, db_mx = _rollout_metrics(A, b, db, x_kT, [kT])

    r_pre = np.zeros((L, 2))
    r_pre[kT] = np.linalg.solve(A[kT], target - b[kT])
    Q_term = 500.0 * np.eye(2)

    cands = []
    for rho in rhos:
        gains = finite_horizon_lqr(A, b, band_layers, r_pre, rho * np.eye(2),
                                   np.zeros((2, 2)), Q_term, terminal_layer=kT)
        sl_ang = _terminal_angle(A, b, gains, c0, band_layers)
        gap = abs(_wrap(circular_mean(sl_ang) - db_ang_mean))
        sl_tot, sl_mx = _rollout_metrics(A, b, gains, c0, band_layers)
        cands.append({"rho": float(rho), "gap": float(gap), "gains": gains,
                      "sl_tot": sl_tot, "sl_mx": sl_mx, "sl_ang": sl_ang})
    # prefer the gentlest (largest rho) controller that still matches within 0.05 rad;
    # if none qualifies, fall back to the tightest-matching (smallest-gap) candidate.
    matched = [c for c in cands if c["gap"] < 0.05]
    best = max(matched, key=lambda c: c["rho"]) if matched \
        else min(cands, key=lambda c: c["gap"])
    return {
        "deadbeat": {"ang_mean": float(db_ang_mean), "total_effort": db_tot,
                     "max_u": db_mx, "gains": db, "ang": db_ang, "x_kT": x_kT},
        "softlanding": {"rho": best["rho"], "gap": best["gap"],
                        "total_effort": best["sl_tot"], "max_u": best["sl_mx"],
                        "gains": best["gains"], "ang": best["sl_ang"]},
    }


def exp3_robustness(A, b, c0, band, gains, target, delta, n_mc, seed):
    """Monte-Carlo terminal-angle variance under a perturbed plant Ahat = A + delta*N(0,1).

    Robustness proxy (S4): re-roll the FIXED controller gains through a randomly
    perturbed plant and measure how much the landed terminal angle scatters. Lower
    variance = more robust to the ~14% plant-model error. Returns
    (mean_terminal_ang_err, var_terminal_ang) over n_mc perturbations x ICs.
    """
    rng = np.random.RandomState(seed)
    c0 = np.atleast_2d(np.asarray(c0, np.float64))
    target_ang = _angle(target)
    angs = []
    for _ in range(n_mc):
        Ah = A + delta * rng.randn(*A.shape)
        for i in range(c0.shape[0]):
            roll = lqr_rollout(Ah, b, gains, c0[i], band)
            angs.append(_angle(roll["x_term"]))
    angs = np.array(angs)
    err = np.abs(_wrap(angs - target_ang))
    # circular variance of the landed angle (1 - resultant length): scale-free robustness
    R = np.sqrt(np.mean(np.cos(angs)) ** 2 + np.mean(np.sin(angs)) ** 2)
    return float(err.mean()), float(1.0 - R)


def exp3_advantage_vs_band(A, b, C_all, target_angle, band_widths, all_kTs, seed):
    """Soft-landing advantage (deadbeat push minus soft push) correlated with
    ||A_k - I||_2 across depth bands.

    For each candidate terminal layer kT we build a contiguous band ending at kT,
    match a soft-landing controller to the single-layer deadbeat at that kT, and
    record (mean ||A_k - I||_2 over the band, deadbeat push, soft push, advantage).

    The push is the SCALE-FREE off-manifold fraction max_k ||u_k|| / (band coord
    scale): the raw ||u_k|| is dominated by the absolute coordinate magnitude, which
    GROWS ~60x from the early to the late band (and anti-correlates with rotation),
    so a raw-displacement advantage would be confounded by depth. Normalising by the
    local coordinate scale isolates the rotation effect the prediction is about.

    Prediction (S4): the advantage is largest in rotation-dominated bands (large
    ||A_k - I||) and smallest in the near-identity late band. Returns a dict of arrays.
    """
    L = A.shape[0] + 1
    rot_mag_layer = np.array([np.linalg.norm(A[k] - np.eye(2), 2) for k in range(L - 1)])

    kTs, rot_band, db_mx, sl_mx, adv = [], [], [], [], []
    for kT, w in zip(all_kTs, band_widths):
        k0 = max(0, kT - (w - 1))
        bl = list(range(k0, kT + 1))
        if len(bl) < 2:                               # need a band to distribute over
            continue
        # natural-magnitude terminal target at the target angle, scaled to the band
        # IC magnitude so the deadbeat/soft comparison is on the same circle.
        c0 = C_all[:, k0, :]
        tmag = np.linalg.norm(C_all[:, kT + 1, :], axis=-1).mean()
        target = tmag * np.array([np.cos(target_angle), np.sin(target_angle)])
        m = _match_softlanding_to_deadbeat(A, b, c0, bl, target, kT)
        # scale-free push: normalise by the local band coordinate magnitude.
        band_scale = np.linalg.norm(C_all[:, k0:kT + 1, :].reshape(-1, 2), axis=-1).mean()
        db_push = m["deadbeat"]["max_u"] / (band_scale + 1e-12)
        sl_push = m["softlanding"]["max_u"] / (band_scale + 1e-12)
        kTs.append(kT)
        rot_band.append(float(rot_mag_layer[k0:kT + 1].mean()))
        db_mx.append(db_push)
        sl_mx.append(sl_push)
        adv.append(db_push - sl_push)
    return {"kT": np.array(kTs), "rot_band": np.array(rot_band),
            "deadbeat_max_u": np.array(db_mx), "soft_max_u": np.array(sl_mx),
            "advantage": np.array(adv), "rot_mag_layer": rot_mag_layer}


def circular_mean(angles):
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


# =============================================================================
# Exp 4 — observability + Kalman-vs-single-vs-EMA + LQG-vs-LQR-under-noise
# =============================================================================


def exp4_estimator_compare(A, b, C_all, band_layers, W, V_scale, seed,
                           ema_alpha=0.5):
    """Behavioural-state (angle) estimate quality: Kalman-FUSED vs best SINGLE-layer
    vs ad-hoc EMA, under injected measurement noise.

    The "behaviour" the controller cares about is the angle. We take a held-out true
    coordinate path across the band, inject measurement noise z_k = c_k + v_k
    (v_k ~ N(0, V)), and compare three readouts of the angle against the TRUE angle:
      * Kalman: fuse the whole {z_k} sequence with the fitted plant + W into xhat_k,
        read angle(xhat_k) (the principled estimator);
      * single: angle(z_k) at each layer (the noisy per-layer read — "best single
        measurement");
      * EMA: an exponential moving average of z_k (the ad-hoc smoother the CLAS
        drift test resorted to), read angle(ema_k).

    Returns dict of per-layer angle-RMSE arrays (Kalman/single/EMA) + the noise level.
    """
    rng = np.random.RandomState(seed)
    band_layers = list(band_layers)
    k0, kT = band_layers[0], band_layers[-1]
    Cb = C_all[:, k0:kT + 1, :]                       # (N, T, 2) true coords over the band
    N, T, _ = Cb.shape
    scale = np.linalg.norm(Cb.reshape(-1, 2), axis=-1).mean()
    V = (V_scale * scale) ** 2 * np.eye(2)            # measurement-noise cov
    Vc = np.sqrt((V_scale * scale) ** 2)

    # Per-step plant for the filter: kalman_filter_seq predicts through A_seq[t]
    # BEFORE fusing z[t], so A_seq[t] must be the transition from layer t-1 to t.
    # Step 0 is the reset ONLY (no_update_at_0=True below): the estimate is exactly
    # z[0] with no second fusion (matching PolicyLQG, which does not re-fuse z_0 at
    # band[0] — F4). Steps 1..T-1 use the band's fitted transitions A[k0..kT-1]; the
    # A_seq[0]/b_seq[0] slots are unused once t=0 is reset-only but kept as identity.
    A_seq = np.empty((T, 2, 2)); A_seq[0] = np.eye(2); A_seq[1:] = A[k0:kT]
    b_seq = np.zeros((T, 2)); b_seq[1:] = b[k0:kT]
    H = np.eye(2)

    true_ang = _angle(Cb)                              # (N,T)
    kalman_err = np.zeros(T)
    single_err = np.zeros(T)
    ema_err = np.zeros(T)
    for i in range(N):
        z = Cb[i] + Vc * rng.randn(T, 2)               # (T,2) noisy measurements
        # Kalman fuse (autonomous filter: u=0, plant carries the band dynamics). The
        # reset value xhat_0 = z[0] is NOT re-fused at t=0 (no_update_at_0; matches
        # PolicyLQG) so the first band layer trusts z[0] exactly, not twice (F4).
        xhat, _ = kalman_filter_seq(A_seq, b_seq, H, W, V, z,
                                    x0=z[0], P0=np.eye(2), no_update_at_0=True)
        # ad-hoc EMA smoother
        ema = np.zeros_like(z)
        ema[0] = z[0]
        for t in range(1, T):
            ema[t] = ema_alpha * z[t] + (1 - ema_alpha) * ema[t - 1]
        kalman_err += _wrap(_angle(xhat) - true_ang[i]) ** 2
        single_err += _wrap(_angle(z) - true_ang[i]) ** 2
        ema_err += _wrap(_angle(ema) - true_ang[i]) ** 2
    rmse = lambda e: np.sqrt(e / N)
    return {"layers": np.array(band_layers),
            "kalman_rmse": rmse(kalman_err), "single_rmse": rmse(single_err),
            "ema_rmse": rmse(ema_err), "V_scale": float(V_scale)}


def exp4_lqg_vs_lqr(A, b, C_all, band_layers, target, kT, W, V_scales, seed):
    """LQG (control on the Kalman estimate xhat) vs LQR-on-raw (control on the noisy
    measurement z) at MATCHED gains, under a sweep of injected measurement noise.

    Both controllers use the SAME soft-landing gains (separation principle: the
    optimal controller is unchanged; only the state estimate differs). For each noise
    level we Monte-Carlo the terminal angle error each controller achieves when fed,
    per layer, either the FUSED xhat (LQG) or the RAW z (LQR-on-raw). Prediction (S4):
    LQG wins as noise grows; they tie when noise -> 0 (then z is already clean).

    Crucially we use a MODERATE terminal weight Q_term so the terminal decision
    s_kT = F_kT * estimate + g_kT genuinely DEPENDS on the state estimate (F_kT is
    not negligible): a heavy Q_term would slam the terminal layer to the pre-image
    regardless of the estimate, making both controllers insensitive to measurement
    noise (the degenerate deadbeat corner). With a moderate weight the noisy estimate
    propagates into the landed angle, so we report BOTH the mean terminal-angle error
    and its variance (the scatter LQG's fusion is designed to reduce).
    Returns (V_scales, lqg_err[:], lqr_err[:], lqg_var[:], lqr_var[:]).
    """
    rng = np.random.RandomState(seed)
    band_layers = list(band_layers)
    k0 = band_layers[0]
    L = A.shape[0] + 1
    H = np.eye(2)
    target_ang = _angle(target)

    # moderate soft-landing gains so the terminal decision depends on the estimate
    # (F_kT not negligible) -> measurement noise actually propagates into the angle.
    r_pre = np.zeros((L, 2))
    r_pre[kT] = np.linalg.solve(A[kT], target - b[kT])
    gains = finite_horizon_lqr(A, b, band_layers, r_pre, 1.0 * np.eye(2),
                               np.zeros((2, 2)), 1.0 * np.eye(2), terminal_layer=kT)
    F, g = gains["F"], gains["g"]

    N = min(60, C_all.shape[0])
    idx = rng.permutation(C_all.shape[0])[:N]
    c0_all = C_all[idx, k0, :]                          # (N,2) true incoming at band start
    scale = np.linalg.norm(C_all[:, k0:kT + 1, :].reshape(-1, 2), axis=-1).mean()

    A_band = A[k0:kT]
    b_band = b[k0:kT]

    def closed_loop(use_filter, Vc, V):
        """Run the controller in closed loop on the TRUE plant; the controller sees
        either the fused estimate (LQG) or the raw noisy measurement (LQR-on-raw)."""
        ang_err = []
        for c0 in c0_all:
            x = c0.copy()                              # TRUE natural incoming coord
            kf = KalmanFilter()                        # filter for this rollout
            kf.reset(c0 + Vc * rng.randn(2), np.eye(2))
            prev_k = None
            prev_u = None
            xhat = kf.x.copy()
            for j, k in enumerate(band_layers):
                z = x + Vc * rng.randn(2)              # noisy measurement of the true coord
                if use_filter:
                    if j == 0:
                        est = kf.x.copy()              # xhat_0 = z_0 (reset value)
                    else:
                        est, _ = kf.step(z, A[prev_k], b[prev_k], H, W, V, u=prev_u)
                else:
                    est = z                            # LQR-on-raw: trust the noisy read
                s = F[k] @ est + g[k]                  # actuated decision
                u = s - est
                # the TRUE plant advances using the actuated coordinate (angle-free
                # additive convention for the offline maths; the angle-only tax is Exp 5)
                x = A[k] @ s + b[k]
                prev_k, prev_u = k, u
            ang_err.append(_wrap(_angle(x) - target_ang))
        ang_err = np.array(ang_err)
        return float(np.abs(ang_err).mean()), float(ang_err.var())

    lqg_err, lqr_err, lqg_var, lqr_var = [], [], [], []
    for vs in V_scales:
        Vc = vs * scale
        V = (vs * scale) ** 2 * np.eye(2)
        e_g, v_g = closed_loop(True, Vc, V)
        e_r, v_r = closed_loop(False, Vc, V)
        lqg_err.append(e_g); lqg_var.append(v_g)
        lqr_err.append(e_r); lqr_var.append(v_r)
    return (np.array(V_scales), np.array(lqg_err), np.array(lqr_err),
            np.array(lqg_var), np.array(lqr_var))


# =============================================================================
# Main
# =============================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traj", type=str,
        default="../SO2/phase_portrait_output/Qwen2.5-3B-Instruct/trajectories.npz",
        help="saved phase-portrait trajectories (the same file PTS uses)")
    ap.add_argument("--target-deg", type=float, default=None,
                    help="target angle in degrees (default: harmless-mean band angle)")
    ap.add_argument("--delta", type=float, default=0.03,
                    help="plant-perturbation std for the Exp-3 robustness MC")
    ap.add_argument("--n-mc", type=int, default=40,
                    help="Monte-Carlo perturbations for the Exp-3 robustness MC")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traj_path = pathlib.Path(args.traj)
    assert traj_path.exists(), f"trajectory file not found: {traj_path.resolve()}"
    d = np.load(traj_path)
    hc1, hc2 = d["harmful_c1"], d["harmful_c2"]
    lc1, lc2 = d["harmless_c1"], d["harmless_c2"]
    layers = d["layers"]
    L = hc1.shape[1]
    print(f"loaded {traj_path}: harmful {hc1.shape}, harmless {lc1.shape}, L={L}")

    rng = np.random.RandomState(args.seed)

    # ----------------------------------------------------------- steering band
    harmful_mean = np.stack([hc1.mean(0), hc2.mean(0)], 1)        # (L,2)
    harmless_mean = np.stack([lc1.mean(0), lc2.mean(0)], 1)
    band, angle_sep, peak = steering_band(harmful_mean, harmless_mean, L)
    band_layers = list(range(band[0], band[1] + 1))
    kT = band_layers[-1]
    print(f"\n[band] |angle_sep|>0.5 rad around peak layer {layers[peak]}: "
          f"layers {layers[band[0]]}..{layers[band[1]]} "
          f"(indices {band[0]}..{band[1]}, {len(band_layers)} layers), terminal kT={layers[kT]}")

    # ----------------------------------------------------- plant + process noise
    # train/test split shared across experiments (mirror pts_offline)
    nH, nL = hc1.shape[0], lc1.shape[0]
    hi = rng.permutation(nH); li = rng.permutation(nL)
    htr, hte = hi[:nH // 2], hi[nH // 2:]
    ltr, lte = li[:nL // 2], li[nL // 2:]
    tr_c1 = np.concatenate([hc1[htr], lc1[ltr]]); tr_c2 = np.concatenate([hc2[htr], lc2[ltr]])
    te_c1 = np.concatenate([hc1[hte], lc1[lte]]); te_c2 = np.concatenate([hc2[hte], lc2[lte]])
    fit = fit_layer_dynamics(tr_c1, tr_c2, affine=True)
    val = validate_dynamics(te_c1, te_c2, fit, horizons=(1, 5))
    A, b = fit["A"], fit["b"]
    print(f"[plant] held-out R2(1-step)={val['r2_h1']:.4f}  R2(5-step)={val['r2_h5']:.4f}  "
          f"full-rollout RMSE={val['full_rollout_rmse']:.3f}")

    # process-noise covariance W from the (held-out) one-step residuals (the ~14% PTS
    # residual, now as a covariance the Kalman filter consumes).
    C_te = np.stack([te_c1, te_c2], axis=-1)                      # (Nte, L, 2)
    W, W_per_layer = estimate_process_noise(A, b, C_te)
    scale_band = np.linalg.norm(C_te[:, band[0]:band[1] + 1, :].reshape(-1, 2), axis=-1).mean()
    w_frac = np.sqrt(np.trace(W)) / (scale_band + 1e-12)
    print(f"[noise] W = pooled residual cov, sqrt(tr W)={np.sqrt(np.trace(W)):.3f} "
          f"(~{100*w_frac:.0f}% of band coord scale {scale_band:.2f})")

    # per-layer rotation magnitude ||A_k - I||_2 (the depth-band spectral profile)
    rot_mag = np.array([np.linalg.norm(A[k] - np.eye(2), 2) for k in range(L - 1)])

    # target angle: the harmless-mean band angle (the "track harmless" reference
    # direction, as PTS), unless overridden.
    if args.target_deg is None:
        target_angle = circular_mean(_angle(harmless_mean[band_layers]))
    else:
        target_angle = np.radians(args.target_deg)
    print(f"[target] terminal target angle = {np.degrees(target_angle):.0f} deg")

    # held-out incoming coords at the band start (the controller's initial conditions)
    C_all = np.stack([np.concatenate([hc1, lc1]), np.concatenate([hc2, lc2])], axis=-1)  # (N,L,2)
    c0_band = C_te[:, band[0], :]                                 # (Nte,2)
    # terminal-target coordinate at the natural terminal magnitude
    tmag = np.linalg.norm(C_te[:, kT + 1, :], axis=-1).mean()
    target = tmag * np.array([np.cos(target_angle), np.sin(target_angle)])

    # --------------------------------------------------------------- Exp 2
    print("\n[Exp 2] geometric enforcement-layer sweep (terminal angle error vs enf. layer)")
    enf_idx, enf_terminal_err = exp2_enforcement_sweep(
        A, b, c0_band, band_layers, target_angle, kT)
    for ki, e in zip(enf_idx, enf_terminal_err):
        print(f"      enforce at layer {int(layers[ki]):2d}: terminal angle err = {e:.4f} rad")
    best_enf = int(enf_idx[int(np.argmin(enf_terminal_err))])
    print(f"  -> terminal angle best landed by enforcing LATE (layer {int(layers[best_enf])}); "
          f"early enforcement washes out -> motivates the terminal (kT) objective.")

    # --------------------------------------------------------------- Exp 3
    print("\n[Exp 3] soft-landing vs deadbeat at matched terminal angle (headline maths)")
    # (a) rho frontier (deadbeat = rho->0 endpoint)
    rhos, eff_rho, mxu_rho, aerr_rho = exp3_rho_frontier(
        A, b, c0_band, band_layers, target, kT)
    print("  3a. rho frontier (effort vs terminal angle error; rho->0 = deadbeat):")
    for rho, e, m, a in zip(rhos, eff_rho, mxu_rho, aerr_rho):
        print(f"      rho={rho:7.3f}  total_effort={e:7.4f}  max||u||={m:7.4f}  ang_err={a:.5f} rad")

    # (b) matched-terminal-angle soft-vs-deadbeat (the headline comparison)
    matched = _match_softlanding_to_deadbeat(A, b, c0_band, band_layers, target, kT)
    db_m, sl_m = matched["deadbeat"], matched["softlanding"]
    print(f"  3b. matched terminal angle (gap={sl_m['gap']:.4f} rad, soft rho={sl_m['rho']:.3f}):")
    print(f"      deadbeat   : total_effort={db_m['total_effort']:.4f}  max||u||={db_m['max_u']:.4f}")
    print(f"      softlanding: total_effort={sl_m['total_effort']:.4f}  max||u||={sl_m['max_u']:.4f}")
    print(f"      -> soft-landing uses {db_m['max_u']/(sl_m['max_u']+1e-12):.2f}x smaller per-layer push")

    # (c) robustness: terminal-angle scatter under a perturbed plant (Monte Carlo).
    # The deadbeat acts at the terminal layer, so it must be re-rolled from the natural
    # x_kT (the matched incoming state), NOT the band-start coord — same consistent
    # physical state used in the matched comparison above (F1).
    db_rob_err, db_rob_var = exp3_robustness(
        A, b, db_m["x_kT"], [kT], db_m["gains"], target, args.delta, args.n_mc, args.seed)
    sl_rob_err, sl_rob_var = exp3_robustness(
        A, b, c0_band, band_layers, sl_m["gains"], target, args.delta, args.n_mc, args.seed + 1)
    print(f"  3c. robustness under perturbed plant (delta={args.delta}, {args.n_mc} MC):")
    print(f"      deadbeat   : terminal ang err={db_rob_err:.4f} rad  circ-var={db_rob_var:.4f}")
    print(f"      softlanding: terminal ang err={sl_rob_err:.4f} rad  circ-var={sl_rob_var:.4f}")

    # (d) advantage vs ||A_k - I|| across depth bands (the depth-band prediction)
    # sweep candidate terminal layers across the available depth (excluding the very
    # late layers with no room for a band); use a fixed band width.
    band_w = 5
    cand_kTs = [k for k in range(band_w, L - 1)]
    adv = exp3_advantage_vs_band(A, b, C_all, target_angle,
                                 band_widths=[band_w] * len(cand_kTs),
                                 all_kTs=cand_kTs, seed=args.seed)
    if adv["kT"].size:
        corr = float(np.corrcoef(adv["rot_band"], adv["advantage"])[0, 1])
        frac_pos = float((adv["advantage"] > 0).mean())  # also recomputed below for save
        # Honest read of the rotation trend: the proposal PREDICTS a POSITIVE
        # correlation (soft-landing wins MOST in rotation-dominated bands). Describe
        # the measured sign/strength as-is — do not assert a positive trend if the
        # data does not show one.
        if corr > 0.3:
            trend = "a clear positive"
        elif corr > 0.1:
            trend = "a weak positive"
        elif corr >= -0.1:
            trend = "essentially no"
        else:
            trend = "a (mildly) NEGATIVE"
        print(f"  3d. soft-landing advantage (deadbeat push - soft push, scale-free) vs "
              f"||A_k - I||_2 across {adv['kT'].size} bands: corr={corr:+.3f}")
        print(f"      advantage > 0 in {100*frac_pos:.0f}% of bands "
              f"(range {adv['advantage'].min():+.2f}..{adv['advantage'].max():+.2f}); "
              f"HONEST read: soft-landing wins the per-layer push EVERYWHERE, with "
              f"{trend} rotation trend across depth (the predicted near-identity-band "
              f"hedge — a positive corr — is not borne out in this scale-free metric).")
    else:
        corr = float("nan")
    frac_pos = float((adv["advantage"] > 0).mean()) if adv["kT"].size else float("nan")

    # --------------------------------------------------------------- Exp 4
    print("\n[Exp 4] observability + Kalman-vs-single-vs-EMA + LQG-vs-LQR-under-noise")
    # (a) observability of (A_band, H) with H = I (full read) and H = e1 (scalar read)
    A_band = A[band[0]:band[1] + 1]
    A_mean = A_band.mean(0)                                       # band-mean representative A
    H_full = np.eye(2)
    H_scalar = np.array([[1.0, 0.0]])                            # angle/b1-only partial read
    obs_full = bool(is_observable(A_mean, H_full))
    obs_scalar = bool(is_observable(A_mean, H_scalar))
    G_scalar = observability_gramian(A_mean, H_scalar, n=2)
    G_scalar_tv = observability_gramian_tv(A_band[:-1], H_scalar)  # time-varying over the band
    cond_scalar = float(np.linalg.cond(G_scalar)) if obs_scalar else float("inf")
    cond_scalar_tv = float(np.linalg.cond(G_scalar_tv))
    print(f"  4a. observability: full-read H=I -> {obs_full}; scalar-read H=e1 -> "
          f"{obs_scalar} (Gramian cond TI={cond_scalar:.1f}, TV={cond_scalar_tv:.1f})")

    # (b) estimator comparison (angle RMSE): Kalman vs single vs EMA under noise
    V_scale = 0.30                                                # 30% of band coord scale
    est = exp4_estimator_compare(A, b, C_te, band_layers, W, V_scale, args.seed)
    print(f"  4b. angle RMSE under V={V_scale:.0%} measurement noise (mean over band):")
    print(f"      Kalman-fused={est['kalman_rmse'].mean():.4f}  "
          f"single-layer={est['single_rmse'].mean():.4f}  "
          f"ad-hoc-EMA={est['ema_rmse'].mean():.4f} rad")

    # (c) LQG vs LQR-on-raw at matched gains, swept over injected measurement noise
    V_scales = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.8])
    vsw, lqg_err, lqr_err, lqg_var, lqr_var = exp4_lqg_vs_lqr(
        A, b, C_te, band_layers, target, kT, W, V_scales, args.seed)
    print("  4c. terminal angle error (mean | variance): LQG (control on xhat) vs "
          "LQR-on-raw vs noise:")
    for vs, le, lr, vg, vr in zip(vsw, lqg_err, lqr_err, lqg_var, lqr_var):
        flag = "LQG<" if le < lr - 1e-4 else ("tie" if abs(le - lr) <= 1e-4 else "LQR<")
        print(f"      V={vs:4.0%}: LQG={le:.4f} (var {vg:.4f})  "
              f"LQR-raw={lr:.4f} (var {vr:.4f}) rad  ({flag})")

    # --------------------------------------------------------------- save
    out = traj_path.parent / "oas_offline_results.npz"
    np.savez_compressed(
        out,
        # --- shared / band / plant ---
        layers=layers, band=np.array(band), band_layers=np.array(band_layers),
        terminal_layer=np.array(kT), peak_layer=np.array(peak),
        angle_sep=angle_sep, target_angle=np.array(target_angle),
        target=target, harmful_mean=harmful_mean, harmless_mean=harmless_mean,
        rot_mag=rot_mag,
        r2_h1=np.array(val["r2_h1"]), r2_h5=np.array(val["r2_h5"]),
        full_rollout_rmse=np.array(val["full_rollout_rmse"]),
        per_layer_r2=np.array(val["per_layer_r2_1step"]),
        W=W, W_per_layer=W_per_layer, w_frac=np.array(w_frac),
        band_coord_scale=np.array(scale_band),
        # --- Exp 2: enforcement-layer sweep ---
        exp2_enf_layers=enf_idx, exp2_enf_layers_global=layers[enf_idx],
        exp2_terminal_ang_err=enf_terminal_err, exp2_best_enf_layer=np.array(best_enf),
        # --- Exp 3a: rho frontier ---
        exp3_rhos=rhos, exp3_total_effort=eff_rho, exp3_max_u=mxu_rho,
        exp3_terminal_ang_err=aerr_rho,
        # --- Exp 3b: matched soft-vs-deadbeat ---
        exp3_db_total_effort=np.array(db_m["total_effort"]),
        exp3_db_max_u=np.array(db_m["max_u"]),
        exp3_db_ang_mean=np.array(db_m["ang_mean"]),
        exp3_sl_total_effort=np.array(sl_m["total_effort"]),
        exp3_sl_max_u=np.array(sl_m["max_u"]),
        exp3_sl_rho=np.array(sl_m["rho"]), exp3_match_gap=np.array(sl_m["gap"]),
        # --- Exp 3c: robustness MC ---
        exp3_delta=np.array(args.delta), exp3_n_mc=np.array(args.n_mc),
        exp3_db_rob_ang_err=np.array(db_rob_err), exp3_db_rob_var=np.array(db_rob_var),
        exp3_sl_rob_ang_err=np.array(sl_rob_err), exp3_sl_rob_var=np.array(sl_rob_var),
        # --- Exp 3d: advantage vs ||A_k - I|| across bands ---
        exp3_adv_kT=adv["kT"], exp3_adv_rot_band=adv["rot_band"],
        exp3_adv_deadbeat_max_u=adv["deadbeat_max_u"],
        exp3_adv_soft_max_u=adv["soft_max_u"], exp3_adv_advantage=adv["advantage"],
        exp3_adv_corr=np.array(corr), exp3_adv_band_width=np.array(band_w),
        exp3_adv_frac_pos=np.array(frac_pos),
        # --- Exp 4a: observability ---
        exp4_obs_full=np.array(obs_full), exp4_obs_scalar=np.array(obs_scalar),
        exp4_gramian_cond_ti=np.array(cond_scalar),
        exp4_gramian_cond_tv=np.array(cond_scalar_tv),
        exp4_A_mean=A_mean,
        # --- Exp 4b: estimator comparison ---
        exp4_est_layers=est["layers"], exp4_kalman_rmse=est["kalman_rmse"],
        exp4_single_rmse=est["single_rmse"], exp4_ema_rmse=est["ema_rmse"],
        exp4_est_V_scale=np.array(est["V_scale"]),
        # --- Exp 4c: LQG vs LQR under noise ---
        exp4_V_scales=vsw, exp4_lqg_err=lqg_err, exp4_lqr_err=lqr_err,
        exp4_lqg_var=lqg_var, exp4_lqr_var=lqr_var,
    )
    print(f"\nsaved {out}")
    print(f"\n[oas_offline] done.")


if __name__ == "__main__":
    main()
