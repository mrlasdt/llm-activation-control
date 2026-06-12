"""OAS — finite-horizon LQ tracking with a TERMINAL cost (Idea 1, soft landing).

Implements Section 1 of OAS_SPEC.md: the depth-domain finite-horizon LQR that
emits a smooth, bounded sequence of small rotations across a band of layers and
lands the in-plane coordinate at a target angle only by a chosen terminal layer
`kT` — the principled generalisation of the single-layer deadbeat reset that the
existing Angular Steering (and the Exp-3 baseline) use.

Actuator convention (MUST match PTS exactly, OAS_SPEC S0):

    state    x_k = the *natural incoming* coordinate at layer k (what the plant
             delivers, before actuation);
    decision s_k = the *actuated* coordinate that flows downstream, s_k = x_k + u_k
             (control u_k = s_k - x_k);
    dynamics x_{k+1} = A_k s_k + b_k = A_k (x_k + u_k) + b_k.

The physical actuator is the norm-preserving SO(2) reset (it realises only the
*angle* of s_k and keeps ||x_k||); that magnitude tax is measured elsewhere
(Exp 5). This module is the additive-control planner — pure numpy on the 2D plane
coordinates, no model.

Cost (OAS_SPEC S1.1), over the actuated band k0..kT:

    J = sum_{k=k0}^{kT} [ (s_k - r_k)' Q_k (s_k - r_k) + (s_k - x_k)' R (s_k - x_k) ]

with the soft-landing config Q_k = Q_stage (small/zero) for k < kT and
Q_{kT} = Q_term (large), R = rho I (rho > 0). The deadbeat corner is a single
layer kT with Q_term huge and rho -> 0.

The backward recursion (OAS_SPEC S1.2 — VERIFIED, implemented verbatim) builds the
value function V_k(x) = x' S_k x - 2 v_k' x + const with sentinel S_{kT+1} = 0,
v_{kT+1} = 0, and for k = kT, kT-1, ..., k0:

    M_k = Q_k + R + A_k' S_{k+1} A_k                 # 2x2 SPD
    e_k = Q_k r_k + A_k' (v_{k+1} - S_{k+1} b_k)      # 2-vector
    F_k = M_k^{-1} R                                  # feedback gain on x_k
    g_k = M_k^{-1} e_k                                # feedforward
    s_k*(x) = F_k x + g_k        ( u_k = (F_k - I) x + g_k )
    S_k = R - R M_k^{-1} R                            # symmetric PSD
    v_k = R g_k

The authoritative correctness gate is the condensed-QP gold cross-check in the
self-test: stack the whole horizon decision U = [s_k0; ...; s_kT], write
J(U) = 1/2 U'PU + q'U + c by eliminating the states via x_{k+1} = A_k s_k + b_k,
solve U* = -P^{-1} q, roll the Riccati policy forward through the SAME dynamics,
and assert the two s-trajectories and costs agree to < 1e-8 (the analogue of
pts_mpc test [1]/[7]).
"""

from __future__ import annotations

import sys
import pathlib

# sys.path shim (OAS_SPEC S0): expose BOTH the shared lib and the sibling PTS dir
# so `from pts_dynamics import ...`, `from utils import ...`, `from observables
# import ...` all resolve. OAS is self-contained: it imports only the shared lib
# and pts_dynamics from the PTS dir (the pure-numpy plant fit) — nothing else.
_R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "pytorch_pure"))
sys.path.insert(0, str(_R / "research_proposals" / "PTS"))

import numpy as np


# =============================================================================
# Backward Riccati recursion  (OAS_SPEC S1.2 — implemented verbatim)
# =============================================================================


def _as_band(band) -> list[int]:
    """Validate the actuated band: ascending, contiguous list of layer indices."""
    band = [int(k) for k in band]
    assert len(band) >= 1, "band must contain at least one layer"
    assert all(band[i + 1] == band[i] + 1 for i in range(len(band) - 1)), \
        "band must be ascending and contiguous"
    return band


def finite_horizon_lqr(A, b, band, r, R, Q_stage, Q_term, *, terminal_layer=None):
    """Solve the finite-horizon LQ-tracking problem with a terminal cost.

    Args:
        A: (L-1, 2, 2) fitted plant matrices (full grid, indexed by layer).
        b: (L-1, 2) fitted affine drifts (full grid).
        band: list of actuated layer indices (ascending, contiguous).
        r: (L, 2) per-layer reference; r[terminal_layer] is the terminal target.
        R: (2, 2) control-effort weight (rho I); SPD.
        Q_stage: (2, 2) small/zero intermediate state weight (k < kT).
        Q_term: (2, 2) large terminal state weight (k == kT).
        terminal_layer: kT; defaults to band[-1].

    Returns dict (OAS_SPEC S1.3):
        {"F": {k:(2,2)}, "g": {k:(2,)}, "S": {k:(2,2)}, "v": {k:(2,)},
         "band": band, "terminal_layer": kT}

    The per-layer policy is s_k*(x) = F_k x + g_k (control u_k = s_k - x_k).
    """
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    r = np.asarray(r, np.float64)
    R = np.asarray(R, np.float64)
    Q_stage = np.asarray(Q_stage, np.float64)
    Q_term = np.asarray(Q_term, np.float64)
    band = _as_band(band)
    kT = band[-1] if terminal_layer is None else int(terminal_layer)
    # The terminal layer is the LAST actuated layer (OAS_SPEC S1.1: "land the angle
    # at the target only by a chosen LATE layer kT"). The backward recursion applies
    # Q_term only at k==kT and Q_stage elsewhere; permitting kT < band[-1] would leave
    # the layers after kT actuated with the small Q_stage and a stale sentinel-derived
    # gain — NOT what "terminal layer = last actuated layer" means. Every caller passes
    # kT = band[-1]; we assert it rather than silently mis-actuate the tail. (F5)
    assert kT == band[-1], \
        "terminal_layer must be the LAST actuated band layer (kT == band[-1])"

    F: dict[int, np.ndarray] = {}
    g: dict[int, np.ndarray] = {}
    S: dict[int, np.ndarray] = {}
    v: dict[int, np.ndarray] = {}

    # Sentinel beyond the band: S_{kT+1} = 0, v_{kT+1} = 0.
    S_next = np.zeros((2, 2))
    v_next = np.zeros(2)

    # Backward pass k = kT, kT-1, ..., k0.
    for k in reversed(band):
        Q_k = Q_term if k == kT else Q_stage
        Ak = A[k]
        bk = b[k]
        M_k = Q_k + R + Ak.T @ S_next @ Ak                      # 2x2 SPD
        e_k = Q_k @ r[k] + Ak.T @ (v_next - S_next @ bk)        # 2-vector
        Minv = np.linalg.inv(M_k)
        F_k = Minv @ R                                          # feedback on x_k
        g_k = Minv @ e_k                                        # feedforward
        S_k = R - R @ Minv @ R                                  # symmetric PSD
        S_k = 0.5 * (S_k + S_k.T)                               # symmetrise
        v_k = R @ g_k
        F[k], g[k], S[k], v[k] = F_k, g_k, S_k, v_k
        S_next, v_next = S_k, v_k

    return {"F": F, "g": g, "S": S, "v": v, "band": band, "terminal_layer": kT}


def deadbeat_gains(A, b, kT, target, *, R_eps=1e-12, Q_term_scale=1.0):
    """The R -> 0 deadbeat corner: a single actuated layer kT that lands the *next*
    coordinate x_{kT+1} exactly on `target` (OAS_SPEC S1.2 check 2).

    Returns the same gains dict as finite_horizon_lqr for the degenerate
    single-layer band [kT] with R = R_eps I (-> 0). To land x_{kT+1} = A_kT s* + b_kT
    on `target`, the *actuated* coordinate must be slammed to the pre-image
    s* = A_kT^{-1}(target - b_kT). The terminal-layer policy with S_{kT+1}=0 gives
    s* = (Q_term + R)^{-1}(R x + Q_term r), so as R -> 0 we get s* -> r; we therefore
    set the LQ reference at kT to that pre-image r_{kT} = A_kT^{-1}(target - b_kT).
    In the limit this slams the actuated angle to the pre-image and the autonomous
    A_kT step then carries it exactly onto `target` at the next coordinate.

    Note (OAS_SPEC S1.3): the *physical* deadbeat baseline in the drivers is simply
    the fixed-target-angle reset at one layer, so a driver can use the target angle
    directly; this convenience is the LQ-consistent algebraic equivalent that makes
    the next coordinate land on target.
    """
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    target = np.asarray(target, np.float64)
    L = A.shape[0] + 1
    pre_image = np.linalg.solve(A[kT], target - b[kT])   # A_kT^{-1}(target - b_kT)
    r = np.zeros((L, 2))
    r[kT] = pre_image
    R = R_eps * np.eye(2)
    Q_term = Q_term_scale * np.eye(2)
    Q_stage = np.zeros((2, 2))
    return finite_horizon_lqr(A, b, [kT], r, R, Q_stage, Q_term, terminal_layer=kT)


# =============================================================================
# Rollout / effort  (OAS_SPEC S1.3)
# =============================================================================


def lqr_rollout(A, b, gains, x0, band):
    """Roll the Riccati policy s_k = F_k x_k + g_k forward through the dynamics.

    Starting from the natural incoming coordinate x_{k0} = x0, at each band layer:

        s_k     = F_k x_k + g_k         (actuated coordinate / decision)
        u_k     = s_k - x_k             (additive control)
        x_{k+1} = A_k s_k + b_k         (natural incoming coordinate at next layer)

    Args:
        A, b: (L-1,2,2)/(L-1,2) fitted plant (full grid).
        gains: dict from finite_horizon_lqr ({"F","g","band",...}).
        x0: (2,) natural incoming coordinate at band[0].
        band: list of actuated layer indices (ascending, contiguous).

    Returns dict:
        {"x": (n,2) natural states x_{k0..kT}, "s": (n,2) actuated decisions,
         "u": (n,2) controls, "x_term": (2,) x_{kT+1} (the landed coordinate),
         "band": band}   where n = len(band).
    """
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    band = _as_band(band)
    F, g = gains["F"], gains["g"]

    x = np.asarray(x0, np.float64).copy()
    xs, ss, us = [], [], []
    for k in band:
        s = F[k] @ x + g[k]
        u = s - x
        xs.append(x.copy())
        ss.append(s.copy())
        us.append(u.copy())
        x = A[k] @ s + b[k]          # natural incoming coordinate at layer k+1
    return {"x": np.array(xs), "s": np.array(ss), "u": np.array(us),
            "x_term": x.copy(), "band": band}


def total_effort(u_seq) -> float:
    """Sum_k ||u_k|| — the total off-manifold push over the band (OAS_SPEC S1.3)."""
    u_seq = np.asarray(u_seq, np.float64)
    return float(np.linalg.norm(u_seq, axis=-1).sum())


# =============================================================================
# Condensed-QP gold cross-check  (OAS_SPEC S1.2 / S1.4) — used by the self-test
# =============================================================================


def _build_condensed_qp(A, b, band, r, R, Q_stage, Q_term, kT, x0):
    """Stack the whole horizon decision U = [s_k0; ...; s_kT] and write the cost
    J(U) = 1/2 U'PU + q'U + c by eliminating the states via x_{k+1}=A_k s_k+b_k.

    States are eliminated as affine functions of the decisions: starting from the
    fixed incoming x_{k0} = x0,

        x_{k0}    = x0                                            (constant)
        x_{k+1}   = A_k s_k + b_k        for k in band.

    So x_k = sum_{j<k in band} (prod of A's) A_j s_j + (drift) — built iteratively
    as x_k = Xx[k] @ U + xc[k]. The cost couples each s_k (a 2-block of U, selector
    E_k) with x_k through the effort term (s_k - x_k)' R (s_k - x_k) and with the
    reference through (s_k - r_k)' Q_k (s_k - r_k).

    Returns (P, q, c) with P SPD; the unconstrained minimiser is U* = -P^{-1} q.
    This mirrors pts_mpc.build_condensed_qp (the gold cross-check pattern).
    """
    n = 2
    m = len(band)
    nU = n * m
    idx = {k: i for i, k in enumerate(band)}     # band layer -> decision block

    def E(k):
        """Selector S_k: 2 x nU picking the s_k block out of U."""
        Ek = np.zeros((n, nU))
        Ek[:, n * idx[k]:n * idx[k] + n] = np.eye(n)
        return Ek

    # State maps x_k = Xx[k] @ U + xc[k], built forward over the band.
    Xx = {band[0]: np.zeros((n, nU))}
    xc = {band[0]: np.asarray(x0, np.float64).copy()}
    for k in band[:-1]:
        Xx[k + 1] = A[k] @ E(k) + np.zeros((n, nU))    # x_{k+1} = A_k s_k + b_k
        # note: A_k s_k = A_k (E_k U); plus the affine b_k. (Earlier x's drop out
        # because the decision IS s_k, not a control relative to x_k.)
        xc[k + 1] = b[k].copy()

    P = np.zeros((nU, nU))
    q = np.zeros(nU)
    c = 0.0
    for k in band:
        Q_k = Q_term if k == kT else Q_stage
        Ek = E(k)
        # State (reference) term: (s_k - r_k)' Q_k (s_k - r_k), s_k = E_k U.
        P += 2.0 * (Ek.T @ Q_k @ Ek)
        q += 2.0 * (Ek.T @ (-Q_k @ r[k]))
        c += r[k] @ Q_k @ r[k]
        # Effort term: (s_k - x_k)' R (s_k - x_k), with x_k = Xx[k] U + xc[k].
        D = Ek - Xx[k]                              # d_k = D U - xc[k] = s_k - x_k
        d0 = -xc[k]
        P += 2.0 * (D.T @ R @ D)
        q += 2.0 * (D.T @ R @ d0)
        c += d0 @ R @ d0
    P = 0.5 * (P + P.T)
    return P, q, float(c)


def _qp_cost(P, q, c, U):
    return float(0.5 * U @ P @ U + q @ U + c)


def _rollout_cost(A, b, gains, x0, band, r, R, Q_stage, Q_term, kT):
    """Direct (forward-simulation) cost of the rolled-out Riccati policy."""
    roll = lqr_rollout(A, b, gains, x0, band)
    s, x = roll["s"], roll["x"]
    J = 0.0
    for i, k in enumerate(band):
        Q_k = Q_term if k == kT else Q_stage
        es = s[i] - r[k]
        eu = s[i] - x[i]
        J += es @ Q_k @ es + eu @ R @ eu
    return float(J), roll


# =============================================================================
# Self-test  (OAS_SPEC S1.4)
# =============================================================================


if __name__ == "__main__":
    rng = np.random.RandomState(7)
    n = 2

    # Synthetic stable affine plant (rotation-dominated, like the early/mid band).
    L = 14
    A = np.stack([0.92 * np.array([[np.cos(0.3), -np.sin(0.3)],
                                   [np.sin(0.3), np.cos(0.3)]])
                  + 0.04 * rng.randn(2, 2) for _ in range(L - 1)])
    b = 0.1 * rng.randn(L - 1, 2)

    band = list(range(3, 10))                       # k0..kT = 3..9
    kT = band[-1]
    x0 = rng.randn(2)

    # Reference: terminal target at the target angle with a natural magnitude;
    # intermediate references arbitrary (Q_stage will make them nearly irrelevant).
    target = np.array([np.cos(2.0), np.sin(2.0)])
    r = rng.randn(L, 2) * 0.3
    r[kT] = target

    R = 0.05 * np.eye(2)
    Q_stage = 0.01 * np.eye(2)
    Q_term = 50.0 * np.eye(2)

    # ---- (a) condensed-QP gold cross-check (the AUTHORITATIVE gate) ----------
    gains = finite_horizon_lqr(A, b, band, r, R, Q_stage, Q_term)
    P, q, c = _build_condensed_qp(A, b, band, r, R, Q_stage, Q_term, kT, x0)
    U_star = np.linalg.solve(P, -q)                 # U* = -P^{-1} q (convex)
    J_star, roll = _rollout_cost(A, b, gains, x0, band, r, R, Q_stage, Q_term, kT)
    U_roll = roll["s"].reshape(-1)                  # stacked s_k0..s_kT
    s_err = float(np.abs(U_star - U_roll).max())
    J_qp = _qp_cost(P, q, c, U_star)
    cost_err = abs(J_qp - J_star)
    # P SPD (convex) and the QP minimiser optimal (gradient ~ 0).
    grad = P @ U_star + q
    print(f"[a] condensed-QP gold: max |s_Riccati - s_QP| = {s_err:.2e}, "
          f"|J_Riccati - J_QP| = {cost_err:.2e}, ||grad|| = {np.linalg.norm(grad):.2e}")
    assert np.all(np.linalg.eigvalsh(P) > 0), "condensed P must be SPD"
    assert s_err < 1e-8 and cost_err < 1e-8

    # ---- terminal-layer sanity (OAS_SPEC S1.2 check 1) -----------------------
    # Single terminal layer, S_{kT+1}=0: s* = (Q_term+R)^{-1}(R x + Q_term r).
    Qt = 3.0 * np.eye(2)
    Rt = 0.4 * np.eye(2)
    g1 = finite_horizon_lqr(A, b, [kT], r, Rt, Q_stage, Qt, terminal_layer=kT)
    xx = rng.randn(2)
    s_pol = g1["F"][kT] @ xx + g1["g"][kT]
    s_ref = np.linalg.solve(Qt + Rt, Rt @ xx + Qt @ r[kT])
    print(f"[b] terminal-layer weighted-average: max err = "
          f"{np.abs(s_pol - s_ref).max():.2e}")
    assert np.abs(s_pol - s_ref).max() < 1e-10
    # R -> 0 => s* -> r (deadbeat/slam at this layer); R -> inf => s* -> x (no move).
    g_slam = finite_horizon_lqr(A, b, [kT], r, 1e-9 * np.eye(2), Q_stage, Qt,
                                terminal_layer=kT)
    s_slam = g_slam["F"][kT] @ xx + g_slam["g"][kT]
    g_noop = finite_horizon_lqr(A, b, [kT], r, 1e9 * np.eye(2), Q_stage, Qt,
                                terminal_layer=kT)
    s_noop = g_noop["F"][kT] @ xx + g_noop["g"][kT]
    assert np.linalg.norm(s_slam - r[kT]) < 1e-4 and np.linalg.norm(s_noop - xx) < 1e-4

    # ---- (b) deadbeat limit lands the terminal coordinate on target ----------
    db = deadbeat_gains(A, b, kT, target)
    roll_db = lqr_rollout(A, b, db, x0=rng.randn(2), band=[kT])
    land_err = float(np.linalg.norm(roll_db["x_term"] - target))
    print(f"[c] deadbeat lands x_{{kT+1}} on target: err = {land_err:.2e}")
    assert land_err < 1e-6

    # ---- (c) monotone effort-vs-error frontier as rho sweeps -----------------
    # Sweep R = rho I up: total_effort must fall and terminal angle error rise,
    # monotonically. This is the pure soft-landing config (Q_stage = 0, "don't
    # force the angle early") so the ONLY tension is terminal-target vs effort —
    # exactly the deadbeat (rho->0) <-> distributed (rho up) spectrum. The terminal
    # reference is the pre-image of `target`, so x_{kT+1} = A_kT s_kT + b_kT lands
    # on `target` exactly at the deadbeat corner and drifts off as effort is taxed.
    Q_stage0 = np.zeros((2, 2))
    r_pre = r.copy()
    r_pre[kT] = np.linalg.solve(A[kT], target - b[kT])
    rhos = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0])
    efforts, ang_errs = [], []
    target_ang = np.arctan2(target[1], target[0])
    for rho in rhos:
        gk = finite_horizon_lqr(A, b, band, r_pre, rho * np.eye(2), Q_stage0, Q_term)
        rk = lqr_rollout(A, b, gk, x0, band)
        efforts.append(total_effort(rk["u"]))
        # terminal landed coordinate x_{kT+1}: its angle vs the target angle.
        ang = np.arctan2(rk["x_term"][1], rk["x_term"][0])
        ang_errs.append(abs(np.arctan2(np.sin(ang - target_ang),
                                       np.cos(ang - target_ang))))
    efforts = np.array(efforts)
    ang_errs = np.array(ang_errs)
    print("[d] effort-vs-error frontier (rho up -> effort down, ang-err up):")
    for rho, e, a in zip(rhos, efforts, ang_errs):
        print(f"      rho={rho:7.3f}  total_effort={e:8.4f}  ang_err={a:8.5f} rad")
    # Monotone non-increasing effort, monotone non-decreasing angle error.
    assert np.all(np.diff(efforts) <= 1e-9), "effort must fall as rho rises"
    assert np.all(np.diff(ang_errs) >= -1e-9), "angle error must rise as rho rises"

    # ---- (d) soft-landing reaches the deadbeat terminal angle at lower max||u||
    # Single-layer deadbeat at kT vs a soft-landing band over k0..kT, BOTH landing
    # x_{kT+1} at (essentially) the same terminal angle. Soft landing must use a
    # strictly smaller per-layer push max_k ||u_k||.
    x0c = rng.randn(2)
    # deadbeat: single layer kT slams the next coordinate exactly onto target.
    db_gains = deadbeat_gains(A, b, kT, target)
    db_roll = lqr_rollout(A, b, db_gains, x0c, [kT])
    db_ang = np.arctan2(db_roll["x_term"][1], db_roll["x_term"][0])
    db_maxu = float(np.linalg.norm(db_roll["u"], axis=-1).max())
    # soft landing over the full band, same terminal landing convention (pre-image
    # reference, Q_stage = 0 so the angle is only required by the terminal layer)
    # and a tight Q_term so it lands on the SAME terminal angle as the deadbeat.
    r_pre_e = np.zeros((L, 2))
    r_pre_e[kT] = np.linalg.solve(A[kT], target - b[kT])
    sl_gains = finite_horizon_lqr(A, b, band, r_pre_e, 0.02 * np.eye(2),
                                  np.zeros((2, 2)), 500.0 * np.eye(2))
    sl_roll = lqr_rollout(A, b, sl_gains, x0c, band)
    sl_ang = np.arctan2(sl_roll["x_term"][1], sl_roll["x_term"][0])
    sl_maxu = float(np.linalg.norm(sl_roll["u"], axis=-1).max())
    ang_gap = abs(np.arctan2(np.sin(sl_ang - db_ang), np.cos(sl_ang - db_ang)))
    print(f"[e] soft-landing vs deadbeat at matched terminal angle "
          f"(gap={ang_gap:.4f} rad): max||u|| soft={sl_maxu:.4f} < "
          f"deadbeat={db_maxu:.4f}")
    assert ang_gap < 0.05, "soft landing must reach the same terminal angle"
    assert sl_maxu < db_maxu, "soft landing must use a smaller per-layer push"

    print("[oas_lqr self-test] OK")
