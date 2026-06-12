"""OAS — Kalman filter / EKF + observability (research_proposal_OAS.md, Idea 2).

Implements the depth-domain observer pillar of OAS (OAS_SPEC.md S2): a latent
behavioural state x_k is driven by the *fitted 2x2 affine plant* (reuse PTS) with
process noise, and observed noisily/partially through the per-layer geometric
read-out z_k. A Kalman filter fuses the whole trajectory of weak measurements into
the minimum-variance estimate x_hat_k that the LQR (oas_lqr.py) controls; by the
separation principle the two are designed independently (LQG).

Linear-Gaussian model (S2.1), with the OAS actuator convention B_k = A_k (the
control u_k is added to the state x_k *before* propagation, so the input enters as
A_k u_k, exactly as PTS rollout: x_{k+1} = A_k (x_k + u_k) + b_k):

    x_{k+1} = A_k x_k + B_k u_k + b_k + w_k,   w_k ~ N(0, W)     (process)
    z_k     = H_k x_k + v_k,                   v_k ~ N(0, V)     (measurement)

Standard recursion (implemented exactly, Joseph-form covariance update for the
numerical stability the spec asks for):

    predict:  x_hat^- = A x_hat + B u + b ;            P^- = A P A' + W
    update:   S = H P^- H' + V ;  K = P^- H' S^{-1}
              x_hat = x_hat^- + K (z - H x_hat^-)
              P = (I - K H) P^- (I - K H)' + K V K'   (Joseph form)

Everything here is pure numpy on the 2D plane coordinates (no model, no torch), so
the whole observer claim is validated offline from the saved trajectory file.

Self-tests (S2.6) at the bottom; run `python oas_observer.py` -> prints
`[oas_observer self-test] OK`.
"""

from __future__ import annotations

import sys
import pathlib

# --- sys.path shim (OAS_SPEC.md S0): make BOTH the shared lib and the sibling PTS
# --- dir importable, so `from pts_dynamics import ...` / `from utils import ...`
# --- work no matter where python is launched from.
_R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "pytorch_pure"))
sys.path.insert(0, str(_R / "research_proposals" / "PTS"))

import numpy as np


# =============================================================================
# Kalman filter (time-varying capable)
# =============================================================================


class KalmanFilter:
    """Discrete-time Kalman filter for the OAS linear-Gaussian plant (S2.1).

    Time-varying-capable: A, b, H, W, V are passed to `.step` per layer, so the
    same object filters across the depth band with the fitted per-layer dynamics.
    `B_k = A_k` is the OAS actuator convention (control added to the state before
    propagation, exactly as the PTS rollout); pass u=0 for the autonomous filter.

    Usage:
        kf = KalmanFilter(); kf.reset(x0, P0)
        for z, A, b, H, W, V in ...: xhat, P = kf.step(z, A, b, H, W, V, u=u)
    """

    def __init__(self):
        self.x = None      # current state estimate (n,)
        self.P = None      # current covariance (n,n)

    def reset(self, x0, P0):
        """Initialise the estimate x_hat_0 = x0 and covariance P_0 = P0."""
        self.x = np.asarray(x0, np.float64).reshape(-1).copy()
        self.P = np.asarray(P0, np.float64).copy()
        assert self.P.shape == (self.x.shape[0], self.x.shape[0]), \
            "P0 must be (n,n) matching x0"
        return self.x, self.P

    def step(self, z, A, b, H, W, V, u=0):
        """One predict+update step. Returns (x_hat, P) after fusing measurement z.

        Args:
            z: (m,) measurement at this layer.
            A: (n,n) transition; b: (n,) affine drift; H: (m,n) measurement map.
            W: (n,n) process-noise cov; V: (m,m) measurement-noise cov.
            u: (n,) control applied to the *previous* state (B_k = A_k convention);
               default 0 (autonomous). Pass the control that produced this transition.
        """
        A = np.asarray(A, np.float64)
        b = np.asarray(b, np.float64).reshape(-1)
        H = np.atleast_2d(np.asarray(H, np.float64))
        W = np.asarray(W, np.float64)
        V = np.atleast_2d(np.asarray(V, np.float64))
        z = np.asarray(z, np.float64).reshape(-1)
        u = np.zeros_like(self.x) if np.isscalar(u) and u == 0 else \
            np.asarray(u, np.float64).reshape(-1)

        # --- predict (B_k = A_k: input enters as A u, like x_{k+1}=A(x+u)+b) ----
        x_pred = A @ (self.x + u) + b                 # x_hat^-
        P_pred = A @ self.P @ A.T + W                 # P^-
        P_pred = 0.5 * (P_pred + P_pred.T)            # keep symmetric

        # --- update ------------------------------------------------------------
        S = H @ P_pred @ H.T + V                      # innovation covariance (m,m)
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T  # K = P^- H' S^{-1}, solve form
        innov = z - H @ x_pred                        # innovation
        x_new = x_pred + K @ innov

        # Joseph form: P = (I-KH) P^- (I-KH)' + K V K'  (PSD-preserving, S2.1).
        n = self.x.shape[0]
        ImKH = np.eye(n) - K @ H
        P_new = ImKH @ P_pred @ ImKH.T + K @ V @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        self.x, self.P = x_new, P_new
        return self.x, self.P


def kalman_filter_seq(A, b, H, W, V, z_seq, x0, P0, u_seq=None,
                      no_update_at_0=False):
    """Filter a whole trajectory of measurements (S2.5).

    Each of A, b, H, W, V may be a single time-invariant array OR a per-step
    sequence (length = len(z_seq)). Returns (xhat:(T,n), P:(T,n,n)).

    Args:
        z_seq: (T, m) measurements, one per layer in the band.
        x0, P0: initial estimate / covariance.
        u_seq: (T, n) controls (u applied to produce step t), or None (autonomous).
        no_update_at_0: if True, treat t=0 as the RESET only — the first estimate is
            exactly x0 (no predict, no measurement fusion), so z[0] is not counted
            twice when x0 is itself derived from z[0]. This matches PolicyLQG's
            convention (it does NOT re-fuse z_0 at band[0]); the standard recursion
            resumes at t=1. (F4)
    """
    z_seq = np.asarray(z_seq, np.float64)
    z_seq = z_seq.reshape(z_seq.shape[0], -1)
    T = z_seq.shape[0]

    # Per-step access: a sequence carries one extra leading axis (length T) over
    # its time-invariant base shape (A:2, b:1, H:2, W:2, V:2). `pick` slices it.
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    H = np.asarray(H, np.float64)
    W = np.asarray(W, np.float64)
    V = np.asarray(V, np.float64)

    def pick(arr, t, base_ndim):
        return arr[t] if arr.ndim == base_ndim + 1 else arr

    kf = KalmanFilter()
    kf.reset(x0, P0)
    n = kf.x.shape[0]
    xhat = np.zeros((T, n))
    Ps = np.zeros((T, n, n))
    for t in range(T):
        if t == 0 and no_update_at_0:
            # Reset-only first step: estimate IS x0, no second fusion of z[0] (F4).
            xhat[0] = kf.x
            Ps[0] = kf.P
            continue
        At = pick(A, t, 2)
        bt = pick(b, t, 1)
        Ht = np.atleast_2d(pick(H, t, 2))
        Wt = pick(W, t, 2)
        Vt = np.atleast_2d(pick(V, t, 2))
        ut = 0 if u_seq is None else np.asarray(u_seq, np.float64)[t]
        x, P = kf.step(z_seq[t], At, bt, Ht, Wt, Vt, u=ut)
        xhat[t] = x
        Ps[t] = P
    return xhat, Ps


# =============================================================================
# Process-noise estimation from the fitted plant (reuse PTS residuals, S2.2)
# =============================================================================


def estimate_process_noise(A, b, c_traj):
    """Process-noise covariance W from one-step residuals of the fitted affine plant.

    The PTS plant has a ~14% residual; here we turn that residual into the
    covariance the Kalman filter consumes. For each layer transition k and each
    sample i the residual is eps = c_{k+1} - (A_k c_k + b_k); W is the covariance of
    the pooled residuals (and W_per_layer the per-transition covariance).

    Args:
        A: (L-1, 2, 2); b: (L-1, 2) fitted dynamics.
        c_traj: (N, L, 2) coordinate paths (e.g. phase_portrait trajectories).

    Returns:
        W: (2, 2) pooled residual covariance.
        W_per_layer: (L-1, 2, 2) per-transition residual covariance.
    """
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    C = np.asarray(c_traj, np.float64)
    assert C.ndim == 3 and C.shape[-1] == 2, "c_traj must be (N,L,2)"
    N, L, _ = C.shape
    assert A.shape[0] == L - 1, "A must have L-1 transitions for an (N,L,2) c_traj"

    resids = []                                   # pooled (N*(L-1), 2)
    W_per_layer = np.zeros((L - 1, 2, 2))
    for k in range(L - 1):
        pred = C[:, k, :] @ A[k].T + b[k]         # (N,2) one-step prediction
        eps = C[:, k + 1, :] - pred               # (N,2) residual
        resids.append(eps)
        # per-layer covariance about the per-layer residual mean (the drift A_k,b_k
        # already absorbs the OLS mean, so this is essentially the bias-free cov).
        W_per_layer[k] = np.cov(eps, rowvar=False) if N > 1 else np.zeros((2, 2))
    R = np.concatenate(resids, axis=0)            # (N*(L-1), 2)
    W = np.cov(R, rowvar=False)
    W = 0.5 * (W + W.T)                           # symmetrise
    return W, W_per_layer


# =============================================================================
# Observability (S2.3)
# =============================================================================


def observability_matrix(A, H, n=2):
    """Observability matrix O = [H; HA; ...; H A^{n-1}] for time-invariant (A,H).

    Args:
        A: (n,n) representative transition (e.g. band-mean of the fitted A_k).
        H: (m,n) measurement map.
        n: state dimension (default 2).
    Returns: O of shape (m*n, n).
    """
    A = np.asarray(A, np.float64)
    H = np.atleast_2d(np.asarray(H, np.float64))
    rows = []
    Apow = np.eye(A.shape[0])
    for _ in range(n):
        rows.append(H @ Apow)
        Apow = Apow @ A
    return np.vstack(rows)


def is_observable(A, H, tol=1e-9):
    """True iff the pair (A,H) is observable (full-rank observability matrix)."""
    A = np.asarray(A, np.float64)
    nx = A.shape[0]
    O = observability_matrix(A, H, n=nx)
    return int(np.linalg.matrix_rank(O, tol=tol)) == nx


def observability_gramian(A, H, n):
    """Finite-horizon observability Gramian G = sum_{j=0}^{n-1} (A')^j H' H A^j.

    A continuous *degree of observability*: the spectrum / condition number of G
    quantifies how well each state direction is excited by the measurements over an
    n-step window. Returns the (nx,nx) Gramian; its condition number (well-defined
    only when observable) is the scalar degree-of-observability used downstream.

    Documented caveat: the OAS plant is time-VARYING (A_k per layer); this Gramian
    uses a single representative A (the band-mean) as a tractable proxy. The
    time-varying observability Gramian sums (Phi_{j,0})' H' H Phi_{j,0} over the
    actual transition products Phi and is the honest object for a depth band.
    """
    A = np.asarray(A, np.float64)
    H = np.atleast_2d(np.asarray(H, np.float64))
    nx = A.shape[0]
    G = np.zeros((nx, nx))
    Apow = np.eye(nx)
    for _ in range(n):
        G = G + Apow.T @ (H.T @ H) @ Apow
        Apow = A @ Apow
    return 0.5 * (G + G.T)


def observability_gramian_tv(A_seq, H_seq):
    """Time-varying observability Gramian over a depth band (the honest object).

    G = sum_k Phi_{k,0}' H_k' H_k Phi_{k,0}, where Phi_{k,0} = A_{k-1}...A_0 (the
    state-transition product, Phi_{0,0}=I). A_seq:(T-1?,n,n) supplies the per-step
    transitions; H_seq:(T,m,n) the per-step measurement maps. Use this when the
    band's dynamics rotate appreciably (early/mid layers); the time-invariant
    `observability_gramian` is the band-mean proxy.
    """
    A_seq = np.asarray(A_seq, np.float64)
    H_seq = np.asarray(H_seq, np.float64)
    # A single H given as a 2D (m,n) matrix is broadcast over the WHOLE horizon
    # defined by A_seq: there are len(A_seq) transitions -> len(A_seq)+1 states
    # (Phi_{0,0}=I ... Phi_{len(A_seq),0}), so we sum that many terms. A per-step
    # H_seq:(T,m,n) instead fixes the horizon to its own length T.
    broadcast_H = H_seq.ndim == 2
    T = (A_seq.shape[0] + 1) if broadcast_H else H_seq.shape[0]
    nx = H_seq.shape[-1]
    G = np.zeros((nx, nx))
    Phi = np.eye(nx)
    for k in range(T):
        Hk = np.atleast_2d(H_seq if broadcast_H else H_seq[k])
        G = G + Phi.T @ (Hk.T @ Hk) @ Phi
        if k < A_seq.shape[0]:
            Phi = A_seq[k] @ Phi
    return 0.5 * (G + G.T)


# =============================================================================
# Steady-state (algebraic) Kalman gain — DARE for the LG self-test cross-check
# =============================================================================


def kalman_dare(A, H, W, V, iters=5000, tol=1e-13):
    """Steady-state Kalman filter via the (filtering) Riccati / DARE iteration.

    Iterates the predicted-covariance Riccati recursion
        P^- <- A (P^- - P^- H'(H P^- H' + V)^{-1} H P^-) A' + W
    to its fixed point P_inf, then forms the steady-state gain K_inf = P_inf H'
    (H P_inf H' + V)^{-1}. This is the algebraic filter the time-varying KF must
    converge to on a time-invariant observable system (S2.6a).

    Returns dict: {"P": P_inf (predicted cov), "K": K_inf (steady gain)}.
    """
    A = np.asarray(A, np.float64)
    H = np.atleast_2d(np.asarray(H, np.float64))
    W = np.asarray(W, np.float64)
    V = np.atleast_2d(np.asarray(V, np.float64))
    nx = A.shape[0]
    P = W.copy()
    for _ in range(iters):
        S = H @ P @ H.T + V
        K = np.linalg.solve(S.T, (P @ H.T).T).T            # P H' S^{-1}
        P_upd = P - K @ H @ P                              # (I-KH)P^-
        P_new = A @ P_upd @ A.T + W
        P_new = 0.5 * (P_new + P_new.T)
        if np.max(np.abs(P_new - P)) < tol:
            P = P_new
            break
        P = P_new
    S = H @ P @ H.T + V
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    return {"P": P, "K": K}


# =============================================================================
# EKF on the circle (S2.4) — SO(2) variant for Exp 5 / circle nonlinearity risk
# =============================================================================


def ekf_angle(z_seq, W, V, x0=None, P0=None, dt=1.0):
    """Extended Kalman filter for the angle on the circle (S2.4).

    State x = [phi, omega] (angle and its per-step rate); constant-velocity process
        phi_{k+1}   = phi_k + dt * omega_k
        omega_{k+1} = omega_k                          (+ process noise W)
    scalar measurement z = phi (H = [1, 0]) with a WRAPPED innovation
        innov = atan2(sin(z - phi^-), cos(z - phi^-))
    so a measurement near +pi and a prediction near -pi are correctly fused as a
    small correction rather than a ~2pi jump. The model is linear; the only
    nonlinearity is the wrap, handled in the innovation (an EKF on the circle).

    Args:
        z_seq: (T,) angle measurements (radians, may wrap across +/-pi).
        W: (2,2) process-noise cov; V: scalar/(1,1) measurement-noise variance.
        x0: (2,) initial [phi, omega] (default [z_seq[0], 0]); P0: (2,2) init cov.
        dt: step size (default 1, one layer per step).

    Returns:
        phi_hat: (T,) filtered angle (wrapped to (-pi, pi]).
        omega_hat: (T,) filtered rate.
        P: (T,2,2) covariances.
    """
    z_seq = np.asarray(z_seq, np.float64).reshape(-1)
    T = z_seq.shape[0]
    W = np.asarray(W, np.float64)
    V = float(np.asarray(V).reshape(-1)[0])
    A = np.array([[1.0, dt], [0.0, 1.0]])         # constant-velocity transition
    H = np.array([[1.0, 0.0]])
    x = np.array([z_seq[0], 0.0]) if x0 is None else np.asarray(x0, np.float64).copy()
    P = np.eye(2) if P0 is None else np.asarray(P0, np.float64).copy()

    phi_hat = np.zeros(T)
    omega_hat = np.zeros(T)
    Ps = np.zeros((T, 2, 2))
    for k in range(T):
        # predict
        x = A @ x
        P = A @ P @ A.T + W
        P = 0.5 * (P + P.T)
        # update with WRAPPED innovation
        S = float((H @ P @ H.T).reshape(())) + V
        K = (P @ H.T).reshape(-1) / S             # (2,)
        innov = np.arctan2(np.sin(z_seq[k] - x[0]), np.cos(z_seq[k] - x[0]))
        x = x + K * innov
        ImKH = np.eye(2) - np.outer(K, H.reshape(-1))
        P = ImKH @ P @ ImKH.T + np.outer(K, K) * V   # Joseph form
        P = 0.5 * (P + P.T)
        phi_hat[k] = np.arctan2(np.sin(x[0]), np.cos(x[0]))  # wrap report to (-pi,pi]
        omega_hat[k] = x[1]
        Ps[k] = P
    return phi_hat, omega_hat, Ps


# =============================================================================
# Self-tests  (OAS_SPEC.md S2.6)
# =============================================================================


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # ---- (a) synthetic LG system: filtered MSE < raw-measurement MSE, and the
    # ----     time-varying KF steady-state gain matches the algebraic DARE gain ---
    A = np.array([[0.8, 0.15], [-0.1, 0.7]])      # stable time-invariant 2x2
    b = np.array([0.05, -0.02])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])        # full (noisy) measurement
    W = np.array([[0.02, 0.005], [0.005, 0.03]])  # process noise (the ~14% residual)
    V = np.array([[0.25, 0.0], [0.0, 0.25]])      # heavy measurement noise

    T = 400
    x = np.array([1.0, -0.5])
    Wc = np.linalg.cholesky(W)
    Vc = np.linalg.cholesky(V)
    xs = np.zeros((T, 2)); zs = np.zeros((T, 2))
    for k in range(T):
        zs[k] = H @ x + Vc @ rng.randn(2)
        xs[k] = x
        x = A @ x + b + Wc @ rng.randn(2)
    xhat, Ps = kalman_filter_seq(A, b, H, W, V, zs, x0=zs[0], P0=np.eye(2))
    mse_filt = float(np.mean(np.sum((xhat - xs) ** 2, axis=1)))
    mse_raw = float(np.mean(np.sum((zs - xs) ** 2, axis=1)))   # H=I so z is a raw x-estimate
    print(f"[a] LG: filtered MSE={mse_filt:.4f}  raw-measurement MSE={mse_raw:.4f}")
    assert mse_filt < mse_raw, "Kalman fusion must beat the raw measurement"

    ss = kalman_dare(A, H, W, V)                  # algebraic steady-state filter
    # the running KF gain at the end of the sequence: K = P^- H' S^{-1} from P_{T-1}.
    P_end = Ps[-1]
    P_pred_end = A @ P_end @ A.T + W
    S_end = H @ P_pred_end @ H.T + V
    K_run = np.linalg.solve(S_end.T, (P_pred_end @ H.T).T).T
    gain_err = float(np.max(np.abs(K_run - ss["K"])))
    print(f"[a] steady-state gain: running KF K vs DARE K  max|diff|={gain_err:.2e}")
    assert gain_err < 1e-4, "running KF gain must match the algebraic DARE gain"

    # ---- (b) PARTIAL observability: dim x = 2, SCALAR z. An OBSERVABLE pair lets
    # ----     the fused SEQUENCE recover x where a single measurement cannot; an
    # ----     UNOBSERVABLE pair is flagged is_observable=False ---------------------
    # Observable pair: we only ever measure x[0], but A couples x[0]<-x[1], so the
    # sequence of x[0] observations reveals x[1] (classic position-only -> velocity).
    A2 = np.array([[1.0, 1.0], [0.0, 1.0]])       # x[1] feeds x[0] next step
    b2 = np.zeros(2)
    H_obs = np.array([[1.0, 0.0]])                # scalar measurement of x[0] only
    Wsmall = 1e-4 * np.eye(2)
    Vscal = np.array([[0.04]])
    assert is_observable(A2, H_obs), "(A2,H_obs) should be observable"
    G = observability_gramian(A2, H_obs, n=2)
    cond_obs = float(np.linalg.cond(G))
    print(f"[b] observable pair: is_observable=True, Gramian cond={cond_obs:.2f}")

    Tb = 60
    xb = np.array([0.0, 0.3])                     # nonzero velocity to be recovered
    xsb = np.zeros((Tb, 2)); zsb = np.zeros((Tb, 1))
    Wbc = np.linalg.cholesky(Wsmall); Vbc = np.sqrt(Vscal[0, 0])
    for k in range(Tb):
        zsb[k, 0] = (H_obs @ xb)[0] + Vbc * rng.randn()
        xsb[k] = xb
        xb = A2 @ xb + Wbc @ rng.randn(2)
    xhat_b, Pb = kalman_filter_seq(A2, b2, H_obs, Wsmall, Vscal, zsb,
                                   x0=np.array([zsb[0, 0], 0.0]), P0=np.eye(2))
    # second-half MSE on the UNMEASURED component x[1] (steady state, transient gone)
    half = Tb // 2
    mse_x1_fused = float(np.mean((xhat_b[half:, 1] - xsb[half:, 1]) ** 2))
    # "single measurement" can say nothing about x[1] -> best guess is the prior mean 0
    mse_x1_single = float(np.mean((0.0 - xsb[half:, 1]) ** 2))
    print(f"[b] unmeasured x[1] MSE: fused={mse_x1_fused:.4f}  single-meas={mse_x1_single:.4f}")
    assert mse_x1_fused < 0.25 * mse_x1_single, "fusing the sequence must recover x[1]"

    # Unobservable pair: H sees only x[0] and A is block-diagonal, so x[1] never
    # influences any measurement -> rank-deficient observability matrix.
    A_un = np.array([[0.9, 0.0], [0.0, 0.8]])
    assert not is_observable(A_un, H_obs), "(A_un,H_obs) must be unobservable"
    print(f"[b] unobservable pair (diagonal A, x[0]-only H): is_observable=False OK")

    # The time-VARYING Gramian must reduce to the time-invariant one on a constant-A
    # band (sum over the FULL horizon). Two equivalent inputs: a single H broadcast
    # over the band, and the explicit per-step H_seq:(T,m,n). Both must agree with
    # observability_gramian over n = len(A_seq)+1 states.
    A_band = np.stack([A2] * 4)                   # 4 transitions -> 5 states
    G_ti = observability_gramian(A2, H_obs, n=A_band.shape[0] + 1)
    G_tv_bcast = observability_gramian_tv(A_band, H_obs)            # single H -> broadcast
    G_tv_steps = observability_gramian_tv(A_band, np.stack([H_obs] * (A_band.shape[0] + 1)))
    print(f"[b] TV Gramian: broadcast vs TI max|diff|={np.max(np.abs(G_tv_bcast - G_ti)):.2e}, "
          f"per-step vs TI max|diff|={np.max(np.abs(G_tv_steps - G_ti)):.2e}")
    assert np.max(np.abs(G_tv_bcast - G_ti)) < 1e-9, "broadcast TV Gramian must sum the full horizon"
    assert np.max(np.abs(G_tv_steps - G_ti)) < 1e-9, "per-step TV Gramian must match TI on constant A"

    # no_update_at_0: the first estimate must be EXACTLY x0 (reset only, z[0] not
    # re-fused), matching PolicyLQG's band[0] convention (F4). The default
    # (update-at-0) path runs a full predict+update at t=0; with a prior x0 whose
    # measured component differs from z[0] the second fusion MOVES the estimate, so
    # the reset-only path (which trusts x0) must differ from it at t=0.
    x0_seed = np.array([zsb[0, 0] + 1.0, 0.7])         # deliberately off from z[0]
    xhat_nf, _ = kalman_filter_seq(A2, b2, H_obs, Wsmall, Vscal, zsb,
                                   x0=x0_seed, P0=np.eye(2), no_update_at_0=True)
    xhat_def, _ = kalman_filter_seq(A2, b2, H_obs, Wsmall, Vscal, zsb,
                                    x0=x0_seed, P0=np.eye(2), no_update_at_0=False)
    assert np.allclose(xhat_nf[0], x0_seed, atol=1e-12), \
        "no_update_at_0 must leave the first estimate exactly at x0 (no second fusion)"
    assert not np.allclose(xhat_nf[0], xhat_def[0], atol=1e-6), \
        "the default path re-fuses z[0] at t=0, so it must differ from the reset-only x0"
    print(f"[b] no_update_at_0: xhat_0 == x0 (reset only), default re-fuses z_0 "
          f"(|diff|={np.max(np.abs(xhat_nf[0] - xhat_def[0])):.3e})")

    # ---- (c) degeneracy: V->0 => x_hat -> z (trust the measurement);
    # ----                  W->0 => x_hat -> model prediction (trust the model) ----
    Hc = np.eye(2)
    # V -> 0
    kf = KalmanFilter(); kf.reset(np.array([5.0, -5.0]), np.eye(2))
    z_meas = np.array([0.3, -0.7])
    xc, _ = kf.step(z_meas, A, b, Hc, W, 1e-12 * np.eye(2))
    print(f"[c] V->0: x_hat={xc} ~= z={z_meas}  max|diff|={np.max(np.abs(xc - z_meas)):.2e}")
    assert np.max(np.abs(xc - z_meas)) < 1e-5, "V->0 must give x_hat -> z"
    # W -> 0 (and a confident prior): the predict step dominates, x_hat -> A x + b.
    x_prev = np.array([0.4, 0.1])
    kf2 = KalmanFilter(); kf2.reset(x_prev, 1e-12 * np.eye(2))
    model_pred = A @ x_prev + b
    xc2, _ = kf2.step(np.array([9.0, 9.0]), A, b, Hc, 1e-12 * np.eye(2), V)
    print(f"[c] W->0: x_hat={xc2} ~= model={model_pred}  max|diff|={np.max(np.abs(xc2 - model_pred)):.2e}")
    assert np.max(np.abs(xc2 - model_pred)) < 1e-5, "W->0 must give x_hat -> model prediction"

    # ---- (d) EKF tracks a WRAPPING ramp angle without divergence -----------------
    Td = 200
    omega_true = 0.2                              # rad/step ramp -> wraps every ~31 steps
    phi_true = _wrap(np.arange(Td) * omega_true)
    z_ang = _wrap(phi_true + 0.15 * rng.randn(Td))   # noisy wrapped measurements
    W_ekf = np.array([[1e-4, 0.0], [0.0, 1e-5]])
    phi_hat, omega_hat, P_ekf = ekf_angle(z_ang, W_ekf, V=0.15 ** 2,
                                          x0=np.array([phi_true[0], 0.0]))
    half_d = Td // 2
    ang_err = _wrap(phi_hat[half_d:] - phi_true[half_d:])
    rmse_ang = float(np.sqrt(np.mean(ang_err ** 2)))
    om_err = float(np.mean(np.abs(omega_hat[half_d:] - omega_true)))
    print(f"[d] EKF wrapping ramp: angle RMSE={rmse_ang:.4f} rad  "
          f"omega_hat~{omega_hat[-1]:.3f} (true {omega_true})  |om_err|={om_err:.3f}")
    assert np.all(np.isfinite(phi_hat)), "EKF diverged (non-finite)"
    assert rmse_ang < 0.1, "EKF must track the wrapping ramp tightly"
    assert om_err < 0.05, "EKF must recover the ramp rate omega"

    print("[oas_observer self-test] OK")
