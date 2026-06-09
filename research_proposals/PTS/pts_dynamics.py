"""PTS — per-layer 2x2 linear dynamics model in the Angular Steering plane.

Implements the cheap, offline dynamics model the Predictive Trajectory Steering
(PTS) proposal (research_proposal_PTS.md, S5.1) relies on:

    c_{k+1} ~= A_k c_k + b_k          (A_k in R^{2x2}, b_k in R^2)

fit by ordinary least squares from contrastive forward passes. This is option
(a) of S5.1 — "from contrastive data (cheap, offline)" — which the proposal argues
is sufficient (citing arXiv:2603.12541, low-order linear depth dynamics) so we do
NOT need full d x d Jacobians. The affine term b_k absorbs the layer's mean drift
(layernorm bias, residual offset) that a purely linear A_k c_k cannot represent.

Everything here is pure numpy on the 2D coordinates c = (b1.z, b2.z); it never
touches the model, so the entire dynamics-modelling claim can be validated offline
from a saved trajectory file.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Fitting
# =============================================================================


def _lstsq_ridge(X: np.ndarray, Y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve min_W ||X W - Y||^2 + ridge ||W||^2.  X:(N,p) Y:(N,m) -> W:(p,m).

    Uses an SVD-based least squares (np.linalg.lstsq) so rank-deficient designs —
    e.g. layer 0, whose last-token activation is identical across prompts because
    the chat-template suffix is shared — yield the stable min-norm solution rather
    than a singular-matrix error. (Such layers are never actuated anyway.)
    """
    if ridge > 0.0:
        p = X.shape[1]
        Xa = np.concatenate([X, np.sqrt(ridge) * np.eye(p)], axis=0)
        Ya = np.concatenate([Y, np.zeros((p, Y.shape[1]))], axis=0)
        return np.linalg.lstsq(Xa, Ya, rcond=None)[0]
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def fit_layer_dynamics(
    c1: np.ndarray,
    c2: np.ndarray,
    affine: bool = True,
    ridge: float = 0.0,
) -> dict:
    """Fit a 2x2 (affine) linear map per layer transition from coordinate paths.

    Args:
        c1, c2: arrays of shape (N, L) — the b1 / b2 coordinates of N prompts at
            each of L layers (e.g. phase_portrait.compute_phase_trajectories output,
            or fresh residual-stream coordinates).
        affine: if True fit c_{k+1} = A_k c_k + b_k, else pure-linear (b_k = 0).
        ridge: Tikhonov regularisation on the fitted parameters.

    Returns dict:
        - "A": (L-1, 2, 2)
        - "b": (L-1, 2)
        - "affine": bool
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    assert c1.shape == c2.shape and c1.ndim == 2, "c1,c2 must be (N,L)"
    N, L = c1.shape
    C = np.stack([c1, c2], axis=-1)  # (N, L, 2)

    A = np.zeros((L - 1, 2, 2))
    b = np.zeros((L - 1, 2))
    for k in range(L - 1):
        X = C[:, k, :]      # (N, 2)
        Y = C[:, k + 1, :]  # (N, 2)
        if affine:
            Xb = np.concatenate([X, np.ones((N, 1))], axis=1)  # (N, 3)
            W = _lstsq_ridge(Xb, Y, ridge)                     # (3, 2)
            A[k] = W[:2].T
            b[k] = W[2]
        else:
            W = _lstsq_ridge(X, Y, ridge)                      # (2, 2)
            A[k] = W.T
    return {"A": A, "b": b, "affine": bool(affine)}


# =============================================================================
# Rollout / prediction
# =============================================================================


def rollout(
    A: np.ndarray,
    b: np.ndarray,
    c0: np.ndarray,
    controls: np.ndarray | None = None,
    start: int = 0,
) -> np.ndarray:
    """Roll the fitted dynamics forward from c0.

    With the PTS actuator convention B_k = I, a control u_k is *added to the
    post-state* at layer k (the coordinate the actuator commits to), so the
    recursion is

        s_0   = c0 + u_0
        s_{k} = A_{start+k-1} s_{k-1} + b_{start+k-1} + u_k     (k >= 1)

    where s_k is the actuated coordinate at layer (start+k). With controls=None
    this is the autonomous open-loop prediction s_k = c0 propagated by A,b.

    Args:
        A, b: (L-1, 2, 2) / (L-1, 2) fitted dynamics.
        c0: (..., 2) initial coordinate(s) at layer `start`.
        controls: (..., H, 2) per-layer additive controls, or None.
        start: index of c0 within the layer grid (controls/A indexed from here).

    Returns:
        traj: (..., H+1, 2) actuated coordinates s_0..s_H (H = #controls, or
        the number of remaining transitions if controls is None).
    """
    c0 = np.asarray(c0, dtype=np.float64)
    lead = c0.shape[:-1]
    if controls is None:
        H = A.shape[0] - start
        controls = np.zeros(lead + (H, 2))
    else:
        controls = np.asarray(controls, dtype=np.float64)
        H = controls.shape[-2]

    traj = np.zeros(lead + (H + 1, 2))
    s = c0 + controls[..., 0, :]         # actuated state at layer `start`
    traj[..., 0, :] = s
    for k in range(1, H + 1):
        Ak = A[start + k - 1]            # (2,2)
        bk = b[start + k - 1]            # (2,)
        s = s @ Ak.T + bk                # autonomous propagation across the block
        if k <= H - 1:                   # last state (k==H) is terminal: no control
            s = s + controls[..., k, :]
        traj[..., k, :] = s
    return traj


def autonomous_predict(A: np.ndarray, b: np.ndarray, c_k: np.ndarray, k: int,
                       steps: int = 1) -> np.ndarray:
    """Predict c_{k+steps} from c_k under the autonomous fitted dynamics."""
    c = np.asarray(c_k, dtype=np.float64)
    for j in range(steps):
        c = c @ A[k + j].T + b[k + j]
    return c


# =============================================================================
# Validation
# =============================================================================


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def validate_dynamics(
    c1: np.ndarray,
    c2: np.ndarray,
    fit: dict,
    horizons: tuple[int, ...] = (1, 5),
) -> dict:
    """Multi-horizon prediction accuracy of the fitted dynamics on (held-out) data.

    For each h in `horizons`, predict c_{k+h} from the *true* c_k under the
    autonomous model and report RMSE (in coordinate units), normalised RMSE
    (by the per-layer coordinate scale), and R^2, aggregated over layers/samples.
    Also reports the full open-loop rollout from layer 0 (the hardest test).

    Returns dict with per-horizon metrics and per-layer 1-step R^2.
    """
    c1 = np.asarray(c1, np.float64)
    c2 = np.asarray(c2, np.float64)
    C = np.stack([c1, c2], axis=-1)  # (N, L, 2)
    N, L, _ = C.shape
    A, b = fit["A"], fit["b"]

    out: dict = {"horizons": list(horizons)}

    for h in horizons:
        preds, trues = [], []
        for k in range(0, L - h):
            pred = autonomous_predict(A, b, C[:, k, :], k, steps=h)  # (N,2)
            preds.append(pred)
            trues.append(C[:, k + h, :])
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        err = np.linalg.norm(preds - trues, axis=-1)
        scale = np.linalg.norm(trues, axis=-1).mean() + 1e-12
        out[f"rmse_h{h}"] = float(np.sqrt((err ** 2).mean()))
        out[f"nrmse_h{h}"] = float(np.sqrt((err ** 2).mean()) / scale)
        out[f"r2_h{h}"] = _r2(trues, preds)

    # per-layer 1-step R^2 (diagnostic for which depths the linear model fits)
    per_layer_r2 = []
    for k in range(L - 1):
        pred = C[:, k, :] @ A[k].T + b[k]
        per_layer_r2.append(_r2(C[:, k + 1, :], pred))
    out["per_layer_r2_1step"] = per_layer_r2

    # full open-loop rollout from layer 0 (compounding error)
    roll = rollout(A, b, C[:, 0, :])          # (N, L, 2)
    err = np.linalg.norm(roll - C, axis=-1)   # (N, L)
    out["full_rollout_rmse_per_layer"] = err.mean(axis=0).tolist()
    out["full_rollout_rmse"] = float(err.mean())

    return out


if __name__ == "__main__":
    # Self-test on synthetic stable affine dynamics + recovery.
    rng = np.random.RandomState(0)
    L, N = 12, 400
    A_true = np.stack([0.9 * np.eye(2) + 0.05 * rng.randn(2, 2) for _ in range(L - 1)])
    b_true = 0.2 * rng.randn(L - 1, 2)
    C = np.zeros((N, L, 2))
    C[:, 0, :] = rng.randn(N, 2)
    for k in range(L - 1):
        C[:, k + 1, :] = C[:, k, :] @ A_true[k].T + b_true[k] + 0.01 * rng.randn(N, 2)
    fit = fit_layer_dynamics(C[..., 0], C[..., 1], affine=True)
    a_err = np.abs(fit["A"] - A_true).max()
    b_err = np.abs(fit["b"] - b_true).max()
    val = validate_dynamics(C[..., 0], C[..., 1], fit, horizons=(1, 5))
    print(f"[pts_dynamics self-test] A_err={a_err:.3e} b_err={b_err:.3e} "
          f"r2_h1={val['r2_h1']:.4f} r2_h5={val['r2_h5']:.4f} "
          f"full_rollout_rmse={val['full_rollout_rmse']:.3e}")
    assert a_err < 0.05 and b_err < 0.05 and val["r2_h1"] > 0.98
    print("[pts_dynamics self-test] OK")
