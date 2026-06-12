"""OAS — forward-pass-coupled multi-layer steering (the in-the-loop glue).

This is the piece that turns the OAS planner (oas_lqr.py) and observer
(oas_observer.py) into a *closed-in-depth* controller on a live forward pass
(OAS_SPEC.md S3): a hook on every actuated band layer reads the in-plane
coordinate c_k of the residual stream, asks a policy for the target angle
theta_k, and applies the norm-preserving SO(2) reset — the verified Angular
Steering actuator, identical in semantics to pts_controller._apply_reset /
clas_controller.make_clas_hook.

Every controller routes through the SAME actuator and differs only in how theta_k
is produced:
  * PolicyNone            — read-only (records, keeps the current angle)
  * PolicyDeadbeat        — single-layer slam to a fixed angle (Angular Steering,
                            the Exp-3 baseline)
  * PolicyMultiAngle      — one fixed angle held across a BAND (Exp-1 authority lever)
  * PolicySoftLandingLQR  — open-loop LQR policy s_k = F_k c_k + g_k ; theta=angle(s_k)
  * PolicyLQG             — Kalman-filter the per-layer coordinate, then apply the
                            LQR policy on the filtered estimate: theta=angle(F_k xhat_k+g_k)

Actuator convention (MUST match PTS exactly, OAS_SPEC S0):

    state    x_k = the *natural incoming* coordinate at layer k (the live c_k the
             hook reads, before actuation);
    decision s_k = F_k x_k + g_k  (the actuated coordinate that should flow
             downstream); control u_k = s_k - x_k;
    physical realization: the hook keeps ||x_k|| and sets the angle to
             theta_k = atan2(s_k[1], s_k[0]) (norm-preserving SO(2) reset). The
             dropped magnitude ||s_k|| vs ||x_k|| is the Exp-5 "norm-preserving tax".

Each policy maps the measured coordinates c:(M,2) torch at a layer to angles
theta:(M,) torch in radians (M = batch * positions flattened) plus an optional
additive control (M,2) for logging. The LQR/LQG policies run on the CPU in numpy
(2D plane, tiny), then theta is pushed back to the device. PolicyLQG runs the
oas_observer.KalmanFilter sequentially across the band within one forward pass
(hooks fire in layer order) and MUST reset filter state per forward pass — the
driver calls `.reset()` before each prefill/decode-step, and the policy also
self-guards by detecting layer_idx == band[0].

Self-tests (S3.1) at the bottom; run `python oas_controller.py` -> prints
`[oas_controller self-test] OK`. No model is needed.
"""

from __future__ import annotations

import sys
import pathlib

# --- sys.path shim (OAS_SPEC.md S0): expose BOTH the shared lib and the sibling
# --- PTS dir, so `from oas_lqr import ...`, `from oas_observer import ...`,
# --- `from pts_dynamics import ...`, `from utils import ...` all resolve no matter
# --- where python is launched from. OAS is self-contained: it imports only the
# --- shared lib, pts_dynamics (the pure-numpy plant fit), and its own oas_* modules.
_R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "pytorch_pure"))
sys.path.insert(0, str(_R / "research_proposals" / "PTS"))

import numpy as np
import torch

from oas_lqr import finite_horizon_lqr
from oas_observer import KalmanFilter


# =============================================================================
# Shared state + actuator
# =============================================================================


class OASState:
    """Mutable steering state shared by all layer hooks for one decode/prefill.

    `record` toggles logging of the per-layer measured coordinate, applied angle,
    and (for LQR/LQG) the additive control / filtered estimate, into `.log` keyed
    by layer index — used to measure planned-vs-realised tracking error. Mirrors
    pts_controller.PTSState.
    """

    def __init__(self, policy, record: bool = False):
        self.policy = policy
        self.enabled = True
        self.record = record
        self.log: dict[int, dict] = {}

    def reset_log(self):
        self.log = {}

    def reset_policy(self):
        """Reset the policy's per-forward-pass internal state (e.g. the LQG filter).

        The driver MUST call this before each prefill / decode-step so the observer
        starts fresh at the band's first layer. Stateless policies are unaffected.
        """
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()


def _apply_reset(hidden, b1, b2, theta):
    """Norm-preserving SO(2) reset of the in-plane component to angle `theta`.

    hidden: (B,S,d); b1,b2: (d,); theta: (B,S) radians. Returns steered (B,S,d).
    Identical semantics to pts_controller._apply_reset / clas_controller.make_clas_hook /
    Angular Steering: it keeps the in-plane magnitude ||proj(hidden)|| and only sets
    the angle, so the dropped ||s_k||-vs-||x_k|| magnitude is the Exp-5 tax.
    """
    c1 = hidden @ b1                                  # (B,S)
    c2 = hidden @ b2
    proj = c1.unsqueeze(-1) * b1 + c2.unsqueeze(-1) * b2
    scale = torch.sqrt(c1 ** 2 + c2 ** 2).unsqueeze(-1)
    ct = torch.cos(theta).to(hidden.dtype).unsqueeze(-1)   # (B,S,1)
    st = torch.sin(theta).to(hidden.dtype).unsqueeze(-1)
    steer = ct * b1 + st * b2                         # (B,S,d)
    return hidden - proj + scale * steer


def make_oas_layer_hook(layer_idx: int, b1: torch.Tensor, b2: torch.Tensor,
                        state: OASState):
    """Forward hook for one actuated layer; tuple-aware (HF decoder layers).

    Reads the live in-plane coordinate c_k of the layer output, asks the policy for
    the per-token target angle theta_k (and an optional additive control for the
    log), and applies the norm-preserving SO(2) reset. Mirrors
    pts_controller.make_pts_layer_hook.
    """

    def hook(module, inp, out):
        if not state.enabled:
            return out
        hidden = out[0] if isinstance(out, tuple) else out
        _b1 = b1.to(hidden.device, hidden.dtype)
        _b2 = b2.to(hidden.device, hidden.dtype)

        c1 = (hidden @ _b1)                            # (B,S)
        c2 = (hidden @ _b2)
        B, S = c1.shape
        coords = torch.stack([c1, c2], dim=-1).reshape(-1, 2)  # (B*S, 2)

        theta_flat, ctrl = state.policy(layer_idx, coords)     # (B*S,) [, (B*S,2)]
        theta = theta_flat.reshape(B, S)

        if state.record:
            entry = state.log.setdefault(layer_idx, {"c": [], "theta": [], "u": []})
            entry["c"].append(coords.detach().float().cpu().numpy())
            entry["theta"].append(theta_flat.detach().float().cpu().numpy())
            if ctrl is not None:
                entry["u"].append(np.asarray(ctrl))

        steered = _apply_reset(hidden, _b1, _b2, theta)
        if isinstance(out, tuple):
            return (steered,) + tuple(out[1:])
        return steered

    return hook


def attach_oas_hooks(module_dict, layers, b1, b2, state: OASState):
    """Return [(module, hook), ...] for utils.add_hooks over the actuated layers."""
    return [
        (module_dict[f"model.layers.{k}"], make_oas_layer_hook(k, b1, b2, state))
        for k in layers
    ]


# =============================================================================
# Policies  (coords c:(M,2) torch on device -> theta:(M,) torch, [control:(M,2)])
# =============================================================================


def _keep_angle(coords):
    """The read-only angle of the live coordinate (no behavioural change)."""
    return torch.atan2(coords[:, 1], coords[:, 0])


class PolicyNone:
    """Read-only: keep the current angle (records coordinates, applies no steering)."""

    def __call__(self, layer_idx, coords):
        return _keep_angle(coords), None


class PolicyDeadbeat:
    """Single-layer slam to a fixed target angle (the existing Angular Steering, the
    Exp-3 deadbeat baseline). At the one actuated layer the angle is set to
    `target_angle` regardless of the incoming coordinate; every other layer keeps
    its angle. This is the degenerate R->0, Q_term->inf, no-observer corner of OAS
    (OAS_SPEC S0): the *physical* deadbeat baseline simply uses the target angle
    directly, so this policy needs no plant fit."""

    def __init__(self, target_angle: float, layer: int):
        self.target = float(target_angle)
        self.layer = int(layer)

    def __call__(self, layer_idx, coords):
        if layer_idx == self.layer:
            theta = torch.full((coords.shape[0],), self.target,
                               device=coords.device, dtype=torch.float32)
        else:
            theta = _keep_angle(coords)
        return theta, None


class PolicyMultiAngle:
    """One fixed target angle held across a BAND of layers (the Exp-1 authority
    lever). Same fixed-angle actuation as PolicyDeadbeat but applied at every layer
    in `layers` — the multi-layer generalisation CLAS left open. Distinct from the
    soft-landing LQR (which lands the angle only at the terminal layer with small
    per-layer pushes); this one forces the SAME angle at every band layer."""

    def __init__(self, target_angle: float, layers):
        self.target = float(target_angle)
        self.layers = set(int(k) for k in layers)

    def __call__(self, layer_idx, coords):
        if layer_idx in self.layers:
            theta = torch.full((coords.shape[0],), self.target,
                               device=coords.device, dtype=torch.float32)
        else:
            theta = _keep_angle(coords)
        return theta, None


class PolicySoftLandingLQR:
    """Open-loop soft-landing LQR (OAS_SPEC S3 / Idea 1).

    At each band layer the actuated coordinate is the Riccati decision
    s_k = F_k c_k + g_k (gains from oas_lqr.finite_horizon_lqr / build_softlanding),
    where c_k is the *measured* natural incoming coordinate the hook reads. The hook
    then realises only the angle theta_k = angle(s_k) (norm-preserving). This is the
    in-the-loop analogue of oas_lqr.lqr_rollout: feeding the live c_k as x_k means
    the policy emits exactly the rollout's s_k (and hence its angle) when the
    actuator is additive — the angle-only realization introduces the known Exp-5 tax.

    `gains` is the dict from finite_horizon_lqr ({"F","g","band","terminal_layer",...});
    `ref` is the (L,2) per-layer reference (kept for parity with PolicyLQG / logging,
    not needed for the affine policy evaluation). Layers outside the band keep angle.
    """

    def __init__(self, gains: dict, ref):
        self.F = gains["F"]
        self.g = gains["g"]
        self.band = set(int(k) for k in gains["band"])
        self.ref = None if ref is None else np.asarray(ref, np.float64)

    def __call__(self, layer_idx, coords):
        device = coords.device
        if layer_idx not in self.band:
            return _keep_angle(coords), None
        xi = coords.detach().float().cpu().numpy()            # (M,2) measured x_k
        s = xi @ self.F[layer_idx].T + self.g[layer_idx]      # (M,2) s_k = F_k x_k + g_k
        theta_np = np.arctan2(s[:, 1], s[:, 0])
        theta = torch.from_numpy(np.asarray(theta_np, np.float32)).to(device)
        u = s - xi                                            # additive control (log)
        return theta, u


class PolicyLQG:
    """Closed-loop LQG (OAS_SPEC S3 / Ideas 1+2): observer + LQR by separation.

    The latent behavioural state x_k is observed noisily/partially through the
    per-layer coordinate z_k = H_k x_k + v_k (here z_k = the measured c_k, H = I). A
    Kalman filter (oas_observer.KalmanFilter) fuses the trajectory of measurements
    across the band within ONE forward pass into the minimum-variance estimate
    xhat_k, and the LQR acts on that estimate: the actuated coordinate is
    s_k = F_k xhat_k + g_k, realised as the angle theta_k = angle(s_k).

    The filter steps sequentially as the band hooks fire in layer order:
      * at band[0] it resets to xhat_0 = z_0 (the first measurement) with P0;
      * at every later band layer it predicts using the previous transition
        (A_{k-1}, b_{k-1}) and the control u_{k-1} = s_{k-1} - xhat_{k-1} that was
        actually commanded (B_k = A_k convention), then updates with the new
        measurement z_k.
    It MUST reset per forward pass: the driver calls .reset() before each
    prefill/decode-step, and the policy ALSO self-guards by re-initialising whenever
    it sees layer_idx == band[0]. With V -> 0 the update trusts the measurement
    fully (xhat_k -> z_k = c_k), so PolicyLQG collapses to PolicySoftLandingLQR.

    `kalman_cfg` carries the noise model and plant the filter needs:
        {"A": (L-1,2,2), "b": (L-1,2), "H": (2,2), "W": (2,2)|(L-1,2,2),
         "V": (2,2), "P0": (2,2)}
    A/b are the fitted plant (same grid as the gains); W may be pooled or per-layer.
    """

    def __init__(self, gains: dict, ref, kalman_cfg: dict):
        self.F = gains["F"]
        self.g = gains["g"]
        band = [int(k) for k in gains["band"]]
        self.band = band
        self.band_set = set(band)
        self.k0 = band[0]
        self.ref = None if ref is None else np.asarray(ref, np.float64)

        cfg = kalman_cfg
        self.A = np.asarray(cfg["A"], np.float64)             # (L-1,2,2)
        self.b = np.asarray(cfg["b"], np.float64)             # (L-1,2)
        self.H = np.atleast_2d(np.asarray(cfg.get("H", np.eye(2)), np.float64))
        W = np.asarray(cfg["W"], np.float64)
        # W may be a single pooled (2,2) cov or a per-layer (L-1,2,2) stack.
        self.W_per_layer = W.ndim == 3
        self.W = W
        self.V = np.atleast_2d(np.asarray(cfg["V"], np.float64))
        self.P0 = np.asarray(cfg.get("P0", np.eye(2)), np.float64)

        # Per-forward-pass mutable state (one filter bank per flattened token). These
        # are (re)built lazily on the first band layer of each forward pass, because
        # only then do we know M (batch * positions).
        self._kfs: list[KalmanFilter] | None = None
        self._prev_layer: int | None = None
        self._prev_u: np.ndarray | None = None        # (M,2) control commanded at k-1

    def reset(self):
        """Drop the per-forward-pass filter bank (called by the driver each pass)."""
        self._kfs = None
        self._prev_layer = None
        self._prev_u = None

    def _W_at(self, k):
        return self.W[k] if self.W_per_layer else self.W

    def __call__(self, layer_idx, coords):
        device = coords.device
        if layer_idx not in self.band_set:
            return _keep_angle(coords), None
        z = coords.detach().float().cpu().numpy().astype(np.float64)   # (M,2) measurements
        M = z.shape[0]

        # Reset the filter bank at the band's first layer (self-guard + driver hook).
        if layer_idx == self.k0 or self._kfs is None or len(self._kfs) != M:
            self._kfs = [KalmanFilter() for _ in range(M)]
            for i in range(M):
                self._kfs[i].reset(z[i], self.P0)             # xhat_0 = z_0
            self._prev_layer = None
            self._prev_u = None

        xhat = np.zeros((M, 2))
        if layer_idx == self.k0 and self._prev_layer is None:
            # First band layer: no transition to predict through; estimate is the
            # reset value (xhat_0 = z_0). (We do not double-fuse z_0 here.)
            for i in range(M):
                xhat[i] = self._kfs[i].x
        else:
            # Predict through the previous transition (A_{k-1}, b_{k-1}) with the
            # control u_{k-1} we commanded, then update with this layer's z.
            kp = self._prev_layer
            A_p, b_p = self.A[kp], self.b[kp]
            W_p = self._W_at(kp)
            for i in range(M):
                u_prev = None if self._prev_u is None else self._prev_u[i]
                u_arg = 0 if u_prev is None else u_prev
                x_i, _ = self._kfs[i].step(z[i], A_p, b_p, self.H, W_p, self.V, u=u_arg)
                xhat[i] = x_i

        # LQR on the filtered estimate: s_k = F_k xhat_k + g_k.
        Fk, gk = self.F[layer_idx], self.g[layer_idx]
        s = xhat @ Fk.T + gk                                  # (M,2)
        u = s - xhat                                          # control commanded now
        theta_np = np.arctan2(s[:, 1], s[:, 0])
        theta = torch.from_numpy(np.asarray(theta_np, np.float32)).to(device)

        # Remember what we commanded so the NEXT predict can use it (B_k=A_k).
        self._prev_layer = layer_idx
        self._prev_u = u.copy()
        return theta, u


# =============================================================================
# Offline solve: build the soft-landing gains (OAS_SPEC S3)
# =============================================================================


def build_softlanding(A, b, band, ref, target, R_rho, Q_term,
                      Q_stage=0.0, terminal_layer=None):
    """Build the soft-landing LQR gains via oas_lqr.finite_horizon_lqr.

    Sets the terminal reference r[kT] = target (a point at the target angle with the
    natural magnitude), R = R_rho * I (effort), Q_term (terminal state weight), and
    Q_stage * I (small/zero intermediate weight). Returns the gains dict that
    PolicySoftLandingLQR / PolicyLQG consume.

    Args:
        A: (L-1,2,2); b: (L-1,2) fitted plant (full grid).
        band: actuated layer indices (ascending, contiguous).
        ref: (L,2) per-layer reference; its terminal slot is overwritten with target.
        target: (2,) terminal target coordinate (target angle, natural magnitude).
        R_rho: scalar effort weight rho (R = rho I, rho > 0).
        Q_term: (2,2) OR scalar terminal state weight (scalar -> q*I).
        Q_stage: (2,2) OR scalar intermediate state weight (default 0 -> "don't
            force the angle early"; only the terminal layer must land it).
        terminal_layer: kT; defaults to band[-1].
    """
    A = np.asarray(A, np.float64)
    b = np.asarray(b, np.float64)
    L = A.shape[0] + 1
    r = np.zeros((L, 2)) if ref is None else np.asarray(ref, np.float64).copy()
    band = [int(k) for k in band]
    kT = band[-1] if terminal_layer is None else int(terminal_layer)
    r[kT] = np.asarray(target, np.float64)

    R = float(R_rho) * np.eye(2)
    Qt = Q_term if np.ndim(Q_term) == 2 else float(Q_term) * np.eye(2)
    Qs = Q_stage if np.ndim(Q_stage) == 2 else float(Q_stage) * np.eye(2)
    return finite_horizon_lqr(A, b, band, r, R, Qs, Qt, terminal_layer=kT)


# =============================================================================
# Self-test  (OAS_SPEC.md S3.1) — NO model
# =============================================================================


if __name__ == "__main__":
    import oas_lqr

    torch.manual_seed(0)
    rng = np.random.RandomState(11)

    # ---- (a) _apply_reset is norm-preserving and angle-exact on the plane --------
    d = 16
    b1 = torch.randn(d); b1 = b1 / b1.norm()
    b2 = torch.randn(d); b2 = b2 - (b2 @ b1) * b1; b2 = b2 / b2.norm()
    hidden = torch.randn(2, 3, d)                          # (B,S,d)
    theta_cmd = torch.tensor([[0.3, -1.2, 2.5], [-2.0, 0.1, 1.7]])
    steered = _apply_reset(hidden, b1, b2, theta_cmd)
    p_in = torch.sqrt((hidden @ b1) ** 2 + (hidden @ b2) ** 2)
    p_out = torch.sqrt((steered @ b1) ** 2 + (steered @ b2) ** 2)
    ang_out = torch.atan2(steered @ b2, steered @ b1)
    print(f"[a] _apply_reset: ||proj|| err={float((p_in - p_out).abs().max()):.2e}, "
          f"angle err={float((torch.atan2(torch.sin(ang_out - theta_cmd), torch.cos(ang_out - theta_cmd))).abs().max()):.2e}")
    assert torch.allclose(p_in, p_out, atol=1e-5), "reset must be norm-preserving"
    ang_err = torch.atan2(torch.sin(ang_out - theta_cmd), torch.cos(ang_out - theta_cmd))
    assert float(ang_err.abs().max()) < 1e-4, "reset must realise the commanded angle"
    # the off-plane component is untouched (only the in-plane part is rewritten)
    perp = hidden - (hidden @ b1).unsqueeze(-1) * b1 - (hidden @ b2).unsqueeze(-1) * b2
    perp_s = steered - (steered @ b1).unsqueeze(-1) * b1 - (steered @ b2).unsqueeze(-1) * b2
    assert torch.allclose(perp, perp_s, atol=1e-5), "off-plane component must be untouched"

    # ---- shared synthetic stable affine plant for (b)/(c) ------------------------
    L = 14
    A = np.stack([0.92 * np.array([[np.cos(0.3), -np.sin(0.3)],
                                   [np.sin(0.3), np.cos(0.3)]])
                  + 0.04 * rng.randn(2, 2) for _ in range(L - 1)])
    bb = 0.1 * rng.randn(L - 1, 2)
    band = list(range(3, 10))
    kT = band[-1]
    target = np.array([np.cos(2.0), np.sin(2.0)])
    ref = rng.randn(L, 2) * 0.3
    gains = build_softlanding(A, bb, band, ref, target, R_rho=0.05, Q_term=50.0,
                              Q_stage=0.01)
    # build_softlanding must reproduce a direct finite_horizon_lqr solve.
    r_chk = ref.copy(); r_chk[kT] = target
    gains_ref = finite_horizon_lqr(A, bb, band, r_chk, 0.05 * np.eye(2),
                                   0.01 * np.eye(2), 50.0 * np.eye(2))
    assert all(np.allclose(gains["F"][k], gains_ref["F"][k]) and
               np.allclose(gains["g"][k], gains_ref["g"][k]) for k in band)

    # ---- (b) PolicySoftLandingLQR reproduces oas_lqr.lqr_rollout angles ----------
    # The rollout feeds the natural incoming x_k into s_k = F_k x_k + g_k; the policy
    # does the same when handed the live coordinate as x_k. So, walking the rollout
    # layer by layer and asking the policy for the angle at each x_k must reproduce
    # angle(s_k) from the rollout EXACTLY (the additive-actuator equivalence; the
    # physical angle-only reset introduces the separately-measured Exp-5 tax).
    x0 = rng.randn(2)
    roll = oas_lqr.lqr_rollout(A, bb, gains, x0, band)
    sl_pol = PolicySoftLandingLQR(gains, ref)
    max_ang_err = 0.0
    max_s_err = 0.0
    for i, k in enumerate(band):
        xk = roll["x"][i]                                  # natural incoming coord
        coords = torch.tensor(xk, dtype=torch.float32).reshape(1, 2)
        theta, u = sl_pol(k, coords)
        # policy s_k must equal the rollout s_k; angle must match angle(s_k).
        s_roll = roll["s"][i]
        s_pol = xk + u[0]
        max_s_err = max(max_s_err, float(np.abs(s_pol - s_roll).max()))
        ang_roll = np.arctan2(s_roll[1], s_roll[0])
        ae = abs(np.arctan2(np.sin(float(theta[0]) - ang_roll),
                            np.cos(float(theta[0]) - ang_roll)))
        max_ang_err = max(max_ang_err, ae)
        # control logged correctly: u = s - x
        assert np.allclose(u[0], s_roll - xk, atol=1e-4)
    # layers outside the band keep their angle (read-only).
    coords_out = torch.tensor([[0.4, -0.7]], dtype=torch.float32)
    theta_out, u_out = sl_pol(0, coords_out)
    assert u_out is None and abs(float(theta_out[0]) - np.arctan2(-0.7, 0.4)) < 1e-5
    print(f"[b] PolicySoftLandingLQR vs lqr_rollout: max|s|err={max_s_err:.2e}, "
          f"max angle err={max_ang_err:.2e}")
    assert max_s_err < 1e-4 and max_ang_err < 1e-4

    # ---- (c) PolicyLQG with V -> 0 equals PolicySoftLandingLQR -------------------
    # With vanishing measurement noise the Kalman update trusts the measurement
    # fully, so xhat_k -> z_k = c_k and the LQG policy collapses onto the open-loop
    # soft-landing LQR. We drive BOTH policies through the same forward pass
    # (the live coordinate at each band layer is the rollout's natural x_k) and
    # require the produced angles to agree.
    kcfg = {"A": A, "b": bb, "H": np.eye(2),
            "W": 1e-6 * np.eye(2), "V": 1e-12 * np.eye(2), "P0": 1e-6 * np.eye(2)}
    lqg_pol = PolicyLQG(gains, ref, kcfg)
    lqg_pol.reset()
    # Use a small batch (M=2 tokens) to exercise the per-token filter bank.
    x0a, x0b = rng.randn(2), rng.randn(2)
    roll_a = oas_lqr.lqr_rollout(A, bb, gains, x0a, band)
    roll_b = oas_lqr.lqr_rollout(A, bb, gains, x0b, band)
    sl_pol2 = PolicySoftLandingLQR(gains, ref)
    max_lqg_err = 0.0
    for i, k in enumerate(band):
        coords = torch.tensor(np.stack([roll_a["x"][i], roll_b["x"][i]]),
                              dtype=torch.float32)               # (2,2)
        theta_lqg, _ = lqg_pol(k, coords)
        theta_sl, _ = sl_pol2(k, coords)
        de = torch.atan2(torch.sin(theta_lqg - theta_sl), torch.cos(theta_lqg - theta_sl))
        max_lqg_err = max(max_lqg_err, float(de.abs().max()))
    print(f"[c] PolicyLQG(V->0) vs PolicySoftLandingLQR: max angle err={max_lqg_err:.2e}")
    assert max_lqg_err < 1e-4, "LQG with V->0 must equal the open-loop soft-landing LQR"

    # reset() drops the per-pass filter bank so a fresh pass starts clean.
    lqg_pol.reset()
    assert lqg_pol._kfs is None and lqg_pol._prev_u is None

    # ---- deadbeat / multi-angle / none sanity -----------------------------------
    db = PolicyDeadbeat(target_angle=1.0, layer=5)
    coords = torch.randn(4, 2)
    th_on, _ = db(5, coords); th_off, _ = db(4, coords)
    assert torch.allclose(th_on, torch.full((4,), 1.0)), "deadbeat must slam its layer"
    assert torch.allclose(th_off, torch.atan2(coords[:, 1], coords[:, 0])), \
        "deadbeat must keep angle off its layer"
    ma = PolicyMultiAngle(target_angle=-0.5, layers=[5, 6, 7])
    th6, _ = ma(6, coords); th8, _ = ma(8, coords)
    assert torch.allclose(th6, torch.full((4,), -0.5)) and \
        torch.allclose(th8, torch.atan2(coords[:, 1], coords[:, 0]))
    none_pol = PolicyNone()
    thn, un = none_pol(3, coords)
    assert un is None and torch.allclose(thn, torch.atan2(coords[:, 1], coords[:, 0]))

    # ---- hook wiring sanity (no model): attach via a fake module_dict ------------
    class _FakeLayer(torch.nn.Module):
        def forward(self, x):
            return (x,)                                    # tuple-style HF output

    mods = {f"model.layers.{k}": _FakeLayer() for k in band}
    state = OASState(PolicySoftLandingLQR(gains, ref), record=True)
    pairs = attach_oas_hooks(mods, band, b1, b2, state)
    assert len(pairs) == len(band)
    # fire one hook manually and confirm it reshapes / records / steers a tuple out.
    hk = make_oas_layer_hook(band[0], b1, b2, state)
    h_in = torch.randn(2, 3, d)
    out = hk(mods[f"model.layers.{band[0]}"], (h_in,), (h_in,))
    assert isinstance(out, tuple) and out[0].shape == h_in.shape
    assert band[0] in state.log and len(state.log[band[0]]["theta"]) == 1
    # the recorded angle equals the realised in-plane angle of the steered output.
    realised = torch.atan2(out[0] @ b2, out[0] @ b1).reshape(-1).detach().numpy()
    assert np.allclose(realised, state.log[band[0]]["theta"][0], atol=1e-4)
    state.reset_policy()                                   # smoke the driver hook

    print("[oas_controller self-test] OK")
