"""PTS — forward-pass-coupled multi-layer steering for Predictive Trajectory Steering.

This is the piece that makes PTS a *closed-in-depth* controller rather than a
fixed-angle open-loop one (research_proposal_PTS.md S5.2, S5.4): a hook on every
actuated layer reads the in-plane coordinate c_k of the live residual stream,
asks a policy for the target angle theta_k, and applies the norm-preserving SO(2)
reset (the verified Angular-Steering actuator, identical to clas_controller.make_clas_hook).

Every controller is routed through the SAME actuator and differs only in how
theta_k is produced:
  * PolicyMPC        — receding-horizon 2D MPC (pts_mpc.MPCController)        [PTS]
  * PolicyLQR        — projected-to-2D unconstrained LQR (the A-LQR analogue)
  * PolicyFixedAngle — one global angle at the actuated layers (Angular Steering)
  * PolicyNone       — read-only (records coordinates, applies no steering)

Each policy maps the measured coordinates c:(M,2) at a layer to angles theta:(M,)
in radians, where M = batch * positions flattened. The MPC/LQR policies run on the
CPU in numpy (the QP is 2H~12 vars; tiny), then theta is pushed back to the device.
"""

from __future__ import annotations

import numpy as np
import torch

from pts_mpc import MPCController, angle_from_control


# =============================================================================
# Shared state + actuator
# =============================================================================


class PTSState:
    """Mutable steering state shared by all layer hooks for one decode/prefill.

    `record` toggles logging of the per-layer measured coordinate, applied angle,
    and (for MPC/LQR) the additive control, into `.log` keyed by layer index — used
    to measure planned-vs-realised tracking error.
    """

    def __init__(self, policy, record: bool = False):
        self.policy = policy
        self.enabled = True
        self.record = record
        self.log: dict[int, dict] = {}

    def reset_log(self):
        self.log = {}


def _apply_reset(hidden, b1, b2, theta):
    """Norm-preserving SO(2) reset of the in-plane component to angle `theta`.

    hidden: (B,S,d); b1,b2: (d,); theta: (B,S) radians. Returns steered (B,S,d).
    Identical semantics to clas_controller.make_clas_hook / Angular Steering.
    """
    c1 = hidden @ b1                                  # (B,S)
    c2 = hidden @ b2
    proj = c1.unsqueeze(-1) * b1 + c2.unsqueeze(-1) * b2
    scale = torch.sqrt(c1 ** 2 + c2 ** 2).unsqueeze(-1)
    ct = torch.cos(theta).to(hidden.dtype).unsqueeze(-1)   # (B,S,1)
    st = torch.sin(theta).to(hidden.dtype).unsqueeze(-1)
    steer = ct * b1 + st * b2                         # (B,S,d)
    return hidden - proj + scale * steer


def make_pts_layer_hook(layer_idx: int, b1: torch.Tensor, b2: torch.Tensor,
                        state: PTSState):
    """Forward hook for one actuated layer; tuple-aware (HF decoder layers)."""

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


def attach_pts_hooks(module_dict, layers, b1, b2, state: PTSState):
    """Return [(module, hook), ...] for utils.add_hooks over the actuated layers."""
    return [
        (module_dict[f"model.layers.{k}"], make_pts_layer_hook(k, b1, b2, state))
        for k in layers
    ]


# =============================================================================
# Policies  (coords c:(M,2) torch on device -> theta:(M,) torch, [control:(M,2)])
# =============================================================================


class PolicyNone:
    """Read-only: keep the current angle (no behavioural change)."""

    def __call__(self, layer_idx, coords):
        theta = torch.atan2(coords[:, 1], coords[:, 0])
        return theta, None


class PolicyFixedAngle:
    """Angular Steering baseline: one global angle at every actuated layer."""

    def __init__(self, theta_rad: float, layers):
        self.theta = float(theta_rad)
        self.layers = set(layers)

    def __call__(self, layer_idx, coords):
        if layer_idx in self.layers:
            theta = torch.full((coords.shape[0],), self.theta,
                               device=coords.device, dtype=torch.float32)
        else:
            theta = torch.atan2(coords[:, 1], coords[:, 0])
        return theta, None


class PolicyMPC:
    """PTS: receding-horizon 2D MPC produces the per-layer adaptive angle."""

    def __init__(self, mpc: MPCController):
        self.mpc = mpc

    def __call__(self, layer_idx, coords):
        device = coords.device
        xi = coords.detach().float().cpu().numpy()            # (M,2)
        u0, theta_np = self.mpc.control(layer_idx, xi)        # (M,2),(M,)
        theta = torch.from_numpy(np.asarray(theta_np, np.float32)).to(device)
        return theta, u0


class PolicyLQR:
    """Unconstrained projected-to-2D LQR (the A-LQR analogue) — exactly the
    unconstrained limit of PTS-MPC. The unconstrained MPC first control is affine
    in the state, u0(c) = c0_k - K_k (c - r_k), where K_k is the feedback gain AND
    c0_k = u0(r_k) is the control the MPC still commands when the state already sits
    on the reference (nonzero whenever the dynamics aren't identity or future
    references differ). theta = angle(c + u)."""

    def __init__(self, gains: dict, ref: np.ndarray, layers):
        self.K = gains                       # {layer: {"K":(2,2), "c0":(2,)}}
        self.ref = np.asarray(ref, np.float64)
        self.layers = set(layers)

    def __call__(self, layer_idx, coords):
        device = coords.device
        if layer_idx not in self.layers or layer_idx not in self.K:
            theta = torch.atan2(coords[:, 1], coords[:, 0])
            return theta, None
        xi = coords.detach().float().cpu().numpy()            # (M,2)
        g = self.K[layer_idx]
        u = g["c0"] - (xi - self.ref[layer_idx]) @ g["K"].T    # (M,2)
        theta_np = angle_from_control(xi, u)
        theta = torch.from_numpy(np.asarray(theta_np, np.float32)).to(device)
        return theta, u


def lqr_gain_schedule(A, b, ref, layers, q_pos=1.0, r_ctrl=0.05, qf_scale=4.0,
                      horizon=8):
    """Per-layer unconstrained MPC first control, expressed as gain + affine offset,
    in the PTS actuator convention (control added to the actuated state, B=I).
    Returns {layer: {"K": (2,2), "c0": (2,)}} where u0(xi) = c0 - K (xi - ref_k).

    This IS the unconstrained limit of PTS-MPC (the "A-LQR projected to 2D"
    baseline): since u0(xi) = -P^{-1} q(xi) is affine in xi, K = -du0/dxi (exact via
    central difference) and c0 = u0(ref_k). Dropping c0 — as a naive -K(xi-ref) LQR
    would — is wrong whenever A_k != I or future references differ from where the
    autonomous dynamics carry ref_k.
    """
    from pts_mpc import build_condensed_qp, assemble_q, solve_qp
    A = np.asarray(A, np.float64)
    Q = q_pos * np.eye(2)
    R = r_ctrl * np.eye(2)
    Qf = qf_scale * q_pos * np.eye(2)
    L = A.shape[0] + 1
    gains = {}
    for k in layers:
        Hk = min(horizon, L - 1 - k)
        if Hk < 1:
            continue
        cqp = build_condensed_qp(A, b, k, Hk, Q, R, Qf)
        r_slice = ref[k:k + Hk + 1]
        base = ref[k]
        c0 = solve_qp(cqp["P"], assemble_q(cqp, base, r_slice), None, Hk)[:2]  # u0(ref_k)
        cols = []
        for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0])):
            up = solve_qp(cqp["P"], assemble_q(cqp, base + e, r_slice), None, Hk)[:2]
            um = solve_qp(cqp["P"], assemble_q(cqp, base - e, r_slice), None, Hk)[:2]
            cols.append(-(up - um) / 2.0)        # -du0/dxi column = K column
        gains[k] = {"K": np.stack(cols, axis=1), "c0": c0}   # (2,2),(2,)
    return gains


if __name__ == "__main__":
    # Smoke test the policies on a tiny synthetic plane (no model needed).
    torch.manual_seed(0)
    d = 16
    b1 = torch.randn(d); b1 = b1 / b1.norm()
    b2 = torch.randn(d); b2 = b2 - (b2 @ b1) * b1; b2 = b2 / b2.norm()
    L = 10
    A = np.stack([0.9 * np.eye(2) for _ in range(L - 1)])
    bb = np.zeros((L - 1, 2))
    ref = np.tile(np.array([1.0, 0.0]), (L, 1))
    mpc = MPCController(A, bb, ref, layers=list(range(2, 8)), H=4, u_max=0.5)

    state = PTSState(PolicyMPC(mpc), record=True)
    hidden = torch.randn(2, 3, d)                # (B,S,d)
    c1 = hidden @ b1; c2 = hidden @ b2
    coords = torch.stack([c1, c2], -1).reshape(-1, 2)
    theta, u = state.policy(4, coords)
    assert theta.shape == (6,) and u.shape == (6, 2)
    steered = _apply_reset(hidden, b1, b2, theta.reshape(2, 3))
    # reset is norm-preserving on the plane: ||proj(steered)|| == ||proj(hidden)||
    p_in = torch.sqrt((hidden @ b1) ** 2 + (hidden @ b2) ** 2)
    p_out = torch.sqrt((steered @ b1) ** 2 + (steered @ b2) ** 2)
    assert torch.allclose(p_in, p_out, atol=1e-5)
    # and the realised angle matches the commanded angle
    ang_out = torch.atan2(steered @ b2, steered @ b1).reshape(-1)
    assert torch.allclose(ang_out, theta, atol=1e-4)

    # the LQR baseline must EXACTLY reproduce the unconstrained MPC first control
    gains = lqr_gain_schedule(A, bb, ref, layers=list(range(2, 8)), horizon=4)
    assert all(g["K"].shape == (2, 2) and g["c0"].shape == (2,) for g in gains.values())
    mpc_unc = MPCController(A, bb, ref, layers=list(range(2, 8)), H=4, u_max=None)
    lqr_pol = PolicyLQR(gains, ref, list(range(2, 8)))
    max_err = 0.0
    for k in range(2, 8):
        xi = torch.tensor([[0.3, -0.4], [1.1, 0.2]])
        u_mpc, _ = mpc_unc.control(k, xi.numpy())
        _, u_lqr = lqr_pol(k, xi)
        max_err = max(max_err, float(np.abs(u_mpc - u_lqr).max()))
    assert max_err < 1e-6, f"LQR baseline != unconstrained MPC (err {max_err})"
    print(f"[pts_controller self-test] OK  (norm-preserving, angle-exact, "
          f"LQR==unconstrained-MPC to {max_err:.1e})")
