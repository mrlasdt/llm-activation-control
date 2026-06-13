"""CASA distributed control — the k×k plant + bounded-u MPC over the cone band.

CASA_PROPOSAL.md §3 "Distributed (MPC) variant" / §4 Experiment 3. This is the
first place PTS's lookahead/constraint apparatus meets a NON-norm-preserving
(additive) actuator it can actually shape.

Blunt CASA ablation removes the *full* cone coordinate at *every* band layer. The
distributed variant instead asks: what is the *minimal-perturbation* schedule of
additive pushes `u_j` across the band that drives the LATE-layer cone coordinate
to a target (≈ the harmless-mean coordinate, i.e. "no refusal") — spreading the
removal so total ‖u‖ (the coherence tax proxy) is minimized?

It generalizes PTS's validated 2×2 affine plant `c_{k+1} ≈ A_k c_k + b_k`
(held-out R²≈0.999) to the **k-dim cone coordinate** `c = B h ∈ ℝ^k`, and the
PTS condensed-QP / FISTA bounded-`u` MPC to n=k dimensions. Two differences from
PTS that matter:

  1. Actuator is **additive** (B_ctrl = I): the control `u_j` is applied directly
     to the residual stream as `h += Bᵀ u_j`. There is NO angle conversion — the
     PTS `angle_from_control` collapse (which reduced 2D control to one realized
     DoF and made PTS inert) does not happen here. This is exactly the lever the
     PTS verdict named as "untested".
  2. The state is the **refusal-specific cone coordinate** (`casa_cone`), not the
     noisy AS plane.

Reuses the dimension-agnostic `assemble_q` and `dlqr` from `pts_mpc`; provides
k-generalized `build_condensed_qp` / `solve_qp` (PTS hardcodes n=2).

Self-test (synthetic, no model):  python casa_control.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "PTS"))            # pts_mpc
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))   # shared lib

from pts_mpc import assemble_q, dlqr, qp_objective            # noqa: E402  (dim-agnostic)


# =============================================================================
# k-dim affine plant  c_{k+1} = A_k c_k + b_k   (generalizes PTS's 2×2)
# =============================================================================


def fit_cone_plant(coords, affine=True, ridge=1e-3):
    """Fit per-layer affine dynamics of the k-dim cone coordinate.

    coords: (N, L, k) cone-coordinate trajectories (N prompts, L layers, k dims).
    Returns dict A:(L-1,k,k), b:(L-1,k). Least squares per layer transition
    c_{l+1} ≈ A_l c_l (+ b_l), ridge-regularized."""
    coords = np.asarray(coords, np.float64)
    N, L, k = coords.shape
    A = np.zeros((L - 1, k, k)); b = np.zeros((L - 1, k))
    for l in range(L - 1):
        X = coords[:, l, :]                                   # (N,k)
        Y = coords[:, l + 1, :]                               # (N,k)
        if affine:
            X = np.concatenate([X, np.ones((N, 1))], 1)       # (N,k+1)
        G = X.T @ X + ridge * np.eye(X.shape[1])
        W = np.linalg.solve(G, X.T @ Y)                       # (k+1,k) or (k,k)
        if affine:
            A[l] = W[:k].T; b[l] = W[k]
        else:
            A[l] = W.T
    return {"A": A, "b": b, "affine": affine}


def rollout(A, b, c0, controls=None, start=0):
    """Actuated rollout under the additive convention (B_ctrl=I):
        s_0 = c0 + u_0 ;  s_j = A_{start+j-1} s_{j-1} + b_{start+j-1} (+ u_j if j<H).
    c0:(...,k); controls:(...,H,k) or None; returns (...,H+1,k) actuated states
    (or autonomous (...,1,k) if controls is None)."""
    A = np.asarray(A); b = np.asarray(b); c0 = np.asarray(c0, np.float64)
    if controls is None:
        return c0[..., None, :]
    H = controls.shape[-2]
    s = c0 + controls[..., 0, :]
    states = [s]
    for j in range(1, H + 1):
        s = s @ A[start + j - 1].T + b[start + j - 1]
        if j < H:
            s = s + controls[..., j, :]
        states.append(s)
    return np.stack(states, axis=-2)                          # (...,H+1,k)


def plant_r2(coords, fit, horizon=1):
    """Held-out-style 1- or H-step R² of the fitted plant (reported like PTS)."""
    coords = np.asarray(coords, np.float64)
    N, L, k = coords.shape
    A, b = fit["A"], fit["b"]
    errs, tots = 0.0, 0.0
    mu = coords.reshape(-1, k).mean(0)
    for l in range(L - 1 - horizon + 1):
        s = coords[:, l, :]
        for h in range(horizon):
            s = s @ A[l + h].T + b[l + h]
        tgt = coords[:, l + horizon, :]
        errs += ((s - tgt) ** 2).sum()
        tots += ((tgt - mu) ** 2).sum()
    return 1.0 - errs / (tots + 1e-12)


# =============================================================================
# k-dim condensed QP + FISTA  (PTS's, generalized from n=2 to n=k)
# =============================================================================


def build_condensed_qp(A, b, k, H, Q, R, Qf, n):
    """Condense the layer-horizon OCP into a dense QP for an n-dim state/control.
    Identical to pts_mpc.build_condensed_qp but with n parameterized (PTS hardcodes
    n=2). See that docstring for the derivation."""
    Hs = H + 1
    M = np.zeros((Hs, Hs, n, n))
    for j in range(Hs):
        M[j, j] = np.eye(n)
        for i in range(j - 1, -1, -1):
            M[j, i] = A[k + j - 1] @ M[j - 1, i]
    G = np.zeros((n * Hs, n * H)); F = np.zeros((n * Hs, n)); phi = np.zeros(n * Hs)
    for j in range(Hs):
        F[n * j:n * j + n] = M[j, 0]
        acc = np.zeros(n)
        for i in range(j):
            acc = acc + M[j, i + 1] @ b[k + i]
        phi[n * j:n * j + n] = acc
        for i in range(min(j, H - 1) + 1):
            G[n * j:n * j + n, n * i:n * i + n] = M[j, i]
    Qblk = np.zeros((n * Hs, n * Hs))
    for j in range(H):
        Qblk[n * j:n * j + n, n * j:n * j + n] = Q
    Qblk[n * H:n * H + n, n * H:n * H + n] = Qf
    Rblk = np.kron(np.eye(H), R)
    P = 2.0 * (G.T @ Qblk @ G + Rblk); P = 0.5 * (P + P.T)
    L_lip = float(np.linalg.eigvalsh(P)[-1])
    return {"P": P, "G": G, "F": F, "phi": phi, "Qblk": Qblk, "Rblk": Rblk,
            "H": H, "n": n, "L": L_lip}


def _project_ball(U, u_max, H, n):
    Ur = U.reshape(U.shape[:-1] + (H, n))
    nrm = np.linalg.norm(Ur, axis=-1, keepdims=True)
    scale = np.minimum(1.0, u_max / (nrm + 1e-12))
    return (Ur * scale).reshape(U.shape)


def solve_qp(P, q, u_max, H, n, L=None, iters=80, tol=1e-9):
    """Minimise ½UᵀPU + qᵀU s.t. ‖u_j‖₂ ≤ u_max per step. n-dim generalization of
    pts_mpc.solve_qp (closed-form unconstrained + FISTA constrained)."""
    q = np.asarray(q, np.float64); lead = q.shape[:-1]
    q2 = q.reshape(-1, q.shape[-1])
    U_unc = np.linalg.solve(P, -q2.T).T
    if u_max is None or not np.isfinite(u_max):
        return U_unc.reshape(lead + (n * H,))
    if L is None:
        L = float(np.linalg.eigvalsh(P)[-1])
    step = 1.0 / L
    U = _project_ball(U_unc, u_max, H, n); Y = U.copy(); t = 1.0
    for _ in range(iters):
        grad = Y @ P + q2
        U_new = _project_ball(Y - step * grad, u_max, H, n)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = U_new + ((t - 1.0) / t_new) * (U_new - U)
        if np.max(np.abs(U_new - U)) < tol:
            U = U_new; break
        U, t = U_new, t_new
    return U.reshape(lead + (n * H,))


# =============================================================================
# Additive cone MPC controller  (NO angle conversion — the key difference)
# =============================================================================


class ConeMPC:
    """Receding-horizon MPC over the k-dim cone coordinate with an ADDITIVE
    actuator. `.control(layer, xi)` returns the first additive control u_0:(...,k)
    that the actuator adds (via Bᵀ u_0) to the residual stream — no angle step."""

    def __init__(self, A, b, ref, layers, H=6, q_pos=1.0, r_ctrl=0.05,
                 qf_scale=4.0, u_max=None, fista_iters=60):
        self.A = np.asarray(A, np.float64); self.b = np.asarray(b, np.float64)
        self.ref = np.asarray(ref, np.float64)                # (L,k)
        self.n = self.A.shape[1]
        self.layers = list(layers)
        self.L = min(self.A.shape[0] + 1, self.ref.shape[0])
        self.H = H; self.u_max = u_max; self.fista_iters = fista_iters
        self.Q = q_pos * np.eye(self.n)
        self.R = r_ctrl * np.eye(self.n)
        self.Qf = qf_scale * q_pos * np.eye(self.n)
        self._cqp = {}; self._refstack = {}
        for kk in self.layers:
            Hk = min(H, self.L - 1 - kk)
            if Hk < 1:
                continue
            self._cqp[kk] = build_condensed_qp(self.A, self.b, kk, Hk,
                                               self.Q, self.R, self.Qf, self.n)
            self._refstack[kk] = self.ref[kk:kk + Hk + 1]
        if not self._cqp:
            raise ValueError("no actuated layer has a positive horizon")

    def reset(self):
        """No-op: the MPC is stateless (re-solves each call). Present so it satisfies the
        same controller contract as the stateful ConePID for make_controller_hooks."""
        pass

    def control(self, k, xi):
        """Additive first control u_0:(...,k) for measured cone coordinate xi:(...,k)."""
        if k not in self._cqp:
            return np.zeros_like(np.asarray(xi, np.float64))
        cqp = self._cqp[k]
        q = assemble_q(cqp, xi, self._refstack[k])
        U = solve_qp(cqp["P"], q, self.u_max, cqp["H"], self.n,
                     L=cqp["L"], iters=self.fista_iters)
        return U[..., :self.n]


# =============================================================================
# Distributed-ablation hook factory (model-in-the-loop, additive cone actuator)
# =============================================================================


def make_controller_hooks(module_dict, band, B, controller):
    """Controller-agnostic forward hooks: at each band layer, measure the cone coordinate
    c = B h, ask `controller.control(layer, xi)` for the additive control u0, apply h += u0 @ B.

    `controller` is ANY object exposing `.control(layer, xi:(M,k)) -> u0:(M,k)` (and optionally
    `.reset()`). This is the single deployment surface shared by ConeMPC, ConeLQR, ConePID and
    the RecordingController wrapper, so the head-to-head varies only the control law.

    B:(k,d) orthonormal cone basis (torch); applied across `band`. Stateful controllers (PID)
    reset themselves when they see the first band layer (the forward pass enters the band there)."""
    layerset = set(band)
    if hasattr(controller, "reset"):
        controller.reset()

    def mk(layer):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if layer not in layerset:
                return out
            Bd = B.to(h.device, h.dtype)
            c = (h @ Bd.t())                                  # (B,S,k)
            shp = c.shape
            xi = c.reshape(-1, shp[-1]).float().cpu().numpy()
            u0 = controller.control(layer, xi)                # (M,k)
            u = torch.from_numpy(np.ascontiguousarray(u0)).to(h.device, h.dtype)
            u = u.reshape(shp)
            steered = h + u @ Bd
            return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered
        return hook

    return [(module_dict[f"model.layers.{j}"], mk(j)) for j in band]


# back-compat alias (ConeMPC is the original consumer)
def make_distributed_hooks(module_dict, band, B, mpc: ConeMPC):
    """Deprecated name for make_controller_hooks (kept so existing callers keep working)."""
    return make_controller_hooks(module_dict, band, B, mpc)


# =============================================================================
# Self-test (synthetic — no model)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.RandomState(0)
    for n in (1, 3, 4):                                       # k-dim generalization
        # ---- 1. plant fit recovers a known affine system at high R² ----
        L, N = 12, 200
        A_true = np.stack([0.9 * np.eye(n) + 0.05 * rng.randn(n, n) for _ in range(L - 1)])
        b_true = 0.1 * rng.randn(L - 1, n)
        c = np.zeros((N, L, n)); c[:, 0, :] = rng.randn(N, n)
        for l in range(L - 1):
            c[:, l + 1, :] = c[:, l, :] @ A_true[l].T + b_true[l] + 0.01 * rng.randn(N, n)
        fit = fit_cone_plant(c)
        r2 = plant_r2(c, fit, horizon=1)
        assert r2 > 0.99, f"n={n} plant R²={r2:.4f} too low"

        # ---- 2. condensation: P,q reproduce simulated cost (ground truth) ----
        Q, R, Qf = np.eye(n), 0.03 * np.eye(n), 3.0 * np.eye(n)
        kL, H = 2, 5
        cqp = build_condensed_qp(A_true, b_true, kL, H, Q, R, Qf, n)
        xi = rng.randn(n); ref = rng.randn(H + 1, n)
        q = assemble_q(cqp, xi, ref)
        def sim_cost(U):
            s = xi + U[:n]; J = 0.0
            for j in range(H + 1):
                Qj = Qf if j == H else Q; e = s - ref[j]; J += e @ Qj @ e
                if j < H:
                    J += U[n * j:n * j + n] @ R @ U[n * j:n * j + n]
                    s = A_true[kL + j] @ s + b_true[kL + j]
                    if j + 1 <= H - 1:
                        s = s + U[n * (j + 1):n * (j + 1) + n]
            return J
        const = (xi @ cqp["F"].T + cqp["phi"] - ref.reshape(-1)) @ cqp["Qblk"] @ \
                (xi @ cqp["F"].T + cqp["phi"] - ref.reshape(-1))
        rel = max(abs(qp_objective(cqp["P"], q, U := rng.randn(n * H)) + const - sim_cost(U))
                  / (abs(sim_cost(U)) + 1e-9) for _ in range(50))
        assert rel < 1e-8, f"n={n} condensation rel err {rel:.2e}"

        # ---- 3. unconstrained optimal (grad≈0) + constrained feasible & better ----
        U_unc = solve_qp(cqp["P"], q, None, H, n)
        assert np.linalg.norm(cqp["P"] @ U_unc + q) < 1e-7
        u_max = 0.25
        U_con = solve_qp(cqp["P"], q, u_max, H, n, iters=400)
        feas = np.linalg.norm(U_con.reshape(H, n), axis=-1).max()
        assert feas <= u_max + 1e-6
        assert qp_objective(cqp["P"], q, U_con) <= \
               qp_objective(cqp["P"], q, _project_ball(U_unc, u_max, H, n)) + 1e-6

        # ---- 4. ADDITIVE closed loop drives the cone coordinate to ref=0 ----
        nstep = 18
        A_cl = np.stack([0.9 * np.eye(n) + 0.05 * rng.randn(n, n)] * (nstep + 2))
        b_cl = np.zeros((nstep + 2, n)); refz = np.zeros((nstep + 2, n))
        mpc = ConeMPC(A_cl, b_cl, refz, layers=list(range(nstep)), H=6, u_max=None)
        s = rng.randn(n) * 2.0; n0 = np.linalg.norm(s)
        for kk in range(nstep):
            u0 = mpc.control(kk, s)
            s = s + u0                                        # additive actuation
            s = A_cl[kk] @ s + b_cl[kk]                       # true dynamics
        assert np.linalg.norm(s) < 0.05 * n0, f"n={n} closed loop did not contract"
        print(f"[casa_control n={n}] OK  plantR²={r2:.4f}  condRel={rel:.1e}  "
              f"closed-loop ‖c‖ {n0:.2f}->{np.linalg.norm(s):.2e}")

    print("[casa_control self-test] ALL OK — k-dim plant fit, condensation, "
          "bounded-u FISTA, additive closed-loop contraction")
