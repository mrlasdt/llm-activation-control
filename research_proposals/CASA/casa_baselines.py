"""CASA control-law baselines — P / PID / LQR on the SAME cone plant as the MPC.

The fair head-to-head (CASA_PROPOSAL §4 / BASELINES). These controllers all expose the
**same contract as `casa_control.ConeMPC`** — `.control(layer, xi:(...,k)) -> u0:(...,k)`
and `.reset()` — so they deploy through the (controller-agnostic) hook factory unchanged.
The ONLY thing that varies between them and the MPC is the control LAW: same plant `{A,b}`,
same reference `ref` (the harmless-mean cone coordinate per layer), same band, same per-token
budget `u_max`. This isolates the law as the sole explanatory variable, which is exactly the
P → PID → LQR → MPC ladder the user's instinct ("predictive control always beats PID/LQR")
asks us to test.

The three laws cope with the affine drift toward the setpoint differently — the crux:
  * **P**   u = Kp·(ref−c)                         — proportional only ⇒ steady-state offset.
  * **PID** u = Kp·e + Ki·Σe + Kd·Δe (e=ref−c)     — integral kills the offset (model-free).
  * **LQR** u = −K(c−ref) [+ feedforward]          — optimal multivariable feedback; with the
            K,S from the DARE on each layer's A. WITHOUT feedforward it (like P) leaves a
            steady-state offset; WITH the model-based feedforward g it cancels the known drift
            — the same information the MPC plans over.
  * **MPC** (casa_control.ConeMPC) — receding-horizon constrained QP; the only law that can
            enforce ‖u‖≤u_max exactly and plan the drift over a finite horizon.

Control-theory note this experiment tests: for an UNCONSTRAINED quadratic-cost LTI problem the
infinite-horizon LQR is *optimal*, and finite-horizon MPC only approximates it — so MPC has a
strict edge over LQR only when the ‖u‖ constraint binds, the horizon/terminal cost matters, or
the plant is nonlinear. The self-test below demonstrates the unconstrained-MPC≈LQR equivalence
directly (`_selftest_mpc_equals_lqr`).

Self-test (synthetic, no model):  python casa_baselines.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "PTS"))            # pts_mpc.dlqr
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))   # shared lib

from pts_mpc import dlqr                                       # noqa: E402  (dim-agnostic DARE)


# =============================================================================
# helpers
# =============================================================================


def _clamp_ball(u, u_max):
    """Bound each row's L2 norm to u_max (the SAME per-token budget the MPC enforces
    internally via FISTA ball projection). u:(...,k)."""
    if u_max is None or not np.isfinite(u_max):
        return u
    nrm = np.linalg.norm(u, axis=-1, keepdims=True)
    return u * np.minimum(1.0, u_max / (nrm + 1e-12))


def _dare_cross(A, B, Q, N, R, iters=4000, tol=1e-12):
    """Discrete LQR with a cross term: cost Σ xᵀQx + 2xᵀNu + uᵀRu, x⁺=Ax+Bu.
    Returns (K, S) with u=-Kx. Iterated Riccati. (k is tiny, so this is cheap.)"""
    A = np.asarray(A, float); B = np.asarray(B, float)
    Q = np.asarray(Q, float); N = np.asarray(N, float); R = np.asarray(R, float)
    S = Q.copy()
    K = np.zeros((B.shape[1], A.shape[0]))
    for _ in range(iters):
        Gm = R + B.T @ S @ B
        Hm = B.T @ S @ A + N.T
        K = np.linalg.solve(Gm, Hm)
        S_new = Q + A.T @ S @ A - Hm.T @ np.linalg.solve(Gm, Hm)
        S_new = 0.5 * (S_new + S_new.T)
        if np.max(np.abs(S_new - S)) < tol:
            S = S_new; break
        S = S_new
    return K, S


# =============================================================================
# PID  (and P as the Ki=Kd=0 special case)  — over the layer "time" axis
# =============================================================================


class ConePID:
    """Discrete PID over the layer axis, tracking the cone reference.

    e_l = ref_l − c_l;  u_l = Kp·e_l + Ki·Σ_{j≤l} e_j + Kd·(e_l − e_{l-1}),  clamped to u_max.
    This is the in-loop, per-(batch×position) analog of PID-AcT / S-PID (both track a
    diff-in-means / LFS setpoint error with a PID law over depth). The integral accumulator
    optionally resets every `i_reset` layers (anti-windup; S-PID resets every 10 layers,
    PIDsteering.py:174). State is allocated lazily to match the incoming batch and reset per
    forward pass via `.reset()` (called at the first band layer by the hook factory).
    """

    def __init__(self, ref, layers, kp=1.0, ki=0.1, kd=0.01, u_max=None, i_reset=0):
        self.ref = np.asarray(ref, np.float64)                # (L,k)
        self.layers = list(layers)
        self.first = self.layers[0]
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd)
        self.u_max = u_max
        self.i_reset = int(i_reset)
        self.n = self.ref.shape[1]
        self._isum = None; self._eprev = None; self._step = 0

    def reset(self):
        self._isum = None; self._eprev = None; self._step = 0

    def control(self, k, xi):
        xi = np.asarray(xi, np.float64)                       # (M,k)
        if k == self.first:                                   # new forward pass through the band
            self.reset()
        e = self.ref[k] - xi                                  # (M,k) tracking error
        if self._isum is None:
            self._isum = np.zeros_like(e); self._eprev = np.zeros_like(e); self._step = 0
        if self.i_reset and self._step > 0 and self._step % self.i_reset == 0:
            self._isum = np.zeros_like(e)                     # periodic anti-windup reset
        self._isum = self._isum + e
        d = e - self._eprev
        u = self.kp * e + self.ki * self._isum + self.kd * d
        self._eprev = e; self._step += 1
        return _clamp_ball(u, self.u_max)


def ConeP(ref, layers, kp=1.0, u_max=None):
    """Pure proportional controller — the bottom of the ladder (= a tracking version of the
    blunt additive ablation). u_l = Kp·(ref_l − c_l), clamped."""
    return ConePID(ref, layers, kp=kp, ki=0.0, kd=0.0, u_max=u_max)


# =============================================================================
# LQR  (per-layer DARE gains + optional model-based feedforward) — fixed-gain
# =============================================================================


class ConeLQR:
    """Infinite-horizon LQR per layer for the EXACT objective the MPC optimizes, tracking the
    cone reference. Gains are PRECOMPUTED once (the classical-LQR distinction from MPC, which
    re-optimizes online with the hard constraint).

    Plant (matches casa_control.ConeMPC / rollout): the control is added to the residual-stream
    OUTPUT of a layer, so the actuated state s_l = c_l + u_l is what the cost penalises and what
    propagates: c_{l+1} = A_l (c_l + u_l) + b_l. In error coords e_l=c_l−ref_l this is an LQR
    with control matrix B=A_l and a cross term (cost ‖c_l+u_l−ref_l‖²_Q + ‖u_l‖²_R), i.e.
    Q̃=Q, Ñ=Q, R̃=Q+R, Ã=B̃=A_l. The gain is the gain-scheduled cross-term DARE on each A_l.

    control:  e_l = c_l − ref_l;  u_l = −K_l e_l + g_l,  clamped to u_max.
      * feedforward=False — regulation only (the honest A-LQR analog): like P/LQR it leaves a
        steady-state offset against the affine drift.
      * feedforward=True  — inverse-dynamics feedforward g_l = A_l⁻¹(ref_{l+1} − b_l) − ref_l
        cancels the known drift (the same model info the MPC plans over), so the closed loop
        tracks ref with zero steady-state error.
    """

    def __init__(self, A, b, ref, layers, q_pos=1.0, r_ctrl=0.05, u_max=None,
                 feedforward=True, ridge=1e-9):
        self.A = np.asarray(A, np.float64); self.b = np.asarray(b, np.float64)
        self.ref = np.asarray(ref, np.float64)
        self.layers = list(layers)
        self.first = self.layers[0]
        self.n = self.A.shape[1]
        self.u_max = u_max
        self.feedforward = feedforward
        Q = q_pos * np.eye(self.n); R = r_ctrl * np.eye(self.n)
        self.K = {}; self.g = {}; self.rho = {}
        Lmax = min(self.A.shape[0], self.ref.shape[0] - 1)
        for l in self.layers:
            if l >= Lmax:                                     # no transition out of this layer
                continue
            Al = self.A[l]
            K, _ = _dare_cross(Al, Al, Q, Q, Q + R)           # cross-term DARE, B=A_l
            self.K[l] = K
            self.rho[l] = float(np.max(np.abs(np.linalg.eigvals(Al @ (np.eye(self.n) - K)))))
            if feedforward:
                rhs = self.ref[l + 1] - self.b[l]
                self.g[l] = np.linalg.solve(Al + ridge * np.eye(self.n), rhs) - self.ref[l]
            else:
                self.g[l] = np.zeros(self.n)

    def reset(self):
        pass                                                   # stateless

    def control(self, k, xi):
        if k not in self.K:
            return np.zeros_like(np.asarray(xi, np.float64))
        xi = np.asarray(xi, np.float64)                        # (M,k)
        delta = xi - self.ref[k]                               # (M,k)
        u = -(delta @ self.K[k].T) + self.g[k]                 # (M,k)
        return _clamp_ball(u, self.u_max)


# =============================================================================
# realized-effort recorder (matched-effort accounting)
# =============================================================================


class RecordingController:
    """Wrap any controller to log realized control effort ‖u_l‖ per .control call, so the
    head-to-head can report behaviour/coherence at *equal measured effort*, not just equal cap.

    Also accumulates the mean **actuated** cone coordinate s_l = c_l + u_l per layer, so the
    coherence SURROGATE (Σ_l Mahalanobis(s_l, on-manifold density)) can be measured per
    condition and correlated with genNLL (CALM Phase 1 / `calm_mpc.mahalanobis_trajectory`)."""

    def __init__(self, inner):
        self.inner = inner
        self.layers = getattr(inner, "layers", None)
        self.reset_effort()

    def reset_effort(self):
        self._sum = 0.0; self._cnt = 0; self._per_layer = {}
        self._scoord_sum = {}; self._scoord_cnt = {}        # actuated-coord accumulators

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def control(self, k, xi):
        u = self.inner.control(k, xi)
        ua = np.asarray(u, np.float64)
        nrm = float(np.linalg.norm(ua, axis=-1).mean())
        self._sum += nrm; self._cnt += 1
        self._per_layer.setdefault(k, []).append(nrm)
        s = np.asarray(xi, np.float64) + ua                 # actuated cone coordinate
        s2 = s.reshape(-1, s.shape[-1])
        self._scoord_sum[k] = self._scoord_sum.get(k, 0.0) + s2.sum(0)
        self._scoord_cnt[k] = self._scoord_cnt.get(k, 0) + s2.shape[0]
        return u

    @property
    def mean_effort(self):
        return self._sum / self._cnt if self._cnt else 0.0

    def per_layer_effort(self):
        return {k: float(np.mean(v)) for k, v in sorted(self._per_layer.items())}

    def actuated_means(self):
        """{layer: mean actuated cone coordinate s_l:(k,)} over all recorded .control calls."""
        return {k: self._scoord_sum[k] / self._scoord_cnt[k] for k in sorted(self._scoord_sum)}


# =============================================================================
# Self-test (synthetic — no model)
# =============================================================================

def _rand_spd(rng, n, scale=1.0):
    M = rng.randn(n, n)
    return scale * (M @ M.T / n + np.eye(n))


def _selftest_lqr_regulation(rng):
    """ConeLQR (regulation, ref=0, b=0) yields a stable closed loop and contracts to 0
    under the actuated-output plant c⁺=A(c+u)."""
    for n in (1, 3, 4):
        A = 0.9 * np.eye(n) + 0.05 * rng.randn(n, n)
        Aseq = np.stack([A] * 8); bseq = np.zeros((8, n)); ref = np.zeros((9, n))
        lqr = ConeLQR(Aseq, bseq, ref, layers=list(range(8)), q_pos=1.0, r_ctrl=0.05,
                      feedforward=False)
        assert lqr.rho[0] < 1.0, f"n={n} closed loop not stable (rho={lqr.rho[0]:.3f})"
        x = rng.randn(n) * 3.0; n0 = np.linalg.norm(x)
        for l in range(8):
            u = lqr.control(l, x[None, :])[0]
            x = A @ (x + u)                                    # actuated-output plant
        assert np.linalg.norm(x) < 0.5 * n0, f"n={n} regulation did not contract"
    print("[lqr regulation] OK — stable closed loop (rho<1), contracts to 0")


def _selftest_lqr_feedforward(rng):
    """With a constant drift + constant nonzero ref, feedforward LQR reaches zero
    steady-state error while regulation-only LQR leaves an offset (the P/LQR-vs-PID point)."""
    n = 3
    A = 0.7 * np.eye(n) + 0.03 * rng.randn(n, n)
    T = 60
    Aseq = np.stack([A] * (T + 1)); bseq = np.tile(0.5 * rng.randn(n), (T + 1, 1))
    ref_vec = rng.randn(n); ref = np.tile(ref_vec, (T + 2, 1))
    layers = list(range(T))
    err = {}
    for ff in (False, True):
        lqr = ConeLQR(Aseq, bseq, ref, layers, q_pos=1.0, r_ctrl=0.05, feedforward=ff)
        x = rng.randn(n)
        for l in layers:
            u = lqr.control(l, x[None, :])[0]
            x = A @ (x + u) + bseq[l]
        err[ff] = np.linalg.norm(x - ref_vec)
    assert err[True] < 1e-3, f"feedforward steady-state error too high: {err[True]:.2e}"
    assert err[False] > 10 * err[True], "regulation should leave a larger offset than feedforward"
    print(f"[lqr feedforward] OK — ss error ff=True {err[True]:.2e} << ff=False {err[False]:.2e}")


def _selftest_pid(rng):
    """PID integral removes the steady-state offset that pure-P leaves (constant drift)."""
    n = 3
    A = 0.6 * np.eye(n) + 0.02 * rng.randn(n, n)
    T = 80; drift = 0.4 * rng.randn(n); ref_vec = rng.randn(n)
    ref = np.tile(ref_vec, (T + 2, 1)); layers = list(range(T))
    err = {}
    for label, ctrl in (("P", ConeP(ref, layers, kp=0.5)),
                        ("PID", ConePID(ref, layers, kp=0.5, ki=0.15, kd=0.01))):
        ctrl.reset(); x = rng.randn(n)
        for l in layers:
            u = ctrl.control(l, x[None, :])[0]
            x = A @ (x + u) + drift
        err[label] = np.linalg.norm(x - ref_vec)
    assert err["PID"] < err["P"], f"PID ({err['PID']:.3f}) should beat P ({err['P']:.3f})"
    assert err["PID"] < 0.2, f"PID steady-state error too high: {err['PID']:.3f}"
    print(f"[pid] OK — steady-state error P {err['P']:.3f} -> PID {err['PID']:.3f}")


def _selftest_pid_reset(rng):
    """control() at the first band layer resets state (per-forward-pass), and i_reset works."""
    n = 2; ref = np.zeros((10, n)); layers = [2, 3, 4, 5]
    pid = ConePID(ref, layers, kp=1.0, ki=1.0, kd=0.0, i_reset=0)
    x = np.ones((1, n))
    u_a = pid.control(2, x); _ = pid.control(3, x)
    # a fresh pass (layer==first) must reset the integral, so u at layer 2 repeats
    u_b = pid.control(2, x)
    assert np.allclose(u_a, u_b), "first-layer reset failed"
    print("[pid reset] OK — integral resets at the first band layer each forward pass")


def _selftest_clamp(rng):
    for n in (1, 4):
        u = rng.randn(7, n) * 5.0
        uc = _clamp_ball(u, 0.3)
        assert np.all(np.linalg.norm(uc, axis=-1) <= 0.3 + 1e-9), "clamp exceeded budget"
        assert _clamp_ball(u, None) is u and np.allclose(_clamp_ball(u, np.inf), u)
    print("[clamp] OK — per-row L2 ball, inf/None passthrough")


def _selftest_mpc_equals_lqr(rng):
    """The control-theory crux, in code: on a time-invariant unconstrained plant, the
    receding-horizon MPC with terminal cost Qf = the (cross-term) DARE cost-to-go S∞
    reproduces the infinite-horizon LQR control. MPC's edge over LQR therefore requires a
    binding constraint / finite-horizon mismatch — not present here."""
    import casa_control as ctrl
    from pts_mpc import assemble_q
    for n in (1, 3):
        A = 0.8 * np.eye(n) + 0.04 * rng.randn(n, n)
        T = 60
        Aseq = np.stack([A] * T); bseq = np.zeros((T, n)); ref = np.zeros((T + 1, n))
        q_pos, r_ctrl = 1.0, 0.05
        # cross-term DARE for the MPC's exact cost (actuated state s=c+u costed)
        K, S = _dare_cross(A, A, q_pos * np.eye(n), q_pos * np.eye(n),
                           (q_pos + r_ctrl) * np.eye(n))
        # unconstrained MPC, long horizon, terminal Qf = S∞
        l0, H = 2, 16
        cqp = ctrl.build_condensed_qp(Aseq, bseq, l0, H, q_pos * np.eye(n),
                                      r_ctrl * np.eye(n), S, n)
        x = rng.randn(1, n)
        q = assemble_q(cqp, x, ref[l0:l0 + H + 1])
        U = ctrl.solve_qp(cqp["P"], q, None, H, n, L=cqp["L"])
        u_mpc = U[..., :n][0]
        u_lqr = -(x[0] @ K.T)
        assert np.allclose(u_mpc, u_lqr, atol=1e-5), \
            f"n={n} unconstrained MPC(Qf=S∞) != LQR: {np.abs(u_mpc-u_lqr).max():.2e}"
    print("[mpc==lqr] OK — unconstrained MPC with Qf=S∞ reproduces the infinite-horizon LQR")


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    _selftest_clamp(rng)
    _selftest_lqr_regulation(rng)
    _selftest_lqr_feedforward(rng)
    _selftest_pid(rng)
    _selftest_pid_reset(rng)
    _selftest_mpc_equals_lqr(rng)
    print("[casa_baselines self-test] ALL OK — P/PID/LQR laws + MPC≡LQR equivalence")
