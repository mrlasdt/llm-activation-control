"""CALM — Coherence-Aware Lookahead MPC.

The BASELINES head-to-head found the control LAW is a wash for layer-domain refusal
(P ≈ LQR ≈ MPC); the bound is the only lever. The diagnosis: refusal strength↔coherence
is a *saturating hump*, not a frontier, so MPC's constrained optimization has nothing to
trade against. The literature names the real open problem — stronger steering pushes
activations OFF-MANIFOLD → degradation (IDS arXiv:2510.13285; Dynamic Activation
Composition arXiv:2406.17563) — but existing fixes are heuristic scalar intensity dials.

CALM puts coherence INSIDE the controller. `CoherenceMPC` augments `casa_control.ConeMPC`'s
purely-quadratic tracking cost with an **on-manifold density penalty** — a per-layer
anisotropic Mahalanobis term on the *actuated* cone state:

    J = Σ_l  ‖s_l − ref_l‖²_Q   +   κ·(s_l − μ_l)ᵀ Σ_l⁻¹ (s_l − μ_l)   +   ‖u_l‖²_R
              └ suppress refusal ┘     └ stay in the on-distribution density ┘   └ effort ┘
        s.t. ‖u_l‖ ≤ u_max,   s_l = c_l + u_l   (additive actuator)

where `(μ_l, Σ_l)` is the on-distribution (harmless) cone-coordinate density. The density
term is QUADRATIC, so it folds straight into PTS/CASA's condensed QP and is solved by the
SAME FISTA ball-projection — no new solver. It is exactly what P/LQR/clip cannot represent
(they track a point or clip a norm; they have no local geometry): it lets the MPC spend
control freely along high-variance / robust cone directions and gently along brittle
low-variance ones. κ traces the strength↔coherence frontier; κ=0 recovers ConeMPC exactly.

Reuses `casa_control.build_condensed_qp` / `solve_qp` and `pts_mpc.assemble_q` UNCHANGED —
the density term is assembled with the same `assemble_q(cqp, xi, target)` machinery, just
with `Qblk → κ·blkdiag(Σ_l⁻¹)` and `ref → μ`.

Self-test (synthetic, no model):  python calm_mpc.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))           # casa_control
sys.path.insert(0, str(_HERE.parents[1] / "PTS"))            # pts_mpc.assemble_q
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))   # shared lib

from casa_control import build_condensed_qp, solve_qp        # noqa: E402
from pts_mpc import assemble_q, qp_objective                 # noqa: E402


# =============================================================================
# on-manifold density  (μ_l, Σ_l⁻¹) per layer from on-distribution cone coords
# =============================================================================


def fit_cone_density(coords, ridge=1e-2, normalize=True):
    """Per-layer Gaussian density of the k-dim cone coordinate on a reference
    (e.g. harmless) distribution: mean μ_l and inverse covariance Σ_l⁻¹.

    coords: (N, L, k) cone-coordinate trajectories. Returns dict mu:(L,k),
    Sinv:(L,k,k). If `normalize`, each Σ_l⁻¹ is rescaled to mean-eigenvalue 1
    (trace = k) so the Mahalanobis penalty is magnitude-comparable to an isotropic
    Q=I tracking cost — i.e. κ is a clean *relative* weight and only the ANISOTROPY
    (eigenvalue ratios) of Σ_l⁻¹ enters, not its absolute scale."""
    coords = np.asarray(coords, np.float64)
    N, L, k = coords.shape
    mu = coords.mean(0)                                       # (L,k)
    Sinv = np.zeros((L, k, k))
    for l in range(L):
        X = coords[:, l, :] - mu[l]
        S = (X.T @ X) / max(N - 1, 1) + ridge * np.eye(k)
        Si = np.linalg.inv(S)
        if normalize:
            Si = Si * (k / np.trace(Si))                      # mean eigenvalue -> 1
        Sinv[l] = 0.5 * (Si + Si.T)
    return {"mu": mu, "Sinv": Sinv}


def mahalanobis_trajectory(coords_actuated, mu, Sinv, layers):
    """Σ_{l∈layers} (s_l − μ_l)ᵀ Σ_l⁻¹ (s_l − μ_l) for a single actuated trajectory
    `coords_actuated`: dict layer->(k,) (or (k,)-mean). The coherence SURROGATE: how
    far the actuated cone trajectory sits from the on-distribution density. Used to
    validate (does it predict genNLL?) and to report per condition."""
    tot = 0.0
    for l in layers:
        if l not in coords_actuated or l >= len(mu):
            continue
        e = np.asarray(coords_actuated[l], np.float64) - mu[l]
        tot += float(e @ Sinv[l] @ e)
    return tot


# =============================================================================
# Coherence-Aware MPC  (ConeMPC + anisotropic on-manifold density term)
# =============================================================================


class CoherenceMPC:
    """Receding-horizon additive MPC over the k-dim cone coordinate, with the on-manifold
    density penalty folded into the condensed QP. Same contract as `casa_control.ConeMPC`:
    `.control(layer, xi:(...,k)) -> u0:(...,k)` and `.reset()`. κ=0 ≡ ConeMPC."""

    def __init__(self, A, b, ref, layers, Sinv, mu=None, kappa=0.0, H=6,
                 q_pos=1.0, r_ctrl=0.05, qf_scale=4.0, u_max=None, fista_iters=60):
        self.A = np.asarray(A, np.float64); self.b = np.asarray(b, np.float64)
        self.ref = np.asarray(ref, np.float64)                # (L,k)
        self.Sinv = np.asarray(Sinv, np.float64)              # (L,k,k)
        self.mu = self.ref if mu is None else np.asarray(mu, np.float64)   # density centre
        self.kappa = float(kappa)
        self.n = self.A.shape[1]
        self.layers = list(layers)
        self.L = min(self.A.shape[0] + 1, self.ref.shape[0])
        self.H = H; self.u_max = u_max; self.fista_iters = fista_iters
        Q = q_pos * np.eye(self.n)
        R = r_ctrl * np.eye(self.n)
        Qf = qf_scale * q_pos * np.eye(self.n)
        self._cqp = {}; self._dens = {}; self._P = {}; self._Lip = {}
        self._refstack = {}; self._mustack = {}
        for kk in self.layers:
            Hk = min(H, self.L - 1 - kk)
            if Hk < 1:
                continue
            cqp = build_condensed_qp(self.A, self.b, kk, Hk, Q, R, Qf, self.n)
            self._cqp[kk] = cqp
            self._refstack[kk] = self.ref[kk:kk + Hk + 1]
            # density block over the Hk+1 horizon stages: Wblk = κ·blkdiag(Σ_{l}⁻¹)
            Hs = Hk + 1; n = self.n
            Wblk = np.zeros((n * Hs, n * Hs))
            for j in range(Hs):
                lj = min(kk + j, self.Sinv.shape[0] - 1)
                Wblk[n * j:n * j + n, n * j:n * j + n] = self.kappa * self.Sinv[lj]
            G = cqp["G"]
            P_dens = 2.0 * (G.T @ Wblk @ G); P_dens = 0.5 * (P_dens + P_dens.T)
            P_tot = cqp["P"] + P_dens
            self._P[kk] = P_tot
            self._Lip[kk] = float(np.linalg.eigvalsh(P_tot)[-1])
            # a lightweight cqp-like dict so assemble_q computes the density linear term
            self._dens[kk] = {"G": G, "F": cqp["F"], "phi": cqp["phi"], "Qblk": Wblk}
            self._mustack[kk] = self.mu[kk:kk + Hk + 1]
        if not self._cqp:
            raise ValueError("no actuated layer has a positive horizon")

    def reset(self):
        """No-op: stateless (re-solves each call). Matches the controller contract."""
        pass

    def control(self, k, xi):
        if k not in self._cqp:
            return np.zeros_like(np.asarray(xi, np.float64))
        cqp = self._cqp[k]
        q = assemble_q(cqp, xi, self._refstack[k])            # tracking linear term
        if self.kappa != 0.0:
            q = q + assemble_q(self._dens[k], xi, self._mustack[k])   # density linear term
        U = solve_qp(self._P[k], q, self.u_max, cqp["H"], self.n,
                     L=self._Lip[k], iters=self.fista_iters)
        return U[..., :self.n]


# =============================================================================
# Self-test (synthetic — no model)
# =============================================================================

def _selftest_kappa0_equals_conempc(rng):
    """κ=0 reproduces casa_control.ConeMPC to numerical precision (the density term
    vanishes), so CALM is a strict generalization."""
    from casa_control import ConeMPC
    for n in (1, 3, 4):
        L = 14
        A = np.stack([0.9 * np.eye(n) + 0.05 * rng.randn(n, n) for _ in range(L - 1)])
        b = 0.1 * rng.randn(L - 1, n)
        ref = 0.2 * rng.randn(L, n)
        Sinv = np.stack([_rand_spd(rng, n) for _ in range(L)])
        layers = list(range(L - 1))
        cone = ConeMPC(A, b, ref, layers, H=6, u_max=0.4)
        calm = CoherenceMPC(A, b, ref, layers, Sinv=Sinv, kappa=0.0, H=6, u_max=0.4)
        worst = 0.0
        for kk in layers:
            xi = rng.randn(5, n)
            worst = max(worst, float(np.abs(cone.control(kk, xi) - calm.control(kk, xi)).max()))
        assert worst < 1e-9, f"n={n} κ=0 != ConeMPC (max diff {worst:.2e})"
    print("[calm κ=0] OK — κ=0 reproduces ConeMPC exactly")


def _selftest_cost_matches_bruteforce(rng):
    """The assembled (P_tot, q_tot) reproduce the TRUE augmented cost — tracking +
    κ-weighted Mahalanobis density + effort — along the rollout, to ~1e-8."""
    for n in (1, 3):
        L = 16
        A = np.stack([0.85 * np.eye(n) + 0.04 * rng.randn(n, n) for _ in range(L - 1)])
        b = 0.1 * rng.randn(L - 1, n)
        ref = 0.3 * rng.randn(L, n); mu = 0.3 * rng.randn(L, n)
        Sinv = np.stack([_rand_spd(rng, n) for _ in range(L)])
        q_pos, r_ctrl, qf_scale, kappa = 1.0, 0.05, 4.0, 0.7
        Q = q_pos * np.eye(n); R = r_ctrl * np.eye(n); Qf = qf_scale * np.eye(n)
        kk, H = 2, 6
        calm = CoherenceMPC(A, b, ref, list(range(L - 1)), Sinv=Sinv, mu=mu,
                            kappa=kappa, H=H, q_pos=q_pos, r_ctrl=r_ctrl, qf_scale=qf_scale)
        cqp = calm._cqp[kk]; Hk = cqp["H"]
        xi = rng.randn(n)
        q = assemble_q(cqp, xi, calm._refstack[kk]) + assemble_q(calm._dens[kk], xi, calm._mustack[kk])

        def sim_cost(U):
            s = xi + U[:n]; J = 0.0
            for j in range(Hk + 1):
                Qj = Qf if j == Hk else Q
                e = s - ref[kk + j]; J += e @ Qj @ e
                em = s - mu[kk + j]; J += kappa * (em @ Sinv[kk + j] @ em)
                if j < Hk:
                    uj = U[n * j:n * j + n]; J += uj @ R @ uj
                    s = A[kk + j] @ s + b[kk + j]
                    if j + 1 <= Hk - 1:
                        s = s + U[n * (j + 1):n * (j + 1) + n]
            return J

        # const = the parts of (X-ref)'Qblk(X-ref)+κ(X-μ)'Wblk(X-μ) not captured by qp_objective
        G, F, phi = cqp["G"], cqp["F"], cqp["phi"]
        Qblk = cqp["Qblk"]; Wblk = calm._dens[kk]["Qblk"]
        d_r = xi @ F.T + phi - calm._refstack[kk].reshape(-1)
        d_m = xi @ F.T + phi - calm._mustack[kk].reshape(-1)
        const = d_r @ Qblk @ d_r + d_m @ Wblk @ d_m
        rel = max(abs(qp_objective(calm._P[kk], q, U := rng.randn(n * Hk)) + const - sim_cost(U))
                  / (abs(sim_cost(U)) + 1e-9) for _ in range(50))
        assert rel < 1e-8, f"n={n} augmented-cost rel err {rel:.2e}"
    print("[calm cost] OK — assembled P,q match the true tracking+density+effort cost (1e-8)")


def _selftest_kappa_pulls_to_manifold(rng):
    """With ref pulling AWAY from the density centre μ, increasing κ keeps the closed-loop
    cone trajectory closer to μ (the on-manifold pull strengthens) — the coherence knob."""
    n = 3; L = 22
    A = np.stack([0.8 * np.eye(n) + 0.03 * rng.randn(n, n) for _ in range(L - 1)])
    b = np.zeros((L - 1, n))
    mu = np.zeros((L, n))                                      # density centre at 0
    ref = np.tile(2.0 * np.ones(n), (L, 1))                   # tracking target far from μ
    # anisotropic Σ⁻¹: cheap (small penalty) along dim 0, brittle (large) along the rest
    base = np.diag([0.2] + [3.0] * (n - 1))
    Sinv = np.stack([base] * L)
    layers = list(range(L - 1))
    dist = {}
    for kappa in (0.0, 5.0):
        calm = CoherenceMPC(A, b, ref, layers, Sinv=Sinv, mu=mu, kappa=kappa, H=6, u_max=None)
        s = rng.randn(n)
        acc = []
        for kk in layers:
            u = calm.control(kk, s[None, :])[0]
            s = s + u
            acc.append(np.linalg.norm(s - mu[kk]))            # distance to manifold centre
            s = A[kk] @ s + b[kk]
        dist[kappa] = float(np.mean(acc))
    assert dist[5.0] < dist[0.0], \
        f"larger κ should pull toward μ: κ=0 {dist[0.0]:.3f} vs κ=5 {dist[5.0]:.3f}"
    print(f"[calm κ-pull] OK — mean ‖s−μ‖ κ=0 {dist[0.0]:.3f} -> κ=5 {dist[5.0]:.3f} (closer)")


def _selftest_density_fit(rng):
    """fit_cone_density recovers a known anisotropy and the normalization (trace=k)."""
    N, L, k = 4000, 6, 3
    true_cov = np.diag([4.0, 1.0, 0.25])
    coords = rng.randn(N, L, k) @ np.linalg.cholesky(true_cov).T
    coords = coords.reshape(N, L, k) + np.arange(L)[None, :, None]   # per-layer mean shift
    dens = fit_cone_density(coords, normalize=True)
    assert np.allclose(dens["mu"][:, 0], np.arange(L), atol=0.1), "mean recovery failed"
    Si = dens["Sinv"][0]
    assert abs(np.trace(Si) - k) < 1e-6, f"normalization off: trace {np.trace(Si):.4f}"
    order = np.argsort(np.diag(Si))                          # 1/var: dim2 (small var) largest
    assert list(order) == [0, 1, 2], "anisotropy ordering wrong"
    print("[calm density] OK — μ, anisotropy, and trace=k normalization recovered")


def _rand_spd(rng, n, scale=1.0):
    M = rng.randn(n, n)
    return scale * (M @ M.T / n + np.eye(n))


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    _selftest_density_fit(rng)
    _selftest_kappa0_equals_conempc(rng)
    _selftest_cost_matches_bruteforce(rng)
    _selftest_kappa_pulls_to_manifold(rng)
    print("[calm_mpc self-test] ALL OK — density fit, κ=0≡ConeMPC, augmented-cost "
          "exactness, κ on-manifold pull")
