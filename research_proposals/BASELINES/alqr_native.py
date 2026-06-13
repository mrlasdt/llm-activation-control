"""Native A-LQR (Skifstad/Yang/Chou, "Local Linearity ... Model-Based Linear Optimal Control").

Faithful reimplementation of the Activation-LQR control law on HF models, cross-checked
against the authors' code (BASELINES_PROVENANCE.md §1). The pieces:

  * feature direction (diff-of-means)  v_k = e_k/‖e_k‖,  μ_k = ‖e_k‖,  e_k = z̄_{k,+} − z̄_{k,−}.
  * LFS setpoint (adaptive, per layer)  β*_k = λ·μ_k.
  * layer Jacobians  A_k = ∂φ_k/∂z at the nominal z̄_{k,+}, B_k := I  (via column-wise JVP,
    `linearize_jvp_streamed_gpu` in the reference).
  * finite-horizon backward Riccati (B=I), `tv_lqr_noB`, from S_T=Qf → gains K_k.
  * control law  u_k = (β*_k − v_kᵀ z_k)·K_k v_k   (the rank-1 deviation δz_k=−α_k v_k means
    we only ever need w_k := K_k v_k — a single d-vector per layer), added to the residual
    output. A-LQR = last token; A-LQR+ = all token positions (jailbreak).
  * defaults: Q=q·I, R=r·I, Qf=qf·I; q=r=10, qf=1 general (refusal q=0.1, r=10, qf=0.1, λ=1).

Self-test (synthetic, no model):  python alqr_native.py
Model-in-the-loop entry points: `feature_signal`, `layer_jacobians`, `alqr_gains`, `ALQRSteer`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))   # shared lib
sys.path.insert(0, str(_HERE.parents[1] / "PTS"))            # pts_mpc.dlqr (cross-check)
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))           # casa_cone.residual_means


# =============================================================================
# Riccati  (time-varying, B=I)  — the reference's time_varying_lqr_noB
# =============================================================================


def tv_lqr_noB(A, Q, R, S_T):
    """Finite-horizon backward Riccati for x_{k+1}=A_k x_k + u_k (B=I), tracking cost
    Σ δzᵀQδz + δuᵀRδu + δz_TᵀQfδz_T. Returns gains K:(T,d,d) with u_k=−K_k δz_k.

      P = S_{k+1} + R ;  F = S_{k+1} A_k ;  G = Q + A_kᵀ S_{k+1} A_k
      K_k = P⁻¹ F ;  S_k = G − Fᵀ P⁻¹ F        (matches lqr_utils.time_varying_lqr_noB)
    """
    A = np.asarray(A, np.float64)
    T, n, _ = A.shape
    Q = np.broadcast_to(Q, (T, n, n)); R = np.broadcast_to(R, (T, n, n))
    S = np.asarray(S_T, np.float64).copy()
    K = np.zeros((T, n, n))
    for t in range(T - 1, -1, -1):
        At = A[t]
        P = S + R[t]
        F = S @ At
        G = Q[t] + At.T @ S @ At
        Pinv = np.linalg.inv(P)
        K[t] = Pinv @ F
        S = G - F.T @ Pinv @ F
        S = 0.5 * (S + S.T)
    return K


# =============================================================================
# feature directions + LFS
# =============================================================================


def feature_signal(hmean_pos, hmean_neg):
    """e_k = z̄_{k,+} − z̄_{k,−}; returns (e:(L,d), v:(L,d) unit, mu:(L,)).
    `+` is the desired/benign side (e.g. harmless / non-toxic), `−` the contrastive side."""
    e = np.asarray(hmean_pos, np.float64) - np.asarray(hmean_neg, np.float64)
    mu = np.linalg.norm(e, axis=-1)
    v = e / (mu[:, None] + 1e-12)
    return e, v, mu


def lfs_setpoints(mu, lam=1.0):
    """β*_k = λ·μ_k (the adaptive Linear Feature Setpoint)."""
    return lam * np.asarray(mu, np.float64)


# =============================================================================
# Jacobians via column-wise JVP  (faithful to linearize_jvp_streamed_gpu)
# =============================================================================


def jacobian_jvp(fn, x):
    """Full Jacobian of fn at x via forward-mode JVP, one column per basis vector.
    fn: R^n -> R^n (operates on a single vector x:(n,)). Returns A:(n,n) with A[:,j]=∂fn/∂x_j.
    This is the memory-frugal scheme the reference uses for d≈2k–2.3k LLM residuals."""
    x = x.detach()
    n = x.shape[-1]
    A = torch.zeros((n, n), dtype=x.dtype, device=x.device)
    for j in range(n):
        v = torch.zeros_like(x); v[..., j] = 1.0
        _, jvp = torch.autograd.functional.jvp(fn, (x,), (v,), create_graph=False, strict=False)
        A[:, j] = jvp
    return A


def make_block_fn(model, layer_idx, attention_mask, position_ids, position_embeddings):
    """Wrap transformer block `layer_idx` as a last-token map z_last -> z_last_out, holding the
    rest-of-sequence context (mask, rotary) fixed. Mirrors lqr_utils.tf_block_wrapper +
    transformerBlockControl. Used at the nominal z̄ to get A_k = ∂(block out)/∂(block in)."""
    block = model.model.layers[layer_idx]

    def fn(z_last):                                          # z_last:(d,)
        x = z_last.unsqueeze(0).unsqueeze(0)                 # (1,1,d) — single last token
        out = block(x, attention_mask=attention_mask, position_ids=position_ids,
                    position_embeddings=position_embeddings)
        h = out[0] if isinstance(out, tuple) else out
        return h[0, -1, :]
    return fn


def layer_jacobians(model, nominal_z, device, layers=None):
    """A_k for each layer at the nominal last-token activation nominal_z:(L,d).
    Returns A:(len(layers), d, d) numpy. EXPENSIVE (d JVPs × #layers) — model-in-the-loop;
    run on the band or a sampled subset for diagnostics. position context is a length-1 seq."""
    model.eval()
    L = model.config.num_hidden_layers
    layers = list(range(L) if layers is None else layers)
    d = model.config.hidden_size
    pos_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    hs = torch.zeros((1, 1, d), device=device, dtype=next(model.parameters()).dtype)
    pos_emb = model.model.rotary_emb(hs, pos_ids)
    out = np.zeros((len(layers), d, d), np.float64)
    for i, l in enumerate(layers):
        z = torch.as_tensor(nominal_z[l], device=device,
                            dtype=next(model.parameters()).dtype)
        fn = make_block_fn(model, l, None, pos_ids, pos_emb)
        out[i] = jacobian_jvp(fn, z).float().cpu().numpy()
    return out


def alqr_gains(A, q=10.0, r=10.0, qf=1.0):
    """A-LQR gains via the B=I Riccati with Q=q·I, R=r·I, Qf=qf·I. A:(T,d,d)."""
    A = np.asarray(A, np.float64); n = A.shape[1]
    return tv_lqr_noB(A, q * np.eye(n), r * np.eye(n), qf * np.eye(n))


# =============================================================================
# closed-loop A-LQR steering controller (model-in-the-loop)
# =============================================================================


class ALQRSteer:
    """A-LQR closed-loop controller. Precompute w_k = K_k v_k (rank-1: the only thing the
    control needs since δz_k = −α_k v_k). Online at each layer output:
        β = v_kᵀ z ;  α = β*_k − β ;  z += α · w_k.
    all_tokens=True replicates across token positions (A-LQR+)."""

    def __init__(self, v, beta_star, gains, layers=None, all_tokens=False):
        self.v = np.asarray(v, np.float64)                   # (L,d) unit
        self.beta_star = np.asarray(beta_star, np.float64)   # (L,)
        self.L = self.v.shape[0]
        self.layers = set(range(self.L) if layers is None else layers)
        # w_k = K_k v_k
        self.w = np.zeros_like(self.v)
        for k in range(min(self.L, gains.shape[0])):
            self.w[k] = gains[k] @ self.v[k]
        self.all_tokens = all_tokens

    def hooks(self, module_dict):
        vt = torch.from_numpy(self.v); wt = torch.from_numpy(self.w)
        bt = torch.from_numpy(self.beta_star)

        def mk(layer):
            v = vt[layer]; w = wt[layer]; bstar = float(bt[layer])

            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if layer not in self.layers:
                    return out
                vv = v.to(h.device, h.dtype); ww = w.to(h.device, h.dtype)
                if self.all_tokens:
                    beta = h @ vv                            # (B,S)
                    alpha = (bstar - beta).unsqueeze(-1)     # (B,S,1)
                    steered = h + alpha * ww
                else:
                    beta = h[:, -1, :] @ vv                  # (B,)
                    alpha = (bstar - beta).unsqueeze(-1)     # (B,1)
                    h = h.clone()
                    h[:, -1, :] = h[:, -1, :] + alpha * ww
                    steered = h
                return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered
            return hook

        return [(module_dict[f"model.layers.{j}"], mk(j)) for j in sorted(self.layers)]


# =============================================================================
# Self-test (synthetic — no LLM)
# =============================================================================


def _selftest_riccati(rng):
    """tv_lqr_noB on a time-invariant plant converges (long horizon) to the DARE gain
    dlqr(A, I, Q, R)."""
    from pts_mpc import dlqr
    for n in (1, 3, 5):
        A = 0.9 * np.eye(n) + 0.05 * rng.randn(n, n)
        T = 200
        Aseq = np.stack([A] * T)
        Q = np.eye(n); R = 3.0 * np.eye(n)
        K = tv_lqr_noB(Aseq, Q, R, Q.copy())
        Kdare = dlqr(A, np.eye(n), Q, R)["K"]
        assert np.allclose(K[0], Kdare, atol=1e-6), \
            f"n={n} tv_lqr_noB head gain != DARE: {np.abs(K[0]-Kdare).max():.2e}"
    print("[riccati] OK — B=I time-varying Riccati → DARE gain at long horizon")


def _selftest_jvp(rng):
    """jacobian_jvp on a tiny torch MLP matches autograd.functional.jacobian and finite diff."""
    torch.manual_seed(0)
    n = 6
    net = torch.nn.Sequential(torch.nn.Linear(n, n), torch.nn.Tanh(), torch.nn.Linear(n, n))
    net.double().eval()
    x = torch.randn(n, dtype=torch.float64)
    fn = lambda z: net(z)
    A = jacobian_jvp(fn, x)
    A_ref = torch.autograd.functional.jacobian(fn, x)
    assert torch.allclose(A, A_ref, atol=1e-8), f"JVP != autograd jacobian: {(A-A_ref).abs().max():.2e}"
    # finite-difference spot check
    eps = 1e-5
    for j in (0, 3):
        xp = x.clone(); xp[j] += eps; xm = x.clone(); xm[j] -= eps
        fd = (fn(xp) - fn(xm)) / (2 * eps)
        assert torch.allclose(A[:, j], fd, atol=1e-4), f"col {j} != finite diff"
    print("[jvp jacobian] OK — matches autograd jacobian & finite differences on a tiny MLP")


def _selftest_control_identity(rng):
    """u_k = (β*−vᵀz)·(K_k v_k) equals K_k @ δz with δz = −α v (the rank-1 reduction)."""
    n = 4
    K = rng.randn(n, n); v = rng.randn(n); v = v / np.linalg.norm(v)
    z = rng.randn(n); bstar = 0.7
    alpha = bstar - v @ z
    w = K @ v
    u_rank1 = alpha * w
    u_full = K @ (alpha * v)                                 # = -K δz with δz=-αv
    assert np.allclose(u_rank1, u_full, atol=1e-12), "rank-1 control identity failed"
    print("[control identity] OK — (β*−vᵀz)·K v == K(α v), so only w=Kv is needed")


def _selftest_lfs(rng):
    e = rng.randn(5, 8) * np.array([1, 2, 3, 4, 5])[:, None]
    _, v, mu = feature_signal(e, np.zeros_like(e))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0), "v not unit"
    assert np.allclose(lfs_setpoints(mu, 2.0), 2.0 * mu), "LFS scale wrong"
    # the per-layer setpoint normalizes activation-norm variation: β*/μ == λ (const)
    assert np.allclose(lfs_setpoints(mu, 1.5) / mu, 1.5), "β*/μ should equal λ"
    print("[lfs] OK — unit v, β*_k=λμ_k (per-layer norm-adaptive setpoint)")


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    _selftest_riccati(rng)
    _selftest_jvp(rng)
    _selftest_control_identity(rng)
    _selftest_lfs(rng)
    print("[alqr_native self-test] ALL OK — Riccati≡DARE, JVP Jacobian, rank-1 control, LFS")
