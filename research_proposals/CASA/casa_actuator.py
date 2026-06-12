"""CASA actuator — bounded additive control over a k-dim refusal *cone*.

This is the first-class generalization of the validated prototype hook
(`../PTS/verify/additive_subspace_steer.py::make_ablate_hook`). Where Angular
Steering uses a *norm-preserving SO(2) rotation* of a 2D plane (which collapses
the 2D plan to a single realized DoF — the angle — and caps out on Gemma; see
`../PTS/PTS_README.md` Verdict), CASA uses a **norm-CHANGING additive** push along
a k-dim subspace `U` (orthonormal rows), applied across a layer band.

The subspace `U` is meant to be a *refusal concept cone* in the sense of
Wollschläger et al., "The Geometry of Refusal in LLMs: Concept Cones and
Representational Independence" (ICML 2025): a basis `B=[b_1..b_k]` whose
non-negative span `{Σ λ_i b_i : λ_i ≥ 0}` is entirely refusal-mediating. The
basis is discovered in `casa_cone.py`; this module only *applies* it. Keeping
discovery (what subspace) and actuation (how to push on it) separate is the whole
point — CASA's failed L2 used a blunt SVD subspace here; the cone is the fix.

Two operations, matching the paper (its Eq. 1 / Eq. 2), both made bounded:

  directional ablation (DE-refusal):   p = U h ;  u = clip(-ρ·p, ‖u‖≤u_max) ;  h' = h + Uᵀ u
  activation addition  (INDUCE refuse): h' = h + α · ŵ            (ŵ a unit cone vector)

Levers exposed (CASA proposal §2):
  * L1  — additive, norm-CHANGING actuator (vs rotation).           [always]
  * L2  — k>1: U has k orthonormal rows = the cone basis.           [k arg]
  * L5  — `u_max`: per-(token) bound on ‖u‖ = the coherence dial.   [u_max arg]
  * ρ   — removal fraction in [0,1]: a second, unbounded de-refusal dial.
  * cone_clip — remove only the *in-cone* (λ_i≥0) part of the coordinate,
                i.e. push the activation to the cone boundary instead of fully
                out of span(U). This is the literal "steer within the concept
                cone" actuator the cone geometry invites (off by default; the
                paper's measured operation is full-subspace ablation).

Orientation convention: basis rows are oriented so a *positive* coordinate
`p_i = b_i·h` means "more refusal" (e.g. b = mean(harmful) − mean(harmless)).
Ablation removes positive projection; `cone_clip` removes only positive
projection. `casa_cone.py` orients its output the same way.

No model dependency — pure torch. Self-test:  python casa_actuator.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# =============================================================================
# Subspace helpers
# =============================================================================


def orthonormalize(U: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal-row basis spanning the rows of U ((k,d) -> (k',d)).

    Uses a reduced QR on Uᵀ. Drops numerically-dependent rows (k' may be < k).
    The DIM-per-layer "subspace" is generally NOT orthonormal; the RCO cone basis
    already is — orthonormalizing it is then a no-op up to sign.
    """
    if U.ndim != 2:
        raise ValueError(f"U must be (k,d), got {tuple(U.shape)}")
    if U.shape[0] > U.shape[1]:                          # k>d: more basis rows than dims
        raise ValueError(f"cone dim k={U.shape[0]} exceeds ambient d={U.shape[1]}")
    Q, R = torch.linalg.qr(U.t())                       # Q:(d,k), R:(k,k) since k<=d
    keep = R.diagonal().abs() > 1e-6
    Qk = Q[:, keep]                                     # (d,k')
    # preserve the original row orientation (sign of b_i·q_i > 0)
    signs = torch.sign((U[keep] * Qk.t()).sum(-1, keepdim=True))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (Qk.t() * signs)                             # (k',d), orthonormal rows


# =============================================================================
# Config + hooks
# =============================================================================


@dataclass
class ConeActuator:
    """Bounded additive control on a k-dim refusal cone.

    U        : (k,d) orthonormal-row basis (the cone). Pass through
               `orthonormalize` first if unsure.
    mode      : "ablate" (de-refusal) | "add" (induce refusal) | "off".
    u_max     : per-token L2 bound on the ablation control ‖u‖ (lever L5).
                None/inf = unbounded (exact projection-out when ρ=1).
    rho       : removal fraction in [0,1] for ablation (1 = full).
    cone_clip : ablate only the non-negative (in-cone) part of the coordinate.
    add_dir   : (d,) unit direction for "add" mode (a cone vector ŵ=Uᵀs/‖·‖, s≥0).
    alpha     : addition strength for "add" mode.
    """

    U: torch.Tensor
    mode: str = "ablate"
    u_max: float | None = None
    rho: float = 1.0
    cone_clip: bool = False
    add_dir: torch.Tensor | None = None
    alpha: float = 0.0

    def __post_init__(self):
        if self.mode not in ("ablate", "add", "off"):
            raise ValueError(f"mode must be ablate|add|off, got {self.mode}")
        if self.U.ndim != 2:
            raise ValueError(f"U must be (k,d), got {tuple(self.U.shape)}")
        # cache device/dtype-converted tensors per (device,dtype)
        self._cache: dict = {}

    @property
    def k(self) -> int:
        return self.U.shape[0]

    def _cast(self, h: torch.Tensor):
        key = (h.device, h.dtype)
        if key not in self._cache:
            U = self.U.to(h.device, h.dtype)
            w = None
            if self.add_dir is not None:
                w = self.add_dir.to(h.device, h.dtype)
            self._cache[key] = (U, w)
        return self._cache[key]

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the actuator to a residual-stream tensor h:(B,S,d) -> (B,S,d)."""
        if self.mode == "off":
            return h
        U, w = self._cast(h)
        if self.mode == "add":
            if w is None:
                raise ValueError("add mode needs add_dir")
            return h + self.alpha * w
        # ---- ablation (de-refusal) ----
        p = h @ U.t()                                   # (B,S,k) coordinate in cone
        target = p.clamp(min=0.0) if self.cone_clip else p
        u = -self.rho * target                          # desired control (B,S,k)
        if self.u_max is not None and math.isfinite(self.u_max):
            nrm = u.norm(dim=-1, keepdim=True)
            scale = torch.clamp(
                torch.where(nrm > 0, self.u_max / (nrm + 1e-6), torch.zeros_like(nrm)),
                max=1.0,
            )
            u = u * scale
        return h + u @ U                                # (B,S,d), norm-CHANGING

    def hook(self):
        """Return a tuple-aware forward hook for HF decoder layers."""

        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            steered = self.apply(h)
            if isinstance(out, tuple):
                return (steered,) + tuple(out[1:])
            return steered

        return _hook

    def band_hooks(self, module_dict, band):
        """[(module, hook), ...] applying this actuator across a layer band, for
        `pytorch_pure.utils.add_hooks`. Each layer shares one hook (the actuator
        is stateless per-forward), matching the prototype."""
        h = self.hook()
        return [(module_dict[f"model.layers.{j}"], h) for j in band]


# Backwards-compatible functional constructor mirroring the prototype's
# `make_ablate_hook(U, u_max)` (full-subspace ablation, ρ=1, no cone-clip).
def make_ablate_hook(U: torch.Tensor, u_max: float | None):
    return ConeActuator(U=U, mode="ablate", u_max=u_max).hook()


# =============================================================================
# Self-test (no model)
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    d, k = 64, 4
    # random orthonormal cone basis
    U = orthonormalize(torch.randn(k, d))
    assert U.shape[0] == k
    assert torch.allclose(U @ U.t(), torch.eye(k), atol=1e-5), "U not orthonormal"

    h = torch.randn(2, 5, d)

    # 1) full ablation (u_max=inf, rho=1) == exact projection-out  h - UᵀU h
    act = ConeActuator(U=U, mode="ablate", u_max=None)
    h_abl = act.apply(h)
    h_proj = h - (h @ U.t()) @ U
    assert torch.allclose(h_abl, h_proj, atol=1e-5), "full ablation != projection-out"
    # the cone coordinate is gone after ablation
    assert (h_abl @ U.t()).abs().max() < 1e-4, "coordinate not removed"
    # it is norm-CHANGING (the whole point vs rotation): norm strictly drops
    assert (h_abl.norm(dim=-1) < h.norm(dim=-1)).all(), "ablation should drop norm"

    # 2) bounded ablation respects ‖u‖ ≤ u_max exactly when it binds
    umax = 0.5
    actb = ConeActuator(U=U, mode="ablate", u_max=umax)
    u_real = (actb.apply(h) - h) @ U.t()                # realized control in coord
    # ‖u‖ in ambient == ‖u‖ in coord (U orthonormal); must be ≤ umax (+eps)
    assert u_real.norm(dim=-1).max() <= umax + 1e-4, "u_max bound violated"
    # and it binds here (coordinate norm exceeds umax for random h)
    assert (h @ U.t()).norm(dim=-1).max() > umax, "test ill-posed: bound never binds"

    # 3) rho=0.5 removes half the coordinate (unbounded)
    actr = ConeActuator(U=U, mode="ablate", u_max=None, rho=0.5)
    p0 = h @ U.t()
    p1 = actr.apply(h) @ U.t()
    assert torch.allclose(p1, 0.5 * p0, atol=1e-5), "rho=0.5 should halve coordinate"

    # 4) cone_clip removes only the non-negative coordinates, leaves negatives
    actc = ConeActuator(U=U, mode="ablate", u_max=None, cone_clip=True)
    p_after = actc.apply(h) @ U.t()
    neg = p0 < 0
    assert torch.allclose(p_after[neg], p0[neg], atol=1e-5), "cone_clip touched negatives"
    assert (p_after[~neg].abs() < 1e-4).all(), "cone_clip left positive coordinate"

    # 5) add mode injects a unit cone direction at strength alpha
    s = torch.rand(k).clamp(min=1e-3)                   # non-negative cone coeffs
    w = (s @ U); w = w / w.norm()                       # unit cone vector
    acta = ConeActuator(U=U, mode="add", add_dir=w, alpha=2.0)
    delta = acta.apply(h) - h
    assert torch.allclose(delta, (2.0 * w).expand_as(delta), atol=1e-5), "add wrong"

    # 6) hook is tuple-aware (HF decoder layers return tuples)
    hk = act.hook()
    out_tuple = hk(None, None, (h, "kv"))
    assert isinstance(out_tuple, tuple) and torch.allclose(out_tuple[0], h_abl)
    out_bare = hk(None, None, h)
    assert torch.is_tensor(out_bare) and torch.allclose(out_bare, h_abl)

    # 7) k=1 reduces to constrained directional ablation along one axis
    u1 = orthonormalize(torch.randn(1, d))
    a1 = ConeActuator(U=u1, mode="ablate", u_max=None)
    h1 = a1.apply(h)
    assert torch.allclose(h1, h - (h @ u1.t()) @ u1, atol=1e-5)

    # 8) off mode is identity
    assert torch.allclose(ConeActuator(U=U, mode="off").apply(h), h)

    print("[casa_actuator self-test] OK — "
          "projection-out, ‖u‖-bound, ρ-dial, cone-clip, add, tuple-hook, k=1, off")
