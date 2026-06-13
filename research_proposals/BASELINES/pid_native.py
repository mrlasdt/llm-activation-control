"""Native PID Steering (Nguyen et al., "Activation Steering with a Feedback Controller").

Faithful, model-in-the-loop reimplementation of the PID-AcT / PID-jailbreak control law on
HF models, cross-checked against the authors' code (see BASELINES_PROVENANCE.md §3):

  * error signal  r(k) = mean_target(k) − mean_source(k)   (per-layer diff-in-means; the
    jailbreak variant normalizes activations before averaging — `normed=True`).
  * PID over the LAYER axis (precomputed steering vectors):
        u(k) = Kp·r(k) + Ki·Σ_{j≤k} r(j) + Kd·(r(k) − r(k−1))            (Eqn 17, Lemma 1)
    integral = cumsum over layers, derivative = Δ over layers; existing methods (ActAdd /
    DirAblate / Mean-AcT) are the Ki=Kd=0, Kp=1 pure-P special case.
  * actuator: ActAdd  ρ(h,u)=h+α·u   or  DirAblate  ρ(h,u)=h−ûûᵀh.
  * gains (from the authors' repos): jailbreak `Kp=1.0, Ki=0.3, Kd=0.01` (normed dirs →
    directional ablation); toxicity `Kp=1, Ki≈0.005–0.056, Kd≈0.01` (additive Mean-AcT).

The Fig-3 diagnostic (`steady_state_signal`) reproduces the paper's headline mechanism plot:
the alignment ⟨ē(0), ē(k)⟩ across layers under P / PI / PID — P plateaus (steady-state error),
PI drives it to zero (with overshoot), PID damps the overshoot.

Self-test (synthetic, no model):  python pid_native.py
Model-in-the-loop entry points are `diff_in_means`, `PIDSteer`, `steady_state_signal`.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))   # shared lib
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))           # casa_cone.residual_means

# (imported lazily inside model-using functions so the synthetic self-test needs no model libs)


# =============================================================================
# PID over the layer axis  (precomputed steering vectors)
# =============================================================================


def pid_vectors(r, kp=1.0, ki=0.0, kd=0.0):
    """Turn the per-layer error signal r:(L,d) into PID steering vectors u:(L,d).

    u(k) = Kp·r(k) + Ki·Σ_{j≤k} r(j) + Kd·(r(k) − r(k−1)),  with r(-1):=0 (so the derivative
    at k=0 is r(0)). Integral is the inclusive cumulative sum (matches the authors' cumsum;
    note the harness rolls r(1)=0 at the embedding layer). Pure-P (ki=kd=0) == diff-in-means."""
    r = np.asarray(r, np.float64)
    integral = np.cumsum(r, axis=0)
    deriv = np.empty_like(r)
    deriv[0] = r[0]
    deriv[1:] = r[1:] - r[:-1]
    return kp * r + ki * integral + kd * deriv


class PIDSteer:
    """Deploy precomputed PID steering vectors u(k) at each layer's residual-stream output.

    actuator='actadd'   : h += alpha · u(k)            (ActAdd; toxicity / additive jailbreak)
    actuator='ablate'   : h -= (h·û)û  with û=u/‖u‖     (DirAblate; the refusal-removal actuator)
    layers: which layer indices to steer (default all that have a vector).
    """

    def __init__(self, u, layers=None, actuator="actadd", alpha=1.0):
        self.u = np.asarray(u, np.float64)                    # (L,d)
        self.L = self.u.shape[0]
        self.layers = set(range(self.L) if layers is None else layers)
        self.actuator = actuator
        self.alpha = float(alpha)

    @staticmethod
    def apply_actuator(h, u, actuator="actadd", alpha=1.0):
        """ρ_steer(h,u): ActAdd h+α·u, or DirAblate h−(h·û)û. h:(...,d), u:(d,)."""
        if actuator == "actadd":
            return h + alpha * u
        un = u / (u.norm() + 1e-8)
        return h - (h @ un).unsqueeze(-1) * un

    def hooks(self, module_dict):
        ut = torch.from_numpy(self.u)

        def mk(layer):
            uvec = ut[layer]

            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if layer not in self.layers:
                    return out
                steered = self.apply_actuator(h, uvec.to(h.device, h.dtype),
                                              self.actuator, self.alpha)
                return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered
            return hook

        return [(module_dict[f"model.layers.{j}"], mk(j)) for j in sorted(self.layers)]


# =============================================================================
# offline diff-in-means error signal  (model-in-the-loop)
# =============================================================================


def diff_in_means(model, tokenizer, target_prompts, source_prompts, device, normed=False, bs=8):
    """r(k) = mean_target(k) − mean_source(k) per layer at the last token.

    target = desired/benign side (harmless, non-toxic); source = undesired (harmful, toxic).
    normed=True normalizes each activation to unit norm before averaging (the jailbreak variant,
    llama_many_layers.py:510). Returns (L,d) numpy."""
    import casa_cone as cone
    if not normed:
        mt = cone.residual_means(model, tokenizer, target_prompts, device, bs=bs)
        ms = cone.residual_means(model, tokenizer, source_prompts, device, bs=bs)
        return mt - ms
    mt = _normed_means(model, tokenizer, target_prompts, device, bs)
    ms = _normed_means(model, tokenizer, source_prompts, device, bs)
    return mt - ms


def _normed_means(model, tokenizer, prompts, device, bs=8):
    from utils import tokenize_instructions_fn, add_hooks
    L = model.config.num_hidden_layers
    acc = {l: [] for l in range(L)}
    md = dict(model.named_modules())

    def mk(l):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = h[:, -1, :].float()
            acc[l].append((v / (v.norm(dim=-1, keepdim=True) + 1e-8)).cpu())
        return hook

    hooks = [(md[f"model.layers.{l}"], mk(l)) for l in range(L)]
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                inp = tokenize_instructions_fn(prompts[i:i + bs], tokenizer)
                model(input_ids=inp.input_ids.to(device),
                      attention_mask=inp.attention_mask.to(device))
    return np.stack([torch.cat(acc[l], 0).mean(0).numpy() for l in range(L)], 0)


# =============================================================================
# Fig-3 diagnostic: steady-state error signal ⟨ē(0), ē(k)⟩ under P / PI / PID
# =============================================================================


def steady_state_signal(model, tokenizer, target_prompts, source_prompts, device,
                        r=None, gains=(("P", 1, 0, 0), ("PI", 1, 0.1, 0), ("PID", 1, 0.1, 0.02)),
                        normed=False, bs=8):
    """Reproduce the paper's Fig-3 mechanism plot. For each gain triple, steer the SOURCE
    prompts with the PID vectors and measure, per layer, the alignment of the realized mean
    error ē(k)=mean_target(k)−mean_source_steered(k) with the initial error ē(0). A nonzero
    plateau = residual steady-state error (P); →0 = the integral removed the bias (PI/PID).
    Returns {name: signal:(L,)}."""
    import casa_cone as cone
    from utils import tokenize_instructions_fn, add_hooks
    if r is None:
        r = diff_in_means(model, tokenizer, target_prompts, source_prompts, device,
                          normed=normed, bs=bs)
    mt = cone.residual_means(model, tokenizer, target_prompts, device, bs=bs)   # target means (L,d)
    e0 = (mt[0] - cone.residual_means(model, tokenizer, source_prompts, device, bs=bs)[0])
    e0u = e0 / (np.linalg.norm(e0) + 1e-8)
    out = {}
    md = dict(model.named_modules())
    L = model.config.num_hidden_layers
    for name, kp, ki, kd in gains:
        u = pid_vectors(r, kp, ki, kd)
        steer = PIDSteer(u, layers=range(1, L), actuator="actadd", alpha=1.0)
        acc = {l: [] for l in range(L)}

        def mk(l):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                acc[l].append(h[:, -1, :].float().mean(0).cpu())
                return out
            return hook
        cap = [(md[f"model.layers.{l}"], mk(l)) for l in range(L)]
        with add_hooks(module_forward_hooks=steer.hooks(md) + cap):
            with torch.no_grad():
                for i in range(0, len(source_prompts), bs):
                    inp = tokenize_instructions_fn(source_prompts[i:i + bs], tokenizer)
                    model(input_ids=inp.input_ids.to(device),
                          attention_mask=inp.attention_mask.to(device))
        msteer = np.stack([torch.stack(acc[l]).mean(0).numpy() for l in range(L)], 0)
        ek = mt - msteer                                       # realized error per layer (L,d)
        out[name] = ek @ e0u                                   # ⟨ē(k), ê(0)⟩ (L,)
    return out


# =============================================================================
# Self-test (synthetic — no model)
# =============================================================================


def _selftest_recurrence():
    rng = np.random.RandomState(0)
    L, d = 7, 5
    r = rng.randn(L, d)
    kp, ki, kd = 1.3, 0.2, 0.05
    u = pid_vectors(r, kp, ki, kd)
    # manual reference
    ref = np.zeros_like(r); isum = np.zeros(d); rprev = np.zeros(d)
    for k in range(L):
        isum = isum + r[k]
        d_term = r[k] - (rprev if k > 0 else 0.0 * rprev) if k > 0 else r[k]
        ref[k] = kp * r[k] + ki * isum + kd * d_term
        rprev = r[k]
    assert np.allclose(u, ref, atol=1e-10), "PID recurrence mismatch"
    # pure-P == diff-in-means
    assert np.allclose(pid_vectors(r, 1.0, 0.0, 0.0), r), "P-control should equal r"
    print("[pid recurrence] OK — matches Kp·r+Ki·Σr+Kd·Δr; P-only == diff-in-means")


def _selftest_actuator():
    """ActAdd adds α·u; DirAblate removes the u-component (orthogonal residual)."""
    rng = np.random.RandomState(1)
    d = 6
    h = torch.tensor(rng.randn(3, d))
    u = torch.tensor(rng.randn(d))
    # ActAdd
    ha = PIDSteer.apply_actuator(h, u, "actadd", alpha=0.7)
    assert torch.allclose(ha, h + 0.7 * u), "actadd mismatch"
    # DirAblate: result has ~zero component along u
    hb = PIDSteer.apply_actuator(h, u, "ablate")
    un = (u / u.norm())
    comp = hb @ un
    assert torch.allclose(comp, torch.zeros_like(comp), atol=1e-5), \
        f"ablate left a u-component: {comp.abs().max():.2e}"
    print("[pid actuator] OK — ActAdd=h+α·u, DirAblate removes the u-direction")


def _selftest_integral_cumsum():
    """Integral term equals the running cumulative sum (open-loop PID-AcT form)."""
    rng = np.random.RandomState(2)
    r = rng.randn(5, 3)
    u = pid_vectors(r, kp=0.0, ki=1.0, kd=0.0)               # pure integral
    assert np.allclose(u, np.cumsum(r, axis=0)), "integral != cumsum"
    u2 = pid_vectors(r, kp=0.0, ki=0.0, kd=1.0)              # pure derivative
    assert np.allclose(u2[0], r[0]) and np.allclose(u2[1:], r[1:] - r[:-1]), "deriv mismatch"
    print("[pid integral/deriv] OK — integral==cumsum, derivative==Δ over layers")


if __name__ == "__main__":
    _selftest_recurrence()
    _selftest_integral_cumsum()
    _selftest_actuator()
    print("[pid_native self-test] ALL OK — PID recurrence/integral/derivative + actuators "
          "(closed-loop steady-state removal is shown by casa_baselines.ConePID + the model "
          "Fig-3 diagnostic)")
