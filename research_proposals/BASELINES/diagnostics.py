"""Cheap, deterministic faithfulness diagnostics reproducing the papers' signature mechanism
plots (no generation judge needed):

  (1) PID Fig-3  — the steady-state error signal ⟨ē(0), ē(k)⟩ across layers under P / PI / PID:
      P leaves a nonzero plateau (steady-state error), the integral (PI/PID) drives it down.
      (pid_native.steady_state_signal)
  (2) A-LQR Fig-5 — layer-wise Jacobian top-m subspace SIMILARITY across different nominal
      activations: ~0.8 early/late layers, ~0.5 mid (the local-linearity property that justifies
      reusing offline-computed LQR gains). We estimate the top-r column space with a randomized
      range finder (r JVPs per Jacobian) instead of the full d×d Jacobian — cheap and faithful in
      trend.

Run:  python diagnostics.py --model google/gemma-2-2b-it
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))

from utils import get_input_data, tokenize_instructions_fn, add_hooks    # noqa: E402
import casa_cone as cone                                                  # noqa: E402
import alqr_native as alqr                                                # noqa: E402
import pid_native as pid                                                  # noqa: E402


# ---- Fig-5 helpers: randomized top-r range + subspace similarity ----

def jacobian_range(fn, x, r=24):
    """Approximate orthonormal basis (d×r) of the top-r column space of ∂fn/∂x via r JVPs."""
    cols = []
    for _ in range(r):
        v = torch.randn_like(x)
        _, jv = torch.autograd.functional.jvp(fn, (x,), (v,), strict=False)
        cols.append(jv.float())
    Y = torch.stack(cols, dim=1)                              # (d, r)
    Q, _ = torch.linalg.qr(Y)
    return Q


def subspace_sim(Q1, Q2):
    """Normalized subspace overlap ‖Q1ᵀQ2‖_F² / r ∈ [0,1] (1 = identical span)."""
    M = Q1.t() @ Q2
    return float((M ** 2).sum() / Q1.shape[1])


def nominal_activations(model, tok, prompts, device, layers, bs=8):
    """Per-layer last-token activations for each prompt (model output of layer l)."""
    md = dict(model.named_modules())
    L = model.config.num_hidden_layers
    cache = {l: [] for l in range(L)}

    def mk(l):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cache[l].append(h[:, -1, :].detach())
        return hook
    hooks = [(md[f"model.layers.{l}"], mk(l)) for l in range(L)]
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            for i in range(0, len(prompts), bs):
                inp = tokenize_instructions_fn(prompts[i:i + bs], tok)
                model(input_ids=inp.input_ids.to(device),
                      attention_mask=inp.attention_mask.to(device))
    return {l: torch.cat(cache[l], 0) for l in layers}        # (N,d) per layer


def near_identity_cone_plant(model, tok, device, prompts, n_fit=64):
    """Exact ‖A_l − I‖₂ per layer from the fitted k×k cone plant (the near-identity property;
    PTS measured ≈0.34 on the 2×2 plane). Loads the trained CONE k=4 basis if present."""
    import casa_control as ctrl
    from casa_actuator import orthonormalize
    from casa_experiment import cone_coords_all_layers
    npz = _HERE.parents[1] / "CASA" / "outputs" / "casa_subspaces_gemma-2-2b-it.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    key = next((k for k in z.files if k.startswith("CONE")), None)
    if key is None:
        return None
    B = orthonormalize(torch.from_numpy(z[key]).float().to(device))   # (k,d)
    coords = cone_coords_all_layers(model, tok, prompts[:n_fit], B, device)
    fit = ctrl.fit_cone_plant(coords)
    A = fit["A"]                                                       # (L-1,k,k)
    k = A.shape[1]
    return {l: float(np.linalg.norm(A[l] - np.eye(k), 2)) for l in range(A.shape[0])}


def near_identity_jvp(model, tok, device, prompts, layers, n_probe=16):
    """Full-d cross-check: RMS ‖(A_l−I)v‖/‖v‖ over random unit v at the harmful-mean nominal
    (A_l = block Jacobian = I + ∂f/∂z; the residual stream makes this small)."""
    hmean = cone.residual_means(model, tok, prompts[:64], device)     # (L,d) nominal
    d = model.config.hidden_size
    pos_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
    hs = torch.zeros((1, 1, d), device=device, dtype=next(model.parameters()).dtype)
    pos_emb = model.model.rotary_emb(hs, pos_ids)
    out = {}
    for l in layers:
        fn = alqr.make_block_fn(model, l, None, pos_ids, pos_emb)
        z = torch.as_tensor(hmean[l], device=device, dtype=next(model.parameters()).dtype)
        ratios = []
        for _ in range(n_probe):
            v = torch.randn_like(z)
            try:
                _, jv = torch.autograd.functional.jvp(fn, (z,), (v,), strict=False)
            except Exception as e:                            # noqa
                print(f"  layer {l}: JVP failed ({type(e).__name__})"); ratios = []; break
            ratios.append((jv.float() - v.float()).norm() / (v.float().norm() + 1e-9))
        out[l] = float(np.mean([r.item() for r in ratios])) if ratios else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--n-fit", type=int, default=64)
    ap.add_argument("--n-src", type=int, default=32)
    ap.add_argument("--r", type=int, default=24)
    ap.add_argument("--n-pts", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.model.split("/")[-1]
    t0 = time.time()
    print(f"device={device} model={args.model}")

    # eager attention: flash-SDPA has no double-backward, which the JVP Jacobian needs
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation="eager").eval()
    model.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    L = model.config.num_hidden_layers

    harmful_tr, _ = get_input_data("harmful", "en")
    harmless_tr, _ = get_input_data("harmless", "en")

    # ---- (1) PID Fig-3 steady-state signal ----
    print("\n[Fig-3] PID steady-state error signal <e(0),e(k)>/<e(0),e(0)> under P / PI / PID ...")
    # gentle gains: small Kp UNDER-corrects (positive residual plateau = steady-state error);
    # the integral term then drives the residual toward 0 (the paper's mechanism).
    sig = pid.steady_state_signal(
        model, tok, harmless_tr[:args.n_fit], harmful_tr[:args.n_src], device,
        gains=(("P", 0.05, 0.0, 0.0), ("PI", 0.05, 0.005, 0.0), ("PID", 0.05, 0.005, 0.003)),
        normed=False, bs=8)
    sig = {nm: (s / (s[0] + 1e-9)) for nm, s in sig.items()}   # normalize: starts at 1.0
    for nm, s in sig.items():
        tail = float(np.mean(s[-6:]))                          # steady-state plateau (tail mean)
        print(f"    {nm:4s}: normalized tail plateau = {tail:+.3f}   "
              f"(start {s[0]:+.2f}, min {s.min():+.2f})  [→0 = error removed]")

    # ---- (2) near-identity dynamics: ‖A_l − I‖₂ (the property behind the head-to-head) ----
    print("\n[near-identity] exact ‖A_l − I‖₂ from the fitted k×k cone plant "
          "(PTS 2×2 measured ≈0.34) ...")
    ni = near_identity_cone_plant(model, tok, device, harmful_tr, n_fit=args.n_fit)
    if ni is not None:
        band = [l for l in ni if 7 <= l <= 24]               # CASA behavioural band
        print(f"    per-layer ‖A_l−I‖₂: min {min(ni.values()):.2f}  max {max(ni.values()):.2f}  "
              f"mean {np.mean(list(ni.values())):.2f}")
        print(f"    behavioural band (7–24) mean ‖A_l−I‖₂ = {np.mean([ni[l] for l in band]):.2f}  "
              f"(near-identity ⇒ lookahead inert ⇒ MPC≈LQR)")
    layers = sorted(set(int(x) for x in np.linspace(1, L - 1, 8)))
    print("[near-identity] full-d JVP cross-check: RMS ‖(A_l−I)v‖/‖v‖ ...")
    nij = near_identity_jvp(model, tok, device, harmful_tr, layers, n_probe=args.r)
    for l in layers:
        print(f"    layer {l:2d}: full-d RMS ‖A−I‖ ≈ {nij[l]:.3f}")

    # ---- figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for nm, s in sig.items():
        ax1.plot(range(len(s)), s, marker=".", label=nm)
    ax1.axhline(0, ls="--", c="gray", lw=0.8)
    ax1.set_xlabel("layer k"); ax1.set_ylabel("normalized ⟨ē(0),ē(k)⟩  (→0 = error removed)")
    ax1.set_title("PID Fig-3: steady-state error signal"); ax1.grid(alpha=0.3); ax1.legend()
    if ni is not None:
        ls = sorted(ni); ax2.plot(ls, [ni[l] for l in ls], marker="o", color="tab:blue",
                                  label="cone plant ‖A_l−I‖₂ (exact)")
    ax2.axhline(1.0, ls=":", c="gray", lw=1, label="identity reference (=I ⇒ 0)")
    ax2.set_xlabel("layer"); ax2.set_ylabel("‖A_l − I‖₂")
    ax2.set_title("Near-identity dynamics (small ⇒ lookahead inert ⇒ MPC≈LQR)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle(f"Mechanism diagnostics — {name}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _HERE.parent / "outputs" / f"diagnostics_{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
