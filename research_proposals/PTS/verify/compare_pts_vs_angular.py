"""Side-by-side TEXT comparison: PTS vs Angular Steering on a chosen model.

Generates continuations for the same harmful prompts under four conditions, all
sharing the verified norm-preserving SO(2) reset actuator and the same 2D steering
plane, so the only thing that varies is HOW the per-layer angle is chosen:

  1. baseline                no steering
  2. Reset-1L (residual)     norm-preserving angle reset at the single selected layer's
                             RESIDUAL stream (model.layers.{L} output). NOTE: this is
                             NOT canonical Angular Steering — the published method hooks
                             input_layernorm and is far weaker (see verify/angular_sweep.py).
  3. Reset-band (residual)   the SAME fixed-angle residual reset at every band layer
                             (matched actuation footprint to PTS)
  4. PTS-MPC (resid band)    per-layer ADAPTIVE angle from the MPC tracking the harmless
                             reference trajectory (Predictive Trajectory Steering)

  All four reset the RESIDUAL stream; only the angle-selection differs (none / fixed-1L /
  fixed-band / adaptive-band). Canonical Angular Steering (input_layernorm, single layer)
  is a separate, much weaker actuator — test it with verify/angular_sweep.py.

All steer toward the same target (the harmless-mean direction = the de-refusal
direction, where harmful prompts have headroom). We also print the first-token
refusal/compliance margin per condition as a quantitative readout.

Run:  python pts_vs_angular.py --model google/gemma-2-2b-it
      python pts_vs_angular.py --model Qwen/Qwen2.5-3B-Instruct   # smoke test
"""

import argparse
import gc
import math
import sys
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# this file lives in research_proposals/PTS/verify/ ; add the shared lib and the
# PTS module dir (pts_dynamics/pts_mpc/pts_controller) to the path.
_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "pytorch_pure"))   # shared lib
sys.path.insert(0, str(_HERE.parents[1]))                    # PTS modules

from utils import get_input_data, tokenize_instructions_fn, add_hooks
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn
from pts_dynamics import fit_layer_dynamics, validate_dynamics
from pts_mpc import MPCController, reference_trajectory
from pts_controller import (PTSState, PolicyMPC, PolicyFixedAngle, attach_pts_hooks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--plane-samples", type=int, default=192)
    ap.add_argument("--n-fit", type=int, default=192)
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--u-max-frac", type=float, default=0.5,
                    help="PTS budget as a fraction of the band reference scale")
    ap.add_argument("--angle", type=float, default=None,
                    help="Angular-Steering angle in degrees (default: harmless-dir angle)")
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.model.split("/")[-1]
    print(f"device={device}  model={args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", torch_dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    module_dict = dict(model.named_modules())
    L = model.config.num_hidden_layers
    print(f"loaded: {L} layers")

    harmful_train, harmful_test = get_input_data("harmful", "en")
    harmless_train, harmless_test = get_input_data("harmless", "en")

    # ---- steering plane (shared) ----
    ha = extract_all_layer_activations(model, harmful_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    la = extract_all_layer_activations(model, harmless_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    plane = compute_steering_plane(ha, la)
    b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
    STEER_LAYER = int(plane["selected_key"].split("_")[1])
    del ha, la; gc.collect(); torch.cuda.empty_cache()
    print(f"plane: {plane['selected_key']} -> steer layer {STEER_LAYER}; b1.b2={(b1@b2).item():.1e}")
    margin_fn, R, C = make_margin_fn(tokenizer, device)

    # ---- residual-stream coords -> dynamics + band + reference ----
    def residual_coords(prompts, bs=8):
        cache = {k: [] for k in range(L)}

        def mk(k):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                last = h[:, -1, :]
                _b1 = b1.to(last.dtype); _b2 = b2.to(last.dtype)
                cache[k].append(torch.stack([(last @ _b1).float().cpu(),
                                             (last @ _b2).float().cpu()], -1))
            return hook
        hooks = [(module_dict[f"model.layers.{k}"], mk(k)) for k in range(L)]
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                for i in range(0, len(prompts), bs):
                    inp = tokenize_instructions_fn(prompts[i:i + bs], tokenizer)
                    model(input_ids=inp.input_ids.to(device),
                          attention_mask=inp.attention_mask.to(device))
        return np.stack([torch.cat(cache[k], 0).numpy() for k in range(L)], axis=1)

    print("fitting 2x2 dynamics on the residual stream...")
    hc = residual_coords(harmful_train[:args.n_fit]); gc.collect(); torch.cuda.empty_cache()
    lc = residual_coords(harmless_train[:args.n_fit]); gc.collect(); torch.cuda.empty_cache()
    fit = fit_layer_dynamics(np.concatenate([hc, lc])[..., 0],
                             np.concatenate([hc, lc])[..., 1], affine=True)
    A, bvec = fit["A"], fit["b"]
    val = validate_dynamics(np.concatenate([hc, lc])[..., 0],
                            np.concatenate([hc, lc])[..., 1], fit, horizons=(1, 5))
    harmful_mean, harmless_mean = hc.mean(0), lc.mean(0)
    ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
    sep = np.abs(np.arctan2(np.sin(ang(harmful_mean) - ang(harmless_mean)),
                            np.cos(ang(harmful_mean) - ang(harmless_mean))))
    peak = int(np.argmax(sep)); hi = sep > 0.5
    lo = peak; top = peak
    while lo - 1 >= 0 and hi[lo - 1]: lo -= 1
    while top + 1 < L - 1 and hi[top + 1]: top += 1
    band = list(range(lo, top + 1))
    ref = reference_trajectory(harmful_mean, harmless_mean, option="A")
    ref_scale = float(np.linalg.norm(harmless_mean[band], axis=-1).mean())
    if args.angle is None:
        fixed_angle = float(np.arctan2(np.sin(ang(harmless_mean[band])).mean(),
                                       np.cos(ang(harmless_mean[band])).mean()))
    else:
        fixed_angle = math.radians(args.angle)
    u_max = args.u_max_frac * ref_scale
    print(f"dynamics R2(1step)={val['r2_h1']:.4f} R2(5step)={val['r2_h5']:.4f}")
    print(f"band {band[0]}..{band[-1]} ({len(band)} layers); ref_scale~{ref_scale:.2f}; "
          f"u_max={u_max:.2f}; Angular angle={np.degrees(fixed_angle):.0f}deg")

    # ---- generation under each policy ----
    def generate(prompts, policy, layers):
        hooks = (attach_pts_hooks(module_dict, layers, b1, b2, PTSState(policy))
                 if policy is not None else [])
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                g = model.generate(ids, attention_mask=attn,
                                   max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)

    def margin(prompts, policy, layers):
        hooks = (attach_pts_hooks(module_dict, layers, b1, b2, PTSState(policy))
                 if policy is not None else [])
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
        return margin_fn(out.logits[:, -1, :]).cpu().numpy()

    mpc = MPCController(A, bvec, ref, layers=band, H=args.horizon, q_pos=1.0,
                        r_ctrl=0.02, qf_scale=4.0, u_max=u_max)
    conditions = [
        ("baseline",            None,                                         []),
        ("Reset-1L(resid)",     PolicyFixedAngle(fixed_angle, [STEER_LAYER]), [STEER_LAYER]),
        ("Reset-band(resid)",   PolicyFixedAngle(fixed_angle, band),          band),
        ("PTS-MPC(resid-band)", PolicyMPC(mpc),                               band),
    ]

    gen_prompts = harmful_test[:args.n_gen]
    print("\n" + "=" * 78)
    print(f"first-token refusal/compliance margin (high=refuse, low=comply), "
          f"mean over {len(harmful_test[:24])} harmful prompts:")
    margins = {}
    for label, pol, layers in conditions:
        m = margin(harmful_test[:24], pol, layers)
        margins[label] = float(m.mean())
        print(f"  {label:18s} margin={m.mean():+6.2f}")

    print("\n" + "=" * 78)
    print("GENERATIONS (greedy):")
    texts = {label: generate(gen_prompts, pol, layers) for label, pol, layers in conditions}
    for i, p in enumerate(gen_prompts):
        print(f"\n### prompt {i+1}: {p}")
        for label, _, _ in conditions:
            t = texts[label][i].replace("\n", " ").strip()
            print(f"  [{label:16s}] {t[:240]}")

    # ---- save ----
    out = pathlib.Path(__file__).resolve().parent / f"pts_vs_angular_{name}.txt"
    with open(out, "w") as f:
        f.write(f"model={args.model}  steer_layer={STEER_LAYER}  band={band[0]}..{band[-1]}  "
                f"angle={np.degrees(fixed_angle):.0f}deg  u_max={u_max:.2f}\n")
        f.write(f"dynamics R2(1step)={val['r2_h1']:.4f}\n")
        f.write("margins: " + "  ".join(f"{k}={v:+.2f}" for k, v in margins.items()) + "\n\n")
        for i, p in enumerate(gen_prompts):
            f.write(f"### prompt {i+1}: {p}\n")
            for label, _, _ in conditions:
                f.write(f"  [{label}] {texts[label][i].strip()}\n")
            f.write("\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
