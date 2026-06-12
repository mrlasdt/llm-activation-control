"""Side-by-side TEXT comparison: OAS vs Angular Steering on a chosen model.

Mirrors ../../PTS/verify/compare_pts_vs_angular.py, but the steered conditions are
the OAS controllers (soft-landing LQR + LQG observer) instead of PTS-MPC. Default
model is google/gemma-2-2b-it (the model reputed to resist steering).

The same harmful prompts are continued under SIX conditions. The honest distinction
the verify/ README insists on — there are TWO actuators both loosely called
"angular steering" — is built in explicitly:

  1. baseline                no steering.

  2. Canonical-AS(1L,LN)     the REAL, published Angular Steering (Vu & Nguyen,
                             NeurIPS'25), via the repo's own
                             utils.get_angular_steering_output_hook: it rotates the
                             in-plane component at the `model.layers.{L}.input_layernorm`
                             OUTPUT (the normalised pre-attention activation) of a
                             SINGLE selected layer. This is the faithful "Angular
                             Steering" — and the verify/ runs show it is nearly inert
                             on Gemma (and weak on Qwen). For the full angle sweep use
                             ../../PTS/verify/angular_sweep.py.

  3. Reset-1L(resid)         norm-preserving angle reset at the SAME single layer's
                             RESIDUAL stream (model.layers.{L} output). This is the
                             degenerate R->0 / single-layer corner of OAS — a much
                             stronger actuator than (2), at a different hook point.

  4. Reset-band(resid)       the SAME fixed de-refusal angle held across the OAS band
                             (matched actuation footprint to OAS; = PolicyMultiAngle).

  5. OAS-SoftLanding(band)   the OAS Idea-1 controller: a finite-horizon LQR that
                             distributes small per-layer rotations and lands the angle
                             at the terminal band layer kT (PolicySoftLandingLQR).

  6. OAS-LQG(band)           OAS Ideas 1+2: a Kalman observer filters the per-layer
                             coordinate, the LQR acts on the estimate (PolicyLQG). On a
                             clean readout (no injected noise) this ~= condition 5.

Conditions 3-6 all reset the RESIDUAL stream and steer toward the SAME target (the
harmless-mean / de-refusal direction); only the per-layer angle SELECTION differs
(fixed-1L / fixed-band / soft-landing-LQR / LQG). Condition 2 is the canonical method
at its own (weaker, layernorm) hook point. We print the first-token
refusal/compliance margin per condition as a quantitative readout alongside the text.

⚠️  Labelling, per ../../PTS/verify/README.md: "Angular Steering" proper = condition 2.
Conditions 3/4 are a *residual-stream rotation reset*, NOT canonical Angular Steering;
they are included so OAS is compared against a like-for-like (same-actuator) fixed-angle
control as well as against the published method.

Run:  python compare_oas_vs_angular.py --model google/gemma-2-2b-it
      python compare_oas_vs_angular.py --model Qwen/Qwen2.5-3B-Instruct   # contrast
"""

import argparse
import gc
import math
import sys
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# this file lives in research_proposals/OAS/verify/ ; expose the shared lib, the OAS
# modules (oas_controller/oas_observer/oas_lqr), and the PTS dir (pts_dynamics).
_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "pytorch_pure"))          # shared lib
sys.path.insert(0, str(_HERE.parents[1]))                           # OAS modules (oas_*)
sys.path.insert(0, str(_HERE.parents[2] / "PTS"))                   # pts_dynamics

from utils import (get_input_data, tokenize_instructions_fn, add_hooks,
                   get_angular_steering_output_hook)
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn
from pts_dynamics import fit_layer_dynamics, validate_dynamics
from oas_observer import estimate_process_noise
from oas_controller import (OASState, attach_oas_hooks, build_softlanding,
                            PolicyDeadbeat, PolicyMultiAngle,
                            PolicySoftLandingLQR, PolicyLQG)


def circular_mean(angles: np.ndarray) -> float:
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--plane-samples", type=int, default=192)
    ap.add_argument("--n-fit", type=int, default=192)
    ap.add_argument("--n-obs", type=int, default=24, help="harmful prompts for the margin")
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--angle", type=float, default=None,
                    help="de-refusal target angle in degrees (default: harmless-dir angle)")
    ap.add_argument("--kt", type=int, default=None,
                    help="OAS terminal layer (default: enforcement-sweep argmin = the "
                         "single-layer that most de-refuses)")
    ap.add_argument("--adaptive-mode", type=int, default=0, choices=[0, 1],
                    help="canonical Angular Steering mask: 0=always steer, 1=adaptive")
    ap.add_argument("--rho", type=float, default=0.05,
                    help="OAS soft-landing effort weight R = rho*I")
    ap.add_argument("--q-term", type=float, default=50.0,
                    help="OAS soft-landing terminal state weight Q_term")
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

    # ---- steering plane (shared by every condition) ----
    ha = extract_all_layer_activations(model, harmful_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    la = extract_all_layer_activations(model, harmless_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    plane = compute_steering_plane(ha, la)
    b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
    STEER_LAYER = int(plane["selected_key"].split("_")[1])
    del ha, la; gc.collect(); torch.cuda.empty_cache()
    print(f"plane: {plane['selected_key']} -> steer layer {STEER_LAYER}; b1.b2={(b1@b2).item():.1e}")
    margin_fn, R_ids, C_ids = make_margin_fn(tokenizer, device)

    # ---- residual-stream coords -> 2x2 affine plant + band + de-refusal target ----
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
    all_c = np.concatenate([hc, lc])
    fit = fit_layer_dynamics(all_c[..., 0], all_c[..., 1], affine=True)
    A, bvec = fit["A"], fit["b"]
    val = validate_dynamics(all_c[..., 0], all_c[..., 1], fit, horizons=(1, 5))
    W_pool, _W_per = estimate_process_noise(A, bvec, all_c)

    harmful_mean, harmless_mean = hc.mean(0), lc.mean(0)
    ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
    sep = np.abs(np.arctan2(np.sin(ang(harmful_mean) - ang(harmless_mean)),
                            np.cos(ang(harmful_mean) - ang(harmless_mean))))
    peak = int(np.argmax(sep)); hi = sep > 0.5
    lo = peak; top = peak
    while lo - 1 >= 0 and hi[lo - 1]: lo -= 1
    while top + 1 < L - 1 and hi[top + 1]: top += 1
    band = list(range(lo, top + 1))
    if len(band) < 2:                                 # guard: keep a steerable band
        band = list(range(max(0, STEER_LAYER - 1), min(L - 1, STEER_LAYER + 2)))
    ref_scale = float(np.linalg.norm(harmless_mean[band], axis=-1).mean())

    # the single best fixed de-refusal angle = circular mean of the harmless-ref band
    # angle; the terminal target = a point at that angle with the natural magnitude.
    if args.angle is None:
        target_ang = circular_mean(ang(harmless_mean[band]))
    else:
        target_ang = math.radians(args.angle)
    target = ref_scale * np.array([math.cos(target_ang), math.sin(target_ang)])
    rot_band = float(np.mean([np.linalg.norm(A[k] - np.eye(2), 2) for k in band]))
    print(f"dynamics R2(1step)={val['r2_h1']:.4f} R2(5step)={val['r2_h5']:.4f}")
    print(f"geometric band {band[0]}..{band[-1]} ({len(band)} layers); ref_scale~{ref_scale:.2f}; "
          f"de-refusal angle={np.degrees(target_ang):.0f}deg; ||A_k-I||~{rot_band:.2f}")

    # low-level: mean first-token margin under an explicit hook list
    def margin_with_hooks(prompts, hooks):
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
        return margin_fn(out.logits[:, -1, :]).cpu().numpy()

    # ---- ENFORCEMENT-LAYER SWEEP (OAS Exp 2): single-layer residual deadbeat reset to
    # ---- the de-refusal angle at each band layer; the strongest de-refusal (lowest
    # ---- margin) is the behavioural sweet spot, and we land the OAS controller there
    # ---- instead of at the geometric band-end (which the OAS prototype showed is too late).
    enf_prompts = harmful_test[:args.n_obs]
    base_enf = float(margin_with_hooks(enf_prompts, []).mean())
    print(f"\nenforcement-layer sweep (single-layer residual deadbeat @ {np.degrees(target_ang):.0f}deg; "
          f"baseline margin {base_enf:+.2f}, lower = more de-refused):")
    enf = {}
    for ly in band:
        st = OASState(PolicyDeadbeat(target_ang, ly))
        m = float(margin_with_hooks(enf_prompts, attach_oas_hooks(module_dict, [ly], b1, b2, st)).mean())
        enf[ly] = m
        print(f"    enforce@layer {ly:2d}: margin={m:+6.2f}")
    kT = int(args.kt) if args.kt is not None else min(enf, key=enf.get)   # strongest de-refusal
    band_sl = list(range(band[0], kT + 1))               # actuate band[0]..kT, land at kT
    print(f"  -> kT (enforcement sweet spot) = layer {kT} "
          f"(single-layer margin {enf.get(kT, float('nan')):+.2f}); "
          f"OAS soft-landing band {band_sl[0]}..{band_sl[-1]} ({len(band_sl)} layers)")

    # ---- OAS offline solve (once): soft-landing gains over band[0]..kT, landing at kT ----
    sl_gains = build_softlanding(A, bvec, band_sl, ref=None, target=target,
                                 R_rho=args.rho, Q_term=args.q_term, Q_stage=0.0)
    # measurement-noise guess for the observer ~ process noise (clean real readout =>
    # LQG ~= soft-landing; the observer's value shows up only under injected noise,
    # which the prototype's Exp 4 tests — here it confirms LQG doesn't *hurt*).
    kcfg = {"A": A, "b": bvec, "H": np.eye(2), "W": W_pool, "V": W_pool, "P0": np.eye(2)}

    # canonical Angular Steering config (its own input_layernorm hook point, plane layer)
    ln_name = f"model.layers.{STEER_LAYER}.input_layernorm"
    steering_config = {"first_direction": plane["b1"].cpu().numpy(),
                       "second_direction": plane["b2"].cpu().numpy()}
    target_deg = float(np.degrees(target_ang))

    # ---- per-condition hook builders (fresh state each call; LQG self-resets per
    # ---- forward pass via its band[0] self-guard, exactly as the prototype relies on) ----
    def hooks_for(label):
        if label == "baseline":
            return []
        if label == "Canonical-AS(1L,LN)":
            hook = get_angular_steering_output_hook(steering_config, target_deg, args.adaptive_mode)
            return [(module_dict[ln_name], hook)]
        if label == "Reset-1L@kT(resid)":
            st = OASState(PolicyDeadbeat(target_ang, kT))
            return attach_oas_hooks(module_dict, [kT], b1, b2, st)
        if label == "Reset-band→kT(resid)":
            st = OASState(PolicyMultiAngle(target_ang, band_sl))
            return attach_oas_hooks(module_dict, band_sl, b1, b2, st)
        if label == "OAS-SoftLand(→kT)":
            st = OASState(PolicySoftLandingLQR(sl_gains, ref=None))
            return attach_oas_hooks(module_dict, band_sl, b1, b2, st)
        if label == "OAS-LQG(→kT)":
            st = OASState(PolicyLQG(sl_gains, ref=None, kalman_cfg=kcfg))
            return attach_oas_hooks(module_dict, band_sl, b1, b2, st)
        raise ValueError(label)

    labels = ["baseline", "Canonical-AS(1L,LN)", "Reset-1L@kT(resid)", "Reset-band→kT(resid)",
              "OAS-SoftLand(→kT)", "OAS-LQG(→kT)"]

    def margin(prompts, label):
        hooks = hooks_for(label)
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
        return margin_fn(out.logits[:, -1, :]).cpu().numpy()

    def generate(prompts, label):
        hooks = hooks_for(label)
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                g = model.generate(ids, attention_mask=attn,
                                   max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)

    # ---- quantitative readout: first-token refusal/compliance margin ----
    hp = harmful_test[:args.n_obs]
    print("\n" + "=" * 78)
    print(f"first-token refusal/compliance margin (high=refuse, low=comply), "
          f"mean over {len(hp)} harmful prompts:")
    margins = {}
    base_m = None
    for label in labels:
        m = margin(hp, label)
        margins[label] = float(m.mean())
        if label == "baseline":
            base_m = margins[label]
        delta = "" if base_m is None else f"  (Δ vs baseline {margins[label]-base_m:+.2f})"
        print(f"  {label:22s} margin={m.mean():+6.2f}{delta}")

    # ---- qualitative text ----
    gen_prompts = harmful_test[:args.n_gen]
    print("\n" + "=" * 78)
    print("GENERATIONS (greedy):")
    texts = {label: generate(gen_prompts, label) for label in labels}
    for i, p in enumerate(gen_prompts):
        print(f"\n### prompt {i+1}: {p}")
        for label in labels:
            t = texts[label][i].replace("\n", " ").strip()
            print(f"  [{label:22s}] {t[:240]}")

    # ---- save transcript (filename tagged with kT so the band-end run is preserved) ----
    out = _HERE.parent / "outputs" / f"compare_oas_vs_angular_{name}_kT{kT}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"model={args.model}  steer_layer={STEER_LAYER}  geometric_band={band[0]}..{band[-1]}  "
                f"kT(enforcement sweet spot)={kT}  OAS_band={band_sl[0]}..{band_sl[-1]}  "
                f"de-refusal angle={target_deg:.0f}deg  rho={args.rho}  Q_term={args.q_term}  "
                f"adaptive_mode={args.adaptive_mode}\n")
        f.write(f"dynamics R2(1step)={val['r2_h1']:.4f} R2(5step)={val['r2_h5']:.4f}; "
                f"||A_k-I||(band)~{rot_band:.2f}\n")
        f.write("enforcement sweep (single-layer residual deadbeat margin per band layer): "
                + "  ".join(f"L{ly}={mv:+.2f}" for ly, mv in enf.items()) + "\n")
        f.write("NOTE: 'Canonical-AS(1L,LN)' = published Angular Steering (input_layernorm, 1 layer). "
                "'Reset-*'/'OAS-*' rewrite the RESIDUAL stream (a different, stronger actuator). "
                "kT is chosen by the enforcement sweep; OAS actuates band[0]..kT and lands at kT.\n")
        f.write("margins: " + "  ".join(f"{k}={v:+.2f}" for k, v in margins.items()) + "\n\n")
        for i, p in enumerate(gen_prompts):
            f.write(f"### prompt {i+1}: {p}\n")
            for label in labels:
                f.write(f"  [{label}] {texts[label][i].strip()}\n")
            f.write("\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
