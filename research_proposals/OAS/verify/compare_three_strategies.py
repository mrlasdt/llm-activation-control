"""Three-way TEXT comparison of rotation-steering STRATEGIES on a chosen model.

All three use the SAME extracted de-refusal direction (the harmless-mean direction
in the best-layer steering plane {b1,b2}) and the SAME norm-preserving SO(2) reset
actuator on the RESIDUAL stream (model.layers.{k} output). They differ ONLY in
*where* and *how aggressively* the rotation is applied across depth:

  0. Canonical-AS(1L)      the PUBLISHED Angular Steering (input_layernorm output, single
                           layer) — included as the faithful baseline; ~inert on Gemma.
  1. ResidualRot(all-L)    hard reset to the extracted direction at EVERY layer, on the
                           RESIDUAL stream. (= the user's definition: "extract the
                           direction from the best layer, steer all layers to it" — but
                           a residual reset, NOT the canonical input_layernorm method.)
  2. Deadbeat(1L@best)     a single hard reset at the BEST layer only.
  3. OAS-SoftLand(->best)  the OAS finite-horizon LQR: rotate GENTLY from an early
     OAS-LQG(->best)       layer, distributing small per-layer rotations that LAND at
                           the extracted direction by the best layer (Idea 1); LQG adds
                           the Kalman observer (Idea 2). On a clean readout LQG ~= LQR.

"Best layer" = the enforcement-sweep sweet spot: the single layer whose deadbeat
reset most de-refuses (lowest first-token refusal margin). Deadbeat slams there; OAS
lands there; AngularSteer ignores it and slams everywhere.

⚠️  Labelling honesty (../../PTS/verify/README.md): "AngularSteer(all-L)" here is a
RESIDUAL-stream reset at all layers — this is the user's working definition, NOT the
*published* canonical Angular Steering (Vu & Nguyen), which resets a SINGLE layer's
`input_layernorm` output and is essentially inert on this repo's models (see
../../PTS/verify/angular_sweep.py and compare_oas_vs_angular.py's Canonical-AS row).

Per condition we print the first-token refusal/compliance margin and greedy
generations, and save a transcript to outputs/.

Run:  python compare_three_strategies.py --model google/gemma-2-2b-it
      python compare_three_strategies.py --model Qwen/Qwen2.5-3B-Instruct   # contrast
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
    ap.add_argument("--best-layer", type=int, default=None,
                    help="best layer for deadbeat slam / OAS landing (default: enforcement argmin)")
    ap.add_argument("--oas-start", type=int, default=None,
                    help="first layer of the OAS soft-landing band (default: geometric band start)")
    ap.add_argument("--no-final-layer", action="store_true",
                    help="exclude the final layer from AngularSteer(all-L) (default: include all)")
    ap.add_argument("--rho", type=float, default=0.05, help="OAS soft-landing effort weight R=rho*I")
    ap.add_argument("--q-term", type=float, default=50.0, help="OAS soft-landing terminal weight")
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

    # ---- steering plane (the best layer's direction, shared by every condition) ----
    ha = extract_all_layer_activations(model, harmful_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    la = extract_all_layer_activations(model, harmless_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    plane = compute_steering_plane(ha, la)
    b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
    STEER_LAYER = int(plane["selected_key"].split("_")[1])
    del ha, la; gc.collect(); torch.cuda.empty_cache()
    print(f"plane: {plane['selected_key']} -> extracted from layer {STEER_LAYER}; b1.b2={(b1@b2).item():.1e}")
    margin_fn, R_ids, C_ids = make_margin_fn(tokenizer, device)

    # ---- residual-stream coords -> 2x2 affine plant + band + de-refusal direction ----
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
    if len(band) < 2:
        band = list(range(max(0, STEER_LAYER - 1), min(L - 1, STEER_LAYER + 2)))
    ref_scale = float(np.linalg.norm(harmless_mean[band], axis=-1).mean())

    # the extracted de-refusal DIRECTION (one angle, used by all three strategies)
    if args.angle is None:
        target_ang = circular_mean(ang(harmless_mean[band]))
    else:
        target_ang = math.radians(args.angle)
    target = ref_scale * np.array([math.cos(target_ang), math.sin(target_ang)])
    target_deg = float(np.degrees(target_ang))
    # canonical (published) Angular Steering hooks the input_layernorm OUTPUT, single
    # layer — the faithful method, included so the inert published baseline sits next
    # to the residual-reset variants (see verify/README labelling caveat).
    ln_name = f"model.layers.{STEER_LAYER}.input_layernorm"
    steering_config = {"first_direction": plane["b1"].cpu().numpy(),
                       "second_direction": plane["b2"].cpu().numpy()}
    print(f"dynamics R2(1step)={val['r2_h1']:.4f} R2(5step)={val['r2_h5']:.4f}; "
          f"geometric band {band[0]}..{band[-1]}; de-refusal angle={target_deg:.0f}deg")

    # ---- low-level margin helper + enforcement-layer sweep -> the BEST layer ----
    def margin_with_hooks(prompts, hooks):
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
        return margin_fn(out.logits[:, -1, :]).cpu().numpy()

    enf_prompts = harmful_test[:args.n_obs]
    base_enf = float(margin_with_hooks(enf_prompts, []).mean())
    print(f"\nenforcement-layer sweep (single-layer deadbeat @ {target_deg:.0f}deg; "
          f"baseline {base_enf:+.2f}, lower = more de-refused):")
    enf = {}
    for ly in band:
        st = OASState(PolicyDeadbeat(target_ang, ly))
        enf[ly] = float(margin_with_hooks(enf_prompts, attach_oas_hooks(module_dict, [ly], b1, b2, st)).mean())
        print(f"    enforce@layer {ly:2d}: margin={enf[ly]:+6.2f}")
    best = int(args.best_layer) if args.best_layer is not None else min(enf, key=enf.get)
    oas_start = int(args.oas_start) if args.oas_start is not None else band[0]
    oas_start = min(oas_start, best - 1) if best > 0 else 0
    oas_band = list(range(oas_start, best + 1))               # early -> best (land at best)
    all_layers = list(range(0, L - 1 if args.no_final_layer else L))
    print(f"  -> BEST layer (enforcement sweet spot) = {best} (margin {enf.get(best, float('nan')):+.2f})")
    print(f"  strategies: Canonical-AS = published input_layernorm @{STEER_LAYER} (1 layer);  "
          f"ResidualRot = residual reset all {len(all_layers)} layers {all_layers[0]}..{all_layers[-1]};  "
          f"Deadbeat = 1 residual reset @{best};  OAS = soft-land {oas_band[0]}..{oas_band[-1]} ({len(oas_band)} layers)")

    # ---- OAS offline solve: soft-landing gains over [oas_start..best], land at best ----
    sl_gains = build_softlanding(A, bvec, oas_band, ref=None, target=target,
                                 R_rho=args.rho, Q_term=args.q_term, Q_stage=0.0)
    kcfg = {"A": A, "b": bvec, "H": np.eye(2), "W": W_pool, "V": W_pool, "P0": np.eye(2)}

    # ---- the three strategies (fresh state per call; LQG self-resets per forward pass) ----
    def hooks_for(label):
        if label == "baseline":
            return []
        if label == "Canonical-AS(1L)":     # the PUBLISHED method: input_layernorm, 1 layer
            return [(module_dict[ln_name],
                     get_angular_steering_output_hook(steering_config, target_deg, 0))]
        if label == "ResidualRot(all-L)":    # residual reset at all layers (the user's "AS")
            return attach_oas_hooks(module_dict, all_layers, b1, b2,
                                    OASState(PolicyMultiAngle(target_ang, all_layers)))
        if label == "Deadbeat(1L@best)":
            return attach_oas_hooks(module_dict, [best], b1, b2,
                                    OASState(PolicyDeadbeat(target_ang, best)))
        if label == "OAS-SoftLand(->best)":
            return attach_oas_hooks(module_dict, oas_band, b1, b2,
                                    OASState(PolicySoftLandingLQR(sl_gains, ref=None)))
        if label == "OAS-LQG(->best)":
            return attach_oas_hooks(module_dict, oas_band, b1, b2,
                                    OASState(PolicyLQG(sl_gains, ref=None, kalman_cfg=kcfg)))
        raise ValueError(label)

    labels = ["baseline", "Canonical-AS(1L)", "ResidualRot(all-L)", "Deadbeat(1L@best)",
              "OAS-SoftLand(->best)", "OAS-LQG(->best)"]

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
        m = margin_with_hooks(hp, hooks_for(label))
        margins[label] = float(m.mean())
        if label == "baseline":
            base_m = margins[label]
        delta = "" if base_m is None else f"  (Δ {margins[label]-base_m:+.2f})"
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

    # ---- save transcript ----
    out = _HERE.parent / "outputs" / f"compare_three_strategies_{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"model={args.model}  plane_layer={STEER_LAYER}  best_layer={best}  "
                f"AngularSteer=all-layers(0..{all_layers[-1]})  OAS_band={oas_band[0]}..{oas_band[-1]}  "
                f"de-refusal angle={target_deg:.0f}deg  rho={args.rho}  Q_term={args.q_term}\n")
        f.write(f"dynamics R2(1step)={val['r2_h1']:.4f}\n")
        f.write("enforcement sweep (single-layer deadbeat margin per band layer): "
                + "  ".join(f"L{ly}={mv:+.2f}" for ly, mv in enf.items()) + "\n")
        f.write("NOTE: 'Canonical-AS(1L)' = the PUBLISHED Angular Steering (input_layernorm output, "
                "1 layer) and is ~inert on Gemma. 'ResidualRot(all-L)'/'Deadbeat'/'OAS-*' rewrite the "
                "RESIDUAL stream (model.layers.{k} output) — a different, much stronger actuator; these "
                "are NOT canonical Angular Steering.\n")
        f.write("margins: " + "  ".join(f"{k}={v:+.2f}" for k, v in margins.items()) + "\n\n")
        for i, p in enumerate(gen_prompts):
            f.write(f"### prompt {i+1}: {p}\n")
            for label in labels:
                f.write(f"  [{label}] {texts[label][i].strip()}\n")
            f.write("\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
