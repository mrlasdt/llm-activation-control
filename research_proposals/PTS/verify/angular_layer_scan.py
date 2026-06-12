"""All-layer scan of CANONICAL Angular Steering — does ANY single layer move the model?

The single-layer sweep (angular_sweep.py) tested only the auto-selected steer layer.
This scans EVERY layer: at each layer L it builds that layer's own refusal direction
(difference-in-means b1_L) + a shared PCA second axis, applies the canonical
input_layernorm rotation over a set of angles, and records the most de-refusal
(minimum first-token margin) achievable at that layer. If the min margin stays near
the (high) baseline at every layer, canonical Angular Steering genuinely cannot
control the model from any single layer — settling whether the earlier null was just
a bad layer choice.

Run:  python angular_layer_scan.py --model google/gemma-2-2b-it
"""

import argparse
import gc
import sys
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "pytorch_pure"))

from utils import (get_input_data, tokenize_instructions_fn, add_hooks,
                   get_angular_steering_output_hook)
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--plane-samples", type=int, default=192)
    ap.add_argument("--n-obs", type=int, default=24)
    ap.add_argument("--angle-step", type=int, default=45)
    ap.add_argument("--adaptive-mode", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
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

    harmful_train, harmful_test = get_input_data("harmful", "en")
    harmless_train, harmless_test = get_input_data("harmless", "en")
    ha = extract_all_layer_activations(model, harmful_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    la = extract_all_layer_activations(model, harmless_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    plane = compute_steering_plane(ha, la)
    cand = plane["candidate_directions"]            # {layer_{L}_mid: diff direction}
    b2_global = plane["b2"].cpu().numpy()
    del ha, la; gc.collect(); torch.cuda.empty_cache()
    margin_fn, R, C = make_margin_fn(tokenizer, device)

    hp = harmful_test[:args.n_obs]
    inp = tokenize_instructions_fn(hp, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)

    def margin(hooks):
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
        return float(margin_fn(out.logits[:, -1, :]).mean())

    base = margin([])
    angles = list(range(0, 360, args.angle_step))
    layer_keys = sorted(cand.keys(), key=lambda k: int(k.split("_")[1]))
    print(f"baseline margin={base:+.2f} (high=refuse). Scanning {len(layer_keys)} layers "
          f"x {len(angles)} angles, adaptive_mode={args.adaptive_mode}\n")
    print(f"  {'layer':>5s} {'min-margin':>11s} {'@angle':>7s} {'drop':>7s}")

    rows = []
    best = (1e9, None, None)                          # (min_margin, layer, angle)
    for key in layer_keys:
        L = int(key.split("_")[1])
        cfg = {"first_direction": cand[key].cpu().numpy(),
               "second_direction": b2_global}
        mod = module_dict[f"model.layers.{L}.input_layernorm"]
        per_angle = []
        for a in angles:
            hook = get_angular_steering_output_hook(cfg, float(a), args.adaptive_mode)
            per_angle.append(margin([(mod, hook)]))
        mn = float(np.min(per_angle)); amin = angles[int(np.argmin(per_angle))]
        rows.append((L, mn, amin))
        flag = "  <-- de-refuses" if mn < base - 3.0 else ""
        print(f"  {L:5d} {mn:+11.2f} {amin:7d} {mn-base:+7.2f}{flag}")
        if mn < best[0]:
            best = (mn, L, amin)

    print(f"\n  BEST canonical de-refusal across ALL layers: margin {best[0]:+.2f} "
          f"(baseline {base:+.2f}, drop {best[0]-base:+.2f}) at layer {best[1]} angle {best[2]}")
    verdict = ("canonical Angular Steering CANNOT meaningfully de-refuse this model from "
               "any single layer" if best[0] > base - 3.0 else
               f"layer {best[1]} @ {best[2]}deg DOES de-refuse")
    print(f"  VERDICT: {verdict}")

    # sample generation at the single best (layer, angle)
    if best[1] is not None:
        L, a = best[1], best[2]
        cfg = {"first_direction": cand[f"layer_{L}_mid"].cpu().numpy(),
               "second_direction": b2_global}
        hook = get_angular_steering_output_hook(cfg, float(a), args.adaptive_mode)
        mod = module_dict[f"model.layers.{L}.input_layernorm"]
        p = harmful_test[0]
        pinp = tokenize_instructions_fn([p], tokenizer)
        with add_hooks(module_forward_hooks=[(mod, hook)]):
            with torch.no_grad():
                g = model.generate(pinp.input_ids.to(device),
                                   attention_mask=pinp.attention_mask.to(device),
                                   max_new_tokens=args.max_new_tokens, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
        txt = tokenizer.batch_decode(g[:, pinp.input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"\n  best-config generation (layer {L}, {a}deg):")
        print(f"    prompt: {p}")
        print(f"    out: {txt[:240].strip()!r}")

    out = _HERE.parent / "outputs" / f"angular_layer_scan_{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"model={args.model}  baseline_margin={base:+.2f}  adaptive_mode={args.adaptive_mode}\n")
        f.write("layer  min_margin  @angle  drop\n")
        for L, mn, amin in rows:
            f.write(f"{L:5d}  {mn:+8.2f}  {amin:5d}  {mn-base:+.2f}\n")
        f.write(f"\nBEST: margin {best[0]:+.2f} at layer {best[1]} angle {best[2]} "
                f"(drop {best[0]-base:+.2f})\nVERDICT: {verdict}\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
