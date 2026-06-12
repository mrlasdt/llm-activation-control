"""CANONICAL Angular Steering — angle sweep, faithful to the repo's own method.

This is the method as Angular Steering (Vu & Nguyen, NeurIPS'25) actually applies
it, using the repo's own `utils.get_angular_steering_output_hook`:
  * hook point = `model.layers.{L}.input_layernorm` OUTPUT (the normalised
    pre-attention activation), NOT the residual stream;
  * a SINGLE selected layer (the max-cosine-similarity "steer layer"), NOT a band;
  * an adaptive mask (adaptive_mode=1 steers only positions whose projection onto
    the feature direction is positive; mode=0 steers always);
  * swept over the full 0–360° of the steering plane.

Use this to check whether canonical Angular Steering actually controls a given
model — in particular Gemma, which is reputed to resist steering. Contrast with
`compare_pts_vs_angular.py`, whose "Angular(band)" condition resets the RESIDUAL
STREAM across 14–18 layers at once (a much more aggressive, non-canonical actuation
that can flip behaviour even when the canonical single-layer method cannot).

For each (adaptive_mode, angle) it reports the first-token refusal/compliance
margin over harmful prompts; it also prints a sample generation at a few angles.

Run:  python angular_sweep.py --model google/gemma-2-2b-it
      python angular_sweep.py --model Qwen/Qwen2.5-3B-Instruct
"""

import argparse
import gc
import sys
import pathlib

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "pytorch_pure"))   # shared lib

from utils import (get_input_data, tokenize_instructions_fn, add_hooks,
                   get_angular_steering_output_hook)
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--plane-samples", type=int, default=192)
    ap.add_argument("--n-obs", type=int, default=24, help="harmful prompts for the margin")
    ap.add_argument("--step", type=int, default=30, help="angle step in degrees")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gen-angles", type=int, nargs="*", default=[0, 90, 180, 270])
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

    # same plane as the comparison script (so the direction is identical)
    ha = extract_all_layer_activations(model, harmful_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    la = extract_all_layer_activations(model, harmless_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    plane = compute_steering_plane(ha, la)
    STEER_LAYER = int(plane["selected_key"].split("_")[1])
    del ha, la; gc.collect(); torch.cuda.empty_cache()
    margin_fn, R, C = make_margin_fn(tokenizer, device)

    # canonical hook applies at the input_layernorm OUTPUT of the selected layer
    module_name = f"model.layers.{STEER_LAYER}.input_layernorm"
    steering_config = {
        "first_direction": plane["b1"].cpu().numpy(),
        "second_direction": plane["b2"].cpu().numpy(),
    }
    print(f"canonical Angular Steering at {module_name} (single layer {STEER_LAYER})")

    hp = harmful_test[:args.n_obs]

    def margin_at(angle_deg, adaptive_mode):
        hook = get_angular_steering_output_hook(steering_config, float(angle_deg), adaptive_mode)
        inp = tokenize_instructions_fn(hp, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
        with add_hooks(module_forward_hooks=[(module_dict[module_name], hook)]):
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
        return float(margin_fn(out.logits[:, -1, :]).mean())

    def generate_at(prompt, angle_deg, adaptive_mode):
        inp = tokenize_instructions_fn([prompt], tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        if angle_deg is None:                       # baseline: no steering hook
            hooks = []
        else:
            hook = get_angular_steering_output_hook(steering_config, float(angle_deg), adaptive_mode)
            hooks = [(module_dict[module_name], hook)]
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                g = model.generate(ids, attention_mask=attn, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)[0]

    angles = list(range(0, 360, args.step))
    print("\n" + "=" * 70)
    print(f"first-token refusal margin (high=refuse, low=comply) over {len(hp)} harmful prompts")
    print(f"  {'angle':>6s}  {'adaptive=0':>11s}  {'adaptive=1':>11s}")
    table = {}
    for a in angles:
        m0 = margin_at(a, 0)
        m1 = margin_at(a, 1)
        table[a] = (m0, m1)
        print(f"  {a:6d}  {m0:+11.2f}  {m1:+11.2f}")
    # baseline reference
    base = generate_at(harmful_test[0], None, 0)

    print("\n" + "=" * 70)
    print(f"sample generations at angles {args.gen_angles} (adaptive_mode=0):")
    prompt = harmful_test[0]
    print(f"prompt: {prompt}")
    print(f"  [baseline] {base[:200].strip()!r}")
    gens = {}
    for a in args.gen_angles:
        g = generate_at(prompt, a, 0)
        gens[a] = g
        print(f"  [angle {a:3d}] {g[:200].strip()!r}")

    out = _HERE.parent / "outputs" / f"angular_sweep_{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"model={args.model}  canonical AS at {module_name} (layer {STEER_LAYER})\n")
        f.write("angle  adaptive=0  adaptive=1  (first-token refusal margin)\n")
        for a in angles:
            f.write(f"{a:5d}  {table[a][0]:+8.2f}  {table[a][1]:+8.2f}\n")
        f.write(f"\nprompt: {prompt}\n  [baseline] {base.strip()}\n")
        for a in args.gen_angles:
            f.write(f"  [angle {a}] {gens[a].strip()}\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
