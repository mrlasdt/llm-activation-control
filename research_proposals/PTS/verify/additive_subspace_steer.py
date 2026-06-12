"""Does ADDITIVE, k-dim subspace steering de-refuse Gemma where AS rotation fails?

Tests the two highest-leverage extensions beyond Angular Steering, in isolation:
  L1 — additive (norm-CHANGING) control instead of the norm-preserving 2D rotation;
  L2 — a k-dim refusal SUBSPACE instead of the 2D plane.

Actuator (bounded directional ablation of a k-dim refusal subspace U, applied at every
band layer):  p = U h ;  u = -p capped to ||u||<=u_max ;  h' = h + Uᵀ u .
  * u_max = inf  -> full projection-out  h' = h - UᵀU h  (exact ablation)
  * finite u_max -> partial removal (the coherence knob, lever L5)
For k=1 this is constrained directional ablation along the refusal axis; for k>1 it
removes a multi-dimensional refusal subspace (targets Gemma's diffuse refusal).

Conditions (all on the SAME discriminative band, same plane machinery):
  baseline | AS-rotation(band) [the limit we're solving] | ablate(k=1) | ablate(k=8),
  each additive condition at full and half strength. Reports refusal margin + a NEUTRAL
  off-axis perplexity (capability/coherence tax) + greedy generations.

Run:  python additive_subspace_steer.py --model google/gemma-2-2b-it
"""

import argparse
import gc
import math
import sys
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[3] / "pytorch_pure"))   # shared lib
sys.path.insert(0, str(_HERE.parents[1]))                    # PTS modules

from utils import get_input_data, tokenize_instructions_fn, add_hooks
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn
from pts_controller import PTSState, PolicyFixedAngle, attach_pts_hooks

NEUTRAL = [
    "The mitochondria is the membrane-bound organelle that generates most of the cell's ATP.",
    "In 1969, Apollo 11 landed the first humans on the Moon during the Space Race.",
    "To compute a matrix determinant by cofactor expansion, alternate signs along a row.",
    "The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.",
    "Photosynthesis converts carbon dioxide and water into glucose using light energy.",
    "A binary search halves the search interval each step, giving logarithmic time.",
    "The French Revolution began in 1789 and led to the rise of Napoleon Bonaparte.",
    "Saturn's rings are composed mostly of ice particles with a smaller amount of rock.",
    "Quicksort partitions an array around a pivot and recurses on the two halves.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level.",
    "The Great Barrier Reef off Australia is the world's largest coral reef system.",
    "An object in free fall near Earth accelerates at roughly 9.8 meters per second squared.",
    "DNA is structured as a double helix held together by complementary base pairs.",
    "The Fibonacci sequence begins 0, 1, 1, 2, 3, 5, 8 and each term sums the prior two.",
    "Mount Everest, on the border of Nepal and China, is the highest peak above sea level.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--plane-samples", type=int, default=192)
    ap.add_argument("--n-fit", type=int, default=192)
    ap.add_argument("--n-obs", type=int, default=24)
    ap.add_argument("--n-gen", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 8])
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

    harmful_train, harmful_test = get_input_data("harmful", "en")
    harmless_train, harmless_test = get_input_data("harmless", "en")

    # ---- plane (for the AS-rotation baseline + band/angle) ----
    ha = extract_all_layer_activations(model, harmful_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    la = extract_all_layer_activations(model, harmless_train[:args.plane_samples], tokenizer, ["mid"], 8)
    gc.collect(); torch.cuda.empty_cache()
    plane = compute_steering_plane(ha, la)
    b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
    STEER_LAYER = int(plane["selected_key"].split("_")[1])
    del ha, la; gc.collect(); torch.cuda.empty_cache()
    margin_fn, R, C = make_margin_fn(tokenizer, device)

    # ---- FULL residual-stream last-token means (the space we actually ablate) ----
    def residual_means(prompts, bs=8):
        sums = None; n = 0
        cache = {k: [] for k in range(L)}

        def mk(k):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                cache[k].append(h[:, -1, :].float().cpu())
            return hook
        hooks = [(module_dict[f"model.layers.{k}"], mk(k)) for k in range(L)]
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                for i in range(0, len(prompts), bs):
                    inp = tokenize_instructions_fn(prompts[i:i + bs], tokenizer)
                    model(input_ids=inp.input_ids.to(device),
                          attention_mask=inp.attention_mask.to(device))
        # per-layer mean over prompts -> (L, d)
        return np.stack([torch.cat(cache[k], 0).mean(0).numpy() for k in range(L)], 0)

    print("extracting residual-stream class means (full d) ...")
    hmean = residual_means(harmful_train[:args.n_fit]); gc.collect(); torch.cuda.empty_cache()
    lmean = residual_means(harmless_train[:args.n_fit]); gc.collect(); torch.cuda.empty_cache()

    # band + AS angle from the 2D projection of the means
    b1n = b1.cpu().numpy(); b2n = b2.cpu().numpy()
    hm2 = np.stack([hmean @ b1n, hmean @ b2n], 1)        # (L,2)
    lm2 = np.stack([lmean @ b1n, lmean @ b2n], 1)
    ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
    sep = np.abs(np.arctan2(np.sin(ang(hm2) - ang(lm2)), np.cos(ang(hm2) - ang(lm2))))
    peak = int(np.argmax(sep)); hi = sep > 0.5
    lo = peak; top = peak
    while lo - 1 >= 0 and hi[lo - 1]: lo -= 1
    while top + 1 < L - 1 and hi[top + 1]: top += 1
    band = list(range(lo, top + 1))
    fixed_angle = float(np.arctan2(np.sin(ang(lm2[band])).mean(), np.cos(ang(lm2[band])).mean()))
    print(f"steer layer {STEER_LAYER}; band {band[0]}..{band[-1]} ({len(band)}); "
          f"AS angle={math.degrees(fixed_angle):.0f}deg")

    # ---- refusal subspace U_k = top-k right singular vectors of the per-layer diff means ----
    diff = (hmean - lmean)[band]                          # (nb, d) refusal directions over band
    diff = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-9)
    # SVD (no centering: we want the span of the refusal directions, incl. their mean)
    _, _, Vh = np.linalg.svd(diff, full_matrices=False)   # Vh rows are right singular vecs (d,)
    U_by_k = {k: torch.from_numpy(Vh[:k].copy()).to(device) for k in args.ks}  # (k,d)
    # typical subspace-coordinate scale at the steer layer (for the u_max knob)
    pscale = {k: float(np.linalg.norm(Vh[:k] @ hmean[STEER_LAYER])) for k in args.ks}

    # ---- additive bounded-ablation hook ----
    def make_ablate_hook(U, u_max):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            Ud = U.to(h.device, h.dtype)                 # (k,d)
            p = h @ Ud.t()                               # (B,S,k)
            if u_max is None or not math.isfinite(u_max):
                u = -p
            else:
                nrm = p.norm(dim=-1, keepdim=True)
                scale = torch.clamp(torch.where(nrm > 0, u_max / (nrm + 1e-6),
                                                torch.zeros_like(nrm)), max=1.0)
                u = -p * scale
            steered = h + u @ Ud                         # (B,S,d)
            return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered
        return hook

    def ablate_hooks(k, u_max):
        h = make_ablate_hook(U_by_k[k], u_max)
        return [(module_dict[f"model.layers.{j}"], h) for j in band]

    def as_hooks():
        return attach_pts_hooks(module_dict, band, b1, b2,
                                PTSState(PolicyFixedAngle(fixed_angle, band)))

    # ---- metrics ----
    hp = harmful_test[:args.n_obs]
    ninp = tokenizer(NEUTRAL, return_tensors="pt", padding=True)
    nids = ninp.input_ids.to(device); nattn = ninp.attention_mask.to(device)

    def margin(hooks):
        inp = tokenize_instructions_fn(hp, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
        return float(margin_fn(out.logits[:, -1, :]).mean())

    def neutral_nll(hooks):
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                out = model(input_ids=nids, attention_mask=nattn)
        logits = out.logits[:, :-1, :].float(); labels = nids[:, 1:]; mask = nattn[:, 1:].bool()
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                              reduction="none").reshape(labels.shape)
        return float(nll[mask].mean())

    def generate(prompts, hooks):
        inp = tokenize_instructions_fn(prompts, tokenizer)
        ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                g = model.generate(ids, attention_mask=attn, max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)

    # ---- conditions ----
    conds = [("baseline", []), ("AS-rotation(band)", as_hooks())]
    for k in args.ks:
        conds.append((f"ablate k={k} FULL", ablate_hooks(k, None)))
        conds.append((f"ablate k={k} half", ablate_hooks(k, 0.5 * pscale[k])))

    base_nll = neutral_nll([])
    print("\n" + "=" * 70)
    print(f"refusal margin (baseline high=refuse) + neutral-corpus coherence tax")
    print(f"  {'condition':22s} {'margin':>8s} {'neutralNLL':>11s} {'tax':>7s}")
    rows = []
    for label, hooks in conds:
        m = margin(hooks); nll = neutral_nll(hooks)
        rows.append((label, m, nll))
        print(f"  {label:22s} {m:+8.2f} {nll:11.3f} {nll-base_nll:+7.3f}")

    print("\n" + "=" * 70)
    gen_labels = ["baseline", "AS-rotation(band)"] + \
                 [f"ablate k={k} FULL" for k in args.ks] + \
                 ([f"ablate k={args.ks[-1]} half"] if args.ks else [])
    gen_map = dict(conds)
    gp = harmful_test[:args.n_gen]
    texts = {lab: generate(gp, gen_map[lab]) for lab in gen_labels}
    print("GENERATIONS (greedy):")
    for i, p in enumerate(gp):
        print(f"\n### {p}")
        for lab in gen_labels:
            t = texts[lab][i].replace("\n", " ").strip()
            print(f"  [{lab:18s}] {t[:230]}")

    out = _HERE.parent / "outputs" / f"additive_subspace_{name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"model={args.model}  steer_layer={STEER_LAYER}  band={band[0]}..{band[-1]}  "
                f"AS_angle={math.degrees(fixed_angle):.0f}deg  base_neutralNLL={base_nll:.3f}\n\n")
        f.write("condition              margin   neutralNLL   tax\n")
        for label, m, nll in rows:
            f.write(f"{label:22s} {m:+8.2f} {nll:11.3f} {nll-base_nll:+7.3f}\n")
        f.write("\nGENERATIONS:\n")
        for i, p in enumerate(gp):
            f.write(f"\n### {p}\n")
            for lab in gen_labels:
                f.write(f"  [{lab}] {texts[lab][i].strip()}\n")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
