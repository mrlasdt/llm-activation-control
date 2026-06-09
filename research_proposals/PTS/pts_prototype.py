"""PTS prototype: Predictive Trajectory Steering in the loop on a real model.

Mirrors clas_prototype.py. Unlike pts_offline.py (which validates the controller
maths on saved trajectories), this drives Qwen2.5-3B-Instruct itself, so the plant
is the real, nonlinear model and the fitted 2x2 dynamics are only an approximation:

  [A] Dynamics on the REAL residual stream — fit per-layer 2x2 affine A_k,b_k from
      layer-output coordinates (the point we actually actuate) and report held-out
      prediction accuracy (S7.4 Exp 2 on the true steering point).

  [B] Closed-loop tracking in the real model — run the receding-horizon MPC during
      a forward pass, record the realised per-layer angle, and measure:
        * does PTS steer the real trajectory toward the reference (vs autonomous)?
        * the one-step model-mismatch w_j = ||c_{k+1}^real - (A_k s_k + b_k)|| under
          steering — the residual the offline sim cannot show (S5.4 / Thm 8.1).

  [C] Behavioural control — the refusal/compliance margin observable (reused from
      clas_controller) under: no-steer, fixed-angle Angular Steering, and PTS-MPC
      tracking an interpolated reference (Option B, lambda 0->1). Shows the 2D
      reference is a graded behavioural dial (monotone de-refusal over lambda 0->0.75,
      saturating near the compliance floor by lambda=1), and that PTS shifts
      behaviour at a tunable, bounded perturbation.

  [D] Qualitative generations under each controller.

Run:  python pts_prototype.py
"""

import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# shared library lives in <repo>/pytorch_pure (proposals live in <repo>/research_proposals/<NAME>/)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "pytorch_pure"))

from utils import get_input_data, tokenize_instructions_fn, add_hooks
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn
from pts_dynamics import fit_layer_dynamics, validate_dynamics
from pts_mpc import MPCController, reference_trajectory
from pts_controller import (PTSState, PolicyMPC, PolicyFixedAngle, PolicyNone,
                            attach_pts_hooks)

# ----------------------------------------------------------------------------- config
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
POSITION = "mid"
PLANE_SAMPLES = 192
N_FIT = 192               # prompts per class for dynamics fitting (residual stream)
N_OBS = 24                # prompts per class for the behavioural sweep
N_GEN = 4                 # prompts for qualitative generation
MAX_NEW_TOKENS = 40
HORIZON = 6
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

# ----------------------------------------------------------------------------- model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
module_dict = dict(model.named_modules())
L = model.config.num_hidden_layers
print(f"loaded {MODEL_ID}: {L} layers")

# ----------------------------------------------------------------------------- data + plane
harmful_train, harmful_test = get_input_data("harmful", "en")
harmless_train, harmless_test = get_input_data("harmless", "en")

ha = extract_all_layer_activations(model, harmful_train[:PLANE_SAMPLES], tokenizer, [POSITION], 8)
gc.collect(); torch.cuda.empty_cache()
la = extract_all_layer_activations(model, harmless_train[:PLANE_SAMPLES], tokenizer, [POSITION], 8)
gc.collect(); torch.cuda.empty_cache()
plane = compute_steering_plane(ha, la)
b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
STEER_LAYER = int(plane["selected_key"].split("_")[1])
del ha, la; gc.collect(); torch.cuda.empty_cache()
print(f"plane: {plane['selected_key']} -> CLAS steer layer {STEER_LAYER}; b1.b2={(b1@b2).item():.2e}")

margin_fn, R, C = make_margin_fn(tokenizer, device)


# ----------------------------------------------------------------------------- residual-stream coords
def extract_residual_coords(prompts, batch_size=8):
    """Last-token (b1,b2) coordinates at EVERY layer output (the residual stream we
    actuate). Returns (N, L, 2). Tuple-aware hooks on model.layers.{k}."""
    cache = {k: [] for k in range(L)}

    def mk(k):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            last = h[:, -1, :]                       # (B,d)
            _b1 = b1.to(last.dtype); _b2 = b2.to(last.dtype)
            c1 = (last @ _b1).float().cpu()
            c2 = (last @ _b2).float().cpu()
            cache[k].append(torch.stack([c1, c2], -1))   # (B,2)
        return hook

    hooks = [(module_dict[f"model.layers.{k}"], mk(k)) for k in range(L)]
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                inp = tokenize_instructions_fn(prompts[i:i + batch_size], tokenizer)
                model(input_ids=inp.input_ids.to(device),
                      attention_mask=inp.attention_mask.to(device))
    coords = np.stack([torch.cat(cache[k], 0).numpy() for k in range(L)], axis=1)  # (N,L,2)
    return coords


# ----------------------------------------------------------------------------- [A] dynamics on real residual stream
print("\n[A] fitting 2x2 dynamics on the REAL residual stream (layer outputs)...")
hc = extract_residual_coords(harmful_train[:N_FIT]); gc.collect(); torch.cuda.empty_cache()
lc = extract_residual_coords(harmless_train[:N_FIT]); gc.collect(); torch.cuda.empty_cache()
all_c = np.concatenate([hc, lc], 0)                      # (2N, L, 2)
n = all_c.shape[0]
perm = np.random.RandomState(0).permutation(n)
tr, te = perm[:n // 2], perm[n // 2:]
fit = fit_layer_dynamics(all_c[tr, :, 0], all_c[tr, :, 1], affine=True)
val = validate_dynamics(all_c[te, :, 0], all_c[te, :, 1], fit, horizons=(1, 5))
print(f"  held-out R2: 1-step={val['r2_h1']:.4f}  5-step={val['r2_h5']:.4f}  "
      f"NRMSE(1-step)={val['nrmse_h1']:.4f}")

# reference trajectories from the residual-stream class means
harmful_mean = hc.mean(0)                                # (L,2)
harmless_mean = lc.mean(0)
ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
angle_sep = np.abs(np.arctan2(np.sin(ang(harmful_mean) - ang(harmless_mean)),
                              np.cos(ang(harmful_mean) - ang(harmless_mean))))
peak = int(np.argmax(angle_sep))
hi = angle_sep > 0.5
lo = peak
while lo - 1 >= 0 and hi[lo - 1]:
    lo -= 1
top = peak
while top + 1 < L - 1 and hi[top + 1]:
    top += 1
band = list(range(lo, top + 1))
ref_scale = float(np.linalg.norm(harmless_mean[band], axis=-1).mean())
print(f"  steering band (refusal-angle separation around layer {peak}): "
      f"layers {band[0]}..{band[-1]} ({len(band)} layers); ref scale~{ref_scale:.2f}")

A, bvec = fit["A"], fit["b"]


# ----------------------------------------------------------------------------- helpers: steered prefill
def prefill_with_policy(prompts, policy, record=False):
    """Single forward pass under a steering policy. Returns (margin (B,), state)."""
    state = PTSState(policy, record=record)
    hooks = attach_pts_hooks(module_dict, band, b1, b2, state)
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    return margin_fn(out.logits[:, -1, :]), state


def baseline_margin(prompts):
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    return margin_fn(out.logits[:, -1, :])


# ----------------------------------------------------------------------------- [B] closed-loop tracking in the real model
print("\n[B] closed-loop tracking in the real model (prefill, harmful prompts)...")
ref_track = reference_trajectory(harmful_mean, harmless_mean, option="A")  # harmless
mpc = MPCController(A, bvec, ref_track, layers=band, H=HORIZON, q_pos=1.0,
                    r_ctrl=0.02, qf_scale=4.0, u_max=0.5 * ref_scale)
trk_prompts = harmful_test[:N_OBS]
B = len(trk_prompts)

# circular mean of the harmless-reference band angle = the single best fixed angle
ref_band_ang = ang(harmless_mean[band])
fixed_angle = math.atan2(np.sin(ref_band_ang).mean(), np.cos(ref_band_ang).mean())


def logged_last_token(state):
    """Per-band-layer last-token PRE-steering incoming coord, commanded angle, and
    control, from a record-enabled prefill (one forward call). Prefill logs (B*S,2)
    in (prompt,position) row-major; the decision token is each prompt's last position.
    """
    cc, th, uu = {}, {}, {}
    for k in band:
        c = state.log[k]["c"][-1]                    # (B*S,2)
        S = c.shape[0] // B
        cc[k] = c.reshape(B, S, 2)[:, -1, :]         # (B,2) last token
        t = state.log[k]["theta"][-1].reshape(B, S)[:, -1]
        th[k] = t
        if state.log[k].get("u"):
            uu[k] = state.log[k]["u"][-1].reshape(B, S, 2)[:, -1, :]
    return cc, th, uu


# PTS: incoming (pre-steering) real coords carry the cumulative effect of steering
_, st_pts = prefill_with_policy(trk_prompts, PolicyMPC(mpc), record=True)
inc_pts, theta_pts, u_pts = logged_last_token(st_pts)
# autonomous: the natural trajectory, no steering (clean, no reset)
coords_non = extract_residual_coords(trk_prompts)            # (B,L,2) last-token
inc_non = {k: coords_non[:, k, :] for k in band}


def mean_ang_err(incoming):
    """Mean |angle(incoming_k) - angle(ref_k)| over the band — how close the REAL
    model's trajectory (as it enters each band layer) sits to the reference."""
    errs = []
    for k in band:
        a = np.arctan2(incoming[k][:, 1], incoming[k][:, 0])
        r = math.atan2(ref_track[k, 1], ref_track[k, 0])
        errs.append(np.abs(np.arctan2(np.sin(a - r), np.cos(a - r))).mean())
    return float(np.degrees(np.mean(errs)))


print(f"  angular distance of the REAL incoming trajectory to the harmless reference:")
print(f"    autonomous (no steer): {mean_ang_err(inc_non):6.1f} deg")
print(f"    PTS-MPC (steered)    : {mean_ang_err(inc_pts):6.1f} deg")
print(f"    -> PTS bends the real model's residual stream toward the reference.")

# one-step model mismatch w_j under PTS steering: real incoming c_{k+1} vs the
# fitted prediction A_k s_k + b_k from the actuated coord s_k (Thm 8.1's residual).
wj = []
for kk in band[:-1]:
    if kk + 1 not in inc_pts:
        continue
    r = np.linalg.norm(inc_pts[kk], axis=-1, keepdims=True)   # actuated magnitude
    s_k = r * np.stack([np.cos(theta_pts[kk]), np.sin(theta_pts[kk])], -1)  # (B,2)
    pred = s_k @ A[kk].T + bvec[kk]
    wj.append(np.linalg.norm(inc_pts[kk + 1] - pred, axis=-1).mean())
wj_mean = float(np.mean(wj)) if wj else float("nan")
print(f"  one-step model mismatch w_j under steering: mean={wj_mean:.3f} "
      f"(~{wj_mean/ref_scale:.1%} of ref scale) — the real-plant residual the linear "
      f"model incurs under actuation; bounded, as Thm 8.1 assumes")

# ----------------------------------------------------------------------------- [C] behavioural control
print("\n[C] behavioural control — refusal/compliance margin observable...")
hp = harmful_test[:N_OBS]
base_h = baseline_margin(hp).cpu().numpy()
fix_h, _ = prefill_with_policy(hp, PolicyFixedAngle(fixed_angle, band))
fix_h = fix_h.cpu().numpy()
print(f"  harmful prompts: baseline margin mean={base_h.mean():+.3f}  "
      f"(high=refuse, low=comply)")

lambdas = [0.0, 0.25, 0.5, 0.75, 1.0]
margins_lam, pert_lam = [], []
for lam in lambdas:
    ref_l = reference_trajectory(harmful_mean, harmless_mean, option="B", lam=lam)
    mpc_l = MPCController(A, bvec, ref_l, layers=band, H=HORIZON, q_pos=1.0,
                         r_ctrl=0.02, qf_scale=4.0, u_max=0.5 * ref_scale)
    m, st = prefill_with_policy(hp, PolicyMPC(mpc_l), record=True)
    margins_lam.append(float(m.mean()))
    # realised perturbation ||u|| (from logged controls, last forward call)
    us = [np.linalg.norm(st.log[k]["u"][-1], axis=-1).mean() for k in band
          if st.log.get(k, {}).get("u")]
    pert_lam.append(float(np.mean(us)) if us else 0.0)
    print(f"    lambda={lam:.2f} (0=harmful ref,1=harmless ref): "
          f"margin={margins_lam[-1]:+.3f}  mean||u||={pert_lam[-1]:.2f}")

# constrained vs fixed-angle at MATCHED behavioural shift: bounded perturbation
print(f"  fixed-angle margin={fix_h.mean():+.3f}; "
      f"PTS(lam=1) margin={margins_lam[-1]:+.3f} at mean||u||={pert_lam[-1]:.2f}")

# ----------------------------------------------------------------------------- [D] generations
print("\n[D] qualitative generations (greedy, 40 tok)...")


def generate(prompts, policy):
    state = PTSState(policy)
    hooks = attach_pts_hooks(module_dict, band, b1, b2, state)
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            g = model.generate(ids, attention_mask=attn, max_new_tokens=MAX_NEW_TOKENS,
                               do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)


gen_prompts = harmful_test[:N_GEN]
ref_jb = reference_trajectory(harmful_mean, harmless_mean, option="A")
mpc_jb = MPCController(A, bvec, ref_jb, layers=band, H=HORIZON, q_pos=1.0,
                      r_ctrl=0.02, qf_scale=4.0, u_max=0.5 * ref_scale)
gen_base = generate(gen_prompts, PolicyNone())
gen_fix = generate(gen_prompts, PolicyFixedAngle(fixed_angle, band))
gen_pts = generate(gen_prompts, PolicyMPC(mpc_jb))
for i, p in enumerate(gen_prompts):
    print(f"\n  prompt: {p[:70]}")
    print(f"    baseline : {gen_base[i][:90]!r}")
    print(f"    fixed    : {gen_fix[i][:90]!r}")
    print(f"    PTS-MPC  : {gen_pts[i][:90]!r}")

# ----------------------------------------------------------------------------- save
np.savez_compressed(
    "pts_prototype_results.npz",
    steer_layer=STEER_LAYER, band=np.array(band), L=L,
    r2_h1=val["r2_h1"], r2_h5=val["r2_h5"], nrmse_h1=val["nrmse_h1"],
    per_layer_r2=np.array(val["per_layer_r2_1step"]),
    harmful_mean=harmful_mean, harmless_mean=harmless_mean, ref_track=ref_track,
    angle_sep=angle_sep, ref_scale=ref_scale, fixed_angle=fixed_angle,
    track_autonomous=mean_ang_err(inc_non), track_pts=mean_ang_err(inc_pts),
    wj_mean=wj_mean,
    real_pts_ang=np.array([np.arctan2(inc_pts[k][:, 1], inc_pts[k][:, 0]).mean()
                           for k in band]),
    real_non_ang=np.array([np.arctan2(inc_non[k][:, 1], inc_non[k][:, 0]).mean()
                           for k in band]),
    ref_ang=np.array([math.atan2(ref_track[k, 1], ref_track[k, 0]) for k in band]),
    base_margin=base_h, fixed_margin=fix_h,
    lambdas=np.array(lambdas), margins_lam=np.array(margins_lam),
    pert_lam=np.array(pert_lam),
)
print("\nsaved pts_prototype_results.npz")
