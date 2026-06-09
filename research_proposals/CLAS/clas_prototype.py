"""CLAS prototype: measure G(theta) on the continuous behavioral margin, find the
monotone band, and run a first output-feedback thermostat loop.

Run:  python clas_prototype.py
"""

import gc
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# shared library lives in <repo>/pytorch_pure
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "pytorch_pure"))

from utils import get_input_data, tokenize_instructions_fn, add_hooks
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from clas_controller import SteerState, make_clas_hook, make_margin_fn, OuterThermostat

# ----------------------------------------------------------------------------- config
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
POSITION = "mid"
PLANE_SAMPLES = 192
N_OBS = 24                 # prompts per class for the G(theta) sweep / sanity
N_DEMO = 8                 # harmful prompts for the thermostat decode demo
MAX_NEW_TOKENS = 32
SWEEP_DEG = list(range(0, 360, 15))
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
print(f"loaded {MODEL_ID}: {model.config.num_hidden_layers} layers")

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
print(f"plane: {plane['selected_key']} -> steer layer {STEER_LAYER}, b1.b2={(b1@b2).item():.2e}")

steer_module = module_dict[f"model.layers.{STEER_LAYER}"]
state = SteerState()
margin_fn, R, C = make_margin_fn(tokenizer, device)
print(f"observable: |R|={len(R)} refusal ids, |C|={len(C)} compliance ids")


# ----------------------------------------------------------------------------- helpers
def prefill_margin(prompts, theta=None):
    """Batched prefill; return per-prompt margin at the first generated position.
    theta=None -> no steering; scalar -> same angle all rows; (B,) tensor -> per-row.
    Left-padding handled via position_ids. Returns a (B,) tensor on device."""
    inputs = tokenize_instructions_fn(prompts, tokenizer)
    ids = inputs.input_ids.to(device)
    attn = inputs.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    hooks = []
    if theta is not None:
        state.enabled = True
        state.theta = theta if torch.is_tensor(theta) else torch.tensor(float(theta), device=device)
        hooks = [(steer_module, make_clas_hook(b1, b2, state))]
    else:
        state.enabled = False
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    state.enabled = True
    return margin_fn(out.logits[:, -1, :])


def decode_single(prompt, controller=None, fixed_theta=None, steer=True,
                  max_new_tokens=MAX_NEW_TOKENS):
    """batch=1 manual KV-cached greedy decode. Records (y_t, theta_t) per step.
    controller given -> thermostat; else fixed_theta held; steer=False -> no hook."""
    inputs = tokenize_instructions_fn([prompt], tokenizer)
    ids = inputs.input_ids.to(device)
    attn = inputs.attention_mask.to(device)
    hooks = []
    if steer:
        state.enabled = True
        theta = (controller.reset(1, device) if controller is not None
                 else torch.tensor([float(fixed_theta)], device=device))
        state.theta = theta
        hooks = [(steer_module, make_clas_hook(b1, b2, state))]
    else:
        state.enabled = False
        theta = torch.tensor([0.0], device=device)

    ys, ths, gen = [], [], []
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            y = margin_fn(logits)
            ys.append(y.item()); ths.append(theta.item())
            nxt = logits.argmax(-1); gen.append(nxt.item())
            for _ in range(max_new_tokens - 1):
                if controller is not None:
                    theta = controller.update(y)
                    state.theta = theta
                attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)
                out = model(input_ids=nxt[:, None], attention_mask=attn,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                y = margin_fn(logits)
                ys.append(y.item()); ths.append(theta.item())
                nxt = logits.argmax(-1); gen.append(nxt.item())
                if nxt.item() == tokenizer.eos_token_id:
                    break
    state.enabled = True
    return {"y": ys, "theta": ths, "ids": gen,
            "text": tokenizer.decode(gen, skip_special_tokens=True)}


# ----------------------------------------------------------------------------- 0. self-check
print("\n[0] decode-loop self-check vs model.generate (no steering)...")
p0 = harmful_test[0]
man_ids = decode_single(p0, steer=False, max_new_tokens=16)
inp = tokenize_instructions_fn([p0], tokenizer)
with torch.no_grad():
    g = model.generate(inp.input_ids.to(device), attention_mask=inp.attention_mask.to(device),
                       max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id)
ref_ids = g[0, inp.input_ids.shape[1]:].tolist()
mids = man_ids["ids"]
k = 0
while k < min(len(mids), len(ref_ids)) and mids[k] == ref_ids[k]:
    k += 1
print(f"  manual : {man_ids['text']!r}")
print(f"  generate:{tokenizer.decode(ref_ids, skip_special_tokens=True)!r}")
print(f"  greedy tokens agree for first {k}/{min(len(mids),len(ref_ids))} tokens "
      f"(tail divergence = bf16 tie-breaking; first token MUST match: {mids[0]==ref_ids[0]})")

# ----------------------------------------------------------------------------- 1. observable sanity
print("\n[1] observable sign check (no steering): harmful margin should exceed harmless...")
mh = prefill_margin(harmful_test[:N_OBS], theta=None).cpu().numpy()
ml = prefill_margin(harmless_test[:N_OBS], theta=None).cpu().numpy()
print(f"  harmful  margin: mean={mh.mean():+.3f}  (n={len(mh)})")
print(f"  harmless margin: mean={ml.mean():+.3f}  (n={len(ml)})")

# ----------------------------------------------------------------------------- 2. G(theta)
print("\n[2] measuring G(theta) on the continuous margin...")
gh, gl = [], []
hp, lp = harmful_test[:N_OBS], harmless_test[:N_OBS]
for d in SWEEP_DEG:
    th = math.radians(d)
    gh.append(float(prefill_margin(hp, theta=th).mean()))
    gl.append(float(prefill_margin(lp, theta=th).mean()))
    print(f"  theta={d:3d}  harmful_margin={gh[-1]:+.3f}  harmless_margin={gl[-1]:+.3f}")
gh = np.array(gh); gl = np.array(gl); deg = np.array(SWEEP_DEG)

# identify monotone rising band on the harmful margin, below the peak
peak_i = int(gh.argmax())
peak_deg = int(deg[peak_i])
# band: from the angle where the rise begins up to just below the peak
rise = gh - gh.min()
thr = gh.min() + 0.15 * (gh.max() - gh.min())
lo_i = int(np.argmax(rise > (0.15 * rise.max())))  # first index meaningfully above floor
band_lo_deg = int(deg[lo_i])
band_hi_deg = max(band_lo_deg + 30, peak_deg - 15)
# slope sign on the band (harmful margin increasing with theta toward peak)
slope_sign = 1.0 if gh[peak_i] >= gh[lo_i] else -1.0
print(f"  -> peak at theta={peak_deg} (margin={gh.max():+.3f}); "
      f"band=[{band_lo_deg},{band_hi_deg}] deg; slope_sign={slope_sign:+.0f}")

# ----------------------------------------------------------------------------- 3. thermostat
# Behaviorally meaningful signal: the FIRST-TOKEN refuse/comply decision y_0.
# Closed-loop demo: a MIXED batch (harmful + harmless) regulated to a COMMON
# behavioral setpoint y*. The two groups have G(theta) curves offset by ~7 margin
# units, so NO single fixed angle satisfies both; the feedback loop gives each input
# its own angle. The "disturbance" rejected is the input's intrinsic harmful/harmless
# bias. Gains are scaled by the MEASURED plant slope gamma (tune against gamma).
N_ITERS = 30
theta_nom = math.radians((band_lo_deg + band_hi_deg) / 2)
band_rad = (math.radians(band_lo_deg), math.radians(band_hi_deg))

# measured slope across the band -> principled gains. The slope is NON-uniform
# (steeper near theta~90-120), so tune against gamma_HI (the max local slope), not
# the average, or the loop goes unstable where the plant is steepest (proposal: §4.2).
rad = np.radians(deg)
dgh = np.gradient(gh, rad)
band_mask = (deg >= band_lo_deg) & (deg <= band_hi_deg)
gamma_hi = float(np.max(np.abs(dgh[band_mask])))
gamma_avg = (gh[int(np.where(deg == band_hi_deg)[0][0])] - gh[int(np.where(deg == band_lo_deg)[0][0])]) \
    / (math.radians(band_hi_deg) - math.radians(band_lo_deg))
Kp, Ki = 0.6 / gamma_hi, 0.5 / gamma_hi     # worst-case loop gain Kp*gamma_hi=0.6 (eigvals in unit disk)

# common setpoint reachable by BOTH groups: midway between their margins at nominal
mh_nom = float(prefill_margin(hp, theta=theta_nom).mean())
ml_nom = float(prefill_margin(lp, theta=theta_nom).mean())
y_star = 0.5 * (mh_nom + ml_nom)
print(f"\n[3] closed-loop regulation of the first-token decision, MIXED batch: "
      f"theta_nom={math.degrees(theta_nom):.0f}deg band=[{band_lo_deg},{band_hi_deg}]deg")
print(f"    slope gamma_hi={gamma_hi:.2f} (avg {gamma_avg:.2f}) margin/rad -> Kp={Kp:.3f} Ki={Ki:.3f} "
      f"(worst-case loop gain Kp*gamma_hi={Kp*gamma_hi:.2f})")
print(f"    y*={y_star:+.3f}  (harmful@nom={mh_nom:+.2f}, harmless@nom={ml_nom:+.2f})")

nh = N_DEMO // 2
demo = harmful_test[:nh] + harmless_test[:nh]
is_harm = np.array([1] * nh + [0] * nh)
B = len(demo)

# --- open-loop baseline: one fixed angle for everyone ---
y_fixed = prefill_margin(demo, theta=torch.full((B,), theta_nom, device=device)).cpu().numpy()

# --- closed-loop thermostat (vectorized over the batch) ---
ctrl = OuterThermostat(theta_nom, y_star, Kp, Ki, band_rad, slope_sign=slope_sign)
theta = ctrl.reset(B, device)
traj_y, traj_th = [], []
for it in range(N_ITERS):
    y = prefill_margin(demo, theta=theta)                 # observe at current angle
    traj_y.append(y.cpu().numpy()); traj_th.append(theta.detach().cpu().numpy())
    theta = ctrl.update(y)                                # PI -> next angle
y_final = prefill_margin(demo, theta=theta).cpu().numpy()
theta_final_deg = np.degrees(theta.detach().cpu().numpy())
traj_y = np.array(traj_y); traj_th = np.degrees(np.array(traj_th))

print(f"\n  open-loop (fixed {math.degrees(theta_nom):.0f}deg): "
      f"mean|y*-y|={np.abs(y_star-y_fixed).mean():.3f}  std(y)={y_fixed.std():.3f}  "
      f"[harmful y={y_fixed[is_harm==1].mean():+.2f}, harmless y={y_fixed[is_harm==0].mean():+.2f}]")
print(f"  closed-loop (PI thermostat):        "
      f"mean|y*-y|={np.abs(y_star-y_final).mean():.3f}  std(y)={y_final.std():.3f}  "
      f"[harmful y={y_final[is_harm==1].mean():+.2f}, harmless y={y_final[is_harm==0].mean():+.2f}]")
print("  convergence (mean|err| over iterations):")
for it in range(0, N_ITERS, 3):
    print(f"    iter {it:2d}: mean|y*-y|={np.abs(y_star-traj_y[it]).mean():.3f}  "
          f"std(y)={traj_y[it].std():.3f}  mean theta={traj_th[it].mean():.0f}deg")
print(f"  final per-input angles (deg):  harmful={np.round(theta_final_deg[is_harm==1]).astype(int).tolist()}")
print(f"                                 harmless={np.round(theta_final_deg[is_harm==0]).astype(int).tolist()}")
print("  -> each group converges to a DIFFERENT angle to reach the same behavioral setpoint.")

# ----------------------------------------------------------------------------- save
np.savez_compressed(
    "clas_prototype_results.npz",
    deg=deg, gh=gh, gl=gl, peak_deg=peak_deg, band=[band_lo_deg, band_hi_deg],
    y_star=y_star, theta_nom_deg=math.degrees(theta_nom),
    y_fixed=y_fixed, y_final=y_final, traj_y=traj_y, traj_th=traj_th,
)
print("\nsaved clas_prototype_results.npz")
