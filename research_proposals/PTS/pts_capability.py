"""(b) The effectiveness-vs-capability axis that pts_offline/pts_prototype omitted —
the real deliverable of proposal S7.4 Exp 3 ("Pareto curves of steering effectiveness
vs model capability (PPL)") and Exp 4 ("perplexity plots showing bounded degradation
under PTS vs unbounded under baselines").

Fixes the documented gap in PTS_README "Discrepancies" #3/#4: we previously showed
PTS bounds the *perturbation* ||u|| (the mechanism S8.2 claims protects coherence);
here we measure whether that actually buys coherence.

Setup (Qwen2.5-3B): reference = harmless mean (the de-refusal target). Sweep the PTS
budget u_max from tiny to large; for each, measure
  * EFFECTIVENESS  = refusal/compliance margin on harmful prompts (lower = de-refused),
  * CAPABILITY TAX = perplexity on held-out clean text with the steering hooks ON
                     (how much ordinary-language fluency the steering destroys),
  * realised mean ||u||.
Compared against no-steer (baseline) and fixed-angle Angular Steering (one strong,
unbounded-perturbation point). The claim to test: PTS traces a frontier of bounded
capability tax that fixed-angle cannot — at matched de-refusal, lower PPL damage.

Run:  python pts_capability.py
"""

import gc
import sys
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "pytorch_pure"))

from utils import get_input_data, tokenize_instructions_fn, add_hooks
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn
from pts_dynamics import fit_layer_dynamics
from pts_mpc import MPCController, reference_trajectory
from pts_controller import (PTSState, PolicyMPC, PolicyFixedAngle, PolicyNone,
                            attach_pts_hooks)

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PLANE_SAMPLES = 192
N_FIT = 192
N_HARM = 24          # harmful prompts for effectiveness
N_CLEAN = 32         # clean harmless prompts for the capability (PPL) probe
HORIZON = 6
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
module_dict = dict(model.named_modules())
L = model.config.num_hidden_layers
print(f"loaded {MODEL_ID}: {L} layers")

harmful_train, harmful_test = get_input_data("harmful", "en")
harmless_train, harmless_test = get_input_data("harmless", "en")

ha = extract_all_layer_activations(model, harmful_train[:PLANE_SAMPLES], tokenizer, ["mid"], 8)
gc.collect(); torch.cuda.empty_cache()
la = extract_all_layer_activations(model, harmless_train[:PLANE_SAMPLES], tokenizer, ["mid"], 8)
gc.collect(); torch.cuda.empty_cache()
plane = compute_steering_plane(ha, la)
b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
del ha, la; gc.collect(); torch.cuda.empty_cache()
margin_fn, R, C = make_margin_fn(tokenizer, device)


def extract_residual_coords(prompts, batch_size=8):
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
            for i in range(0, len(prompts), batch_size):
                inp = tokenize_instructions_fn(prompts[i:i + batch_size], tokenizer)
                model(input_ids=inp.input_ids.to(device),
                      attention_mask=inp.attention_mask.to(device))
    return np.stack([torch.cat(cache[k], 0).numpy() for k in range(L)], axis=1)


print("\nfitting dynamics + band on the real residual stream...")
hc = extract_residual_coords(harmful_train[:N_FIT]); gc.collect(); torch.cuda.empty_cache()
lc = extract_residual_coords(harmless_train[:N_FIT]); gc.collect(); torch.cuda.empty_cache()
all_c = np.concatenate([hc, lc], 0)
fit = fit_layer_dynamics(all_c[..., 0], all_c[..., 1], affine=True)
A, bvec = fit["A"], fit["b"]
harmful_mean, harmless_mean = hc.mean(0), lc.mean(0)
ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
sep = np.abs(np.arctan2(np.sin(ang(harmful_mean) - ang(harmless_mean)),
                        np.cos(ang(harmful_mean) - ang(harmless_mean))))
peak = int(np.argmax(sep)); hi = sep > 0.5
lo = peak;  hib = peak
while lo - 1 >= 0 and hi[lo - 1]: lo -= 1
while hib + 1 < L - 1 and hi[hib + 1]: hib += 1
band = list(range(lo, hib + 1))
ref = reference_trajectory(harmful_mean, harmless_mean, option="A")
ref_scale = float(np.linalg.norm(harmless_mean[band], axis=-1).mean())
fixed_angle = float(np.arctan2(np.sin(ang(harmless_mean[band])).mean(),
                               np.cos(ang(harmless_mean[band])).mean()))
print(f"  band {band[0]}..{band[-1]} ({len(band)}); ref_scale~{ref_scale:.2f}; "
      f"fixed_angle={np.degrees(fixed_angle):.0f}deg")

# ----------------------------------------------------------------- metrics
harm = harmful_test[:N_HARM]
clean = harmless_test[:N_CLEAN]      # on-axis (harmless distribution) PPL probe
# NEUTRAL, off-refusal-axis corpus — the honest general-capability probe (steering
# toward "harmless" should not help these the way it trivially helps harmless prompts).
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
    "The printing press, developed by Gutenberg, accelerated the spread of literacy.",
    "DNA is structured as a double helix held together by complementary base pairs.",
    "The Fibonacci sequence begins 0, 1, 1, 2, 3, 5, 8 and each term sums the prior two.",
    "Mount Everest, on the border of Nepal and China, is the highest peak above sea level.",
]
_n = tokenizer(NEUTRAL, return_tensors="pt", padding=True)
NEUTRAL_IDS = _n.input_ids.to(device)
NEUTRAL_ATTN = _n.attention_mask.to(device)


def effectiveness(policy):
    """mean refusal/compliance margin on harmful prompts (high=refuse).
    policy=None -> true baseline with no hooks."""
    hooks = attach_pts_hooks(module_dict, band, b1, b2, PTSState(policy)) if policy else []
    inp = tokenize_instructions_fn(harm, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    return margin_fn(out.logits[:, -1, :]).mean().item()


def realized_u(policy):
    state = PTSState(policy, record=True)
    hooks = attach_pts_hooks(module_dict, band, b1, b2, state)
    inp = tokenize_instructions_fn(harm, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            model(input_ids=ids, attention_mask=attn, position_ids=pos)
    us = []
    for k in band:
        if state.log.get(k, {}).get("u"):
            us.append(np.linalg.norm(np.asarray(state.log[k]["u"][-1]), axis=-1).mean())
    return float(np.mean(us)) if us else 0.0


def _nll(ids, attn, policy):
    """Mean teacher-forced token NLL (nats) with steering hooks ON (policy=None -> none)."""
    hooks = attach_pts_hooks(module_dict, band, b1, b2, PTSState(policy)) if policy else []
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn)
    logits = out.logits[:, :-1, :].float()
    labels = ids[:, 1:]
    mask = attn[:, 1:].bool()
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                          reduction="none").reshape(labels.shape)
    return float(nll[mask].mean().item())


def perplexity(policy):
    """On-axis capability probe: NLL on the harmless (Alpaca) distribution. CONFOUNDED
    with the steering direction (steering toward harmless trivially helps harmless text);
    reported only for contrast with the neutral probe."""
    inp = tokenize_instructions_fn(clean, tokenizer)
    return _nll(inp.input_ids.to(device), inp.attention_mask.to(device), policy)


def perplexity_neutral(policy):
    """Honest general-capability probe: NLL on a NEUTRAL off-axis corpus (encyclopedic
    / coding / math). A real capability tax shows up here as rising NLL."""
    return _nll(NEUTRAL_IDS, NEUTRAL_ATTN, policy)


# ----------------------------------------------------------------- sweep
print("\n[Exp 3/4] effectiveness vs capability frontier (reference = harmless mean):")
base_margin = effectiveness(None)
base_ppl = perplexity(None)
base_neu = perplexity_neutral(None)
print(f"  baseline (no steer): margin={base_margin:+.2f}  onaxis-NLL={base_ppl:.3f}  "
      f"NEUTRAL-NLL={base_neu:.3f}")

budgets = np.array([0.1, 0.2, 0.35, 0.5, 0.75, 1.0]) * ref_scale
rows = []
for um in budgets:
    mpc = MPCController(A, bvec, ref, layers=band, H=HORIZON, q_pos=1.0,
                        r_ctrl=0.02, qf_scale=4.0, u_max=um)
    pol = PolicyMPC(mpc)
    m = effectiveness(pol)
    ppl = perplexity(pol)
    neu = perplexity_neutral(pol)
    u = realized_u(pol)
    rows.append((um, m, ppl, u, neu))
    print(f"  PTS u_max={um:5.2f}: margin={m:+6.2f}  neutral-NLL={neu:6.3f} "
          f"(tax {neu-base_neu:+.3f})  onaxis {ppl-base_ppl:+.3f}  mean||u||={u:.2f}")

fix_margin = effectiveness(PolicyFixedAngle(fixed_angle, band))
fix_ppl = perplexity(PolicyFixedAngle(fixed_angle, band))
fix_neu = perplexity_neutral(PolicyFixedAngle(fixed_angle, band))
print(f"  fixed-angle ({np.degrees(fixed_angle):.0f}deg): margin={fix_margin:+.2f}  "
      f"neutral-NLL={fix_neu:.3f} (tax {fix_neu-base_neu:+.3f})  onaxis {fix_ppl-base_ppl:+.3f}")

rows = np.array(rows)
print("\n  --- frontier read (NEUTRAL = the honest capability tax) ---")
derf_pts = base_margin - rows[:, 1]
derf_fix = base_margin - fix_margin
j = int(np.argmin(np.abs(derf_pts - derf_fix)))
print(f"  fixed-angle de-refuses by {derf_fix:.1f} at neutral tax {fix_neu-base_neu:+.3f} NLL.")
print(f"  PTS matching that de-refusal (u_max={rows[j,0]:.2f}) costs neutral tax "
      f"{rows[j,4]-base_neu:+.3f} NLL "
      f"({'LOWER' if rows[j,4]<fix_neu else 'HIGHER/EQUAL'} than fixed-angle).")
print(f"  partial-steer option fixed-angle lacks: u_max={rows[1,0]:.2f} gives margin "
      f"{rows[1,1]:+.2f} (de-refuse {derf_pts[1]:.1f}) at neutral tax {rows[1,4]-base_neu:+.3f}.")

np.savez_compressed(
    pathlib.Path(__file__).resolve().parent / "pts_capability_results.npz",
    band=np.array(band), ref_scale=ref_scale, fixed_angle=fixed_angle,
    base_margin=base_margin, base_ppl=base_ppl, base_neu=base_neu,
    budgets=rows[:, 0], pts_margin=rows[:, 1], pts_ppl=rows[:, 2], pts_umax=rows[:, 3],
    pts_neu=rows[:, 4], fix_margin=fix_margin, fix_ppl=fix_ppl, fix_neu=fix_neu,
)
print("\nsaved pts_capability_results.npz")
