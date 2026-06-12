"""OAS prototype: Observer-Based, Soft-Landing Angular Steering in the loop on the
real model (Qwen2.5-3B-Instruct, A10G) — OAS_SPEC.md S5.

Mirrors pts_prototype.py structure: a config block, the steering plane + the fitted
2x2 affine plant on the REAL residual stream (extract_all_layer_activations /
compute_steering_plane + a residual-coord extractor), prefill_with_policy,
baseline_margin, and a generate helper. Unlike oas_offline.py (which validates the
control maths on saved trajectories, model-free), this drives the live model, so the
plant is the real nonlinear network and the fitted dynamics are only an approximation.

Four experiments, run in the gated order the proposal demands (S5):

  Exp 1 — AUTHORITY GATE (the headline kill-switch). Sweep the steered band WIDTH
          1 -> many contiguous layers around the steer layer, and for each width
          measure the REALIZED behavioural range (a G-sweep of a behavioural margin
          over many target angles) for (a) REFUSAL [known strong] and (b) a SUSTAINED
          SENTIMENT attribute built from generation-eliciting contrastive instructions
          ("write upbeat..." vs "write bleak...", reimplemented here — recipe READ
          from ../CLAS/clas_drift_test.py but NOT imported) plus an authority gate.
          Compare PolicyMultiAngle (fixed angle held across the band) against the
          soft-landing LQR. PASS iff multi-layer materially raises authority vs a
          single layer. This is the lever CLAS left open.

  Exp 2 — behavioural enforcement-layer sweep. Single-layer deadbeat at the target
          angle, slide the enforcement layer across the band, measure the behavioural
          margin -> behavioural-effect-vs-enforcement-layer curve (cross-checks the
          offline geometric curve and picks the terminal layer kT).

  Exp 3 — soft-landing vs deadbeat at MATCHED terminal effect. Coherence =
          KL(steered || unsteered) of the next-token distribution (+ a short-gen
          perplexity proxy); robustness = margin variance under a perturbed plant;
          plus qualitative generations. Predict soft-landing matches effect at lower
          coherence cost.

  Exp 4 — observer value under injected readout noise. Inject measurement noise into
          the per-layer coordinate read by the controller and compare LQG (Kalman-
          filtered estimate -> LQR) against LQR-on-raw: the filtered angle should be
          stabler / smoother (the principled EMA). Confirms the offline result in-model.

Imports ONLY the shared lib (utils, phase_portrait, observables), pts_dynamics (the
pure-numpy plant fit, treated as shared per OAS_SPEC S0), and OAS's own oas_* modules.
Never imports from ../CLAS or ../PTS (except pts_dynamics).

Run:  python oas_prototype.py              # full prototype (expensive)
      python oas_prototype.py --smoke       # tiny N / few layers / short gen (dry-run)
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import pathlib
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- sys.path shim (OAS_SPEC.md S0): expose BOTH the shared lib and the sibling PTS
# --- dir, so `from utils import ...`, `from phase_portrait import ...`,
# --- `from observables import ...`, `from pts_dynamics import ...` and the local
# --- `from oas_* import ...` all resolve no matter where python is launched from.
_R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_R / "pytorch_pure"))
sys.path.insert(0, str(_R / "research_proposals" / "PTS"))

from utils import get_input_data, tokenize_instructions_fn, add_hooks
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from observables import make_margin_fn
from pts_dynamics import fit_layer_dynamics, validate_dynamics

from oas_observer import estimate_process_noise
from oas_controller import (OASState, attach_oas_hooks, build_softlanding,
                            PolicyNone, PolicyDeadbeat, PolicyMultiAngle,
                            PolicySoftLandingLQR, PolicyLQG)


# ----------------------------------------------------------------------------- logging
class _Tee:
    """Mirror stdout to a logfile (oas_prototype.log), like a `tee`."""

    def __init__(self, path):
        self.term = sys.stdout
        self.fh = open(path, "w")

    def write(self, s):
        self.term.write(s)
        self.fh.write(s)

    def flush(self):
        self.term.flush()
        self.fh.flush()


# ----------------------------------------------------------------------------- config
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
POSITION = "mid"

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true",
                help="tiny N / few layers / short gen — dry-run the whole pipeline")
ap.add_argument("--seed", type=int, default=0)
args, _ = ap.parse_known_args()
SMOKE = args.smoke

if SMOKE:
    PLANE_SAMPLES = 16        # prompts per class for the plane
    N_FIT = 16                # prompts per class for the dynamics fit
    N_OBS = 4                 # prompts for the behavioural sweeps
    N_GEN = 2                 # prompts for qualitative generation
    MAX_NEW_TOKENS = 8
    N_GRID = 4                # G-sweep grid (target angles) in Exp 1
    MAX_WIDTH = 3             # max band width tested in Exp 1
    N_ENF = 3                 # enforcement layers sampled in Exp 2
    N_RHO = 3                 # rho values in Exp 3
    N_PERTURB = 4             # plant-perturbation Monte-Carlo draws (Exp 3 robustness)
    N_NOISE = 3               # noise levels in Exp 4
    FIT_BATCH = 8
else:
    PLANE_SAMPLES = 192
    N_FIT = 192
    N_OBS = 24
    N_GEN = 4
    MAX_NEW_TOKENS = 40
    N_GRID = 12
    MAX_WIDTH = 9
    N_ENF = 8
    N_RHO = 6
    N_PERTURB = 16
    N_NOISE = 5
    FIT_BATCH = 8

torch.manual_seed(args.seed)
np.random.seed(args.seed)

sys.stdout = _Tee("oas_prototype.log")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== OAS prototype {'[SMOKE]' if SMOKE else '[FULL]'} ===  device={device}")


# ----------------------------------------------------------------------------- model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
module_dict = dict(model.named_modules())
L = model.config.num_hidden_layers
print(f"loaded {MODEL_ID}: {L} layers")


# ----------------------------------------------------------------------------- helpers
def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def circular_mean(angles):
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def find_band(harmful_mean, harmless_mean, thresh=0.5):
    """Refusal-angle-separation band: the contiguous run around the peak where the
    harmful/harmless mean trajectories are angularly separated (> thresh rad). Same
    recipe as pts_offline / pts_prototype, so OAS shares PTS's steering band."""
    ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
    sep = np.abs(_wrap(ang(harmful_mean) - ang(harmless_mean)))
    peak = int(np.argmax(sep))
    hi = sep > thresh
    lo = peak
    while lo - 1 >= 0 and hi[lo - 1]:
        lo -= 1
    top = peak
    while top + 1 < L - 1 and hi[top + 1]:
        top += 1
    return list(range(lo, top + 1)), sep, peak


def make_residual_extractor(b1, b2):
    """Build a closure that returns last-token (b1,b2) coords at EVERY layer output
    (the residual stream we actuate). Returns (N, L, 2). Tuple-aware hooks on
    model.layers.{k} — identical to pts_prototype.extract_residual_coords."""

    def extract(prompts, batch_size=FIT_BATCH):
        cache = {k: [] for k in range(L)}

        def mk(k):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                last = h[:, -1, :]
                _b1 = b1.to(last.dtype); _b2 = b2.to(last.dtype)
                c1 = (last @ _b1).float().cpu()
                c2 = (last @ _b2).float().cpu()
                cache[k].append(torch.stack([c1, c2], -1))
            return hook

        hooks = [(module_dict[f"model.layers.{k}"], mk(k)) for k in range(L)]
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    inp = tokenize_instructions_fn(prompts[i:i + batch_size], tokenizer)
                    model(input_ids=inp.input_ids.to(device),
                          attention_mask=inp.attention_mask.to(device))
        return np.stack([torch.cat(cache[k], 0).numpy() for k in range(L)], axis=1)

    return extract


# ----------------------------------------------------------------------------- sentiment plane recipe
# Reimplemented from ../CLAS/clas_drift_test.py (READ, NOT imported): the sentiment
# direction that controls the model's OWN output is the diff-in-means of
# generation-eliciting contrastive instructions ("write upbeat..." vs "write
# bleak..."), NOT a read-a-review direction.
_SENT_TOPICS = [
    "the weather", "a city", "food", "work", "travel", "music", "mornings",
    "the future", "a walk in the park", "technology", "friends", "the ocean",
    "school", "a holiday", "a new job", "the economy", "sports", "art", "cooking",
    "the news", "growing older", "a small town", "the internet", "family dinners",
    "winter", "summer", "a long drive", "moving house", "a first date", "retirement",
    "coffee", "books", "the night sky", "a garden", "city traffic", "a rainy day",
]
_SENT_POS_T = "Write an upbeat, joyful, deeply positive and optimistic paragraph about {}."
_SENT_NEG_T = "Write a bleak, miserable, deeply negative and pessimistic paragraph about {}."

_SENT_POS_WORDS = ["great", "good", "love", "wonderful", "amazing", "happy", "beautiful",
                   "excellent", "best", "joy", "joyful", "fantastic", "delight",
                   "brilliant", "perfect", "enjoy", "lovely", "bright", "hopeful", "warm"]
_SENT_NEG_WORDS = ["bad", "terrible", "hate", "awful", "worst", "sad", "ugly", "horrible",
                   "disappointing", "fail", "poor", "boring", "disgusting", "miserable",
                   "wrong", "bleak", "dark", "cold", "grim", "dreadful"]


def build_sentiment_contrastive(n):
    pos, neg, i = [], [], 0
    while len(pos) < n:
        t = _SENT_TOPICS[i % len(_SENT_TOPICS)]
        pos.append(_SENT_POS_T.format(t)); neg.append(_SENT_NEG_T.format(t)); i += 1
    return pos[:n], neg[:n]


def make_sentiment_margin_fn():
    """log P(positive word) - log P(negative word) on the next-token distribution —
    the sustained-attribute readout (reimplemented from the CLAS drift test)."""
    def id_set(words):
        ids = set()
        for w in words:
            for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
                enc = tokenizer.encode(v, add_special_tokens=False)
                if enc:
                    ids.add(enc[0])
        return torch.tensor(sorted(ids), device=device)

    POS, NEG = id_set(_SENT_POS_WORDS), id_set(_SENT_NEG_WORDS)

    def margin(logits):
        lp = F.log_softmax(logits.float(), dim=-1)
        return torch.logsumexp(lp[:, POS], dim=-1) - torch.logsumexp(lp[:, NEG], dim=-1)

    return margin


# ----------------------------------------------------------------------------- data + REFUSAL plane/plant
print("\n[setup] refusal plane + plant on the real residual stream...")
harmful_train, harmful_test = get_input_data("harmful", "en")
harmless_train, harmless_test = get_input_data("harmless", "en")

ha = extract_all_layer_activations(model, harmful_train[:PLANE_SAMPLES], tokenizer, [POSITION], FIT_BATCH)
gc.collect(); torch.cuda.empty_cache()
la = extract_all_layer_activations(model, harmless_train[:PLANE_SAMPLES], tokenizer, [POSITION], FIT_BATCH)
gc.collect(); torch.cuda.empty_cache()
plane = compute_steering_plane(ha, la)
b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
STEER_LAYER = int(plane["selected_key"].split("_")[1])
del ha, la; gc.collect(); torch.cuda.empty_cache()
print(f"  refusal plane: {plane['selected_key']} -> steer layer {STEER_LAYER}; "
      f"b1.b2={(b1 @ b2).item():.2e}")

refusal_margin_fn, _R_ids, _C_ids = make_margin_fn(tokenizer, device)

extract_residual_coords = make_residual_extractor(b1, b2)

# fit the 2x2 affine plant on the REAL residual stream (the point we actuate)
print("  fitting 2x2 dynamics on the real residual stream...")
hc = extract_residual_coords(harmful_train[:N_FIT]); gc.collect(); torch.cuda.empty_cache()
lc = extract_residual_coords(harmless_train[:N_FIT]); gc.collect(); torch.cuda.empty_cache()
all_c = np.concatenate([hc, lc], 0)
nall = all_c.shape[0]
perm = np.random.RandomState(args.seed).permutation(nall)
tr, te = perm[:nall // 2], perm[nall // 2:]
fit = fit_layer_dynamics(all_c[tr, :, 0], all_c[tr, :, 1], affine=True)
val = validate_dynamics(all_c[te, :, 0], all_c[te, :, 1], fit, horizons=(1, 5))
A, bvec = fit["A"], fit["b"]
print(f"  held-out R2: 1-step={val['r2_h1']:.4f}  5-step={val['r2_h5']:.4f}  "
      f"NRMSE(1-step)={val['nrmse_h1']:.4f}")

# process-noise covariance for the observer (the PTS ~14% residual as a covariance)
W_pool, W_per_layer = estimate_process_noise(A, bvec, all_c)
print(f"  process-noise W (pooled): diag={np.diag(W_pool)}  "
      f"||W||_2={np.linalg.norm(W_pool, 2):.3f}")

harmful_mean = hc.mean(0)
harmless_mean = lc.mean(0)
band, angle_sep, peak = find_band(harmful_mean, harmless_mean)
# guard: ensure the band is wide enough to actually sweep widths in smoke mode.
if len(band) < 2:
    c = STEER_LAYER
    band = list(range(max(0, c - 1), min(L - 1, c + 2)))
ref_scale = float(np.linalg.norm(harmless_mean[band], axis=-1).mean())
kT = band[-1]
print(f"  refusal steering band (|angle_sep|>0.5 around layer {peak}): "
      f"layers {band[0]}..{band[-1]} ({len(band)} layers); ref_scale~{ref_scale:.2f}; kT={kT}")

# the single best fixed refusal angle = circular mean of the harmless-ref band angle
ang = lambda v: np.arctan2(v[:, 1], v[:, 0])
refusal_target_ang = circular_mean(ang(harmless_mean[band]))
# the target coordinate at the target angle with the natural (band-mean) magnitude
refusal_target = ref_scale * np.array([math.cos(refusal_target_ang),
                                       math.sin(refusal_target_ang)])
print(f"  best fixed refusal (de-refusal) angle = {math.degrees(refusal_target_ang):.0f} deg")

# rotation magnitude of the plant per layer (the soft-landing-advantage predictor)
rot_mag = np.array([np.linalg.norm(A[k] - np.eye(2), 2) for k in range(L - 1)])
print(f"  ||A_k - I||_2 over band = {rot_mag[band].mean():.3f} "
      f"(small -> near-identity late band; soft-landing wins predicted where this is large)")


# ----------------------------------------------------------------------------- prefill / generate harness
def prefill_with_policy(prompts, layers, policy, margin_fn, record=False):
    """Single steered forward pass over the actuated `layers`. Returns (margin, state,
    logits). Mirrors pts_prototype.prefill_with_policy; resets the policy's
    per-forward-pass state (the LQG filter) first, exactly as OAS_SPEC S3 requires."""
    state = OASState(policy, record=record)
    state.reset_policy()                       # fresh observer per pass (S3)
    hooks = attach_oas_hooks(module_dict, layers, b1, b2, state)
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    logits = out.logits[:, -1, :]
    return margin_fn(logits), state, logits


def baseline_logits(prompts):
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    return out.logits[:, -1, :]


def baseline_margin(prompts, margin_fn):
    return margin_fn(baseline_logits(prompts))


def generate(prompts, layers, policy, n=MAX_NEW_TOKENS):
    state = OASState(policy)
    state.reset_policy()
    hooks = attach_oas_hooks(module_dict, layers, b1, b2, state)
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            g = model.generate(ids, attention_mask=attn, max_new_tokens=n,
                               do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)


def behavioural_range(prompts, layers, margin_fn, n_grid, policy_factory):
    """Realized behavioural range: sweep `n_grid` target angles over the full circle,
    apply the policy at each angle across `layers`, return per-angle mean margins. The
    range (max - min) is the steering AUTHORITY at this bandwidth (the G-sweep)."""
    thetas = np.linspace(-math.pi, math.pi, n_grid, endpoint=False)
    margins = []
    for th in thetas:
        pol = policy_factory(float(th), layers)
        m, _, _ = prefill_with_policy(prompts, layers, pol, margin_fn)
        margins.append(float(m.mean()))
    return thetas, np.array(margins)


def contiguous_widths(center_layer, max_width):
    """Bands of growing width centred (as best as possible) on the steer layer,
    width 1, 2, 3, ... clamped to [0, L-2] (we never actuate the final layer)."""
    bands = []
    for w in range(1, max_width + 1):
        half = (w - 1) // 2
        lo = center_layer - half
        hi = lo + w - 1
        lo = max(0, lo); hi = min(L - 2, lo + w - 1); lo = max(0, hi - w + 1)
        bands.append(list(range(lo, hi + 1)))
    return bands


# =============================================================================
# Exp 1 — AUTHORITY GATE (run FIRST; the headline kill-switch)
# =============================================================================
print("\n" + "=" * 72)
print("[Exp 1] AUTHORITY GATE — does multi-layer rotation beat single-layer?")
print("=" * 72)

exp1 = {}   # attribute -> dict of arrays for saving


def softlanding_factory(target_ang_scalar, layers):
    """A factory making a soft-landing LQR policy that lands `target_ang_scalar` at
    the band's terminal layer (used as a PolicyMultiAngle alternative in Exp 1)."""
    tgt = ref_scale * np.array([math.cos(target_ang_scalar), math.sin(target_ang_scalar)])
    gains = build_softlanding(A, bvec, layers, ref=None, target=tgt,
                              R_rho=0.05, Q_term=50.0, Q_stage=0.0)
    return PolicySoftLandingLQR(gains, ref=None)


for attr_name, margin_fn, obs_prompts in [
    ("refusal", refusal_margin_fn, harmful_test[:N_OBS]),
]:
    print(f"\n  --- attribute: {attr_name} ---")
    widths = contiguous_widths(STEER_LAYER if STEER_LAYER <= L - 2 else kT, MAX_WIDTH)
    auth_multi, auth_soft, band_widths = [], [], []
    for bnd in widths:
        w = len(bnd)
        # PolicyMultiAngle: SAME fixed angle held across the band (the Exp-1 lever)
        _, m_multi = behavioural_range(obs_prompts, bnd, margin_fn, N_GRID,
                                       lambda th, ly: PolicyMultiAngle(th, ly))
        rng_multi = float(m_multi.max() - m_multi.min())
        # soft-landing LQR: lands the angle only at the terminal layer
        _, m_soft = behavioural_range(obs_prompts, bnd, margin_fn, N_GRID,
                                      softlanding_factory)
        rng_soft = float(m_soft.max() - m_soft.min())
        auth_multi.append(rng_multi); auth_soft.append(rng_soft); band_widths.append(w)
        print(f"    width={w:2d} (layers {bnd[0]}..{bnd[-1]}): "
              f"multi-angle authority={rng_multi:6.3f}  soft-landing authority={rng_soft:6.3f}")
        gc.collect(); torch.cuda.empty_cache()
    auth_multi = np.array(auth_multi); auth_soft = np.array(auth_soft)
    band_widths = np.array(band_widths)
    single = auth_multi[0]
    best = auth_multi.max()
    lift = (best - single) / (abs(single) + 1e-9)
    verdict = "PASS (multi-layer raises authority)" if best > single + 0.1 else \
              "FAIL (multi-layer does not materially raise authority)"
    print(f"    GATE [{attr_name}]: single-layer={single:.3f} -> best multi={best:.3f} "
          f"(+{lift*100:.0f}%) -> {verdict}")
    exp1[attr_name] = {"widths": band_widths, "auth_multi": auth_multi,
                       "auth_soft": auth_soft, "single": single, "best": best}

# sentiment attribute: build its OWN plane (generation-eliciting contrastive), then
# reuse the same machinery. We rebuild the plane+plant+band for the sentiment plane.
print("\n  --- attribute: sentiment (own plane: write-upbeat vs write-bleak) ---")
pos_txt, neg_txt = build_sentiment_contrastive(PLANE_SAMPLES)
pos_acts = extract_all_layer_activations(model, pos_txt, tokenizer, [POSITION], FIT_BATCH)
gc.collect(); torch.cuda.empty_cache()
neg_acts = extract_all_layer_activations(model, neg_txt, tokenizer, [POSITION], FIT_BATCH)
gc.collect(); torch.cuda.empty_cache()
# compute_steering_plane(A,B) orients b1 toward B's pole; (pos,neg) -> b1 ~ negative
sent_plane = compute_steering_plane(pos_acts, neg_acts)
sb1 = sent_plane["b1"].to(device); sb2 = sent_plane["b2"].to(device)
SENT_STEER_LAYER = int(sent_plane["selected_key"].split("_")[1])
del pos_acts, neg_acts; gc.collect(); torch.cuda.empty_cache()
print(f"    sentiment plane: {sent_plane['selected_key']} -> steer layer {SENT_STEER_LAYER}")

sent_margin_fn = make_sentiment_margin_fn()
NEUTRAL_PROMPTS = [
    "Tell me about a typical morning routine.",
    "Describe what a city looks like at night.",
    "Write a short story about a journey through a forest.",
    "Share your thoughts on modern technology.",
    "Describe a meal someone might cook for friends.",
    "Tell me about the changing of the seasons.",
]
sent_obs = NEUTRAL_PROMPTS[:N_OBS]

# the sentiment sweep is a fixed-angle G-sweep on the sentiment plane at growing band
# widths. The actuator must use the SENTIMENT basis, so we temporarily swap b1/b2 via
# a thin extractor-free path: PolicyMultiAngle only needs the angle, and the hook reads
# whichever b1/b2 it's attached with. We attach hooks with sb1/sb2 here.
def sent_prefill(prompts, layers, policy, margin_fn):
    state = OASState(policy)
    state.reset_policy()
    hooks = attach_oas_hooks(module_dict, layers, sb1, sb2, state)
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    return margin_fn(out.logits[:, -1, :])


def sent_behavioural_range(prompts, layers, n_grid):
    thetas = np.linspace(-math.pi, math.pi, n_grid, endpoint=False)
    margins = []
    for th in thetas:
        m = sent_prefill(prompts, layers, PolicyMultiAngle(float(th), layers),
                         sent_margin_fn)
        margins.append(float(m.mean()))
    return thetas, np.array(margins)


sent_widths = contiguous_widths(SENT_STEER_LAYER if SENT_STEER_LAYER <= L - 2
                                else L - 2, MAX_WIDTH)
sent_auth, sent_band_w = [], []
for bnd in sent_widths:
    _, m = sent_behavioural_range(sent_obs, bnd, N_GRID)
    rng = float(m.max() - m.min())
    sent_auth.append(rng); sent_band_w.append(len(bnd))
    print(f"    width={len(bnd):2d} (layers {bnd[0]}..{bnd[-1]}): "
          f"sentiment authority (G-sweep range)={rng:6.3f}")
    gc.collect(); torch.cuda.empty_cache()
sent_auth = np.array(sent_auth); sent_band_w = np.array(sent_band_w)
s_single, s_best = sent_auth[0], sent_auth.max()
s_verdict = "PASS" if s_best > s_single + 0.1 else "FAIL (weak sustained-attribute authority)"
print(f"    GATE [sentiment]: single={s_single:.3f} -> best multi={s_best:.3f} "
      f"(+{(s_best-s_single)/(abs(s_single)+1e-9)*100:.0f}%) -> {s_verdict}")
exp1["sentiment"] = {"widths": sent_band_w, "auth_multi": sent_auth,
                     "single": s_single, "best": s_best}


# =============================================================================
# Exp 2 — behavioural enforcement-layer sweep (single-layer deadbeat)
# =============================================================================
print("\n" + "=" * 72)
print("[Exp 2] enforcement-layer sweep — single-layer deadbeat at the target angle")
print("=" * 72)
enf_layers = sorted(set(int(round(x)) for x in
                        np.linspace(band[0], band[-1], min(N_ENF, len(band)))))
enf_prompts = harmful_test[:N_OBS]
base_ref = float(baseline_margin(enf_prompts, refusal_margin_fn).mean())
enf_margins = []
for ly in enf_layers:
    pol = PolicyDeadbeat(refusal_target_ang, ly)
    m, _, _ = prefill_with_policy(enf_prompts, [ly], pol, refusal_margin_fn)
    enf_margins.append(float(m.mean()))
    print(f"    enforce@layer {ly:2d}: refusal margin={enf_margins[-1]:+.3f} "
          f"(baseline {base_ref:+.3f}; lower = more de-refused)")
    gc.collect(); torch.cuda.empty_cache()
enf_margins = np.array(enf_margins)
best_enf = enf_layers[int(np.argmin(enf_margins))]
print(f"    -> strongest de-refusal at enforcement layer {best_enf} "
      f"(behavioural-effect-vs-enforcement-layer curve; cross-checks the offline kT)")


# =============================================================================
# Exp 3 — soft-landing vs deadbeat at matched terminal effect
# =============================================================================
print("\n" + "=" * 72)
print("[Exp 3] soft-landing vs deadbeat — coherence (KL) & robustness at matched effect")
print("=" * 72)
exp3_prompts = harmful_test[:N_OBS]
unsteered_logits = baseline_logits(exp3_prompts)
unsteered_lp = F.log_softmax(unsteered_logits.float(), dim=-1)


def kl_to_unsteered(steered_logits):
    """KL(steered || unsteered) of the next-token distribution, per prompt -> mean.
    The coherence proxy (S5 Exp 3): a smaller distributional shift at matched
    behavioural effect = cheaper coherence cost."""
    sp = F.log_softmax(steered_logits.float(), dim=-1)
    p = sp.exp()
    kl = (p * (sp - unsteered_lp)).sum(-1)
    return float(kl.mean()), kl.detach().float().cpu().numpy()


# deadbeat baseline: single-layer slam at the (Exp-2-selected) terminal layer
db_pol = PolicyDeadbeat(refusal_target_ang, kT)
m_db, _, logits_db = prefill_with_policy(exp3_prompts, [kT], db_pol, refusal_margin_fn)
kl_db, kl_db_arr = kl_to_unsteered(logits_db)
eff_db = float(m_db.mean())
# baseline (unsteered) margin sets the SCALE of a meaningful behavioural effect: a
# coherence comparison "at matched effect" is only valid if the soft-landing actually
# reaches the deadbeat's effect, measured as a fraction of the deadbeat's own swing
# off baseline (|baseline - eff_db|). (F2)
base_eff3 = float(refusal_margin_fn(unsteered_logits).mean())
db_swing = abs(base_eff3 - eff_db)
print(f"    deadbeat (layer {kT}): effect(margin)={eff_db:+.3f}  KL={kl_db:.4f}  "
      f"(baseline {base_eff3:+.3f}; deadbeat swing {db_swing:.3f})")

# soft-landing LQR over the band: sweep rho to trace the coherence-vs-effect frontier,
# and find the rho whose effect best MATCHES the deadbeat's, then compare KL there. The
# grid is extended DOWN toward rho->0 (the deadbeat corner) so the soft-landing band
# has the chance to actually reach the single-layer slam's effect (F2).
rhos = np.geomspace(1e-4, 1.0, N_RHO)
sl_eff, sl_kl = [], []
for rho in rhos:
    gains = build_softlanding(A, bvec, band, ref=None, target=refusal_target,
                              R_rho=float(rho), Q_term=50.0, Q_stage=0.0)
    pol = PolicySoftLandingLQR(gains, ref=None)
    m, _, logits = prefill_with_policy(exp3_prompts, band, pol, refusal_margin_fn)
    klv, _ = kl_to_unsteered(logits)
    sl_eff.append(float(m.mean())); sl_kl.append(klv)
    print(f"    soft-landing rho={rho:8.5f}: effect={sl_eff[-1]:+.3f}  KL={klv:.4f}")
    gc.collect(); torch.cuda.empty_cache()
sl_eff = np.array(sl_eff); sl_kl = np.array(sl_kl)
j_match = int(np.argmin(np.abs(sl_eff - eff_db)))
eff_gap = float(abs(sl_eff[j_match] - eff_db))
print(f"    -> closest soft-landing: rho={rhos[j_match]:.5f} effect={sl_eff[j_match]:+.3f} "
      f"KL={sl_kl[j_match]:.4f}  (effect gap to deadbeat = {eff_gap:.3f})")
# Effect-match guard: declare a coherence comparison VALID only when the soft-landing
# reaches within EFF_MATCH_FRAC of the deadbeat's effect (a fraction of the deadbeat's
# own swing off baseline). Otherwise a lower KL is just the trivial consequence of a
# weaker behavioural effect, NOT a coherence advantage at matched effect (F2).
EFF_MATCH_FRAC = 0.25
eff_matched = eff_gap <= EFF_MATCH_FRAC * (db_swing + 1e-9)
if eff_matched:
    coherence_win = sl_kl[j_match] < kl_db
    print(f"    [matched: gap {eff_gap:.3f} <= {EFF_MATCH_FRAC:.0%} of swing {db_swing:.3f}] "
          f"soft-landing coherence {'WIN' if coherence_win else 'no-win'} "
          f"(KL at matched effect): {sl_kl[j_match]:.4f} vs {kl_db:.4f}")
else:
    coherence_win = False
    print(f"    [EFFECT NOT MATCHED: gap {eff_gap:.3f} > {EFF_MATCH_FRAC:.0%} of swing "
          f"{db_swing:.3f}] soft-landing could NOT reach the deadbeat effect over the rho "
          f"grid -> the KL difference ({sl_kl[j_match]:.4f} vs {kl_db:.4f}) is NOT a "
          f"matched-effect coherence claim. This is also evidence the soft-landing band "
          f"lacks the authority to match the single-layer slam at this kT.")

# robustness: terminal-effect variance under a perturbed plant. To be a FAIR
# comparison both controllers must be exposed to the SAME plant-model error (F3): the
# soft-landing's gains are re-solved on the perturbed (Ap, bp), AND the deadbeat's
# commanded angle is DERIVED from the perturbed plant (the actuated angle whose
# perturbed-plant image at kT lands on the refusal target, angle(Ap[kT]^{-1}(target -
# bp[kT]))). So a plant error propagates into BOTH; a plant-free deadbeat (constant
# angle, std==0 by construction) would NOT test robustness at all.
print("    robustness — effect spread under a perturbed plant (delta-perturbed A,b; "
      "BOTH controllers see the perturbation):")
rng_pert = np.random.RandomState(args.seed + 1)
delta = 0.05
db_effs, sl_effs = [], []
for _ in range(N_PERTURB):
    Ap = A + delta * rng_pert.randn(*A.shape)
    bp = bvec + delta * ref_scale * rng_pert.randn(*bvec.shape)
    tgt = refusal_target
    # soft-landing: gains re-solved on the perturbed plant.
    g_sl = build_softlanding(Ap, bp, band, ref=None, target=tgt,
                             R_rho=float(rhos[j_match]), Q_term=50.0, Q_stage=0.0)
    m_sl, _, _ = prefill_with_policy(exp3_prompts, band, PolicySoftLandingLQR(g_sl, None),
                                     refusal_margin_fn)
    sl_effs.append(float(m_sl.mean()))
    # deadbeat: target ANGLE derived from the perturbed plant's pre-image at kT, so a
    # plant-model error moves the deadbeat's commanded angle too (equal-footing).
    s_pre = np.linalg.solve(Ap[kT], tgt - bp[kT])           # Ap[kT]^{-1}(target - bp[kT])
    db_ang_p = float(math.atan2(s_pre[1], s_pre[0]))
    m_dbp, _, _ = prefill_with_policy(exp3_prompts, [kT],
                                      PolicyDeadbeat(db_ang_p, kT),
                                      refusal_margin_fn)
    db_effs.append(float(m_dbp.mean()))
    gc.collect(); torch.cuda.empty_cache()
db_effs = np.array(db_effs); sl_effs = np.array(sl_effs)
# Report the MEASURED spreads on equal footing — no a-priori narrative about which
# "should" win (F3). Smaller std = less sensitive to the ~14% plant-model error.
rob_verdict = ("soft-landing is LESS sensitive" if sl_effs.std() < db_effs.std()
               else "deadbeat is LESS sensitive" if db_effs.std() < sl_effs.std()
               else "tie")
print(f"      deadbeat   effect std under plant noise = {db_effs.std():.4f} "
      f"(mean {db_effs.mean():+.3f})")
print(f"      softlanding effect std under plant noise = {sl_effs.std():.4f} "
      f"(mean {sl_effs.mean():+.3f})  -> measured: {rob_verdict}")

# qualitative generations under each controller
print("\n    qualitative generations (greedy):")
gen_prompts = harmful_test[:N_GEN]
g_base = generate(gen_prompts, band, PolicyNone())
g_db = generate(gen_prompts, [kT], PolicyDeadbeat(refusal_target_ang, kT))
g_sl_gains = build_softlanding(A, bvec, band, ref=None, target=refusal_target,
                               R_rho=float(rhos[j_match]), Q_term=50.0, Q_stage=0.0)
g_sl = generate(gen_prompts, band, PolicySoftLandingLQR(g_sl_gains, None))
for i, p in enumerate(gen_prompts):
    print(f"      prompt: {p[:60]}")
    print(f"        baseline    : {g_base[i][:80]!r}")
    print(f"        deadbeat    : {g_db[i][:80]!r}")
    print(f"        soft-landing: {g_sl[i][:80]!r}")


# =============================================================================
# Exp 4 — observer value under injected readout noise (LQG vs LQR-on-raw)
# =============================================================================
print("\n" + "=" * 72)
print("[Exp 4] observer value — LQG (Kalman) vs LQR-on-raw under injected readout noise")
print("=" * 72)


class _NoisyCoordPolicy:
    """Wrap a base policy so the coordinate it SEES is corrupted by Gaussian readout
    noise (the injected measurement noise V). LQR-on-raw consumes the noisy coord
    directly; LQG (PolicyLQG) instead filters it through the Kalman observer. We
    compare the realized-angle SMOOTHNESS / effect stability across the band."""

    def __init__(self, base, noise_std, seed):
        self.base = base
        self.noise_std = float(noise_std)
        self.rng = np.random.RandomState(seed)
        if hasattr(base, "reset"):
            self.reset = base.reset  # propagate the LQG filter reset

    def __call__(self, layer_idx, coords):
        if self.noise_std > 0:
            noise = torch.from_numpy(
                self.noise_std * self.rng.randn(*coords.shape)).to(coords.dtype).to(coords.device)
            coords = coords + noise
        return self.base(layer_idx, coords)


exp4_prompts = harmful_test[:N_OBS]
noise_levels = np.linspace(0.0, 0.6 * ref_scale, N_NOISE)
# LQG kalman config: measure the coordinate (H=I), process noise = the fitted W,
# measurement noise V = the injected readout-noise covariance (set per level below).
sl_gains_e4 = build_softlanding(A, bvec, band, ref=None, target=refusal_target,
                                R_rho=0.05, Q_term=50.0, Q_stage=0.0)
raw_smooth, lqg_smooth, raw_eff, lqg_eff = [], [], [], []
for v_std in noise_levels:
    Vmat = (v_std ** 2 + 1e-6) * np.eye(2)
    kcfg = {"A": A, "b": bvec, "H": np.eye(2), "W": W_pool, "V": Vmat, "P0": np.eye(2)}
    # LQR-on-raw: soft-landing LQR consuming the NOISY coordinate directly.
    raw_pol = _NoisyCoordPolicy(PolicySoftLandingLQR(sl_gains_e4, None), v_std, args.seed + 7)
    m_raw, st_raw, _ = prefill_with_policy(exp4_prompts, band, raw_pol,
                                           refusal_margin_fn, record=True)
    # LQG: same noisy coordinate, but Kalman-filtered before the LQR.
    lqg_pol = _NoisyCoordPolicy(PolicyLQG(sl_gains_e4, None, kcfg), v_std, args.seed + 7)
    m_lqg, st_lqg, _ = prefill_with_policy(exp4_prompts, band, lqg_pol,
                                           refusal_margin_fn, record=True)

    # smoothness = mean across-band step-to-step variation of the realized angle
    # (last-token), pooled over prompts. The Kalman estimate should be smoother.
    def angle_roughness(state):
        thetas = []
        for k in band:
            if k in state.log and state.log[k]["theta"]:
                thetas.append(state.log[k]["theta"][-1])  # (M,)
        if len(thetas) < 2:
            return float("nan")
        T = np.stack(thetas, 0)                            # (n_band, M)
        d = _wrap(np.diff(T, axis=0))                      # wrapped step deltas
        return float(np.mean(np.abs(d)))

    rr, lr = angle_roughness(st_raw), angle_roughness(st_lqg)
    raw_smooth.append(rr); lqg_smooth.append(lr)
    raw_eff.append(float(m_raw.mean())); lqg_eff.append(float(m_lqg.mean()))
    print(f"    V_std={v_std:6.3f}: angle roughness raw={rr:.4f} vs LQG={lr:.4f} "
          f"| effect raw={raw_eff[-1]:+.3f} LQG={lqg_eff[-1]:+.3f}")
    gc.collect(); torch.cuda.empty_cache()
raw_smooth = np.array(raw_smooth); lqg_smooth = np.array(lqg_smooth)
raw_eff = np.array(raw_eff); lqg_eff = np.array(lqg_eff)
# the headline: as noise grows, LQG should be smoother (smaller roughness) than raw.
hi = -1
print(f"    -> at highest noise V_std={noise_levels[hi]:.3f}: LQG roughness "
      f"{lqg_smooth[hi]:.4f} {'<' if lqg_smooth[hi] < raw_smooth[hi] else '>='} "
      f"raw {raw_smooth[hi]:.4f} "
      f"({'observer earns its place (smoother)' if lqg_smooth[hi] < raw_smooth[hi] else 'no clear smoothing'})")


# =============================================================================
# save
# =============================================================================
save = dict(
    smoke=SMOKE, L=L, steer_layer=STEER_LAYER, band=np.array(band),
    r2_h1=val["r2_h1"], r2_h5=val["r2_h5"], nrmse_h1=val["nrmse_h1"],
    angle_sep=angle_sep, ref_scale=ref_scale, kT=kT, rot_mag=rot_mag,
    refusal_target_ang=refusal_target_ang, W_pool=W_pool,
    # Exp 1
    exp1_refusal_widths=exp1["refusal"]["widths"],
    exp1_refusal_auth_multi=exp1["refusal"]["auth_multi"],
    exp1_refusal_auth_soft=exp1["refusal"]["auth_soft"],
    exp1_sent_widths=exp1["sentiment"]["widths"],
    exp1_sent_auth_multi=exp1["sentiment"]["auth_multi"],
    sent_steer_layer=SENT_STEER_LAYER,
    # Exp 2
    exp2_enf_layers=np.array(enf_layers), exp2_enf_margins=enf_margins,
    exp2_baseline=base_ref, exp2_best_enf=best_enf,
    # Exp 3
    exp3_rhos=rhos, exp3_sl_eff=sl_eff, exp3_sl_kl=sl_kl,
    exp3_db_eff=eff_db, exp3_db_kl=kl_db, exp3_match_idx=j_match,
    exp3_baseline_eff=base_eff3, exp3_db_swing=db_swing,
    exp3_eff_gap=eff_gap, exp3_eff_matched=eff_matched,
    exp3_db_effs_perturbed=db_effs, exp3_sl_effs_perturbed=sl_effs,
    # Exp 4
    exp4_noise=noise_levels, exp4_raw_smooth=raw_smooth, exp4_lqg_smooth=lqg_smooth,
    exp4_raw_eff=raw_eff, exp4_lqg_eff=lqg_eff,
)
np.savez_compressed("oas_prototype_results.npz", **save)
print("\nsaved oas_prototype_results.npz + oas_prototype.log")
print(f"=== OAS prototype {'[SMOKE]' if SMOKE else '[FULL]'} DONE ===")
