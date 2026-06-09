"""CLAS falsification gate — the DRIFT TEST.

Question: under a FIXED steering angle, does a sustained behavioral attribute
(sentiment) hold across a long generation, or does it drift? The CLAS per-token
thermostat is only justified if fixed-angle steering FAILS to hold the attribute.

Modes compared (all at the same steered layer):
  off            : no steering (baseline attribute trajectory)
  prompt_only    : steer the PREFILL, then stop -> tests whether steering decays
                   as un-steered generated tokens accumulate
  fixed_decode   : steer EVERY token at one fixed angle -> tests whether a static
                   angle already holds the attribute flat (if so, thermostat is
                   over-engineering -> drop CLAS)

Attribute = sentiment (the canonical *sustained* steering target). Plane from
rotten_tomatoes pos/neg; behavioral readouts: (a) per-token sentiment logit margin,
(b) decoded-text lexicon sentiment.
"""

import gc
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# shared library lives in <repo>/pytorch_pure
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "pytorch_pure"))

from utils import tokenize_instructions_fn, add_hooks
from phase_portrait import extract_all_layer_activations, compute_steering_plane
from clas_controller import SteerState, make_clas_hook

# ----------------------------------------------------------------------------- config
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
POSITION = "mid"
PLANE_SAMPLES = 128
GEN_TOKENS = 200
SWEEP_DEG = list(range(0, 360, 30))
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

NEUTRAL_PROMPTS = [
    "Tell me about a typical morning routine.",
    "Describe what a city looks like at night.",
    "Write a short story about a journey through a forest.",
    "Share your thoughts on modern technology.",
    "Describe a meal someone might cook for friends.",
    "Tell me about the changing of the seasons.",
]

POS_WORDS = ["great", "good", "love", "wonderful", "amazing", "happy", "beautiful",
             "excellent", "best", "joy", "joyful", "fantastic", "delight", "delightful",
             "brilliant", "perfect", "enjoy", "lovely", "bright", "hopeful", "warm"]
NEG_WORDS = ["bad", "terrible", "hate", "awful", "worst", "sad", "ugly", "horrible",
             "disappointing", "fail", "poor", "boring", "disgusting", "miserable",
             "wrong", "bleak", "bleak", "dark", "cold", "grim", "dreadful"]

# ----------------------------------------------------------------------------- model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
module_dict = dict(model.named_modules())
print(f"loaded {MODEL_ID}")


# ----------------------------------------------------------------------------- sentiment data + plane
def build_contrastive(n):
    """Generation-eliciting contrastive instructions: the diff-in-means captures the
    *generate-positive vs generate-negative* direction (what controls the model's own
    output sentiment), NOT the read-a-review direction."""
    topics = [
        "the weather", "a city", "food", "work", "travel", "music", "mornings",
        "the future", "a walk in the park", "technology", "friends", "the ocean",
        "school", "a holiday", "a new job", "the economy", "sports", "art", "cooking",
        "the news", "growing older", "a small town", "the internet", "family dinners",
        "winter", "summer", "a long drive", "moving house", "a first date", "retirement",
        "coffee", "books", "the night sky", "a garden", "city traffic", "a rainy day",
        "old photographs", "a phone call", "the gym", "a concert", "an airport",
        "a hospital visit", "a wedding", "a funeral", "a birthday", "a reunion",
        "a forest", "a desert", "a mountain", "a river", "a library", "a market",
        "a factory", "a classroom", "a kitchen", "a hotel", "a train ride",
        "the stock market", "social media", "artificial intelligence", "the climate",
        "a startup", "a corporation", "a village", "a megacity", "the deep sea",
    ]
    pos_t = "Write an upbeat, joyful, deeply positive and optimistic paragraph about {}."
    neg_t = "Write a bleak, miserable, deeply negative and pessimistic paragraph about {}."
    pos, neg = [], []
    i = 0
    while len(pos) < n:
        t = topics[i % len(topics)]
        pos.append(pos_t.format(t)); neg.append(neg_t.format(t)); i += 1
    return pos[:n], neg[:n]

pos_txt, neg_txt = build_contrastive(PLANE_SAMPLES)
print(f"contrastive generation instructions: {len(pos_txt)} pos, {len(neg_txt)} neg")
pos_acts = extract_all_layer_activations(model, pos_txt, tokenizer, [POSITION], 8)
gc.collect(); torch.cuda.empty_cache()
neg_acts = extract_all_layer_activations(model, neg_txt, tokenizer, [POSITION], 8)
gc.collect(); torch.cuda.empty_cache()
# compute_steering_plane(A, B) orients b1 toward B's pole; pass (pos, neg) -> b1 ~ negative pole
plane = compute_steering_plane(pos_acts, neg_acts)
b1 = plane["b1"].to(device); b2 = plane["b2"].to(device)
STEER_LAYER = int(plane["selected_key"].split("_")[1])
del pos_acts, neg_acts; gc.collect(); torch.cuda.empty_cache()
steer_module = module_dict[f"model.layers.{STEER_LAYER}"]
state = SteerState()
print(f"sentiment plane: {plane['selected_key']} -> steer layer {STEER_LAYER}, b1.b2={(b1@b2).item():.2e}")


# ----------------------------------------------------------------------------- observables
def make_id_set(words):
    ids = set()
    for w in words:
        for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            enc = tokenizer.encode(v, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
    return torch.tensor(sorted(ids), device=device)

POS_IDS, NEG_IDS = make_id_set(POS_WORDS), make_id_set(NEG_WORDS)

def sent_margin(logits):                       # (B,V)->(B,) log P(pos word) - log P(neg word)
    lp = F.log_softmax(logits.float(), dim=-1)
    return torch.logsumexp(lp[:, POS_IDS], dim=-1) - torch.logsumexp(lp[:, NEG_IDS], dim=-1)

POS_SET = set(w.lower() for w in POS_WORDS)
NEG_SET = set(w.lower() for w in NEG_WORDS)
def lexicon_score(text):                       # net pos-neg sentiment words, normalized by length
    toks = [t.strip(".,!?;:\"'()").lower() for t in text.split()]
    if not toks:
        return 0.0
    return (sum(t in POS_SET for t in toks) - sum(t in NEG_SET for t in toks)) / max(len(toks), 1)


# ----------------------------------------------------------------------------- generation
def gen_record(prompt, theta=None, steer_prefill=False, steer_decode=False, n=GEN_TOKENS):
    """batch=1 greedy decode; record per-token sentiment logit margin. Steering is
    applied during prefill iff steer_prefill, and during each decode step iff
    steer_decode (the hook reads state.enabled)."""
    inp = tokenize_instructions_fn([prompt], tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    th = torch.tensor([float(theta)], device=device) if theta is not None else torch.tensor([0.0], device=device)
    state.theta = th
    hooks = [(steer_module, make_clas_hook(b1, b2, state))]
    ys, gen = [], []
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            state.enabled = steer_prefill
            out = model(input_ids=ids, attention_mask=attn, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            ys.append(sent_margin(logits).item())
            nxt = logits.argmax(-1); gen.append(nxt.item())
            state.enabled = steer_decode
            for _ in range(n - 1):
                attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)
                out = model(input_ids=nxt[:, None], attention_mask=attn,
                            past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                ys.append(sent_margin(logits).item())
                nxt = logits.argmax(-1); gen.append(nxt.item())
                if nxt.item() == tokenizer.eos_token_id:
                    break
    state.enabled = True
    return np.array(ys), tokenizer.decode(gen, skip_special_tokens=True)


def first_margin_batched(prompts, theta):
    inp = tokenize_instructions_fn(prompts, tokenizer)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    state.enabled = True; state.theta = torch.tensor(float(theta), device=device)
    with add_hooks(module_forward_hooks=[(steer_module, make_clas_hook(b1, b2, state))]):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn, position_ids=pos)
    state.enabled = True
    return sent_margin(out.logits[:, -1, :]).cpu().numpy()


# ----------------------------------------------------------------------------- [1] does steering move sentiment?
print("\n[1] sweep: does the sentiment plane respond to theta? (first-token margin)")
sweep = []
for d in SWEEP_DEG:
    m = float(first_margin_batched(NEUTRAL_PROMPTS, math.radians(d)).mean())
    sweep.append(m)
    print(f"  theta={d:3d}  sentiment_margin={m:+.3f}")
sweep = np.array(sweep)
theta_pos_deg = int(SWEEP_DEG[int(sweep.argmax())])   # angle that pushes most positive
authority = float(sweep.max() - sweep.min())
print(f"  -> most-positive angle theta+={theta_pos_deg} deg (margin {sweep.max():+.3f}); "
      f"range {sweep.min():+.2f}..{sweep.max():+.2f}  -> AUTHORITY={authority:.2f}")
if authority < 2.0:
    print("  !! WARNING: steering authority is weak (<2.0). A drift result here is NOT")
    print("     conclusive — fixed-theta would 'drift' simply because the knob is too weak.")
theta_pos = math.radians(theta_pos_deg)

# ----------------------------------------------------------------------------- [2] drift across a long generation
print(f"\n[2] long-gen drift ({GEN_TOKENS} tokens) at theta+={theta_pos_deg} deg, per mode...")
modes = {
    "off":          dict(theta=None,      steer_prefill=False, steer_decode=False),
    "prompt_only":  dict(theta=theta_pos, steer_prefill=True,  steer_decode=False),
    "fixed_decode": dict(theta=theta_pos, steer_prefill=True,  steer_decode=True),
}
results = {m: {"y": [], "lex_early": [], "lex_late": [], "texts": []} for m in modes}
for p in NEUTRAL_PROMPTS:
    for m, kw in modes.items():
        y, text = gen_record(p, **kw)
        results[m]["y"].append(y)
        results[m]["texts"].append(text)
        half = max(len(text.split()) // 2, 1)
        words = text.split()
        results[m]["lex_early"].append(lexicon_score(" ".join(words[:half])))
        results[m]["lex_late"].append(lexicon_score(" ".join(words[half:])))

print("\n  --- sample generations for prompt[0] (eyeball the steering effect) ---")
print(f"  PROMPT: {NEUTRAL_PROMPTS[0]}")
for m in modes:
    print(f"  [{m}] {results[m]['texts'][0][:240]!r}")

def ema(a, alpha=0.1):
    out = np.empty_like(a); acc = a[0]
    for i, v in enumerate(a):
        acc = alpha * v + (1 - alpha) * acc; out[i] = acc
    return out

# align to common length, average EMA across prompts
L = min(min(len(y) for y in results[m]["y"]) for m in modes)
print(f"\n  common length = {L} tokens")
print(f"  {'mode':13s} {'early_margin':>13s} {'late_margin':>12s} {'drift(late-early)':>18s} "
      f"{'lex_early':>10s} {'lex_late':>9s}")
curves = {}
for m in modes:
    arr = np.stack([ema(y[:L]) for y in results[m]["y"]])     # (P, L)
    mean = arr.mean(0)
    curves[m] = mean
    early, late = mean[:L // 4].mean(), mean[-L // 4:].mean()
    le, ll = np.mean(results[m]["lex_early"]), np.mean(results[m]["lex_late"])
    print(f"  {m:13s} {early:>13.3f} {late:>12.3f} {late-early:>18.3f} {le:>10.3f} {ll:>9.3f}")

# ----------------------------------------------------------------------------- plot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(SWEEP_DEG, sweep, "o-", color="purple")
ax[0].axvline(theta_pos_deg, color="green", ls="--", label=f"theta+={theta_pos_deg}°")
ax[0].set_xlabel("steering angle θ (deg)"); ax[0].set_ylabel("first-token sentiment margin")
ax[0].set_title("Sentiment plane responds to θ"); ax[0].legend(); ax[0].grid(alpha=0.3)

colors = {"off": "gray", "prompt_only": "orange", "fixed_decode": "crimson"}
for m in modes:
    ax[1].plot(curves[m], color=colors[m], lw=2, label=m)
ax[1].set_xlabel("generated token position"); ax[1].set_ylabel("sentiment margin (EMA, mean over prompts)")
ax[1].set_title(f"Attribute drift across {GEN_TOKENS}-token generation")
ax[1].legend(); ax[1].grid(alpha=0.3)
fig.suptitle(f"CLAS drift test — sentiment, {MODEL_ID.split('/')[-1]} layer {STEER_LAYER}", fontsize=13)
plt.tight_layout(); plt.savefig("clas_drift_test.png", dpi=130, bbox_inches="tight")
print("\nsaved clas_drift_test.png")

# ----------------------------------------------------------------------------- verdict
fd = curves["fixed_decode"]
fd_drift = fd[-L // 4:].mean() - fd[:L // 4].mean()
po = curves["prompt_only"]
po_drift = po[-L // 4:].mean() - po[:L // 4].mean()
print("\n" + "=" * 64)
print("DRIFT-TEST VERDICT")
print("=" * 64)
print(f"  prompt-only decays over generation?  drift={po_drift:+.3f} "
      f"({'YES, decays toward baseline' if po_drift < -0.3 else 'no clear decay'})")
print(f"  fixed-θ decode holds the attribute?  drift={fd_drift:+.3f} "
      f"({'HOLDS (flat) -> thermostat may be unneeded' if abs(fd_drift) < 0.3 else 'DRIFTS -> thermostat justified'})")
print("  -> If prompt-only decays AND fixed-decode also drifts/overshoots, the per-token")
print("     loop is justified. If fixed-decode holds flat, a static angle suffices (drop CLAS).")
np.savez_compressed("clas_drift_test_results.npz",
                    sweep_deg=np.array(SWEEP_DEG), sweep=sweep, theta_pos_deg=theta_pos_deg,
                    **{f"curve_{m}": curves[m] for m in modes})
