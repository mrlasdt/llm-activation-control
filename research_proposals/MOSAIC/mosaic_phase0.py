"""MOSAIC Phase 0 — the binding-precondition KILL gate (no controller, no training).

Hard gate + fork from MOSAIC_PROPOSAL.md §7. On Gemma-2-2b-it, on a *deliberately
interfering* attribute triple (A=formality, B=reading-level, C=sentiment — formality &
reading-level both load on lexical complexity, sentiment as the third), cheaply measure
the TWO mechanisms that could make a hold-constraint bind:

  (a) CROSS-COUPLING  (-> MOSAIC): does a naive scalar push on attribute A drag B,C off
      their baselines by > eps AND > 1 SE of natural variation?  Quantify the off-diagonal
      cross-Gram + whether a feasible *allocated* (hold-B,C) solution exists.
  (b) REVERSION       (-> SARTRE arm): if you steer an attribute up then RELEASE, does it
      drift back over the generation (a sustained disturbance to reject)?

GO (MOSAIC) iff scalar-A push moves B and/or C by > eps AND > 1 SE, AND a feasible
allocated direction exists.  NO-GO on both -> report the coupling/reversion DIVIDING LINE
as a publishable scoping law (not a dead null).

Directions are quick k=1 difference-in-means (casa_cone.residual_means + dim_directions)
built from style-directive contrastive prompts (CAA-style). The "naive scalar push" is a
bounded additive per-band-layer push  u_l = m * d_{A,l}  (m = frac * pscale).  Readouts:
Flesch-Kincaid grade (dependency-free, reading-level) + two small HF classifiers
(sentiment, formality), auto-oriented by canonical exemplars (robust to label naming).

Run:
  ../../.venv/bin/python mosaic_phase0.py --selftest        # tiny smoke (n=4, 24 tok)
  ../../.venv/bin/python mosaic_phase0.py --run             # full Phase-0 gate
Outputs: outputs/mosaic_phase0_<model>.json (+ .txt summary).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))

from utils import add_hooks, get_input_data, generate_completions, tokenize_instructions_fn  # noqa: E402
from casa_cone import residual_means, dim_directions                                          # noqa: E402

OUT = _HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)
FRONTIER = _HERE.parents[1] / "CASA" / "outputs"

SENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
FORM_MODEL = "s-nlp/roberta-base-formality-ranker"

# ----------------------------------------------------------------------------- attributes
# Style directives appended to a neutral instruction; (high pole, low pole). The push
# d_attr = mean(high)-mean(low) moves toward the HIGH pole; scorers are oriented so the
# HIGH pole scores higher (formal / harder-to-read / positive).
ATTRS = {
    "formality": (
        "Write your response in an extremely formal, professional, and sophisticated register.",
        "Write your response in a very casual, informal, chatty tone with slang and contractions.",
    ),
    "reading": (   # high = harder to read (higher FK grade)
        "Use advanced, complex vocabulary and long, elaborate, multi-clause sentences.",
        "Use only very simple words and short sentences that a young child could understand.",
    ),
    "sentiment": ( # high = positive
        "Write in an overwhelmingly positive, cheerful, enthusiastic, upbeat way.",
        "Write in a very negative, critical, gloomy, pessimistic way.",
    ),
}
ATTR_KEYS = ["formality", "reading", "sentiment"]

# canonical exemplars to AUTO-ORIENT the classifiers (robust to label naming)
FORMAL_EX = [
    "I would be most grateful if you could provide further information regarding this matter.",
    "Please find enclosed the requested documentation for your consideration.",
    "The committee shall convene to deliberate upon the proposed amendments.",
]
INFORMAL_EX = ["hey wanna grab food later lol", "omg this is so cool, can't even", "nah dude that's whack tbh"]
POS_EX = ["I absolutely love this, it is wonderful and makes me so happy!",
          "What a fantastic, delightful day — everything is going great!"]
NEG_EX = ["This is terrible, I hate it and it makes me miserable.",
          "Everything is awful and hopeless; I feel completely wretched."]


# ----------------------------------------------------------------------------- model
def load_model(name, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", torch_dtype=torch.bfloat16).eval()
    return model, tok


def load_band_pscale(model):
    name = model.config._name_or_path.split("/")[-1]
    meta = json.load(open(FRONTIER / f"casa_frontier_{name}.json"))["meta"]
    lo, hi = meta["band"]
    return list(range(lo, hi + 1)), float(meta["pscale"])


# ----------------------------------------------------------------------------- scorers
def _count_syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    syl = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and syl > 1 and not word.endswith(("le", "ye")):
        syl -= 1
    return max(1, syl)


def fk_grade(text):
    """Flesch-Kincaid grade level (deterministic). Higher = harder to read."""
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    if len(words) < 3 or not sents:
        return float("nan")
    nsyl = sum(_count_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sents)) + 11.8 * (nsyl / len(words)) - 15.59


def distinct2(text):
    toks = text.split()
    if len(toks) < 2:
        return float("nan")
    bg = list(zip(toks, toks[1:]))
    return len(set(bg)) / len(bg)


class HFScorer:
    """A 2-class HF sequence classifier; auto-orients to P(high-pole) via exemplars."""

    def __init__(self, name, hi_ex, lo_ex, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModelForSequenceClassification.from_pretrained(
            name, torch_dtype=torch.float32).to(device).eval()
        # orient: choose the class index that scores higher on hi_ex than lo_ex
        p_hi = self._raw(hi_ex)  # (n,C)
        p_lo = self._raw(lo_ex)
        self.idx = int(np.argmax(p_hi.mean(0) - p_lo.mean(0)))
        self.name = name

    @torch.no_grad()
    def _raw(self, texts):
        out = []
        for i in range(0, len(texts), 32):
            enc = self.tok(texts[i:i + 32], return_tensors="pt", padding=True,
                           truncation=True, max_length=256).to(self.device)
            out.append(torch.softmax(self.m(**enc).logits, -1).float().cpu().numpy())
        return np.concatenate(out, 0)

    def score(self, texts):
        return self._raw(texts)[:, self.idx]  # P(high pole) in [0,1]


def score_all(texts, sent, form):
    """Return dict attr -> np.array of per-text scores (nan-safe)."""
    fk = np.array([fk_grade(t) for t in texts], float)
    s = {"reading": fk}
    s["sentiment"] = sent.score(texts) if sent else np.full(len(texts), np.nan)
    s["formality"] = form.score(texts) if form else np.full(len(texts), np.nan)
    s["distinct2"] = np.array([distinct2(t) for t in texts], float)
    return s


# ----------------------------------------------------------------------------- directions
def wrap(prompts, directive):
    return [f"{p}\n\n{directive}" for p in prompts]


def build_directions(model, tok, base_prompts, device, log=print):
    """Per-layer unit DIM direction + raw mean-diff for each attribute (residual-stream)."""
    dirs, gaps = {}, {}
    for a in ATTR_KEYS:
        hi_d, lo_d = ATTRS[a]
        t0 = time.time()
        hmean = residual_means(model, tok, wrap(base_prompts, hi_d), device)  # (L,d)
        lmean = residual_means(model, tok, wrap(base_prompts, lo_d), device)
        d_unit = dim_directions(hmean, lmean)                                 # (L,d) unit
        delta = hmean - lmean                                                  # (L,d) raw
        dirs[a] = torch.from_numpy(d_unit).float()
        gaps[a] = np.linalg.norm(delta, axis=1)                               # (L,) per-layer gap
        log(f"  dir[{a}] built ({time.time()-t0:.1f}s)  mean band gap "
            f"||mu_hi-mu_lo|| = {gaps[a][7:25].mean():.2f}")
    return dirs, gaps


def coupling_geometry(dirs, band):
    """Per-band-mean cosine cross-Gram + hold-feasibility residual for each driven attr."""
    cos = {}
    for i, ai in enumerate(ATTR_KEYS):
        for aj in ATTR_KEYS[i + 1:]:
            c = (dirs[ai][band] * dirs[aj][band]).sum(-1)  # (nb,)
            cos[f"{ai}|{aj}"] = float(c.mean())
    # feasibility: can we move A without (1st-order) moving the other two?
    feas = {}
    for a in ATTR_KEYS:
        others = [o for o in ATTR_KEYS if o != a]
        res = []
        for l in band:
            da = dirs[a][l]
            B = torch.stack([dirs[o][l] for o in others], 0)  # (2,d)
            Q, _ = torch.linalg.qr(B.t())                     # orthonormal basis of span(others)
            proj = (da @ Q) @ Q.t()
            res.append(float((da - proj).norm()))             # ||component of d_A orthogonal to others||
        feas[a] = float(np.mean(res))
    return cos, feas


# ----------------------------------------------------------------------------- push hooks
def push_hooks(model, band, vec_per_layer, device):
    """Additive per-layer push: h += vec_per_layer[l] at each band layer l (tuple-aware)."""
    md = dict(model.named_modules())
    hooks = []

    def mk(v):
        vv = v.to(device, torch.bfloat16)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + vv
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    for l in band:
        hooks.append((md[f"model.layers.{l}"], mk(vec_per_layer[l])))
    return hooks


# ----------------------------------------------------------------------------- experiments
def gen_score(model, tok, prompts, hooks, sent, form, max_new_tokens, bs):
    comps = generate_completions(model, prompts, tok, fwd_hooks=hooks,
                                 batch_size=bs, max_new_tokens=max_new_tokens, temperature=0.0)
    resp = [c["response"] for c in comps]
    return resp, score_all(resp, sent, form)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    print(f"model={args.model}  band={band[0]}..{band[-1]} ({len(band)}L)  pscale={pscale:.1f}")

    # data: neutral instructions, disjoint dir-building vs eval
    train, test = get_input_data("harmless")
    n_dir = 8 if args.selftest else args.n_dir
    n_eval = 4 if args.selftest else args.n_eval
    base = train[:n_dir]
    eval_prompts = test[:n_eval]
    fracs = [0.1] if args.selftest else args.fracs
    mnt = 24 if args.selftest else args.max_new_tokens
    bs = args.bs

    # scorers (auto-oriented)
    print("loading scorers ...")
    sent = form = None
    try:
        sent = HFScorer(SENT_MODEL, POS_EX, NEG_EX, device)
        print(f"  sentiment: {SENT_MODEL} -> P(positive)=class[{sent.idx}]")
    except Exception as e:
        print(f"  !! sentiment scorer FAILED: {e}")
    try:
        form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)
        print(f"  formality: {FORM_MODEL} -> P(formal)=class[{form.idx}]")
    except Exception as e:
        print(f"  !! formality scorer FAILED: {e}")

    # directions + geometry
    print("building directions ...")
    dirs, gaps = build_directions(model, tok, base, device)
    band_t = band
    cos, feas = coupling_geometry(dirs, band_t)
    print("  cross-Gram (band-mean cosine):", {k: round(v, 3) for k, v in cos.items()})
    print("  hold-feasibility residual ||d_A ⟂ span(others)||:", {k: round(v, 3) for k, v in feas.items()})

    # baseline (no push)
    print("baseline generation ...")
    base_resp, base_sc = gen_score(model, tok, eval_prompts, [], sent, form, mnt, bs)
    base_mean = {a: float(np.nanmean(base_sc[a])) for a in ["formality", "reading", "sentiment", "distinct2"]}
    base_se = {a: float(np.nanstd(base_sc[a]) / np.sqrt(np.sum(~np.isnan(base_sc[a])))) for a in
               ["formality", "reading", "sentiment"]}
    print("  baseline:", {k: round(v, 3) for k, v in base_mean.items()}, "SE:", {k: round(v, 3) for k, v in base_se.items()})

    # (a) coupling: drive each attribute, score all three (the 3x3 induced-drift matrix)
    conditions = []
    for driven in ATTR_KEYS:
        d_unit = dirs[driven]
        for frac in fracs:
            m = frac * pscale
            vec = {l: m * d_unit[l] for l in band}
            hooks = push_hooks(model, band, vec, device)
            t0 = time.time()
            resp, sc = gen_score(model, tok, eval_prompts, hooks, sent, form, mnt, bs)
            row = {
                "driven": driven, "frac": frac, "push_norm_per_layer": round(m, 2),
                "scores": {a: float(np.nanmean(sc[a])) for a in ["formality", "reading", "sentiment", "distinct2"]},
                "delta": {a: float(np.nanmean(sc[a]) - base_mean[a]) for a in ["formality", "reading", "sentiment"]},
                "sample": resp[0][:200],
            }
            conditions.append(row)
            d = row["delta"]
            print(f"  drive {driven:9s} frac={frac:<4} | "
                  f"ΔF={d['formality']:+.3f} ΔR={d['reading']:+.2f} ΔS={d['sentiment']:+.3f} | "
                  f"dist2={row['scores']['distinct2']:.2f} ({time.time()-t0:.0f}s)")

    # (b) reversion (SARTRE fork): steer formality for K tokens then RELEASE; does it revert?
    reversion = None
    if not args.selftest:
        print("reversion probe (formality) ...")
        reversion = reversion_probe(model, tok, eval_prompts[: min(16, len(eval_prompts))],
                                    dirs["formality"], band, pscale, args.revert_frac,
                                    args.revert_k, mnt, form, base_mean["formality"], device)
        if reversion:
            print(f"  reversion: held α≈{reversion['alpha']:.2f}  "
                  f"(steered_seg F={reversion['steered_seg_F']:.3f}, released_seg F={reversion['released_seg_F']:.3f}, "
                  f"base F={base_mean['formality']:.3f})")

    verdict = decide(conditions, cos, feas, base_se, reversion, args)
    print("\n==== VERDICT ====")
    for line in verdict["lines"]:
        print(" ", line)

    result = {
        "meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale,
                 "n_dir": n_dir, "n_eval": n_eval, "fracs": fracs, "max_new_tokens": mnt,
                 "sent_model": SENT_MODEL if sent else None, "form_model": FORM_MODEL if form else None,
                 "selftest": args.selftest},
        "baseline": {"mean": base_mean, "se": base_se},
        "cross_gram_cosine": cos, "hold_feasibility_residual": feas,
        "band_gap": {a: float(gaps[a][band[0]:band[-1] + 1].mean()) for a in ATTR_KEYS},
        "conditions": conditions, "reversion": reversion, "verdict": verdict,
    }
    name = args.model.split("/")[-1]
    tag = "_selftest" if args.selftest else ""
    jf = OUT / f"mosaic_phase0_{name}{tag}.json"
    json.dump(result, open(jf, "w"), indent=2)
    (OUT / f"mosaic_phase0_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    print(f"\nsaved {jf}")
    return result


def reversion_probe(model, tok, prompts, d_unit, band, pscale, frac, K, total, form, base_F, device):
    """Generate K tokens with a formality push, then continue UNSTEERED; compare the
    formality of the steered prefix vs the released continuation (and vs baseline)."""
    if form is None:
        return None
    m = frac * pscale
    vec = {l: m * d_unit[l] for l in band}
    hooks = push_hooks(model, band, vec, device)
    inp = tokenize_instructions_fn(prompts, tok)
    ids = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    plen = ids.shape[1]
    with add_hooks(module_forward_hooks=hooks):
        with torch.no_grad():
            g1 = model.generate(ids, attention_mask=attn, max_new_tokens=K, do_sample=False,
                                pad_token_id=tok.pad_token_id)
    # continue WITHOUT hooks
    a2 = torch.ones_like(g1); a2[:, :plen] = attn
    with torch.no_grad():
        g2 = model.generate(g1, attention_mask=a2, max_new_tokens=total, do_sample=False,
                            pad_token_id=tok.pad_token_id)
    steered_seg = tok.batch_decode(g1[:, plen:], skip_special_tokens=True)
    released_seg = tok.batch_decode(g2[:, g1.shape[1]:], skip_special_tokens=True)
    fs = float(np.mean(form.score(steered_seg)))
    fr = float(np.mean(form.score(released_seg)))
    denom = (fs - base_F)
    alpha = float((fs - fr) / denom) if abs(denom) > 1e-6 else float("nan")  # 1=full revert, 0=sticky
    return {"steered_seg_F": fs, "released_seg_F": fr, "base_F": base_F, "alpha": alpha,
            "revert_frac": frac, "revert_k": K}


def decide(conditions, cos, feas, base_se, reversion, args):
    """Apply the pre-registered GO/NO-GO gates and emit the dividing-line numbers."""
    lines = []
    # strongest nonzero push per driven attr
    by_driven = {}
    for c in conditions:
        if c["frac"] == 0:
            continue
        by_driven.setdefault(c["driven"], []).append(c)
    # MOSAIC coupling gate: driving A=formality, do B(reading)/C(sentiment) move > eps AND > 1 SE?
    eps = args.eps  # normalized tolerance on the held attribute
    go_couple = False
    detail = {}
    for driven in ATTR_KEYS:
        rows = sorted(by_driven.get(driven, []), key=lambda r: r["frac"])
        if not rows:
            continue
        # COHERENCE FLOOR: evaluate coupling at the strongest push that is still coherent
        # (distinct-2 >= coh_floor). Never let gibberish (the frac=0.2 collapse) inflate the
        # induced-drift numbers — the program's substring-ASR/genNLL lesson.
        coherent = [r for r in rows if r["scores"]["distinct2"] >= args.coh_floor]
        strong = coherent[-1] if coherent else rows[0]
        incoherent = not coherent
        # authority: does driving A move A itself monotonically/materially?
        a_moves = abs(strong["delta"][driven])
        # induced drift on the other two
        induced = {o: abs(strong["delta"][o]) for o in ATTR_KEYS if o != driven}
        induced_se = {o: abs(strong["delta"][o]) / (base_se[o] + 1e-9) for o in ATTR_KEYS if o != driven}
        detail[driven] = {"frac": strong["frac"], "distinct2": strong["scores"]["distinct2"],
                          "incoherent": incoherent, "self_move": a_moves,
                          "induced": induced, "induced_in_SE": induced_se}
        binds = any((induced[o] > eps_for(o, eps)) and (induced_se[o] > 1.0) for o in induced)
        if driven == "formality" and binds and a_moves > eps_for(driven, eps):
            go_couple = True
        flag = "  [!] no coherent push found" if incoherent else ""
        lines.append(f"drive {driven} @frac={strong['frac']} (dist2={strong['scores']['distinct2']:.2f}{flag}): "
                     f"self Δ={a_moves:.3f}; induced "
                     + ", ".join(f"{o} Δ={induced[o]:.3f} ({induced_se[o]:.1f}×SE)" for o in induced))
    # feasibility: a hold-feasible push for formality exists if d_A has a non-trivial component ⟂ others
    feas_ok = feas.get("formality", 0.0) > args.feas_min
    lines.append(f"cross-Gram cosines (band-mean): " + ", ".join(f"{k}={v:+.3f}" for k, v in cos.items()))
    lines.append(f"hold-feasibility residual (formality ⟂ others) = {feas.get('formality', float('nan')):.3f} "
                 f"(> {args.feas_min} ⇒ a feasible allocated push exists: {feas_ok})")

    mosaic_go = go_couple and feas_ok
    revert_go = bool(reversion and not np.isnan(reversion["alpha"]) and args.alpha_lo <= reversion["alpha"] <= args.alpha_hi)
    if reversion:
        lines.append(f"reversion α={reversion['alpha']:.2f} (SARTRE-arm GO window "
                     f"[{args.alpha_lo},{args.alpha_hi}]: {revert_go})")

    if mosaic_go:
        verdict = "GO (MOSAIC headline): scalar-A push binds the hold-constraint AND an allocated solution is feasible."
    elif revert_go:
        verdict = "GO (SARTRE arm): weak/over-strong instantaneous coupling, but the attribute reverts ⇒ sustained disturbance-rejection binds on the token axis."
    else:
        verdict = ("NO-GO on both ⇒ report the DIVIDING-LINE scoping law: 'allocation pays iff measured "
                   "cross-coupling > ε'. Coupling/reversion numbers above ARE the result.")
    lines.insert(0, verdict)
    return {"verdict": verdict, "mosaic_go": bool(mosaic_go), "sartre_go": revert_go,
            "go_couple": bool(go_couple), "feas_ok": bool(feas_ok), "detail": detail, "lines": lines}


def coupling_matrix(conditions, base_se, coh_floor):
    """Print the 3x3 induced-drift matrix at each driven attr's strongest COHERENT push
    (rows = driven attribute, cols = measured drift on every attribute)."""
    by = {}
    for c in conditions:
        if c["frac"] > 0:
            by.setdefault(c["driven"], []).append(c)
    lines = ["", "Induced-drift matrix (rows=driven, cols=measured Δ; coherent operating point):"]
    label = "driven/meas"
    hdr = f"  {label:<12}" + "".join(f"{a:>22}" for a in ATTR_KEYS) + f"{'frac':>7}{'dist2':>7}"
    lines.append(hdr)
    for driven in ATTR_KEYS:
        rows = sorted(by.get(driven, []), key=lambda r: r["frac"])
        coh = [r for r in rows if r["scores"]["distinct2"] >= coh_floor]
        r = coh[-1] if coh else rows[0]
        cells = []
        for meas in ATTR_KEYS:
            d = r["delta"][meas]
            se = d / (base_se[meas] + 1e-9)
            tag = "(self)" if meas == driven else ""
            cells.append(f"{d:+8.3f} {se:+5.1f}σ{tag:>6}")
        lines.append(f"  {driven:<12}" + "".join(f"{c:>22}" for c in cells)
                     + f"{r['frac']:>7}{r['scores']['distinct2']:>7.2f}")
    return lines


def eps_for(attr, eps):
    """Per-attribute absolute tolerance: reading-level (FK grade) is on a ~0-18 scale, the
    classifier probs on 0-1. Scale eps accordingly so the gate is comparable across readouts."""
    return eps * (5.0 if attr == "reading" else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--reanalyze", action="store_true", help="recompute verdict from saved JSON (no GPU)")
    ap.add_argument("--coh-floor", type=float, default=0.90, dest="coh_floor",
                    help="distinct-2 coherence floor for selecting the operating point")
    ap.add_argument("--n-dir", type=int, default=64, dest="n_dir")
    ap.add_argument("--n-eval", type=int, default=40, dest="n_eval")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.05, help="normalized hold tolerance")
    ap.add_argument("--feas-min", type=float, default=0.30, dest="feas_min")
    ap.add_argument("--revert-frac", type=float, default=0.1, dest="revert_frac")
    ap.add_argument("--revert-k", type=int, default=32, dest="revert_k")
    ap.add_argument("--alpha-lo", type=float, default=0.3, dest="alpha_lo")
    ap.add_argument("--alpha-hi", type=float, default=0.95, dest="alpha_hi")
    args = ap.parse_args()
    if not (args.run or args.selftest or args.reanalyze):
        ap.error("pass --run (full Phase-0 gate), --selftest (tiny smoke), or --reanalyze")
    if args.reanalyze:
        reanalyze(args)
    else:
        run(args)


def reanalyze(args):
    """Recompute the verdict from the saved JSON at the coherent operating point (no model)."""
    name = args.model.split("/")[-1]
    jf = OUT / f"mosaic_phase0_{name}.json"
    R = json.load(open(jf))
    cos, feas = R["cross_gram_cosine"], R["hold_feasibility_residual"]
    base_se, conditions, reversion = R["baseline"]["se"], R["conditions"], R["reversion"]
    verdict = decide(conditions, cos, feas, base_se, reversion, args)
    mat = coupling_matrix(conditions, base_se, args.coh_floor)
    print(f"\n==== RE-ANALYSIS (coherence floor distinct2≥{args.coh_floor}) ====")
    for l in verdict["lines"]:
        print(" ", l)
    for l in mat:
        print(l)
    R["verdict"] = verdict
    R["coupling_matrix_text"] = mat
    R["meta"]["coh_floor"] = args.coh_floor
    json.dump(R, open(jf, "w"), indent=2)
    (OUT / f"mosaic_phase0_{name}.txt").write_text("\n".join(verdict["lines"] + mat) + "\n")
    print(f"\nupdated {jf}")


if __name__ == "__main__":
    main()
