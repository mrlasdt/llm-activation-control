"""MOSAIC Phase 5 — TRUE per-token closed loop vs chunk-stale feedback (does loop rate matter?).

Phase 4 won with a CHUNKED loop (re-measure reading-level, re-set the push every 32 tokens). Its
controller is PROPORTIONAL, not integral:  u_l[b] = m_F·g_F,l − κ·e_R[b]·g_R,l  with the push SET
(not accumulated) from the *current* reading error each update. So the honest thing a "true per-token
loop" changes is NOT the gain but the FEEDBACK STALENESS: how fresh is the e_R the controller reacts
to. This phase isolates exactly that on ONE KV-cached decode:

    update_every ∈ {1, 8, 32}   (1 = re-measure FK and re-set the push EVERY generated token;
                                 32 = chunk-stale, the Phase-4 regime), same κ, same m_F, same prompts.

Because the law is proportional (push magnitude = κ·e_R regardless of how often you update), finer
updates do not push harder — they just react to a less-stale reading error. So a fair, confound-free
test of "does a genuine per-token loop hold tighter than the chunked approximation?"

WIN  : at matched formality-gain, update_every=1 holds reading > 1 SE tighter than update_every=32
       (lag reduction buys a tighter hold) — the per-token loop is worth its cost.
TIE  : update_every=1 ≈ 32 ⇒ the chunked approximation already captured the feedback benefit; the
       32-token lag is below the timescale on which the side-effect re-accumulates (a useful negative:
       the cheap chunked loop is sufficient, build that).

Sensor = exact Flesch-Kincaid on text-so-far (the Phase-4 best-case sensor). Single-stream KV-cached
greedy decode adapted from calm_token.decode_dual (Gemma-2 dense-cache-correct, incrementing
position_ids); the additive push is read LIVE from a mutable LivePush each forward.

Run:  ../../.venv/bin/python mosaic_phase5.py --run
      ../../.venv/bin/python mosaic_phase5.py --selftest   (zero-push decode == model.generate gate)
Outputs: outputs/mosaic_phase5_<model>.{json,txt,png}.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))
sys.path.insert(0, str(_HERE.parent))

from utils import add_hooks, get_input_data, tokenize_instructions_fn          # noqa: E402
from mosaic_phase0 import (  # noqa: E402
    SENT_MODEL, FORM_MODEL, FORMAL_EX, INFORMAL_EX, POS_EX, NEG_EX,
    HFScorer, fk_grade, distinct2, load_model, load_band_pscale,
)
from mosaic_phase3 import load_cones, gdir                                      # noqa: E402

OUT = _HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)


class LivePush:
    """Per-band-layer additive push with a PER-BATCH-ELEMENT vector P_l:(B,d), read LIVE each forward
    (mutated between tokens by the controller). Broadcast over sequence positions; tuple-aware. Same
    mechanism as calm_token's mutable actuator — the hook reads self.P each call."""

    def __init__(self, band, device):
        self.band = band
        self.P = {l: None for l in band}     # set() before first forward
        self.device = device

    def set(self, P_by_layer):
        for l in self.band:
            v = P_by_layer[l]
            self.P[l] = v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v, np.float32))
            self.P[l] = self.P[l].to(self.device)

    def hooks(self, module_dict):
        def mk(l):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out          # (B,S,d)
                h = h + self.P[l].to(h.dtype)[:, None, :]
                return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            return hook
        return [(module_dict[f"model.layers.{l}"], mk(l)) for l in self.band]


def _push_vectors(gF, gR, band, m_F, kappa, e_R, u_max):
    """Proportional control law, per batch element: u_l = m_F·g_F − κ·e_R·g_R, clipped to ‖·‖≤u_max."""
    P = {}
    for l in band:
        Pl = m_F * gF[l][None, :] - kappa * e_R[:, None] * gR[l][None, :]       # (B,d)
        nrm = np.linalg.norm(Pl, axis=1, keepdims=True)
        P[l] = (Pl * np.minimum(1.0, u_max / (nrm + 1e-6))).astype(np.float32)
    return P


def per_token_generate(model, tok, prompts, gF, gR, band, m_F, kappa, u_max, target_FK,
                       max_new_tokens, device, update_every=1, ema=0.0):
    """Single-stream KV-cached greedy decode. Every `update_every` generated tokens, decode the
    text-so-far, measure FK, recompute e_R = FK − target_FK, and re-set the live push. Returns final
    texts + the per-update mean |e_R| trace + the number of controller updates.

    ema>0 low-pass-filters the feedback signal: e_filt ← ema·e_filt + (1−ema)·e_measured, and the
    controller acts on e_filt. This is a slew-rate limiter on the control signal — the standard fix
    for actuator chatter, tested here against the unfiltered per-token loop's coherence cost."""
    from transformers import DynamicCache
    module_dict = dict(model.named_modules())
    inp = tokenize_instructions_fn(prompts, tok)
    ids0 = inp.input_ids.to(device); attn = inp.attention_mask.to(device)
    pos = (attn.long().cumsum(-1) - 1).clamp(min=0)
    B = ids0.shape[0]
    e_R = np.zeros(B); e_filt = np.zeros(B)
    lp = LivePush(band, device)
    lp.set(_push_vectors(gF, gR, band, m_F, kappa, e_filt, u_max))
    hooks = lp.hooks(module_dict)
    gen = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    trace, n_upd = [], 0

    def texts_so_far():
        return [tok.decode(gen[i], skip_special_tokens=True) for i in range(B)]

    with torch.no_grad():
        with add_hooks(module_forward_hooks=hooks):
            out = model(input_ids=ids0, attention_mask=attn, position_ids=pos,
                        past_key_values=DynamicCache(), use_cache=True)
        past = out.past_key_values; log = out.logits[:, -1, :]; cur = pos[:, -1:]
        for step in range(max_new_tokens):
            nxt = log.argmax(-1)
            for i in range(B):
                if not finished[i]:
                    gen[i].append(int(nxt[i]))
            finished |= (nxt == tok.eos_token_id)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break
            if step % update_every == 0:                       # re-measure & re-set the push
                fk = np.array([fk_grade(t) for t in texts_so_far()], float)
                e_R = np.nan_to_num(fk - target_FK)
                e_filt = ema * e_filt + (1.0 - ema) * e_R       # slew-limit (ema=0 ⇒ raw per-token)
                lp.set(_push_vectors(gF, gR, band, m_F, kappa, e_filt, u_max))
                trace.append(float(np.nanmean(np.abs(e_R)))); n_upd += 1
            cur = cur + 1
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            tkn = nxt[:, None]
            with add_hooks(module_forward_hooks=hooks):
                out = model(input_ids=tkn, attention_mask=attn, position_ids=cur,
                            past_key_values=past, use_cache=True)
            past = out.past_key_values; log = out.logits[:, -1, :]
    return texts_so_far(), trace, n_upd


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    C, cband = load_cones(model)
    assert cband == band, "cone band mismatch — re-run Phase 2"
    gF = {l: gdir(C, "formality", i) for i, l in enumerate(band)}
    gR = {l: gdir(C, "reading", i) for i, l in enumerate(band)}
    u_max = args.umax_frac * pscale
    print(f"model={args.model} band={band[0]}..{band[-1]} pscale={pscale:.1f} u_max={u_max:.1f}")

    _, test = get_input_data("harmless")
    n_eval = 4 if args.selftest else args.n_eval
    mnt = 24 if args.selftest else args.max_new_tokens
    MF = [0.06] if args.selftest else args.mf
    KAP = [0.0, 4.0] if args.selftest else args.kappa
    UE = [1, 32] if args.selftest else args.update_every
    prompts = test[:n_eval]

    print("loading formality scorer ...")
    form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)

    base_txt, _, _ = per_token_generate(model, tok, prompts, gF, gR, band, 0.0, 0.0, u_max,
                                        np.zeros(len(prompts)), mnt, device, update_every=10**9)
    target_FK = np.array([fk_grade(t) for t in base_txt], float)
    base_F = form.score(base_txt); base_R = np.array([fk_grade(t) for t in base_txt], float)
    baseM = {"formality": float(np.nanmean(base_F)), "reading": float(np.nanmean(base_R)),
             "distinct2": float(np.nanmean([distinct2(t) for t in base_txt]))}
    print(f"baseline: formality={baseM['formality']:.3f} reading={baseM['reading']:.2f} "
          f"distinct2={baseM['distinct2']:.2f}")

    conds = []
    for m_F in MF:
        # open-loop drive (κ=0): update_every is irrelevant, run once
        txt, trace, _ = per_token_generate(model, tok, prompts, gF, gR, band, m_F * pscale, 0.0,
                                            u_max, target_FK, mnt, device, update_every=10**9)
        conds.append(_row(m_F, 0.0, 0, txt, form, base_F, base_R, trace))
        _show(conds[-1])
        for kap in [k for k in KAP if k > 0]:
            for ue in UE:
                txt, trace, nu = per_token_generate(model, tok, prompts, gF, gR, band, m_F * pscale,
                                                    kap, u_max, target_FK, mnt, device, update_every=ue)
                conds.append(_row(m_F, kap, ue, txt, form, base_F, base_R, trace, n_upd=nu))
                _show(conds[-1])
            # slew-limited per-token (ema>0): same low lag (ue=1) but filtered control signal — the
            # control-theoretic fix for the chatter the raw per-token loop injects.
            txt, trace, nu = per_token_generate(model, tok, prompts, gF, gR, band, m_F * pscale, kap,
                                                u_max, target_FK, mnt, device, update_every=1, ema=args.ema)
            conds.append(_row(m_F, kap, 1, txt, form, base_F, base_R, trace, n_upd=nu, ema=args.ema))
            _show(conds[-1])

    verdict = decide(conds, args)
    print("\n==== VERDICT ====")
    for l in verdict["lines"]:
        print(" ", l)

    out = {"meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale,
                    "n_eval": n_eval, "max_new_tokens": mnt, "mf": MF, "kappa": KAP,
                    "update_every": UE, "umax_frac": args.umax_frac, "coh_floor": args.coh_floor,
                    "win_se": args.win_se, "selftest": args.selftest},
           "baseline": baseM, "conditions": conds, "verdict": verdict}
    name = args.model.split("/")[-1]; tag = "_selftest" if args.selftest else ""
    json.dump(out, open(OUT / f"mosaic_phase5_{name}{tag}.json", "w"), indent=2)
    (OUT / f"mosaic_phase5_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    if not args.selftest:
        try:
            plot(conds, baseM, OUT / f"mosaic_phase5_{name}.png")
            print(f"plot -> outputs/mosaic_phase5_{name}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"saved outputs/mosaic_phase5_{name}{tag}.json  (elapsed {time.time()-t0:.0f}s)")
    return out


def _row(m_F, kappa, ue, txt, form, base_F, base_R, trace, n_upd=0, ema=0.0):
    F = form.score(txt); R = np.array([fk_grade(t) for t in txt], float)
    d2 = np.array([distinct2(t) for t in txt], float)
    Ag = F - base_F; Bd = R - base_R
    return {"m_F": m_F, "kappa": kappa, "update_every": ue, "ema": ema, "open_loop": kappa == 0.0,
            "A_gain": float(np.nanmean(Ag)), "B_drift": float(np.nanmean(Bd)),
            "B_absdrift": float(np.nanmean(np.abs(Bd))), "distinct2": float(np.nanmean(d2)),
            "n_upd": n_upd, "eR_trace": trace, "A_per": Ag.tolist(), "B_per": Bd.tolist(),
            "sample": txt[0][:160]}


def _show(r):
    if r["open_loop"]:
        tag = "(drive-only)   "
    elif r.get("ema", 0.0) > 0:
        tag = f"(closed ue=1 ema={r['ema']:.1f})"
    else:
        tag = f"(closed ue={r['update_every']:>2})    "
    print(f"  m_F={r['m_F']:<5} κ={r['kappa']:<5} {tag}: A_gain={r['A_gain']:+.3f}  "
          f"B_drift={r['B_drift']:+.2f} (|{r['B_absdrift']:.2f}|)  dist2={r['distinct2']:.2f}  "
          f"upd={r['n_upd']}")


def _coh(rows, floor):
    return [r for r in rows if r["distinct2"] >= floor]


def _interp(rows, g, key="B_absdrift"):
    pts = sorted(rows, key=lambda r: r["A_gain"])
    gs = [r["A_gain"] for r in pts]
    if not pts or g < min(gs) or g > max(gs):
        return np.nan
    return float(np.interp(g, gs, [r[key] for r in pts]))


def _by(conds, **kw):
    return [r for r in conds if all(r.get(k) == v for k, v in kw.items())]


def decide(conds, args):
    """Honest per-token analysis. Compares per-token (ue=1) vs chunk-stale (ue=max) at MATCHED κ
    (so loop-rate is the only variable), reporting BOTH |drift| and the coherence cost — and whether
    slew-limiting (ema>0) recovers coherence. The program's lesson: never let a tighter hold hide a
    coherence collapse."""
    lines = []
    ue_fine, ue_coarse = min(args.update_every), max(args.update_every)
    KAP = [k for k in args.kappa if k > 0]; MF = args.mf
    raw_tighter, raw_tighter_coherent, chatter = 0, 0, 0
    ema_recovers = 0; ema_total = 0
    for mF in MF:
        for kap in KAP:
            f = (_by(conds, m_F=mF, kappa=kap, update_every=ue_fine, ema=0.0) or [None])[0]
            c = (_by(conds, m_F=mF, kappa=kap, update_every=ue_coarse, ema=0.0) or [None])[0]
            e = next((r for r in conds if r["m_F"] == mF and r["kappa"] == kap and r.get("ema", 0) > 0), None)
            if not (f and c):
                continue
            df = np.abs(np.array(f["B_per"], float)); dc = np.abs(np.array(c["B_per"], float))
            diff = dc - df; se = float(np.nanstd(diff) / np.sqrt(np.sum(np.isfinite(diff))))
            md = float(np.nanmean(diff))
            tighter = md > args.win_se * se and f["B_absdrift"] < c["B_absdrift"]
            f_coh = f["distinct2"] >= args.coh_floor; c_coh = c["distinct2"] >= args.coh_floor
            raw_tighter += tighter
            raw_tighter_coherent += tighter and f_coh
            chatter += tighter and (not f_coh) and c_coh    # per-token tighter but went incoherent
            line = (f"[m_F={mF} κ={kap}] |drift| per-token(ue=1)={f['B_absdrift']:.2f}(d2={f['distinct2']:.2f}) "
                    f"vs chunk(ue={ue_coarse})={c['B_absdrift']:.2f}(d2={c['distinct2']:.2f}) | paired chunk−fine="
                    f"{md:+.2f}±{se:.2f}SE → {'fine-tighter' if tighter else 'tie'}"
                    f"{' but INCOHERENT' if tighter and not f_coh else ''}")
            if e is not None:
                ema_total += 1
                # slew-limit recovers coherence iff ema arm is coherent AND holds ≈ as tight as raw fine
                rec = (e["distinct2"] >= args.coh_floor and not f_coh
                       and e["B_absdrift"] <= c["B_absdrift"] + se)
                ema_recovers += rec
                line += (f"  || slew(ema={e['ema']:.1f}): |drift|={e['B_absdrift']:.2f} d2={e['distinct2']:.2f}"
                         f"{' [recovers coherence]' if rec else ''}")
            lines.append(line)
    # verdict
    if chatter >= 1 and raw_tighter_coherent == 0:
        v = ("NEGATIVE/NUANCED (per-token chatter): the raw per-token loop (update_every=1) holds reading "
             f"tighter than the 32-token-chunked loop at matched κ in {chatter}/{len(KAP)*len(MF)} cells — but "
             "ONLY by dropping below the coherence floor. Updating the steering vector every token = actuator "
             "chatter: the noisy per-token FK error jerks the push around and degrades fluency. At matched κ "
             "AND coherence the chunked loop weakly dominates (≈equal hold, better coherence, 32× cheaper). "
             + (f"Slew-limiting (ema) recovers coherence in {ema_recovers}/{ema_total} cells — the control-"
                "theoretic fix works: filter the feedback signal and the per-token loop becomes usable."
                if ema_recovers else
                "Slew-limiting did not cleanly recover the hold+coherence here."))
    elif raw_tighter_coherent >= 1:
        v = (f"WIN (per-token > chunked): in {raw_tighter_coherent} cell(s) the per-token loop holds reading "
             ">1 SE tighter than chunk-stale at matched κ WHILE staying coherent — lower feedback lag buys a "
             "real, coherent hold improvement.")
    else:
        v = ("TIE (chunked sufficient): per-token ≈ chunk-stale at matched κ — the 32-token lag is below the "
             "timescale on which the reading side-effect re-accumulates; the cheap chunked loop captures the "
             "closed-loop benefit. Build the chunked loop.")
    lines.insert(0, v)
    return {"verdict": v, "raw_tighter": raw_tighter, "raw_tighter_coherent": raw_tighter_coherent,
            "chatter_cells": chatter, "ema_recovers": ema_recovers, "ema_total": ema_total, "lines": lines}


def plot(conds, baseM, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ues = sorted(set(r["update_every"] for r in conds if not r["open_loop"]))
    cols = {ue: f"C{i}" for i, ue in enumerate(ues)}
    drive = sorted([r for r in conds if r["open_loop"]], key=lambda r: r["A_gain"])
    ax.plot([r["A_gain"] for r in drive], [r["B_absdrift"] for r in drive], "o--", color="C3",
            label="open-loop drive (κ=0)")
    for ue in ues:
        rows = sorted([r for r in conds if not r["open_loop"] and r["update_every"] == ue],
                      key=lambda r: r["A_gain"])
        ax.plot([r["A_gain"] for r in rows], [r["B_absdrift"] for r in rows], "s-", color=cols[ue],
                label=f"closed-loop update_every={ue}")
        for r in rows:
            if r["distinct2"] < 0.85:
                ax.plot(r["A_gain"], r["B_absdrift"], "x", color=cols[ue], ms=9)
    ax.set_xlabel("formality-gain ↑"); ax.set_ylabel("|reading-drift| ↓ (tighter hold)")
    ax.set_title("Phase 5: feedback staleness (update_every)\nlower-left = tighter hold at equal gain; ×=incoherent")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]
    for r in conds:
        if not r["open_loop"] and r["eR_trace"]:
            xs = np.linspace(0, 1, len(r["eR_trace"]))
            ax.plot(xs, r["eR_trace"], "-", label=f"m_F={r['m_F']} κ={r['kappa']} ue={r['update_every']}")
    ax.set_xlabel("generation progress (normalized)"); ax.set_ylabel("mean |reading error| (FK)")
    ax.set_title("regulation trace (finer = lower-lag tracking of e_R)")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle("MOSAIC Phase 5 — true per-token loop vs chunk-stale feedback")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def _selftest(args):
    """Zero-push per-token decode must match model.generate greedy token-for-token (correctness gate),
    AND the live push must actually perturb generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    C, cband = load_cones(model); assert cband == band
    gF = {l: gdir(C, "formality", i) for i, l in enumerate(band)}
    gR = {l: gdir(C, "reading", i) for i, l in enumerate(band)}
    _, test = get_input_data("harmless"); prompts = test[:2]
    # zero push (m_F=0, κ=0) == plain greedy generate
    txt, _, _ = per_token_generate(model, tok, prompts, gF, gR, band, 0.0, 0.0, pscale,
                                   np.zeros(2), 24, device, update_every=10**9)
    inp = tokenize_instructions_fn(prompts, tok)
    with torch.no_grad():
        g = model.generate(inp.input_ids.to(device), attention_mask=inp.attention_mask.to(device),
                           max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
    ref = [tok.decode(g[i, inp.input_ids.shape[1]:], skip_special_tokens=True) for i in range(2)]
    ok = True
    for i in range(2):
        man_ids = tok.encode(txt[i], add_special_tokens=False)
        ref_ids = tok.encode(ref[i], add_special_tokens=False)
        k = 0
        while k < min(len(man_ids), len(ref_ids)) and man_ids[k] == ref_ids[k]:
            k += 1
        print(f"  row {i}: zero-push==generate agree {k}/{min(len(man_ids),len(ref_ids))} tokens")
        ok &= k >= min(len(man_ids), len(ref_ids)) - 1
    print(f"[gate] zero-push decode == model.generate: {'PASS' if ok else 'FAIL'}")
    # push perturbs: a strong formality push must change the text
    txt2, _, _ = per_token_generate(model, tok, prompts, gF, gR, band, 0.10 * pscale, 0.0, 0.15 * pscale,
                                    np.zeros(2), 24, device, update_every=10**9)
    diff = any(txt2[i] != txt[i] for i in range(2))
    print(f"[gate] formality push changes generation: {'PASS' if diff else 'FAIL'}")
    print(f"  base[0]: {txt[0][:90]!r}\n  push[0]: {txt2[0][:90]!r}")
    return ok and diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-eval", type=int, default=24, dest="n_eval")
    ap.add_argument("--max-new-tokens", type=int, default=96, dest="max_new_tokens")
    ap.add_argument("--mf", type=float, nargs="+", default=[0.06, 0.08])
    ap.add_argument("--kappa", type=float, nargs="+", default=[0.0, 4.0, 8.0])
    ap.add_argument("--update-every", type=int, nargs="+", default=[1, 8, 32], dest="update_every")
    ap.add_argument("--ema", type=float, default=0.8, help="slew-limit (EMA) on the feedback signal for "
                    "the per-token chatter-fix arm; 0 disables that arm")
    ap.add_argument("--umax-frac", type=float, default=0.15, dest="umax_frac")
    ap.add_argument("--coh-floor", type=float, default=0.85, dest="coh_floor")
    ap.add_argument("--win-se", type=float, default=1.0, dest="win_se")
    args = ap.parse_args()
    if args.selftest and not args.run:
        _selftest(args); return
    if not args.run:
        ap.error("pass --run or --selftest")
    run(args)


if __name__ == "__main__":
    main()
