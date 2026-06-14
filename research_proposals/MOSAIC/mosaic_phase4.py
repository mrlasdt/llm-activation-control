"""MOSAIC Phase 4 — closed-loop token-level HOLD (output feedback) vs open-loop steering.

Phase 3 found the open-loop linear null-space hold could NOT separate formality from reading-level
(shared frontier) because the coupling is a *semantic entanglement* that re-emerges DOWNSTREAM during
autoregressive generation. Phase 4 tests the move that mechanism points to: a CLOSED-LOOP controller
that drives formality up while REGULATING the *measured* reading-level back to its unsteered baseline
each chunk — output feedback, which can in principle fight generation-time re-entanglement that an
open-loop per-layer projection cannot.

Controller (per band layer l, per prompt b), chunked (STU-PID style):
    u_l[b] = m_F · g_F,l   −   κ · e_R[b] · g_R,l ,   ‖u_l[b]‖ ≤ u_max
    e_R[b] = FK(text-so-far[b]) − target_FK[b]      (Flesch-Kincaid = exact, free, zero-noise sensor)
κ=0 is open-loop drive-only (= Phase-3 scalar). The sensor is the EXACT FK scorer (best case) — a null
here is the strongest possible negative (semantic entanglement is irreducible even with a perfect
output sensor + feedback). The Phase-1 probe (r=0.94) is the path for attributes lacking a cheap scorer.

WIN  : closed-loop (κ>0) holds reading-level > 1 SE tighter than open-loop drive (κ=0) at MATCHED
       formality-gain (coherent) — output feedback beats the open-loop projection Phase 3 tied.
NULL : closed-loop ties / can't hold without killing formality or coherence ⇒ the entanglement is
       irreducible; the side-effect is intrinsic to "formal", not a controllable disturbance.

Run:  ../../.venv/bin/python mosaic_phase4.py --run       (~25-40 min)
      ../../.venv/bin/python mosaic_phase4.py --selftest   (chunk-hook math + tiny e2e)
Outputs: outputs/mosaic_phase4_<model>.{json,txt,png}.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

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


def _relpad(ids, pad):
    """Re-LEFT-pad a batch after a generate chunk (model.generate right-pads sequences that hit EOS;
    feeding that back with right-padding corrupts positions for the next chunk). Strips all pad tokens
    and re-left-pads — valid here since the prompt is left-padded and greedy gen does not emit pad."""
    rows = [r[r != pad] for r in ids]
    m = max(int(t.numel()) for t in rows)
    out = torch.full((len(rows), m), pad, device=ids.device, dtype=ids.dtype)
    for i, t in enumerate(rows):
        out[i, m - t.numel():] = t
    return out


def chunk_hooks(model, band, push_by_layer):
    """Per-band-layer additive push with a PER-BATCH-ELEMENT vector P_l:(B,d) (so each prompt gets its
    own drive+feedback push). Broadcast over sequence positions; tuple-aware."""
    md = dict(model.named_modules())

    def mk(P):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out          # (B,S,d)
            h = h + P.to(h.device, h.dtype)[:, None, :]
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook

    return [(md[f"model.layers.{l}"], mk(push_by_layer[l])) for l in band]


def closed_loop_generate(model, tok, prompts, gF, gR, band, m_F, kappa, u_max,
                         target_FK, n_chunks, chunk_tok, device):
    """Chunked closed loop: drive formality (m_F·g_F) while regulating measured FK toward target_FK
    (−κ·e_R·g_R). Returns final generated texts (list) and the per-chunk mean |e_R| trace."""
    pad = tok.pad_token_id
    inp = tokenize_instructions_fn(prompts, tok)
    ids = inp.input_ids.to(device)
    n_prompt = inp.attention_mask.sum(1).tolist()          # real prompt tokens per row
    B = len(prompts)
    e_R = np.zeros(B)
    trace = []

    def gen_texts():
        out = []
        for b in range(B):
            real = ids[b][ids[b] != pad]                   # [prompt..gen..] (left-pad stripped)
            out.append(tok.decode(real[n_prompt[b]:], skip_special_tokens=True))
        return out

    for c in range(n_chunks):
        push = {}
        for l in band:
            P = m_F * gF[l][None, :] - kappa * e_R[:, None] * gR[l][None, :]   # (B,d)
            nrm = np.linalg.norm(P, axis=1, keepdims=True)
            P = P * np.minimum(1.0, u_max / (nrm + 1e-6))
            push[l] = torch.from_numpy(P.astype(np.float32))
        attn = (ids != pad).long()
        with add_hooks(module_forward_hooks=chunk_hooks(model, band, push)):
            with torch.no_grad():
                ids = model.generate(ids, attention_mask=attn, max_new_tokens=chunk_tok,
                                     do_sample=False, pad_token_id=pad)
        ids = _relpad(ids, pad)                            # re-left-pad (generate right-pads finished seqs)
        fk = np.array([fk_grade(t) for t in gen_texts()], float)
        e_R = np.nan_to_num(fk - target_FK)
        trace.append(float(np.nanmean(np.abs(e_R))))
    return gen_texts(), trace


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    C, cband = load_cones(model)
    assert cband == band
    gF = {l: gdir(C, "formality", i) for i, l in enumerate(band)}   # drive direction (unit, per layer)
    gR = {l: gdir(C, "reading", i) for i, l in enumerate(band)}     # hold direction (unit, per layer)
    print(f"model={args.model} band={band[0]}..{band[-1]} pscale={pscale:.1f}")

    _, test = get_input_data("harmless")
    n_eval = 4 if args.selftest else args.n_eval
    n_chunks = 2 if args.selftest else args.n_chunks
    chunk_tok = 12 if args.selftest else args.chunk_tok
    MF = [0.06] if args.selftest else args.mf
    KAP = [0.0, 6.0] if args.selftest else args.kappa
    u_max = args.umax_frac * pscale
    prompts = test[:n_eval]

    print("loading scorers ...")
    sent = HFScorer(SENT_MODEL, POS_EX, NEG_EX, device)
    form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)

    # baseline (unsteered) → per-prompt target FK + baseline formality
    base_txt, _ = closed_loop_generate(model, tok, prompts, gF, gR, band, 0.0, 0.0, u_max,
                                       np.zeros(len(prompts)), n_chunks, chunk_tok, device)
    target_FK = np.array([fk_grade(t) for t in base_txt], float)
    base_F = form.score(base_txt); base_R = np.array([fk_grade(t) for t in base_txt], float)
    baseM = {"formality": float(np.nanmean(base_F)), "reading": float(np.nanmean(base_R))}
    print(f"baseline: formality={baseM['formality']:.3f} reading={baseM['reading']:.2f} "
          f"distinct2={np.nanmean([distinct2(t) for t in base_txt]):.2f}")

    conds = []
    for m_F in MF:
        for kap in KAP:
            txt, trace = closed_loop_generate(model, tok, prompts, gF, gR, band, m_F * pscale, kap,
                                              u_max, target_FK, n_chunks, chunk_tok, device)
            F = form.score(txt); R = np.array([fk_grade(t) for t in txt], float)
            d2 = np.array([distinct2(t) for t in txt], float)
            Ag = F - base_F; Bd = R - base_R
            row = {"m_F": m_F, "kappa": kap, "open_loop": kap == 0.0,
                   "A_gain": float(np.nanmean(Ag)), "B_drift": float(np.nanmean(Bd)),
                   "B_absdrift": float(np.nanmean(np.abs(Bd))), "distinct2": float(np.nanmean(d2)),
                   "eR_trace": trace, "A_per": Ag.tolist(), "B_per": Bd.tolist(),
                   "sample": txt[0][:160]}
            conds.append(row)
            print(f"  m_F={m_F:<5} κ={kap:<5} {'(drive-only)' if kap==0 else '(closed-loop)'}: "
                  f"A_gain={row['A_gain']:+.3f}  B_drift={row['B_drift']:+.2f} (|{row['B_absdrift']:.2f}|)  "
                  f"dist2={row['distinct2']:.2f}  |e_R| trace={['%.1f'%t for t in trace]}")

    verdict = decide(conds, args)
    print("\n==== VERDICT ====")
    for l in verdict["lines"]:
        print(" ", l)

    out = {"meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale,
                    "n_eval": n_eval, "n_chunks": n_chunks, "chunk_tok": chunk_tok, "mf": MF,
                    "kappa": KAP, "umax_frac": args.umax_frac, "coh_floor": args.coh_floor,
                    "selftest": args.selftest},
           "baseline": baseM, "conditions": conds, "verdict": verdict}
    name = args.model.split("/")[-1]; tag = "_selftest" if args.selftest else ""
    json.dump(out, open(OUT / f"mosaic_phase4_{name}{tag}.json", "w"), indent=2)
    (OUT / f"mosaic_phase4_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    if not args.selftest:
        try:
            plot(conds, OUT / f"mosaic_phase4_{name}.png")
            print(f"plot -> outputs/mosaic_phase4_{name}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"saved outputs/mosaic_phase4_{name}{tag}.json")
    return out


def _coh(rows, floor):
    return [r for r in rows if r["distinct2"] >= floor]


def _interp(rows, g, key="B_absdrift"):
    pts = sorted(rows, key=lambda r: r["A_gain"])
    gs = [r["A_gain"] for r in pts]
    if not pts or g < min(gs) or g > max(gs):
        return np.nan
    return float(np.interp(g, gs, [r[key] for r in pts]))


def decide(conds, args):
    lines = []
    drive = _coh([r for r in conds if r["open_loop"]], args.coh_floor)
    closed = _coh([r for r in conds if not r["open_loop"]], args.coh_floor)
    # does feedback reduce |B-drift| as κ increases at fixed m_F? (sensor/loop is working)
    works = []
    for m_F in sorted(set(r["m_F"] for r in conds)):
        rows = sorted([r for r in conds if r["m_F"] == m_F], key=lambda r: r["kappa"])
        if len(rows) >= 2:
            bd = [r["B_absdrift"] for r in rows]
            works.append(bd[-1] < bd[0] - 1e-6)
            lines.append(f"m_F={m_F}: |B-drift| vs κ {[ (r['kappa'], round(r['B_absdrift'],2), round(r['A_gain'],3)) for r in rows]}")
    loop_modulates = any(works)
    # frontier dominance: closed-loop |B-drift| vs drive-only at matched formality-gain
    if drive and closed:
        gmax = min(max(r["A_gain"] for r in drive), max(r["A_gain"] for r in closed))
        gmin = max(min(r["A_gain"] for r in drive), min(r["A_gain"] for r in closed))
        wins, paired = [], []
        for g in np.linspace(gmin, max(gmin, gmax), 5):
            d_dr = _interp(drive, g); d_cl = _interp(closed, g)
            if np.isfinite(d_dr) and np.isfinite(d_cl):
                # paired SE at the nearest closed point
                rc = min(closed, key=lambda r: abs(r["A_gain"] - g))
                rd = min(drive, key=lambda r: abs(r["A_gain"] - g))
                da = np.abs(np.array(rc["B_per"], float)); dd = np.abs(np.array(rd["B_per"], float))
                diff = dd - da
                se = float(np.nanstd(diff) / np.sqrt(np.sum(np.isfinite(diff))))
                md = float(np.nanmean(diff))
                wins.append(md > args.win_se * se and d_cl < d_dr)
                paired.append((round(float(g), 3), round(d_dr, 2), round(d_cl, 2), round(md, 2), round(se, 2)))
        lines.append("matched-A-gain (g, drive|B|, closed|B|, paired drive−closed, SE): " + str(paired))
        frontier_win = any(wins)
    else:
        frontier_win = False
        lines.append("frontier: insufficient coherent overlap between drive-only and closed-loop")
    if frontier_win:
        v = ("WIN (closed-loop beats open-loop): output feedback holds reading > 1 SE tighter than the "
             "open-loop drive at matched formality-gain — generation-time re-entanglement IS controllable "
             "by output feedback, unlike the open-loop projection Phase 3 tied.")
    elif loop_modulates:
        v = ("NULL (entanglement irreducible): the loop genuinely regulates the sensor (|B-drift| drops "
             "with κ) but only by sacrificing formality-gain and/or coherence — at matched formality-gain "
             "it does NOT hold reading tighter than open-loop. With a PERFECT (FK) sensor, the formality↔"
             "reading entanglement is not a controllable disturbance: 'formal-but-simple' is off-manifold "
             "for the model. Closes the MOSAIC arc — the side-effect is intrinsic, not a control problem.")
    else:
        v = ("INCONCLUSIVE: the closed loop did not measurably modulate reading-drift (check κ range / "
             "sensor / u_max).")
    lines.insert(0, v)
    return {"verdict": v, "frontier_win": bool(frontier_win), "loop_modulates": bool(loop_modulates),
            "lines": lines}


def plot(conds, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    dr = sorted([r for r in conds if r["open_loop"]], key=lambda r: r["A_gain"])
    cl = sorted([r for r in conds if not r["open_loop"]], key=lambda r: r["A_gain"])
    for rows, col, lab, mk in ((dr, "C3", "open-loop drive (κ=0)", "o"), (cl, "C0", "closed-loop (κ>0)", "s")):
        xs = [r["A_gain"] for r in rows]; ys = [r["B_absdrift"] for r in rows]
        ax.plot(xs, ys, mk + "-", color=col, label=lab)
        for r in rows:
            if r["distinct2"] < 0.85:
                ax.plot(r["A_gain"], r["B_absdrift"], "x", color=col, ms=9)
    ax.set_xlabel("formality-gain ↑"); ax.set_ylabel("|reading-drift| ↓ (tighter hold)")
    ax.set_title("Phase 4: closed-loop hold vs open-loop drive\n(lower = tighter hold at equal formality; ×=incoherent)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1]
    for r in conds:
        if not r["open_loop"]:
            ax.plot(range(1, len(r["eR_trace"]) + 1), r["eR_trace"], "-o",
                    label=f"m_F={r['m_F']} κ={r['kappa']}")
    ax.set_xlabel("chunk"); ax.set_ylabel("mean |reading error| (FK)")
    ax.set_title("closed-loop regulation trace (does the loop drive reading-error down?)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("MOSAIC Phase 4 — closed-loop token-level hold (output feedback)")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def _selftest():
    # chunk-hook math: per-batch push added correctly, broadcast over sequence
    d = 16
    P = {0: torch.arange(2 * d, dtype=torch.float32).reshape(2, d)}

    class M:
        def named_modules(self):
            return {"model.layers.0": object()}
    hooks = chunk_hooks(M(), [0], P)
    h = torch.zeros(2, 3, d)
    out = hooks[0][1](None, None, (h, "kv"))
    assert torch.allclose(out[0][0], P[0][0][None].expand(3, d)), "batch-0 push wrong"
    assert torch.allclose(out[0][1], P[0][1][None].expand(3, d)), "batch-1 push wrong"
    print("[mosaic_phase4 self-test] OK — per-batch chunk push broadcasts over sequence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-eval", type=int, default=32, dest="n_eval")
    ap.add_argument("--n-chunks", type=int, default=4, dest="n_chunks")
    ap.add_argument("--chunk-tok", type=int, default=32, dest="chunk_tok")
    ap.add_argument("--mf", type=float, nargs="+", default=[0.04, 0.06, 0.08])
    ap.add_argument("--kappa", type=float, nargs="+", default=[0.0, 4.0, 8.0, 16.0])
    ap.add_argument("--umax-frac", type=float, default=0.15, dest="umax_frac")
    ap.add_argument("--coh-floor", type=float, default=0.85, dest="coh_floor")
    ap.add_argument("--win-se", type=float, default=1.0, dest="win_se")
    args = ap.parse_args()
    if args.selftest and not args.run:
        _selftest(); run(args); return
    if not args.run:
        ap.error("pass --run or --selftest")
    run(args)


if __name__ == "__main__":
    main()
