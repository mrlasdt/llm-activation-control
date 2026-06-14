"""MOSAIC Phase 6 — full MIMO setpoint TRACKING: drive (formality, reading) to commanded setpoints.

Phase 4 won at a HOLD task (drive formality up, regulate reading back to *baseline*). Phase 6 asks the
harder, fully general question: can a closed loop place BOTH outputs at ARBITRARY commanded setpoints
(τ_F, τ_R) — including the ANTI-correlated corner "formal but SIMPLE", which the formality↔reading
semantic entanglement fights hardest? And does a model-based DECOUPLING controller (that knows the
plant's cross-coupling) beat a naive DIAGONAL one (that treats the axes as independent)?

Two-input two-output regulator, chunked, integral action, per batch element b:
    push_l[b] = a_F[b]·g_F,l + a_R[b]·g_R,l ,           ‖push_l[b]‖ ≤ u_max          (actuator)
    y[b] = (formality(text), FK(text)) ,  e[b] = ([τ_F,τ_R] − y[b]) / s            (normalized error)
    diagonal   : a[b] += Kp · ( diag(G)⁻¹ · e[b] )      (correct per-axis scale, IGNORES coupling)
    decoupling : a[b] += Kp · (      G⁻¹  · e[b] )      (inverts the 2×2 plant — cancels coupling)
G is the static plant gain (normalized Δoutput per unit a), calibrated once by probing g_F, g_R alone.
Both controllers get correct diagonal scaling, so the ONLY difference is whether off-diagonal coupling
is accounted for — a clean A/B for "does decoupling help?".

Plant CONTROLLABILITY: if g_F and g_R produce near-collinear output effects, G is ill-conditioned →
the two outputs cannot be placed independently (the anti-correlated setpoint is structurally
unreachable within ‖u‖≤u_max). cond(G) / det(G) is therefore the *quantitative* statement of the
"semantic entanglement" Phase 3 hypothesized — reported as a first-class diagnostic.

WIN  : on the anti-correlated setpoint, decoupling tracks with lower final error than diagonal AND
       stays coherent — closed-loop MIMO extends the reachable set past the entanglement.
NULL : both saturate (high residual error / coherence collapse) at the anti-correlated corner ⇒ the
       reachable set has an intrinsic boundary; cond(G) quantifies how collinear the actuators are.

Run:  ../../.venv/bin/python mosaic_phase6.py --run
      ../../.venv/bin/python mosaic_phase6.py --selftest   (controller math: decoupling REACHES the
            anti-correlated setpoint on a synthetic coupled plant, and a near-collinear plant makes the
            inverse demand unbounded authority. NB: with integral action a *diagonal* controller also
            reaches the setpoint at steady state on a full-rank plant — the real track() difference is
            the TRANSIENT under the bounded-‖a‖ budget, which the model run, not this gate, exercises.)
Outputs: outputs/mosaic_phase6_<model>.{json,txt,png}.
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
    FORM_MODEL, FORMAL_EX, INFORMAL_EX, HFScorer, fk_grade, distinct2,
    load_model, load_band_pscale,
)
from mosaic_phase3 import load_cones, gdir                                      # noqa: E402
from mosaic_phase4 import chunk_hooks, _relpad                                  # noqa: E402

OUT = _HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)


def _push_from_a(a, gF, gR, band, u_max):
    """a:(B,2) actuator commands → per-layer combined push {l:(B,d)}, each row clipped to ‖·‖≤u_max."""
    P = {}
    for l in band:
        Pl = a[:, 0:1] * gF[l][None, :] + a[:, 1:2] * gR[l][None, :]            # (B,d)
        nrm = np.linalg.norm(Pl, axis=1, keepdims=True)
        P[l] = (Pl * np.minimum(1.0, u_max / (nrm + 1e-6))).astype(np.float32)
    return P


def measure(model, tok, prompts, P_by_layer, band, device, form, n_chunks, chunk_tok):
    """Generate with a FIXED per-layer push and return per-prompt (formality, FK, distinct2). Used both
    for plant calibration and as the open-loop generator inside the tracking loop's chunks."""
    pad = tok.pad_token_id
    inp = tokenize_instructions_fn(prompts, tok)
    ids = inp.input_ids.to(device)
    n_prompt = inp.attention_mask.sum(1).tolist()
    B = len(prompts)
    push = {l: torch.from_numpy(P_by_layer[l]) for l in band} if P_by_layer is not None else None

    def texts():
        return [tok.decode(ids[b][ids[b] != pad][n_prompt[b]:], skip_special_tokens=True) for b in range(B)]

    for _ in range(n_chunks):
        attn = (ids != pad).long()
        hooks = chunk_hooks(model, band, push) if push is not None else []
        with add_hooks(module_forward_hooks=hooks):
            with torch.no_grad():
                ids = model.generate(ids, attention_mask=attn, max_new_tokens=chunk_tok,
                                     do_sample=False, pad_token_id=pad)
        ids = _relpad(ids, pad)
    txt = texts()
    return (form.score(txt), np.array([fk_grade(t) for t in txt], float),
            np.array([distinct2(t) for t in txt], float), txt)


def track(model, tok, prompts, gF, gR, band, setpoint, G, s, mode, Kp, u_max, device, form,
          n_chunks, chunk_tok, a_clip):
    """Closed-loop MIMO tracking. setpoint=(τ_F,τ_R). G:(2,2) normalized plant gain. s:(2,) output
    scales. mode∈{diagonal,decoupling}. Integral action on a:(B,2). Returns final scores + traces."""
    pad = tok.pad_token_id
    inp = tokenize_instructions_fn(prompts, tok)
    ids = inp.input_ids.to(device)
    n_prompt = inp.attention_mask.sum(1).tolist()
    B = len(prompts)
    a = np.zeros((B, 2))
    Gd_inv = np.diag(1.0 / np.diag(G))                       # diagonal-only inverse
    G_inv = np.linalg.inv(G)
    M = Gd_inv if mode == "diagonal" else G_inv
    tau = np.array(setpoint, float)
    err_trace, y_trace = [], []

    def texts():
        return [tok.decode(ids[b][ids[b] != pad][n_prompt[b]:], skip_special_tokens=True) for b in range(B)]

    for c in range(n_chunks):
        P = _push_from_a(a, gF, gR, band, u_max)
        push = {l: torch.from_numpy(P[l]) for l in band}
        attn = (ids != pad).long()
        with add_hooks(module_forward_hooks=chunk_hooks(model, band, push)):
            with torch.no_grad():
                ids = model.generate(ids, attention_mask=attn, max_new_tokens=chunk_tok,
                                     do_sample=False, pad_token_id=pad)
        ids = _relpad(ids, pad)
        txt = texts()
        yF = form.score(txt); yR = np.array([fk_grade(t) for t in txt], float)
        y = np.stack([yF, np.nan_to_num(yR, nan=float(np.nanmean(yR)))], 1)     # (B,2)
        e = (tau[None, :] - y) / s[None, :]                                     # normalized error
        a = a + Kp * (e @ M.T)                                                  # integral update
        anrm = np.linalg.norm(a, axis=1, keepdims=True)                         # anti-windup clamp
        a = a * np.minimum(1.0, a_clip / (anrm + 1e-9))
        err_trace.append(float(np.nanmean(np.linalg.norm(e, axis=1))))
        y_trace.append([float(np.nanmean(yF)), float(np.nanmean(yR))])
    txt = texts()
    yF = form.score(txt); yR = np.array([fk_grade(t) for t in txt], float)
    d2 = np.array([distinct2(t) for t in txt], float)
    eF = (tau[0] - yF) / s[0]; eR = (tau[1] - yR) / s[1]
    track_err = np.sqrt(eF ** 2 + np.nan_to_num(eR, nan=0.0) ** 2)
    return {"mode": mode, "setpoint": list(map(float, setpoint)),
            "y_formality": float(np.nanmean(yF)), "y_reading": float(np.nanmean(yR)),
            "track_err": float(np.nanmean(track_err)), "track_err_per": track_err.tolist(),
            "eF_norm": float(np.nanmean(np.abs(eF))), "eR_norm": float(np.nanmean(np.abs(eR))),
            "distinct2": float(np.nanmean(d2)), "err_trace": err_trace, "y_trace": y_trace,
            "a_mean": [float(np.nanmean(a[:, 0])), float(np.nanmean(a[:, 1]))],
            "sample": txt[0][:160]}


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    C, cband = load_cones(model); assert cband == band, "cone band mismatch — re-run Phase 2"
    gF = {l: gdir(C, "formality", i) for i, l in enumerate(band)}
    gR = {l: gdir(C, "reading", i) for i, l in enumerate(band)}
    u_max = args.umax_frac * pscale
    if args.a_clip is None:                                  # anti-windup: bound ‖a‖ near push saturation
        args.a_clip = 1.5 * u_max
    print(f"model={args.model} band={band[0]}..{band[-1]} pscale={pscale:.1f} u_max={u_max:.1f} "
          f"a_clip={args.a_clip:.1f}")

    _, test = get_input_data("harmless")
    n_eval = 4 if args.selftest else args.n_eval
    n_chunks = 3 if args.selftest else args.n_chunks
    chunk_tok = 16 if args.selftest else args.chunk_tok
    prompts = test[:n_eval]

    print("loading formality scorer ...")
    form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)

    # ---- baseline + plant calibration (static gain G in NORMALIZED output units) ----
    bF, bR, bd2, _ = measure(model, tok, prompts, None, band, device, form, n_chunks, chunk_tok)
    y0 = np.array([float(np.nanmean(bF)), float(np.nanmean(bR))])
    s = np.array(args.scales, float)                          # per-axis normalizers [sF, sR]
    a0 = args.probe_frac * pscale
    pF, rF, _, _ = measure(model, tok, prompts, _push_from_a(np.tile([a0, 0.0], (n_eval, 1)),
                                                             gF, gR, band, u_max), band, device, form, n_chunks, chunk_tok)
    pR, rR, _, _ = measure(model, tok, prompts, _push_from_a(np.tile([0.0, a0], (n_eval, 1)),
                                                             gF, gR, band, u_max), band, device, form, n_chunks, chunk_tok)
    # G[:,j] = (Δoutput / s) per unit a_j  → columns are the two actuators' normalized effects
    G = np.array([
        [(np.nanmean(pF) - y0[0]) / s[0] / a0, (np.nanmean(pR) - y0[0]) / s[0] / a0],
        [(np.nanmean(rF) - y0[1]) / s[1] / a0, (np.nanmean(rR) - y0[1]) / s[1] / a0],
    ])
    cond = float(np.linalg.cond(G)); det = float(np.linalg.det(G))
    # collinearity of the two actuators' normalized output effects (cos of G's columns)
    c0, c1 = G[:, 0], G[:, 1]
    col_cos = float(c0 @ c1 / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-12))
    print(f"baseline y0=(F={y0[0]:.3f}, R={y0[1]:.2f}) distinct2={np.nanmean(bd2):.2f}")
    print(f"plant G (normalized Δout/a) =\n{np.array2string(G, precision=4)}")
    print(f"  cond(G)={cond:.2f}  det(G)={det:.4f}  actuator-effect collinearity cos={col_cos:+.3f}")

    # ---- setpoint grid (offsets from baseline; the anti-correlated corner is the headline) ----
    SP = setpoints(y0, args)
    print(f"setpoints (F,R): " + ", ".join(f"{k}=({v[0]:.2f},{v[1]:.1f})" for k, v in SP.items()))

    results = {}
    for name, tau in SP.items():
        results[name] = {}
        for mode in ("diagonal", "decoupling"):
            r = track(model, tok, prompts, gF, gR, band, tau, G, s, mode, args.kp, u_max, device,
                      form, n_chunks, chunk_tok, args.a_clip)
            results[name][mode] = r
            print(f"  [{name:16s}] {mode:10s}: y=(F={r['y_formality']:.3f},R={r['y_reading']:.2f}) "
                  f"trackErr={r['track_err']:.3f} (|eF|={r['eF_norm']:.2f} |eR|={r['eR_norm']:.2f}) "
                  f"dist2={r['distinct2']:.2f}")

    verdict = decide(results, SP, cond, col_cos, args)
    print("\n==== VERDICT ====")
    for l in verdict["lines"]:
        print(" ", l)

    out = {"meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale,
                    "n_eval": n_eval, "n_chunks": n_chunks, "chunk_tok": chunk_tok,
                    "umax_frac": args.umax_frac, "probe_frac": args.probe_frac, "kp": args.kp,
                    "a_clip": args.a_clip, "scales": list(s), "coh_floor": args.coh_floor,
                    "win_se": args.win_se, "selftest": args.selftest},
           "baseline_y": list(y0), "G": G.tolist(), "cond_G": cond, "det_G": det,
           "actuator_collinearity": col_cos, "setpoints": {k: list(v) for k, v in SP.items()},
           "results": results, "verdict": verdict}
    name = args.model.split("/")[-1]; tag = "_selftest" if args.selftest else ""
    json.dump(out, open(OUT / f"mosaic_phase6_{name}{tag}.json", "w"), indent=2)
    (OUT / f"mosaic_phase6_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    if not args.selftest:
        try:
            plot(results, SP, y0, OUT / f"mosaic_phase6_{name}.png")
            print(f"plot -> outputs/mosaic_phase6_{name}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"saved outputs/mosaic_phase6_{name}{tag}.json  (elapsed {time.time()-t0:.0f}s)")
    return out


def setpoints(y0, args):
    """Setpoint grid as offsets from baseline y0=(F,R). The anti-correlated corner (formality UP,
    reading DOWN) is the one the entanglement fights — the headline test."""
    dF, dR = args.dform, args.dread
    Fhi = min(y0[0] + dF, args.form_cap)                      # cap near classifier saturation
    Flo = max(y0[0] - dF, 0.05)
    return {
        "hold_reading":   (Fhi,   y0[1]),        # formality↑, hold reading (Phase-4 sanity)
        "formal_simpler": (Fhi,   y0[1] - dR),   # ANTI-correlated — the hard corner (headline)
        "formal_complex": (Fhi,   y0[1] + dR),   # correlated — easy
        "casual_simpler": (Flo,   y0[1] - dR),   # both down
        "hold_F_simpler": (y0[0], y0[1] - dR),   # hold formality, simplify
    }


# setpoints whose two axes pull AGAINST the natural coupling (the controllability test cases)
ANTI = {"formal_simpler"}


def decide(results, SP, cond, col_cos, args):
    lines = [f"plant: cond(G)={cond:.2f}, actuator-effect collinearity cos={col_cos:+.3f} "
             f"(|cos|→1 ⇒ near-collinear actuators ⇒ anti-correlated setpoints structurally hard)"]
    decoup_wins, both_fail = [], []
    for name in SP:
        di = results[name]["diagonal"]; de = results[name]["decoupling"]
        # paired improvement of decoupling over diagonal on per-prompt tracking error
        ed = np.array(di["track_err_per"], float); ee = np.array(de["track_err_per"], float)
        diff = ed - ee                                        # >0 ⇒ decoupling tracks tighter
        se = float(np.nanstd(diff) / np.sqrt(np.sum(np.isfinite(diff))))
        md = float(np.nanmean(diff))
        de_coh = de["distinct2"] >= args.coh_floor
        win = md > args.win_se * se and de["track_err"] < di["track_err"] and de_coh
        fail = di["track_err"] > args.fail_err and de["track_err"] > args.fail_err
        tag = "ANTI" if name in ANTI else "    "
        lines.append(f"[{tag} {name:16s}] trackErr diag={di['track_err']:.3f} decoup={de['track_err']:.3f} "
                     f"| paired diag−decoup={md:+.3f}±{se:.3f}SE → {'DECOUP-WIN' if win else 'tie'} "
                     f"| decoup dist2={de['distinct2']:.2f}{' [incoherent]' if not de_coh else ''}")
        if win:
            decoup_wins.append(name)
        if fail:
            both_fail.append(name)
    anti_win = any(n in ANTI for n in decoup_wins)
    anti_fail = any(n in ANTI for n in both_fail)
    if anti_win:
        v = (f"WIN (MIMO decoupling extends the reachable set): on the anti-correlated setpoint(s) "
             f"{[n for n in decoup_wins if n in ANTI]}, the decoupling controller (which inverts the "
             f"plant's cross-coupling) tracks > {args.win_se:g} SE tighter than the naive diagonal "
             f"controller AND stays coherent — closed-loop output feedback places BOTH attributes at a "
             f"setpoint the formality↔reading entanglement fights, generalizing the Phase-4 hold win to "
             f"full 2-DoF tracking.")
    elif anti_fail:
        v = (f"NULL/BOUNDARY (intrinsic reachable-set limit): at the anti-correlated corner BOTH "
             f"controllers leave large residual error (cond(G)={cond:.1f}, actuator collinearity "
             f"cos={col_cos:+.2f}) — the two style actuators produce near-collinear output effects, so "
             f"'formal-but-simple' is structurally near-unreachable within ‖u‖≤u_max regardless of "
             f"controller. cond(G) is the quantitative form of the Phase-3 entanglement hypothesis.")
    else:
        v = ("MIXED: decoupling helps on some setpoints but not decisively on the anti-correlated corner "
             "(see per-setpoint lines); report cond(G) and the per-setpoint tracking table.")
    lines.insert(0, v)
    return {"verdict": v, "decoup_wins": decoup_wins, "both_fail": both_fail,
            "anti_win": bool(anti_win), "anti_fail": bool(anti_fail), "lines": lines}


def plot(results, SP, y0, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]                                              # output plane: setpoints vs achieved
    ax.plot(y0[0], y0[1], "k*", ms=14, label="baseline")
    for name, tau in SP.items():
        ax.plot(tau[0], tau[1], "kD", ms=8)
        ax.annotate(name, (tau[0], tau[1]), fontsize=6, alpha=0.7)
        di = results[name]["diagonal"]; de = results[name]["decoupling"]
        ax.plot(di["y_formality"], di["y_reading"], "o", color="C3")
        ax.plot(de["y_formality"], de["y_reading"], "s", color="C0")
        ax.plot([tau[0], di["y_formality"]], [tau[1], di["y_reading"]], "-", color="C3", alpha=0.3)
        ax.plot([tau[0], de["y_formality"]], [tau[1], de["y_reading"]], "-", color="C0", alpha=0.3)
    ax.plot([], [], "o", color="C3", label="diagonal achieved")
    ax.plot([], [], "s", color="C0", label="decoupling achieved")
    ax.plot([], [], "kD", label="setpoint (target)")
    ax.set_xlabel("formality P(formal)"); ax.set_ylabel("reading FK grade")
    ax.set_title("Phase 6: setpoint tracking in the output plane\n(shorter line to ◆ = better tracking)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[1]                                              # per-setpoint tracking error bars
    names = list(SP.keys()); x = np.arange(len(names))
    dia = [results[n]["diagonal"]["track_err"] for n in names]
    dec = [results[n]["decoupling"]["track_err"] for n in names]
    ax.bar(x - 0.2, dia, 0.4, color="C3", label="diagonal")
    ax.bar(x + 0.2, dec, 0.4, color="C0", label="decoupling")
    for i, n in enumerate(names):
        if n in ANTI:
            ax.annotate("ANTI", (i, max(dia[i], dec[i])), ha="center", fontsize=7, color="purple")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("final tracking error ‖e‖ (normalized) ↓"); ax.legend()
    ax.set_title("tracking error by setpoint (lower = better)")
    fig.suptitle("MOSAIC Phase 6 — MIMO setpoint tracking: decoupling vs diagonal control")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def _selftest():
    """Controller math on a SYNTHETIC coupled plant: decoupling drives e→0; diagonal leaves
    cross-coupling residual. No model."""
    rng = np.random.RandomState(0)
    # true plant: y = Gp · a, with strong off-diagonal coupling (the entanglement)
    Gp = np.array([[1.0, 0.6], [0.5, 1.0]])
    s = np.array([1.0, 1.0]); tau = np.array([1.0, -1.0])     # anti-correlated target
    for mode in ("diagonal", "decoupling"):
        a = np.zeros(2)
        Gd_inv = np.diag(1.0 / np.diag(Gp)); G_inv = np.linalg.inv(Gp)
        M = Gd_inv if mode == "diagonal" else G_inv
        for _ in range(200):
            y = Gp @ a
            e = (tau - y) / s
            a = a + 0.3 * (M @ e)
        y = Gp @ a
        err = np.linalg.norm(tau - y)
        print(f"  [{mode:10s}] final y={np.array2string(y, precision=3)} ‖e‖={err:.4f}")
        if mode == "decoupling":
            assert err < 1e-3, f"decoupling should reach the anti-correlated setpoint (‖e‖={err})"
    # diagonal on a coupled plant with integral action still converges (it's stable) — the model
    # difference is the TRANSIENT + the ‖u‖ budget; assert decoupling needs no more authority here:
    cond = np.linalg.cond(Gp)
    print(f"  synthetic cond(G)={cond:.2f} (well-conditioned ⇒ both reach it given unbounded a)")
    # near-collinear plant: decoupling demands huge a (controllability loss)
    Gsing = np.array([[1.0, 0.98], [1.0, 1.0]])
    a = np.linalg.inv(Gsing) @ np.array([1.0, -1.0])
    print(f"  near-collinear cond(G)={np.linalg.cond(Gsing):.1f} ⇒ decoupling ‖a‖={np.linalg.norm(a):.1f} "
          f"(blows up — the anti-correlated setpoint needs unbounded authority)")
    assert np.linalg.norm(a) > 10, "near-collinear plant should demand large authority"
    print("[mosaic_phase6 self-test] OK — decoupling reaches the anti-correlated setpoint; near-collinear "
          "plant ⇒ inverse demands unbounded authority (diagonal also converges at steady state — the "
          "discriminator is the bounded-‖a‖ transient, tested in the model run, not this gate)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-eval", type=int, default=16, dest="n_eval")
    ap.add_argument("--n-chunks", type=int, default=6, dest="n_chunks")
    ap.add_argument("--chunk-tok", type=int, default=20, dest="chunk_tok")
    ap.add_argument("--umax-frac", type=float, default=0.15, dest="umax_frac")
    ap.add_argument("--probe-frac", type=float, default=0.06, dest="probe_frac")
    ap.add_argument("--kp", type=float, default=0.5, help="integral loop gain")
    ap.add_argument("--a-clip", type=float, default=None, dest="a_clip",
                    help="anti-windup clamp on ‖a‖ (default = 1.5×u_max, set in run() once pscale known)")
    ap.add_argument("--scales", type=float, nargs=2, default=[0.2, 4.0],
                    help="per-axis output normalizers [sF, sR]")
    ap.add_argument("--dform", type=float, default=0.15, help="formality setpoint offset")
    ap.add_argument("--dread", type=float, default=3.0, help="reading (FK) setpoint offset")
    ap.add_argument("--form-cap", type=float, default=0.95, dest="form_cap")
    ap.add_argument("--coh-floor", type=float, default=0.85, dest="coh_floor")
    ap.add_argument("--fail-err", type=float, default=0.7, dest="fail_err",
                    help="normalized tracking error above which a setpoint counts as 'not reached'")
    ap.add_argument("--win-se", type=float, default=1.0, dest="win_se")
    args = ap.parse_args()
    if args.selftest and not args.run:
        _selftest(); return
    if not args.run:
        ap.error("pass --run or --selftest")
    run(args)


if __name__ == "__main__":
    main()
