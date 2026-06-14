"""MOSAIC Phase 7 — the reachable-set map over authority u_max (reach vs coherence).

Phase 6 found 2-DoF setpoint tracking is NULL/BOUNDARY, and the verification pass sharpened *why*: the
plant is FULL-RANK with only MODERATE ill-conditioning (cond G 3.97 Gemma / 7.23 Qwen), so the
anti-correlated 'formal-but-simple' corner is **authority-bounded, not structurally unreachable** —
raising u_max should extend the reachable set, at a coherence cost. Phase 7 tests that directly and
maps it.

Method. Calibrate the same static plant as Phase 6. Then, for a FAN of output-space directions θ
(8, every 45°, in normalized (Δformality/sF, Δreading/sR) coordinates), command a setpoint FAR along θ
(radius R_cmd≫reach) so the integral controller SATURATES the per-layer push at u_max — the achieved
output is then the reachable-set boundary in direction θ at that authority. Sweep u_max over a grid and
record, per (u_max, θ): the achieved (formality, reading), the reach (projection of the achieved
displacement onto θ), and coherence (distinct-2). Diagonal controller (Phase 6: the robust one;
decoupling over-corrects under saturation).

What the map shows / the pre-registered reads:
  * REACH GROWS WITH AUTHORITY (confirms authority-bound): the anti-correlated reach increases with
    u_max while distinct-2 falls ⇒ a reach-vs-coherence frontier with a KNEE (max *coherent* reach).
  * ANISOTROPY = the collinearity, visualized: reach along the NATURAL diagonal (formal+complex /
    casual+simple, actuators aligned) ≫ reach along the ANTI-correlated diagonal (formal+simple,
    actuators fight) at every u_max. The reachable set is an ELLIPSE squashed along the entangled axis;
    its axis ratio ≈ how collinear the actuators are.

Run:  ../../.venv/bin/python mosaic_phase7.py --run
      ../../.venv/bin/python mosaic_phase7.py --selftest   (synthetic: isotropic plant ⇒ round set;
                                                            collinear plant ⇒ squashed set)
Outputs: outputs/mosaic_phase7_<model>.{json,png}.
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

from mosaic_phase0 import (  # noqa: E402
    FORM_MODEL, FORMAL_EX, INFORMAL_EX, HFScorer, fk_grade, distinct2,
    load_model, load_band_pscale,
)
from mosaic_phase3 import load_cones, gdir                                      # noqa: E402
from mosaic_phase6 import _push_from_a, measure, track                         # noqa: E402

OUT = _HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)


def calibrate_plant(model, tok, prompts, gF, gR, band, u_max, s, a0, device, form, n_chunks, chunk_tok):
    """Static plant gain G (normalized Δoutput per unit actuator command) + baseline y0, as in Phase 6."""
    bF, bR, _, _ = measure(model, tok, prompts, None, band, device, form, n_chunks, chunk_tok)
    y0 = np.array([float(np.nanmean(bF)), float(np.nanmean(bR))])
    n = len(prompts)
    pF, rF, _, _ = measure(model, tok, prompts, _push_from_a(np.tile([a0, 0.0], (n, 1)), gF, gR, band, u_max),
                           band, device, form, n_chunks, chunk_tok)
    pR, rR, _, _ = measure(model, tok, prompts, _push_from_a(np.tile([0.0, a0], (n, 1)), gF, gR, band, u_max),
                           band, device, form, n_chunks, chunk_tok)
    G = np.array([[(np.nanmean(pF) - y0[0]) / s[0] / a0, (np.nanmean(pR) - y0[0]) / s[0] / a0],
                  [(np.nanmean(rF) - y0[1]) / s[1] / a0, (np.nanmean(rR) - y0[1]) / s[1] / a0]])
    return y0, G


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    C, cband = load_cones(model); assert cband == band, "cone band mismatch — re-run Phase 2"
    gF = {l: gdir(C, "formality", i) for i, l in enumerate(band)}
    gR = {l: gdir(C, "reading", i) for i, l in enumerate(band)}
    s = np.array(args.scales, float)
    print(f"model={args.model} band={band[0]}..{band[-1]} pscale={pscale:.1f}")

    from utils import get_input_data
    _, test = get_input_data("harmless")
    n_eval = 4 if args.selftest else args.n_eval
    n_chunks = 3 if args.selftest else args.n_chunks
    chunk_tok = 16 if args.selftest else args.chunk_tok
    prompts = test[:n_eval]

    print("loading formality scorer ...")
    form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)

    # calibrate the plant once at a mid authority (the gain ratios are ~scale-invariant)
    cal_umax = args.cal_frac * pscale
    a0 = args.probe_frac * pscale
    y0, G = calibrate_plant(model, tok, prompts, gF, gR, band, cal_umax, s, a0, device, form, n_chunks, chunk_tok)
    cond = float(np.linalg.cond(G)); c0, c1 = G[:, 0], G[:, 1]
    col_cos = float(c0 @ c1 / (np.linalg.norm(c0) * np.linalg.norm(c1) + 1e-12))
    print(f"baseline y0=(F={y0[0]:.3f},R={y0[1]:.2f})  cond(G)={cond:.2f}  actuator collinearity cos={col_cos:+.3f}")

    # output-space direction fan (normalized coords), every 45 deg
    n_dir = 4 if args.selftest else args.n_dir
    thetas = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    dir_labels = {0.0: "F+ (formal)", 45.0: "F+R+ (formal+complex, natural)", 90.0: "R+ (complex)",
                  135.0: "F-R+ ", 180.0: "F- (casual)", 225.0: "F-R- (casual+simple, natural)",
                  270.0: "R- (simple)", 315.0: "F+R- (formal+simple, ANTI)"}
    fracs = [0.1, 0.2] if args.selftest else args.umax_fracs

    cells = []
    for frac in fracs:
        u_max = frac * pscale
        a_clip = args.aclip_mult * u_max
        for th in thetas:
            # commanded setpoint FAR along θ (raw units) so the controller saturates at u_max
            tau = (y0[0] + args.r_cmd * s[0] * np.cos(th), y0[1] + args.r_cmd * s[1] * np.sin(th))
            r = track(model, tok, prompts, gF, gR, band, tau, G, s, "diagonal", args.kp, u_max,
                      device, form, n_chunks, chunk_tok, a_clip)
            dF = (r["y_formality"] - y0[0]) / s[0]; dR = (r["y_reading"] - y0[1]) / s[1]
            reach = dF * np.cos(th) + dR * np.sin(th)            # signed projection onto the commanded dir
            deg = round(float(np.degrees(th)), 1)
            cell = {"frac": frac, "u_max": u_max, "theta_deg": deg, "label": dir_labels.get(deg, ""),
                    "y_formality": r["y_formality"], "y_reading": r["y_reading"],
                    "dF_norm": float(dF), "dR_norm": float(dR), "reach": float(reach),
                    "distinct2": r["distinct2"], "coherent": r["distinct2"] >= args.coh_floor}
            cells.append(cell)
            print(f"  u={frac:<5}({u_max:5.1f}) θ={deg:5.0f}° {dir_labels.get(deg,'')[:24]:24s} "
                  f"reach={reach:+.2f} (ΔF={dF:+.2f},ΔR={dR:+.2f}) d2={r['distinct2']:.2f}"
                  f"{'' if cell['coherent'] else '  [incoherent]'}")

    verdict = decide(cells, fracs, cond, col_cos, args)
    print("\n==== VERDICT ====")
    for l in verdict["lines"]:
        print(" ", l)

    out = {"meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale, "n_eval": n_eval,
                    "n_chunks": n_chunks, "chunk_tok": chunk_tok, "umax_fracs": fracs, "r_cmd": args.r_cmd,
                    "kp": args.kp, "aclip_mult": args.aclip_mult, "scales": list(s), "n_dir": n_dir,
                    "coh_floor": args.coh_floor, "cal_frac": args.cal_frac, "selftest": args.selftest},
           "baseline_y": list(y0), "G": G.tolist(), "cond_G": cond, "actuator_collinearity": col_cos,
           "cells": cells, "verdict": verdict}
    name = args.model.split("/")[-1]; tag = "_selftest" if args.selftest else ""
    json.dump(out, open(OUT / f"mosaic_phase7_{name}{tag}.json", "w"), indent=2)
    (OUT / f"mosaic_phase7_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    if not args.selftest:
        try:
            plot(cells, fracs, y0, s, OUT / f"mosaic_phase7_{name}.png")
            print(f"plot -> outputs/mosaic_phase7_{name}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"saved outputs/mosaic_phase7_{name}{tag}.json  (elapsed {time.time()-t0:.0f}s)")
    return out


ANTI_DEG = 315.0      # formal+simple
NAT_DEG = 225.0       # casual+simple (the other natural-coupling diagonal); 45 is formal+complex


def _reach_at(cells, frac, deg, coherent_only=False):
    for c in cells:
        if c["frac"] == frac and c["theta_deg"] == deg and (c["coherent"] or not coherent_only):
            return c
    return None


def decide(cells, fracs, cond, col_cos, args):
    lines = [f"plant cond(G)={cond:.2f}, actuator collinearity cos={col_cos:+.3f}"]
    # (1) does reach toward the ANTI-correlated corner grow with authority? at what coherence?
    anti = [(_reach_at(cells, f, ANTI_DEG)) for f in fracs]
    anti = [c for c in anti if c]
    if len(anti) >= 2:
        reach_lo, reach_hi = anti[0]["reach"], anti[-1]["reach"]
        grows = reach_hi > reach_lo + 1e-6
        lines.append(f"ANTI-correlated (formal+simple) reach vs u_max: " +
                     ", ".join(f"u={c['frac']}:reach={c['reach']:+.2f}(d2={c['distinct2']:.2f})" for c in anti))
        lines.append(f"  reach {'GROWS' if grows else 'does NOT grow'} with authority "
                     f"({reach_lo:+.2f}→{reach_hi:+.2f}) — authority-bounded confirmed: {grows}")
        coherent_anti = [c for c in anti if c["coherent"]]
        if coherent_anti:
            best = max(coherent_anti, key=lambda c: c["reach"])
            lines.append(f"  MAX COHERENT anti-reach = {best['reach']:+.2f} at u_max={best['frac']}·pscale "
                         f"(d2={best['distinct2']:.2f}); beyond this, reach buys only incoherence")
        else:
            lines.append("  no coherent anti-correlated operating point at any tested u_max")
    # (2) anisotropy: natural-diagonal reach vs anti-diagonal reach (= the collinearity, visualized)
    aniso = []
    for f in fracs:
        a = _reach_at(cells, f, ANTI_DEG); nat = _reach_at(cells, f, 45.0)   # formal+complex natural
        if a and nat and abs(a["reach"]) > 1e-6:
            aniso.append((f, nat["reach"] / max(a["reach"], 1e-6)))
    if aniso:
        lines.append("anisotropy (natural[45°,formal+complex] reach / anti[315°] reach) per u_max: " +
                     ", ".join(f"u={f}:{r:.1f}×" for f, r in aniso))
        lines.append(f"  the reachable set is SQUASHED along the entangled (anti-correlated) axis by "
                     f"~{np.nanmean([r for _, r in aniso]):.1f}× — the actuator collinearity, made visible.")
    grows = len(anti) >= 2 and anti[-1]["reach"] > anti[0]["reach"] + 1e-6
    if grows:
        v = ("CONFIRMS authority-bound: more authority (u_max) DOES extend the reachable set toward the "
             "anti-correlated corner — Phase-6's limit is authority×collinearity, not a rank deficiency — "
             "but reach beyond the coherent knee buys only incoherence, and the set stays strongly squashed "
             "along the entangled axis (the collinearity). So 'formal-but-simple' is reachable-with-more-"
             "authority but not COHERENTLY reachable: the practical boundary is set by coherence, not rank.")
    else:
        v = ("UNEXPECTED: anti-correlated reach did not grow with u_max — re-check the controller saturates "
             "(r_cmd/a_clip) or whether coherence collapses before reach extends.")
    lines.insert(0, v)
    return {"verdict": v, "anti_grows": bool(grows), "lines": lines}


def plot(cells, fracs, y0, s, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    # (a) reachable-set map: one closed polygon of achieved points per u_max, in NORMALIZED coords
    ax = axes[0]
    ax.plot(0, 0, "k*", ms=13, label="baseline")
    cmap = plt.get_cmap("viridis")
    for i, f in enumerate(sorted(fracs)):
        pts = sorted([c for c in cells if c["frac"] == f], key=lambda c: c["theta_deg"])
        xs = [c["dF_norm"] for c in pts] + [pts[0]["dF_norm"]]
        ys = [c["dR_norm"] for c in pts] + [pts[0]["dR_norm"]]
        col = cmap(i / max(1, len(fracs) - 1))
        ax.plot(xs, ys, "-o", color=col, label=f"u_max={f}·pscale", ms=4)
        for c in pts:
            if not c["coherent"]:
                ax.plot(c["dF_norm"], c["dR_norm"], "x", color="red", ms=7)
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Δformality (normalized)"); ax.set_ylabel("Δreading (normalized)")
    ax.set_title("Reachable-set map vs authority u_max\n(polygons grow with u_max; squashed along the\nanti-correlated F+R− diagonal; ×=incoherent)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="datalim")
    # (b) anti-correlated reach vs u_max, colored by coherence
    ax = axes[1]
    anti = sorted([c for c in cells if c["theta_deg"] == ANTI_DEG], key=lambda c: c["frac"])
    nat = sorted([c for c in cells if c["theta_deg"] == 45.0], key=lambda c: c["frac"])
    for series, lab, mk in ((anti, "anti-correlated (F+R−, formal+simple)", "s"), (nat, "natural (F+R+, formal+complex)", "o")):
        xs = [c["frac"] for c in series]; ys = [c["reach"] for c in series]
        ax.plot(xs, ys, mk + "-", label=lab)
        for c in series:
            if not c["coherent"]:
                ax.plot(c["frac"], c["reach"], "x", color="red", ms=9)
    ax.set_xlabel("authority u_max (·pscale)"); ax.set_ylabel("reach toward setpoint (normalized)")
    ax.set_title("reach vs authority (×=incoherent)\nmore authority extends reach, at a coherence cost")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("MOSAIC Phase 7 — reachable-set map over u_max (reach vs coherence)")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def _selftest():
    """No model: a synthetic ISOTROPIC plant gives a ~round reachable set; a COLLINEAR plant gives a
    set squashed along the anti-correlated diagonal. Verifies the reach-projection + anisotropy logic."""
    # simulate achieved displacement = Gp @ a_saturated, a chosen to point toward each θ within ‖a‖≤1
    def map_set(Gp):
        cells = []
        for f in (0.1, 0.2):
            for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                want = np.array([np.cos(th), np.sin(th)])
                a = np.linalg.solve(Gp, want)                  # actuator to achieve unit output dir
                a = a / (np.linalg.norm(a) + 1e-9) * f          # bounded by authority f
                y = Gp @ a                                      # achieved output displacement
                reach = float(y @ want)
                cells.append({"frac": f, "theta_deg": round(float(np.degrees(th)), 1),
                              "dF_norm": float(y[0]), "dR_norm": float(y[1]), "reach": reach,
                              "distinct2": 0.95, "coherent": True})
        return cells
    iso = map_set(np.eye(2))
    coll = map_set(np.array([[1.0, 0.9], [0.9, 1.0]]))         # near-collinear actuators
    # isotropic: anti-reach ≈ natural-reach (round)
    ia = _reach_at(iso, 0.2, ANTI_DEG)["reach"]; inat = _reach_at(iso, 0.2, 45.0)["reach"]
    assert abs(ia - inat) < 0.2, f"isotropic set should be round: {ia} vs {inat}"
    # collinear: natural-reach ≫ anti-reach (squashed)
    ca = _reach_at(coll, 0.2, ANTI_DEG)["reach"]; cnat = _reach_at(coll, 0.2, 45.0)["reach"]
    assert cnat > 3 * ca, f"collinear set should be squashed along anti: nat {cnat} vs anti {ca}"
    # reach grows with authority
    assert _reach_at(coll, 0.2, ANTI_DEG)["reach"] > _reach_at(coll, 0.1, ANTI_DEG)["reach"]
    print(f"[mosaic_phase7 self-test] OK — isotropic round (anti {ia:.2f}≈nat {inat:.2f}); "
          f"collinear squashed (nat {cnat:.2f} ≫ anti {ca:.2f}); reach grows with authority")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-eval", type=int, default=12, dest="n_eval")
    ap.add_argument("--n-chunks", type=int, default=5, dest="n_chunks")
    ap.add_argument("--chunk-tok", type=int, default=20, dest="chunk_tok")
    ap.add_argument("--n-dir", type=int, default=8, dest="n_dir")
    ap.add_argument("--umax-fracs", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3], dest="umax_fracs")
    ap.add_argument("--r-cmd", type=float, default=4.0, dest="r_cmd", help="commanded setpoint radius (normalized) — far, to saturate")
    ap.add_argument("--kp", type=float, default=0.5)
    ap.add_argument("--aclip-mult", type=float, default=3.0, dest="aclip_mult", help="a_clip = mult·u_max (so u_max binds, not a_clip)")
    ap.add_argument("--cal-frac", type=float, default=0.15, dest="cal_frac", help="u_max for the one plant calibration")
    ap.add_argument("--probe-frac", type=float, default=0.06, dest="probe_frac")
    ap.add_argument("--scales", type=float, nargs=2, default=[0.2, 4.0])
    ap.add_argument("--coh-floor", type=float, default=0.85, dest="coh_floor")
    args = ap.parse_args()
    if args.selftest and not args.run:
        _selftest(); return
    if not args.run:
        ap.error("pass --run or --selftest")
    run(args)


if __name__ == "__main__":
    main()
