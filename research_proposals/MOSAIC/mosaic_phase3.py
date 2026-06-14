"""MOSAIC Phase 3 — the headline: constrained control ALLOCATION vs the best scalar push.

Drive attribute A to higher intensity while HOLDING attribute B at its unsteered baseline, using
the A-cone's null-space. Pre-registered (sharpened by Phase 2: the baseline is the best *PLS-scalar*
push, not crude diff-in-means):

  WIN  : the allocation push (in the A-cone, ⊥ B's readout) reaches a given A-gain with > 1 SE LESS
         B-drift than the scalar push (paired, same prompts; at matched A-tracking; coherent), i.e.
         the (A-gain, B-drift) Pareto frontier of allocation DOMINATES scalar's — a "control law >
         bound" result on the side-effect axis that a null-space-free scalar structurally cannot reach.
  k=1  : the single-direction arms (RepInd / k_A=1) must NOT achieve it (no null-space to exploit).
  SCOPED NULL: if allocation only ties scalar at matched effort/A-gain ⇒ report the dividing line
         "allocation pays iff residual coupling > ε" (Phase-2 drift/gain as predictor).

Per-layer allocation is closed-form for one equality hold + a norm bound (no fragile solver):
  s* = aₐ − (⟨aₐ,q⟩/⟨q,q⟩) q ,  q = Uₐ·g_B ,  u = m·Uₐᵀ(s*/‖s*‖).
The general QP (multi-hold + hard KL budget) drops in via casa_control.solve_qp later. Additive
actuator throughout (the robust Phase-0/1/2 mechanism); cones/maps loaded from Phase 2.

Run:  ../../.venv/bin/python mosaic_phase3.py --run       (~35-50 min)
      ../../.venv/bin/python mosaic_phase3.py --selftest   (synthetic alloc math + tiny e2e)
Outputs: outputs/mosaic_phase3_<model>.{json,txt,png}.
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

from utils import get_input_data, generate_completions                        # noqa: E402
from mosaic_phase0 import (  # noqa: E402
    SENT_MODEL, FORM_MODEL, FORMAL_EX, INFORMAL_EX, POS_EX, NEG_EX,
    HFScorer, fk_grade, distinct2, load_model, load_band_pscale, push_hooks,
)

OUT = _HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)

# (drive A, hold B) pairs; both are the Phase-1/2 passing attributes
PAIRS = [("formality", "reading"), ("reading", "formality")]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def load_cones(model):
    z = np.load(OUT / f"mosaic_cones_{model.config._name_or_path.split('/')[-1]}.npz")
    band = list(z["band"])
    C = {}
    for a in ("formality", "reading"):
        C[a] = {"U": z[f"{a}_pls_U"], "a": z[f"{a}_pls_a"]}  # (L,k,d), (L,k)
    return C, band


def gdir(C, a, li):
    """Activation-space unit direction that increases attribute a at band index li (= aₐ·Uₐ)."""
    return unit(C[a]["a"][li] @ C[a]["U"][li])


def method_push(C, A, B, band, m, method):
    """Per-layer additive push vector (d,) for the given method. Returns {l: np.array(d)}.
       scalar  : push along A's intensity gradient gₐ (best PLS scalar — drags B via coupling).
       alloc   : push IN A's k-cone, ⊥ B's readout (the MOSAIC null-space allocation).
       repind  : push gₐ orthogonalized to g_B in activation space (single decoupled direction, no cone)."""
    vec = {}
    for li, l in enumerate(band):
        UA = C[A]["U"][li]; aA = C[A]["a"][li]            # (k,d),(k,)
        gB = gdir(C, B, li)                                # (d,)
        if method == "scalar":
            uw = unit(aA @ UA)
        elif method == "alloc":
            q = UA @ gB                                    # (k,) B-direction inside A-cone
            s = aA - (aA @ q) / (q @ q + 1e-12) * q        # slope ⊥ q within the cone
            # collapse guard: if removing the B-component leaves ~no A-moving DoF (the k=1 case,
            # no null-space), the push is genuinely zero — do NOT let unit() amplify numerical noise.
            uw = unit(s @ UA) if np.linalg.norm(s) > 1e-6 * np.linalg.norm(aA) else np.zeros_like(gB)
        elif method == "repind":
            gA = unit(aA @ UA)
            uw = unit(gA - (gA @ gB) * gB)                 # single direction, orthogonalized to g_B
        else:
            raise ValueError(method)
        vec[l] = (m * uw).astype(np.float32)
    return vec


def gen_scores(model, tok, prompts, vec, band, device, sent, form, mnt, bs):
    hooks = [] if vec is None else push_hooks(model, band, {l: torch.from_numpy(v) for l, v in vec.items()}, device)
    resp = [c["response"] for c in generate_completions(model, prompts, tok, fwd_hooks=hooks,
                                                        batch_size=bs, max_new_tokens=mnt, temperature=0.0)]
    f = form.score(resp); r = np.array([fk_grade(t) for t in resp], float)
    d2 = np.array([distinct2(t) for t in resp], float)
    return {"formality": f, "reading": r, "distinct2": d2, "resp": resp}


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    C, cband = load_cones(model)
    assert cband == band, "cone band mismatch — re-run Phase 2"
    print(f"model={args.model} band={band[0]}..{band[-1]} pscale={pscale:.1f} k={C['formality']['U'].shape[1]}")

    _, test = get_input_data("harmless")
    n_eval = 4 if args.selftest else args.n_eval
    mnt = 24 if args.selftest else args.max_new_tokens
    fracs = [0.04, 0.1] if args.selftest else args.fracs
    bs = args.bs
    prompts = test[:n_eval]

    print("loading scorers ...")
    sent = HFScorer(SENT_MODEL, POS_EX, NEG_EX, device)
    form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)

    base = gen_scores(model, tok, prompts, None, band, device, sent, form, mnt, bs)
    baseM = {a: float(np.nanmean(base[a])) for a in ("formality", "reading")}
    print(f"baseline: formality={baseM['formality']:.3f} reading={baseM['reading']:.2f} "
          f"distinct2={np.nanmean(base['distinct2']):.2f}")

    results = {}
    for A, B in PAIRS:
        print(f"\n==== drive {A}  /  hold {B} ====")
        sweep = {meth: [] for meth in ("scalar", "alloc", "repind")}
        for meth in ("scalar", "alloc", "repind"):
            for frac in fracs:
                vec = method_push(C, A, B, band, frac * pscale, meth)
                sc = gen_scores(model, tok, prompts, vec, band, device, sent, form, mnt, bs)
                Ag = sc[A] - base[A]                 # per-prompt A-gain
                Bd = sc[B] - base[B]                 # per-prompt B-drift (hold target: 0)
                row = {"frac": frac, "A_gain": float(np.nanmean(Ag)),
                       "B_drift": float(np.nanmean(Bd)), "B_absdrift": float(np.nanmean(np.abs(Bd))),
                       "distinct2": float(np.nanmean(sc["distinct2"])),
                       "A_per": Ag.tolist(), "B_per": Bd.tolist()}
                sweep[meth].append(row)
                print(f"  {meth:6s} frac={frac:<5} A_gain={row['A_gain']:+.3f}  "
                      f"B_drift={row['B_drift']:+.2f} (|{row['B_absdrift']:.2f}|)  dist2={row['distinct2']:.2f}")
        results[f"{A}->{B}"] = {"A": A, "B": B, "baseA": baseM[A], "baseB": baseM[B], "sweep": sweep}

    verdict = decide(results, args)
    print("\n==== VERDICT ====")
    for l in verdict["lines"]:
        print(" ", l)

    out = {"meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale,
                    "k": int(C['formality']['U'].shape[1]), "n_eval": n_eval, "fracs": fracs,
                    "max_new_tokens": mnt, "coh_floor": args.coh_floor, "selftest": args.selftest},
           "baseline": baseM, "results": results, "verdict": verdict}
    name = args.model.split("/")[-1]; tag = "_selftest" if args.selftest else ""
    json.dump(out, open(OUT / f"mosaic_phase3_{name}{tag}.json", "w"), indent=2)
    (OUT / f"mosaic_phase3_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    if not args.selftest:
        try:
            plot(results, OUT / f"mosaic_phase3_{name}.png")
            print(f"plot -> outputs/mosaic_phase3_{name}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"saved outputs/mosaic_phase3_{name}{tag}.json")
    return out


def _coherent(rows, floor):
    return [r for r in rows if r["distinct2"] >= floor]


def _interp_bdrift(rows, target_gain):
    """|B-drift| of a method at a target A-gain, linear-interpolated over its coherent sweep
    (by A_gain). Returns (absdrift, frac_used) or (nan, None) if target unreachable coherently."""
    pts = sorted(_coherent(rows, -1), key=lambda r: r["A_gain"])  # already coherent-filtered upstream
    gs = [r["A_gain"] for r in pts]
    if not pts or target_gain < min(gs) or target_gain > max(gs):
        return float("nan"), None
    for i in range(1, len(pts)):
        if gs[i] >= target_gain:
            lo, hi = pts[i - 1], pts[i]
            w = (target_gain - lo["A_gain"]) / (hi["A_gain"] - lo["A_gain"] + 1e-9)
            return (1 - w) * lo["B_absdrift"] + w * hi["B_absdrift"], hi["frac"]
    return pts[-1]["B_absdrift"], pts[-1]["frac"]


# materiality threshold on the held attribute's drift (≈ Phase-1 baseline noise): below this, the
# scalar push barely perturbs B ⇒ there is nothing for allocation to fix (an EXPECTED tie, not a null).
COUPLING_EPS = {"reading": 1.0, "formality": 0.05, "sentiment": 0.05}


def decide(results, args):
    lines = []
    perdir = {}
    for key, R in results.items():
        A, B = R["A"], R["B"]
        coh = {m: _coherent(R["sweep"][m], args.coh_floor) for m in ("scalar", "alloc", "repind")}
        gmax = lambda m: max([r["A_gain"] for r in coh[m]], default=0.0)
        g_target = min(gmax("scalar"), gmax("alloc")) * args.match_frac
        d_sc, _ = _interp_bdrift(coh["scalar"], g_target)
        d_al, _ = _interp_bdrift(coh["alloc"], g_target)
        d_rp, _ = _interp_bdrift(coh["repind"], g_target)
        nearest = lambda m: (min(coh[m], key=lambda r: abs(r["A_gain"] - g_target)) if coh[m] else None)
        rs, ra, rr = nearest("scalar"), nearest("alloc"), nearest("repind")
        # paired >1 SE test (alloc vs scalar), same prompts, at matched A-gain operating points
        paired_win, md, se = False, float("nan"), float("nan")
        if rs and ra:
            ds = np.abs(np.array(rs["B_per"], float)); da = np.abs(np.array(ra["B_per"], float))
            diff = ds - da
            se = float(np.nanstd(diff) / np.sqrt(np.sum(np.isfinite(diff))))
            md = float(np.nanmean(diff))
            paired_win = md > args.win_se * se and md > 0
        # is this direction materially coupled? (scalar drags B above the noise floor)
        scalar_drift = abs(d_sc) if np.isfinite(d_sc) else (abs(rs["B_drift"]) if rs else 0.0)
        coupled = scalar_drift > COUPLING_EPS.get(B, 0.05)
        win = coupled and paired_win and np.isfinite(d_al) and d_al < d_sc
        # does the cone (k>1) matter? compare alloc vs repind (single decoupled direction)
        cone_helps = np.isfinite(d_al) and np.isfinite(d_rp) and d_al < d_rp - 1e-6
        perdir[key] = {"A": A, "B": B, "g_target": g_target, "scalar_drift": scalar_drift,
                       "coupled": bool(coupled), "d_scalar": d_sc, "d_alloc": d_al, "d_repind": d_rp,
                       "paired_diff": md, "paired_se": se, "paired_win": bool(paired_win),
                       "win": bool(win), "cone_helps": bool(cone_helps)}
        lines.append(f"[{A}->{B}] coupling(scalar |B-drift|@matched-A={g_target:+.2f})={scalar_drift:.2f} "
                     f"→ {'MATERIAL' if coupled else 'weak (expected tie)'}")
        lines.append(f"          |B-drift|: scalar={d_sc:.2f}  alloc={d_al:.2f}  repind/k1={d_rp:.2f}  "
                     f"| paired scalar−alloc={md:+.2f}±{se:.2f}SE → {'WIN' if win else ('tie' if not coupled else 'no-win')}; "
                     f"cone>single:{cone_helps}")
    wins = [k for k, p in perdir.items() if p["win"]]
    coupled_dirs = [k for k, p in perdir.items() if p["coupled"]]
    worse = [k for k, p in perdir.items() if p["coupled"] and np.isfinite(p["d_alloc"])
             and np.isfinite(p["d_scalar"]) and p["d_alloc"] > p["d_scalar"] + p["paired_se"]]
    if wins and not worse:
        v = (f"GO (MOSAIC WIN): on the materially-coupled direction(s) {wins}, constrained allocation "
             f"holds the off-target attribute with >{args.win_se:g} SE less drift than the best PLS-scalar "
             f"push at matched A-tracking — control law > bound on the side-effect axis. Ties on weakly-"
             f"coupled direction(s) are expected (nothing to allocate); consistent with the dividing line "
             f"'allocation pays iff coupling > ε'.")
    elif coupled_dirs and not wins:
        v = (f"SCOPED NULL (pre-registered): even on the coupled direction(s) {coupled_dirs}, allocation "
             f"only ties the best PLS-scalar — residual coupling after a good direction is too small to "
             f"exploit. Report the dividing line (allocation pays iff coupling > ε; Phase-2 drift/gain).")
    else:
        v = ("INCONCLUSIVE: no materially-coupled direction at coherent operating points (a good PLS "
             "direction already decoupled them) — report the coupling magnitudes; the hold-constraint "
             "does not bind here.")
    lines.insert(0, v)
    return {"verdict": v, "perdir": perdir, "wins": wins, "lines": lines}


def plot(results, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 5.5))
    if len(results) == 1:
        axes = [axes]
    for ax, (key, R) in zip(axes, results.items()):
        A, B = R["A"], R["B"]
        for meth, col, mk in (("scalar", "C3", "o"), ("alloc", "C0", "s"), ("repind", "C2", "^")):
            rows = sorted(R["sweep"][meth], key=lambda r: r["A_gain"])
            xs = [r["A_gain"] for r in rows]; ys = [r["B_absdrift"] for r in rows]
            d2 = [r["distinct2"] for r in rows]
            ax.plot(xs, ys, mk + "-", color=col, label=meth)
            for x, y, d in zip(xs, ys, d2):  # mark incoherent points hollow
                if d < 0.85:
                    ax.plot(x, y, "x", color=col, ms=9)
        ax.set_xlabel(f"A-gain ({A}) ↑ more steering"); ax.set_ylabel(f"|B-drift| ({B}) ↓ better hold")
        ax.set_title(f"drive {A} / hold {B}\n(lower = tighter hold at equal A-gain; ×=incoherent)")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("MOSAIC Phase 3 — allocation (hold B) vs best scalar push: side-effect Pareto frontier")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def _selftest():
    rng = np.random.RandomState(0)
    d, k = 48, 4
    UA = np.linalg.qr(rng.randn(d, k))[0].T            # (k,d) orthonormal rows
    aA = rng.randn(k)
    gB = UA[0] * 0.7 + UA[1] * 0.3; gB = gB / np.linalg.norm(gB)   # B-dir lies partly in A-cone
    # craft C["B"] so gdir(C,"B",0) == gB exactly (U_B = gB row, a_B = [1])
    C = {"A": {"U": UA[None], "a": aA[None]},
         "B": {"U": gB[None, None], "a": np.array([[1.0]])}}
    band = [0]
    # alloc push must be ⊥ gB; scalar must NOT be
    va = method_push(C, "A", "B", band, 1.0, "alloc")[0]
    vs = method_push(C, "A", "B", band, 1.0, "scalar")[0]
    assert abs(va @ gB) < 1e-5, f"alloc not ⊥ g_B ({va @ gB})"
    assert abs(vs @ gB) > 1e-2, "scalar should have a B-component (coupling)"
    # alloc push stays in the A-cone span
    P = UA.T @ UA                                       # projector onto span(UA)
    assert np.linalg.norm(P @ va - va) < 1e-5, "alloc push left the A-cone"
    # magnitude honored
    assert abs(np.linalg.norm(va) - 1.0) < 1e-5 and abs(np.linalg.norm(vs) - 1.0) < 1e-5
    # k_A=1: A-cone is 1-dim along aA; the in-cone ⊥gB push must collapse (no null-space)
    C1 = {"A": {"U": unit(aA @ UA)[None, None], "a": np.array([[1.0]])},
          "B": C["B"]}
    v1 = method_push(C1, "A", "B", band, 1.0, "alloc")[0]
    # in 1-dim cone, s = a - (a·q/q·q)q with scalar q ⇒ s=0 ⇒ push ~0
    assert np.linalg.norm(v1) < 1e-4, f"k=1 alloc should collapse (no null-space), got ‖v‖={np.linalg.norm(v1)}"
    print("[mosaic_phase3 self-test] OK — alloc ⊥ g_B & in-cone; scalar couples; ‖u‖=m; k=1 cone collapses")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-eval", type=int, default=40, dest="n_eval")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.02, 0.04, 0.06, 0.08, 0.1, 0.12])
    ap.add_argument("--max-new-tokens", type=int, default=96, dest="max_new_tokens")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--coh-floor", type=float, default=0.85, dest="coh_floor")
    ap.add_argument("--match-frac", type=float, default=0.8, dest="match_frac",
                    help="target A-gain = match_frac × min(max coherent A-gain of scalar, alloc)")
    ap.add_argument("--win-se", type=float, default=1.0, dest="win_se")
    args = ap.parse_args()
    if args.selftest and not args.run:
        _selftest(); run(args); return
    if not args.run:
        ap.error("pass --run or --selftest")
    run(args)


if __name__ == "__main__":
    main()
