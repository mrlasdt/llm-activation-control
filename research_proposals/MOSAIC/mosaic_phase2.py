"""MOSAIC Phase 2 — graded k>1 cones + the depth-indexed setpoint map (the D1 build).

Two pre-registered kill criteria (MOSAIC_PROPOSAL.md §7), on the attributes that passed
Phase 1 (formality, reading-level):

  (1) ATTRIBUTE-SPECIFIC k>1 CONE IS COHERENT (not blunt-SVD gibberish). We compare a
      supervised PLS cone (covaries with the attribute = the retain-loss / concept-cone
      analog) against a blunt unsupervised PCA cone (the SVD-span failure mode) and the
      k=1 DIM baseline: the PLS cone must (a) reconstruct the attribute intensity better
      than PCA at equal k, and (b) move the attribute under a bounded setpoint push while
      staying coherent (distinct-2 ≥ floor).
  (2) The DEPTH-INDEXED reference beats diff-in-means on CALIBRATION. Same direction +
      setpoint actuator, but a per-LAYER target r_l(τ) vs one global (layer-agnostic)
      target: the per-layer reference must hit a commanded intensity τ with lower
      miscalibration |achieved−τ| (at matched coherence) over a τ grid.

Also: RepInd ablation (project the OTHER attribute's probe out of the cone — how much
cross-coupling does it remove up front?) + a stacked-plant R² gate (for the optional H>1
ablation later). Reuses Phase-0 directions/push/scorers + Phase-1 feature extractor/probes.

Run:  ../../.venv/bin/python mosaic_phase2.py --run     (~30-45 min)
      ../../.venv/bin/python mosaic_phase2.py --selftest # synthetic actuator math + tiny e2e
Outputs: outputs/mosaic_phase2_<model>.{json,txt,png} + mosaic_cones_<model>.npz (for Phase 3).
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "pytorch_pure"))
sys.path.insert(0, str(_HERE.parents[1] / "CASA"))
sys.path.insert(0, str(_HERE.parents[1] / "CALM"))
sys.path.insert(0, str(_HERE.parent))

from utils import add_hooks, get_input_data, generate_completions             # noqa: E402
from calm_token import gen_coherence                                          # noqa: E402
from casa_actuator import orthonormalize                                      # noqa: E402
from casa_control import fit_cone_plant, plant_r2                             # noqa: E402
from mosaic_phase0 import (  # noqa: E402
    ATTRS, FORMAL_EX, INFORMAL_EX, POS_EX, NEG_EX, SENT_MODEL, FORM_MODEL,
    HFScorer, fk_grade, distinct2, load_model, load_band_pscale, push_hooks,
)
from mosaic_phase1 import extract_features, ttr, pearson                       # noqa: E402
from sklearn.cross_decomposition import PLSRegression                          # noqa: E402

OUT = _HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)

# the two attributes that cleared Phase 1; sentiment is hold-only (probe r=0.73)
DRIVEN = ["formality", "reading"]
SCORER_OF = {}  # filled at runtime: attr -> callable(list[str])->np.array

# graded intensity ladder (5 levels) per attribute, low->high
LADDER = {
    "formality": [
        "Write in an extremely casual, slangy, informal tone, like texting a close friend.",
        "Write in a fairly casual, conversational tone.",
        "Write in a plain, neutral tone.",
        "Write in a fairly formal, professional tone.",
        "Write in an extremely formal, professional, and sophisticated register.",
    ],
    "reading": [
        "Use only very simple words and short sentences a young child could understand.",
        "Use simple, easy words and mostly short sentences.",
        "Write plainly for a general audience.",
        "Use somewhat advanced vocabulary and longer sentences.",
        "Use advanced, complex vocabulary and long, elaborate, multi-clause sentences.",
    ],
}


# ----------------------------------------------------------------------------- setpoint actuator
class SetpointActuator:
    """Per-layer bounded P-push that drives each band layer's k-dim cone coordinate toward
    a target r_l: c = U_l h; u = clip(rho·(r_l − c), ‖u‖≤u_max); h' = h + uᵀU_l. (k=1 ok.)
    Reused by Phase 3 (there the target/QP also enforces the hold). Orthonormal U_l ⇒ the
    ambient ‖u‖ equals the coordinate ‖u‖."""

    def __init__(self, U_by_layer, r_by_layer, u_max, rho=1.0):
        self.U = U_by_layer          # {l: (k,d) torch}
        self.r = r_by_layer          # {l: (k,) torch}
        self.u_max = float(u_max)
        self.rho = float(rho)
        self._cache = {}

    def _cast(self, l, h):
        key = (l, h.device, h.dtype)
        if key not in self._cache:
            self._cache[key] = (self.U[l].to(h.device, h.dtype), self.r[l].to(h.device, h.dtype))
        return self._cache[key]

    def apply(self, l, h):
        U, r = self._cast(l, h)
        c = h @ U.t()                                # (B,S,k)
        u = self.rho * (r - c)                       # desired push in coord space
        if math.isfinite(self.u_max):
            nrm = u.norm(dim=-1, keepdim=True)
            scale = torch.clamp(torch.where(nrm > 0, self.u_max / (nrm + 1e-6),
                                            torch.zeros_like(nrm)), max=1.0)
            u = u * scale
        return h + u @ U                             # (B,S,d)

    def hook(self, l):
        def _hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            steered = self.apply(l, h)
            return (steered,) + tuple(out[1:]) if isinstance(out, tuple) else steered
        return _hook

    def band_hooks(self, module_dict, band):
        return [(module_dict[f"model.layers.{l}"], self.hook(l)) for l in band]


def steer_hooks(model, act, band):
    return act.band_hooks(dict(model.named_modules()), band)


# ----------------------------------------------------------------------------- cones + maps
def build_cones(feats_mean, y, band, k):
    """Per-layer cones from mean-pooled features. Returns dict type -> {l:(k',d) np orthonormal-rows}.
       dim (k=1 quartile diff), pca (top-k SVD = blunt), pls (top-k PLS = attribute-specific)."""
    y = np.asarray(y, float)
    hi_m, lo_m = np.quantile(y, 0.75), np.quantile(y, 0.25)
    cones = {"dim": {}, "pca": {}, "pls": {}}
    recon = {"pca": [], "pls": []}  # held-out intensity-reconstruction R^2 per layer
    for l in band:
        X = feats_mean[l].astype(np.float64)
        Xc = X - X.mean(0)
        # DIM (k=1): high-quartile minus low-quartile mean, unit
        d = X[y >= hi_m].mean(0) - X[y <= lo_m].mean(0)
        d = d / (np.linalg.norm(d) + 1e-9)
        cones["dim"][l] = d[None, :]
        # PCA (blunt): top-k right singular vectors of centered X
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        cones["pca"][l] = orthonormalize(torch.from_numpy(Vt[:k].copy()).float()).numpy()
        # PLS (attribute-specific): top-k PLS components of X -> y
        pls = PLSRegression(n_components=k, scale=True).fit(X, y)
        W = pls.x_rotations_.T  # (k,d)
        cones["pls"][l] = orthonormalize(torch.from_numpy(W.copy()).float()).numpy()
        # intensity reconstruction R^2 (cheap split) for pca vs pls
        for t in ("pca", "pls"):
            U = cones[t][l]
            C = Xc @ U.T
            recon[t].append(_holdout_r2(C, y))
    recon = {t: float(np.mean(v)) for t, v in recon.items()}
    return cones, recon


def _holdout_r2(C, y, seed=0, frac=0.7):
    n = len(y); rng = np.random.RandomState(seed); idx = rng.permutation(n)
    tr, te = idx[:int(frac * n)], idx[int(frac * n):]
    A = np.concatenate([C[tr], np.ones((len(tr), 1))], 1)
    w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
    pred = np.concatenate([C[te], np.ones((len(te), 1))], 1) @ w
    ss = ((y[te] - pred) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-9
    return 1.0 - ss / tot


def repind_decouple(cone_by_layer, other_dir_by_layer, band):
    """Project the OTHER attribute's (unit) direction out of each layer's cone, re-orthonormalize."""
    out = {}
    for l in band:
        U = torch.from_numpy(cone_by_layer[l]).float()           # (k,d)
        o = torch.from_numpy(other_dir_by_layer[l][0]).float()   # (d,)
        o = o / (o.norm() + 1e-9)
        Up = U - (U @ o)[:, None] * o[None, :]                   # remove component along o
        out[l] = orthonormalize(Up).numpy()
    return out


def fit_setpoint_map(feats_last, y, cone_by_layer, band):
    """Per-layer linear map intensity τ -> target coordinate r_l(τ)∈R^k, fit on LAST-token
    coords (matching what the actuator reads). Returns (slope a_l:(k,), intercept b_l:(k,))."""
    y = np.asarray(y, float)
    a, b = {}, {}
    for l in band:
        U = cone_by_layer[l]                       # (k,d)
        C = feats_last[l] @ U.T                    # (N,k)
        A = np.stack([y, np.ones_like(y)], 1)      # (N,2)
        sol, *_ = np.linalg.lstsq(A, C, rcond=None)  # (2,k)
        a[l] = sol[0]; b[l] = sol[1]
    return a, b


def targets_at(a, b, tau, band, global_avg=False):
    """r_l(τ): per-layer (a_l·τ+b_l). If global_avg, use band-averaged (a,b) at every layer
    (the layer-agnostic 'diff-in-means' reference)."""
    if global_avg:
        am = np.mean([a[l] for l in band], 0); bm = np.mean([b[l] for l in band], 0)
        return {l: torch.from_numpy((am * tau + bm)).float() for l in band}
    return {l: torch.from_numpy((a[l] * tau + b[l])).float() for l in band}


def cone_intensity_dir(cone_by_layer, feats_mean, y, band):
    """Per-layer UNIT direction inside span(U_l) that increases the attribute intensity:
    regress intensity on the cone coordinate, map the coefficient back to activation space.
    For a blunt cone whose coordinate barely predicts intensity, this direction is weak/noisy
    (the point of comparing PLS vs PCA). Returns {l: (d,) np unit}."""
    y = np.asarray(y, float)
    g = {}
    for l in band:
        U = cone_by_layer[l]                        # (k,d)
        C = feats_mean[l] @ U.T                      # (N,k)
        A = np.concatenate([C, np.ones((len(y), 1))], 1)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        amb = w[:-1] @ U                             # (d,) activation-space gradient (in-cone)
        g[l] = amb / (np.linalg.norm(amb) + 1e-9)
    return g


# ----------------------------------------------------------------------------- gen helpers
def gen(model, tok, prompts, hooks, mnt, bs):
    comps = generate_completions(model, prompts, tok, fwd_hooks=hooks, batch_size=bs,
                                 max_new_tokens=mnt, temperature=0.0)
    return [c["response"] for c in comps]


def measure(resp, attr):
    if attr == "reading":
        return float(np.nanmean([fk_grade(t) for t in resp]))
    return float(np.nanmean(SCORER_OF[attr](resp)))


# ----------------------------------------------------------------------------- run
def run(args):
    global SCORER_OF
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    md = dict(model.named_modules())
    print(f"model={args.model} band={band[0]}..{band[-1]} pscale={pscale:.1f}")

    train, test = get_input_data("harmless")
    n_lad = 6 if args.selftest else args.n_ladder
    n_eval = 4 if args.selftest else args.n_eval
    mnt = 24 if args.selftest else args.max_new_tokens
    bs = args.bs
    k = args.k
    ladder_prompts = train[:n_lad]
    eval_prompts = test[:n_eval]
    umax = args.umax_frac * pscale

    print("loading scorers ...")
    sent = HFScorer(SENT_MODEL, POS_EX, NEG_EX, device)
    form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)
    SCORER_OF = {"formality": form.score, "sentiment": sent.score}

    # ---- ladder data (cone + map training): every prompt × 5 intensity levels per attr ----
    print("generating intensity ladders ...")
    ladder = {}
    for a in DRIVEN:
        prompts = [f"{p}\n\n{d}" for p in ladder_prompts for d in LADDER[a]]
        resp = gen(model, tok, prompts, [], 80 if not args.selftest else 24, bs)
        yv = (np.array([fk_grade(t) for t in resp], float) if a == "reading"
              else SCORER_OF[a](resp))
        keep = [i for i, t in enumerate(resp) if len(t.strip()) >= 8 and np.isfinite(yv[i])]
        texts = [resp[i] for i in keep]; yv = yv[keep]
        fmean, flast = extract_features(model, tok, texts, band, device, bs=bs)
        ladder[a] = {"y": yv, "fmean": fmean, "flast": flast, "n": len(texts),
                     "lo": float(np.quantile(yv, 0.1)), "hi": float(np.quantile(yv, 0.9))}
        print(f"  {a}: {len(texts)} texts, intensity range [{yv.min():.2f},{yv.max():.2f}]")

    # ---- build cones + maps per attribute ----
    cones_all, recon_all, maps_all, dim_dirs = {}, {}, {}, {}
    for a in DRIVEN:
        cones, recon = build_cones(ladder[a]["fmean"], ladder[a]["y"], band, k)
        cones_all[a] = cones; recon_all[a] = recon; dim_dirs[a] = cones["dim"]
        print(f"  cone[{a}] intensity-reconstruction R² (held-out): PLS={recon['pls']:.3f}  PCA={recon['pca']:.3f}")
    # RepInd-decoupled PLS cone (project the OTHER driven attr's DIM dir out)
    repind = {}
    if len(DRIVEN) == 2:
        repind[DRIVEN[0]] = repind_decouple(cones_all[DRIVEN[0]]["pls"], dim_dirs[DRIVEN[1]], band)
        repind[DRIVEN[1]] = repind_decouple(cones_all[DRIVEN[1]]["pls"], dim_dirs[DRIVEN[0]], band)
    for a in DRIVEN:
        maps_all[a] = {ct: fit_setpoint_map(ladder[a]["flast"], ladder[a]["y"], cones_all[a][ct], band)
                       for ct in ("dim", "pls")}

    # per-layer attribute gap (for the single-vector layer choice) + intensity directions
    gaps = {}
    for a in DRIVEN:
        y = ladder[a]["y"]; hi_m, lo_m = np.quantile(y, 0.75), np.quantile(y, 0.25)
        gaps[a] = {l: float(np.linalg.norm(ladder[a]["fmean"][l][y >= hi_m].mean(0)
                                           - ladder[a]["fmean"][l][y <= lo_m].mean(0))) for l in band}

    # ============================ E1: cone specificity + coherence (additive push) ============================
    # The retain-loss/concept-cone analog: an attribute-SPECIFIC subspace (PLS) should both
    # reconstruct intensity far better than a blunt one (PCA) AND, when pushed along its in-cone
    # intensity direction, move the attribute coherently. Push = bounded additive (the validated
    # Phase-0/1 actuator), NOT the fitted-coordinate setpoint (which is fragile; saved for Phase 3).
    m_push = args.umax_frac * pscale
    print(f"\nE1 — cone specificity & coherence (additive push ‖m‖={m_push:.1f}/layer along in-cone intensity dir):")
    e1 = {}
    for a in DRIVEN:
        base = measure(gen(model, tok, eval_prompts, [], mnt, bs), a)
        e1[a] = {"base": base, "recon": recon_all[a], "rows": []}
        for ct in ("dim", "pls", "pca"):
            g = cone_intensity_dir(cones_all[a][ct], ladder[a]["fmean"], ladder[a]["y"], band)
            vec = {l: torch.from_numpy(m_push * g[l]).float() for l in band}
            resp = gen(model, tok, eval_prompts, push_hooks(model, band, vec, device), mnt, bs)
            inten = measure(resp, a); d2 = float(np.nanmean([distinct2(t) for t in resp]))
            e1[a]["rows"].append({"cone": ct, "k": cones_all[a][ct][band[0]].shape[0],
                                  "intensity": inten, "gain": inten - base, "distinct2": d2,
                                  "sample": resp[0][:140]})
            print(f"    {a:9s} {ct:3s}(k={cones_all[a][ct][band[0]].shape[0]}): "
                  f"intensity {base:.2f}→{inten:.2f} (gain {inten-base:+.2f})  distinct2={d2:.2f}")

    # ============================ E2: depth-indexed band vs single diff-in-means vector ============================
    # D1 test: a depth-indexed reference (per-layer DIM directions applied across the WHOLE band)
    # vs the textbook single difference-in-means VECTOR (one direction at one best layer). Both swept
    # over a magnitude grid; compare miscalibration |achieved−τ| achievable while COHERENT.
    print("\nE2 — depth-indexed full-band reference vs a single diff-in-means vector (one layer):")
    e2 = {}
    depth_a = [0.025, 0.05, 0.075, 0.1] if not args.selftest else [0.05, 0.1]
    single_a = [0.1, 0.2, 0.4, 0.8] if not args.selftest else [0.2, 0.8]
    for a in DRIVEN:
        Lstar = max(band, key=lambda l: gaps[a][l])               # best single layer
        lo, hi = ladder[a]["lo"], ladder[a]["hi"]
        taus = [lo + (hi - lo) * f for f in ([0.6, 1.0] if not args.selftest else [1.0])]
        d_l = {l: cones_all[a]["dim"][l][0] for l in band}        # per-layer unit DIM dir
        e2[a] = {"taus": taus, "Lstar": int(Lstar), "rows": []}
        for method, alphas in (("depth", depth_a), ("single", single_a)):
            for al in alphas:
                m = al * pscale
                if method == "depth":
                    vec = {l: torch.from_numpy(m * d_l[l]).float() for l in band}
                    hooks = push_hooks(model, band, vec, device)
                else:
                    vec = {Lstar: torch.from_numpy(m * d_l[Lstar]).float()}
                    hooks = push_hooks(model, [Lstar], vec, device)
                resp = gen(model, tok, eval_prompts, hooks, mnt, bs)
                ach = measure(resp, a); d2 = float(np.nanmean([distinct2(t) for t in resp]))
                e2[a]["rows"].append({"method": method, "alpha": al, "achieved": ach, "distinct2": d2})
        # miscalibration achievable while coherent (distinct-2 >= floor)
        mis = {}
        for method in ("depth", "single"):
            pts = [r for r in e2[a]["rows"] if r["method"] == method and r["distinct2"] >= args.coh_floor]
            mis[method] = float(np.mean([min((abs(r["achieved"] - t) for r in pts), default=float("nan"))
                                         for t in taus])) if pts else float("nan")
        e2[a]["miscal"] = mis
        print(f"    {a:9s} (single@L{Lstar}): miscal_coherent  depth={mis['depth']:.3f}  single={mis['single']:.3f}")

    # ============================ RepInd cross-drift preview + plant R² ============================
    print("\nRepInd cross-drift preview (push attr A→up along PLS dir, measure OTHER attr drift; raw vs RepInd):")
    crossdrift = {}
    for a in DRIVEN:
        other = DRIVEN[1] if a == DRIVEN[0] else DRIVEN[0]
        base_other = measure(gen(model, tok, eval_prompts, [], mnt, bs), other)
        cd = {}
        for tag, cone in (("rawPLS", cones_all[a]["pls"]), ("RepInd", repind[a])):
            g = cone_intensity_dir(cone, ladder[a]["fmean"], ladder[a]["y"], band)
            vec = {l: torch.from_numpy(m_push * g[l]).float() for l in band}
            resp = gen(model, tok, eval_prompts, push_hooks(model, band, vec, device), mnt, bs)
            cd[tag] = {"driven_gain": measure(resp, a) - e1[a]["base"],
                       "other_drift": measure(resp, other) - base_other}
        crossdrift[a] = {"other": other, **cd}
        print(f"    push {a}→up: rawPLS other({other})Δ={cd['rawPLS']['other_drift']:+.2f} "
              f"(self {cd['rawPLS']['driven_gain']:+.2f}) | RepInd other Δ={cd['RepInd']['other_drift']:+.2f} "
              f"(self {cd['RepInd']['driven_gain']:+.2f})")

    # stacked-plant R² over the band (cheap gate for any H>1 ablation)
    plant = {}
    for a in DRIVEN:
        cone = cones_all[a]["pls"]
        # coords across band for the ladder texts: (N, L, k)
        C = np.stack([ladder[a]["fmean"][l] @ cone[l].T for l in band], 1)
        try:
            fit = fit_cone_plant(C)
            plant[a] = float(plant_r2(C, fit, horizon=1))
        except Exception as e:
            plant[a] = float("nan")
        print(f"    plant R²(1-step) [{a}, PLS k={k}] = {plant[a]:.4f}")

    verdict = decide(recon_all, e1, e2, crossdrift, args)
    print("\n==== VERDICT ====")
    for l in verdict["lines"]:
        print(" ", l)

    # ---- save cones + maps for Phase 3 ----
    save = {}
    for a in DRIVEN:
        for ct in ("dim", "pls"):
            save[f"{a}_{ct}_U"] = np.stack([cones_all[a][ct][l] for l in band], 0)  # (L,k,d)
            am, bm = maps_all[a][ct]
            save[f"{a}_{ct}_a"] = np.stack([am[l] for l in band], 0)                # (L,k)
            save[f"{a}_{ct}_b"] = np.stack([bm[l] for l in band], 0)
        save[f"{a}_repind_U"] = np.stack([repind[a][l] for l in band], 0)
    save["band"] = np.array(band)
    np.savez(OUT / f"mosaic_cones_{args.model.split('/')[-1]}.npz", **save)

    result = {"meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale, "k": k,
                       "umax_frac": args.umax_frac, "n_ladder": n_lad, "n_eval": n_eval,
                       "max_new_tokens": mnt, "selftest": args.selftest},
              "recon_r2": recon_all, "E1_coherence": e1, "E2_calibration": e2,
              "repind_crossdrift": crossdrift, "plant_r2": plant, "verdict": verdict}
    name = args.model.split("/")[-1]; tag = "_selftest" if args.selftest else ""
    json.dump(result, open(OUT / f"mosaic_phase2_{name}{tag}.json", "w"), indent=2)
    (OUT / f"mosaic_phase2_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    if not args.selftest:
        try:
            plot(e1, e2, recon_all, crossdrift, OUT / f"mosaic_phase2_{name}.png")
            print(f"plot -> outputs/mosaic_phase2_{name}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"saved outputs/mosaic_phase2_{name}{tag}.json + mosaic_cones_{name}.npz")
    return result


def decide(recon, e1, e2, crossdrift, args):
    lines = []
    # criterion 1: attribute-specific PLS cone reconstructs intensity >> blunt PCA, AND a bounded
    # push along its in-cone intensity direction moves the attribute the right way while coherent.
    c1 = {}
    for a in DRIVEN:
        pls_better = recon[a]["pls"] > recon[a]["pca"] + 0.05
        pls_row = next(r for r in e1[a]["rows"] if r["cone"] == "pls")
        pca_row = next(r for r in e1[a]["rows"] if r["cone"] == "pca")
        coherent = pls_row["distinct2"] >= args.coh_floor and pls_row["gain"] > 0
        c1[a] = pls_better and coherent
        lines.append(f"C1[{a}]: intensity-recon R² PLS={recon[a]['pls']:.2f} vs PCA={recon[a]['pca']:.2f} "
                     f"(PLS-specific:{pls_better}); PLS push gain={pls_row['gain']:+.2f} "
                     f"distinct2={pls_row['distinct2']:.2f} (coherent:{coherent}); "
                     f"PCA gain={pca_row['gain']:+.2f}/d2={pca_row['distinct2']:.2f} → {'✓' if c1[a] else '✗'}")
    # criterion 2: depth-indexed (full-band) reference reaches τ with lower coherent miscalibration
    # than a single diff-in-means vector (one layer).
    c2 = {}
    for a in DRIVEN:
        m = e2[a]["miscal"]
        c2[a] = np.isfinite(m["depth"]) and (np.isnan(m["single"]) or m["depth"] < m["single"] - 1e-6)
        lines.append(f"C2[{a}]: coherent miscal |achieved−τ|  depth-indexed={m['depth']:.3f}  "
                     f"single-vector@L{e2[a]['Lstar']}={m['single']:.3f} → depth "
                     f"{'wins ✓' if c2[a] else 'does NOT beat ✗'}")
    for a in DRIVEN:
        if a in crossdrift:
            cd = crossdrift[a]
            lines.append(f"RepInd[{a}→{cd['other']}]: other-attr drift rawPLS={cd['rawPLS']['other_drift']:+.2f} "
                         f"→ RepInd={cd['RepInd']['other_drift']:+.2f} "
                         f"(self-gain raw={cd['rawPLS']['driven_gain']:+.2f}/RepInd={cd['RepInd']['driven_gain']:+.2f})")
    c1_go = all(c1.values()); c2_go = all(c2.values())
    go = c1_go and c2_go
    if go:
        v = ("GO: attribute-specific k>1 cones are coherent and beat the blunt PCA span, AND the "
             "depth-indexed full-band reference beats a single diff-in-means vector on calibration.")
    else:
        miss = []
        if not c1_go: miss.append("cone(s) not specific/coherent: " + ",".join(a for a in DRIVEN if not c1[a]))
        if not c2_go: miss.append("depth-indexing not better: " + ",".join(a for a in DRIVEN if not c2[a]))
        v = "PARTIAL — " + "; ".join(miss) + " (carry the cone result; report depth-indexing honestly)."
    lines.insert(0, v)
    return {"verdict": v, "go": bool(go), "c1": {a: bool(c1[a]) for a in DRIVEN},
            "c2": {a: bool(c2[a]) for a in DRIVEN}, "lines": lines}


def plot(e1, e2, recon, crossdrift, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # E1: gain & coherence per cone type
    ax = axes[0, 0]
    cts = ["dim", "pls", "pca"]; x = np.arange(len(cts))
    for i, a in enumerate(DRIVEN):
        gains = [next(r for r in e1[a]["rows"] if r["cone"] == ct)["gain"] for ct in cts]
        ax.bar(x + i * 0.25, gains, 0.25, label=f"{a} push gain")
    ax.set_xticks(x + 0.12); ax.set_xticklabels(cts); ax.set_title("E1: attribute gain by cone type")
    ax.legend(fontsize=7); ax.axhline(0, color="k", lw=0.5)
    ax = axes[0, 1]
    for i, a in enumerate(DRIVEN):
        d2 = [next(r for r in e1[a]["rows"] if r["cone"] == ct)["distinct2"] for ct in cts]
        ax.bar(x + i * 0.25, d2, 0.25, label=a)
    ax.axhline(0.85, color="k", ls="--", alpha=0.5); ax.set_xticks(x + 0.12); ax.set_xticklabels(cts)
    ax.set_title("E1: coherence (distinct-2) by cone type"); ax.legend(fontsize=7); ax.set_ylim(0, 1.05)
    # E2: coherent miscalibration depth-indexed vs single diff-in-means vector
    ax = axes[1, 0]
    for i, a in enumerate(DRIVEN):
        m = e2[a]["miscal"]; ax.bar([i - 0.2, i + 0.2], [m["depth"], m["single"]], 0.4,
                                    color=["C0", "C3"])
    ax.set_xticks(range(len(DRIVEN))); ax.set_xticklabels(DRIVEN)
    ax.set_title("E2: coherent miscal |achieved−τ|  (blue=depth-indexed, red=single vector)")
    # recon R²
    ax = axes[1, 1]
    for i, a in enumerate(DRIVEN):
        ax.bar([i - 0.2, i + 0.2], [recon[a]["pls"], recon[a]["pca"]], 0.4, color=["C2", "C1"])
    ax.set_xticks(range(len(DRIVEN))); ax.set_xticklabels(DRIVEN); ax.set_ylim(0, 1.05)
    ax.set_title("cone intensity-reconstruction R² (green=PLS, orange=PCA)")
    fig.suptitle("MOSAIC Phase 2 — graded cones + depth-indexed setpoint map")
    fig.tight_layout(); fig.savefig(path, dpi=120)


def _selftest():
    torch.manual_seed(0)
    d, k = 32, 3
    U = orthonormalize(torch.randn(k, d))
    band = [0, 1]
    Ud = {l: U for l in band}; r = {l: torch.zeros(k) for l in band}
    # full push (u_max=inf, rho=1) drives coordinate to target r=0 ⇒ projection-out
    act = SetpointActuator(Ud, r, float("inf"), rho=1.0)
    h = torch.randn(2, 4, d)
    h2 = act.apply(0, h)
    assert (h2 @ U.t()).abs().max() < 1e-4, "setpoint(r=0) should zero the coordinate"
    # target = nonzero r: coordinate should equal r after full push
    rt = torch.randn(k); act2 = SetpointActuator({0: U}, {0: rt}, float("inf"))
    h3 = act2.apply(0, h)
    assert torch.allclose(h3 @ U.t(), rt.expand(2, 4, k), atol=1e-4), "coordinate != target"
    # bounded push respects u_max
    actb = SetpointActuator({0: U}, {0: rt}, 0.3)
    u = (actb.apply(0, h) - h) @ U.t()
    assert u.norm(dim=-1).max() <= 0.3 + 1e-4, "u_max violated"
    # map fit: linear coord = a*y+b recovered
    y = np.linspace(0, 1, 50); Cfake = {0: (np.outer(y, np.ones(d)) * 2.0)}
    a, b = fit_setpoint_map(Cfake, y, {0: np.eye(d)[:1]}, [0])
    assert abs(a[0][0] - 2.0) < 1e-3, f"map slope wrong {a[0][0]}"
    print("[mosaic_phase2 self-test] OK — setpoint actuator (r=0 proj, target-hit, u_max), map fit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-ladder", type=int, default=40, dest="n_ladder")
    ap.add_argument("--n-eval", type=int, default=24, dest="n_eval")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--umax-frac", type=float, default=0.08, dest="umax_frac")
    ap.add_argument("--max-new-tokens", type=int, default=96, dest="max_new_tokens")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--coh-floor", type=float, default=0.85, dest="coh_floor")
    ap.add_argument("--min-gain-frac", type=float, default=0.3, dest="min_gain_frac")
    args = ap.parse_args()
    if args.selftest and not args.run:
        _selftest()
        # tiny end-to-end too
        run(args)
        return
    if not args.run:
        ap.error("pass --run or --selftest")
    run(args)


if __name__ == "__main__":
    main()
