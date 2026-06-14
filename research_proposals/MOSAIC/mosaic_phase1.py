"""MOSAIC Phase 1 — graded sensors + monotone-frontier verification (the precondition
the prior nulls lacked).  GO gate from MOSAIC_PROPOSAL.md §7:

  (a) PROBE: a linear probe a_t = w·h+b on the residual stream predicts the external
      attribute score at held-out Pearson r > 0.85  (a closed loop chasing a biased
      sensor manufactures fake wins — this is a HARD gate).
  (b) MONOTONE FRONTIER: sweeping the bounded additive push, the driven attribute's
      intensity rises MONOTONICALLY over the coarse-to-mid (coherent) range — a genuine
      strength<->coherence trade-off, NOT a refusal-style hump (lose on both axes).

Reuses Phase-0 building blocks (directions, push, scorers) + CALM's gen_coherence.
Run:  ../../.venv/bin/python mosaic_phase1.py --run     (~30-50 min, single GPU)
      ../../.venv/bin/python mosaic_phase1.py --selftest (tiny smoke)
Outputs: outputs/mosaic_phase1_<model>.{json,txt,png} + mosaic_probes_<model>.npz
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
sys.path.insert(0, str(_HERE.parents[1] / "CALM"))
sys.path.insert(0, str(_HERE.parent))

from utils import add_hooks, get_input_data, generate_completions             # noqa: E402
from calm_token import gen_coherence                                          # noqa: E402
from mosaic_phase0 import (  # noqa: E402
    ATTRS, ATTR_KEYS, FORMAL_EX, INFORMAL_EX, POS_EX, NEG_EX, SENT_MODEL, FORM_MODEL,
    HFScorer, fk_grade, distinct2, load_model, load_band_pscale, build_directions,
    push_hooks, wrap,
)
from sklearn.linear_model import RidgeCV  # noqa: E402

OUT = _HERE.parent / "outputs"
OUT.mkdir(exist_ok=True)

POOL_STYLES = [   # (label, directive) — span all 3 attribute axes for probe-pool diversity
    ("formal", ATTRS["formality"][0]), ("casual", ATTRS["formality"][1]),
    ("complex", ATTRS["reading"][0]), ("simple", ATTRS["reading"][1]),
    ("positive", ATTRS["sentiment"][0]), ("negative", ATTRS["sentiment"][1]),
    ("neutral", ""),
]


def ttr(text):
    w = text.split()
    return len(set(w)) / len(w) if w else float("nan")


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 2 else float("nan")


# ----------------------------------------------------------------------------- probe features
@torch.no_grad()
def extract_features(model, tok, texts, band, device, bs=16, maxlen=128):
    """Per-band-layer residual features for each text: mean-pooled over real tokens AND
    last-token. Forward the raw text (right-padded so default positions are correct)."""
    md = dict(model.named_modules())
    cache, pooled, lasts = {}, {l: [] for l in band}, {l: [] for l in band}

    def mk(l):
        def hook(mod, inp, out):
            cache[l] = out[0] if isinstance(out, tuple) else out
        return hook

    hooks = [(md[f"model.layers.{l}"], mk(l)) for l in band]
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        with add_hooks(module_forward_hooks=hooks):
            for i in range(0, len(texts), bs):
                enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                          truncation=True, max_length=maxlen)
                ids = enc.input_ids.to(device); attn = enc.attention_mask.to(device)
                model(input_ids=ids, attention_mask=attn)
                m = attn.unsqueeze(-1).float()
                last_idx = attn.sum(1) - 1  # right-pad: last real token index
                for l in band:
                    h = cache[l].float()
                    pooled[l].append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu())
                    lasts[l].append(h[torch.arange(h.shape[0]), last_idx].cpu())
    finally:
        tok.padding_side = old
    return ({l: torch.cat(pooled[l], 0).numpy() for l in band},
            {l: torch.cat(lasts[l], 0).numpy() for l in band})


def fit_probe(feats_by_layer, labels, band, seed=0, train_frac=0.7):
    """RidgeCV per band layer; held-out Pearson r. Returns (best_layer, best_r, per_layer_r,
    probe_dict). Standardizes features (mean/std stored for deployment)."""
    n = len(labels)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = idx[:ntr], idx[ntr:]
    y = np.asarray(labels, float)
    ok = np.isfinite(y)
    per_layer, best = {}, (None, -1.0, None)
    for l in band:
        X = feats_by_layer[l]
        mtr = tr[ok[tr]]; mte = te[ok[te]]
        mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-6
        Xs = (X - mu) / sd
        reg = RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0, 1e4])
        reg.fit(Xs[mtr], y[mtr])
        r = pearson(reg.predict(Xs[mte]), y[mte])
        per_layer[l] = r
        if r > best[1]:
            best = (l, r, {"w": reg.coef_, "b": float(reg.intercept_), "mu": mu, "sd": sd,
                           "alpha": float(reg.alpha_), "layer": l})
    return best[0], best[1], per_layer, best[2]


# ----------------------------------------------------------------------------- experiments
def build_probe_pool(model, tok, pool_prompts, max_new_tokens, bs):
    """Generate a style-diverse text pool (every prompt under every style directive)."""
    prompts = []
    for p in pool_prompts:
        for _, d in POOL_STYLES:
            prompts.append(f"{p}\n\n{d}" if d else p)
    comps = generate_completions(model, prompts, tok, batch_size=bs,
                                 max_new_tokens=max_new_tokens, temperature=0.0)
    return [c["response"] for c in comps]


def frontier(model, tok, dirs, eval_prompts, band, pscale, fracs, driven, sent, form,
             device, max_new_tokens, bs):
    """Sweep push magnitude for one driven attribute; measure intensity + coherence."""
    d_unit = dirs[driven]
    rows = []
    for frac in fracs:
        m = frac * pscale
        hooks = [] if frac == 0 else push_hooks(model, band, {l: m * d_unit[l] for l in band}, device)
        comps = generate_completions(model, eval_prompts, tok, fwd_hooks=hooks, batch_size=bs,
                                     max_new_tokens=max_new_tokens, temperature=0.0)
        resp = [c["response"] for c in comps]
        if driven == "reading":
            inten = float(np.nanmean([fk_grade(t) for t in resp]))
        elif driven == "formality":
            inten = float(np.nanmean(form.score(resp))) if form else float("nan")
        else:
            inten = float(np.nanmean(sent.score(resp))) if sent else float("nan")
        gnll = gen_coherence(model, tok, device, eval_prompts, resp, max_new_tokens)
        rows.append({"frac": frac, "intensity": inten, "genNLL": gnll,
                     "ttr": float(np.nanmean([ttr(t) for t in resp])),
                     "distinct2": float(np.nanmean([distinct2(t) for t in resp])),
                     "sample": resp[0][:160]})
    return rows


# ----------------------------------------------------------------------------- run
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_model(args.model, device)
    band, pscale = load_band_pscale(model)
    print(f"model={args.model} band={band[0]}..{band[-1]} pscale={pscale:.1f}")

    train, test = get_input_data("harmless")
    n_dir = 8 if args.selftest else args.n_dir
    n_pool = 5 if args.selftest else args.n_pool
    n_eval = 4 if args.selftest else args.n_eval
    fracs = [0.0, 0.1] if args.selftest else args.fracs
    mnt = 24 if args.selftest else args.max_new_tokens
    bs = args.bs
    base = train[:n_dir]
    pool_prompts = train[n_dir:n_dir + n_pool]
    eval_prompts = test[:n_eval]

    print("loading scorers ...")
    sent = HFScorer(SENT_MODEL, POS_EX, NEG_EX, device)
    form = HFScorer(FORM_MODEL, FORMAL_EX, INFORMAL_EX, device)

    # ---- (a) PROBE calibration -------------------------------------------------
    print(f"building probe pool ({n_pool}×{len(POOL_STYLES)} texts) ...")
    pool = build_probe_pool(model, tok, pool_prompts, max_new_tokens=80 if not args.selftest else 24, bs=bs)
    pool = [t if len(t.strip()) >= 8 else "n/a" for t in pool]
    labels = {"formality": form.score(pool), "reading": np.array([fk_grade(t) for t in pool], float),
              "sentiment": sent.score(pool)}
    print("extracting probe features ...")
    pooled_f, last_f = extract_features(model, tok, pool, band, device, bs=bs)
    probes, probe_r, probe_best_layer, per_layer_r = {}, {}, {}, {}
    saved = {}
    for a in ATTR_KEYS:
        bl, br, pl, pdict = fit_probe(pooled_f, labels[a], band)
        bl2, br2, _, _ = fit_probe(last_f, labels[a], band)  # last-token diagnostic
        probe_r[a] = br; probe_best_layer[a] = bl; per_layer_r[a] = pl
        probes[a] = pdict
        saved[f"{a}_w"] = pdict["w"]; saved[f"{a}_b"] = np.array([pdict["b"]])
        saved[f"{a}_mu"] = pdict["mu"]; saved[f"{a}_sd"] = pdict["sd"]
        saved[f"{a}_layer"] = np.array([bl])
        print(f"  probe[{a}]: best held-out r={br:.3f} @layer{bl} (mean-pool); "
              f"last-token r={br2:.3f} @layer{bl2}")
    np.savez(OUT / f"mosaic_probes_{args.model.split('/')[-1]}.npz", **saved)

    # ---- (b) MONOTONE FRONTIER -------------------------------------------------
    print("building directions ...")
    dirs, _ = build_directions(model, tok, base, device)
    frontiers = {}
    for driven in ATTR_KEYS:
        t0 = time.time()
        rows = frontier(model, tok, dirs, eval_prompts, band, pscale, fracs, driven,
                        sent, form, device, mnt, bs)
        frontiers[driven] = rows
        print(f"  frontier[{driven}] ({time.time()-t0:.0f}s):")
        for r in rows:
            print(f"    frac={r['frac']:<5} intensity={r['intensity']:7.3f}  "
                  f"genNLL={r['genNLL']:.3f}  ttr={r['ttr']:.2f}  dist2={r['distinct2']:.2f}")

    verdict = decide(probe_r, frontiers, args)
    print("\n==== VERDICT ====")
    for l in verdict["lines"]:
        print(" ", l)

    result = {
        "meta": {"model": args.model, "band": [band[0], band[-1]], "pscale": pscale,
                 "n_dir": n_dir, "n_pool_texts": len(pool), "n_eval": n_eval, "fracs": fracs,
                 "coh_floor": args.coh_floor, "max_new_tokens": mnt, "selftest": args.selftest},
        "probe": {"held_out_r": probe_r, "best_layer": probe_best_layer, "per_layer_r": per_layer_r},
        "frontiers": frontiers, "verdict": verdict,
    }
    name = args.model.split("/")[-1]
    tag = "_selftest" if args.selftest else ""
    json.dump(result, open(OUT / f"mosaic_phase1_{name}{tag}.json", "w"), indent=2)
    (OUT / f"mosaic_phase1_{name}{tag}.txt").write_text("\n".join(verdict["lines"]) + "\n")
    if not args.selftest:
        try:
            plot(frontiers, per_layer_r, probe_r, band, args.coh_floor, OUT / f"mosaic_phase1_{name}.png")
            print(f"plot -> outputs/mosaic_phase1_{name}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    print(f"saved outputs/mosaic_phase1_{name}{tag}.json")
    return result


def decide(probe_r, frontiers, args):
    lines = []
    # (a) probe gate
    probe_ok = {a: (probe_r[a] > args.r_gate) for a in ATTR_KEYS}
    lines.append("PROBE r (held-out, vs external scorer): "
                 + ", ".join(f"{a}={probe_r[a]:.3f}{'✓' if probe_ok[a] else '✗'}" for a in ATTR_KEYS)
                 + f"  (gate r>{args.r_gate})")
    # (b) monotonicity over the coherent range
    mono_ok = {}
    for a in ATTR_KEYS:
        rows = frontiers[a]
        coh = [r for r in rows if r["distinct2"] >= args.coh_floor]
        fr = [r["frac"] for r in coh]; inten = [r["intensity"] for r in coh]
        sp = spearman(fr, inten)
        # "no interior peak before collapse": intensity at the largest coherent frac is ~the max
        peak_at_end = (len(inten) >= 2 and (inten[-1] >= max(inten) - 1e-6 or
                       abs(inten[-1] - max(inten)) <= 0.1 * (max(inten) - min(inten) + 1e-9)))
        mono = (not np.isnan(sp)) and sp >= args.mono_gate and peak_at_end and len(coh) >= 3
        mono_ok[a] = mono
        # is it a hump? (intensity drops while still coherent)
        hump = len(inten) >= 3 and (max(inten) - inten[-1]) > 0.2 * (max(inten) - min(inten) + 1e-9) and inten.index(max(inten)) < len(inten) - 1
        lines.append(f"FRONTIER[{a}]: coherent fracs={fr} | intensity Spearman(frac)= {sp:.2f} "
                     f"| peak-at-strongest-coherent={peak_at_end} | hump={hump} -> monotone {'✓' if mono else '✗'}")
    probe_go = all(probe_ok.values())
    mono_go = all(mono_ok.values())
    go = probe_go and mono_go
    if go:
        v = "GO: probes calibrate (r>{:.2f}) AND the strength↔coherence dose-response is monotone (not a hump).".format(args.r_gate)
    else:
        miss = []
        if not probe_go:
            miss.append("probe(s) below r-gate: " + ",".join(a for a in ATTR_KEYS if not probe_ok[a]))
        if not mono_go:
            miss.append("non-monotone/hump attribute(s): " + ",".join(a for a in ATTR_KEYS if not mono_ok[a]))
        v = "PARTIAL/NO-GO — " + "; ".join(miss) + ". Report per-attribute; carry only the attributes that pass."
    lines.insert(0, v)
    return {"verdict": v, "go": bool(go), "probe_ok": {a: bool(probe_ok[a]) for a in ATTR_KEYS},
            "mono_ok": {a: bool(mono_ok[a]) for a in ATTR_KEYS}, "lines": lines}


def plot(frontiers, per_layer_r, probe_r, band, coh_floor, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, a in zip(axes.flat[:3], ATTR_KEYS):
        rows = frontiers[a]
        fr = [r["frac"] for r in rows]; inten = [r["intensity"] for r in rows]
        gnll = [r["genNLL"] for r in rows]; d2 = [r["distinct2"] for r in rows]
        ax.plot(fr, inten, "o-", color="C0", label="intensity (external)")
        ax.set_xlabel("push frac (·pscale)"); ax.set_ylabel(f"{a} intensity", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax2 = ax.twinx()
        ax2.plot(fr, gnll, "s--", color="C3", label="genNLL (incoherence↑)")
        ax2.plot(fr, d2, "^:", color="C2", label="distinct-2")
        ax2.axhline(coh_floor, color="C2", ls=":", alpha=0.4)
        ax2.set_ylabel("coherence (genNLL / distinct-2)")
        # shade the coherent region
        coh_fr = [r["frac"] for r in rows if r["distinct2"] >= coh_floor]
        if coh_fr:
            ax.axvspan(min(coh_fr), max(coh_fr), color="green", alpha=0.06)
        ax.set_title(f"{a}: dose-response (probe r={probe_r[a]:.2f})")
        l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, fontsize=7, loc="upper left")
    axp = axes.flat[3]
    for a in ATTR_KEYS:
        axp.plot(band, [per_layer_r[a][l] for l in band], "o-", label=f"{a} (best {probe_r[a]:.2f})")
    axp.axhline(0.85, color="k", ls="--", alpha=0.5, label="r=0.85 gate")
    axp.set_xlabel("layer"); axp.set_ylabel("held-out probe r"); axp.set_title("probe linear-readability by layer")
    axp.legend(fontsize=7)
    fig.suptitle("MOSAIC Phase 1 — graded sensors + monotone strength↔coherence frontier")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2-2b-it")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-dir", type=int, default=64, dest="n_dir")
    ap.add_argument("--n-pool", type=int, default=32, dest="n_pool")
    ap.add_argument("--n-eval", type=int, default=32, dest="n_eval")
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2])
    ap.add_argument("--max-new-tokens", type=int, default=128, dest="max_new_tokens")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--r-gate", type=float, default=0.85, dest="r_gate")
    ap.add_argument("--mono-gate", type=float, default=0.9, dest="mono_gate")
    ap.add_argument("--coh-floor", type=float, default=0.85, dest="coh_floor")
    args = ap.parse_args()
    if not (args.run or args.selftest):
        ap.error("pass --run or --selftest")
    run(args)


if __name__ == "__main__":
    main()
