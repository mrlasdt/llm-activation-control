"""CASA figures — the behaviour/coherence Pareto that decides lever L2.

Reads the structured results written by `casa_experiment.py`
(`outputs/casa_cone_<model>.json`) and draws:

  (left)  de-refusal margin (↓ better) vs neutral-corpus coherence tax (→ worse),
          one point per subspace-construction method. The headline: the CONE
          points sit in the low-margin / low-tax corner where the blunt SVD
          subspace cannot reach (high tax for any de-refusal).
  (right) ASR (↑ better) vs coherence tax (→ worse) — same story in behaviour space.

Run:  python casa_plot.py [outputs/casa_cone_gemma-2-2b-it.json]
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve()


def _style(label):
    if label.startswith("CONE") and "MPC" in label:
        return dict(color="tab:purple", marker="D")
    if label.startswith("CONE"):
        return dict(color="tab:green", marker="*")
    if label.startswith("RDO"):
        return dict(color="tab:blue", marker="o")
    if label.startswith("SVD"):
        return dict(color="tab:red", marker="s")
    if label.startswith("DIM"):
        return dict(color="tab:orange", marker="^")
    return dict(color="gray", marker=".")


def main():
    jpath = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
             else _HERE.parent / "outputs" / "casa_cone_gemma-2-2b-it.json")
    data = json.loads(pathlib.Path(jpath).read_text())
    meta, rows = data["meta"], data["rows"]
    # one representative point per label: full ablation (umax None) where present
    by_label = {}
    for r in rows:
        key = r["label"]
        # prefer the full-ablation row (umax None) as the representative
        if key not in by_label or (r["umax"] is None and by_label[key]["umax"] is not None):
            by_label[key] = r

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    base_m = meta["base_margin"]
    ax1.axhline(base_m, ls=":", c="k", lw=1, label=f"baseline margin {base_m:+.1f}")
    ax1.axhline(0, ls="--", c="gray", lw=0.8)
    # right-panel y-axis: prefer the StrongREJECT score (honest ASR) if present
    use_sr = any(r.get("sr_score") is not None for r in by_label.values())
    yk = "sr_score" if use_sr else "asr"
    ylab = ("StrongREJECT score (↑ better; honest ASR)" if use_sr
            else "substring attack success rate (↑ better)")
    for label, r in by_label.items():
        st = _style(label)
        ax1.scatter(r["nll_tax"], r["margin"], s=160, edgecolor="k", zorder=3, **st)
        ax1.annotate(label, (r["nll_tax"], r["margin"]), fontsize=8,
                     xytext=(5, 4), textcoords="offset points")
        yv = r.get(yk)
        if yv is not None:
            ax2.scatter(r["nll_tax"], yv, s=160, edgecolor="k", zorder=3, **st)
            ax2.annotate(label, (r["nll_tax"], yv), fontsize=8,
                         xytext=(5, 4), textcoords="offset points")
    ax1.set_xlabel("neutral-corpus coherence tax  ΔNLL  (→ worse)")
    ax1.set_ylabel("de-refusal margin  (↓ better, <0 complies)")
    ax1.set_title("Behaviour vs coherence (lower-left = win)")
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("neutral-corpus coherence tax  ΔNLL  (→ worse)")
    ax2.set_ylabel(ylab)
    ax2.set_title("De-refusal vs coherence (upper-left = win)")
    ax2.grid(alpha=0.3)
    r2 = meta.get("plant_r2_1step")
    sup = (f"CASA concept-cone (L2 rescue) — {meta['model'].split('/')[-1]}  "
           f"band {meta['band'][0]}..{meta['band'][1]}"
           + (f"   |   k×k plant R²(1-step)={r2:.3f}" if r2 else ""))
    fig.suptitle(sup, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = jpath.with_suffix(".png") if str(jpath).endswith(".json") else \
        _HERE.parent / "outputs" / "casa_cone.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
