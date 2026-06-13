"""CALM — plot the strength↔coherence frontier from a --frontier run.

Reads CASA/outputs/casa_frontier_<model>.json and renders:
  (left)   StrongREJECT (de-refusal) vs genNLL (coherence) — THE frontier. Each law
           is a curve over its strength grid; cMPC is shown per κ. Up-and-left = better
           (more de-refusal at lower genNLL). Tests: does cMPC dominate P/MPC?
  (mid)    StrongREJECT vs realized effort — the matched-effort view.
  (right)  coherence surrogate (cone-space Mahalanobis) vs genNLL — the surrogate
           VALIDATION: if these correlate, the cone-space density CALM optimizes is a
           faithful coherence proxy (justifies Phase 2's quadratic cost).

Run:  python plot_frontier.py [path-to-json]
"""

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve()


def _law_of(label):
    # "CONE k=4+P" -> "P"; "CONE k=4+cMPC@0.5" -> "cMPC@0.5"; "CONE k=4+MPC" -> "MPC"
    return label.split("+", 1)[1] if "+" in label else label


def main():
    if len(sys.argv) > 1:
        jpath = pathlib.Path(sys.argv[1])
    else:
        cands = sorted((_HERE.parents[1] / "CASA" / "outputs").glob("casa_frontier_*.json"))
        if not cands:
            print("no casa_frontier_*.json found — run casa_experiment.py --frontier first")
            return
        jpath = cands[-1]
    data = json.loads(jpath.read_text())
    meta = data["meta"]; rows = data["rows"]
    name = meta["model"].split("/")[-1]
    base_sr = meta.get("base_sr_score")

    # group control-law rows by law (P / MPC / cMPC@κ); skip the static-ablation anchors
    laws = {}
    anchors = []
    for r in rows:
        if r.get("realized_effort") is None:                 # static-ablation anchor (CONE/RDO)
            if r.get("sr_score") is not None:
                anchors.append(r)
            continue
        laws.setdefault(_law_of(r["label"]), []).append(r)

    def srt(rs, key):
        rs = [r for r in rs if r.get(key) is not None and r.get("sr_score") is not None]
        return sorted(rs, key=lambda r: r[key])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.4))
    cmap = plt.get_cmap("tab10")
    colors = {law: cmap(i) for i, law in enumerate(sorted(laws))}

    # (left) StrongREJECT vs genNLL — the frontier
    for law in sorted(laws):
        rs = srt(laws[law], "gen_nll")
        x = [r["gen_nll"] for r in rs]; y = [r["sr_score"] for r in rs]
        marker = "o" if law.startswith("cMPC") else ("s" if law == "MPC" else "^")
        ax1.plot(x, y, marker=marker, color=colors[law], label=law, lw=1.6, ms=6, alpha=0.9)
    for r in anchors:
        ax1.scatter([r["gen_nll"]], [r["sr_score"]], marker="*", s=140, color="black", zorder=5)
        ax1.annotate(r["label"].split("+")[0], (r["gen_nll"], r["sr_score"]),
                     fontsize=7, xytext=(4, 4), textcoords="offset points")
    if base_sr is not None and meta.get("base_gen_nll") is not None:
        ax1.scatter([meta["base_gen_nll"]], [base_sr], marker="x", s=80, color="gray", label="baseline")
    ax1.set_xlabel("genNLL  (coherence — lower = more fluent)")
    ax1.set_ylabel("StrongREJECT  (de-refusal — higher = stronger)")
    ax1.set_title("Strength↔coherence frontier\n(up & left = dominates)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    # (mid) StrongREJECT vs realized effort — matched-effort view
    for law in sorted(laws):
        rs = srt(laws[law], "realized_effort")
        x = [r["realized_effort"] for r in rs]; y = [r["sr_score"] for r in rs]
        marker = "o" if law.startswith("cMPC") else ("s" if law == "MPC" else "^")
        ax2.plot(x, y, marker=marker, color=colors[law], label=law, lw=1.6, ms=6, alpha=0.9)
    ax2.set_xlabel("realized effort  mean‖u_l‖")
    ax2.set_ylabel("StrongREJECT")
    ax2.set_title("De-refusal vs realized control effort")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    # (right) surrogate validation: cone-space Mahalanobis vs genNLL
    sx, sy = [], []
    for law in sorted(laws):
        rs = [r for r in laws[law] if r.get("surrogate") is not None and r.get("gen_nll") is not None]
        x = [r["surrogate"] for r in rs]; y = [r["gen_nll"] for r in rs]
        ax3.scatter(x, y, color=colors[law], label=law, s=40, alpha=0.85)
        sx += x; sy += y
    corr = float(np.corrcoef(sx, sy)[0, 1]) if len(sx) > 2 else float("nan")
    ax3.set_xlabel("coherence surrogate  Σ Mahalanobis(s_l, harmless density)")
    ax3.set_ylabel("genNLL (measured)")
    ax3.set_title(f"Surrogate validation\nPearson r(surrogate, genNLL) = {corr:.2f}")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=8)

    fig.suptitle(f"CALM — coherence-aware MPC frontier — {name} "
                 f"(plant R²={meta.get('plant_r2_1step', float('nan')):.4f}, "
                 f"n={meta.get('n_obs')})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = _HERE.parent / "outputs" / f"calm_frontier_{name}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print(f"surrogate↔genNLL Pearson r = {corr:.3f}  (high ⇒ cone-space density is a "
          "faithful coherence proxy)")


if __name__ == "__main__":
    main()
