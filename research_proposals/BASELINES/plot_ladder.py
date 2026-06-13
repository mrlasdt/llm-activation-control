"""Head-to-head figure: the P → PID → LQR → MPC ladder on the shared cone plant.

Reads the CASA experiment JSON (rows tagged `CONE k=*+{P,PID,LQR,LQR+ff,MPC}` with
`realized_effort`, `sr_score`, `gen_nll`, `margin`, `umax`) and draws:

  (left)  StrongREJECT srScore (↑ de-refusal) vs realized control effort mean‖u‖ (→),
          one connected curve per control law across the u_max budgets — the matched-effort
          comparison: does MPC dominate the others at equal effort?
  (right) srScore (↑) vs coherence genNLL (→ worse) — behaviour/fluency Pareto.

Run:  python plot_ladder.py [../CASA/outputs/casa_cone_gemma-2-2b-it.json]
"""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve()

_LAW_STYLE = {
    "P":      dict(color="tab:orange", marker="^"),
    "PID":    dict(color="tab:green",  marker="s"),
    "LQR":    dict(color="tab:blue",   marker="o"),
    "LQR+ff": dict(color="tab:cyan",   marker="v"),
    "MPC":    dict(color="tab:purple", marker="D"),
}


def _law(label):
    return label.split("+", 1)[1] if "+" in label else label


def main():
    jpath = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
             else _HERE.parents[1] / "CASA" / "outputs" / "casa_cone_gemma-2-2b-it.json")
    data = json.loads(pathlib.Path(jpath).read_text())
    meta, rows = data["meta"], data["rows"]
    laws = {}
    for r in rows:
        lab = r["label"]
        if "+" not in lab:
            continue
        law = _law(lab)
        if law not in _LAW_STYLE:
            continue
        laws.setdefault(law, []).append(r)
    if not laws:
        print("no ladder rows (CONE k=*+{P,PID,LQR,LQR+ff,MPC}) in", jpath); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    base_sr = meta.get("base_sr_score")
    if base_sr is not None:
        ax1.axhline(base_sr, ls=":", c="k", lw=1, label=f"baseline sr {base_sr:.2f}")
        ax2.axhline(base_sr, ls=":", c="k", lw=1)
    for law, rs in laws.items():
        rs = sorted(rs, key=lambda r: (r.get("realized_effort") or 0.0))
        st = _LAW_STYLE[law]
        eff = [r.get("realized_effort") for r in rs]
        sr = [r.get("sr_score") for r in rs]
        gn = [r.get("gen_nll") for r in rs]
        if all(e is not None for e in eff) and all(s is not None for s in sr):
            ax1.plot(eff, sr, "-", **st, alpha=0.85, label=law)
            for r in rs:
                ax1.scatter(r["realized_effort"], r["sr_score"], s=150, edgecolor="k",
                            zorder=3, **st)
                tag = "∞" if r["umax"] is None else f"{r['umax']:.0f}"
                ax1.annotate(tag, (r["realized_effort"], r["sr_score"]), fontsize=7,
                             xytext=(4, 4), textcoords="offset points")
        if all(g is not None for g in gn) and all(s is not None for s in sr):
            ax2.plot(gn, sr, "-", **st, alpha=0.85, label=law)
            for r in rs:
                ax2.scatter(r["gen_nll"], r["sr_score"], s=150, edgecolor="k", zorder=3, **st)

    ax1.set_xlabel("realized control effort  mean‖u_l‖  (→)")
    ax1.set_ylabel("StrongREJECT srScore  (↑ de-refusal)")
    ax1.set_title("Behaviour vs effort — matched-effort ladder")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel("coherence genNLL  (→ worse / gibberish)")
    ax2.set_ylabel("StrongREJECT srScore  (↑ de-refusal)")
    ax2.set_title("Behaviour vs coherence")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    r2 = meta.get("plant_r2_1step")
    fig.suptitle(f"Control-law head-to-head on the shared cone plant — "
                 f"{meta['model'].split('/')[-1]}  band {meta['band'][0]}..{meta['band'][1]}"
                 + (f"   |   plant R²(1-step)={r2:.3f}" if r2 else ""), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _HERE.parent / "outputs" / "ladder_pareto.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
