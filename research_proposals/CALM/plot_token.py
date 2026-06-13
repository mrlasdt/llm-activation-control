"""CALM Phase 3 — plot the token-domain dynamic-vs-static comparison.

Reads CALM/outputs/calm_token_<model>.json and renders:
  (left)   StrongREJECT vs genNLL — static-hold anchors vs the KL-thermostat conditions.
           A dynamic WIN = a thermostat point strictly UPPER-LEFT of the static frontier
           (higher de-refusal at lower genNLL). Decides the GO/NO-GO on token-MPC.
  (mid)    per-token realized strength α_t and KL_t vs token index (example prompts) —
           shows whether the feedback actually MODULATES or is pinned (the "is it doing
           anything" diagnostic).
  (right)  α_std per condition — feedback that never moves is just static at a fixed level.

Run:  python plot_token.py [path-to-json]
"""

import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = pathlib.Path(__file__).resolve()


def main():
    if len(sys.argv) > 1:
        jpath = pathlib.Path(sys.argv[1])
    else:
        cands = sorted((_HERE.parent / "outputs").glob("calm_token_*.json"))
        if not cands:
            print("no calm_token_*.json — run calm_token.py --run first"); return
        jpath = cands[-1]
    data = json.loads(jpath.read_text())
    meta = data["meta"]; res = data["results"]
    name = meta["model"].split("/")[-1]

    def kind(tag):
        return tag.split("|")[0]                              # baseline / static / bang / prop
    colors = {"static": "tab:gray", "bang": "tab:red", "prop": "tab:orange", "baseline": "black"}

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.4))

    # (left) StrongREJECT vs genNLL — static frontier vs thermostat cloud
    for tag, d in res.items():
        if d.get("sr_score") is None:
            continue
        k = kind(tag); mk = {"static": "s", "bang": "^", "prop": "o", "baseline": "x"}.get(k, "o")
        ax1.scatter([d["gen_nll"]], [d["sr_score"]], marker=mk, s=90, color=colors.get(k, "tab:blue"),
                    edgecolor="k", lw=0.4, zorder=3)
        ax1.annotate(tag.replace("static|", "").replace("bang|", "b:").replace("prop|", "p:"),
                     (d["gen_nll"], d["sr_score"]), fontsize=6, xytext=(3, 3),
                     textcoords="offset points")
    # static frontier line
    st = sorted([d for t, d in res.items() if kind(t) == "static" and d.get("sr_score") is not None],
                key=lambda d: d["gen_nll"])
    if st:
        ax1.plot([d["gen_nll"] for d in st], [d["sr_score"] for d in st], "--",
                 color="tab:gray", lw=1.2, label="static frontier", zorder=2)
    for k, c in colors.items():
        ax1.scatter([], [], marker={"static": "s", "bang": "^", "prop": "o", "baseline": "x"}[k],
                    color=c, label=k)
    ax1.set_xlabel("genNLL (coherence — lower = fluent)")
    ax1.set_ylabel("StrongREJECT (de-refusal)")
    ax1.set_title("Token-domain: dynamic vs static\n(thermostat upper-left of static = WIN)")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=8)

    # (mid) alpha_t and KL_t vs token index for a couple of dynamic conditions
    dyn = [t for t in res if kind(t) in ("bang", "prop")]
    for tag in dyn[:3]:
        a = np.array(res[tag]["alpha_t"], float)
        a = np.where(np.isfinite(a), a, np.nan)
        ax2.plot(a, lw=1.2, label=f"α_t {tag.split('|')[1]}")
    ax2.set_xlabel("token index"); ax2.set_ylabel("realized strength u_max")
    ax2.set_title("Feedback modulation of strength\n(flat = pinned = not adapting)")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
    ax2b = ax2.twinx()
    if dyn:
        klex = np.array(res[dyn[0]]["kl_t_ex"], float)
        ax2b.plot(np.nanmean(klex, 0), color="purple", alpha=0.4, lw=1.0, label="mean KL_t")
        ax2b.set_ylabel("KL_t (nats)", color="purple")

    # (right) alpha_std per condition
    tags = [t for t in res if t != "baseline"]
    asd = [res[t].get("alpha_std", 0.0) for t in tags]
    cols = [colors.get(kind(t), "tab:blue") for t in tags]
    ax3.barh(range(len(tags)), asd, color=cols)
    ax3.set_yticks(range(len(tags))); ax3.set_yticklabels(tags, fontsize=6)
    ax3.set_xlabel("α_std (per-token strength variation)")
    ax3.set_title("Is the feedback doing anything?\n(α_std≈0 ⇒ effectively static)")
    ax3.grid(alpha=0.3, axis="x")

    fig.suptitle(f"CALM Phase 3 — token-domain sustained steering — {name} "
                 f"(n={meta.get('n')}, {meta.get('max_new_tokens')} tok, "
                 f"baseline SR={meta.get('base_sr_score')})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = _HERE.parent / "outputs" / f"calm_token_{name}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # numeric verdict
    if st:
        # build static frontier as (genNLL, sr) and check if any dynamic point dominates
        import numpy as _np
        sf = _np.array([(d["gen_nll"], d["sr_score"]) for d in st])
        wins = []
        for t in dyn:
            d = res[t]
            if d.get("sr_score") is None:
                continue
            # dominated-by-static? find static points with genNLL<=this and compare sr
            better = [(g, s) for g, s in sf if g <= d["gen_nll"] + 1e-9]
            best_static_sr = max([s for _, s in better], default=-1)
            margin = d["sr_score"] - best_static_sr
            wins.append((t, margin, d["sr_score"], d["gen_nll"], d.get("alpha_std", 0.0)))
        wins.sort(key=lambda x: -x[1])
        print("\ndynamic-vs-static (Δsr at ≤ matched genNLL; >+0.03 with α_std>0 = candidate win):")
        for t, mg, sr, gn, asd_ in wins:
            print(f"  {t:22s} Δsr {mg:+.3f}  (sr {sr:.3f} @ genNLL {gn:.3f}, α_std {asd_:.1f})")


if __name__ == "__main__":
    main()
