"""Figures for the follow-up experiments:
  (a) pts_lookahead_results.npz  — does the MPC exploit horizon H, and where?
  (b) pts_capability_results.npz — effectiveness vs (honest, neutral) capability tax.
Run:  python pts_followup_plot.py  ->  pts_followup.png
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
la = np.load(HERE / "pts_lookahead_results.npz")
cap = np.load(HERE / "pts_capability_results.npz")

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# (a1) real-data band scan: H=1 vs H=8 tracking vs required swing
sep = np.degrees(la["scan_sep"]); e1 = la["scan_err_h1"]; e8 = la["scan_err_h8"]
order = np.argsort(sep)
ax[0, 0].plot(sep[order], e1[order], "o-", color="darkorange", label="H=1 (myopic)")
ax[0, 0].plot(sep[order], e8[order], "o-", color="crimson", label="H=8 (lookahead)")
ax[0, 0].axvspan(0, 30, color="green", alpha=0.08, label="feasible (gap ≤1.3°)")
ax[0, 0].axvspan(30, sep.max() + 5, color="red", alpha=0.06, label="infeasible (both fail)")
ax[0, 0].set_xlabel("required angular swing to reference (deg)")
ax[0, 0].set_ylabel("angular tracking error (deg)")
ax[0, 0].set_title("(a1) Real fitted bands: lookahead never\nturns a failure into a success")
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

# (a2) synthetic positive control: gap grows with rotation (machinery works)
psi = la["syn_psi"]
ax[0, 1].plot(psi, la["syn_err_h1"], "o-", color="darkorange", label="H=1")
ax[0, 1].plot(psi, la["syn_err_h8"], "o-", color="crimson", label="H=8")
ax2 = ax[0, 1].twinx()
ax2.bar(psi, la["syn_gap"], width=2.5, color="steelblue", alpha=0.3)
ax2.set_ylabel("H=1 − H=8 gap (deg)  [bars]", color="steelblue")
ax[0, 1].set_xlabel("imposed rotation ψ per layer (deg)")
ax[0, 1].set_ylabel("angular tracking error (deg)")
ax[0, 1].set_title("(a2) Synthetic positive control: the MPC\nDOES use H when dynamics rotate")
ax[0, 1].legend(fontsize=8, loc="lower left"); ax[0, 1].grid(alpha=0.3)

# (b1) effectiveness frontier: de-refusal vs steering budget
um = cap["pts_umax"]; mar = cap["pts_margin"]
ax[1, 0].plot(um, mar, "o-", color="crimson", label="PTS-MPC (u_max sweep)")
ax[1, 0].axhline(float(cap["base_margin"]), color="gray", ls="--", label="baseline (refuse)")
ax[1, 0].axhline(float(cap["fix_margin"]), color="royalblue", ls=":", label="fixed-angle")
ax[1, 0].axhline(0, color="k", lw=0.6, alpha=0.5)
ax[1, 0].set_xlabel("realised perturbation  mean ‖u‖")
ax[1, 0].set_ylabel("refusal–compliance margin")
ax[1, 0].set_title("(b) Effectiveness: tunable de-refusal,\nsaturates at the compliance boundary (~0)")
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

# (b2) honest capability tax (NEUTRAL corpus) vs de-refusal achieved
base_m = float(cap["base_margin"]); base_neu = float(cap["base_neu"])
derf = base_m - mar; tax = cap["pts_neu"] - base_neu
ax[1, 1].plot(derf, tax, "o-", color="crimson", label="PTS-MPC")
ax[1, 1].scatter([base_m - float(cap["fix_margin"])], [float(cap["fix_neu"]) - base_neu],
                 color="royalblue", s=110, marker="X", zorder=5, label="fixed-angle")
ax[1, 1].axhline(0, color="k", lw=0.6, alpha=0.5)
ax[1, 1].set_xlabel("de-refusal achieved  (baseline − margin)")
ax[1, 1].set_ylabel("neutral capability tax  ΔNLL (nats)")
ax[1, 1].set_title("(b) Honest capability tax (off-axis corpus):\nnegligible for both; slightly lower for PTS")
ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

fig.suptitle("PTS follow-up — lookahead stress-test (a) + effectiveness/capability (b), Qwen2.5-3B",
             fontsize=14)
plt.tight_layout()
plt.savefig(HERE / "pts_followup.png", dpi=130, bbox_inches="tight")
print(f"saved {HERE/'pts_followup.png'}")
