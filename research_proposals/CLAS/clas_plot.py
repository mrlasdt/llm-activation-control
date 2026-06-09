"""Plot CLAS prototype results from clas_prototype_results.npz."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

d = np.load("clas_prototype_results.npz")
deg, gh, gl = d["deg"], d["gh"], d["gl"]
band = d["band"]; peak = int(d["peak_deg"]); y_star = float(d["y_star"])
nom = float(d["theta_nom_deg"])
y_fixed, y_final = d["y_fixed"], d["y_final"]
traj_y, traj_th = d["traj_y"], d["traj_th"]   # (iters, B)
B = y_fixed.shape[0]; nh = B // 2

fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# --- (1) the measured plant G(theta) ---
ax[0].axvspan(band[0], band[1], color="gold", alpha=0.18, label="monotone band")
ax[0].plot(deg, gh, "o-", color="crimson", label="harmful margin")
ax[0].plot(deg, gl, "o-", color="royalblue", label="harmless margin")
ax[0].axhline(y_star, color="green", ls="--", lw=1.5, label=f"setpoint y*={y_star:.1f}")
ax[0].axvline(peak, color="black", ls=":", alpha=0.6, label=f"peak {peak}°")
ax[0].set_xlabel("steering angle θ (deg)"); ax[0].set_ylabel("behavioral margin  y = logP(R) − logP(C)")
ax[0].set_title("Measured plant  G(θ)  (continuous observable)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# --- (2) convergence of the closed loop ---
for i in range(B):
    c = "crimson" if i < nh else "royalblue"
    ax[1].plot(np.abs(y_star - traj_y[:, i]), color=c, alpha=0.4, lw=1)
ax[1].plot(np.abs(y_star - traj_y).mean(1), color="black", lw=2.5, label="mean |y*−y|")
ax[1].set_xlabel("control iteration"); ax[1].set_ylabel("|y* − y|  (behavioral error)")
ax[1].set_title("Closed-loop convergence\n(red=harmful, blue=harmless)")
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

# --- (3) open-loop vs closed-loop behavioral spread ---
x = np.arange(B)
ax[2].axhline(y_star, color="green", ls="--", lw=1.5, label=f"setpoint y*={y_star:.1f}")
ax[2].scatter(x, y_fixed, c=["crimson" if i < nh else "royalblue" for i in range(B)],
              marker="x", s=90, label="open-loop (fixed θ)")
ax[2].scatter(x, y_final, c=["crimson" if i < nh else "royalblue" for i in range(B)],
              marker="o", s=90, label="closed-loop (thermostat)")
for i in range(B):
    ax[2].plot([i, i], [y_fixed[i], y_final[i]], color="gray", alpha=0.4, lw=0.8)
ax[2].set_xlabel("prompt (0–3 harmful, 4–7 harmless)"); ax[2].set_ylabel("first-token behavioral margin y")
ax[2].set_title(f"Open-loop spread σ={y_fixed.std():.2f}  →  closed-loop σ={y_final.std():.2f}")
ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)

fig.suptitle("Closed-Loop Angular Steering (CLAS) prototype — Qwen2.5-3B, layer 27", fontsize=14)
plt.tight_layout()
plt.savefig("clas_prototype.png", dpi=130, bbox_inches="tight")
print("saved clas_prototype.png")
