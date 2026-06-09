# Archive — CLAS (Closed-Loop Angular Steering)

**Dropped 2026-06-09.** This folder holds the CLAS proposal and prototype, retired
after a falsification experiment. Kept (not deleted) because the result is a useful
**negative finding** and the prototype code is reusable.

## Why CLAS was dropped

CLAS proposed an *output-feedback thermostat*: measure a behavioral signal off the
model's own logits, and adjust the SO(2) steering angle per token to hold a behavioral
setpoint. Two findings sank it:

1. **The refusal framing is degenerate.** Refusal is a *one-shot* decision (made at the
   first token) and is already *saturated* (100% on Qwen2.5-3B), and the first-token
   margin is nearly an affine function of the internal angle (so "output feedback" ≈
   "state feedback" — the claimed novelty collapses). A one-shot decision needs
   *feedforward* (invert the measured `G(θ)`), not a token loop.

2. **Sustained attributes aren't angular-steerable (the drift test).** A token loop only
   makes sense for an attribute that persists/drifts over a long generation (e.g.
   sentiment). But angular steering is **norm-preserving** — it only re-points the part
   of the activation lying in the 2D plane — so its authority ∝ how much of the attribute
   lives in that plane. For the dominant refusal feature that's large (sweep range ~18,
   flips behavior); for sentiment it's negligible (range ~1.6, and steered vs unsteered
   200-token generations were **byte-identical**).

**The bind:** strongly-steerable attributes (refusal) are one-shot → no loop value;
loop-needing attributes (sentiment) aren't steerable by this actuator. No niche.

**Scope of the negative result:** single-layer angular *reset*, auto-selected layer,
instruction-position contrastive directions, Qwen2.5-3B-Instruct. Not ruled out:
multi-layer / additive actuators (which have unbounded authority but overlap PTS).

## What's reusable here

- `clas_controller.py` — a **mutable-angle steering hook** (reads the commanded angle
  fresh each forward pass), a **behavioral output observable** (refusal/compliance
  log-prob margin), and a **PI thermostat** (anti-windup + band clamp + slope sign-guard).
- `clas_prototype.py` — measures `G(θ)` on a continuous observable; the **manual
  KV-cached decode loop** (verified against `model.generate`) is the generally useful bit.
- `clas_drift_test.py` — the falsification experiment (sentiment plane + 3-mode drift).
- `*.png`, `*.npz` — the figures and saved results.

## Active proposals (kept)

- `../research_proposal_SO2.md` — energy/Lyapunov state-feedback over depth.
- `../research_proposal_PTS.md` — predictive MPC in the 2D plane (implemented:
  `../pytorch_pure/pts_*.py`).
