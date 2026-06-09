# Predictive Trajectory Steering (PTS)

Implementation of the proposal in [`../research_proposal_PTS.md`](../research_proposal_PTS.md):
**Model Predictive Control in the 2D Angular Steering plane**. PTS replaces the
fixed scalar/angle setpoint of prior steering methods with a **2D reference
trajectory** through the steering plane, and tracks it with a constrained,
receding-horizon **MPC across layers** whose output is a per-layer adaptive
Angular-Steering angle.

This is the PTS sibling of the CLAS deliverable (`clas_*.py`). It shares the
verified pieces of the repo — the steering plane `{b1,b2}` from
`phase_portrait.compute_steering_plane`, the norm-preserving SO(2) reset actuator
from `clas_controller`, and the continuous refusal/compliance margin observable —
and adds the four PTS-specific components.

## Modules

| File | What it is | Validated by |
|------|-----------|--------------|
| `pts_dynamics.py` | Per-layer 2×2 **affine** dynamics `c_{k+1}≈A_k c_k + b_k`, fit by least squares from contrastive forward passes (proposal §5.1a). Rollout + multi-horizon validation. | `python pts_dynamics.py` (synthetic recovery) |
| `pts_mpc.py` | The MPC core: reference trajectories (§4.3 A/B/C/D), condensed-QP construction (§5.2), a **dependency-free QP solver** (closed-form unconstrained + FISTA for the `‖u‖≤u_max` constraint), discrete LQR via the DARE (the "A-LQR projected to 2D" baseline), and the §5.4 angle conversion. | `python pts_mpc.py` (7 self-tests) |
| `pts_controller.py` | Forward-pass-coupled **multi-layer steering hooks** + policies (`PolicyMPC`, `PolicyLQR`, `PolicyFixedAngle`, `PolicyNone`). Every controller routes through the same verified reset actuator. | `python pts_controller.py` (norm-preserving + angle-exact) |
| `pts_offline.py` | Validates the controller maths on the **saved** `trajectories.npz` — **no GPU** (Exp 1,2,3,4,7). | `python pts_offline.py` |
| `pts_prototype.py` | **Model-in-the-loop** on Qwen2.5-3B: dynamics on the real residual stream, closed-loop tracking, behavioural control, generations. | `python pts_prototype.py` (A10G, ~1 min) |
| `pts_plot.py` | Figures from both result files. | `python pts_plot.py` |

```bash
# numerical core (no model, deterministic)
python pts_dynamics.py && python pts_mpc.py && python pts_controller.py
python pts_offline.py          # -> phase_portrait_output/<model>/pts_offline_results.npz
# model in the loop (needs the GPU + Qwen2.5-3B)
python pts_prototype.py        # -> pts_prototype_results.npz
python pts_plot.py             # -> pts_offline.png, pts_prototype.png
```

## The one modelling decision that matters

Angular Steering's actuator is a **norm-preserving SO(2) reset**: it discards the
in-plane angle and sets it to a target, keeping `‖proj_plane‖`. The MPC, however,
plans an **additive** control `u_k` under the linear model `c_{k+1}=A_k c_k + B_k u_k`
with `B_k=I`. PTS reconciles these exactly as proposal §5.4 prescribes: the MPC
plans the 2D point `c_k+u_k`, and the device realises only its **angle**
`θ_k = atan2(c2+u2, c1+u1)` at the incoming magnitude. So PTS controls the angular
coordinate of the plane; the radial coordinate evolves autonomously (by construction
of rotation-based steering). The gap between the additive plan and the angle-only
realisation is precisely the linearisation residual `w_j` of the tracking bound
(§8.1) — we **measure** it in the real model (≈14% of the coordinate scale).

## Results (Qwen2.5-3B-Instruct)

### Offline — controller validation on saved trajectories (`pts_offline.png`)

- **Exp 1 — 2D carries more information than 1D.** Harmful-vs-harmless 5-fold CV
  AUC: **1D (b₁ projection) 0.863 → 2D (plane) 0.958** (+0.095); the peak layer is
  perfectly separable in both. The steering plane's second axis is not redundant.
- **Exp 2 — the 2×2 affine model is accurate.** Held-out **R² = 0.996 (1-step),
  0.969 (5-step)**; the affine term `b_k` beats the pure-linear fit. A 256-byte/layer
  model reproduces the layerwise coordinate evolution — no Jacobians needed (§5.1).
- **Exp 3 — tracking-vs-perturbation Pareto.** The honest axis is the *realised*
  in-plane perturbation `‖u‖` (how far off the reachable manifold the actuation
  pushes — cf. non-surjectivity, arXiv:2604.09839). FixedAngle and unconstrained
  LQR ignore the budget and sit at one **high-perturbation** point. **At matched
  perturbation `‖u‖≈6.4`, PTS-MPC tracks 0.3° vs FixedAngle's 13.5°**, and PTS can
  dial perturbation all the way down to `‖u‖≈0.86`, which the baselines cannot.
- **Exp 3b — lookahead is benign here (honest negative).** H=1…8 all give ≈19° at a
  tight budget, because the behavioural late band has near-identity dynamics
  (`‖A_k−I‖₂≈0.34`). PTS's value in this band is its **2D reference** and its
  **constraint handling**, not its horizon. Lookahead would matter in
  rotation-dominated bands — out of scope here, flagged not hidden.
- **Exp 4 — only PTS respects the budget.** PTS max `‖u‖ = u_max` exactly;
  unconstrained LQR overshoots the same budget by **6.3×**.
- **Exp 7 — efficiency.** QP solve ≈ **0.15 µs/prompt** (unconstrained/explicit) to
  **22 µs/prompt** (constrained FISTA, batched), ≪ a single 3B-layer forward.
  Storage **48 B/layer** (all `A_k,b_k`) vs A-LQR's **16.8 MB/layer** d×d gain
  (**≈350,000× smaller**).

### Model-in-the-loop — the real, nonlinear plant (`pts_prototype.png`)

- **[A]** Dynamics fit on the **real residual stream**: held-out **R² = 0.999
  (1-step), 0.989 (5-step)** — the linear model is, if anything, *more* accurate on
  the actual steering point than on the layernorm trajectories.
- **[B]** PTS **bends the real model's residual stream**: the incoming trajectory's
  angular distance to the reference drops from **139° (autonomous) → 69.5° (PTS)**.
  Perfect tracking is not reached — the nonlinear plant fights the linear plan — but
  the one-step mismatch `w_j` stays **bounded at ≈14% of the coordinate scale**. (The
  §8.1 contraction bound is proven only for the additive planning model; here `w_j`
  is *measured*-bounded for the deployed norm-preserving actuator, not proven — see
  "Discrepancies" below.)
- **[C]** The 2D reference is a **graded behavioural dial.** Interpolating the
  reference from the harmful-mean to the harmless-mean trajectory (Option B,
  λ:0→1) sweeps the refusal/compliance margin **+11.7 (refuse) → +5.8 (λ=0.5) → ≈0
  (comply)** — monotone de-refusal over λ∈[0, 0.75], saturating near the compliance
  floor by λ=1. Graded behavioural control that a single fixed angle cannot target.
- **[D]** Coherent de-refusal under PTS: *"I'm sorry, but I can't assist…"* →
  *"Creating a program to hack… is an interesting challenge"*. The bounded
  perturbation keeps the output fluent.

**Polarity, stated honestly:** tracking the *harmless-prompt* (benign Alpaca)
trajectory removes the refusal feature, i.e. it is the **de-refusal / jailbreak**
direction — which is where the observable has headroom, since harmful prompts
already refuse at baseline (the CLAS finding: baseline harmful refusal is saturated
at 100%). The safety-positive direction is the opposite reference (λ→0, harmful
mean), which preserves refusal. PTS gives **bidirectional, graded** control; this
implementation demonstrates control of the behaviour, not a safety claim.

## Discrepancies with the proposal text (where this implementation is more careful)

An adversarial review of this implementation against the proposal surfaced four
places where the **proposal text overstates** what its own mechanics deliver; the
code here handles them honestly, and they are called out so no claim is taken on
faith:

1. **Angle-only actuation drops the magnitude (§5.1/§5.4).** The OCP plans an
   *additive* update with `B_k=I` and a cost penalizing the full 2D distance
   `‖c−τ*‖` (magnitude *and* angle). The deployed Angular-Steering actuator is
   norm-preserving — it realizes only the *angle* of `c+u` and keeps `‖c‖`. So the
   MPC optimizes over a radial component the device cannot move. The proposal frames
   §5.4 as "just modifies the target angle" and never states this; this code models
   the additive plan and the angle-only realization separately and **measures** the
   gap (the `w_j` residual: `pts_offline.py` `actuator='additive'` vs `'angle'`;
   `pts_prototype.py` [B]).
2. **§8.1 contraction is proven only for the additive model.** `rho=max‖A_k−B_k K_k‖`
   and the geometric-contraction claim hold for the MPC's internal `B=I` plant
   (`pts_mpc.py` self-test [7] validates exactly that). For the realized
   norm-preserving actuator there is no contraction proof — `w_j` is *measured*
   bounded (≈14% of scale), not proven.
3. **Exp 3 here is a tracking-vs-perturbation Pareto**, not the proposal's
   "effectiveness vs capability (PPL / TinyBenchmarks)" Pareto. Both axes
   (angular tracking error, realized `‖u‖`) are control-internal; the
   behavioural/capability axis is delivered separately and partially by the
   prototype's margin dial + coherent generations, not by a PPL sweep.
4. **Exp 4 here is the `‖u‖`-budget / constraint-satisfaction check (§8.2)**, not the
   proposal's "perplexity-stability across the angle sweep" plot. The constraint is
   shown to bound the perturbation (the mechanism §8.2 claims protects coherence);
   the perplexity measurement that would *confirm* the coherence payoff is not run.

## What is and isn't implemented

Implemented and validated: §4.3 references A/B (and the C/D scaffolding), §5
condensed MPC + dependency-free constrained solver + DARE baseline (the LQR
baseline is *exactly* the unconstrained limit of PTS-MPC, isolating the
constraint's effect), §5.4 angle conversion, the per-layer adaptive-angle
actuation, §7 Exp 1/2/7 fully and Exp 3/4 as control-internal proxies (see
Discrepancies 3–4), and empirical checks of the §8.1 residual and §8.4
plane-sufficiency intuition (2D>1D).

Out of scope for this pass (documented, not silently dropped): §6 token-horizon
two-level MPC; the PPL/capability axis of Exp 3 and the perplexity-stability plot
of Exp 4; full Exp 5 ablation over reference Options C(tube)/D and Exp 6 long-form
drift; the PID-AcT / ODESteer external baselines (only the in-repo fixed-angle and
the LQR analogue are compared); multi-model generalisation (§7.1).
