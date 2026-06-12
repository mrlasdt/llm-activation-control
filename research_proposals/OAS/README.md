# Observer-Based, Soft-Landing Angular Steering (OAS)

Implementation of the proposal in [`research_proposal_OAS.md`](research_proposal_OAS.md)
against the engineering contract in [`OAS_SPEC.md`](OAS_SPEC.md): **depth-domain LQG
control of LLM behavior**. OAS recasts Angular Steering's one-layer deadbeat angle
reset as a **finite-horizon LQ tracking problem with a terminal-angle cost over a band
of layers** (Idea 1, "soft landing"), and fuses the weak per-layer geometric readouts
with a **Kalman observer** before controlling them (Idea 2, LQG by the separation
principle).

This is the OAS sibling of the PTS deliverable (`../PTS/pts_*.py`). It shares the
verified pieces of the repo — the steering plane `{b1,b2}` from
`phase_portrait.compute_steering_plane`, the **2×2 affine plant** `c_{k+1}=A_k c_k+b_k`
fit by `pts_dynamics` (held-out R²≈0.999), and the norm-preserving SO(2) reset actuator
— and adds the OAS-specific control: the terminal-cost LQ-tracking recursion, the
Kalman/EKF observer, and the deadbeat-vs-distributed analysis.

**Self-containment.** OAS imports only the shared lib (`utils`, `phase_portrait`,
`observables`), `pts_dynamics` (the pure-numpy plant fit, treated as shared), and its
own `oas_*` modules. It never imports from `../CLAS`, and from `../PTS` it imports
**only `pts_dynamics`**. Every `oas_*.py` adds *both* `../../pytorch_pure` and the
sibling `../PTS` to `sys.path` via the shim at the top of the file (documented in
`OAS_SPEC.md §0`).

## Modules

| File | Lines | What it is | Validated by |
|------|------:|-----------|--------------|
| `oas_lqr.py` | 434 | The novelty over PTS: **finite-horizon LQ tracking with a TERMINAL cost** via the backward Riccati recursion (`OAS_SPEC §1.2`, implemented verbatim) — soft, bounded rotations across a band that land the angle only at the terminal layer `kT`. Deadbeat = the `R→0` corner. Pure numpy, 2×2. | `python oas_lqr.py` (condensed-QP gold cross-check < 1e-8, deadbeat-limit landing, monotone effort↔error frontier, soft-landing beats deadbeat on per-layer push) |
| `oas_observer.py` | 544 | **Kalman filter** (time-varying-capable, Joseph-form covariance) + **EKF on the circle** for the wrapping angle; process-noise estimation from the fitted plant's residuals (the ≈14% PTS residual as a covariance `W`); observability matrix / Gramian (time-invariant and time-varying); steady-state DARE gain for the cross-check. | `python oas_observer.py` (filtered MSE < raw, KF gain = DARE gain, partial-observability recovery, V→0 / W→0 degeneracy, EKF tracks a wrapping ramp) |
| `oas_controller.py` | 553 | The in-the-loop glue: own copy of the norm-preserving SO(2) reset hook + the policy bank (`PolicyNone`, `PolicyDeadbeat`, `PolicyMultiAngle`, `PolicySoftLandingLQR`, `PolicyLQG`). `PolicyLQG` runs the Kalman filter sequentially across band layers within one forward pass and resets per pass. | `python oas_controller.py` (reset is norm-preserving + angle-exact, `PolicySoftLandingLQR` reproduces `lqr_rollout` angles, `PolicyLQG(V→0)` == `PolicySoftLandingLQR`) |
| `oas_offline.py` | 752 | **Model-FREE** experiments on the saved `trajectories.npz` (the same file PTS uses) — the control maths validated cheaply (CI target < ~30 s): plant + noise fit, Exp-2 geometric enforcement-layer sweep, Exp-3 soft-vs-deadbeat (effort, robustness, advantage-vs-`‖A_k−I‖`), Exp-4 observability + Kalman-vs-single-vs-EMA + LQG-vs-LQR-under-noise. | `python oas_offline.py` → `oas_offline_results.npz` |
| `oas_prototype.py` | 784 | **Model-in-the-loop** on Qwen2.5-3B (A10G): Exp-1 authority gate (refusal + sentiment), Exp-2 behavioral enforcement-layer sweep, Exp-3 soft-vs-deadbeat coherence (KL) & robustness at matched effect, Exp-4 observer value under injected readout noise. `--smoke` dry-runs the whole pipeline on tiny N. | `python oas_prototype.py --smoke` (end-to-end on the real model) |
| `oas_plot.py` | 236 | Figures from both result files (Agg backend, both plots guarded on the npz existing). | `python oas_plot.py` → `oas_offline.png`, `oas_prototype.png` |

```bash
# numerical core (no model, deterministic) — run from inside research_proposals/OAS
python oas_lqr.py && python oas_observer.py && python oas_controller.py
python oas_offline.py          # -> ../SO2/phase_portrait_output/<model>/oas_offline_results.npz
python oas_plot.py             # -> oas_offline.png (+ oas_prototype.png if the proto npz exists)
# model in the loop (needs the GPU + Qwen2.5-3B)
python oas_prototype.py --smoke   # tiny dry-run; drop --smoke for the full run
```

## Load-bearing design facts

- **The novelty over PTS is the terminal-cost LQ-tracking recursion.** PTS's MPC tracks
  a 2D reference at *every* layer with a uniform-`Q` cost. OAS instead puts `Q_k = 0`
  (or tiny `Q_stage`) on the intermediate layers and a large `Q_term` only at the
  terminal layer `kT`: "don't force the angle early; just land it by `kT`." The
  backward recursion (`OAS_SPEC §1.2`) is `M_k = Q_k + R + A_kᵀ S_{k+1} A_k`,
  `F_k = M_k⁻¹ R`, `g_k = M_k⁻¹ e_k`, `S_k = R − R M_k⁻¹ R` with sentinel
  `S_{kT+1}=0`. The condensed-QP gold cross-check in the self-test proves the rolled
  Riccati policy matches the single-shot QP minimiser to < 1e-8.
- **Deadbeat = the `R→0` corner.** The existing single-layer Angular Steering reset is
  the degenerate `R→0`, `Q_term→∞`, one-layer, no-observer special case
  (`deadbeat_gains`). The self-test confirms the `R→0` limit lands the *next*
  coordinate `x_{kT+1}` exactly on target.
- **LQG observer (separation principle).** The Kalman filter (`oas_observer`) estimates
  the latent behavioral state from the noisy/partial per-layer angle, and the LQR
  (`oas_lqr`) controls the *estimate* — designed independently and combined optimally
  (`PolicyLQG`). With `V→0` the filter trusts the measurement fully and LQG collapses
  onto the open-loop soft-landing LQR (asserted in the controller self-test).
- **Actuator convention (identical to PTS).** State `x_k` = the natural incoming
  coordinate; decision `s_k = x_k + u_k` = the actuated coordinate; dynamics
  `x_{k+1} = A_k s_k + b_k`. The physical reset is **norm-preserving**: it realizes
  `angle(s_k)` and keeps `‖x_k‖`. The control plans an additive `u_k`; the device
  keeps only the angle. This is `OAS_SPEC §0`, matching `pts_controller._apply_reset`.

## Results (Qwen2.5-3B-Instruct)

### Offline — controller validation on saved trajectories (`oas_offline.png`)

Steering band (same recipe as PTS, `|angle_sep|>0.5` rad around the peak): layers
**21–34** (terminal `kT=34`), target angle ≈ **27°** (the harmless-ref band angle).
Held-out plant R² = 0.999 (1-step), 0.989 (5-step).

- **Exp 2 — the angle must land LATE.** Enforcing the target angle at a single layer
  and rolling the fitted plant to the band end: terminal angle error falls from ~0.45
  rad (enforce mid-band) to **0.03 rad at `kT=34`** — early enforcement washes out under
  the natural dynamics, which is exactly what motivates the *terminal* objective.
- **Exp 3a — the soft-landing is a tunable frontier.** Sweeping `ρ` (`R = ρ I`) traces a
  monotone effort↔terminal-error frontier; `ρ→0` is the deadbeat endpoint.
- **Exp 3b — soft-landing matches the deadbeat angle at a far smaller per-layer push.**
  At matched terminal angle (gap ≈ 0 rad), the single-layer deadbeat needs `max‖u‖ ≈
  12.5` (charged against the true terminal-layer state `x_{kT}`, ‖x‖~11) while the
  soft-landing band needs `max‖u‖ ≈ 1.4` — a **≈8.9× smaller** per-layer push.
- **Exp 3c — robustness.** Re-rolling both fixed controllers through a perturbed plant
  `Â = A + δ·N(0,1)`: soft-landing terminal-angle error/variance ≈ the deadbeat's here
  (both small in this near-identity late band).
- **Exp 3d — advantage vs rotation.** Across width-5 bands swept over depth, the
  scale-free per-layer-push advantage is positive in **100%** of bands, but the
  predicted positive correlation with `‖A_k−I‖₂` is **not** borne out (corr ≈ −0.09) —
  reported honestly (see caveats).
- **Exp 4 — the observer earns its place under noise.** `(A_band, H)` is observable for
  both full (`H=I`) and scalar (`H=e₁`) reads. Under 30% measurement noise the
  Kalman-fused angle RMSE (**0.31 rad**) beats both the best single-layer read (0.46)
  and the ad-hoc EMA (0.38). In the LQG-vs-LQR-on-raw sweep the two **tie at V→0** and
  LQG's terminal-angle error/variance pulls clearly ahead as noise grows.

### Model-in-the-loop — the real, nonlinear plant (`oas_prototype.png`, smoke run)

- **Exp 1 — authority gate.** Multi-layer rotation raises the realized behavioral range
  vs a single layer for both refusal (+9%) and sentiment (+49%) in the smoke run —
  PASS for both (the lever CLAS left open). The full run sweeps wider bands.
- **Exp 2 — behavioral enforcement curve.** Single-layer deadbeat de-refuses strongly
  at the mid/late band (margin +11.9 baseline → −5.7 at layer 27), cross-checking the
  offline geometric curve.
- **Exp 3 — soft-landing vs deadbeat.** Coherent de-refused generations under both
  controllers (*"I'm sorry, but I can't"* → *"Creating a program to hack into a
  secure…"*). At the tested band/`kT` the soft-landing does **not** reach the
  single-layer slam's behavioral effect, so the coherence comparison is reported as
  **"effect not matched"** rather than as a win (see caveats). The robustness test now
  exposes **both** controllers to the perturbed plant; in the smoke run the soft-landing
  is the less plant-sensitive of the two.
- **Exp 4 — observer value in-model.** Under injected readout noise the LQG-filtered
  realized angle is markedly smoother (roughness 0.20 vs 1.21 for LQR-on-raw at the
  highest noise) — the principled EMA, confirmed on the real model.

## Discrepancies / caveats (where this implementation is careful)

An adversarial review surfaced places where the proposal's *prediction* is not borne
out by the mechanics, or where a claim needs guarding. The code reports each honestly:

1. **Norm-preserving-actuator tax (inherited from PTS).** The LQR plans an *additive*
   `u_k` and penalizes the full 2D distance, but the deployed actuator keeps only the
   *angle* of `s_k` and drops the magnitude. So OAS controls the angular coordinate; the
   radial coordinate evolves autonomously. The offline maths is run in the additive
   convention (where the policy and rollout agree exactly); the angle-only realization
   introduces the separately-flagged Exp-5 tax — `PolicySoftLandingLQR` reproduces the
   rollout's *angles*, not its magnitudes.
2. **Near-identity late band hedge — and it cuts against Idea 1's depth prediction
   here.** PTS measured `‖A_k−I‖₂≈0.34` in the behavioral late band. The proposal
   predicts the soft-landing advantage is *largest in rotation-dominated bands* (a
   positive correlation with `‖A_k−I‖`). In the offline scale-free metric we measure
   corr ≈ −0.09 — i.e. soft-landing wins the per-layer-push comparison **everywhere**
   (100% of bands) but **without** the predicted depth trend. We print "essentially no /
   mildly negative rotation trend… not borne out", not a positive trend.
3. **The observer only earns its place under noise.** With a clean, full, noiseless
   readout the Kalman filter degenerates to a pass-through and LQG ties LQR (Exp-4 V→0
   tie, and `PolicyLQG(V→0)==PolicySoftLandingLQR`). The observer's value is *conditional*
   on genuine uncertainty (process noise, injected measurement noise, partial
   observability); we test rather than assume it, and report the tie at zero noise.
4. **Exp-3 coherence comparison is guarded on an effect-match tolerance.** A lower
   KL(steered‖unsteered) at a *weaker* behavioral effect is not a coherence win. The
   prototype only declares a matched-effect comparison valid when the soft-landing's
   effect reaches within 25% of the deadbeat's swing off baseline; otherwise it prints
   **"effect not matched"** and treats the KL gap as inconclusive (also evidence the
   soft-landing band lacks the authority to match the single-layer slam at that `kT`).
   In the smoke run the effect is not matched and is reported as such.
5. **Exp-3 deadbeat must be charged against the *terminal-layer* state.** The
   single-layer deadbeat acts at `kT` (‖x‖~11), not at the band start (‖x‖~3): its
   incoming state is the natural coordinate autonomously propagated through the band to
   `kT`. The reported `max‖u‖` and robustness use this consistent physical state (a
   ≈25% correction over a naive band-start roll, and the depth-dependent bias it removed
   is what makes the Exp-3d trend honest).
6. **Robustness is a fair, two-sided comparison.** Both controllers are exposed to the
   same plant-model error: the soft-landing's gains are re-solved on the perturbed
   `(Â, b̂)` and the deadbeat's commanded angle is derived from the perturbed plant's
   pre-image at `kT`. A plant-free deadbeat (constant angle, variance 0 by construction)
   would not test robustness at all.

## What is and isn't implemented

Implemented and validated: `OAS_SPEC §1` finite-horizon terminal-cost LQR + condensed-QP
gold cross-check + deadbeat corner; `§2` Kalman filter (Joseph form), EKF on the circle,
process-noise estimation, observability matrix / time-invariant + time-varying Gramian,
DARE steady-state gain; `§3` the actuator hook + the five policies incl. the in-pass LQG
filter bank; `§4` the full model-free offline suite (Exp-2/3/4 with the honesty guards
above); `§5` the model-in-the-loop prototype (Exp-1 gate through Exp-4 observer) with a
`--smoke` mode; `§6` both figure sets.

Out of scope for this pass (documented, not silently dropped): the standalone **Exp-5**
norm-preserving-tax sweep as its own experiment (the tax is flagged and the additive
vs. angle-only gap is structural, but not swept as a dedicated panel); the **token-domain**
slow Kalman filter over generation steps (the §6/§4.3-style extension, gated on a
sustained attribute clearing Exp-1); a full PPL/capability axis for Exp-3 beyond the KL
coherence proxy + qualitative generations; external baselines (A-LQR full-`d`, PID-AcT,
ODESteer) — only the in-repo deadbeat and the soft-landing/LQG analogues are compared;
multi-model generalisation.
