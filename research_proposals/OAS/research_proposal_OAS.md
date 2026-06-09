# Observer-Based, Soft-Landing Angular Steering (OAS): Depth-Domain LQG Control of LLM Behavior

## 0. One-paragraph summary

Angular Steering forces the in-plane component of an activation, inside a fixed 2D plane `{b1, b2}`, to a target angle `θ` in **one deadbeat step at one layer**. Two pathologies follow: (i) a single large reset is an aggressive, fragile intervention that fights the model's own layer dynamics and degrades coherence; (ii) it commits to a single fixed `θ` with no estimate of where the activation is actually heading. We propose **Observer-Based, Soft-Landing Angular Steering (OAS)** — a recasting of angular steering as **finite-horizon LQG control over network depth**. Two pillars, each fixing one CLAS failure mode:

1. **Soft landing (multi-step, terminal target).** Instead of slamming the angle to target at one layer, an LQR with a **terminal-angle cost** and a **control-effort cost** emits a *smooth, bounded sequence of small rotations across a band of layers* that lands the activation at the target angle only by a chosen late layer — not at the first. The deadbeat single-layer reset is the degenerate `R→0`, one-layer special case. Hypothesis: distributing the rotation is *more stable* (better coherence, more robust to plant-model error, and it cooperates with rather than fights the natural dynamics).
2. **Observer (Kalman + LQR = LQG).** Rather than relying on one weak, possibly-circular direct measurement of behavior (the rock CLAS foundered on), we model a **latent behavioral state** observed *noisily and partially* through the per-layer geometric angle, and a **Kalman filter fuses the whole trajectory of weak measurements** into a confident state estimate that the LQR controls. The **separation principle** lets us design the estimator and the controller independently.

OAS is grounded in what the sibling work already established: PTS validated that the per-layer in-plane dynamics are an excellent **2×2 affine** model (held-out R²≈0.999), so the linear plant the Kalman filter and LQR need *exists and is cheap* (no Jacobians). OAS is differentiated from A-LQR (full-`d` additive LQR, **full-state, no observer**), PTS (MPC, **full-state, no observer**, trajectory tracking), SO2 (energy state-feedback), and CLAS (output feedback, dropped) by **two things no prior method has: a state observer under explicit partial-observability, and a terminal-target / deadbeat-vs-distributed analysis of the rotation actuator.** We are explicit up front about the threats this inherits — the CLAS **authority** lesson (a gate, not an assumption) and PTS's **near-identity late band** (which limits where idea 1 can pay off) — and make both the first experiments.

---

## 1. Problem Statement & Gap

### 1.1 Where we are, after CLAS

The repo now contains three depth-domain control proposals over the same 2D plane — SO2 (energy/Lyapunov state-feedback), PTS (predictive MPC, *implemented*), A-LQR (the published full-`d` LQR) — plus the dropped CLAS (output-feedback over tokens). The post-mortem on CLAS gave two hard lessons that this proposal is built to respect:

- **L1 — the actuator is a deadbeat reset.** The verified hook (`archive/pytorch_pure/clas_controller.py`, `pytorch_pure/utils.py`) computes `h_out = h − Ph + ‖Ph‖·(cosθ·b1 + sinθ·b2)`: it discards the incoming in-plane angle and *sets* it to `θ` in one step. That is the most aggressive controller there is — full-gain, all-poles-at-zero, at a single layer.
- **L2 — direct behavioral observation is weak.** A single-layer/single-token readout of behavior is noisy, often saturated, and for the first-token case nearly an affine function of the internal angle (circular). And the angular actuator has authority only for *dominant* features.

### 1.2 The gap OAS fills

Every prior depth-domain controller here is **full-state and single-objective**: A-LQR and PTS both assume the state they regulate is directly and cleanly measured (`z_k` / `c_k`), and they regulate it at *every* step (per-layer setpoint or full-trajectory tracking). Neither asks two questions that follow directly from L1 and L2:

1. **(L1 →) Must we hit the target angle at the steered layer at all?** Or can a *distributed, bounded* controller let the natural dynamics do most of the work and only *land* the angle at the target by a late layer — trading a single violent reset for many gentle nudges, gaining coherence and robustness? This is **deadbeat vs. LQR**, and **terminal-cost vs. tracking**, applied to the rotation actuator. No proposal here studies it.
2. **(L2 →) Can we avoid needing a strong direct measurement at all** by *fusing* the many weak per-layer (and per-token) geometric observations into an estimate of a latent behavioral state — i.e. a **Kalman observer**, the optimal version of the ad-hoc EMA the CLAS drift-test resorted to? No proposal here has an observer; all assume full observability.

### 1.3 Positioning

| | Angular Steering | A-LQR | PTS (MPC) | SO2 (energy) | CLAS (dropped) | **OAS (this)** |
|---|---|---|---|---|---|---|
| Domain | depth | depth | depth (+token MPC) | depth | token | **depth** |
| Actuator | deadbeat angle reset, 1 layer | additive `+K v`, all layers | additive→angle, plane | rotation/angle | angle reset, per token | **distributed bounded rotation over a layer band** |
| Objective | fixed angle | per-layer scalar setpoint | track 2D reference *trajectory* | energy descent | output setpoint | **terminal angle target + control cost (soft landing)** |
| State estimate | — | **full-state (measured)** | **full-state (measured `c_k`)** | full-state (φ,ω) | output | **Kalman-estimated latent state (partial obs)** |
| Plant model | none | `d×d` Jacobians | `2×2` affine | optional `F_k` | scalar slope | **`2×2` affine (reuse PTS) + process-noise model** |
| Deadbeat-vs-distributed studied? | no | no | no | no | no | **yes (primary question)** |
| Observer under noise? | no | no | no | no | no | **yes (Kalman; separation principle)** |

**The two load-bearing novelties:** (a) a **state observer** for steering (LQG, not LQR/MPC); (b) a **terminal-target, distributed (soft-landing) rotation** controller and the empirical claim that it beats the deadbeat reset on coherence/robustness. Everything else (the 2D plane, the affine plant, the angle formulation) is borrowed and credited.

---

## 2. The Two Ideas, Formalized

### 2.1 Idea 1 — Soft landing: deadbeat is fragile; distribute the rotation, land at a terminal layer

"Single-step is not stable" is not a claim about asymptotic stability (a forward pass is finite). It is three precise, testable claims about why a one-layer deadbeat reset is a *bad controller*:

- **(a) Coherence / off-manifold.** A deadbeat reset can move the activation by up to the full in-plane diameter `2r` in one layer (when it flips `φ` by ≈π). Even though it is norm-preserving *on the plane*, that is a large, abrupt displacement of the residual stream that downstream layers were not "expecting" — the mechanism behind coherence loss / PPL spikes, and adjacent to the non-surjectivity result (Mishra et al., 2026). Spreading the *same net rotation* over a band of layers keeps every step small and near-manifold.
- **(b) Robustness to plant error.** Deadbeat control (`R→0`, all poles at 0) is the *least* robust LQR to model mismatch. PTS measured the affine-model residual at ≈14% of scale — non-trivial process noise. An LQR with `R>0` trades a little terminal accuracy for robustness to exactly this error.
- **(c) Fighting the natural dynamics.** If we reset the angle early, the model's own layer dynamics (especially the rotation-dominated early/mid layers) rotate it away again, so the reset must be re-asserted — wasted authority spent fighting the network. A controller that *only* requires the target by a late layer lets the natural dynamics carry the activation most of the way and corrects only the residual.

**The control formulation that captures all of this** is a finite-horizon LQR with a **terminal** angle cost and a **control-effort** cost, over a steered band `k = k0 … L`:

```
J = Σ_{k=k0}^{L-1} [ (c_k − c*_k)ᵀ Q_k (c_k − c*_k) + u_kᵀ R u_k ]
        + (c_L − c_target)ᵀ Q_L (c_L − c_target)
```

with **small/zero intermediate `Q_k`** ("don't force the angle early") and **large terminal `Q_L`** ("must land at target by layer L"). Tuning the two knobs recovers the whole spectrum:

- `R → 0`, single layer, `Q_L → ∞` ⇒ **the existing deadbeat reset** (a degenerate special case).
- `R > 0`, band of layers, terminal `Q_L` ⇒ **soft landing** — a smooth, bounded, distributed rotation that arrives at target by layer `L`.

Idea 1's second half ("we only need the angle at mid/last layer") is the **terminal-cost** structure itself, *plus* an empirical question — **which** terminal layer matters — answered by Experiment 2.

> **Actuator caveat, stated honestly (inherited from PTS).** The deployed rotation actuator is norm-preserving: it realizes only the *angle* of the commanded in-plane vector and drops its magnitude. So the cleanest state for OAS is the **angle `φ_k` on the circle** (1-D, with `ω_k` as in SO2), not the full 2D coordinate `c_k` whose magnitude we cannot independently command. We develop the controller in `c_k` for linearity (the affine plant is linear in `c_k`) and **project the commanded `c_k+u_k` back to its angle** at actuation — and we *measure*, not assume, how much the dropped magnitude degrades tracking (Exp. 5). Alternatively we linearize the angular dynamics directly (EKF, §2.2) and control `φ` on SO(2). Both are evaluated.

### 2.2 Idea 2 — Observer: estimate a latent behavioral state by fusing weak measurements (LQG)

CLAS needed *one strong* behavioral measurement and could not get one. OAS instead assumes measurements are **weak, noisy, and partial**, and *fuses* them. Model a latent behavioral state `x_k` (e.g. `[φ_k, ω_k]`, or an augmented vector with a slow "behavioral intent" component) with a linear-Gaussian state-space model:

```
process:      x_{k+1} = A_k x_k + B_k u_k + w_k,     w_k ~ N(0, W_k)   (W_k ← PTS's measured ~14% model residual)
measurement:  z_k     = H_k x_k + v_k,               v_k ~ N(0, V_k)   (per-layer angle read-out, noisy/partial)
```

A **Kalman filter** gives the minimum-variance estimate `x̂_k` by the standard recursion:

```
predict:  x̂_k⁻ = A x̂_{k-1} + B u_{k-1};   P_k⁻ = A P_{k-1} Aᵀ + W
update:   K_k = P_k⁻ Hᵀ (H P_k⁻ Hᵀ + V)⁻¹;  x̂_k = x̂_k⁻ + K_k (z_k − H x̂_k⁻);  P_k = (I − K_k H) P_k⁻
```

The **LQR** of §2.1 then acts on `x̂_k` instead of a raw measurement. By the **separation principle**, the Kalman filter and the LQR are designed independently and combine optimally (LQG).

**Why an observer is not over-engineering here** (the honest version): a Kalman filter degenerates to a pass-through if the state is measured exactly and noiselessly. Its value in OAS is conditional on genuine uncertainty, and we claim three concrete sources, each of which we *test* rather than assume:

1. **Process noise.** The 2×2 affine plant has ≈14% residual (PTS). The filter uses the model where it is confident and the measurement where it is not — provably better than either alone.
2. **Per-token noise (the EMA we already needed).** In the drift test we smoothed the noisy per-token signal with an *ad-hoc* EMA. A Kalman filter is the *optimal* EMA when a state-space model is available — same role, principled gain, with an uncertainty estimate `P_k` for free.
3. **Latent / partial observability.** If behavior is governed by a latent state larger than the 2D angle (the orthogonal-complement and token-context that the plane does not capture), then `z_k = H x_k + v_k` with `dim x > dim z`, and fusing a *sequence* of angle observations recovers `x` where one observation cannot. (Observability of `(A, H)` is checkable from the data; Exp. 4.)

If Exp. 4 shows none of these hold — the angle is a clean, full, noiseless readout — then the observer adds nothing and OAS reduces to depth-LQR (still novel via Idea 1). We say so in advance.

---

## 3. Plant: the depth state-space model (reuse, don't rebuild)

PTS already fit and validated the plant OAS needs: per layer, `c_{k+1} = A_k c_k + b_k` with held-out R²≈0.999 (1-step), 0.989 (5-step), no Jacobians. OAS reuses this directly and adds:

- **The control channel `B_k`.** Because steering is an in-plane displacement, `B_k = I` in plane coordinates (we add `u_k` to `c_k`), exactly as PTS. For the rotational actuator, `u_k` is realized as an angle (the §2.1 caveat).
- **A process-noise covariance `W_k`** estimated from the affine-model residuals (the ≈14% PTS measured) — this is what the Kalman filter consumes.
- **Layer-band structure.** PTS found the behavioral late band (layers ≈20–34, around the steer layer 27) is **near-identity** (`‖A_k − I‖₂ ≈ 0.34`), while early layers are rotation-dominated. This is *central* to OAS and cuts both ways (see Risks): in the near-identity band the natural dynamics neither help much nor fight much, so "distributing vs. deadbeat" may matter *less* there and *more* in the rotation-dominated early/mid band. OAS therefore studies the controller **across depth bands**, predicting idea 1's payoff is largest where `‖A_k − I‖` is largest.

---

## 4. The OAS controller (putting it together)

```
                         ┌─────────────────────── per forward pass (depth k0 … L) ───────────────────────┐
   target angle φ_target │                                                                                │
            │            │   ┌──────────┐    x̂_k    ┌──────────────┐  u_k (small rotation)  ┌──────────┐ │
            └──►(LQR with │   │  KALMAN  ├──────────►│ LQR feedback  ├───────────────────────►│ rotation │ │
              terminal Q_L│   │  filter  │           │ (terminal Q_L,│                        │  actuator │ │ at each
              & cost R)   │   │ fuses z_k│◄──────────│  effort R)    │                        │ at layer k│ │ steered
                          │   └────▲─────┘  z_k      └──────────────┘                        └────┬─────┘ │ layer
                          │        │  (noisy per-layer angle readout)                              │       │
                          │        └────────────────────── activations c_k ◄────────────────────────┘       │
                          └────────────────────────────────────────────────────────────────────────────────┘
   Deadbeat reset = the degenerate corner: single layer, R→0, Q_L→∞, no filter.
```

- **Offline (cheap, batched):** fit `{A_k, b_k}` and residual `W_k` (reuse PTS); pick the steered band and terminal layer `L`; solve the finite-horizon LQR (a backward Riccati recursion — tiny, 2×2) for feedback gains `{K_k}`; design the Kalman gains. All `2×2`. No Jacobians, no online QP.
- **Online (per forward pass):** at each steered layer, read `c_k`, Kalman-update `x̂_k`, compute `u_k = −K_k x̂_k`, actuate the (projected) rotation. Per-layer cost is a handful of `2×2` ops — far cheaper than CLAS's re-prefill loop or A-LQR's `d×d`.
- **Token extension (optional):** run a second, slow Kalman filter over generation steps (the principled EMA) to track and hold behavior across a long output — but only for attributes that clear the authority gate (Exp. 1).

---

## 5. Hypotheses & Experiments (falsifiable; gated)

> **Experiment 1 is a hard gate — run it first, exactly as the CLAS drift test should have been run.**

**Exp. 1 — AUTHORITY GATE (kill-switch).** Does *distributed multi-layer* rotation clear the authority bar that *single-layer* failed? Sweep the steered band width (1 → many layers) and measure the realized behavioral range (`G` sweep) for (a) refusal [known strong] and (b) a sustained attribute [sentiment; known weak single-layer]. *Pass:* multi-layer materially raises authority for the target attribute. *Fail:* if even multi-layer can't move the attribute, OAS — like CLAS — has no actuation authority for it; restrict OAS to the strong-feature (refusal) regime or stop. This directly tests the "multi-layer lever" left open when CLAS was dropped.

**Exp. 2 — Where must the angle be correct? (idea 1b).** Enforce the target angle at a *single* layer and sweep that layer across depth; measure the behavioral outcome. Prediction (from "behavior is read late"): a late terminal layer suffices and early enforcement is washed out by the natural dynamics. Deliverable: behavioral-effect-vs-enforcement-layer curve → justifies the *terminal* (not per-layer) objective and picks `L`.

**Exp. 3 — Soft landing vs. deadbeat (idea 1a, the headline).** At *matched terminal behavioral effect*, compare the single-layer deadbeat reset against the LQR soft-landing controller on **coherence (perplexity / KL to the unsteered distribution)** and **robustness (effect variance under prompt perturbations and under a deliberately perturbed plant model)**. Prediction: soft landing matches the behavioral effect at materially lower coherence cost, and the gap is **largest in the rotation-dominated early/mid band** (smallest in the near-identity late band — the honest hedge from PTS). *Falsified if* deadbeat is on the same coherence–effect Pareto front everywhere.

**Exp. 4 — Does the observer earn its place? (idea 2).** (i) Test observability of `(A, H)` and quantify measurement/process noise from data. (ii) Compare behavioral-state prediction from a **Kalman-fused** estimate vs. the **best single-layer** measurement vs. the **ad-hoc EMA**. (iii) Compare **LQG** (control on `x̂`) vs. **LQR-on-raw-measurement** at matched effort. *Pass:* fusion beats single-measurement and LQG beats raw-feedback (esp. under injected noise / partial observability). *Falsified if* the raw single measurement is already a clean full readout — in which case we report that and drop the filter (OAS → depth-LQR).

**Exp. 5 — The norm-preserving-actuator tax.** Quantify how much tracking degrades when the planned `c_k+u_k` is projected to its angle (magnitude dropped). Compare angle-only actuation vs. an additive control (magnitude kept) as an upper bound. Decides whether to control `φ` on SO(2) directly (EKF) or `c_k` in the plane.

**Baselines (re-used / from siblings):** Angular Steering (deadbeat), A-LQR (full-`d` LQR, no observer), PTS-MPC (full-state), SO2 energy. Metrics: realized behavioral range, refusal/ASR + LlamaGuard (where the eval stack permits), perplexity/KL coherence, effect variance (robustness), and control-engineering metrics (terminal error, total control effort `Σ‖u_k‖`, sensitivity to plant perturbation).

---

## 6. Implementation Plan (mapped to existing code)

| File | Status | Role in OAS |
|---|---|---|
| `pytorch_pure/phase_portrait.py` | exists | reuse `compute_phase_trajectories` for per-layer `φ_k, ω_k, r_k, c_k` (the observations `z_k`) and the `r_k` observability guard |
| `pytorch_pure/pts_dynamics.py` (+ offline) | exists (PTS) | reuse the fitted `2×2` affine `{A_k, b_k}` **and its residuals** → plant + `W_k` for the Kalman filter |
| `archive/pytorch_pure/clas_controller.py` | archived | reuse the **mutable-angle hook** and the **manual KV-cached decode loop** (verified vs `model.generate`) for actuation and the token-domain extension |
| `oas_observer.py` | **new (~120 LOC)** | Kalman filter (and an EKF variant for `φ` on SO(2)); observability test; noise estimation from residuals; all with executable self-tests (mirror PTS's dependency-free style) |
| `oas_lqr.py` | **new (~120 LOC)** | finite-horizon LQR via backward Riccati (`2×2`) with terminal `Q_L` + effort `R`; deadbeat as the `R→0` corner |
| `oas_controller.py` | **new (~100 LOC)** | glue: offline gain solve, online predict/update/actuate over the steered band; LQG assembly |
| `oas_prototype.py` | **new** | Exp. 1 (authority gate) → Exp. 2 (enforcement-layer sweep) → Exp. 3 (soft-landing vs deadbeat) → Exp. 4 (observer value); figures + npz |

**Milestones.** Wk1: **Exp. 1 gate** (multi-layer authority) — go/no-go. Wk1–2: reuse PTS plant + fit `W_k`; Exp. 2 (where the angle must land). Wk2–3: `oas_lqr` + soft-landing controller; **Exp. 3** (the headline coherence/robustness result). Wk3–4: `oas_observer` + LQG; Exp. 4–5. Wk4+: token-domain Kalman extension *iff* Exp. 1 passed for a sustained attribute; writing.

---

## 7. Novelty & Contributions (narrowed, honest)

1. **A state observer for activation steering (LQG, not LQR/MPC).** First steering controller to estimate a *latent, partially-observed* behavioral state with a Kalman filter and control the estimate (separation principle) — distinct from A-LQR/PTS (full-state) and SO2 (full-state energy). The principled replacement for the EMA the CLAS drift test improvised.
2. **Deadbeat-vs-distributed, and terminal-vs-tracking, for the rotation actuator.** First to frame the existing single-layer reset as a degenerate deadbeat corner and show (or refute) that a soft-landing, terminal-target LQR preserves coherence and robustness better — with the depth-band dependence predicted from PTS's spectral profile.
3. **A reusable, dependency-free `2×2` LQG-over-depth controller** that needs no Jacobians and no online QP — cheaper than A-LQR (`d×d`) and PTS (online QP).

**Explicitly not novel** (credited): the 2D plane and rotation actuator (Angular Steering); the `2×2` affine plant (PTS); LQR over depth and the tracking-error idea (A-LQR); the angular/`φ,ω` state and SO(2) framing (SO2); Kalman/LQG/separation principle (classical control).

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Authority** — multi-layer rotation still can't move the target attribute (the CLAS killer) | Medium (Low for refusal) | Fatal | **Exp. 1 is a hard gate, run first.** Scope to strong features if it fails for sustained ones. |
| **Near-identity late band** weakens idea 1 — if natural dynamics barely move/fight the activation near the steer layer, deadbeat ≈ distributed there | Medium | Moderate | Predicted by PTS; **test across depth bands** (Exp. 3); claim the soft-landing win specifically in the rotation-dominated early/mid band, and report honestly where it doesn't help. |
| **Observer over-engineering** — if `c_k` is a clean full noiseless readout, Kalman degenerates | Medium | Moderate | **Exp. 4 decides.** If fusion doesn't beat the single measurement, report it and drop to depth-LQR (still novel via idea 1). No silent inclusion. |
| **Norm-preserving actuator tax** — controller plans `c_k+u_k`, actuator keeps only the angle | High | Moderate | Quantify (Exp. 5); control `φ` on SO(2) via EKF if the magnitude loss is material. |
| **Circle nonlinearity** — Kalman/LQR are linear; `φ` wraps | Certain | Low | Work in linear `c_k` coords (angle as nonlinear readout) **or** EKF/UKF on `φ`; both implemented and compared. |
| **Coherence cost of multi-layer steering** (Vu & Nguyen caveat for 3B models) | Medium | Moderate | The soft-landing's *purpose* is to reduce this; measured directly as the Exp. 3 outcome (perplexity). |
| **Overlap with PTS** | Medium | Moderate | OAS's distinct claims are the **observer** and the **deadbeat-vs-distributed/terminal** analysis; PTS is full-state trajectory-tracking MPC. Keep the comparison head-to-head and concede shared ground (plane, plant). |

---

## 9. References

**Repository papers.**
- Vu, H. M., & Nguyen, T. M. (2025). *Angular Steering: Behavior Control via Rotation in Activation Space.* NeurIPS 2025. (Norm-preserving rotation actuator; multi-layer coherence caveat.)
- Nguyen, D. V., et al. (2026). *Activation Steering with a Feedback Controller* (PID-AcT). ICLR 2026. arXiv:2510.04309.
- Skifstad, J., Yang, X. A., & Chou, G. (2026). *Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control* (A-LQR). arXiv:2604.19018. (LQR over depth, full-state, no observer; tracking-error bound.)

**Sibling proposals.**
- `research_proposal_SO2.md` — energy/Lyapunov state-feedback over depth (`φ_k, ω_k`).
- `research_proposal_PTS.md` (+ `PTS_README.md`) — MPC in the plane; the **validated `2×2` affine plant**, the **near-identity late band**, and the **norm-preserving-actuator discrepancy** OAS builds on.
- `archive/research_proposal_CLAS.md` — the dropped output-feedback attempt; its post-mortem (L1/L2) motivates OAS.

**Control theory.**
- Kalman, R. E. (1960). *A New Approach to Linear Filtering and Prediction Problems.* J. Basic Eng. (The filter.)
- Anderson, B. D. O., & Moore, J. B. (1979). *Optimal Filtering.* Prentice Hall. (Kalman/LQG.)
- Åström, K. J., & Murray, R. M. (2008). *Feedback Systems.* Princeton. (LQG, separation principle.)
- Franklin, G. F., Powell, J. D., & Workman, M. (1998). *Digital Control of Dynamic Systems.* Addison-Wesley. (Finite-horizon LQR, deadbeat control, Riccati recursion.)
- Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed.). Prentice Hall. (EKF, observability.)
