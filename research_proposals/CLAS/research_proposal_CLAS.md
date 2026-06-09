# Closed-Loop Angular Steering (CLAS): Output-Feedback Control of LLM Behavior by Rotation on SO(2)

> **STATUS: DROPPED (2026-06-09).** A cheap falsification ("drift test") killed the pivot before a full build, after the original refusal framing also failed (one-shot + saturated + circular observable). Decisive findings:
> 1. **Refusal** — the one attribute with strong angular-steering authority (~18 margin range, flips behavior) — is a **one-shot** decision (decided at the first token, already saturated at 100% on Qwen2.5-3B), so a per-token feedback loop has no job; the decision is better set feedforward via the measured `G(θ)`.
> 2. **Sustained** attributes that *would* justify a token loop (sentiment) have **negligible** angular-steering authority: sweep range only ~1.6, and steered vs unsteered 200-token generations came out **byte-identical**. Mechanism: the norm-preserving rotation only re-points the activation's *in-plane* component, and little of a subtle/distributed attribute lives in a single mid-layer 2D plane (vs. the dominant, safety-trained refusal feature).
>
> Net: the angular actuator can steer **dominant one-shot** features or leave **sustained** features unmoved — but not a sustained feature it can actually move — so CLAS's closed-loop *output-feedback thermostat* has no demonstrable niche **on this actuator**. Scope of the negative result: single-layer angular reset, auto-selected layer, instruction-position directions, Qwen2.5-3B. Effort consolidated on `research_proposal_SO2.md` and `research_proposal_PTS.md`. Reusable parts (mutable-angle hook, output observable, manual KV-cached decode loop) are archived alongside. See `archive/README.md`. The proposal text below is preserved unchanged for the record.

## 0. One-paragraph summary

Angular Steering controls an LLM by forcing the in-plane component of an activation, inside a fixed 2D plane `{b1, b2}`, to a *fixed absolute angle* `theta`. This is **open-loop**: the angle is chosen by an offline sweep and never reacts to what the model actually emits. We propose **Closed-Loop Angular Steering (CLAS)**, an *output-feedback thermostat* whose only actuator is the SO(2) steering angle. Once per generated token, CLAS measures a **behavioral observable read off the model's own output distribution** — the controlled variable we actually care about — forms an error against a behavioral setpoint, and adjusts the commanded steering angle with integral disturbance rejection and anti-windup. CLAS is differentiated from every prior method along **one sharp axis: what is measured and fed back.** Angular Steering measures nothing; SO2, PTS, A-LQR, and PID-AcT all feed back an *internal activation coordinate* (an angle, a 2D point, a feature strength, or a difference-in-means); **none closes a loop on a measured behavioral output of the model.** A second, smaller differentiator is *reactive PI* on that output versus PTS's *predictive MPC* on an internal trajectory. We are explicit about what is **not** novel: the cascaded token/layer hierarchy (PTS already proposes it), the SO(2) angular formulation and wrapped-angle error (SO2), the norm-preserving rotation actuator (Angular Steering), and the PI-integral steady-state-error argument (PID-AcT). We are also explicit about a hard empirical constraint our own pilot data imposes: on Qwen2.5-3B at the single steered layer the repo uses, the refusal-vs-angle map is **unimodal, not monotone**, and the headline refusal task is **saturated** (baseline harmful refusal already 100%). The proposal therefore (i) confines the controller to an empirically identified monotone operating band with a sign-guarded integrator, (ii) replaces the saturated binary endpoint with a continuous, real-valued behavioral margin that has dynamic range, and (iii) honestly scopes the contribution to **output-feedback regulation** rather than overclaiming a multi-loop cascade or universal stability. We give the corrected control formulation, a corrected stability analysis, and a falsifiable plan with all four prior methods as baselines, mapped onto the *actual* code in `pytorch_pure/utils.py`, `phase_portrait.py`, and `steering_validation.ipynb`.

---

## 1. Problem Statement & Gap

### 1.1 The object of control (as the code actually implements it)

Following Angular Steering (Vu & Nguyen, NeurIPS 2025), fix a behavior (e.g. refusal). Extract a feature direction by difference-in-means over contrastive data and a second axis, Gram–Schmidt them into an orthonormal basis `{b1, b2}`, and form the plane projector `P = b1 b1^T + b2 b2^T`. Any activation `h` decomposes as `h = P h + (I-P) h`, with in-plane coordinates `c(h) = [b1^T h, b2^T h]^T ∈ R^2`, magnitude `r = ||c(h)||`, and angle `phi = atan2(b2^T h, b1^T h)`.

**The actuator is an absolute-angle reset, not an incremental rotation.** The verified hook in `utils.py` (`get_angular_steering_output_hook`, mirrored by `make_angular_steering_hook` in `steering_validation.ipynb`) computes

```
projected = (b1^T h) b1 + (b2^T h) b2,   scale = ||projected||,   steer = cos(theta) b1 + sin(theta) b2,
h_out = h - projected + scale * steer.
```

This **discards the incoming in-plane angle `phi` entirely** and forces the in-plane component to the *absolute* angle `theta`, preserving its magnitude `scale`. Formally, the post-hook in-plane angle is `phi_out = theta` for **any** input `phi`. It is *not* the relative rotation `phi_out = phi + u`. This is norm-preserving on the plane (`||h_out|| = ||h||`), which is the real advantage of the rotation family: it cannot inflate the activation norm the way unbounded additive steering can (Mishra et al., 2026, non-surjectivity). Our control authority is the scalar (per-token) **commanded angle**; we never add a vector and we never compute Jacobians.

> *Correction relative to an earlier draft and to the differentiation literature:* the actuator is a deadbeat **set-point assignment** of the in-plane angle, so any control law that "rotates by the error amount and integrates the residual" within a single layer is vacuous — the reset erases `phi` in one shot. This single fact reshapes the entire design (Sec. 2): there is no intra-layer drift for an inner loop to integrate away, so CLAS is a **single-loop output-feedback controller over tokens**, not a depth-integrating cascade.

### 1.2 The gap

Angular Steering picks one global `theta`, chosen by a sweep, fixed before generation, and **never reacts to the model's behavior**. Two failures follow:

1. **No behavioral disturbance rejection across tokens.** Behavior drifts over a long generation; the prompt and the partial completion act as a time-varying disturbance. A fixed angle over-steers easy prompts (coherence loss, gibberish at large effective rotation on small models) and under-steers hard ones (jailbreak leakage). Nothing closes the loop on the realized behavior.
2. **No principled setpoint in behavior-space.** The operator tunes an internal angle, not the thing they care about (a refusal margin). There is no mechanism to *hold* a behavioral target against disturbance.

### 1.3 What "closed-loop feedback control" means here, and exactly where each prior method falls short

A genuine feedback controller must: **(M)** measure an observable online; **(E)** form an error against a setpoint; **(F)** feed that error back to adjust the action. We additionally require, for *behavioral* control, that the measured observable be a function of the **model's realized output distribution** — the thing we want to regulate — not merely an internal state we hope correlates with it. This is the distinction between *state feedback* (measure an internal `z_k`) and *output feedback* (measure `y_t = h(Unembed(z_L))`, a behavioral readout). The four prior methods:

- **Angular Steering** (Vu & Nguyen, 2025): no M, no E, no F — pure open-loop absolute-angle assignment.
- **SO2 energy-damping proposal** (`research_proposal_SO2.md`): has M/E/F, but the loop is **state feedback over depth**. It already uses the activation **angle** `phi_k = atan2(b2^T h, b1^T h)` as the controlled variable, the **wrapped-angle / SO(2)-Log error**, and damps an internal energy `V(phi, omega)`. The controlled variable is an internal coordinate.
- **PTS / MPC proposal** (`research_proposal_PTS.md`): has M/E/F and is **predictive trajectory tracking** of the *internal* 2D coordinate `c_k` against a reference *trajectory*, using a fitted `2x2` plant and a receding-horizon QP. **Crucially, PTS Sec. 6.3 ("Two-Level MPC") already proposes a token-level outer loop wrapping a layer-level inner loop, "analogous to hierarchical MPC,"** for long-form drift. So the cascade/hierarchy idea is *prior art*, not ours. PTS's outer loop re-measures the internal `c_t` and re-plans an MPC reference.
- **PID-AcT** (Nguyen et al., ICLR 2026; arXiv:2510.04309): P/PI/PID **over the layer index** of the difference-in-means `r(k)`, additive in full `d`-dim space. Its base variant feeds back an offline reference; we explicitly **concede** that its **Mean-AcT** variant *does* re-measure the running internal means online (recomputed at each layer accounting for prior interventions). So the honest dividing line is **not** "online vs offline" — it is **internal-state vs output**: even Mean-AcT measures internal means, never the output distribution.
- **A-LQR** (Skifstad et al., 2026; arXiv:2604.19018): genuinely closed-loop — `u_k* = (beta*_k - v_k^T z_k) K_k v_k` re-measures the live activation `z_k` and scales the perturbation by the live feature-strength error. But it is **full-`d` additive LQR over depth** with a **scalar internal setpoint** `beta*_k` (a projection), and the observable is an internal feature strength, not an output behavior. The actuator is `+K_k v_k`, not a rotation.

**The CLAS thesis, narrowed to what survives scrutiny:** *No existing method closes a loop on a measured behavioral **output** of the model (a readout of `Unembed(z_L)`); they all regulate an internal coordinate, feature strength, or difference-in-means.* CLAS does exactly that, with the SO(2) angle as the manipulated variable and a reactive PI law (vs PTS's predictive MPC). The inner/cascade structure, the SO(2) formulation, the rotation actuator, and the PI steady-state-error result are **not** claimed as novel.

### 1.4 Positioning table

| | Angular Steering | PID-AcT (+Mean-AcT) | A-LQR | SO2 (energy) | PTS (MPC) | **CLAS (ours)** |
|---|---|---|---|---|---|---|
| Manipulated variable (actuator) | absolute angle `theta` | additive vec in `R^d` | additive vec in `R^d` | rotation/angle | angle (via `c∈R^2`) | **absolute steering angle `theta`** |
| Loop topology | open-loop | feedforward / online over depth | state-feedback over depth | state-feedback over depth | predictive depth loop **+ token-level two-level MPC (Sec. 6.3)** | **single token-level output-feedback loop** |
| What is measured (M) | nothing | diff-means `r(k)` (offline) / running means (Mean-AcT) | live activation `z_k` | live angle `phi_k`, `omega_k` | live coord `c_k` | **behavioral output `y_t` = f(Unembed(z_L))** |
| **Observable type** | — | **internal diff-means / means** | **internal feature strength** | **internal angle** | **internal 2D coord** | **realized OUTPUT behavior** |
| Setpoint | fixed angle | diff-means ref | scalar `beta*_k` | target angle `phi*` | reference *trajectory* | **behavioral setpoint `y*` (one knob)** |
| Disturbance rejection | none | I-term over depth | LQR (no integral) | damping (no integral) | constraint/penalty | **integral over tokens (sign-guarded)** |
| Anticipative? | no | no | no | no | yes (horizon `H`) | **no (reactive PI)** |
| Plant model needed | no | no | `d×d` Jacobians | optional `F_k` fit | `2×2` per layer | **scalar slope `G'` of the output map (estimated online)** |
| Coherence handling | none | none | none | barrier/separatrix | `‖u‖≤u_max` | **angle clamp + norm-preservation (empirical PPL guardrail)** |
| Guarantee offered | none | depth steady-state-error removed | LFS tracking bound | energy monotonicity | trajectory tracking | **token-loop zero steady-state error *within a monotone band*** |

**The single decisive row is "Observable type": realized OUTPUT behavior (CLAS) vs. internal coordinate / feature / diff-means (every other method).** The "Loop topology" and "Manipulated variable" rows are shared ground (PTS is also cascaded; SO2 also uses the angle), not differentiators, and we no longer claim otherwise.

---

## 2. Control Formulation

CLAS is a **single output-feedback loop over tokens**. We deliberately drop the depth-integrating "inner loop" of the earlier draft, because (a) the actuator is an absolute-angle reset (Sec. 1.1), so there is no intra-layer residual to integrate, and (b) the validated pipeline steers at a **single** layer (`STEER_LAYER = 27` of 36 in `steering_validation.ipynb`), so there is no multi-layer depth loop to close. Sec. 2.5 discusses an *optional* multi-layer variant and states honestly what it would require.

Notation: `t` indexes generated tokens; angles live on `S^1`; angular subtraction is the wrapped difference `wrap(a) = atan2(sin a, cos a) ∈ (-pi, pi]`.

### 2.1 The behavioral plant and its output observable

The variable we regulate is the model's **behavior at the output**. Define a scalar **behavioral observable** read from the realized output at token `t`. We require it to be **real-valued, low-latency, and have dynamic range** (the saturated binary refusal label does not qualify — see Sec. 5):

- **Refusal/compliance logit margin (primary, continuous):**
  `y_t = logsumexp_{j∈R} ℓ_t[j] − logsumexp_{j∈C} ℓ_t[j]`,
  where `ℓ_t` are the next-token logits the model emits at step `t`, `R` is a refusal first-token id set (e.g. `{"I", "Sorry", "Cannot", "As"}` continuations) and `C` a compliance set (e.g. `{"Sure", "Here", "Step", "1"}`). This is a direct readout of `Unembed(z_L)` and is **continuous even where the binary refusal label is pinned at the rail**, restoring dynamic range.
- **Multi-step behavioral observable (primary for the headline claim):** an EMA over the running completion of a refusal/compliance classifier applied to the *decoded text so far*. This **cannot be reduced to a single-layer angle** and is the observable we use to defend H2 against the circularity objection (Sec. 5.1).
- **Linear behavioral probe (control):** `y_t = w_probe^T z_L^{(t)} + b`, a probe **trained from scratch** on held-out data (it does **not** exist in the repo; §4 is only a substring matcher). Treated as an explicitly-circular easy case.

Crucially `y_t` is a function of the *output* `ℓ_t = Unembed(z_L)`, i.e. an output-feedback signal in the precise sense of the state-space template `(x' = g(x,u), y = h(x,u))`. The outer plant is the unknown, nonlinear map from the commanded steering angle to this behavioral readout. **Because the angle commanded during token `t`'s forward pass must be chosen *before* the logits of step `t` exist, the loop has an unavoidable unit transport delay:**

```
y_t = G(theta_{t-1}) + d_t,                                            (A1, delayed)
```

where `theta_{t-1}` is the angle held during the forward pass that produced `ℓ_t`, `G` is the behavioral map, and `d_t` is the behavioral disturbance from prompt + partial completion. The delay is structural and is carried through the stability analysis (Sec. 4.2).

### 2.2 The empirical behavioral map `G` and its operating band (measured, not assumed)

We refuse to assume `G` is monotone. The repo's own full-circle sweep (`steering_validation.ipynb` cell 21, Qwen2.5-3B, layer 27) gives the realized refusal-vs-angle curve:

| `theta` (deg) | 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 |
|---|---|---|---|---|---|---|---|---|
| harmful refusal | 0% | 0% | 0% | 87.5% | 93.8% | **100%** | 93.8% | 75% |
| harmless refusal | 0% | 0% | 0% | 0% | 12.5% | 12.5% | 6.2% | 0% |

This is **unimodal/periodic on `S^1`** (flat near zero for `theta∈[0,90]`, peaking near `theta≈225`), **not globally monotone** — on a circle it cannot be. It is also *opposite in sign convention* to the label in `compute_steering_plane` (which names `theta=0` the "safe pole," yet `theta=0` yields 0% refusal on harmful prompts). CLAS therefore **first measures `G(theta)`** on the chosen model/layer/observable, then:

1. **Identifies a monotone operating band** `[theta_lo, theta_hi]` (here roughly the rising flank `theta∈[90,225]`) on which `0 < gamma_lo ≤ G'(theta) ≤ gamma_hi`, and reports `gamma_lo, gamma_hi`, the band edges, and the peak location.
2. **Sets the nominal angle `theta_nom`** strictly inside the band and on the correct side of the peak.
3. **Clamps the commanded angle to the band** (`sat_band`), not merely to a magnitude — clamping alone does not prevent crossing the peak unless the band itself excludes it.

This makes the monotonicity assumption (I1, Sec. 4.1) an **empirically verified, band-local** fact rather than a global assertion.

### 2.3 Error, thermostat law, and sign-guarded integrator

**Setpoint.** A behavioral target `y*` (e.g. "logit margin = +m, comfortably refusing"). This is a *single interpretable knob* the operator sets — unlike a per-layer scalar `beta*_k` (A-LQR) or a full reference trajectory (PTS).

**Error and PI thermostat over tokens (with anti-windup and a sign guard):**

```
epsilon_t = y* − y_t                                                   (output error)
ĝ_t       = online slope estimate of G' from recent (theta, y) pairs   (sign/gain monitor)
sigma_t   = clamp( sigma_{t-1} + epsilon_t,  −Sig_max, +Sig_max )       (integrator, anti-windup)
theta_t   = sat_band( theta_nom + Kp_out * epsilon_t + Ki_out * sigma_t )   (B1)
```

The integrator gives **zero steady-state behavioral error for constant disturbances** *within the monotone band*: persistent disturbance (a stubborn jailbreak prompt) is rejected by accumulating `theta_t` until `y_t = y*`. The **sign guard** addresses the unimodality directly: `ĝ_t` is estimated online (regression of `y` on `theta` over a short window); if `|ĝ_t|` falls below a floor or `ĝ_t` changes sign (the loop is approaching the peak / a flat region), CLAS **freezes the integrator and relaxes toward `theta_nom`**, because plain clamp anti-windup does **not** prevent a sign-flip-driven runaway. `sat_band` keeps operation inside the monotone region by construction.

### 2.4 Why a thermostat, and why this is genuinely output feedback

Like a thermostat measuring *room temperature* (the output we care about) rather than *furnace valve position* (an internal state), CLAS measures behavior, not the in-plane angle. This is the distinction from SO2 (measures `phi_k`), A-LQR (feature strength), PTS (`c_k`), and PID-AcT/Mean-AcT (diff-means/means). The actuator happens to be the same SO(2) machinery, and the loop is a textbook discrete PI — neither of which we claim as novel. **The novelty is the observable.**

### 2.5 Optional multi-layer variant (stated honestly, not the headline)

If one wanted a layer-level component, the *only* defensible version given the absolute-reset actuator is **deadbeat set-point assignment over a band of steered layers** (not integral drift rejection): command each steered layer's absolute angle so that, after downstream layers re-drift (measured as `delta_k = mean(omega_k)` by `phase_portrait.py`), the **end-of-stack** in-plane angle hits a target. This requires:

- Steering at a **band** of layers, which Vu & Nguyen report can break coherence on 3B models (PPL spikes) unless gated — so multi-layer steering is a *research risk*, not a free lunch.
- An **observability guard**: `phase_portrait.py` already returns the in-plane magnitude `r_k`; where `r_k` is small, `phi_k = atan2(c2, c1)` is the angle of a near-zero 2-vector and is numerically meaningless. The observer must gate/skip layers with `r_k` below a threshold.
- Reusing a **single-layer-fit `{b1,b2}`** across the band (cheap but may capture little variance off the selected layer) **or** fitting per-layer planes (cost).

We **do not** include this in the headline claim. The headline CLAS is single-layer output feedback over tokens. The multi-layer deadbeat variant is an ablation (Sec. 5.4), gated on whether residual end-of-stack drift after single-layer steering is non-negligible *and* `r_k` is large enough for `phi_k` to be well-defined.

---

## 3. The Closed-Loop Architecture

```
                       OUTPUT-FEEDBACK (TOKEN) LOOP — THERMOSTAT, one update per generated token t
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                                                                                      │
  │   y*  ──►(+)──► epsilon_t ─► [ PI + anti-windup + SIGN GUARD ]  ─►  theta_t  (commanded angle, in    │
  │          ▲ −                 Kp_out, Ki_out, sat_band, ĝ_t                    monotone band)          │
  │          │                                                          │                                 │
  │          │ y_t  (MEASURED OUTPUT BEHAVIOR; available only AFTER     │  held during NEXT forward pass  │
  │          │       the forward pass for token t  →  unit delay)       ▼   (transport delay, Eq. A1)     │
  │          │                                       ┌───────────────────────────────────────────────┐   │
  │  behavioral_readout( logits_t )  ◄─────────────  │  LLM forward pass for token t (frozen weights)  │   │
  │  = logit margin / decoded-text EMA / probe       │                                                 │   │
  │          ▲                                       │   z_1 ─►f_1─► ... ─► z_{STEER} ─► ... ─► z_L     │   │
  │          │                                       │                       ▲                          │   │
  │   logits_t = Unembed(z_L)                        │                       │  ACTUATOR (utils.py hook)│   │
  │                                                  │            h_out = h − Ph + ||Ph||·              │   │
  │                                                  │              (cos theta_{t-1} b1 + sin · b2)     │   │
  │                                                  │            ABSOLUTE-ANGLE RESET: phi_out=theta   │   │
  │                                                  └───────────────────────────────────────────────┘   │
  │                                                                                                      │
  │   ONE integrator: sigma_t (rejects token-level behavioral disturbance d_t, sign-guarded).            │
  │   ONE measurement: y_t (REALIZED OUTPUT behavior).   ONE actuator: the absolute steering angle.      │
  └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Reading the diagram: during token `t`'s forward pass, the existing `utils.py` hook resets the in-plane angle at the steered layer to `theta_{t-1}` (the value the controller computed from the *previous* token's behavioral readout). After the pass, CLAS reads `y_t` off the emitted logits, forms `epsilon_t`, updates the sign-guarded integrator, and computes `theta_t` for the next pass. The operator sets only `y*`.

---

## 4. Stability & Robustness Analysis

We give *informal* results and are explicit that this is **standard discrete PI / singular-perturbation theory applied to a new output-feedback interconnection** — the only piece keyed to the novel observable is the outer steady-state-error result. We do **not** claim a depth-loop theorem (there is no depth loop), and we have **removed** the earlier draft's incorrect inner-loop "Theorem 1" entirely (its claimed closed-loop matrix had a structural eigenvalue of exactly 1 for all gains, so it could never be exponentially stable).

### 4.1 Modelling assumptions (idealizations, stated honestly)

- **(I1) Band-local monotonicity.** `G(theta)` is monotone with bounded slope `0 < gamma_lo ≤ G'(theta) ≤ gamma_hi` **only on the identified operating band** `[theta_lo, theta_hi]`. *This is measured first (Sec. 2.2), not assumed*, and the controller is confined to the band by `sat_band`. **Idealization:** outside the band `G'` flips sign (unimodality); handled by `sat_band` + the sign guard, not by clamp anti-windup alone.
- **(I2) Bounded, slowly-varying disturbance.** `|d_t| ≤ Δ`. **Idealization:** `d_t` can jump at a hard prompt boundary (e.g. an adversarial trigger token); the integrator then needs a few tokens to recover (quantified in Exp. H4, including a step-disturbance test).
- **(I3) Adequate authority within the band.** `theta_lo, theta_hi` admit the `theta_*` that achieves `y*`; otherwise anti-windup prevents windup but tracking degrades gracefully (and we report it).
- **(I4) Saturation/headroom.** The chosen observable has dynamic range at the operating point. *This is the binding constraint for the refusal task* (baseline harmful refusal is already 100%; see Sec. 5), and is the reason we use the continuous margin / multi-step observable rather than the saturated binary.

### 4.2 Token loop — zero steady-state error within the band, with transport delay

**Informal Theorem (delayed thermostat tracking).** On the monotone band, linearize `G` to slope `gamma ∈ [gamma_lo, gamma_hi]`. With the unit transport delay (A1), the closed-loop output is `y_t = gamma·theta_{t-1} + d_t`. Substituting the PI law (B1) (and writing `epsilon_t = y* − y_t`) yields the **delay-augmented** error recursion

```
epsilon_t = (1 − gamma Kp_out) epsilon_{t-1} − gamma Ki_out sigma_{t-1} + (d* − d_t),
sigma_{t-1} = sigma_{t-2} + epsilon_{t-1},
```

equivalently a second-order recursion in `epsilon` whose characteristic polynomial is `z^2 − (2 − gamma Kp_out − gamma Ki_out) z + (1 − gamma Kp_out)`. **Both roots lie inside the unit disk** under the Jury conditions (sufficient form: `0 < gamma Ki_out < gamma Kp_out < 2` and `gamma(2 Kp_out + Ki_out) < 4`, i.e. gains tuned against `gamma_hi` **with a margin reserved for the one-step delay**), giving `epsilon_t → 0` and, by the integrator, **zero steady-state error for constant `d_t`**. For slowly-varying `d_t`, ISS holds: `limsup_t |y* − y_t| ≤ c · sup_t |d_t − d_{t-1}|`. **We state plainly that the unit delay shrinks the stable gain region relative to a delay-free PI** (the earlier draft's delay-free recursion was incorrect); we tune `Kp_out, Ki_out` against both `gamma_hi` and the delay, and verify the margin empirically (Exp. 6) rather than claim it universally. *Contrast:* integral-free baselines (Angular Steering, A-LQR) retain a nonzero offset against persistent behavioral disturbance; this is the standard PI steady-state-error argument (PID-AcT Prop. 1), here re-used **on a behavioral output** rather than on diff-means.

### 4.3 Why a single loop suffices, and the singular-perturbation note (NOT a small-gain claim)

We **withdraw** the earlier draft's `H_∞` "small-gain Theorem 3," which was vacuous: it simultaneously asserted `||T_in||≈0` *and* that the small-gain product certified robustness — contradictory, and merely the timescale-separation argument in disguise. The honest statement is a **singular-perturbation / timescale-separation** argument: the LLM forward pass completes within one token step, so from the token loop's perspective the map `theta → y` is a static (delayed) gain `G`. With a *single* steered layer there is exactly one actuator update per token; there is no second integrator and hence no "two integrators fighting." If the optional multi-layer deadbeat variant (Sec. 2.5) is used, the layer-level set-points are feedforward *within* the pass (they settle trivially before the token loop acts), so the token loop still sees a static map — that is the boundary-layer condition.

### 4.4 Coherence: norm-preservation, not a spurious "certificate"

Absolute-angle assignment is **exactly norm-preserving on the plane** (`||h_out|| = ||h||`), which avoids the scale-blowup failure mode of unbounded additive steering. We explicitly **do not** claim that bounding the displacement norm bounds coherence: a small-norm but adversarially-directed in-plane move can still degrade downstream behavior (indeed the absolute reset can move the activation by up to the full in-plane diameter `2r` when it flips `phi` by ≈π, e.g. the `theta≈180–225` regime). We also retract the earlier claim that additive methods "cannot" bound their perturbation — PTS's `‖u‖≤u_max` and clipping `‖K_k α_k v_k‖` are such bounds. **The genuine rotation advantage is norm-preservation, full stop.** Coherence is measured empirically by perplexity (Sec. 5), and `sat_band` keeps operation in the validated band.

---

## 5. Experimental Plan

### 5.1 Falsifiable hypotheses (re-scoped to the real dynamic range and the circularity risk)

- **H1 (output feedback holds a behavioral setpoint where open-loop drifts).** Over long generations, CLAS keeps the continuous behavioral observable `y_t` within a band around `y*`; fixed-angle Angular Steering drifts. *Falsified if* CLAS's settling band is no tighter than fixed-angle's drift band.
- **H2 (output feedback beats state feedback for behavior — tested where they DECOUPLE).** The circularity risk is real: for the *single-token logit margin*, `y_t` is nearly affine in `phi_L`, so "output feedback" ≈ "state feedback on the last-layer angle," and the decisive ablation could be a wash. We therefore **pre-register the decoupling regime**: the **multi-step decoded-text observable** (Sec. 2.1), which is a function of *many* emitted tokens and cannot be reduced to a single-layer angle. *Claim, restricted to this regime:* CLAS reaches a behavioral target at lower mean `|theta − theta_nom|` (hence lower ΔPPL) than any internal-coordinate controller at matched behavior. *Falsified if* internal-only control matches CLAS on the effectiveness–coherence Pareto front even on the multi-step observable. The single-token-margin case is reported as the easy-but-circular control.
- **H3 (zero steady-state behavioral error, in the band, on a non-saturated observable).** Under a sustained adversarial prompt, CLAS's integrator drives the continuous margin `y_t` to within `δ` of `y*` and holds it; integral-free baselines retain a persistent offset. *Measured on the continuous margin, not the binary refusal label*, because the binary is pinned at the rail (baseline harmful refusal = 100%, so the binary has ~6 points of range at this layer — see Sec. 5.3). *Falsified if* offsets are statistically indistinguishable.
- **H4 (recovery from a behavioral step disturbance).** Injecting an adversarial trigger token mid-generation, CLAS's loop recovers `y_t` to the setpoint within `T_rec` tokens; fixed-angle does not recover. *Falsified if* recovery times are equal.
- **H5 (cost of the per-token control loop, measured honestly).** CLAS's online cost is the **extra Python-side per-token control step plus loss of fused batched `generate`** (Sec. 6), *not* "two gathers for free." We **measure** wall-clock throughput vs. baseline `model.generate` and report the penalty; we do **not** assert "within 2%." *Falsified if* the penalty makes 512-token × multi-method × ablation evaluation infeasible within budget (mitigation: every-`N`-token updates as the default).

### 5.2 Baselines (the four required, plus controls) — with honest engineering cost

1. **No steering** (reference PPL/capability; baseline harmful refusal 100%, harmless 0%).
2. **Angular Steering** — fixed `theta`, `adaptive_mode∈{0,1}` (open-loop; `utils.py` as-is).
3. **PID-AcT** (+Mean-AcT) — full-`d` additive PID on diff-means / running means over depth. **Re-implemented from scratch** (arXiv:2510.04309); not in repo.
4. **A-LQR** — full-`d` LQR with offline Jacobian gains, LFS scalar setpoint. **Re-implemented from scratch** (arXiv:2604.19018); plus a projected-to-2D efficiency-matched control.
5. **SO2 energy-damping** — state-feedback over depth (from `research_proposal_SO2.md`). **Re-implemented from scratch.**
6. **PTS-MPC** — `2×2` plant, receding-horizon QP, **including its Sec. 6.3 two-level token/layer variant** (the fairest comparison to CLAS's token loop). **Re-implemented from scratch.**
7. **CLAS ablations** (Sec. 5.4).

> **Engineering honesty:** four of six baselines do not exist in the repo and are first-class re-implementation work with a shared tuning protocol and matched compute (same model, same steered layer(s), same observable for behavioral metrics). We fix per-token vs per-completion measurement identically across methods.

### 5.3 Tasks, metrics, and the dynamic-range problem

- **Dynamic-range pre-check (gates everything).** Baseline harmful refusal on Qwen2.5-3B is already **100%**, and the strongest steering moves it only to ~94% (bypass) — the saturated binary has almost no headroom. **Therefore the primary controlled variable is the continuous logit margin / multi-step observable**, which has range even where the binary is railed. We additionally evaluate on a **larger model and/or the induce-refusal-on-harmless direction** (which the sweep shows is weak and controllable: harmless refusal 0%→12.5%) to obtain a regime with genuine actuation authority before asserting H3/H4.
- **Refusal / jailbreak** (primary): AdvBench held-out (`get_harmful_instructions` test split), Alpaca harmless (`get_harmless_instructions`). Metrics: refusal rate (substring + LlamaGuard3 *if* vLLM is available — see Sec. 6), the **continuous refusal-margin trajectory** (the controlled variable), and **steady-state behavioral error** `|y* − mean_t y_t|`.
- **Capability preservation** (constraint): perplexity (WikiText) and TinyBenchmarks **iff** the vLLM eval stack is stood up; otherwise a substring+PPL core (stated, not implied turnkey).
- **Control-quality metrics** (borrowed from control engineering): rise time, overshoot, settling band, integral-absolute-error `Σ|epsilon_t|`, **recovery time after a step disturbance**, and an empirical gain margin from a reference-step response (Exp. 6).

### 5.4 Ablations (isolating the one real contribution)

- **Output feedback vs. state feedback at matched cost** — token loop on `y_t` (multi-step observable) vs. token loop on internal `phi_L` only. **The decisive test of the thesis**, run specifically in the decoupling regime (H2).
- **Observable choice** — single-token logit margin (circular/easy) vs. multi-step decoded-text EMA (decoupled) vs. trained probe.
- **P vs PI vs PID; sign-guard on/off; integrator-freeze threshold** — isolates integral disturbance rejection *and* the unimodality safeguard.
- **`sat_band` width / `theta_nom` placement** — confines to the measured monotone band; shows runaway when the band wrongly includes the peak (negative control).
- **Update period `N`** (every token vs. every `N` tokens) — throughput vs. tracking (ties to H5).
- **Optional multi-layer deadbeat variant** (Sec. 2.5) with the `r_k` observability guard — gated on measured residual drift; reports whether multi-layer steering preserves coherence on 3B.

### 5.5 Primary deliverable

Two figures: (1) effectiveness–coherence Pareto fronts for all six baselines + CLAS on the **continuous** observable, annotated with mean `|theta − theta_nom|`; (2) a step-/disturbance-response panel showing CLAS reaching and *holding* the behavioral setpoint where open-loop and integral-free methods drift or offset — plus the measured `G(theta)` map with the monotone band and peak annotated.

---

## 6. Implementation Plan (mapped to the *actual* code)

The repo provides the actuator and the diagnostics; CLAS's new work is the **per-token control loop**, the **behavioral observable**, and the **four baselines**. We correct every code claim from the earlier draft.

| File | Status | Change |
|---|---|---|
| `pytorch_pure/utils.py` | exists (`get_angular_steering_output_hook`, `_get_rotation_args`) — **performs an absolute-angle reset of a fixed unit vector to `theta`; does NOT measure `phi_k` and does NOT rotate by `phi+u`.** | **Generalize the hook** so the absolute target `theta` becomes a value read from a shared controller object each forward pass: `steer = cos(theta_t) b1 + sin(theta_t) b2` with `theta_t` mutated by the token loop. Keep the absolute-reset semantics (it is a deadbeat set-point and is the right primitive here). Add `phi = atan2(h@b2, h@b1)` (computed in fp32) **only** as a diagnostic/observability readout, plus `r = sqrt((h@b1)^2+(h@b2)^2)` for the optional multi-layer guard. Keep `adaptive_mode`. |
| `pytorch_pure/phase_portrait.py` | exists (`compute_phase_trajectories` returns `phi, omega, r, c1, c2`; `compute_steering_plane` selects ONE layer and orients `b1` toward the harmless pole) | **Reuse directly** to (a) build `{b1,b2}`, (b) measure natural drift `delta_k = mean(omega_k)` and **in-plane magnitude `r_k`** (the observability guard), (c) supply diagnostics for the optional multi-layer variant. Note the orientation caveat: `b1` is labeled "safe" but the sweep shows refusal peaks toward the *opposite* pole — fix the sign convention against the measured `G`. |
| `pytorch_pure/steering_validation.ipynb` | exists (§4 **binary substring matcher** `is_refusal`; §13 hook; §21 full-circle sweep; baseline harmful refusal 100%) | **Primary harness.** §21's sweep becomes the empirical `G(theta)` map (identifies the monotone band, `gamma_lo/hi`, peak). **Build the continuous logit-margin observable** (define `R`/`C` token-id sets; logsumexp margin) — it does **not** exist; §4 is only a substring matcher and produces a single binary label *after* full decoding, which cannot drive a per-token integrator. **Replace the opaque `generate_completions` call with a manual autoregressive decode loop** (next row). Add Pareto-front, `G(theta)`, and disturbance-response plots. |
| `pytorch_pure/clas_controller.py` | **new (~150 LOC)** | `OuterThermostat` (PI, anti-windup, `sat_band`, **online slope estimate + sign guard**), `behavioral_readout(logits, decoded_so_far, z_L)` (logit-margin / multi-step EMA / probe), `angular_ema(cos,sin)` for circle-correct smoothing (EMA of unit vectors, then `atan2`, never of raw `phi`), and `make_clas_hook(plane, controller_state)`. No Jacobians, no QP. |
| **per-token decode loop** (in `clas_controller.py` / harness) | **new — core engineering risk; does NOT exist** | `generate_completions` calls a single opaque `model.generate` with **no** `output_scores`, `return_dict_in_generate`, `StoppingCriteria`, or `LogitsProcessor`. CLAS needs a **manual token-by-token decode loop** (or a `LogitsProcessor` that writes `theta_t` into shared hook state): manage the KV cache, extract last-token logits each step, compute `y_t`, update the `OuterThermostat`, write `theta_t` for the next step, maintain **per-sequence** controller state. **This replaces batched `generate` and likely defeats batching** — benchmarked under H5. |
| `pytorch_pure/generate_responses.py` | exists (calls `generate_completions`) | Add the CLAS generation path (the manual decode loop). |
| **Eval stack** (`llama_guard.py`, `evaluate_jailbreak.py`, `eval_perplexity.py`, `tiny_benchmarks.py`) | exist but **all import `vllm`**, which the `pytorch_pure` path deliberately avoids (`requirements.txt`: "No vLLM needed") | **State the choice explicitly:** either stand up vLLM for LlamaGuard3/TinyBenchmarks/WikiText-PPL, **or** port these to pure PyTorch, **or** fall back to a substring+PPL core. We do not imply turnkey reproducibility. |
| `pytorch_pure/visualization.ipynb` | exists | Add control-quality plots (step/disturbance response, IAE, margin) and the Pareto front. |

**Compute budget (stated, not hidden).** Current notebook caps: `N_EVAL=32`, `N_SWEEP=16`, `MAX_NEW_TOKENS=48`; `get_harmless_instructions` caps Alpaca to `train[:512]/test[:128]`; the sweep already costs ~3–4 s per angle at these caps. The per-token Python control loop defeats fused batching, so a `6 methods × tasks × ablations × 512-token` matrix is a real cost. **Default to every-`N`-token updates** (not a fallback), with a fixed eval `N`, `max_new_tokens`, and seed count, and a GPU-hour estimate reported up front. Verify 512-token generation is affordable under the loop *before* asserting H4.

**Observer note.** The diagnostic angle `phi` and the magnitude `r` are two dot products + an `atan2` at the steered position, cheap inside the existing hook; cast the two scalars to fp32 for the `atan2` (the model runs in bf16). Cross-token smoothing requires the manual decode loop to thread per-sequence `(cos, sin)` EMA state.

**Milestones.** Wk 1: measure `G(theta)` on the continuous observable, fix sign convention, identify band/peak/`gamma`. Wk 2–3: manual decode loop + logit-margin/multi-step observable + `OuterThermostat` with sign guard; H1/H3 on the continuous margin. Wk 3–5: baselines (PID-AcT/Mean-AcT, A-LQR, SO2, PTS incl. its two-level variant) at matched cost; Pareto front. Wk 5–7: disturbance-response (H4), throughput benchmark (H5), the decisive output-vs-state ablation in the decoupling regime (H2). Wk 7–8: larger-model dynamic-range check; writing.

---

## 7. Novelty & Contributions (narrowed to what survives adversarial review)

1. **Output feedback, not state feedback — the sole load-bearing novelty.** CLAS is the first steering controller to close a loop on a **measured behavioral output** of the model (a readout of `Unembed(z_L)`: refusal logit margin / multi-step decoded-text observable), regulating the variable we actually care about. This is distinct from SO2/A-LQR/PTS (internal coordinate / feature / `c_k`) and from **both** PID-AcT and its online **Mean-AcT** variant (internal diff-means / means). The dividing line is **internal-state vs output**, not online-vs-offline.
2. **Reactive PI on a behavioral output, vs PTS's predictive MPC on an internal trajectory.** A secondary differentiator: cheaper (no QP, no `2×2`/`d×d` plant — only an online scalar slope estimate), regulating output rather than an internal reference path.
3. **A sign-guarded integrator confined to an empirically identified monotone band**, which is the correct and honest way to apply integral action to the *unimodal* behavioral map that the repo's own data exhibits (prior steering controllers implicitly assume monotonicity).
4. **A control-theoretic evaluation protocol for steering** keyed to a **continuous** behavioral observable with dynamic range (logit margin / multi-step EMA), including disturbance-response and recovery-time metrics — usable even where the binary refusal endpoint is saturated.

**Explicitly NOT claimed as novel** (credited to prior work): the cascaded token/layer hierarchy (PTS Sec. 6.3); the SO(2) angular formulation and wrapped-angle / Lie-algebra error (SO2); the norm-preserving rotation actuator and the `{b1,b2}` plane (Angular Steering); the PI steady-state-error argument (PID-AcT); and the use of saturation as a constraint (SO2/PTS).

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Behavioral map `G` is non-monotone (CONFIRMED by repo data)** | Certain | High | Measure `G(theta)` first; confine controller to the identified monotone band via `sat_band`; sign-guarded integrator that freezes near the peak. The "zero steady-state error" claim is scoped to the band. |
| **Refusal task is saturated (baseline harmful refusal = 100%, CONFIRMED)** | Certain | High | Use the continuous logit margin / multi-step observable (range even when binary is railed); add a larger model and the induce-on-harmless direction for genuine dynamic range before H3/H4. |
| **Single-token logit margin ≈ affine in `phi_L` ⇒ output feedback ≈ state feedback (circularity)** | High | High (could self-falsify H2) | Pre-register the decoupling regime: run the decisive H2 ablation on the **multi-step** observable (function of many emitted tokens, irreducible to one layer's angle); report the single-token case as the easy/circular control. |
| **Per-token loop kills batching / is costly (CONFIRMED no `output_scores` path)** | High | Medium | Build the manual decode loop as a first-class deliverable; default to every-`N`-token updates; benchmark throughput (H5) rather than asserting it. |
| **Eval tools are vLLM-bound; `pytorch_pure` avoids vLLM** | Certain | Medium | State the choice: stand up vLLM, port to pure PyTorch, or fall back to a substring+PPL core. No turnkey claim. |
| **Four baselines do not exist in the repo** | Certain | Medium | Budget PID-AcT/Mean-AcT, A-LQR, SO2, PTS (incl. two-level) as first-class re-implementations with a shared tuning protocol and matched compute. |
| **Transport delay shrinks stable gain / can oscillate** | Medium | Medium | Carry the unit delay in the recursion (Sec. 4.2); tune against `gamma_hi` *and* the delay; verify margin empirically. |
| **Optional multi-layer variant breaks coherence on 3B / `phi_k` ill-defined where `r_k` small** | Medium | Low (it is optional) | Keep it an ablation; add the `r_k` observability guard; report PPL under multi-layer steering. |
| **Reviewers see it as "just PID / output-feedback relabeled"** | Medium | Medium | Concede everything shared (cascade=PTS, angle=SO2, rotation=Angular Steering, PI-SSE=PID-AcT, Mean-AcT is online); stake the entire claim on the one decisive row — **observable = realized output behavior** — and on the decoupled-regime ablation. |

---

## 9. References

**Repository papers (digested).**
- Vu, H. M., & Nguyen, T. M. (2025). *Angular Steering: Behavior Control via Rotation in Activation Space.* NeurIPS 2025. (Rotation/absolute-angle actuator, plane `{b1,b2}`, adaptive mode; open-loop fixed angle; multi-layer steering coherence caveat on small models.)
- Nguyen, D. V., Pham, N. Y., Vu, H. M., Zhang, L., & Nguyen, T. M. (2026). *Activation Steering with a Feedback Controller* (PID-AcT). ICLR 2026. arXiv:2510.04309. (P/PI/PID over depth on difference-in-means; Prop. 1 steady-state error of P-control; **Mean-AcT** re-measures running means online; full-`d` additive.)
- Skifstad, J., Yang, X. A., & Chou, G. (2026). *Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control* (A-LQR). arXiv:2604.19018. (Local-linearity, LTV/LQR with `d×d` Jacobian gains, scalar internal setpoint `beta*_k`; Thm 4.1/4.2.)

**Sibling proposals in this repository.**
- *Angular Steering as Control on SO(2): Phase Portrait Analysis and Energy-Based Stabilization* (`research_proposal_SO2.md`). (Energy/Lyapunov **state-feedback over depth** on the activation **angle** `phi_k`; wrapped-angle / SO(2) error; separatrix/barrier.)
- *Predictive Trajectory Steering: Model Predictive Control in the Angular Steering Plane* (`research_proposal_PTS.md`). (Receding-horizon MPC tracking a 2D reference trajectory; `2×2` plant; **Sec. 6.3 Two-Level MPC: token-level outer loop + layer-level inner loop, "analogous to hierarchical MPC"** — i.e. the cascade is prior art.)

**Supporting external literature.**
- Nettasinghe, B., & Joseph, V. (2026). *As Language Models Scale, Low-order Linear Depth Dynamics Emerge.* arXiv:2603.12541. (Cheap low-order depth models are accurate.)
- Fernando, J., & Guitchounts, G. (2026). *Dynamics of the Transformer Residual Stream.* arXiv:2605.14258. (Rotation-dominated early layers — source of any residual angular drift `delta_k`.)
- Mishra, S., et al. (2026). *Steered LLM Activations are Non-Surjective.* arXiv:2604.09839. (Additive steering pushes off-manifold; rotation preserves norm — motivates the rotation actuator.)
- Åström, K. J., & Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning.* ISA. (Discrete PI, anti-windup, integral disturbance rejection.)
- Khalil, H. K. (2002). *Nonlinear Systems* (3rd ed.). Prentice Hall. (ISS, singular perturbation / timescale separation.)
- Franklin, G. F., Powell, J. D., & Workman, M. (1998). *Digital Control of Dynamic Systems.* Addison-Wesley. (Jury stability test; delay in discrete loops.)