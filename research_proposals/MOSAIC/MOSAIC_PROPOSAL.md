# MOSAIC — Multi-Objective Steering via Allocation In Cones

*A control-theoretic recasting of **multi-attribute** activation steering as **constrained control
allocation**: drive one graded text attribute to a commanded setpoint while **holding** the others at
their natural baselines, by exploiting the redundant degrees of freedom (the null-space) of a
**multi-concept cone basis**. The first regime in this program where the **control law provably beats
the bound**, because the hold-constraint is **active at the optimum by construction** — something a
scalar push cannot satisfy.*

Status: **proposal** (2026-06-13). Successor to CASA (additive-cone win) + BASELINES (law-is-a-wash) +
CALM (coherence-aware ties). Authored from a deep-research workflow over the 6 reference papers, the 7
prior proposals, the reusable codebase, and a 2024–2026 literature sweep (25 agents). Sits on the
**identical** Gemma-2-2b artifacts (band `[7,24]`, `pscale=180.85`, plant R²≈0.999).

---

## 0. The one falsifiable claim

> On graded text attributes whose concept cones are **non-orthogonal** in the residual stream
> (measured cross-coupling above a pre-registered tolerance), a bounded-additive **null-space
> control-allocation QP** drives attribute **A** to a commanded graded setpoint `τ_A` while holding
> **B, C** within tolerance of their unsteered baselines with **strictly lower off-target drift**
> (> 1 SE at matched A-tracking error **and** matched effort `‖u‖`) than **any** per-attribute scalar
> push — because the hold-B,C constraint is **active at the optimum**, and a scalar (null-space-free)
> actuator provably cannot satisfy it.

If that win does not materialize, the **pre-registered fallback is itself a result**: report the
*dividing line* — "allocation pays iff measured cross-coupling > ε" — as a publishable scoping law,
with the day-one number (`induced-drift-per-unit-A-tracking`) as the predictor. Either way the project
**cannot silently reproduce the saturation null**.

---

## 1. Why now: what the arc already proved, and the one regime it never tested

Your seven proposals converge on a single hard-won meta-finding, now **externally corroborated**:

> For cone-based **de-refusal**, no control sophistication — lookahead (PTS), a coherence-aware cost
> (CALM P1+2), or dynamic per-token feedback (CALM P3) — beats a well-chosen fixed magnitude bound.
> The durable levers are the **actuator** (additive ≫ rotation) and the **bound** (`u_max`); the
> **controller is a wash**. Mechanism: refusal is a **binary scalar that flips and saturates**, so
> strength↔coherence is a **hump, not a monotone frontier**; the constraint never binds against a
> *continuing* objective, and the trade-off is governed by *aggregate* ablation, not its allocation in
> depth or time.

The literature now says the same thing out loud, on **your exact models**:

- **A-LQR** (*Local Linearity… Model-Based Linear Optimal Control*, `papers/`) fits a per-layer linear
  plant and applies LQR with an additive actuator on Gemma-2-2b / Qwen-2.5-3B. It gets **30–50× toxicity
  reduction** (graded) but **underperforms baselines on AdvBench jailbreak** (binary) and needs an
  all-token "A-LQR+" hack to match them. *The control machinery pays on the graded task and loses on the
  binary one — exactly your hump.*
- **PID Steering** (*Activation Steering with a Feedback Controller*, `papers/`; your own group) owns the
  "steering = P-controller, add I/D over **layers**" framing. Its integral gain helps most on **graded
  toxicity** (8×), least on binary jailbreak. The depth-axis control framing is **taken** (and your PTS
  result already showed late-band layer dynamics are near-identity, so depth-lookahead is inert anyway).
- **Arditi et al.** (refusal = a single direction) is the mechanistic root cause: a 1-DoF binary flip has
  **nothing to allocate**.

So the open question the whole program has been circling is precise: **is there a steering regime where a
control *law* — not just the actuator and the bound — is load-bearing?** The synthesis named three
conditions that would make it so: (1) a **non-saturating monotone** frontier; (2) a constraint that
genuinely **binds and reshapes** the optimum; (3) **evolving dynamics** where lookahead has predictive
value. Every prior proposal failed condition (1) by testing on refusal.

**MOSAIC is the first to deliver (1) and (2) by construction, and to honestly decline (3).** It moves to
**graded attributes** (formality, reading-level, sentiment-intensity — monotone dose-response, *not* a
binary flip), and it makes the constraint bind *structurally*: because concept directions live in
**superposition** (non-orthogonal cross-Gram; Geometry-of-Refusal's RepInd shows orthogonal ≠ causally
independent), moving A's coordinate with a naive scalar push **mechanically drags B and C off their
setpoints**. Holding B, C fixed is therefore an **active constraint**, and satisfying it **requires the
redundant DoF of A's cone** (`k_A > 1`) — a null-space a scalar actuator does not have. The win is a
**static property of actuator geometry**, so it sidesteps the hump via condition (2) alone, and it
**pre-registers H=1 ≈ H>1 as a confirmation** (cross-coupling lives in the Gram, not in evolving
dynamics) so it cannot re-open the PTS lookahead wall.

This is also the field's named open problem. **Course-Correction** (AAAI 2026) builds a goal-space over
*these exact attributes* (Flesch-Kincaid reading difficulty, Heylighen–Dewaele formality, length) and
decomposes steering error into **parallel** (miscalibration: over/undershoot of the requested attribute)
vs **orthogonal** (**side-effects**: unintended drift on the others) — and finds *"side effects remain
pervasive even in strong LLMs"* and dominate the failure budget, surviving best-of-128 and RL. **MOSAIC's
constrained allocation is exactly the mechanism that cancels the orthogonal/side-effect error that scalar
and best-of-N steering cannot.**

---

## 2. The idea in one picture

```
Concept cones in the residual stream (non-orthogonal — superposition):

      B (reading-level)
        ^             A (formality)               GOAL:  c_A → τ_A   (track)
        |            /                                    c_B → c_B^base (hold)
        |          /                                      c_C → c_C^base (hold)
        |        /
        |      /  .· a naive scalar push along A  ← drags B, C off baseline (SIDE EFFECT)
        |    /  .·                                   (Course-Correction's dominant failure)
        |  / .·
        |/.·_______________> C (sentiment)

Scalar steering  : u = α·d_A           → 1 DoF, no null-space → cannot hold B,C  ✗
MOSAIC allocation: min ‖c_A+u_A − r_A‖²_Q + ‖u‖²_R
                   s.t.  ‖c_B+u_B − c_B^base‖ ≤ ε_B          ← ACTIVE constraint
                         ‖c_C+u_C − c_C^base‖ ≤ ε_C          ← (binds by construction)
                         ambient KL_t ≤ κ                    ← validated coherence budget (r=0.976)
                         ‖u‖ ≤ u_max                          ← the prior durable lever
                   over the stacked cone basis B_stack=[B_A;B_B;B_C]
                   → uses A-cone's redundant DoF (k_A>1) to hit τ_A *inside* the B,C-hold polytope
```

The headline experiment is a **Pareto plot on the side-effect axis**: A-tracking error (x) vs combined
B,C off-target drift (y), at matched effort and matched ambient-KL. MOSAIC's curve should lie **below**
the best scalar baseline's; the `k_A=1` ablation (no null-space) should **collapse onto** the scalar
curve. That gap **is** "control law > bound."

---

## 3. How this maps to your five directions

| Your direction | How MOSAIC uses it |
|---|---|
| **D1 — optimal setpoints, not diff-in-means** | The reference for A is a **calibrated, depth-indexed map** `τ_A ↦ r_{A,l}` — a *per-layer cone-coordinate setpoint trajectory* fit by regressing layer-wise cone coordinates against a **continuous** attribute label, **not** a single mean-difference vector. The "set of optimal setpoints on a manifold" is literal here: a curve `r_{A,l}(τ_A)` over (layer × commanded-intensity). B, C setpoints are the model's **own unsteered baselines** (a disturbance-rejection "don't-move-these" reference). |
| **D2 — advanced / constrained / energy control** | The core is a **constrained control-allocation QP/MPC** with a **binding** hold-polytope + a **hard ambient-KL energy budget** (the only coherence signal you proved faithful, r=0.976) + the `u_max` bound. This is the one formulation where the constraint is *active*, which is precisely what made P/PID/LQR/MPC tie before (their constraint "barely binds"). |
| **D3 — autoregressive, layer + token** | Allocation runs **per band-layer** `[7,24]` (depth) **and** is re-enforced **per generated token** inside the verified `decode_dual` loop (the reverting-disturbance footing, §6). Tokens are **not** treated independently: the hold is a *sustained* disturbance-rejection problem over the 256-token generation. |
| **D4 — LoRA as a model for control** | **Cut from the critical path** (honest: the codebase freezes the model and has zero training scaffolding — see §10). Retained as a **strictly-gated upside extension** (§13): after a 2-family cone win, train one low-rank ReFT/LoRA per attribute as a learned multi-DoF actuator whose **rank-r column space = "where you can push"** and whose **null-space the allocator exploits** to hold B,C — testing "rank = DoF of an over-actuated actuator." |
| **D5 — your call** | The genuinely fresh framing borrowed from **aerospace control allocation** (redundant actuators + null-space secondary objective), so far **unclaimed** in the LLM-steering literature, plus the falsifiable **"coupling magnitude predicts when allocation pays"** scoping law. |

---

## 4. Control formulation

**State.** Stacked cone coordinate `c = B_stack · h ∈ ℝ^K` at each band layer, where
`B_stack = [B_A; B_B; B_C] ∈ ℝ^{K×d}` stacks the per-attribute orthonormal cone bases,
`K = k_A + k_B + k_C`. Per-attribute blocks are individually orthonormal; **cross-blocks are kept raw**
(the off-diagonal cross-Gram `Π = B_stack B_stackᵀ − I` is the coupling the QP must reject — *do not
pre-orthogonalize it away*, see §5/§9).

**Plant.** For the static allocation core, the model **is** the plant; the linearized layer Jacobian /
cross-Gram is the coupling map. A fitted **stacked k×k affine plant** `c_{l+1} = A_l c_l + b_l`
(`casa_control.fit_cone_plant`, R² gated) is used **only** if the H>1 falsification ablation is run.

**Actuator.** Bounded **additive** multi-concept cone push (`casa_actuator.ConeActuator` generalized from
a single cone `U` to `B_stack`), control `u ∈ ℝ^K`, applied on the residual-stream **output** of
`model.layers.{j}` across the band via `make_controller_hooks`. Norm-changing, multi-DoF. **Rotation is
structurally excluded** (norm-preserving ⇒ 1 realized DoF ⇒ *no null-space to allocate into* ⇒ cannot
solve the hold-fixed problem — this is the cleanest statement yet of why rotation caps out).

**Reference / setpoint.** `r_A = r_{A,l}(τ_A)` (the calibrated depth-indexed map, §5); `r_B, r_C` = the
model's **own unsteered baseline** cone coordinates.

**Objective + constraints (the allocation QP, solved per layer):**

```
min_u   ‖ c_A + u_A − r_{A,l} ‖²_Q  +  ‖u‖²_R
s.t.    ‖ c_B + u_B − r_{B,l} ‖  ≤  ε_B          (HOLD B — the active constraint)
        ‖ c_C + u_C − r_{C,l} ‖  ≤  ε_C          (HOLD C)
        ambient KL_t(steered ‖ unsteered)  ≤  κ   (coherence budget, r=0.976 — token loop)
        ‖u‖  ≤  u_max                             (actuator bound, the prior durable lever)
```

**Horizon.** `H=1` static allocation is the **headline**. `H>1` is run **only** as a falsification
ablation, **pre-registered to tie** (a non-tie is a bonus, not the claim) — so MOSAIC cannot re-derive
the PTS null.

**Why this binds where everything before tied.** Unconstrained ⇒ MPC = LQR = P (you proved this to 1e-5
in `casa_baselines`), so all laws collapse to "pick a magnitude." MOSAIC's hold-polytope is an
**inequality constraint that is active** whenever cross-coupling pushes `c_B, c_C` toward the boundary —
which is exactly the regime Phase 0 selects for. An active constraint is where QP/MPC stop being equal to
a tuned scalar.

---

## 5. The optimal-setpoint design (D1) — beyond difference-in-means

Three concrete departures from the naive single-vector target:

1. **A graded, depth-indexed reference map, not a vector.** Fit `r_{A,l}(τ_A)` by regressing per-layer
   cone coordinates against a **continuous** attribute label (deterministic Flesch-Kincaid;
   Heylighen–Dewaele formality F-score; a calibrated sentiment score) across intensity bins. The output
   is a *reference trajectory* over (layer, commanded-intensity) — your D1 "set of optimal setpoints on a
   manifold," made operational. Ablate it against diff-in-means and report **miscalibration error**
   (Course-Correction's *parallel* component).

2. **Hold setpoints are the model's own baseline**, `r_{B,l} = c_l^B(unsteered)` — a "don't move these"
   disturbance-rejection reference, not a learned pole.

3. **Cone (k>1), not vector.** A's reference lives in a `k_A`-dim **retain-loss concept cone** (`rco`) so
   the QP has the **redundant DoF / null-space** to hit A's scalar-intensity target while staying inside
   the B,C-hold feasible set. This is the geometric reason allocation beats scalar steering — and it
   reuses CASA's exact `k=4` machinery (which already produced a *coherent* k>1 subspace where a blunt SVD
   span was gibberish).

**Critical de-risk (from the adversarial panel):** do **not** pre-decouple the cones with RepInd before
measuring coupling — RepInd would **erase the very cross-coupling the method needs**. Measure **raw**
coupling first (Phase 0); apply RepInd only to the **residual** the controller must reject, and report it
as an ablation (raw vs RepInd-decoupled).

---

## 6. Layer × token structure (D3) and the reverting-disturbance second footing

**Layer level (where the static win lives).** The allocation QP runs at each band layer `[7,24]`,
measuring `c = B_stack h` and emitting the K-dim `u` that hits `r_{A,l}` while holding `c_B, c_C` in the
polytope.

**Token level (the second, independent binding mechanism — grafted from the SARTRE candidate).** The same
hold-constraint is enforced **per generated token** inside the verified `calm_token.decode_dual` loop.
This gives MOSAIC a *second footing* that does not depend on instantaneous geometric coupling: even if the
cones are weakly coupled at a single position, **the model's autoregressive prior drags B, C back toward
its preferred register over the 256-token generation** — so *holding* B, C is a genuine **sustained
disturbance-rejection** problem on the token axis. The token loop reads a calibrated per-token attribute
probe `a_t` (validated against an external scorer — a hard gate) and the ambient `KL_t` (the hard budget),
and updates the allocation state live each forward.

This is **per-token feedback + per-token constraint enforcement**, *not* multi-step token lookahead. H>1
is expected inert and reported as a confirmation. The reverting-disturbance mechanism is what lets MOSAIC
*revive CLAS's dropped sustained-control thesis* — which died only because **rotation lacked authority**;
the additive actuator has the authority a graded sustained attribute needs.

---

## 7. Phased plan — pre-registered kill gates (the Phase-0 fork)

Hard gates front-load all falsification **before any controller is built**, so a NO-GO costs < 1 day.

### Phase 0 — Binding-precondition KILL (HARD GATE; ~1 GPU-hr; no controller, no training)
**This is a fork.** Cheaply measure the **two** mechanisms that could make the constraint bind, and let
the data pick the headline:

- **(a) Cross-coupling (→ MOSAIC).** Build three quick diff-in-means cones for A, B, C
  (`casa_cone.residual_means` + `dim_directions`) on a **deliberately interfering** triple
  (formality↔reading-level couple strongly — both load on lexical complexity; sentiment as the third).
  Apply a bounded scalar-A push to a non-trivial setpoint; **measure induced drift in B, C** (external
  scorers) on held-out generations. Quantify the off-diagonal cross-Gram numerically.
- **(b) Reversion (→ SARTRE arm).** Steer an attribute up at the prompt, **release**, and measure whether
  it **drifts back** over 256 tokens (the autoregressive prior as a reverting disturbance; estimate the
  reversion rate `α`).

> **GO (MOSAIC headline)** iff a scalar-A push moves B and/or C by **more than ε AND > 1 SE** of their
> natural variation, **and** the A-cone has a null-space direction within `‖u‖≤u_max` that hits τ_A with
> materially less B,C drift (a feasible allocated solution exists).
> **GO (SARTRE headline)** iff the released attribute reverts with `α ≈ 0.7–0.9` (a genuine sustained
> disturbance) and held-out token-axis plant R² > 0.9.
> **NO-GO on both** ⇒ neither mechanism binds; **report the measured coupling/reversion as the
> dividing-line law** — a publishable scoping result, not a dead null.

The single number `induced-drift-per-unit-A-tracking`, measured day one, **predicts the eventual win
margin** and is reported regardless of outcome.

### Phase 1 — Graded sensors + monotone-frontier verification (~2 GPU-hr)
Build calibrated per-attribute probes `a_t = σ(w·h + b)` (regress residual projections against the
external scorers). Sweep aggregate cone-push magnitude; plot realized **intensity** *and* **coherence**
(ambient KL, genNLL, **Type-Token-Ratio** ρ=0.81 — **exclude perplexity, ρ=0.00**) vs strength.
> **GO** iff (a) the dose-response is **monotone** over the coarse-to-mid range (no refusal-style hump),
> **and** (b) the online probe `a_t` correlates with the external scorer at **r > 0.85** on held-out text
> (a hard gate — a loop chasing a biased sensor manufactures fake wins; if noisy, OAS's Kalman observer
> earns its niche here). NO-GO if the frontier saturates like refusal.

### Phase 2 — Graded retain-loss cones + the depth-indexed setpoint map (the D1 build)
Retarget `casa_cone.rdo/rco` (retain loss is concept-agnostic) from refusal to each graded attribute via a
low/high-intensity contrastive split; fit `r_{A,l}(τ_A)`. Re-fit the stacked plant R² as a gate (only
needed for the H>1 ablation). Run **both** raw and RepInd-decoupled cones.
> **GO** iff `k_A>1` cones are coherent (held-out tracking error below threshold) **and** the
> depth-indexed reference beats diff-in-means on calibration. NO-GO if the k>1 cone is gibberish (the SVD
> failure mode) or the depth map adds nothing.

### Phase 3 — The allocation QP vs the scalar bound (THE HEADLINE; layer-domain, H=1)
Extend `casa_control.solve_qp`/`build_condensed_qp` with a projection/log-barrier onto the B,C-hold
polytope + the hard ambient-KL budget; deploy via `make_controller_hooks`.
> **PRE-REGISTERED WIN:** the allocation QP reduces combined B,C off-target drift by **> 1 SE** over the
> best per-attribute **scalar** baseline **at matched A-tracking error AND matched total effort `‖u‖`**,
> at equal ambient-KL, **n ≥ 40**; **AND** the `k_A=1` ablation **fails** to achieve it (proving redundant
> DoF is load-bearing). If allocation merely **ties** the scalar bound at matched effort ⇒ declare a
> **SCOPED NULL** and report the coupling-magnitude dividing line.

### Phase 4 — Token-axis hold + second-family confirmation + H>1 falsification (~1 day)
Enforce the hold + KL budget **per token** in `decode_dual` (widen the `set_strength` contract to carry
the K-dim allocation state + read `a_t`); test whether per-token hold beats a front-loaded fixed
allocation as B, C revert. Repeat Phase 3 on **Qwen2.5-3B**. Run H=1 vs H∈{4,8}.
> **Token-hold GO** iff per-token enforcement cuts terminal B,C drift vs front-loading by > 1 SE.
> **Second-family GO** iff Phase-3 dominance reproduces on Qwen2.5-3B (≥ 2 families).
> **H>1 pre-registered to tie** (confirmation, not contribution).

---

## 8. Evaluation

**Attributes / data.** Formality (GYAFC / Heylighen–Dewaele F-score), reading-level (Flesch-Kincaid,
deterministic), sentiment-intensity (RoBERTa sentiment). Prompts: neutral open-ended instructions
(Alpaca harmless path via `utils.get_input_data`). **Reuse Course-Correction's goal-space + open-source
framework** (`github.com/MLD3/steerability`; 64 sources × goals) for the parallel/orthogonal decomposition
— it is purpose-built for exactly this measurement.

**Models.** Gemma-2-2b-it (primary; all artifacts on band `[7,24]`) + Qwen2.5-3B-Instruct (second
family). Llama-3.2-3B deferred behind a 2-family win (the field's reliability bar; cf. Tan et al.).

**Metrics.**
- *Primary:* A-tracking error `|A_achieved − τ_A|` (Course-Correction **parallel**/miscalibration) **and**
  off-target drift `|B−B_base|, |C−C_base|` (**orthogonal**/side-effect). Report the **Pareto frontier**
  of A-tracking vs B,C-hold-violation + a single SPI-style scalar.
- *Coherence:* ambient KL (r=0.976), genNLL, **TTR** (ρ=0.81). **Perplexity excluded** (ρ=0.00 with
  fluency — *Effectiveness-Fluency study*; corroborated by *A Sober Look at Steering Vectors*).
- *Success rate:* fraction hitting τ_A within δ **and** holding B,C within ε.

**Baselines (must include the prior winners and the external SOTA).**
- **Your own bars:** the CASA additive-cone fixed-`u_max` winner; the fixed-bound **P** controller
  ("no controller, just the bound" — the bar every law must clear).
- **Per-attribute scalar steering** (CAA / DiffMean) at best-tuned bound — *the* comparison.
- **Per-attribute PID-AcT and A-LQR run independently** (their I/feedforward over-steer is your documented
  failure — show it on a graded attribute).
- **Prompting intensity modifiers** ("slightly/somewhat/much/extremely") — **mandatory** (AxBench/SteerEval:
  prompting beats steering; you must win the *joint* track-A-and-hold-B,C objective prompting controls only
  loosely).
- **Multi-attribute / null-space SOTA:** K-Steering (classifier-gradient composition), Conceptors (Boolean
  affine composition), MAT-Steer / MSRS (orthogonal-subspace multi-attribute), AlphaSteer-style null-space
  projection — the citation-gap baselines a reviewer will demand.
- **Setpoint SOTA:** Precise Attribute Intensity Control (Pre-Control) — the target-reaching value-function
  baseline for graded control.

**Ablations.** hold-constraint on/off (off must recover scalar steering); `k_A=1` vs `k_A>1` (the
null-space test — the win must *need* redundant DoF); raw vs RepInd-decoupled cones; hard-KL budget on/off;
H=1 vs H>1; depth-indexed reference vs diff-in-means.

---

## 9. Novelty boundary (honest)

**What is NOT claimed as novel** (the literature owns these — cite, don't re-invent):
- "Steering = feedback/PID controller over layers" — **PID Steering** (your own group).
- "LLM layers are locally linear ⇒ LQR with a fitted plant" — **A-LQR**, on your exact models. *The k×k
  affine plant is no longer a clean novelty.*
- "Refusal is a 1-DoF binary flip / steering saturates" — Arditi et al.; *What Can We Actually Steer?*
- "Multi-attribute steering / avoid interference via orthogonal subspaces" — **MAT-Steer, MSRS, ACT,
  K-Steering, Conceptors**.
- "Null-space constraint for utility preservation" — **AlphaSteer** (but for *binary refusal*, single
  attribute, utility-preservation — not a graded multi-attribute *hold/allocation*).
- "Per-token dynamic steering magnitude" / "KL-to-unsteered as the coherence signal" — DAC, DSAS, PIXEL,
  and **your own CALM** (infrastructure, not a contribution).

**What IS new and reviewer-defensible:**
1. **The control-allocation formulation of multi-attribute steering**: track a *graded calibrated
   setpoint* for A while *holding* others at baseline, solved as a **constrained QP with the cone's
   redundant DoF (null-space) as the actuation freedom** — the *aerospace control-allocation* lens,
   unclaimed in LLM steering. Prior multi-attribute work **composes/orthogonalizes**; none formulates
   *setpoint-tracking-under-an-active-hold-constraint* and shows the constraint is load-bearing.
2. **A clean "control law > bound" result on the side-effect axis** — the program's open question — by
   construction (the `k_A=1` ablation collapses to the scalar bound; the `k_A>1` allocation dominates).
3. **The falsifiable scoping law**: *allocation pays iff measured cross-coupling > ε*, with
   induced-drift-per-unit-A-tracking as the predictor — turning even a NO-GO into a result.
4. **Depth-indexed calibrated reference map** `r_{A,l}(τ_A)` (D1) replacing diff-in-means, integrated into
   a closed-loop tracker over **both** layer and token axes.

---

## 10. Codebase reuse map (`file:function`) and what must be built

**Drop-in (verified, self-tested):**
- `CASA/casa_actuator.py::ConeActuator(U, mode, u_max, rho, cone_clip)` — generalize `U → B_stack`;
  `.band_hooks(module_dict, band)`; **`u_max` is read live each forward** (so the token loop can mutate
  allocation state mid-decode — verified by `calm_token._selftest_actuator_mutation`).
- `CASA/casa_cone.py::{residual_means, dim_directions, rdo, rco, generate_targets, orient_to_refusal,
  svd_subspace(neg baseline), TrainConfig}` — retarget the retain-loss objective from refusal to each
  graded attribute (the loss is concept-agnostic).
- `CASA/casa_control.py::{fit_cone_plant→{A,b}, plant_r2, rollout, build_condensed_qp, solve_qp (FISTA
  ball-projection QP), ConeMPC(.control(layer,xi)->u0,.reset()), make_controller_hooks(module_dict, band,
  B, controller)}` — **the single controller-agnostic deployment surface**: a new law just implements
  `.control(layer, xi:(M,K)) -> (M,K)` + `.reset()`.
- `CASA/casa_baselines.py::{ConeP, ConePID, ConeLQR, RecordingController}` — the matched-effort baselines.
- `CALM/calm_token.py::{decode_dual (verified dual-cache Gemma-2 loop, 16/16 vs generate), kl_to_unsteered
  (r=0.976), valid_mean_kl, load_cone_and_band}` — the token-axis harness.
- `CALM/calm_mpc.py::{CoherenceMPC, fit_cone_density, mahalanobis_trajectory}` — **note: the cone-space
  Mahalanobis surrogate does NOT predict genNLL (r=0.20); use ambient KL, not this, as the coherence
  cost.**
- `pytorch_pure/utils.py::{get_input_data, add_hooks, generate_completions}`; `observables.py::make_margin_fn`
  (template for a readout head).
- **Artifacts:** `CASA/outputs/casa_subspaces_gemma-2-2b-it.npz` (band `[7,24]`, `steer_layer=20`,
  `pscale=180.85`, plant R² 1-step 0.9993 / 5-step 0.9955) — sit MOSAIC on the *identical* plant.
- venv: `/home/will/work/llm-activation-control/.venv/bin/python`; run module self-tests first
  (`python casa_actuator.py`, etc.); `casa_experiment.py` is the master runner.

**Must be built (the honest gap list):**
1. **Graded attribute task + continuous readout.** Everything is built around binary de-refusal
   (`casa_judge` is refusal-specific). Need: continuous labels (formality/reading-level/sentiment 0..1) +
   a graded probe head (regression/logistic) — `make_margin_fn`'s hard-coded refusal token list won't do.
2. **The calibrated setpoint map** `τ_A ↦ r_{A,l}` — no fitter produces it today.
3. **The stacked multi-concept basis `B_stack`** + the **allocation QP** (extend `solve_qp` with the
   hold-polytope projection + KL budget; the existing QP tracks a *fixed* harmless-mean reference only).
4. **Widen `decode_dual`'s `set_strength` contract** to carry the K-dim allocation state (today it
   modulates a *scalar* `u_max` on a *fixed* cone).
5. **A monotone-frontier harness** for a non-refusal attribute (`casa_experiment --frontier` sweeps refusal
   only).
6. (LoRA arm only) a trainable low-rank actuator + training loop — *absent today* (model is frozen
   everywhere); this is why D4 is gated, not core.

---

## 11. Risks & mitigations

| Risk (sharpest first) | Mitigation |
|---|---|
| **Weak coupling** → hold never binds → QP collapses to scalar steering = the documented wash. | Phase-0 kill is the **literal first action** and a hard gate, on **deliberately interfering** triples; **don't** pre-decouple with RepInd; pre-register the coupling-magnitude dividing line; the **token-axis reversion** (Phase 4) is a *second* binding footing if instantaneous coupling is weak. |
| **Coupling too strong / nonlinear** → linearized cross-Gram is a poor model. | RepInd-decouple the *residual*; successive-linearization; re-fit stacked plant R² as a gate. |
| **Fine-grained monotonicity cliff** (SteerEval: steering fails at the tightest granularity). | Target **coarse-to-mid** intensity; verify dose-response empirically in Phase 1 before trusting setpoints. |
| **Prompting beats steering** (AxBench). | Must win the **joint** (track-A **and** hold-B,C) objective, which prompting controls only loosely; report prompting as a mandatory baseline. |
| **Biased sensor** → closed loop manufactures a fake tracking win. | Phase-1 probe-vs-external-scorer **r > 0.85 hard gate**; OAS Kalman observer on-hand if the per-token probe is noisy (its earned niche). |
| **Novelty crowding** (MAT-Steer/MSRS/AlphaSteer/K-Steering). | Position as **setpoint-tracking constrained allocation with a binding hold + Pareto-dominance on the side-effect axis + the coupling-predicts-payoff law** — *not* "null-space/multi-attribute steering is new" (§9). |
| **Scope creep** re-opening the lookahead wall. | Keep the headline static H=1; report H=1 ≈ H>1 as a confirmation; defer LoRA (D4) and the 3rd family behind a 2-family win. |

---

## 12. Timeline & success criteria

**Timeline (~1–1.5 weeks single-GPU).** Phase 0 ~1 GPU-hr (the gate) · Phase 1 ~2 GPU-hr · Phase 2 ~1–2
days · Phase 3 ~3–4 GPU-hr · Phase 4 ~1 day. Hard gates at Phase 0/1 front-load falsification, so a NO-GO
costs < 1 day.

**A WIN** (vs another null): at matched A-tracking **and** matched effort **and** equal ambient-KL, the
allocation QP cuts combined B,C drift by **> 1 SE** (~0.05 normalized, n ≥ 40) over the best **scalar**
baseline, **and** `k_A=1` **fails**, **and** it reproduces on a second family. This is a clean **control
law > bound** result on the **side-effect axis** that scalar steering structurally cannot reach.

**A SCOPED NULL** (still publishable): allocation ties the scalar bound at matched effort ⇒ report the
pre-registered dividing line ("allocation pays iff cross-coupling > ε") with the day-one
induced-drift number as predictor. **The project cannot silently reproduce the saturation null.**

---

## 13. LoRA extension (D4) — strictly gated, pure upside

Attempted **only after** a 2-family cone-allocation win. Train one low-rank **ReFT/LoRA** per attribute as
a learned multi-DoF actuator: its rank-`r` column space is "where you can push," and the allocator uses
its **null-space** to hold B, C — directly testing your "rank = DoF of an over-actuated actuator" idea, and
the "LoRA-as-control-variable" view (cf. *Optimal Control View of LoRA*, ECCV 2024; ReFT, NeurIPS 2024;
*Weight Updates as Activation Shifts*). Borrow a Grassmann channel-diversity diagnostic to verify the
null-space is real. Never on the kill path (the codebase has no training scaffolding — building it is the
cost that keeps this gated).

---

## 14. References

**In `papers/` (verified — physically present):**
- Vu & Nguyen. *Angular Steering: Behavior Control via Rotation in Activation Space.* (arXiv:2510.26243)
- *Activation Steering with a Feedback Controller* (**PID Steering**; Nguyen, Vu, Pham, Zhang, Nguyen).
  ICLR 2026. arXiv:2510.04309. *(Your group; owns the depth-axis PID framing.)*
- *Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control* (**A-LQR**).
  arXiv:2604.19018 (per lit-scan; verify). *(Same models; graded win, binary loss — corroborates the hump.)*
- Wollschläger et al. *The Geometry of Refusal in LLMs: Concept Cones and Representational Independence.*
  ICML 2025. *(RDO/RCO retain loss — the cone machinery MOSAIC retargets.)*
- *The Cylindrical Representation Hypothesis in LLMs.* *(Concept = angle+magnitude curve, motivating D1.)*
- *(LoRA paper — D4 background.)*

**Verified via web search (real, IDs confirmed):**
- Chang & Schnabel et al. *A Course Correction in Steerability Evaluation: Revealing Miscalibration and
  Side Effects in LLMs.* AAAI 2026. arXiv:2505.23816. Framework: `github.com/MLD3/steerability`.
  *(The parallel/orthogonal side-effect decomposition + goal-space + dataset MOSAIC targets.)*
- Oozeer, Marks, Barez, Abdullah. *Beyond Linear Steering: Unified Multi-Attribute Control (**K-Steering**).*
  Findings of EMNLP 2025. arXiv:2505.24535. Benchmarks TONEBANK/DEBATEMIX.
- *AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint.* ICLR 2026.
  arXiv:2506.07022. Code: `github.com/AlphaLab-USTC/AlphaSteer`.
- *MSRS / MAT-Steer: Adaptive Multi-Subspace Representation Steering for Attribute Alignment.*
  arXiv:2508.10599. *(Orthogonal-subspace multi-attribute — the closest competitor.)*
- Wu et al. *AxBench: Steering LLMs? Even Simple Baselines Outperform SAEs.* ICML 2025. arXiv:2501.17148.
  *(Prompting/DiffMean > SAEs; mandatory baselines.)*
- Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024.

**From lit-scan (real titles; confirm arXiv IDs before citing in a paper):**
- *In-Distribution Steering (IDS): Balancing Control and Coherence.* arXiv:2510.13285. *(SPI; the
  on-manifold-constrained baseline — test if its constraint binds on a graded attribute.)*
- *Precise Attribute Intensity Control via Targeted Representation Editing (**Pre-Control**).*
  arXiv:2510.12121. Code: `github.com/Pre-Control/pre-control`. *(Graded-setpoint SOTA baseline.)*
- Scialom et al. *Multi-property Steering with Dynamic Activation Composition (**DAC**).* BlackboxNLP @
  EMNLP 2024. arXiv:2406.17563. *(KL-to-unsteered intensity — your CALM already validated this.)*
- *From Steering Vectors to Conceptors: Compositional Affine Activation Steering.* NeurIPS 2025.
- Wu et al. *ReFT: Representation Finetuning.* NeurIPS 2024. arXiv:2404.03592. *(LoRA-arm actuator.)*
- *On the Effectiveness-Fluency Trade-Off in LLM Conditioning.* *(TTR ρ=0.81, perplexity ρ=0.00 — metric
  choice; cross-check with* A Sober Look at Steering Vectors *.)*
- *What Can We Actually Steer? A Multi-Behavior Study* (arXiv:2511.18284); Braun et al. *Understanding
  (Un)Reliability of Steering Vectors* (arXiv:2505.22637); Tan et al. *Analysing Generalisation and
  Reliability of Steering Vectors.* *(Saturation/geometry/reliability — motivation, not novelty.)*

---

*Provenance: synthesized by an adversarial design workflow (25 agents) that read the 6 reference papers,
the 7 prior proposals, the reusable codebase, and a 2024–2026 literature sweep; ranked 5 candidate
framings by a control-skeptic / novelty / feasibility panel (MOSAIC and SARTRE tied at the top); and
integrated SARTRE's reverting-disturbance mechanism + KEEL's hard-KL budget into the winner. See
`FINDINGS_SYNTHESIS.md` for the meta-finding this proposal is engineered to clear.*
