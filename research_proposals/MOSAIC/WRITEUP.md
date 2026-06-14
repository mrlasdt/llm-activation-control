# When is control theory load-bearing for activation steering? Open-loop nulls and a closed-loop win

*MOSAIC — Multi-Objective Steering via Allocation In Cones. A complete technical report on the arc
Phases 0–4 + cross-model replication. Gemma-2-2b-it and Qwen2.5-3B-Instruct, single A10G.
Date: 2026-06-14.*

> **Note on scope.** De-refusal experiments referenced as *prior context* were authorized
> safety/steering research on open-weights models; **this report's experiments steer benign graded text
> attributes** (formality, reading-level, sentiment). All numbers are quoted from the run artifacts
> (`outputs/mosaic_phase{0..4}_*.json`, `*.png`) and cross-checked; see §9.

---

## Abstract

Activation steering controls LLM behaviour by adding a vector to the residual stream. A natural question
is whether *control theory* — feedback, predictive control, constrained optimization — buys anything
beyond the two crude levers everyone already uses: the **actuator** (what you add) and the **bound** (how
much). A prior program of ours answered "no" three times for de-refusal: multi-step lookahead (PTS), a
coherence-aware cost (CALM 1–2), and dynamic per-token feedback (CALM 3) each merely **tied** a
well-chosen fixed magnitude, because refusal is a binary behaviour that *saturates* (a hump, not a
frontier). MOSAIC tests the regime those nulls lacked: **graded, non-saturating, multi-attribute** control
— specifically, *drive one attribute to a target while holding another at baseline* — where a binding
hold-constraint could, in principle, make a control law beat the bound.

We find a sharp two-part result. **(1) Open-loop control still ties.** A constrained *null-space
allocation* QP (push attribute A inside its concept cone, orthogonal to attribute B's readout) does **not**
hold B any tighter than the best scalar push: scalar and allocation lie on **one shared (gain, drift)
frontier**. The reason is mechanistic — for the strongly-coupled pair (formality ↔ reading-level), the
coupling is a **semantic entanglement** ("formal" entails "lexically complex"), not a removable linear
subspace; an open-loop per-layer projection cannot veto re-entanglement that the autoregressive process
re-introduces downstream. This is a **fourth** instance of "control sophistication ties the lever." **(2)
Closed-loop control wins.** A controller that *measures the realized attribute as it generates* and pushes
back holds the off-target attribute **>1 SE tighter at matched driven-gain** — the first time in the
program a control law beats the open-loop lever. It replicates on a second model family **more
decisively** (Qwen2.5-3B: 4–6 SE, near-zero coherence cost; Gemma-2-2b: ~1.7 SE at a modest coherence
tax). Two by-products are independently useful: a *supervised, attribute-specific steering direction*
removes ~80% of the side-effect that difference-in-means produces (the side-effect is mostly a
bad-direction artifact); and the optimal feedback gain scales predictably with the residual-stream scale.

The general principle: **control becomes load-bearing for steering exactly when (1) the target is
graded/non-saturating, (2) the side-effect accumulates over the token axis, and (3) a faithful output
sensor exists — and then it must be closed-loop (output feedback), not open-loop (lookahead, coherence
cost, or static allocation).**

---

## 1. The question

Steering adds `α·v` to the residual stream. The field's working knobs are the **direction** `v` (usually
difference-in-means between contrastive prompts) and the **magnitude** `α` (a swept scalar). A recurring
hope — including in our own group's prior work casting steering as PID/LQR over layers (Vu & Nguyen; the
A-LQR line) — is that *control theory* adds value: predictive horizons, constraints, feedback. **Does it,
beyond the actuator and the bound?**

Our prior program answered "no" for **de-refusal** three times (§2). The honest next question is not "does
control help?" but **"in which regime does control become load-bearing?"** MOSAIC isolates a regime the
nulls never tested: **graded multi-attribute control with a binding hold-constraint.**

## 2. Background — three open-loop nulls (the wall MOSAIC must clear)

| experiment | control sophistication added | actuator/axis | outcome |
|---|---|---|---|
| PTS | multi-step predictive lookahead (MPC horizon) | rotation / depth | **ties** — late-band layer dynamics near-identity (‖A−I‖≈0.34), so H=1 ≈ H=8 |
| CALM 1–2 | coherence cost inside the MPC objective | additive / depth | **ties** — the cone-space density it optimizes does not predict fluency (r=0.20) |
| CALM 3 | dynamic per-token feedback on ambient KL | additive / token | **ties** — refusal saturates; reallocating a fixed aggregate in time changes nothing (best Δ +0.018) |

Across all three, the strength↔outcome trade-off is governed by the **aggregate** intervention, not its
allocation in depth or time. The durable levers were the **actuator** (additive ≫ norm-preserving
rotation) and the **bound** (`u_max`); the controller was a wash. The diagnosis named three conditions
that *would* make control load-bearing: a **non-saturating monotone** target; a **binding** constraint;
and **evolving token dynamics** where feedback has predictive value. MOSAIC was designed to deliver all
three.

## 3. MOSAIC — hypothesis, testbed, metrics

**Hypothesis (original).** On *graded* attributes whose concept cones are non-orthogonal (superposition),
driving attribute A to a setpoint with a scalar push mechanically drags B off its setpoint (the
"side-effect" failure mode that dominates steerability evaluations; Course-Correction, AAAI 2026). A
*constrained allocation* QP that uses the A-cone's redundant degrees of freedom (its null-space) to hit A
while holding B should beat a scalar push on the side-effect axis — a control law beating the bound.

**Testbed.** Three graded attributes, each with a deterministic or learned readout:
- **formality** — `s-nlp/roberta-base-formality-ranker` P(formal);
- **reading-level** — Flesch-Kincaid grade (deterministic, dependency-free);
- **sentiment** — `distilbert-base-uncased-finetuned-sst-2-english` P(positive).

We deliberately selected the **most strongly-coupled pair** so a hold-constraint would bind. Prompts:
neutral open-ended instructions (Alpaca harmless split). Actuator: bounded **additive** push on the
residual-stream output of `model.layers.{l}` across a band, the program's validated lever. **Models:**
Gemma-2-2b-it (26 layers, band [7,24], pscale 180.9) and Qwen2.5-3B-Instruct (36 layers, band [10,33],
pscale 58.2). `pscale` is the typical attribute mean-difference magnitude, so a push of `frac·pscale` is
the same ~4–12% relative perturbation on both models.

**Coherence metric.** We use **distinct-2** / type-token ratio (degeneration/repetition), *not* perplexity
(ρ≈0 with fluency) and *not* genNLL — genNLL conflates "attribute changed" (the intended signal, which
makes text less likely under the unsteered model) with incoherence, so it is the wrong axis for graded
steering. Sentence-level samples were inspected to confirm coherent operating points are genuine, not
degenerate.

## 4. The open-loop arc (Phases 0–3): preconditions met, allocation still ties

### 4.1 Phase 0 — the binding precondition holds
Quick difference-in-means directions for the three attributes; measure whether a scalar push on one drags
the others. **GO:** formality ↔ reading-level direction-cosine **+0.50** (the deliberately-interfering
pair), vs formality↔sentiment −0.26. Induced-drift matrix at a coherent operating point (n=40): driving
formality moves reading **+3.7 FK (3.9σ)** while leaving sentiment flat (1.7σ); every off-target cell
clears >1 SE, so the hold-constraint is active. A hold-feasible push direction exists for every attribute
(orthogonal residual 0.84–0.96 ≫ 0.3). A separate **reversion** probe — steer formality, release — showed
it drifts back over the generation (α≈0.40), i.e. holding is a genuine sustained-disturbance problem.

### 4.2 Phase 1 — graded sensors and a monotone frontier (not a hump)
A linear residual-stream probe predicts the external attribute score at held-out Pearson **r = 0.91
(formality), 0.94 (reading-level), 0.73 (sentiment)**. So faithful per-token sensors exist for formality
and reading-level; sentiment is only weakly linearly readable (and saturates near its classifier ceiling)
and is **demoted to a hold-only / dropped** attribute. Sweeping push magnitude, driven-attribute intensity
rises **strictly monotonically** over the coherent range (formality and reading Spearman = 1.00) — a
genuine strength↔coherence **trade-off frontier, not the refusal hump.** Coherent envelope: push ≲
0.075–0.1·pscale; beyond it the text degenerates. **This is the non-saturating regime the prior nulls
lacked** — so the headline pair is formality ↔ reading-level, which both pass the gates *and* is the
most-coupled pair.

### 4.3 Phase 2 — attribute-specific cones, a depth-indexed reference, and a free lever
- **Attribute-specific k>1 cone (GO).** A supervised PLS cone reconstructs intensity far better than a
  blunt PCA span (held-out R²: formality **0.93 vs 0.75**, reading **0.92 vs 0.53**) — a training-free
  analog of the retain-loss concept cone. (Refinement of our earlier "blunt k>1 is gibberish" finding:
  that was an *ablation* phenomenon; for *additive* setpoint steering both cones move the attribute
  coherently, so PLS wins on conditioning, not on raw coherence. And k>1 does not out-*steer* k=1 — its
  value is the null-space for the hold.)
- **Depth-indexed reference beats a single vector (GO).** A per-layer, full-band reference reaches a
  commanded intensity with lower coherent miscalibration than a single difference-in-means vector at one
  layer (formality 0.120 vs 0.173; reading 1.77 vs 3.63). Plant R²(1-step) 0.96/0.98 (near-identity ⇒ H>1
  inert, pre-registered).
- **A free side-effect lever (the durable practical finding).** A *supervised attribute-specific (PLS)
  direction* causes **~80% less cross-drift than raw difference-in-means at matched attribute gain**
  (drift-per-unit-gain 2.5 vs 12.6). Much of the side-effect the field worries about is a *bad-direction
  artifact*, removed for free by a better direction — before any controller. We also found a fragile-method
  lesson: a setpoint actuator that drives the coordinate to a target *fit from reading attribute-laden
  text* moves the attribute the wrong way (the representation of "having read formal text" ≠ "about to
  write formally"); all measurements use the robust additive push instead.

### 4.4 Phase 3 — the open-loop allocation null, and why
We solve, per band layer, the closed-form constrained allocation: push inside the A-cone, orthogonal to
B's readout direction, bounded — `s* = aₐ − (⟨aₐ,q⟩/⟨q,q⟩)q`, `q = Uₐ·g_B`. Arms: **scalar** (best PLS
push), **alloc** (in-cone ⊥ B), **repind** (single decoupled direction). **Result: SCOPED NULL.** At
matched formality-gain, allocation's reading-drift ties scalar at every coherent point (2.78 vs 3.09;
2.24 vs 2.28; paired `scalar − alloc = −0.17 ± 0.52 SE`), and allocation *sacrifices range* (coherent
formality-gain caps at +0.150 vs scalar's +0.235). **Scalar, allocation, and single-direction-decoupling
all lie on one shared (gain, drift) frontier.**

**Mechanism — semantic entanglement, not a removable subspace.** Three lines of evidence: (i) the shared
frontier; (ii) *symmetric inseparability* — orthogonalizing the *reading* push to formality destroys the
reading effect (you cannot move reading without formality either); (iii) *downstream re-entanglement* —
the open-loop projection zeroes B's coordinate *at the band layers*, but autoregressive generation
re-introduces the complexity because "write formally" entails "write complexly," which a per-layer linear
projection cannot veto. **MOSAIC's original headline (static null-space allocation > scalar) is falsified
for semantically-entangled attributes** — a fourth "control sophistication ties the lever," now because
the coupling is not linearly separable.

## 5. The closed-loop win (Phase 4) and cross-model replication

The mechanism (downstream re-entanglement is a *generation-time* phenomenon) points to the fix:
**output feedback.** A chunked closed-loop controller (4 chunks × 32 tokens, STU-PID style) drives
formality up (fixed `m_F·g_F`) while **regulating the measured reading-level back to its per-prompt
unsteered baseline**: `u_l = m_F·g_F,l − κ·e_R·g_R,l`, `e_R = FK(text-so-far) − target_FK`, bounded
‖u_l‖ ≤ u_max. The sensor is the **exact Flesch-Kincaid score** (free, deterministic) — the best-case
sensor, so a null would have been the strongest possible negative; instead:

**WIN (Gemma-2-2b).** At matched formality-gain, closed-loop (κ=4) holds reading **>1 SE tighter** than
open-loop drive: |reading-drift| **3.66 → 1.69**; matched-gain paired difference **1.25 ± 0.72 SE
(≈1.7 SE)**; the closed-loop (gain, drift) frontier **dominates** the open-loop frontier. The regulation
trace confirms genuine control — drive-only's reading-error stays flat (3.8→3.7), closed-loop drives it
down (4.7→2.4). **This is the first controller-beats-the-lever result across PTS → CALM 1–3 → MOSAIC P3
(all tie) → MOSAIC P4 (win).** It is *not* a free lunch: a **coherence tax** (distinct-2 0.95 → 0.86 at
κ=4) and a **narrow stable gain band** — κ=16 over-corrects (reading-drift climbs back, coherence craters
to 0.63): classic high-gain instability. It is a 3-way (drive, hold, coherence) trade-off where feedback
buys the hold by spending some coherence; optimal κ≈4.

**Replication (Qwen2.5-3B) — WIN, more decisively.** Prerequisites reproduce (Phase 2 on Qwen: PLS recon
R² 0.92/0.94 ≫ PCA 0.62/0.73; depth-indexed beats single-vector; coupling present). Phase 4 on Qwen: at
matched formality-gain, closed-loop (κ=1) holds reading **~2.5× tighter** (|drift| **2.91 → 1.44**),
paired **1.68 ± 0.32 SE (5.3 SE)**, ranging **3.7–6.0 SE** across operating points — far more significant
than Gemma's 1.7 SE — and at the κ=1 sweet spot, **near-baseline coherence** (distinct-2 0.94–0.95 vs
baseline 0.97) with ~90–95% of the formality-gain preserved: a near-clean Pareto win, a *smaller*
coherence tax than Gemma. Same mechanism (e_R regulates down; high-κ over-corrects). The optimal gain
**κ≈1 on Qwen vs ≈4 on Gemma** tracks the pscale ratio (58 vs 180) — the hold push `κ·e_R·g_R` is in
FK-grade units, so the balanced gain scales ~inversely with pscale, exactly as predicted. **The result
holds on two model families** (the cross-family reliability bar; Tan et al.).

## 6. The general principle

Combining the three open-loop nulls with the closed-loop win:

> **Control sophistication is a wash for steering when applied open-loop** — predictive lookahead, a
> coherence-aware cost, multi-layer null-space allocation, or a single decoupled direction all lie on the
> same strength↔side-effect frontier as a well-chosen fixed push, because the trade-off is governed by the
> *aggregate* intervention. **Closed-loop output feedback is load-bearing** — it provably reaches operating
> points open-loop cannot — *exactly when* (1) the target is **graded and non-saturating** (a real
> frontier, not a hump), (2) the disturbance/side-effect **accumulates over the token axis** (so feedback
> has something to correct that compounds), and (3) a **faithful output sensor** exists. The lever was
> never lookahead, coherence cost, or static allocation; it is **feedback on a generation-time signal.**

Corollaries: the actuator (additive ≫ rotation) and the bound remain primary; and the *direction* matters
more than the controller for one-shot side-effects (a supervised attribute-specific direction removes
~80% of them for free).

## 7. Related work & positioning

- **Control-theoretic steering.** PID Steering (Vu & Nguyen; layer-axis P/I/D) and the A-LQR / Local
  Linearity line cast steering as control over **depth**, with *offline* gains — they do not close a loop
  over generated tokens, and both report their gains help on graded toxicity but not on binary jailbreak,
  corroborating our saturation diagnosis on the *same* models. Our contribution is orthogonal: a
  **closed-loop over the token axis** with an output sensor, and the open-loop-vs-closed-loop dividing line.
- **Multi-attribute / null-space steering.** K-Steering (classifier-gradient composition), Conceptors
  (affine composition), MAT-Steer / MSRS (orthogonal-subspace multi-attribute), and AlphaSteer
  (null-space-for-utility on binary refusal) all operate *open-loop*. Phase 3 shows open-loop null-space
  allocation cannot separate semantically-entangled graded attributes — a result that *bounds* this line
  and motivates output feedback.
- **Dynamic / closed-loop steering.** DAC (per-token KL intensity), DSAS/PIXEL (per-token magnitude),
  IDS (on-manifold constraint), Pre-Control (target-reaching value function), FASB/SVF (deviation-adaptive
  / refreshed directions), STU-PID (per-chunk PID for reasoning length). MOSAIC's novelty vs these: a
  *multi-attribute hold* via output feedback, framed as the open-loop-tie / closed-loop-win contrast with
  a mechanistic (semantic-entanglement) explanation, validated cross-family.
- **Steerability evaluation.** Course-Correction (AAAI 2026) names the side-effect problem MOSAIC targets;
  AxBench shows simple baselines (prompting, diff-in-means) beat SAEs — we use diff-in-means/PLS, not SAEs.
- **What we do NOT claim as novel:** "steering = control theory," LQR/PID over layers, null-space or
  orthogonal multi-attribute steering, per-token dynamic magnitude, or KL-as-coherence — all prior art.
  (Citations + arXiv IDs in `MOSAIC_PROPOSAL.md` §14; several future-dated IDs there are flagged "verify.")

## 8. Limitations & threats to validity

1. **Two attributes, one pair.** We chose the most-coupled pair (formality ↔ reading-level), which is also
   the most *semantically entangled* — the hardest case for open-loop allocation (Phase 3) and a strong
   case for closed-loop (Phase 4). A *coupled-but-linearly-separable* pair could allocate open-loop; we did
   not find/test one. Sentiment failed the probe gate and is hold-only.
2. **Sample size / significance.** n=40 (Phases 0–3) / n=32 (Phase 4). The Gemma closed-loop win is ≈1.7
   SE (suggestive); the Qwen win is 4–6 SE (solid). Larger n and a third family would harden Gemma.
3. **Chunked, not per-token.** Phase 4 uses 4×32-token chunks (feedback between chunks), not a true
   per-token loop; a verified per-token decode loop (`calm_token.decode_dual`) is the next step.
4. **Hold-to-baseline only.** We regulate B to its unsteered baseline, not to an arbitrary commanded τ_B
   (the full MIMO setpoint-tracking is untested).
5. **Coherence tax + gain stability.** The win costs some coherence (less on Qwen) and lives in a narrow
   κ band (high κ is unstable). A deployable controller needs gain-scheduling / anti-windup.
6. **Exact sensor.** Reading-level uses the exact FK score; the general-attribute path uses the Phase-1
   probe (r=0.91–0.94), which is noisier — the OAS Kalman-observer machinery is the on-hand mitigation.
7. **Greedy decoding, instruct models, English.** Generalization to sampling, base models, and other
   languages is untested.

## 9. Reproducibility

All code self-tested (`python mosaic_phase{0..4}.py --selftest`); experiments are greedy/deterministic.
Pipeline: `mosaic_phase0.py --run` (coupling gate) → `mosaic_phase1.py --run` (sensors + frontier) →
`mosaic_phase2.py --run` (cones + maps; saves `mosaic_cones_<model>.npz`) → `mosaic_phase3.py --run`
(open-loop allocation) → `mosaic_phase4.py --run` (closed-loop). Second family: write the band/pscale
meta (`casa_frontier_<model>.json`), then `--model Qwen/Qwen2.5-3B-Instruct` for Phases 2 & 4 (κ grid
scaled to pscale). Artifacts + per-phase analysis in `PHASE{0,1,2,3,4}_RESULTS.md`; figures in
`outputs/mosaic_phase{1,2,3,4}_<model>.png`. Reuses the program's shared infra (`pytorch_pure/`, CASA
cone/actuator/plant, CALM decode loop).

## 10. Conclusion

Asking "does control theory help activation steering?" the honest, falsifiable answer from this program is
a dividing line, not a yes/no. **Open-loop control — however sophisticated — ties a well-chosen fixed push
on the strength↔side-effect frontier**, even for graded multi-attribute targets with a binding constraint,
because semantically-entangled attributes are not linearly separable and the trade-off is set by the
aggregate intervention. **Closed-loop output feedback breaks off that frontier** — it holds an off-target
attribute >1 SE tighter at matched driven-gain, replicating across two model families (more decisively on
Qwen) — because the side-effect accumulates over generation and only a loop with a faithful sensor can
observe and correct it. Control becomes load-bearing for steering precisely when the target is graded, the
side-effect accumulates over the token axis, and a faithful output sensor exists — and then it must be
closed-loop. The durable open-loop levers remain the actuator, the bound, and a good (supervised)
direction.
