# When is control theory load-bearing for activation steering? Open-loop nulls and a closed-loop win

*MOSAIC — Multi-Objective Steering via Allocation In Cones. A complete technical report on the arc
Phases 0–6 + cross-model replication. Gemma-2-2b-it and Qwen2.5-3B-Instruct, single A10G.
Date: 2026-06-14. All numbers cross-checked against the run artifacts by an adversarial verification
pass; remaining hedges are deliberate (see §8).*

> **Note on scope.** De-refusal experiments referenced as *prior context* were authorized
> safety/steering research on open-weights models; **this report's experiments steer benign graded text
> attributes** (formality, reading-level, sentiment). Numbers are quoted from
> `outputs/mosaic_phase{0..4}_*.json/.png`, `PHASE*_RESULTS.md`, and the coupling-check
> (`mosaic_phase2_coupling_check.py`); see §9.

---

## Abstract

Activation steering controls LLM behaviour by adding a vector to the residual stream. A natural question
is whether *control theory* — feedback, predictive control, constrained optimization — buys anything
beyond the two crude levers everyone already uses: the **actuator** (what you add) and the **bound** (how
much). A prior program of ours answered "no" three times for de-refusal: multi-step lookahead (PTS), a
coherence-aware cost (CALM 1–2), and dynamic per-token feedback (CALM 3) each merely **tied** a
well-chosen fixed magnitude, because refusal is a binary behaviour that *saturates* (a hump, not a
frontier). MOSAIC tests the regime those nulls lacked: **graded, non-saturating, multi-attribute** control
— *drive one attribute to a target while holding another at baseline* — where a binding hold-constraint
could, in principle, make a control law beat the bound.

We find a two-part result. **(1) Open-loop control still ties.** A constrained *null-space allocation* QP
(push attribute A inside its concept cone, orthogonal to attribute B's readout) does **not** hold B
tighter than the best scalar push at the tested matched gain (paired −0.17 ± 0.52 SE); allocation, scalar,
and a single decoupled direction sit on essentially **one (gain, drift) frontier** (the non-scalar arms
trend slightly tighter but within noise). For the strongly-coupled pair (formality ↔ reading-level) the
coupling is plausibly a **semantic entanglement** ("formal" entails "lexically complex") rather than a
removable linear subspace — our leading hypothesis (§4.4), alongside the milder reading that the residual
coupling is simply small once a good direction is used. **(2) Closed-loop control wins.** A controller
that *measures the realized attribute as it generates* and pushes back holds the off-target attribute
tighter at matched driven-gain. The effect is **decisive on Qwen2.5-3B** (paired **3.7–6.0 SE** across
operating points, near-zero coherence cost) and **directionally consistent but weak on Gemma-2-2b**
(~**1.7 SE**, one paired test, two-sided p≈0.08, at a modest coherence tax) — to our knowledge the
program's first controller-beating-the-open-loop-lever result, carried by the cross-family replication.
Two by-products are independently useful: a *supervised (PLS) steering direction* produces **~80% less
side-effect per unit attribute-gain than raw difference-in-means** (the side-effect is largely a
bad-direction artifact); and the optimal feedback gain scales roughly with the residual-stream scale
(partly confounded by an actuator-bound difference between models, §5).

Two follow-ups **bound** the closed-loop win. **(3) Loop rate is a minor lever (Phase 5).** A *true
per-token* loop does not beat the chunked one by reacting to fresher error; raw per-token updates cause
**actuator chatter** (tighter hold only by going incoherent), and the control-theoretic fix —
**slew-limiting the feedback** — recovers coherence and edges out the chunked loop by only ~1.5–1.7 SE in
one operating cell. The control *signal's smoothness*, not its update frequency, is what matters. **(4) The
*coherent* reachable set is authority-bounded (Phase 6).** Extending from *hold-at-baseline* to full 2-DoF
setpoint *tracking*, neither a diagonal nor a model-based decoupling controller places both attributes at
arbitrary mutually-opposing setpoints within the coherent push budget: the two style actuators' output
effects are **88 % collinear on Gemma (cond G ≈ 4) and 94 % on Qwen (cond G ≈ 7)** — the quantitative,
cross-family form of the semantic entanglement. The limit is *moderate* ill-conditioning × the authority
bound (G is full-rank — reachable in principle with more authority, at a coherence cost), and under that
ill-conditioning the "smart" decoupling controller ties or (on Qwen) actively *underperforms* the naive
one.

Working principle (a hypothesis this program supports, not a proven law; §6): **control becomes
load-bearing for steering when (1) the target is graded/non-saturating, (2) the side-effect accumulates
over the token axis, and (3) a faithful output sensor exists — and then it must be closed-loop (output
feedback), not open-loop (lookahead, coherence cost, or static allocation) — but the win is local:
disturbance-rejection back to baseline, not arbitrary setpoint placement, which the plant's actuator
collinearity bounds.**

---

## 1. The question

Steering adds `α·v` to the residual stream. The field's working knobs are the **direction** `v` (usually
difference-in-means between contrastive prompts) and the **magnitude** `α` (a swept scalar). A recurring
hope — including our own group's prior work casting steering as PID/LQR over layers (Vu & Nguyen; the
A-LQR / "Local Linearity" line) — is that *control theory* adds value: predictive horizons, constraints,
feedback. **Does it, beyond the actuator and the bound?** Our prior program answered "no" for
**de-refusal** three times (§2). The honest next question is **"in which regime does control become
load-bearing?"** MOSAIC isolates a regime the nulls never tested: **graded multi-attribute control with a
binding hold-constraint.**

## 2. Background — three open-loop nulls (the wall MOSAIC must clear)

| experiment | control sophistication added | actuator/axis | outcome |
|---|---|---|---|
| PTS | multi-step predictive lookahead (MPC horizon) | rotation / depth | **ties** — late-band layer dynamics near-identity (‖A−I‖≈0.34), H=1 ≈ H=8 |
| CALM 1–2 | coherence cost inside the MPC objective | additive / depth | **ties** — cone-space density does not predict fluency (r=0.20) |
| CALM 3 | dynamic per-token feedback on ambient KL | additive / token | **ties** — refusal saturates; reallocating a fixed aggregate in time changes nothing (best Δ +0.018) |

Across all three the strength↔outcome trade-off is governed by the **aggregate** intervention, not its
allocation in depth or time. Durable levers: the **actuator** (additive ≫ norm-preserving rotation) and
the **bound** (`u_max`); the controller was a wash. The diagnosis named three conditions that *would* make
control load-bearing: a **non-saturating monotone** target; a **binding** constraint; **evolving token
dynamics** where feedback has predictive value. MOSAIC was designed to deliver all three.

## 3. MOSAIC — hypothesis, testbed, metrics

**Hypothesis (original).** On *graded* attributes whose concept cones are non-orthogonal (superposition),
driving A to a setpoint with a scalar push drags B off its setpoint (the "side-effect" failure mode that
dominates steerability evaluations; Course-Correction). A *constrained allocation* QP that uses the
A-cone's redundant DoF (its null-space) to hit A while holding B should beat a scalar push on the
side-effect axis.

**Testbed.** Three graded attributes with readouts: **formality** (`s-nlp/roberta-base-formality-ranker`
P(formal)); **reading-level** (Flesch-Kincaid grade, deterministic); **sentiment**
(`distilbert-base-uncased-finetuned-sst-2-english` P(positive)). We deliberately selected the **most
strongly-coupled pair** so a hold-constraint would bind. Prompts: neutral Alpaca-harmless instructions.
Actuator: bounded **additive** push on the residual-stream output of `model.layers.{l}` across a band.

**Models & hyperparameters.** Gemma-2-2b-it (26 layers, band [7,24], pscale 180.9) and Qwen2.5-3B-Instruct
(36 layers, band [10,33], pscale 58.2). `pscale` is the typical attribute mean-difference magnitude, so a
push of `frac·pscale` is the same ~4–12% relative perturbation on both models. **The actuator bound
differs between the Phase-4 runs: `u_max = 0.15·pscale` (Gemma) vs `0.20·pscale` (Qwen)** — a second
uncontrolled cross-model difference (§5, §8).

**Coherence metric.** We use **distinct-2** / type-token ratio (degeneration/repetition), *not* perplexity
(ρ≈0 with fluency) and *not* genNLL — genNLL conflates "attribute changed" (the intended signal, which
makes text less likely under the unsteered model) with incoherence. Sentence-level samples were inspected
at coherent operating points; full-length per-condition samples and output-length logging would
strengthen this (§8).

## 4. The open-loop arc (Phases 0–3): preconditions met, allocation still ties

### 4.1 Phase 0 — the binding precondition holds
Quick difference-in-means directions; does a scalar push on one attribute drag the others? **GO:**
formality ↔ reading-level direction-cosine **+0.50** (the deliberately-interfering pair), vs
formality↔sentiment −0.26. Induced-drift matrix at a coherent operating point (n=40): driving formality
moves reading **+3.7 FK (3.9σ)** and sentiment a much smaller **+0.13 (1.7σ)**; every off-target cell
clears >1 SE (so the hold-constraint is active, though the formality↔sentiment coupling is comparatively
weak). A hold-feasible push direction exists for every attribute (orthogonal residual 0.84–0.96 ≫ 0.3). A
**reversion** probe — steer formality, release — shows it drifts back over the generation (α≈0.40): holding
is a genuine sustained-disturbance problem.

### 4.2 Phase 1 — graded sensors and a monotone frontier (not a hump)
A linear residual-stream probe predicts the external attribute score at held-out Pearson **r = 0.91
(formality), 0.94 (reading-level), 0.73 (sentiment)**. Faithful per-token sensors exist for formality and
reading-level; sentiment is only weakly linearly readable (and saturates near its classifier ceiling) and
is **demoted to hold-only / dropped**. Sweeping push magnitude, driven-attribute intensity rises
**strictly monotonically** over the coherent range (formality and reading Spearman = 1.00) — a genuine
strength↔coherence **trade-off frontier, not the refusal hump.** Coherent envelope: push ≲
0.075–0.1·pscale. So the headline pair is formality ↔ reading-level, which both pass the gates *and* is the
most-coupled pair.

### 4.3 Phase 2 — attribute-specific cones, a depth-indexed reference, and a free lever
- **Attribute-specific k>1 cone (GO).** A supervised PLS cone reconstructs intensity far better than a
  blunt PCA span (held-out R²: formality **0.93 vs 0.75**, reading **0.92 vs 0.53**) — a training-free
  analog of the retain-loss concept cone. (Refinement of our earlier "blunt k>1 is gibberish" finding:
  that was an *ablation* phenomenon; for *additive* setpoint steering both cones move the attribute
  coherently, so PLS wins on conditioning, not raw coherence. k>1 does not out-*steer* k=1 — its value is
  the null-space for the hold.)
- **Depth-indexed reference beats a single vector (GO).** A per-layer, full-band reference reaches a
  commanded intensity with lower coherent miscalibration than a single difference-in-means vector at one
  layer (formality 0.120 vs 0.173; reading 1.77 vs 3.63). Plant R²(1-step) 0.96/0.98 (near-identity ⇒ H>1
  inert, pre-registered).
- **A free side-effect lever — the *direction*, from the coupling check** (`mosaic_phase2_coupling_check.py`,
  `PHASE2_RESULTS.md §3`; *not* the phase-2 JSON's RepInd field). Steering formality along a **supervised
  PLS direction** vs the **raw difference-in-means** direction, the induced reading-drift **per unit
  formality-gain** is **2.5 (PLS) vs 12.6 (diff-in-means)** at a coherent push (frac 0.05) — a **~80%
  reduction** in side-effect just from a better direction (drift/gain, which normalizes the small gain
  difference between the arms). Most of the side-effect the field worries about is a *bad-direction
  artifact*. **Caveat:** this is the PLS-vs-diff-in-means comparison; *static* decoupling of the cone
  against the other attribute's direction (RepInd) is a **different** intervention and did **not** help —
  on Gemma it over-corrected (formality→reading drift +0.08 → −1.23). We also found a fragile-method
  lesson: a setpoint actuator that drives the coordinate to a target *fit from reading attribute-laden
  text* moves the attribute the wrong way; all measurements use the robust additive push instead.

### 4.4 Phase 3 — the open-loop allocation null, and the candidate mechanism
We solve, per band layer, the closed-form constrained allocation: push inside the A-cone, orthogonal to
B's readout direction, bounded — `s* = aₐ − (⟨aₐ,q⟩/⟨q,q⟩)q`, `q = Uₐ·g_B`. Arms: **scalar** (best PLS
push), **alloc** (in-cone ⊥ B), **repind** (single decoupled direction). **Result: SCOPED NULL.** At the
tested matched formality-gain (g≈0.12) the alloc-vs-scalar paired difference is **−0.17 ± 0.52 SE** (a
tie); allocation also *sacrifices range* (coherent formality-gain caps at +0.150 vs scalar's +0.235).
Across the coherent sweep the non-scalar arms (alloc, repind) **trend slightly tighter than scalar but
within noise** — we report a tie, not a clean dominance either way. The single paired test was run for
alloc only at one matched gain; we do **not** claim a strict "all three on one frontier" beyond that.

**Candidate mechanism (leading hypothesis): semantic entanglement.** For formality ↔ reading-level the
coupling may be a genuine entanglement ("formal" entails "lexically complex"), not a removable linear
subspace. Evidence: (i) no arm clearly separates the attributes at matched gain; (ii) **symmetric
inseparability** — orthogonalizing the *reading* push to formality collapses the reading effect to ~0 or
negative (in the artifacts), i.e. you cannot move reading without formality either; (iii) **downstream
re-entanglement** — a *conjecture* (not directly measured) that the autoregressive process re-introduces
the complexity that the open-loop per-layer projection removed. **Honest alternative:** the Phase-3 JSON
verdict frames it more mildly — *residual coupling is simply small after a good (PLS) direction* (formality
↔ reading weakly coupled post-PLS; reading→formality near-decoupled). Either way, the operational
conclusion stands: **open-loop static allocation does not beat the scalar push here.** MOSAIC's original
headline (static null-space allocation > scalar) is **not supported** — a fourth "control sophistication
ties the open-loop lever."

## 5. The closed-loop win (Phase 4) and cross-model replication

The candidate mechanism (re-entanglement is a *generation-time* phenomenon) points to **output feedback.**
A chunked closed-loop controller (4 chunks × 32 tokens, STU-PID style) drives formality up (fixed
`m_F·g_F`) while **regulating the measured reading-level back to its per-prompt unsteered baseline**:
`u_l = m_F·g_F,l − κ·e_R·g_R,l`, `e_R = FK(text-so-far) − target_FK`, bounded ‖u_l‖ ≤ u_max. The sensor is
the **exact Flesch-Kincaid score** (free, deterministic) — the best-case sensor, so a null would have been
the strongest possible negative.

**Gemma-2-2b — directional (weak) win.** The closed-loop (gain, drift) frontier lies **below** the
open-loop frontier. At **matched formality-gain (g≈0.148)** the closed-loop holds reading **3.66 → 2.12**
(paired difference **1.25 ± 0.72 SE ≈ 1.7 SE**, two-sided p≈0.08; the best raw closed-loop point reaches
|drift| 1.69 but at a *lower* driven gain 0.122, so it is not the matched-gain comparison). **This is a
single paired test re-expressed via frontier interpolation, not five independent operating points** — so
on Gemma alone it is *suggestive*, not significant at the 0.05 two-sided level. The regulation trace does
confirm genuine control (at m_F=0.06: drive-only reading-error stays flat 4.7→4.5, closed-loop drives it
down 4.7→2.4). **Not a free lunch:** a coherence tax (at the headline m_F=0.04 condition, distinct-2
**0.95 → 0.90 at κ=4**; larger at higher κ) and a narrow stable gain band — **κ=16 over-corrects**
(reading-drift climbs back, distinct-2 craters to ~0.63): classic high-gain instability. (Caution: one
contributing high-drive point, m_F=0.08 κ=4, has distinct-2 0.828, just below our 0.85 coherence floor —
so part of the FK reduction there could be degeneration; the win does **not** rest on it, and the fact
that the *most* degenerate κ=16 point has drift going back **up** argues degeneration does not
monotonically buy the hold. Full-sample inspection is a §8 to-do.)

**Qwen2.5-3B — decisive win (the load-bearing result).** Prerequisites reproduce (Phase 2 on Qwen: PLS
recon R² 0.92/0.94 ≫ PCA 0.62/0.73; depth-indexed beats single-vector; coupling present). Phase 4 on
Qwen, at matched formality-gain, the closed-loop (κ=1) holds reading **~2× tighter** (|drift| **2.91 →
1.44**, i.e. 2.0×; ~2.4× if using the raw κ=1 |drift|=1.23), with paired differences ranging **3.7–6.0 SE
across five interpolated operating points** (1.68 ± 0.32 SE at g=0.088 up to 2.47 ± 0.41 SE at g=0.143) —
genuinely multiple points, since Qwen's drive sweep spans three m_F levels. At the κ≈1–2 sweet spot the
closed-loop preserves ~90–95% of the formality-gain at **near-baseline coherence** (distinct-2 0.94–0.95,
vs the open-loop κ=0 distinct-2 ≈0.96–0.97 reported in the run log) — a *smaller* coherence tax than
Gemma. Same control signature (e_R regulates down; high κ over-corrects).

**Optimal gain across models — consistent with, but not cleanly isolated by, pscale.** The sweet spot is
κ≈4 on Gemma (pscale 180) vs κ≈1 on Qwen (pscale 58). The hold push `κ·e_R·g_R` is in FK-grade units (not
pscale-scaled), so the balanced gain should scale ~inversely with pscale (4/1 ≈ 180/58) — **consistent**
with what we see, but the actuator bound also differs (u_max 0.15 vs 0.20·pscale), giving Qwen more
correction headroom; so this is a suggestive scaling, not a controlled result. A pscale-relative κ would
isolate it (a clean-up for future work). **Net: the closed-loop result holds on two model families** —
decisive on Qwen (3.7–6.0 SE), directionally consistent on Gemma (~1.7 SE) — clearing the cross-family
reliability bar (Tan et al.).

### 5.4 Phase 5 — the true per-token loop: rate is a minor lever, the control *signal* is the variable

Phase 4's win used a chunked loop (re-measure / re-push every 32 tokens). Does a **true per-token** loop —
re-measure FK and re-set the push every generated token, on one KV-cached decode that is token-for-token
identical to `model.generate` under zero push — hold tighter? Phase-4's law is *proportional* (the push is
**set**, not accumulated, from the current error), so the honest variable a per-token loop changes is
**feedback staleness**, not gain. Isolating it (`update_every ∈ {1, 8, 32}`, matched κ, same prompts):

**The naive expectation is false.** At matched κ the *raw* per-token loop (ue=1) holds reading tighter than
the 32-token loop in all 4 Gemma cells by raw |drift|, and **significantly (>1 SE) tighter in 3 of 4** —
but in those 3 cells **only by dropping below the coherence floor** (distinct-2 0.73–0.84; the 4th, κ=4
m_F=0.06, is a coherent tie). (Of the 3, two are *clean* chatter — the per-token point is incoherent while
its chunked reference is coherent; the κ=8/m_F=0.08 cell is ambiguous because the chunked reference is
itself sub-floor.) The working interpretation is **actuator chatter** — updating the steering vector every
token off a noisy per-token FK error perturbs the residual stream more abruptly, degrading fluency (the
steering analogue of bang-bang control wearing the plant) — **but this is a hypothesis, not a measurement:**
we logged the FK-error trace (whose per-token step-size is actually small, ~0.16) but not the *push*
total-variation, so the coherence cost is equally consistent with "the per-token loop simply applies a
larger effective cumulative correction." Either way, at matched κ *and* coherence the cheap chunked loop
weakly dominates (equal hold, better coherence, 32× cheaper).

The control-theoretic fix — **slew-rate-limit / low-pass the feedback** (EMA `e_filt ← 0.8·e_filt +
0.2·e_meas`) — works: it recovers coherence in the cells where raw per-token collapsed, and in the
best-conditioned Gemma cell (m_F=0.08, κ=4) the slew-limited per-token loop holds **1.92 vs the chunked 2.89
at matched coherence** (distinct-2 0.89 vs 0.91, near-matched gain), **paired +0.97 ± 0.57 = +1.7 SE**
(population SE; sample SE +1.66) — a genuine, if suggestive (one cell, same effect size as the Phase-4 Gemma
headline), improvement. (Whether the EMA helps by removing high-frequency chatter or simply by adding lag /
shrinking the push magnitude is not separated here — Qwen, below, hints at the latter.) **Cross-family:** Qwen's raw
per-token loop *stays coherent* (distinct-2 0.86–0.90; its smaller pscale makes per-token push changes
gentler in absolute terms), wins one cell at **+1.5 SE**, and needs no slew limiting (the EMA only adds
lag, loosening the hold).

**Takeaway.** Loop *rate* is a minor lever — a per-token loop edges out the chunked one by ~1.5–1.7 SE *at
best*, in a single operating cell, and only when coherent. What governs whether it stays coherent is the
**smoothness of the control signal** (chatter), not its update frequency: on the larger-pscale model you
must slew-limit, on the smaller one you needn't. The chunked loop is the robust, cheap default on both
families. (Resolves §8 limitation #3; details in `PHASE5_RESULTS.md`.)

### 5.5 Phase 6 — full MIMO setpoint tracking: an authority-bounded reachable set

Phase 4 *held* reading at its baseline. The fully general question: can a closed loop place **both**
(formality, reading) at **arbitrary** commanded setpoints τ — including the anti-correlated
**"formal-but-simple"** corner the entanglement fights — and does a **model-based decoupling** controller
(`a = G⁻¹·K·e`, G the calibrated 2×2 static plant gain) beat a naive **diagonal** one (`diag(G)⁻¹·K·e`)?
Both get correct per-axis scaling; only off-diagonal handling differs.

**The plant is the answer.** Calibrating G (normalized Δoutput per unit actuator command), the two style
actuators' output-effect vectors are **88 % collinear on Gemma (cond G = 3.97)** and **94 % on Qwen (cond G
= 7.23)** — both actuators raise *both* outputs. This near-collinearity is the **quantitative form of the
Phase-3 "semantic entanglement,"** stated as a plant input→output controllability property rather than a
metaphor, and it **replicates and strengthens across families** (Qwen more entangled).

**Result: NULL/BOUNDARY on both families** (at coherent operating points, distinct-2 0.87–0.97 after proper
anti-windup). No setpoint is tracked to low error — formality consistently undershoots its target because
regulating reading and driving formality compete for the same near-collinear push budget within the
coherent authority bound. **This is an *authority* bound, not a rank deficiency:** G is full-rank and cond G
≈ 4 (Gemma) / 7 (Qwen) is only *moderate* ill-conditioning (concern threshold ~30), so the anti-correlated
setpoint is reachable *in principle* — raising u_max would extend reach, but Phase 1 showed u_max ≳
0.15·pscale already costs coherence, so it is the *coherent* reachable set that is bounded.
**Decoupling does not beat diagonal — and is sometimes significantly worse**: on Gemma it *ties* on the
anti-correlated corner (track-err 0.766 vs 0.755, +0.01 SE), is *significantly worse* on `formal_complex`
(−2.0 SE), and wins only one of five setpoints; on Qwen — where cond G is higher — it is **actively worse**
on the anti-corner (1.05 diagonal vs 1.85 decoupling: inverting a more ill-conditioned plant commands a
large opposing push that saturates at u_max and drives reading the *wrong* way, 6.9 target → 11.2). This is
the textbook failure of model-inversion control under **actuator saturation + plant ill-conditioning**. (An
over-driven first Gemma run, Kp=0.5 with windup to ‖a‖=40.7, showed decoupling directionally ahead on the
anti-corner at +1.3 SE — but *incoherently* (distinct-2 0.70); proper anti-windup erases the edge: the
program's "never read a verdict off incoherent text" lesson again.)

**Takeaway.** The closed-loop advantage of Phase 4 is **real but local**: feedback regulates a disturbance
back to baseline, but within the coherent authority budget it does **not** extend to placing two
semantically-entangled attributes at arbitrary, mutually-opposing setpoints. The limit is
**authority × actuator-collinearity (cond G)**, a measured plant property — not a rank-deficiency and not
something the controller can invert away within the coherent budget; a "control sophistication ties (or,
ill-conditioned, *loses to*) the lever" result on the *tracking* axis, the limiting quantity quantified and
replicated. (Resolves §8 limitation #4; details in `PHASE6_RESULTS.md`.)

## 6. The general principle (a working hypothesis)

Combining the three open-loop nulls with the closed-loop win, we propose — as a **conjecture this program
supports**, not a proven law (the evidence is two attributes, one deliberately-most-entangled pair, two
models; §8):

> **Open-loop control is a wash for steering** — predictive lookahead, a coherence-aware cost, multi-layer
> null-space allocation, or a single decoupled direction lie on essentially the same strength↔side-effect
> frontier as a well-chosen fixed push, because the trade-off is governed by the *aggregate* intervention.
> **Closed-loop output feedback is load-bearing** — it reaches operating points open-loop does not —
> *plausibly when* (1) the target is **graded and non-saturating**, (2) the side-effect **accumulates over
> the token axis**, and (3) a **faithful output sensor** exists. The lever is **feedback on a
> generation-time signal**, not lookahead, coherence cost, or static allocation. **But the win is local:**
> it regulates a disturbance back to baseline, and does **not** extend to placing two entangled attributes
> at arbitrary opposing setpoints — that is bounded by **actuator collinearity (cond G)**, a measured plant
> property (Phase 6). And within the closed loop, the control *signal's smoothness* (chatter vs
> slew-limited), not its update *rate*, is the second-order lever (Phase 5).

Corollaries (better supported): the actuator (additive ≫ rotation) and the bound remain primary; the
*direction* matters more than the controller for one-shot side-effects (a supervised PLS direction removes
~80% per unit gain, §4.3); and **model-inversion (decoupling) control is fragile under actuator saturation
+ plant ill-conditioning** — it can underperform a naive diagonal controller (Phase 6, Qwen).

## 7. Related work & positioning

*(The dedicated citation-verification agent did not complete this run; the load-bearing citations below
were spot-checked earlier — AlphaSteer, K-Steering, Course-Correction confirmed real; PID Steering and
Local Linearity are physically in `papers/`. Treat §7 as provisional pending a full citation pass; arXiv
IDs are in `MOSAIC_PROPOSAL.md §14`, several future-dated ones flagged "verify".)*

- **Control-theoretic steering.** PID Steering (Vu & Nguyen) and the A-LQR / Local Linearity line cast
  steering as control over **depth** with *offline* gains — no loop over generated tokens — and both
  report gains that help on graded toxicity but not binary jailbreak, corroborating our saturation
  diagnosis on the *same* models. Our contribution is orthogonal: a **closed-loop over the token axis**
  with an output sensor, and the open-loop-vs-closed-loop dividing line.
- **Multi-attribute / null-space steering.** K-Steering, Conceptors, MAT-Steer / MSRS, AlphaSteer all
  operate *open-loop*. Phase 3 bounds this line: open-loop null-space allocation does not separate the
  (strongly/semantically) coupled graded pair we tested.
- **Dynamic / closed-loop steering.** DAC, DSAS/PIXEL, IDS, Pre-Control, FASB/SVF, STU-PID. MOSAIC's
  angle: a *multi-attribute hold* via output feedback, framed as the open-loop-tie / closed-loop-win
  contrast with a candidate mechanism, replicated cross-family.
- **Steerability evaluation.** Course-Correction names the side-effect problem MOSAIC targets; AxBench
  shows simple baselines beat SAEs (we use diff-in-means/PLS, not SAEs).
- **Not claimed novel:** "steering = control theory," LQR/PID over layers, null-space/orthogonal
  multi-attribute steering, per-token dynamic magnitude, KL-as-coherence — all prior art.

## 8. Limitations & threats to validity

1. **Two attributes, one pair.** The most-coupled (and most *semantically entangled*) pair — hardest for
   open-loop allocation, strong for closed-loop. A *coupled-but-separable* pair could allocate open-loop;
   untested. Sentiment failed the probe gate (hold-only).
2. **Significance.** n=40 (Phases 0–3) / n=32 (Phase 4). **Gemma closed-loop is ≈1.7 SE (p≈0.08, one
   paired test) — suggestive, not significant; the load-bearing evidence is Qwen (3.7–6.0 SE).** Larger n
   and a third family would harden Gemma.
3. **Per-token loop done (Phase 5), with a caveat.** A verified per-token KV-cached loop was built; loop
   *rate* proved a minor lever (chatter, not staleness) and the per-token edge is ~1.5–1.7 SE in one cell.
   The slew-limit β=0.8 and the per-token κ grid were lightly tuned, not swept; a fuller controller-design
   sweep (PI, anti-windup, β) is open.
4. **MIMO tracking done (Phase 6), and bounded.** Arbitrary 2-DoF setpoint tracking was tested and found
   **authority/controllability-limited**; the cond(G) boundary is measured at one probe magnitude and one
   coherent u_max — a reachable-set map over u_max (trading coherence for reach) is the natural next step.
5. **Coherence tax + gain stability + a possible degeneration contribution.** The win costs some coherence
   (less on Qwen); lives in a narrow κ band (high κ unstable); and one high-drive Gemma point dips below
   the coherence floor — full-length samples + output-length logging are needed to fully exclude
   degeneration as a contributor to the FK reduction.
6. **Cross-model confounds.** pscale (180 vs 58) **and** u_max (0.15 vs 0.20·pscale) both differ between
   the Phase-4 runs, so the κ-scaling story is suggestive, not isolated.
7. **Mechanism is a hypothesis.** "Semantic entanglement" / "downstream re-entanglement" is not directly
   measured; the milder "residual coupling is small after a good direction" is a live alternative.
8. **Exact sensor; greedy; instruct models; English.** Reading uses exact FK; the general-attribute path
   uses the noisier Phase-1 probe (the OAS Kalman-observer machinery is the on-hand mitigation).
   Generalization to sampling/base-models/other-languages untested. Citation pass incomplete (§7).

## 9. Reproducibility

All code self-tested (`python mosaic_phase{0..6}.py --selftest`); experiments greedy/deterministic. The
Phase-5 per-token decode passes a token-for-token equality gate vs `model.generate` under zero push;
Phase-6 controller math passes a synthetic-plant gate (decoupling cancels coupling; ill-conditioning ⇒
unreachable). Pipeline: `mosaic_phase0.py --run` → `phase1` → `phase2` (saves `mosaic_cones_<model>.npz`)
→ `phase3` → `phase4` → `phase5 --run` (per-token loop, EMA slew arm) → `phase6 --run` (MIMO tracking).
Direction-choice lever: `mosaic_phase2_coupling_check.py`. Second family: write `casa_frontier_<model>.json`
meta (band + pscale), then `--model Qwen/Qwen2.5-3B-Instruct` (κ / u_max scaled to pscale). Artifacts +
per-phase analysis in `PHASE{0..6}_RESULTS.md`; figures in `outputs/mosaic_phase{1..6}_<model>.png`. Reuses
the program's shared infra (`pytorch_pure/`, CASA cone/actuator/plant, CALM `decode_dual` loop).

## 10. Conclusion

The honest, falsifiable answer to "does control theory help activation steering?" is a dividing line.
**Open-loop control — however sophisticated — ties a well-chosen fixed push on the strength↔side-effect
frontier**, even for a graded multi-attribute target with a binding constraint, because the
(strongly/semantically) coupled pair we tested is not separable by an open-loop linear intervention and
the trade-off is set by the aggregate. **Closed-loop output feedback breaks off that frontier** — it holds
an off-target attribute tighter at matched driven-gain, decisively on Qwen2.5-3B (3.7–6.0 SE) and
directionally on Gemma-2-2b (~1.7 SE) — plausibly because the side-effect accumulates over generation and
only a loop with a faithful sensor can observe and correct it. Control becomes load-bearing for steering
when the target is graded, the side-effect accumulates over the token axis, and a faithful output sensor
exists — and then it must be closed-loop. **But that win is local, and we mapped its edges:** the loop's
update *rate* is a minor lever — a true per-token loop chatters unless slew-limited, and then beats the
cheap chunked loop by only ~1.5–1.7 SE (Phase 5); and full 2-DoF setpoint *tracking* is bounded by
**actuator collinearity** (cond G ≈ 4 Gemma / 7 Qwen — the measured, cross-family form of the entanglement),
where a model-based decoupling controller ties or, ill-conditioned, *loses to* the naive diagonal one
(Phase 6). The durable open-loop levers remain the actuator, the bound, and a good (supervised) direction;
the durable closed-loop lever is **disturbance-rejection feedback with a smooth control signal** — not
lookahead, not static allocation, not model inversion.
