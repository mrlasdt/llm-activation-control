# MOSAIC — Multi-Objective Steering via Allocation In Cones

**Proposal:** [`MOSAIC_PROPOSAL.md`](MOSAIC_PROPOSAL.md). **Status: Phases 0–4 run (2026-06-14).**

> **Results banner (honest).** Phase 0 **GO** (cross-coupling binds; formality↔reading cosine +0.50;
> reversion α≈0.40) → Phase 1 **GO** (probes r=0.91/0.94 > 0.85; strictly-monotone strength↔coherence
> frontier, not a hump) → Phase 2 **GO** (PLS attribute-specific cones recon R² 0.93/0.92 ≫ blunt PCA;
> depth-indexed reference beats a single diff-in-means vector; **durable lever: a PLS direction causes
> ~80% less side-effect than diff-in-means**) → **Phase 3 SCOPED NULL** (open-loop null-space allocation
> does NOT beat the scalar push — scalar/alloc share one frontier; the formality↔reading coupling is a
> *semantic entanglement*, not a linearly-separable subspace) → **Phase 4 WIN — the first
> "control-law > open-loop lever" result in the whole program.** A **closed-loop** controller that drives
> formality up while **regulating the measured reading-level back to baseline each chunk** holds reading
> **>1 SE tighter** than the open-loop push at matched formality-gain (|drift| 3.66→1.69 at κ=4; frontier
> dominates). **Why it works where Phase 3 tied:** the side-effect *accumulates over generation*, so
> *output feedback* sees and corrects the downstream re-entanglement an open-loop per-layer projection
> cannot. **Caveat: not a free lunch** — coherence tax (distinct-2 0.95→0.86 at κ=4) + narrow stable gain
> band (κ≥8 over-corrects/degenerates). **Bottom line:** open-loop steering (lookahead/coherence-cost/
> allocation) can't separate semantically-entangled attributes; **closed-loop output feedback can**, when
> the target is graded, the side-effect accumulates over tokens, and a faithful output sensor exists. See
> `PHASE{0,1,2,3,4}_RESULTS.md`. **Cross-model replication (Qwen2.5-3B): WIN, more decisively** — Phase 2
> prerequisites reproduce (PLS recon R² 0.92/0.94 ≫ PCA), and Phase 4 closed-loop holds reading ~2.5×
> tighter at matched formality-gain (paired **4–6 SE** vs Gemma's 1.7 SE) at the κ=1 sweet spot with
> **near-baseline coherence** (distinct-2 0.94 vs 0.97 — a smaller coherence tax than Gemma). Optimal κ
> scales with pscale (≈1 Qwen / ≈4 Gemma), same over-correction at high κ. **The first "control-law >
> open-loop lever" result in the program now holds on 2 model families.** Next: larger n + true per-token
> loop, coherence-aware gain-scheduling (OAS/PID), full setpoint-tracking MIMO.

## One line

Recast **multi-attribute** steering as **constrained control allocation**: drive a graded attribute A to
a commanded setpoint while **holding** B, C at their natural baselines, using the **null-space** of a
multi-concept cone basis. The first regime in this program where a **control law provably beats the
bound** — because the hold-constraint is **active at the optimum**, which a scalar push cannot satisfy.

## Why this one (and why it should escape the saturation null)

The prior arc (PTS → BASELINES → CALM) proved the controller is a wash for **de-refusal** because refusal
**saturates** (a binary flip ⇒ a hump, not a frontier). A-LQR and PID Steering reproduce this on the same
models. MOSAIC changes the problem on the two axes the synthesis named:
1. **graded, monotone** attributes (formality / reading-level / sentiment) — *not* a binary flip;
2. a constraint that **binds by construction** — superposition makes a scalar-A push drag B, C off
   baseline (the side-effect drift **Course-Correction** names as the dominant failure), so holding B, C
   is an *active* constraint that **needs the cone's redundant DoF**.

It honestly **declines** the third condition (token-axis lookahead): H=1 ≈ H>1 is pre-registered as a
*confirmation*, so it cannot re-derive the PTS null.

## Candidates considered (deep-research workflow, 25 agents; adversarial 3-judge panel)

| Candidate | Idea | Panel total /120 | Verdict |
|---|---|--:|---|
| **MOSAIC** | constrained null-space allocation on multi-concept cones (binding hold) | **89** | **chosen headline** |
| **SARTRE** | sustained token-axis regulation of a *reverting* graded attribute (autoregressive prior = disturbance) | **86** | **grafted in** as MOSAIC's token-axis arm (§6) + the Phase-0 fork / fallback headline |
| KEEL | hard-KL-budget receding-horizon control of reasoning-effort | 71 | KL-budget idea grafted; reasoning-effort deferred |
| SCOPE | track a depth×token reasoning-budget curve (integrating `</think>` latent) | 71 | high novelty, low feasibility — deferred |
| LACA | LoRA bank as an over-actuated plant; allocate among rank channels | 65 | = MOSAIC's gated D4 extension (§13) |

The two top picks tied; the synthesizer made MOSAIC the headline and **integrated SARTRE's
reverting-disturbance mechanism** as the second binding footing, so **Phase 0 is a fork**: measure
cross-coupling (→ MOSAIC) *and* attribute reversion (→ SARTRE) cheaply, and let the data pick the headline.
Both are graded/non-saturating, so both clear the hump.

## How to start (Phase 0, ~1 GPU-hr, no controller, no training)

Build quick diff-in-means cones for a **deliberately interfering** triple (formality↔reading-level couple
strongly; sentiment third) via `casa_cone.residual_means`+`dim_directions`; push A with a scalar additive
actuator (`casa_actuator.ConeActuator`); measure induced B,C drift (Flesch-Kincaid + sentiment scorers)
**and** post-release reversion over 256 tokens. **GO/NO-GO gate + the dividing-line law are in
`MOSAIC_PROPOSAL.md` §7.** Sit on the saved Gemma-2-2b plant via `calm_token.load_cone_and_band`.

## Maps to the requested directions

D1 calibrated depth-indexed setpoint map (not diff-in-means) · D2 constrained allocation QP/MPC + binding
KL budget · D3 layer **and** token-axis closed loop · D4 LoRA-as-actuator (gated extension, §13) ·
D5 aerospace control-allocation lens + the falsifiable coupling-predicts-payoff scoping law.

## Provenance

Full workflow output (per-paper extractions, codebase map, literature with citations, 5 candidates, judge
rankings, integrated design) lives in the run transcript:
`…/693ab676-…/tasks/w36a2q9jq.output` (run `wf_f3f2102c-75d`).
