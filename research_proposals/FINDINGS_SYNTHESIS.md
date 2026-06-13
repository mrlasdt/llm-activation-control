# Findings synthesis — control-theoretic activation steering

A cross-proposal "zoom out": every proposal's verdict with the numbers that decided it, the axes
that actually moved behaviour, and the diagnosis that motivates the next direction (**CALM**). All
numbers are quoted from each proposal's authoritative results file (cited inline). Date: 2026-06-13.

> ⚠️ De-refusal (jailbreak) is the *measured experimental outcome* on an open-weights model
> (authorized safety/steering research), not a product. StrongREJECT + genNLL exist so that
> gibberish is never miscounted as a jailbreak.

---

## 1. The six proposals, by outcome

| Proposal | Core idea | Actuator | Verdict | Decisive numbers |
|---|---|---|---|---|
| **SO2** | Lyapunov/energy state-feedback over depth | rotation | **analysis only** — no behavioural head-to-head | Qwen2.5-3B phase portrait: max angle-sep 3.11 rad @ L28; separatrix crossings harmful **44.8%** vs harmless **5.8%** |
| **CLAS** | per-token output-feedback "thermostat" | rotation | **DROPPED** (2026-06-09) — no niche | refusal `G(θ)` range **~18** but *one-shot/saturates*; sentiment authority **~1.6** (too weak); 200-tok steered vs unsteered gens **byte-identical** |
| **PTS** | predictive MPC tracking a 2D trajectory | rotation | **VERIFIED NEGATIVE** (2026-06-12) | PTS ≈ fixed-angle: Qwen **0.05 vs 0.00**, Gemma **+3.66 vs +3.67**; H=1≈H=8; ‖A−I‖≈**0.34**. Durable win: the 2×2 plant (R²≈**0.999**) |
| **OAS** | LQG (Kalman observer + soft-landing LQR) | rotation | **MIXED** | Exp1 authority gate **passes** (refusal 17.9→20.5 **+15%**, sentiment 1.66→3.19 **+92%**); observer wins under noise (roughness flat **0.19** vs LQR-on-raw 0.22→**1.19**); soft-land **under-commits** (Gemma +4.7 vs deadbeat **−5.50**); **3× more plant-robust** (effect-std 0.16 vs 0.50) |
| **CASA** | bounded **additive** ablation of a refusal **concept cone** + MPC | **additive** | **WIN** (2026-06-12) | StrongREJECT 0.017 → DIM-k1 0.16 → SVD-k1 0.69 → RDO-k1 0.71 → cone-k4 0.68 → **cone+MPC 0.76** at **−0.41** coherence tax; blunt SVD k≥2 ≈ **0** (gibberish) |
| **BASELINES** | P vs PID vs LQR vs MPC head-to-head | additive | **TIE** (2026-06-13) | best budget **P 0.721 ≈ LQR 0.718 ≈ MPC 0.720** (0.003 spread ≪ 1 SE ≈0.05); the **bound** lifts every law **0.64→0.72** |

Sources: `SO2/research_proposal_SO2.md` + `SO2/phase_portrait_output/`, `CLAS/README.md`,
`PTS/PTS_README.md`, `OAS/README.md`, `CASA/CASA_RESULTS.md`, `BASELINES/BASELINES_RESULTS.md`.

---

## 2. The four axes that actually decided behaviour (deterministic comparison)

**Axis 1 — Actuator: rotation vs additive (THE decisive variable).**
Every rotation proposal (SO2/CLAS/PTS/OAS) capped out or died; the one additive proposal (CASA) won.
Rotation is **norm-preserving** → it can only set the in-plane *angle*, collapsing any multi-DoF plan
to **one realized DoF**. On Gemma the best rotation variant reaches margin **+3.67 (still hedging)**;
the additive (norm-changing) ablation reaches **−2.99 with coherent compliance** at +0.10 NLL tax.
*De-refusal requires shrinking the refusal-component magnitude — which rotation structurally cannot do.*

**Axis 2 — Trajectory tracking vs single target (three separable parts, not one verdict).**
- *Reference* (which target direction to track) — **decisive**: CASA bounded-ablation → MPC tracking
  the harmless-mean cone coordinate lifts StrongREJECT **0.712 → 0.764**.
- *Multi-step lookahead* (the predictive horizon) — **inert**: H=1 ≈ H=8 everywhere (PTS, CASA), because
  the late band is near-identity (‖A−I‖ ≈ 0.34 on Qwen 2×2; 0.42–0.86 on Gemma cone) so there is nothing
  to anticipate.
- *Distributing* the control over a trajectory — **helps coherence/robustness, costs authority**: OAS
  soft-landing is 3× more plant-robust but under-commits (Gemma Δ≈−6 vs a concentrated deadbeat Δ≈−16).

So "trajectory tracking" earned its keep only through its **reference**, never its **horizon**.

**Axis 3 — Bound vs control law.**
The **bound is the lever**, the **law is a wash**. Tightening `u_max` (∞ → tight) lifts *every* law
**0.64 → 0.72** StrongREJECT — larger and more consistent than any law-to-law gap (P/PID/LQR/MPC spread
**0.003**). More sophistication can *hurt*: PID **0.703** (integral just adds effort, 10.2 vs 9.2), and
LQR+feedforward is **worst, 0.687** (effort **16 vs 9**, genNLL 1.02) — exactly tracking the harmless
mean **over-steers**, because de-refusal is *suppression*, not reaching a setpoint.

**Axis 4 — Dimensionality & metric traps.**
- k>1 alone does **not** beat k=1 (cone-k4 0.68 ≈ RDO-k1 0.71 ≈ SVD-k1 0.69). But a **retain-loss
  concept cone** is a *coherent* k>1 jailbreak (0.68, neutral ΔNLL −0.71) exactly where a **blunt SVD
  span** at the same k is incoherent gibberish (≈0, neutral ΔNLL +1.5→+2.3). The retain loss is the
  mechanism (refusal-specificity), not the dimensionality.
- **First-token margin is an unreliable proxy** throughout: it over-claims k>1; LQR+ff shows the *most*
  negative margin (−6.92) yet the *lowest* srScore (0.687). Judge full generations (StrongREJECT) + genNLL.

---

## 3. What worked / what didn't (carry these forward)

**Worked (keep):**
1. **Additive (norm-changing) actuator** — the whole ballgame for de-refusal.
2. **The bound** — "don't over-steer" buys +0.08 srScore for every law and keeps the answer fluent.
3. **The reference** — steering toward the harmless-mean direction (+0.05 over a matched-budget ablation).
4. **Retain-loss concept cone** — makes a k>1 subspace refusal-specific and coherent.
5. **The cheap k×k affine plant** (R²≈0.999, ~256 B/layer) — reusable infra (PTS→OAS→CASA).
6. **Observer under measurement noise** (OAS Exp4) — earns its place only when the readout is noisy.

**Didn't (stop doing):**
1. **Rotation actuator for de-refusal** — norm-preserving ⇒ 1 DoF ⇒ caps out.
2. **Multi-step lookahead / predictive horizon** — near-identity layer dynamics ⇒ no value.
3. **PID integral & LQR feedforward** for suppression — over-steer, add cost, hurt coherence.
4. **Soft-landing / gentle distribution** — under-commits authority (gentleness costs authority).
5. **Blunt SVD k>1 subspace** — sweeps in capability directions ⇒ gibberish.
6. **First-token margin as the headline metric** — disagrees with judged behaviour at length.

---

## 4. The diagnosis — why MPC keeps tying (and what would change it)

The recurring null ("law doesn't matter, bound does") is **theory-consistent, not an artifact**:
- **Unconstrained ⇒ MPC = LQR** (proved in `casa_baselines._selftest_mpc_equals_lqr` to 1e-5).
- **Near-identity band ⇒ weak lookahead** (Gemma cone ‖A−I‖₂ band-mean 0.86, full-d RMS 0.42–0.52,
  5-step R²=0.9955).
- **The constraint barely binds**, and the objective is **suppression, not tracking**.

But the deeper reason — the one that reframes the user's MPC hypothesis — is **saturation**:

> For layer-domain refusal, **strength↔coherence is not a frontier; it is a saturating hump.**
> Past the de-refusal threshold, more control loses on *both* axes: the ∞-budget rows have *lower*
> StrongREJECT (0.64) **and** worse genNLL (~1.0). Behaviour is one scalar that flips and saturates,
> so there is no monotone Pareto frontier for a smarter controller to dominate, and the coherence
> constraint never binds against a **continuing** objective.

That is why no control law wins here. To make MPC earn its keep, the *control problem itself* must
change so its powers (binding constraints, genuine lookahead, multi-objective costs) become
load-bearing: a **non-saturating, monotone** strength↔coherence frontier; a coherence term that
**binds and reshapes** the optimum; and/or **evolving dynamics** (the token axis) where lookahead has
predictive value.

---

## 5. Literature gap → the next direction (CALM)

The strength↔coherence trade-off is *the* named open problem in steering: stronger coefficients push
activations **off-manifold** → degradation
([In-Distribution Steering, arXiv:2510.13285](https://arxiv.org/abs/2510.13285);
[Activation Steering survey](https://www.emergentmind.com/topics/activation-steering-in-llms)).
Existing fixes are **heuristic scalar intensity dials** — IDS dynamically tunes steering *intensity*;
[Dynamic Activation Composition (arXiv:2406.17563)](https://arxiv.org/abs/2406.17563) adjusts intensity
by **KL between steered/unsteered next-token distributions**. **None formulate the trade-off as an
optimal-control objective.**

**The gap, and the next proposal — CALM (Coherence-Aware Lookahead MPC):** put coherence
(off-manifold distance / KL) *inside* the controller as an explicit cost/constraint, so MPC's
*constrained* optimization could **dominate the frontier** instead of a heuristic dial — tested both
in the layer domain (an on-manifold density cost) and the regime where lookahead should finally bite
(**token-domain sustained** control on the ambient KL signal, reviving the dropped CLAS thesis with
CASA's additive actuator).

**CALM result (2026-06-13) — a third null that closes the arc.** Neither move beat a fixed bound:
the layer-domain coherence-aware MPC ties the frontier (the cone-space density it optimizes doesn't
predict genNLL, r=0.20), and the token-domain KL-thermostat ties static too (best Δsr +0.018, no
dominance) despite genuinely modulating — though it *did* confirm **ambient KL is a faithful coherence
proxy (r=0.976)**, validating *where* coherence lives. So across **PTS (lookahead) → CALM Phase 1+2
(coherence cost) → CALM Phase 3 (token-domain KL feedback)**, the verdict is consistent: **for
cone-based de-refusal, no control sophistication — lookahead, a coherence-aware cost, or dynamic
per-token feedback — beats a well-chosen fixed magnitude bound.** Behaviour saturates (a hump, not a
frontier) and the trade-off is governed by *aggregate* ablation, not its allocation in depth or time,
so there is no binding-constraint/lookahead structure to exploit. The durable levers remain the
**actuator** (additive vs rotation) and the **bound** — never the controller. See `CALM/CALM_RESULTS.md`.
