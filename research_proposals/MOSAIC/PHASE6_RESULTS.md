# MOSAIC Phase 6 — full MIMO setpoint tracking (Gemma-2-2b-it)

**Question.** Phase 4 won at a HOLD task: drive formality up while regulating reading back to its
*baseline*. Phase 6 asks the fully general 2-DoF question: can a closed loop place **both** outputs at
**arbitrary commanded setpoints** (τ_F, τ_R) — including the anti-correlated **"formal-but-simple"**
corner the entanglement fights — and does a **model-based decoupling** controller (`a = G⁻¹·K·e`, G the
calibrated 2×2 plant gain) beat a naive **diagonal** one (`a = diag(G)⁻¹·K·e`)? Both get correct
per-axis scaling; the only difference is whether off-diagonal coupling is inverted.

## Headline: a quantified reachable-set BOUNDARY — decoupling does not escape it

**The plant is the result.** Calibrating the static gain (normalized Δoutput per unit actuator command):

```
G = [[0.0922, 0.0713]      row 0 = formality,  col 0 = formality-cone actuator a_F
     [0.0331, 0.0796]]     row 1 = reading,    col 1 = reading-cone actuator a_R
cond(G) = 3.97   det(G) = 0.0050   actuator-effect collinearity cos = +0.880
```

Both actuators **raise both outputs** (every entry > 0): pushing formality also raises reading
(+0.033) and pushing reading also raises formality (+0.071). The two actuators' normalized
output-effect vectors are **88 % collinear** — so the two outputs cannot be placed independently
without large, opposing commands that exceed the coherent authority budget. **cond(G)=3.97 / cos=0.88
is the quantitative form of the Phase-3 "semantic entanglement" hypothesis** — stated as a plant
property (input→output controllability), not a metaphor.

**Tracking (6 chunks, integral control, anti-windup ‖a‖≤u_max=27.1, Kp=0.3; all coherent, dist2 0.87–0.94):**

| setpoint (F, R) | | diagonal trackErr | decoupling trackErr | paired diag−decoup | verdict |
|---|---|--:|--:|--:|---|
| hold_reading (0.71, 10.3) | | 0.785 | 0.832 | −0.05 ± 0.14 | tie |
| **formal_simpler (0.71, 7.8)** | **ANTI** | 0.766 | 0.755 | +0.01 ± 0.16 | **tie** |
| formal_complex (0.71, 12.8) | | **0.756** | 0.945 | −0.19 ± 0.09 (**−2.0 SE**) | **diag-win** |
| casual_simpler (0.47, 7.8) | | 1.046 | 0.793 | +0.25 ± 0.23 (1.1 SE) | decoup-win |
| hold_F_simpler (0.59, 7.8) | | 0.708 | 0.773 | −0.07 ± 0.14 | tie |

Two clean facts:
1. **No setpoint is tracked to low error.** Best residual is ~0.71 normalized (≈ 0.1 formality + ≈ 2
   FK-grades off). Formality consistently **undershoots** its +0.12 target (reaches ~0.65, not 0.71):
   regulating reading and driving formality compete for the same near-collinear push budget, and within
   the coherent authority bound (u_max = 0.15·pscale, the Phase-4 coherence edge) neither controller can
   satisfy both. **This is an *authority* bound, not a rank deficiency** — G is full-rank (cond 3.97 is
   only moderate ill-conditioning; the rule-of-thumb concern threshold is ~30), so the anti-correlated
   setpoint is reachable *in principle*; raising u_max would extend reach, but Phase 1 showed u_max ≳
   0.15·pscale already costs coherence, so the *coherent* reachable set is what's bounded.
2. **Decoupling does NOT beat diagonal — and is sometimes significantly worse.** It ties on the headline
   anti-correlated corner (`formal_simpler`), is **significantly worse on `formal_complex` (−2.0 SE,
   diag-win)**, and wins on exactly one setpoint (`casual_simpler`, 1.1 SE). Net: model-based plant
   inversion buys nothing here and, when it over-corrects under saturation, hurts.

> Note: an over-driven first run (Kp=0.5, windup to ‖a‖=40.7) showed decoupling *directionally* ahead
> on the anti-corner (+1.3 SE) — but **incoherently** (dist2 0.70). With proper anti-windup that edge
> vanishes and coherence returns. The lesson is the program's own: never read a verdict off incoherent
> text. The honest, coherent answer is a tie.

## Interpretation — bounding the Phase-4 win

Phase 4's closed-loop advantage was specifically about holding reading **tighter than open-loop at
matched formality-gain** — a *relative* improvement on one axis. Phase 6 shows that advantage **does
not extend to absolute 2-DoF setpoint placement**: the reachable (formality, reading) set within the
coherent authority budget is bounded, and the bound is set by **actuator collinearity (cos 0.88 /
cond(G) 3.97)**, a plant property no controller can invert away without more authority (which costs
coherence). "Formal-but-simple" sits near the edge of that set — reachable in *direction* (you can move
toward it) but not to a tight setpoint.

This is the natural, honest closure of the MIMO arc: **closed-loop feedback separates entangled
attributes in the small (regulate a disturbance back to baseline, Phase 4), but within the *coherent
authority budget* the plant's near-collinear actuators — not the controller's sophistication — set the
limit on placing both attributes at arbitrary, mutually-opposing setpoints.** The limit is
authority×collinearity, not a rank deficiency (cond G ≈ 4 is moderate, G full-rank): more authority
(higher u_max) would extend the reachable set, but at a coherence cost (Phase 1), so it is the
*coherent* reachable set that is bounded. A "control sophistication ties (or, ill-conditioned, *loses
to*) the lever" result on the *tracking* axis, with the limiting quantity (cond G / actuator
collinearity) measured.

Artifacts: `outputs/mosaic_phase6_gemma-2-2b-it.{json,png}`, `outputs/phase6_full.log`.

## Cross-family replication (Qwen2.5-3B-Instruct) — the boundary is a plant property

```
G = [[0.2037, 0.2024]      cond(G) = 7.23   actuator-effect collinearity cos = +0.939
     [0.2690, 0.6630]]      (Gemma was cond 3.97 / cos 0.880)
```

The headline diagnostic **replicates and strengthens**: Qwen's two style actuators are **94 % collinear**
(vs Gemma's 88 %), so the same NULL/BOUNDARY holds — no setpoint reaches low error, the anti-correlated
corner is unreachable to a tight setpoint. **And decoupling is now actively *worse*** (formal_simpler:
diag 1.05 vs decoup 1.85 — the decoupling controller, trying to invert a more ill-conditioned plant,
commands a large opposing push that saturates at u_max and drives reading the *wrong* way, 6.9 target →
11.16). This is the textbook failure of model-inversion control under **actuator saturation + plant
ill-conditioning**: a high `cond(G)` makes `G⁻¹` demand authority the bounded actuator cannot supply, so
the "smart" controller overshoots while the naive diagonal one stays safe. (All Qwen runs coherent,
dist2 0.91–0.97.)

**Two families, same verdict:** 2-DoF style setpoint-tracking is bounded by **actuator collinearity**
(cos 0.88 / 0.94), a measurable plant property; a model-based decoupling controller does not beat — and
under tighter ill-conditioning actively underperforms — the naive diagonal controller. cond(G) is the
cross-family quantitative form of the Phase-3 entanglement.

Artifacts: `outputs/mosaic_phase6_Qwen2.5-3B-Instruct.{json,png}`, `outputs/phase6_qwen.log`.
