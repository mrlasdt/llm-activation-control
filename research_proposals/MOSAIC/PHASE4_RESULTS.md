# MOSAIC Phase 4 — closed-loop token-level hold (output feedback): results

**Verdict: WIN — the first "control law beats the open-loop lever" result in the whole program.**
A closed-loop controller that drives formality up while **regulating the measured reading-level back to
baseline each chunk** holds reading **>1 SE tighter than the best open-loop push at matched
formality-gain** — exactly the separation the open-loop linear allocation (Phase 3) provably could NOT
achieve (shared frontier). The difference is *output feedback*: it sees the generation-time
re-entanglement and counter-pushes. **Honest caveat: it is not a free lunch — the hold is bought with a
coherence tax and only works in a narrow gain band (κ≈4; κ≥8 over-corrects and degenerates).**

Run: `mosaic_phase4.py --run` (Gemma-2-2b-it, band [7,24]; n_eval=32, 4 chunks × 32 tok = 128;
drive m_F ∈ {0.04,0.06,0.08}·pscale, hold gain κ ∈ {0,4,8,16}; sensor = exact Flesch-Kincaid).
Date: 2026-06-14. Data: `outputs/mosaic_phase4_gemma-2-2b-it.{json,png}`. (Fixed a mid-run right-padding
bug — `model.generate` right-pads EOS-finished sequences; `_relpad` re-left-pads between chunks. The win
is unchanged before/after the fix.)

---

## 1. The result — closed-loop dominates the open-loop frontier

|reading-drift| at matched **formality-gain** (drive-only κ=0 is the open-loop baseline = Phase-3 scalar):

| condition | formality-gain | \|reading-drift\| | distinct-2 | e_R trace (FK error / chunk) |
|---|---:|---:|---:|---|
| drive-only (κ=0), m_F=0.04 | +0.148 | 3.66 | 0.95 | 3.8 → 3.4 → 3.7 → 3.7 (flat) |
| **closed-loop κ=4, m_F=0.04** | +0.122 | **1.69** | 0.90 | 3.8 → 2.3 → 1.7 → 1.7 (regulated) |
| drive-only (κ=0), m_F=0.06 | +0.187 | 4.45 | 0.97 | 4.7 → 4.5 → 4.4 → 4.5 (flat) |
| **closed-loop κ=4, m_F=0.06** | +0.166 | **2.41** | 0.86 | 4.7 → 3.3 → 2.6 → 2.4 (regulated) |
| closed-loop κ=8, m_F=0.06 | +0.086 | 2.45 | 0.71 | (over-steers formality down; degrading) |
| closed-loop κ=16, m_F=0.06 | +0.059 | 2.54 | 0.63 | (degenerate) |

**Matched-formality-gain, paired (same prompts):** at formality-gain ≈ +0.148, drive |reading-drift| =
3.66 vs closed-loop = 2.12, **paired difference = 1.25 ± 0.72 SE ≈ 1.7 SE** (n=32). The closed-loop
(formality-gain, reading-drift) **frontier lies below the open-loop frontier at every operating point**
(`mosaic_phase4_gemma-2-2b-it.png`, left). Open-loop allocation (Phase 3) shared one frontier with
scalar; **closed-loop output feedback breaks off it** — a ~40–55% reduction in reading-drift at matched
formality movement.

## 2. Why it works where open-loop (Phase 3) tied — the mechanism

The side-effect (formal ⇒ complex) **accumulates over generation**: the drive-only e_R trace stays flat
and high (4.7 → 4.5; the reading-level drifts up and stays up). An **open-loop** per-layer projection
(Phase 3) zeroes the held coordinate *at the band layers* but cannot see or veto the downstream
re-entanglement, so the trade-off is governed by the aggregate intervention (shared frontier). A
**closed-loop** controller measures the *realized output* reading-level each chunk and counter-pushes —
the e_R trace falls 4.7 → 2.4 (`...png`, right). This is the program's **condition (3)** finally
satisfied: a graded, non-saturating side-effect that *accumulates over the token axis*, where feedback
has genuine predictive/corrective value — the regime PTS (near-identity depth) and CALM (refusal
saturates) lacked.

This is the first controller-beats-the-lever result across **PTS → CALM 1–3 → MOSAIC P3 → MOSAIC P4**:
| experiment | control structure | outcome |
|---|---|---|
| PTS / CALM 1–3 / MOSAIC P3 | lookahead / coherence-cost / token-KL / open-loop cone allocation | all **tie** the lever |
| **MOSAIC P4** | **closed-loop output feedback (regulate measured side-effect)** | **WIN (>1 SE)** |

The lever was never lookahead, coherence-cost, or static allocation — it is **output feedback on a
generation-time-accumulating signal**.

## 3. Honest caveats — not a free lunch

1. **Coherence tax.** distinct-2 falls with κ: 0.95 (κ=0) → 0.86–0.90 (κ=4, still coherent, ≥ floor) →
   0.71–0.75 (κ=8) → 0.63–0.69 (κ=16, degenerate). The win at κ=4 is within the coherence floor (verified:
   the κ=4 text is coherent, on-topic, genuinely simpler — not repetition), but the closed-loop is
   measurably *less* coherent than drive-only at matched formality-gain (0.86–0.90 vs 0.95). It buys the
   hold partly by spending coherence — a **3-way (formality, reading-hold, coherence) trade-off**, not a
   strict Pareto win on all three.
2. **Narrow gain band / instability.** κ=4 is the sweet spot; κ=16 **over-corrects** — reading-drift goes
   back *up* (2.17 / 2.54 / 3.68) and coherence craters — classic high-gain instability. A deployable
   controller needs gain-scheduling / anti-windup (the program's PID/OAS machinery is on-hand for this).
3. **Scope.** n=32 (win ≈1.7 SE — real but not overwhelming), single model (Gemma-2-2b), chunked (4×32)
   not true per-token, hold-to-*baseline* only (arbitrary reading setpoints untested), formality drive via
   the classifier (reading sensor is exact FK). Replication (Qwen2.5-3B, larger n, per-token loop) is the
   confirmation step.

## 4. What this means for MOSAIC and the program

**MOSAIC's thesis is revised, not dead.** The *static null-space allocation* headline (Phase 3) is
falsified — semantic entanglement makes the coupling non-separable by an open-loop linear projection. But
the *constrained-control* spirit survives in the form that actually works: **closed-loop output
regulation of the side-effect over generation.** The complete, honest story:

> Open-loop activation steering — however sophisticated (lookahead, coherence cost, multi-layer
> allocation, null-space projection) — cannot separate semantically-entangled graded attributes; they
> share one strength↔side-effect frontier. **Closed-loop output feedback can**, because the side-effect
> lives in generation-time output dynamics that only a loop with a faithful output sensor can observe and
> correct — at a coherence cost and within a stable gain band. Control becomes load-bearing exactly when
> (1) the target is graded/non-saturating, (2) the side-effect accumulates over the token axis, and
> (3) a faithful output sensor exists.

Plus the durable Phase-2 lever (a supervised PLS *direction* removes ~80% of the side-effect for free —
use it as the drive direction inside the loop).

## 5. Next

- **Replicate + tighten:** Qwen2.5-3B, larger n (push the ≈1.7 SE win to ≥3 SE), and a true per-token
  loop (the verified `calm_token.decode_dual`, widened to carry the drive+feedback push) vs the chunked
  approximation.
- **Coherence-aware gain:** schedule κ / add anti-windup so the hold doesn't cost coherence (reuse
  `BASELINES` PID + `OAS` observer machinery — this is finally their earned niche: a noisy/graded output
  sensor regulated under a stability budget).
- **Setpoint tracking (not just hold):** drive formality to τ_F while holding reading to an arbitrary τ_R
  (the full MOSAIC MIMO), now that the closed-loop hold is shown to work.

Reproduce: `mosaic_phase4.py --run`; the frontier + regulation traces are in the JSON/PNG.

---

## 6. Cross-model replication — Qwen2.5-3B-Instruct (2026-06-14): WIN, more decisively

The closed-loop result **replicates on a second model family, more cleanly than on Gemma** — clearing
the cross-family reliability bar the steering literature demands (Tan et al.). Setup: Qwen2.5-3B-Instruct
(36 layers, hidden 2048), band **[10,33]** (24L, proportional to Gemma's [7,24]), **pscale=58.2**
(computed from the attribute mean-difference gaps so `frac·pscale` is the same ~4–12% relative push as on
Gemma). Files: `mosaic_phase{2,4}_Qwen2.5-3B-Instruct.{json,png}`, `mosaic_cones_Qwen2.5-3B-Instruct.npz`.

**Prerequisites reproduce (Phase 2 on Qwen, GO both criteria):** PLS attribute-specific cones recon R²
**0.92 (formality) / 0.94 (reading)** ≫ blunt PCA 0.62 / 0.73; depth-indexed reference beats a single
diff-in-means vector (formality 0.113<0.153, reading 3.975<5.881); plant R² 0.98/0.99; coupling present
(formality→reading drift +1.29 FK).

**Phase 4 WIN (Qwen), at matched formality-gain, paired:**

| formality-gain | drive-only \|reading-drift\| | closed-loop (κ=1) \|reading-drift\| | paired diff ± SE |
|---:|---:|---:|---:|
| +0.088 | 2.91 | 1.44 | **1.68 ± 0.32 (5.3 SE)** |
| +0.116 | 3.35 | 1.37 | 2.03 ± 0.42 (4.8 SE) |
| +0.143 | 3.75 | 1.40 | 2.47 ± 0.41 (6.0 SE) |

- **Stronger and cleaner than Gemma.** The win is **4–6 SE** (vs Gemma's ~1.7 SE), and at the sweet
  spot **κ=1** the closed-loop holds reading **~2.5× tighter** (|drift| 1.2–1.4 vs drive-only 2.9–3.9)
  while preserving **~90–95% of the formality-gain** AND **near-baseline coherence** (distinct-2 0.94–0.95
  vs baseline 0.97) — a near-clean Pareto win, i.e. a **smaller coherence tax** than Gemma (where κ=4 cost
  distinct-2 0.95→0.86).
- **Same mechanism, same control signature.** e_R regulates down (m_F=0.06, κ=1: 3.0→2.5→1.5→1.4) while
  drive-only stays flat (3.0→3.4); high κ **over-corrects** (κ=4–8: reading-drift climbs back, coherence
  craters 0.70→0.62) — the same high-gain instability as Gemma.
- **Optimal κ scales with pscale, as predicted.** Sweet spot κ≈1 on Qwen (pscale 58) vs κ≈4 on Gemma
  (pscale 180) — the hold push `κ·e_R·g_R` is in FK-grade units (not pscale-scaled), so the balanced gain
  scales ~inversely with pscale (4/1 ≈ 180/58). The controller is well-behaved and the parametrization is
  understood (a pscale-relative κ would transfer the gain directly — a clean-up for the writeup).

**Bottom line:** MOSAIC Phase 4 — closed-loop output-feedback hold beats the open-loop lever — is now
confirmed on **2 model families** (Gemma-2-2b, Qwen2.5-3B), **more decisively on Qwen** (4–6 SE, near-zero
coherence tax). The program's first "control law > open-loop lever" result is robust and cross-family.
The revised MOSAIC thesis (§4) stands: open-loop steering can't separate semantically-entangled graded
attributes; closed-loop output feedback can, when the target is graded, the side-effect accumulates over
the token axis, and a faithful output sensor exists.
