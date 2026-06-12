# CASA — Constrained Additive Subspace Ablation

*Control-theoretic steering with a **norm-changing** actuator. The salvage of PTS:
PTS's machinery (k-dim subspace, per-layer perturbation bound, plant + MPC over a
layer band) becomes load-bearing once the norm-preserving rotation is replaced by
additive bounded ablation.*

Status: **prototype-validated headline, research program open** (2026-06-12).
Prototype: `../PTS/verify/additive_subspace_steer.py`. Provenance: emerged from the
PTS verified-negative — see `../PTS/PTS_README.md` Verdict, last bullet ("where PTS's
machinery could still matter: a non-norm-preserving actuator").

---

## 1. Motivation — the actuator was the bottleneck

Angular Steering (and every control recasting of it so far — SO2, PTS, OAS) shares
one actuator: a **norm-preserving rotation** of the in-plane component to an absolute
angle. We now have direct evidence that *this actuator*, not the planner, is what
fails on Gemma:

- Canonical AS (`input_layernorm`, 1 layer) is **inert** on Gemma-2-2b from every
  layer (best de-refusal margin −2.09, still refusing).
- The aggressive residual-stream rotation band only **partially** de-refuses
  (margin +10.83 → +3.67; ~3–4/6 prompts still hedge/refuse).
- PTS's adaptive per-layer angle adds **nothing** over a fixed angle on the same band
  (+3.66 vs +3.67) — because a norm-preserving actuator collapses the 2D plan to one
  realized DoF (the angle).

Replacing rotation with **additive directional ablation** along the refusal axis,
applied across the discriminative band, flips it cleanly:

| condition (Gemma-2-2b, band 7–24) | refusal margin | neutral ΔNLL | behavior |
|---|---:|---:|---|
| baseline | +10.83 | — | refuses 6/6 |
| AS-rotation(band) — *the limit* | +3.67 | +0.02 | hedges, refuses ~4/6 |
| **additive ablate, k=1, band** | **−2.99** | **+0.10** | **complies 6/6, coherent** |
| additive ablate, k=8, band | +0.32 | +2.33 | **broken** — incoherent gibberish |

k=1 additive ablation de-refuses **every** prompt — including the racism/threat/bomb
prompts the rotation band still refused — at a negligible coherence cost (+0.10 NLL
on a 15-sentence neutral corpus). **The actuator change is the whole fix.** This
directly solves the AS limit on Gemma.

(Mechanistically this is multi-layer directional ablation, à la Arditi et al.; the
contribution here is *not* "we invented ablation" — see §5 — but that the actuator
swap is what unblocks Gemma, and the open frontier it creates.)

## 2. The four levers (status)

| | lever | status | evidence |
|---|---|---|---|
| **L1** | additive (norm-CHANGING) actuator instead of rotation | **VALIDATED** | k=1 band de-refuses Gemma cleanly (table above) |
| **L2** | k-dim refusal *subspace* instead of the 2D plane | **NEGATIVE as built / open** | k=8 top-SVD subspace destroys coherence (+2.33 NLL, gibberish) without de-refusing |
| **L5** | bounded per-layer ‖u‖ = explicit coherence knob | **untested at the binding regime** | the ½-strength cap did not bind at k=1 (half ≡ full) |
| **(PTS)** | k×k plant + MPC to **distribute** the additive push over the band | **untested for additive** | the machinery PTS built, now with an actuator it can actually shape |

**Why L2 backfired (the honest read):** the subspace was the top-k right singular
vectors of the per-layer mean-difference directions. That span is **not** a pure
refusal subspace — it sweeps in capability-bearing directions, so projecting all 8
out at *every* band layer guts the model. So "diffuse refusal ⇒ need k>1" is **not**
supported by this blunt test; on Gemma-2-2b, k=1 multi-layer ablation is sufficient
and best. A genuine k>1 win requires a refusal-*specific* subspace.

## 3. Method

Actuator at each band layer `j`, on the residual-stream output `h_j ∈ ℝ^d`, given a
refusal subspace `U ∈ ℝ^{k×d}` (orthonormal rows):

```
p = U h_j                          # in-subspace coordinate (k-dim)
u = clip(−p, ‖u‖ ≤ u_max)          # remove it, bounded (L5 knob)
h_j' = h_j + Uᵀ u                  # additive, norm-CHANGING
```

- `u_max = ∞` → exact projection-out `h' = h − UᵀU h` (full ablation).
- finite `u_max` → partial removal; `u_max` is the coherence dial (L5).
- `k = 1` → constrained directional ablation along the refusal axis.
- **Distributed (MPC) variant:** rather than removing the full coordinate at every
  layer, use the k×k plant `c_{k+1} ≈ A_k c_k + b_k` (PTS's validated 2×2 plant,
  generalized to k dims) and a bounded-`u` MPC to spread the removal across the band
  — minimal total perturbation that drives the late-layer refusal coordinate to ~0.
  This is the first place PTS's lookahead/constraint apparatus has a non-norm-
  preserving actuator to act on.

## 4. Experiments (kill-fast ordering)

The PTS lesson is baked in: **define the behavioral metric + the trivial baseline
first, and frame each experiment to kill CASA fast.** The trivial baseline is *blunt
k=1 full-band ablation* (the §1 win). Every added mechanism must beat it on the
behavior/coherence Pareto or it is dead weight (exactly what sank PTS).

1. **k × u_max frontier.** `k ∈ {1,2,3,4,6,8} × u_max ∈ {full, ¾, ½, ¼}·pscale`.
   Plot the (de-refusal margin) vs (neutral-corpus coherence tax) Pareto. *Kill
   condition:* if blunt k=1-full already sits on the frontier at near-zero tax (these
   results suggest it does), L5 and k>1 add nothing — report and stop.
2. **Refusal-specific subspace (rescue L2).** Replace top-SVD-of-mean-diffs with
   (a) independent diff-in-means from *multiple* contrastive datasets, or
   (b) directions orthogonalized against a capability/coherence basis (drop any
   component that moves the neutral corpus). Re-test whether a *clean* k>1 subspace
   de-refuses more *completely/robustly* than k=1 without the coherence collapse.
   *Kill condition:* if the clean k>1 subspace still doesn't beat k=1 on completeness
   at equal tax, L2 is dead for this model class.
3. **Distributed (MPC) additive push.** k×k plant + bounded-`u` MPC vs blunt every-
   layer removal, at matched final margin. *Win condition:* lower total coherence tax
   (or robustness to plant perturbation, à la OAS Exp 4) at equal de-refusal. *Kill
   condition:* same margin & tax as blunt removal → PTS failure mode again, drop it.
4. **Generalization.** Reproduce the k=1-band win on Gemma-2-9b and a Llama; confirm
   it is not 2B-specific. Replace the 15-sentence NLL proxy with a real coherence/
   capability suite (e.g., a held-out neutral set + a small benchmark), not just
   first-token margin.

## 5. Honest threats & novelty boundary

- **The basic de-refusal is known.** Multi-layer directional ablation is established
  (Arditi et al., 2024). CASA's claim to novelty is narrowly the **constrained,
  dynamics-aware k>1 coherence frontier**: can control machinery (the ‖u‖ bound + the
  plant + MPC) buy a *more complete or more robust* intervention than blunt ablation
  *at equal coherence cost*? If not, CASA reduces to "ablation works on Gemma," which
  is a useful confirmation but not a new method.
- **The PTS trap, again.** If a per-layer scalar cap on k=1 already saturates de-
  refusal at ~zero tax, the plant/MPC are inert — the same proxy-vs-behavior collapse
  that made PTS a negative result. Experiment 1 is designed to detect this in the
  first run.
- **Single-direction-single-layer is weak on Gemma (~3%, lit.); our authority comes
  from the band.** Be precise in write-ups: the lever is *multi-layer* k=1 ablation,
  not a magic single direction.
- **Measure behavior + real capability first.** Not control-internal proxies.

## 6. Relationship to the other proposals

- **PTS** (verified negative): same plant + MPC + constraint, *norm-preserving*
  actuator → no behavioral gain. CASA is PTS with L1 swapped in; PTS's verdict
  explicitly named this as the open lever.
- **OAS** (active): observer + soft-landing LQR over a band, still the rotation
  actuator. CASA's L1 could be dropped into OAS's controller (ablation-magnitude as
  the LQR control), making the observer/soft-landing story apply to a *moving*
  actuator on Gemma — a possible merge if CASA's L5/MPC earn their place.
- **SO2** (active): energy/Lyapunov state-feedback, also rotation-based; orthogonal.
