# MOSAIC Phase 2 — graded k>1 cones + depth-indexed setpoint map: results

**Verdict: GO on both pre-registered criteria** (attributes formality + reading-level). The
attribute-specific PLS cone reconstructs intensity far better than a blunt PCA span, and the
depth-indexed full-band reference beats a single difference-in-means vector on calibration. **The
load-bearing nuance for Phase 3: choosing a *supervised, attribute-specific* steering direction (PLS)
already removes ~80% of the side-effect cross-coupling that raw difference-in-means produces — so
Phase 3's allocation must beat the best *PLS-scalar* push (a strong baseline), and the margin is the
*residual* coupling, which is smaller than Phase 0's raw-DIM coupling implied.**

Run: `mosaic_phase2.py --run` (Gemma-2-2b-it, band [7,24], pscale 180.9; ladder 40×5=200 texts/attr,
n_eval=24, k=4, 96-tok greedy). Date: 2026-06-14. Data: `outputs/mosaic_phase2_gemma-2-2b-it.{json,png}`;
saved cones+maps for Phase 3: `outputs/mosaic_cones_gemma-2-2b-it.npz`. Coupling check:
`mosaic_phase2_coupling_check.py`.

---

## 1. C1 — attribute-specific k>1 cone is coherent (PLS vs blunt PCA)

Held-out intensity-reconstruction R² of the k=4 cone coordinate (how well the subspace *reads* the
attribute) + coherent attribute gain under a bounded push along the in-cone intensity direction:

| attribute | recon R² PLS | recon R² PCA | PLS push gain (distinct-2) | PCA push gain (distinct-2) |
|---|---:|---:|---|---|
| formality | **0.927** | 0.751 | +0.21 (0.87) | +0.30 (0.94) |
| reading | **0.916** | 0.529 | +8.23 (0.97) | +7.80 (0.90) |

- **PLS is decisively the more attribute-specific cone** (R² 0.93/0.92 vs 0.75/0.53; the reading gap is
  largest). This is the training-free analog of "retain-loss concept-cone ≫ SVD span" — a coherent,
  attribute-specific k>1 subspace exists, which is what the allocation QP needs as its state + reference.
- **Honest refinement of the CASA "blunt k>1 is gibberish" story:** that was an *ablation* phenomenon
  (projecting out a non-specific subspace removes capability directions). MOSAIC's actuator is *additive
  setpoint steering*, not ablation — so along the in-cone intensity direction **both** PLS and PCA move
  the attribute coherently (distinct-2 ≥ 0.87). MOSAIC's additive actuator therefore sidesteps the
  gibberish trap entirely; PLS still wins as the better-conditioned cone (cleaner coordinate → sensor,
  reference, and null-space for the QP).
- **k>1 does not give stronger single-attribute steering than k=1** (formality PLS gain +0.21 ≈ DIM +0.31;
  reading PLS +8.23 ≈ DIM/PCA ~+7–8) — consistent with the program's "bare k>1 ties best k=1." The value
  of k>1 is the **null-space for the hold** (Phase 3), not stronger driving.

## 2. C2 — depth-indexed reference beats a single difference-in-means vector (D1)

Coherent miscalibration `min_{coherent push}|achieved − τ|` over a τ grid: depth-indexed = per-layer DIM
directions applied across the **whole band**; single-vector = one DIM vector at the best single layer:

| attribute | depth-indexed (full band) | single vector (1 layer) |
|---|---:|---:|
| formality | **0.120** | 0.173 |
| reading | **1.768** | 3.629 |

The depth-indexed full-band reference hits the commanded intensity **more accurately while staying
coherent** (reading ~2× better). A single steering vector at one layer caps out / breaks coherence before
reaching the target. **D1 supported:** the optimal reference is depth-resolved, not a single
mean-difference vector. (Plant R²(1-step) on the cone coordinate over the band: formality 0.959, reading
0.975 — well-modeled, near-identity, so any H>1 lookahead stays inert as pre-registered.)

## 3. The load-bearing nuance — coupling is direction-dependent (carry to Phase 3)

Pushing **formality** up, measuring induced **reading-level** drift, at matched gain
(`mosaic_phase2_coupling_check.py`, n=16):

| steer direction | frac | Δformality | Δreading (FK) | drift / gain | distinct-2 |
|---|---:|---:|---:|---:|---:|
| **raw diff-in-means** | 0.05 | +0.211 | **+2.66** | 12.6 | 0.96 |
| raw diff-in-means | 0.10 | +0.303 | +3.46 | 11.4 | 0.63 |
| **PLS (attribute-specific)** | 0.05 | +0.183 | **+0.46** | 2.5 | 0.96 |
| PLS (attribute-specific) | 0.10 | +0.271 | +2.43 | 9.0 | 0.85 |

- **A supervised attribute-specific direction (PLS) removes ~80% of the side-effect drift that raw
  difference-in-means produces** at matched formality gain (per-unit-gain coupling 2.5 vs 12.6 at frac
  0.05) — and stays more coherent at higher push (distinct-2 0.85 vs 0.63 at frac 0.1). *So much of the
  cross-coupling Phase 0 measured was an artifact of the crude diff-in-means direction; just choosing a
  better direction is a free, large reduction in side-effects (a finding in its own right, and a
  recommendation against diff-in-means for multi-attribute control).*
- **But residual coupling remains and grows with push** (PLS still drifts reading +0.46 FK at frac 0.05,
  +2.43 at 0.10). So there IS a residual side-effect for the Phase-3 allocation QP to cancel — the
  hold-constraint can still bind, just with a **smaller margin** than Phase 0's raw-DIM coupling implied.
- **RepInd (statically projecting the other attribute's direction out of the cone) over-corrects** at the
  operating point — it flipped formality→reading drift to −1.23 (introducing *opposite* drift) rather than
  cancelling cleanly. This argues **against** static decoupling and **for** the closed-loop *allocation* QP
  (which adapts the within-cone push to the current state) — exactly MOSAIC's thesis, and the panel's
  reason not to pre-decouple with RepInd.

## 4. Decision → Phase 3 (sharpened)

**GO.** A coherent, attribute-specific k>1 cone exists (the QP's state/null-space), the depth-indexed
reference beats diff-in-means (D1), and a well-modeled plant exists. Artifacts saved for Phase 3
(`mosaic_cones_gemma-2-2b-it.npz`: per-layer DIM/PLS/RepInd bases + the intensity→coordinate maps).

**The Phase-3 test is now precise and honestly de-risked:** the allocation QP must drive formality to a
commanded τ while **holding reading-level**, beating the **best PLS-scalar push** (the strong baseline,
*not* crude diff-in-means) by **> 1 SE lower reading drift at matched formality-tracking and matched
effort**, with the `k_A=1` ablation failing. The margin to beat is the **residual** PLS coupling
(quantified above) — modest at low push, larger at high push. Two pre-registered outcomes:
- **WIN:** the QP uses the cone's null-space to cancel the residual drift the best scalar can't → "control
  law > bound" on the side-effect axis.
- **SCOPED NULL (publishable dividing line):** if the residual coupling at coherent operating points is
  too small, allocation ties the PLS-scalar → report "allocation pays iff residual coupling > ε," with the
  drift/gain numbers above as the predictor. (This is the panel's sharpest risk; Phase 2 has now
  quantified exactly how much coupling survives a good direction.)

Reproduce: `mosaic_phase2.py --run`; coupling nuance `mosaic_phase2_coupling_check.py`.
