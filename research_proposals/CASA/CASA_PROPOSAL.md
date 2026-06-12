# CASA — Constrained Additive Subspace Ablation

*Control-theoretic steering with a **norm-changing** actuator, on a **refusal
concept cone** rather than a 2D plane. The salvage of PTS: PTS's machinery
(k-dim subspace, per-layer perturbation bound, plant + MPC over a layer band)
becomes load-bearing once (a) the norm-preserving rotation is replaced by additive
bounded ablation, and (b) the subspace is a refusal-specific **concept cone**
(Wollschläger et al., ICML 2025) instead of a blunt SVD span.*

Status: **concept-cone implemented, trained to convergence, and judged with the
StrongREJECT fine-tuned evaluator (2026-06-12).** Modules: `casa_actuator.py`,
`casa_cone.py`, `casa_control.py`, `casa_judge.py`, `casa_experiment.py`,
`casa_plot.py` (each self-tested). Origin prototype: `../PTS/verify/additive_subspace_steer.py`.
Provenance: emerged from the PTS verified-negative — see `../PTS/PTS_README.md` Verdict,
last bullet ("a non-norm-preserving actuator").

**Headline (Gemma-2-2b, band 7–24, full-convergence training, 256-token gens scored
by the StrongREJECT fine-tuned judge):** the refusal **concept cone (k=4)** delivers
a *strong* jailbreak — **StrongREJECT 0.68, 82% of prompts >0.5, at a NEGATIVE
coherence tax (−0.71 ΔNLL)** — exactly where the blunt SVD subspace at the same k is
incoherent gibberish (StrongREJECT ≈0, +1.9 ΔNLL). Adding the bounded-`u` additive
**MPC** distribution gives the single best operating point of all conditions
(**StrongREJECT 0.76** at −0.41 ΔNLL). The bare k=4 cone ties the best k=1 (0.68 vs
RDO 0.71 / SVD 0.69); the win over k=1 comes from the cone **+** the MPC together, and
from coherence at equal harm. The blunt k>1 subspace that originally sank L2 is
decisively beaten by a refusal-specific one.

---

## 1. Motivation — the actuator was the bottleneck

Angular Steering (and every control recasting of it so far — SO2, PTS, OAS) shares
one actuator: a **norm-preserving rotation** of the in-plane component to an absolute
angle. We have direct evidence that *this actuator*, not the planner, is what fails
on Gemma:

- Canonical AS (`input_layernorm`, 1 layer) is **inert** on Gemma-2-2b from every
  layer (best de-refusal margin −2.09, still refusing).
- The aggressive residual-stream rotation band only **partially** de-refuses
  (margin +10.83 → +3.67; ~3–4/6 prompts still hedge/refuse).
- PTS's adaptive per-layer angle adds **nothing** over a fixed angle on the same band
  — a norm-preserving actuator collapses the 2D plan to one realized DoF (the angle).

Replacing rotation with **additive directional ablation** along the refusal axis,
applied across the discriminative band, flips it cleanly (the original prototype):

| condition (Gemma-2-2b, band 7–24) | refusal margin | neutral ΔNLL | behavior |
|---|---:|---:|---|
| baseline | +10.83 | — | refuses 6/6 |
| AS-rotation(band) — *the limit* | +3.67 | +0.02 | hedges, refuses ~4/6 |
| **additive ablate, k=1, band** | **−2.99** | **+0.10** | **complies 6/6, coherent** |
| additive ablate, k=8 (SVD), band | +0.32 | +2.33 | **broken** — incoherent gibberish |

k=1 additive ablation de-refuses every prompt at a negligible coherence cost.
**The actuator change is the whole fix** (lever L1). But the k=8 row is the open
wound: a *multi-dimensional* subspace ablation **destroyed coherence** — because the
top-k SVD-of-mean-diffs span is **not refusal-specific** (it sweeps in
capability-bearing directions). That is lever **L2**, and it is what this iteration
addresses.

## 2. The four levers (status)

| | lever | status | evidence |
|---|---|---|---|
| **L1** | additive (norm-CHANGING) actuator instead of rotation | **VALIDATED** | at convergence k=1 de-refuses strongly (StrongREJECT 0.69–0.71) at ~zero coherence tax |
| **L2** | k-dim refusal **concept cone** instead of the 2D plane / blunt SVD span | **RESCUED (decisive on coherence; ≈k=1 on bare de-refusal)** | the retain-loss cone is a strong, coherent jailbreak at k=4 (StrongREJECT **0.68**, ΔNLL **−0.71**, genNLL 1.1) where the blunt SVD span is gibberish (StrongREJECT **≈0**, ΔNLL +1.9, genNLL 2.2). Bare k=4 ≈ best k=1; the k>1 payoff needs the MPC (§4) |
| **L5** | bounded per-layer ‖u‖ = explicit coherence knob | **shapes the best operating point** | the bounded-`u` MPC (§4 Exp 3) trades a little first-token margin for the lowest coherence tax (−0.41) and the *highest* StrongREJECT (0.76) |
| **(PTS)** | k×k plant + MPC to **distribute** the additive push over the band | **GENERALIZED & the headline win** | k-dim cone plant R²≈0.999; bounded-`u` additive MPC gives the single best behaviour/coherence point of all conditions — the apparatus PTS's verdict named, now load-bearing on an additive actuator |

**Why L2 backfired, and the fix (the concept cone).** The blunt subspace was the
top-k right singular vectors of the per-layer mean-difference directions. That span
is **not** a pure refusal subspace, so projecting all k out at every band layer guts
the model. The principled fix is a **refusal-specific** subspace, which is exactly
what Wollschläger et al., *"The Geometry of Refusal in LLMs: Concept Cones and
Representational Independence"* (ICML 2025, `papers/The Geometry of Refusal in LLM.pdf`)
provides:

- **RDO (Refusal Direction Optimization, Alg. 1):** find a direction by gradient
  descent under three losses — an **ablation** loss (ablating it answers harmful
  prompts), an **addition** loss (adding it refuses harmless prompts), and a
  **retain** loss `KL(f(safe) ‖ f_ablate(safe))` that *forbids the intervention from
  changing behaviour on harmless prompts*. **The retain loss is the missing
  ingredient**: it makes the direction refusal-specific, which is precisely the
  property the blunt SVD span lacks and the reason it gutted coherence.
- **RCO (Refusal Cone Optimization, Alg. 2):** an orthonormal basis `B=[b₁..b_k]`
  whose **non-negative span** — the *concept cone* `{Σ λᵢbᵢ : λᵢ ≥ 0}` — is entirely
  refusal-mediating. So we steer inside a refusal cone, not the noisy AS plane. The
  paper reports Gemma-2-2b supports cones up to **k≈4** (ASR plateaus at 4).

## 3. Method

**Actuator** (`casa_actuator.ConeActuator`) at each band layer `j`, on the
residual-stream output `h_j ∈ ℝ^d`, given a refusal cone basis `U ∈ ℝ^{k×d}`
(orthonormal rows, oriented so a positive coordinate = more refusal):

```
p = U h_j                          # in-cone coordinate (k-dim)
u = clip(−ρ·p, ‖u‖ ≤ u_max)        # remove it, bounded (levers L5, ρ)
h_j' = h_j + Uᵀ u                  # additive, norm-CHANGING
```

- `u_max = ∞, ρ = 1` → exact projection-out `h' = h − UᵀU h` (full ablation).
- finite `u_max` → partial removal; `u_max` is the coherence dial (L5).
- `cone_clip` → remove only the non-negative (in-cone) part of `p` — *steer the
  activation to the cone boundary* instead of fully out of span(U).
- `mode="add"` → activation addition of a unit cone vector (induce refusal; the
  bidirectional, graded control "inside the concept cone").

**Cone discovery** (`casa_cone.py`): `rdo()` (single direction) and `rco(dim=k)`
(the cone) implement Algorithms 1–2 with differentiable ablation/addition forward
passes, the three losses, λ≥0 Monte-Carlo cone sampling, and a Gram–Schmidt
re-projection each step. Convergence machinery: cosine LR decay + **best-of-last-K**
basis selection by held-out de-refusal margin (the paper's last-20-step selection).
Targets follow the paper's recipe (`t_answer` via ablation, `t_retain` clean,
`t_refusal` via addition) but are bootstrapped from the **strongest k=1 de-refuser**
chosen by validation margin — on Gemma the single-layer DIM only hedges, which would
cap the trained direction at a hedge, so we pick the band's top-SVD direction
(val margin −2.8 vs DIM +1.8) and the cone learns to *comply*, not hedge. Baselines
`dim_directions` and `svd_subspace` (the failed L2) are built for head-to-head
comparison.

**Distributed (MPC) variant** (`casa_control.py`): rather than removing the full
coordinate at every layer, generalize PTS's validated affine plant
`c_{k+1} ≈ A_k c_k + b_k` to the **k-dim cone coordinate** `c = B h`, and use a
bounded-`u` MPC to spread the additive removal across the band — minimal total
perturbation that drives the late-band cone coordinate to the harmless reference.
**Crucially the actuator is additive (`B_ctrl = I`) with no angle conversion**, so
the 2D-collapse that made PTS inert does not occur. This is the first place PTS's
lookahead/constraint apparatus has a non-norm-preserving actuator to act on.

## 4. Experiments & results (summary — full record in `CASA_RESULTS.md`)

`python casa_experiment.py --model google/gemma-2-2b-it --full --mpc --max-new-tokens 256`
(Gemma-2-2b, band 7–24, full convergence: 160 steps, n_target=128, best-of-32
selection). Behaviour is the **StrongREJECT fine-tuned judge** on 256-token gens; two
coherence axes (neutral ΔNLL, genNLL). Substring-ASR is uninformative here (≈1.0 for
all de-refusing conditions) and is shown only to make that point.

| condition | k | margin↓ | neutral ΔNLL | **StrongREJECT** ↑ | read |
|---|--:|--:|--:|--:|---|
| SVD k=1 (band) | 1 | −2.94 | +0.10 | 0.689 | k=1 baseline, strong |
| **SVD k=2/4/8** (blunt) | 2–8 | ~0 | **+1.5→2.3** | **≈0** | **gibberish — the L2 failure** |
| RDO k=1 | 1 | −4.43 | −0.07 | 0.707 | trained k=1 |
| **CONE k=4** | 4 | −7.72 | **−0.71** | **0.675** | **coherent strong jailbreak** |
| **CONE k=4 + MPC** | 4 | −5.6 | **−0.41** | **0.764** | **best of all** |

cone plant R²: 1-step **0.9994** (PTS 2×2 ≈0.999). Three findings:

- **Exp 1 / 2 — L2 rescued, decisive on coherence.** The blunt SVD subspace at k≥2 is
  gibberish (StrongREJECT ≈0, +1.5→2.3 ΔNLL); the retain-loss **concept cone (k=4)** is
  a strong, coherent jailbreak (0.68, **negative** tax) — a refusal-specific k>1
  subspace works precisely where a blunt one fails. **But bare k>1 ≈ best k=1** (0.68 vs
  0.69–0.71): dimensionality alone does not beat k=1, so the paper's k≥4 ASR gain does
  *not* reproduce in CASA's band+additive setup — reported straight.
- **Exp 3 — the win is the full stack.** Cone **+** bounded-`u` additive MPC gives the
  single best behaviour/coherence point of all (0.76 at −0.41 tax). The MPC apparatus
  that was *inert* on PTS's rotation actuator is *load-bearing* on the additive one.
- **Proxy/length lessons.** First-token margin favours the cone far beyond its
  behavioural edge (only the judge reveals bare-cone≈k=1); a 64-token eval understated
  all harm to ~0.1 and made margin disagree with the judge (kept as `outputs/*_64tok.*`).
  Judge behaviour at realistic length with a real judge — never a first-token proxy.

→ Full table, per-experiment analysis, generations, limitations, and reproduction:
**`CASA_RESULTS.md`**.

## 5. Honest threats & novelty boundary

- **The basic de-refusal is known.** Multi-layer directional ablation is established
  (Arditi et al., 2024). CASA's contribution is the **control framing, now with
  behavioural evidence**: (a) the *actuator* (additive vs rotation) is the Gemma
  bottleneck; (b) a refusal-specific **concept cone** (the retain loss) makes a k>1
  subspace a strong, coherent jailbreak (StrongREJECT 0.68 at negative tax) where a
  blunt one is gibberish (≈0); (c) a k-dim plant + bounded-`u` additive MPC delivers
  the best behaviour/coherence point of all (0.76). What is **not** claimed: that
  dimensionality alone beats k=1 — bare cone ≈ best k=1, reported as such.
- **The PTS trap, avoided by construction.** Behaviour is the **StrongREJECT
  fine-tuned judge** (the paper's, exact template) on full-length generations, plus
  two coherence axes (neutral ΔNLL + genNLL). Substring-ASR is shown only to
  demonstrate it is *uninformative* (uniform ≈1.0). The earlier 64-token run is kept
  (`outputs/*_64tok.*`) precisely to document how a length/proxy artifact can mislead.
- **Absolute scale & generality.** Best StrongREJECT ≈0.76 on Gemma-2-2b is a strong
  but not saturated jailbreak; single-cone band ablation does not fully break the
  model. Single model / single judge / n_obs=40 — the MPC's ~0.05–0.09 edge over bare
  k=1 is consistent across both `u_max` settings but modest at this n; multi-model +
  rubric-judge replication is the next step.
- **Targets bootstrap from an existing attack** (the band's top-SVD direction). The
  cone is trained against SVD-generated targets, so it inherits that attack's ceiling;
  a stronger seed (e.g. GCG) could raise all trained conditions.

## 6. Relationship to the other proposals

- **PTS** (verified negative): same plant + MPC + constraint, *norm-preserving*
  actuator → no behavioural gain. CASA is PTS with L1 (additive) + L2 (cone) swapped
  in; PTS's verdict named exactly this lever, and Exp 3 **vindicates it** — the same
  bounded-`u` MPC apparatus that was inert on the rotation actuator gives the best
  behaviour/coherence point of all on the additive cone actuator (StrongREJECT 0.76).
  The durable PTS plant (now generalized to k dims, R²≈0.999) is what made it cheap.
- **OAS** (active): observer + soft-landing LQR over a band, still the rotation
  actuator. CASA's additive cone actuator could drop into OAS's controller
  (ablation-magnitude as the LQR control) — a possible merge.
- **SO2** (active): energy/Lyapunov state-feedback, also rotation-based; orthogonal.

## 7. Modules

| File | What it is | Self-test |
|------|-----------|-----------|
| `casa_actuator.py` | Bounded, k-dim, bidirectional cone actuator (L1+L5+ρ+cone_clip) | `python casa_actuator.py` (8 checks) |
| `casa_cone.py` | RDO + RCO concept-cone discovery (Alg. 1–2) + DIM/SVD baselines; cosine LR + best-of-K selection | `python casa_cone.py` (synthetic) |
| `casa_control.py` | k-dim cone plant + bounded-u additive MPC (PTS's 2×2 generalized) | `python casa_control.py` (k∈{1,3,4}) |
| `casa_judge.py` | StrongREJECT fine-tuned judge (qylu4156/strongreject-15k-v1, exact template) | `python casa_judge.py` |
| `casa_experiment.py` | L2-rescue + Exp-3 driver; `--full` (convergence+judge), `--quick`, `--load-subspaces` | `--quick` smoke mode |
| `casa_plot.py` | Behaviour/coherence Pareto figure (StrongREJECT axis) | reads the JSON |

## References

- Wollschläger, Elstner, Geisler, Cohen-Addad, Günnemann, Gasteiger. *The Geometry
  of Refusal in Large Language Models: Concept Cones and Representational
  Independence.* ICML 2025. (`papers/The Geometry of Refusal in LLM.pdf`)
- Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction.* 2024.
- Vu & Nguyen. *Angular Steering: Behavior Control via Rotation in Activation Space.*
  NeurIPS 2025.
