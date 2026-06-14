# MOSAIC Phase 3 — constrained allocation vs the best scalar push: results

**Verdict: SCOPED NULL — constrained null-space allocation does NOT beat the best PLS-scalar push.**
The (attribute-gain, side-effect-drift) frontier is **shared** across scalar / allocation /
single-decoupled-direction; the null-space allocation reaches a *different point on the same frontier*
(lower gain **and** lower drift) but never **dominates** it (no >1 SE tighter hold at matched
attribute-tracking). MOSAIC's headline claim — that a binding hold-constraint lets a control law beat
the bound on the side-effect axis — **is not supported for these attributes.** The reason is the
valuable finding, and it is mechanistic.

Run: `mosaic_phase3.py --run` (Gemma-2-2b-it, band [7,24], pscale 180.9, k=4 PLS cones; n_eval=40,
6-frac sweep, 96-tok greedy; both drive directions × {scalar, alloc, repind/k1}). Date: 2026-06-14.
Data: `outputs/mosaic_phase3_gemma-2-2b-it.{json,png}`.

---

## 1. The headline comparison — shared frontier, no dominance

Drive **formality**, hold **reading-level** (the materially-coupled direction). |reading-drift| at
matched formality-gain (interpolated over each method's coherent sweep, distinct-2 ≥ 0.85):

| formality-gain | alloc \|B-drift\| (±SE) | scalar \|B-drift\| @ same gain | result |
|---:|---:|---:|---|
| +0.114 | 2.24 ± 0.39 | 2.28 | ~tie |
| +0.133 | 3.50 ± 0.62 | 2.70 | alloc higher |
| +0.150 | 2.78 ± 0.55 | 3.09 | ~tie |

Paired (same prompts) at the matched operating point: |reading-drift| `scalar − alloc = −0.17 ± 0.52
SE` — a tie, not a >1 SE win. **No coherent operating point exists where allocation holds reading
tighter than scalar by >1 SE.** And allocation **sacrifices attribute range**: it caps at
formality-gain **+0.150** (coherently) while scalar reaches **+0.235** — orthogonalizing the push to
the held direction costs driving authority. The figure (`mosaic_phase3_gemma-2-2b-it.png`) shows all
three methods on one overlapping curve.

Drive **reading**, hold **formality**: formality barely drifts under scalar reading-steering
(|formality-drift| 0.08–0.38 across the whole sweep) — **near-decoupled**, so there is nothing to
allocate (expected tie). Worse, the allocation/orthogonalization **guts the reading authority**
(reading-gain collapses to ~0 or negative at frac ≥ 0.06) — you cannot drive reading without formality.

## 2. Mechanism — the coupling is semantic entanglement, not a removable linear subspace

The formality↔reading coupling is a **genuine semantic entanglement** (formal text *is* inherently more
lexically complex), not a linear-subspace artifact a null-space can remove. Three lines of evidence:

1. **Shared frontier (§1):** no method holds the off-target attribute tighter at matched driven-gain —
   the (gain, drift) trade-off is governed by the *aggregate* intervention, not its allocation in the
   cone null-space.
2. **Symmetric inseparability:** orthogonalizing the *reading* push to the *formality* direction
   destroys the reading effect (reading-gain → ~0/negative). The two attributes cannot be linearly
   separated in activation space in *either* direction.
3. **Downstream re-entanglement:** the hold is an *open-loop, first-order* projection at the band layers
   (push ⊥ the held direction). Even when it zeroes the held attribute's coordinate *there*, the
   autoregressive generation re-introduces the complexity downstream, because "write formally" entails
   "write complexly" in the model's generation process — which a per-layer linear projection cannot veto.

## 3. The through-line — a fourth instance of "control sophistication ties the lever"

| experiment | sophistication added | outcome |
|---|---|---|
| PTS | multi-step lookahead | ties (near-identity plant) |
| CALM 1–2 | coherence cost in the objective | ties (wrong space, r=0.20) |
| CALM 3 | dynamic per-token KL feedback | ties (refusal saturates) |
| **MOSAIC P3** | **cone null-space allocation (binding hold)** | **ties (coupling is semantic, not linearly separable)** |

The strength↔side-effect trade-off is governed by the **aggregate intervention** — not its allocation in
depth (PTS), in time (CALM 3), or now **in the cone null-space (MOSAIC)**. The durable levers remain the
**actuator** (additive ≫ rotation) and the **bound** — plus the one genuine positive this investigation
added (§4).

## 4. The real, durable win of this investigation — the *direction*, not the controller (Phase 2)

The actionable result is from Phase 2, not Phase 3: **a supervised, attribute-specific steering
direction (PLS) causes ~80% less side-effect drift than raw difference-in-means at matched attribute
gain** (drift/gain 2.5 vs 12.6). Most of the side-effect the field worries about (Course-Correction's
"side effects dominate the failure budget") is, for separable-enough attributes, a **bad-direction
artifact** — fixed for free by choosing a better *direction*, not by a constrained controller. That is
the practical recommendation: use a supervised attribute-specific direction; don't expect a null-space
allocation to fix what's left when the residual coupling is semantic.

## 5. Honest caveats — what is NOT ruled out

1. **Attribute choice (the sharpest caveat):** Phase 0 deliberately picked the *most strongly-coupled*
   pair so the hold-constraint would bind — but maximal coupling here means maximal *semantic*
   entanglement, i.e. the **least linearly-separable** case. MOSAIC's premise needs coupling that is a
   *representational* artifact (separable by a learned cone) rather than semantic. formality/reading is
   the worst case for separability; a coupled-but-separable pair (e.g. an attribute whose side-effect is
   a removable stylistic correlate, not an inherent semantic one) could still allocate. *We chose the
   pair that binds the constraint hardest, which is also the pair allocation can least fix.*
2. **Open-loop vs closed-loop:** the linear null-space hold is open-loop. A **closed-loop token-level
   hold** — regulate the *measured* reading-level (the Phase-1 probe, r=0.94) back to baseline each token
   during generation — directly targets the *output* and could succeed where activation-space
   orthogonalization fails (downstream re-entanglement is a generation-time phenomenon, so it needs
   generation-time feedback). This is Phase 4 (the SARTRE arm), now strongly motivated.
3. n=40, single model (Gemma-2-2b), per-prompt Flesch-Kincaid noise (SE ~0.5 FK); k=4 PLS cones;
   first-order hold (the general QP with a hard KL budget / multi-hold was not needed — the simpler
   closed-form already showed the null-space doesn't help here).

## 6. Decision

**MOSAIC's headline (allocation > scalar on the side-effect axis) is falsified for semantically-entangled
attributes** — a clean, pre-registered negative caught by the rigor (the dividing line is now concrete:
allocation cannot beat scalar when the residual coupling is semantic rather than linearly-separable). The
investigation's durable output is the **direction-choice finding (§4)** and the mechanistic
**through-line (§3)**.

Two honest continuations, in priority order:
- **Phase 4 — closed-loop token-level hold (SARTRE):** regulate the measured reading-level to baseline
  while driving formality, using the verified decode loop + the Phase-1 probe. This tests whether
  *output-feedback* beats *open-loop activation projection* against downstream re-entanglement — a
  genuinely different mechanism from Phase 3 and from CALM-3 (which had no faithful graded signal). The
  Phase-0 reversion result (α≈0.40) already showed holding is a real sustained-disturbance problem.
- **Separability re-test:** repeat Phase 3 on a *coupled-but-linearly-separable* attribute pair (measure
  separability first: is the side-effect removable by a learned cone, or semantic?). MOSAIC only wins if
  such a pair exists; if not, the honest conclusion is that the side-effect problem is semantic and
  belongs to output-feedback (Phase 4) or direction-choice (Phase 2), not null-space allocation.

Reproduce: `mosaic_phase3.py --run`; frontier-dominance check in the JSON `results`.
