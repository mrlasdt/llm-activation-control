# MOSAIC Phase 1 — graded sensors + monotone-frontier verification: results

**Verdict: GO for the formality ↔ reading-level pair (the headline coupled pair); sentiment
demoted to a hold-only / externally-scored target (probe fails the sensor gate).** The
strength↔coherence relationship is a **monotone trade-off, not a refusal-style hump** — the
non-saturating regime the prior nulls lacked. This is the precondition for control to be load-bearing.

Run: `../../.venv/bin/python mosaic_phase1.py --run` (Gemma-2-2b-it, band [7,24], pscale 180.9;
probe pool 32×7=224 style-diverse texts, frontier n_eval=32, 128-tok greedy). Date: 2026-06-14.
Data: `outputs/mosaic_phase1_gemma-2-2b-it.{json,png}`; probes: `outputs/mosaic_probes_gemma-2-2b-it.npz`.

---

## 1. Probe calibration — the online sensor gate (r > 0.85)

A linear (ridge) probe on the residual stream (mean-pooled over the response, per band layer),
regressed against the external scorer, held-out Pearson r:

| attribute | best held-out r | layer | last-token r | gate (>0.85) |
|---|---:|---:|---:|:--:|
| **reading-level** (FK grade) | **0.942** | 12 | 0.80 | **PASS** |
| **formality** (P_formal) | **0.912** | 14 | 0.81 | **PASS** |
| sentiment (P_pos) | 0.727 | 7 | 0.66 | **FAIL** |

Formality and reading-level are **strongly linearly readable** off the residual stream (mid-band,
r ≈ 0.91–0.94) — a faithful per-token sensor exists, so a closed loop will track the true attribute,
not a biased proxy. **Sentiment is only weakly linearly readable (r = 0.73)**, *and* it saturates near
the classifier ceiling early (see §2), so it does not earn a probe-based online controller. Per the
pre-registered rule ("carry only the attributes that pass"), MOSAIC's headline becomes the
**formality ↔ reading-level** MIMO — which is exactly the **most strongly-coupled pair** from Phase 0
(direction cosine +0.50). Sentiment is retained as a **hold-only target scored by the external
classifier** (detecting drift needs no probe), or dropped. *If* a probe-based sentiment sensor is later
required, its r=0.73 noise is the documented niche for the **OAS Kalman observer** (the proposal's
contingency).

## 2. The monotone strength↔coherence frontier (the load-bearing finding)

Sweeping the bounded additive push (frac · pscale per band-layer), driven-attribute intensity vs the
coherence/degeneration axes:

**formality** (probe r 0.91):

| frac | intensity P_formal | genNLL | TTR | distinct-2 |
|--:|--:|--:|--:|--:|
| 0.00 | 0.601 | 0.42 | 0.79 | 0.96 |
| 0.025 | 0.744 | 0.67 | 0.79 | 0.97 |
| 0.05 | 0.813 | 1.09 | 0.75 | 0.96 |
| 0.075 | 0.901 | 1.71 | 0.68 | 0.93 |
| 0.10 | 0.941 | 2.15 | 0.62 | 0.87 |
| 0.15 | 0.961 | 2.11 | 0.38 | 0.57 |
| 0.20 | 0.967 | 1.81 | 0.22 | 0.35 |

**reading-level** (probe r 0.94): intensity 11.0 → 15.0 → 17.6 → 22.2 (coherent) → 28.6 → 45.6 → 53.3 FK
grade; distinct-2 0.96 → 0.97 → 0.93 → 0.82 → 0.56 → 0.40.

**Reads:**
1. **Monotone, not a hump.** Over the coherent range (distinct-2 ≥ 0.85), intensity rises **strictly
   monotonically** — formality Spearman(frac, intensity) = **1.00**, reading = **1.00** — while
   coherence (distinct-2) degrades monotonically. You can pick **any** operating point on the trade-off
   curve. This is the qualitative opposite of refusal (CALM's hump, where past threshold the behaviour
   metric *fell* while incoherence rose — losing on both axes). **The frontier a controller can navigate
   exists here.**
2. **Operating envelope:** coherent up to frac ≈ **0.075–0.1**; collapse by 0.15–0.2 (distinct-2 → 0.35,
   emoji/word-salad). So the MOSAIC controller's `u_max` should keep the per-layer push in
   ~[0.05, 0.1]·pscale — re-confirming Phase 0's ceiling and the program's "bound is a lever."
3. **genNLL is NOT a clean coherence axis for graded attributes.** It rises 0.42 → 2.5 with push, but
   *partly because the intended attribute shift makes the text less likely under the unsteered
   (casual-baseline) model* — i.e. genNLL conflates "attribute changed" (the signal) with degeneration.
   **distinct-2 / TTR (repetition/degeneration) are the faithful coherence axes here** and are what the
   coherent-range gate uses. (genNLL even *falls* slightly at frac 0.2 as output collapses into
   low-entropy repetition — another reason not to use it as the coherence cost for this regime; the
   token-axis ambient-KL signal from CALM is the right one for Phase 4.)

**sentiment** (probe r 0.73): intensity **saturates early** — 0.57 → 0.77 → 0.86 → 0.95 by frac 0.075,
then a tiny ceiling dip to 0.94 at 0.1 (Spearman 0.90, a knife-edge driven by saturation noise, not a
hump), then emoji-spam degeneration by 0.15–0.2 (*"🎉🎉🎉🎉🎉"*). So sentiment's dose-response is
essentially monotone-then-**saturated** near the classifier ceiling — less useful as a *driven* graded
dial, fine as a *hold* target.

## 3. Decision → Phase 2

**GO**, scoped to the **formality ↔ reading-level** headline pair:
- both clear the probe gate (r 0.91 / 0.94) ⇒ faithful per-token sensors exist;
- both show a strictly-monotone strength↔coherence frontier (no hump) ⇒ the non-saturating regime;
- they are the most strongly cross-coupled pair (Phase 0 cosine +0.50) ⇒ the hold-constraint will bind.

Sentiment is carried as an **externally-scored hold-only** attribute (or dropped); its weak probe is the
OAS-observer contingency, not a blocker for the headline.

**Phase 2** (next): build the graded **k>1 retain-loss cones** (`casa_cone.rdo/rco` retargeted from
refusal to formality and reading-level via low/high-intensity contrastive splits) and fit the
**depth-indexed setpoint map** `r_{A,l}(τ_A)` that replaces diff-in-means (the D1 contribution); gate on
k>1 cone coherence + the depth map beating a single diff-in-means vector on calibration. Then **Phase 3**:
the constrained allocation QP (drive formality to τ, hold reading at baseline — and vice versa) vs the
best scalar bound, win = >1×SE lower off-target drift at matched tracking + effort, with `k_A=1` failing.

Reproduce: `../../.venv/bin/python mosaic_phase1.py --run`. Probes saved for Phase 3/4 in
`outputs/mosaic_probes_gemma-2-2b-it.npz` (per-attribute w, b, layer, feature mean/std).
