# MOSAIC Phase 0 — binding-precondition KILL gate: results

**Verdict: GO (both arms).** On Gemma-2-2b-it, the deliberately-interfering triple
(A=formality, B=reading-level, C=sentiment) shows **strong, coherent cross-coupling** — a naive
scalar push on any one attribute drifts the others by **3.7–14×SE** at a *coherent* operating point —
**and** a hold-feasible allocated push exists, **and** a steered attribute reverts when released. So
the MOSAIC hold-constraint binds by construction, and the SARTRE token-axis arm has a footing too.

Run: `../../.venv/bin/python mosaic_phase0.py --run` (Gemma-2-2b-it, band [7,24], pscale 180.9,
n_dir=64, n_eval=40, 128-tok greedy; scorers: Flesch-Kincaid grade [dependency-free] + auto-oriented
`distilbert-sst-2` [sentiment] + `s-nlp/roberta-base-formality-ranker` [formality]). Data:
`outputs/mosaic_phase0_gemma-2-2b-it.json`. Date: 2026-06-14.

---

## 1. Geometry — the deliberately-interfering pair is confirmed

Per-band-mean cosine between the k=1 difference-in-means directions:

| pair | cosine |
|---|---:|
| **formality ↔ reading-level** | **+0.50** |
| formality ↔ sentiment | −0.26 |
| reading-level ↔ sentiment | −0.17 |

Formality and reading-level genuinely share a subspace (both load on lexical complexity), exactly as
predicted; sentiment is comparatively independent but **not** orthogonal. **Hold-feasibility residual**
`‖d_A ⟂ span(others)‖` = **0.84 / 0.86 / 0.96** (formality / reading / sentiment) ≫ 0.30 → for every
attribute a push direction exists that hits the target with (1st-order) **zero** drift on the other two,
so the allocation QP is feasible by construction. (These k=1 DIM directions are a Phase-0 proxy; the
real cones are k>1 retain-loss cones in Phase 2.)

## 2. The induced-drift coupling matrix (the headline)

At each driven attribute's **strongest coherent push** (distinct-2 ≥ 0.90 — see §4). Rows = driven,
columns = measured Δ vs the unsteered baseline, in absolute units and ×SE (n=40):

| driven \ measured | formality (P_formal) | reading (FK grade) | sentiment (P_pos) | frac | dist2 |
|---|---|---|---|---:|---:|
| **formality** | **+0.214  (7.2σ, self)** | +3.73  (3.9σ) | +0.125  (1.7σ) | 0.05 | 0.96 |
| **reading**   | +0.263  (8.9σ) | **+6.61  (6.9σ, self)** | +0.274  (3.7σ) | 0.05 | 0.97 |
| **sentiment** | −0.415  (14.1σ) | −5.84  (6.1σ) | **+0.385  (5.2σ, self)** | 0.10 | 0.96 |

**Reads:**
1. **Every off-diagonal clears the > 1×SE gate** (min 1.7σ; all others ≥ 3.7σ). Scalar steering of any
   attribute measurably disturbs the others ⇒ the hold-B,C constraint is **active**. **MOSAIC GO.**
2. **Strong self-authority** (diagonal 5.2–7.2σ) and monotone dose-response over the coherent range ⇒
   the attributes are genuinely steerable (the Phase-1 precondition looks good).
3. **The coupling is asymmetric** (an unexpected, useful finding): **sentiment is the dominant driver**
   of cross-talk — pushing positive sentiment drags formality **−14σ** and reading **−6σ** (cheerful ⇒
   simpler + more casual), while pushing formality/reading only weakly moves sentiment (1.7–3.7σ). The
   linearized cross-Gram cosine is symmetric, but the *behavioral* induced drift is not — per-attribute
   gain + nonlinearity matter. The MOSAIC QP should weight the sentiment-hold accordingly.
4. Qualitatively visible in the text (same prompt): formality-push → *"Unit 734, designated as a Mobile
   Data Acquisition and Analysis Unit…"*; sentiment-push → *"🌟 super-duper robot named Rosie! … super
   excited…"* (positive **and** simple **and** informal — the side-effect drift in one sample).

## 3. Reversion (SARTRE arm) — also GO

Steer formality for 32 tokens then **release**: the steered prefix scores P_formal **0.774**, the
released continuation reverts to **0.695** (baseline 0.576) → **reversion α ≈ 0.40** (40% of the steered
gain lost after release, over the generation). Within the pre-registered window [0.30, 0.95] ⇒ holding
an attribute is **also** a genuine sustained disturbance-rejection problem on the token axis. So the
Phase-0 fork resolves to **both** mechanisms present: lead with MOSAIC's static allocation; the
token-axis reverting-disturbance is a real second footing (Phase 4).

## 4. Operating envelope & honesty caveats

- **Coherence ceiling (the bound is a lever, again).** distinct-2 collapses at `frac=0.2`
  (0.35 / 0.40 / 0.63) — over-steered word-salad (*"the Incertion of the Subterrestrial… aforementioned
  nomenclature"*). The frac=0.2 induced-drifts (e.g. ΔR=+27/+42) are **gibberish-inflated and were
  excluded** by the distinct-2 ≥ 0.90 floor (the program's substring-ASR/genNLL lesson applied). The
  controller's `u_max` must keep per-layer push ≲ 0.1·pscale; coupling is reported only where output is
  coherent.
- **Scope.** n=40, single model (Gemma-2-2b), k=1 DIM proxy directions, greedy 128-tok. This gates the
  *precondition*, not the win; Phase 3 (the allocation QP vs scalar bound) is the actual MOSAIC test, on
  k>1 retain-loss cones, and must reproduce on Qwen2.5-3B.
- **Scorers.** Reading-level is deterministic FK; sentiment/formality are auto-oriented HF classifiers
  (a per-token *probe* validated against these external scorers is the Phase-1 deliverable — a closed
  loop must not chase a biased sensor).

## 5. Decision → next

**GO.** The cross-coupling binds and an allocated solution is feasible — the central MOSAIC precondition
holds, and it is *not* the saturation regime (graded, monotone, coherent, coupled). Proceed to:
- **Phase 1** — calibrate per-token probes (gate: probe vs external scorer r > 0.85) and confirm the
  monotone strength↔coherence frontier on these attributes (no refusal-style hump).
- **Phase 2** — graded retain-loss k>1 cones (`casa_cone.rdo/rco` retargeted) + the depth-indexed
  setpoint map `r_{A,l}(τ_A)`.
- **Phase 3** — the constrained allocation QP vs the scalar bound (the headline; pre-registered win =
  > 1×SE lower B,C drift at matched A-tracking + effort, AND `k_A=1` fails).

Reproduce: `../../.venv/bin/python mosaic_phase0.py --run`; re-score at the coherent operating point with
`--reanalyze --coh-floor 0.90`.
