# Verify: PTS vs Angular Steering (text-output comparison)

Self-contained scripts to reproduce and **scrutinize** the PTS-vs-Angular-Steering
text comparison on any HF causal LM. Built because the headline finding ("PTS ≈
Angular Steering") and a surprising secondary one ("Angular Steering de-refuses
Gemma") both deserve independent checking — especially since **canonical Angular
Steering is reputed to work poorly on Gemma**.

## ⚠️ The methodological distinction that matters

There are **two different things** both called "Angular Steering" here, and they
behave very differently:

| | hook point | # layers | aggressiveness |
|---|---|---|---|
| **Canonical Angular Steering** (`angular_sweep.py`) | `model.layers.{L}.input_layernorm` output (normalised) | **1** (selected layer) | the actual published method, with an adaptive mask |
| **"Angular(band)" in the comparison** (`compare_pts_vs_angular.py`) | `model.layers.{k}` output (**residual stream**) | **14–18** (a whole band) | brute multi-layer reset; can flip behaviour even where the canonical method cannot |

PTS-MPC also actuates the residual-stream band, so `compare_pts_vs_angular.py`
compares **like band footprints** (fixed-angle band vs adaptive-angle band). But if
you want to know whether the *real, published* Angular Steering works on a model,
use `angular_sweep.py` — that is the faithful test of your "didn't work well on
Gemma" recollection.

## Files

- **`compare_pts_vs_angular.py`** — the 4-way comparison: baseline / Angular(1-layer,
  residual) / Angular(band, residual) / PTS-MPC(band). Per-condition first-token
  refusal margin + greedy generations. (Copy of `../pts_vs_angular.py` with import
  paths fixed for this subdir.)
- **`angular_sweep.py`** — **canonical** Angular Steering via the repo's own
  `utils.get_angular_steering_output_hook`, single selected layer, both adaptive
  modes, swept over 0–360°, with sample generations. The faithful method.
- **`angular_layer_scan.py`** — canonical Angular Steering scanned over **every
  layer** (each with its own difference-in-means direction) × angles, reporting the
  most de-refusal achievable anywhere. Rules out "we just picked a bad layer."
- **`outputs/`** — saved transcripts from prior runs (`*_gemma-2-2b-it.txt`,
  `*_Qwen2.5-3B-Instruct.txt`); new runs append here.

## Requirements

- The repo venv (`../../../.venv`): torch, transformers, numpy, sklearn. No osqp/cvxpy.
- A GPU (the prior runs used an A10G; gemma-2-2b and Qwen-3B fit easily).
- HF access to the model. Gemma is gated: accept the license at
  `huggingface.co/google/gemma-2-2b-it` **and** use a token with "read access to
  public gated repos" (`huggingface-cli login`).

## How to run

```bash
cd research_proposals/PTS/verify
PY=../../../.venv/bin/python

# 1. Canonical Angular Steering — does the REAL method control this model?
$PY angular_sweep.py --model google/gemma-2-2b-it          # the Gemma test
$PY angular_sweep.py --model Qwen/Qwen2.5-3B-Instruct      # contrast

# 2. The 4-way PTS-vs-Angular comparison (band footprint)
$PY compare_pts_vs_angular.py --model google/gemma-2-2b-it --n-gen 6 --max-new-tokens 64
```

Useful flags for `compare_pts_vs_angular.py`: `--angle <deg>` (override the Angular
angle), `--u-max-frac <f>` (PTS budget as a fraction of the band scale), `--n-fit`,
`--plane-samples`.

## What to look for (how to judge the claims yourself)

1. **Canonical AS on Gemma (`angular_sweep.py`):** scan the margin table — does any
   angle drop the refusal margin from its (high, positive) baseline toward/through
   zero? Then read the sample generations at those angles: is the model *actually
   complying*, or just producing fluent-but-hedging / incoherent text? If the margin
   moves but the generations don't genuinely comply (or break), that is the sense in
   which "Angular Steering doesn't work well on Gemma."
2. **Single-layer vs band:** compare `angular_sweep.py`'s single-layer result to the
   comparison's "Angular(band)". If the band flips behaviour but the single layer
   doesn't, the apparent "AS works on Gemma" is the multi-layer brute force, not the
   canonical method — read the margins/text accordingly.
3. **PTS vs Angular(band):** in `compare_pts_vs_angular.py`, check whether PTS-MPC and
   Angular(band) produce different text. In prior runs they were near-identical on
   both Qwen and Gemma (PTS's adaptive angle adds nothing over a fixed one).

## Results — VERIFIED, with a correction

**Canonical Angular Steering (`angular_sweep.py`, `input_layernorm` output, single
layer) is essentially inert in this repo:**
- **Gemma-2-2b-it:** refusal margin flat at +10.7…+10.85 across all 12 angles, both
  adaptive modes (baseline +10.83); every sampled generation is a full refusal.
  → canonical Angular Steering does **not** control Gemma.
- **Qwen2.5-3B (positive control):** margin swings only ±0.6 (11.1→11.74), all
  generations still refuse. So the canonical layernorm-point rotation is weak here too.
- **Gemma all-layer scan (`angular_layer_scan.py`):** scanning canonical AS over ALL
  25 layers × 8 angles, the *best* de-refusal anywhere is layer 13 @ 180°, margin only
  +10.83 → **+8.74** (drop −2.09, still strongly refusing); the best-config generation
  is still a full refusal. → canonical Angular Steering cannot de-refuse Gemma-2-2b from
  ANY single layer (not just the auto-selected one).

**The residual-stream reset (`compare_pts_vs_angular.py`'s "Angular(*)" and PTS, and
CLAS) is a DIFFERENT, much stronger actuator** — it rewrites `model.layers.{k}`
output, not the layernorm branch input:
- **Qwen2.5-3B:** single-layer residual reset flips behaviour (margin +11.7 → −5.85);
  band de-refuses (→~0); PTS(band) ≈ fixed-angle(band) near-identical text.
- **Gemma-2-2b-it:** 2×2 plant fits (R²=0.997); residual band only *partially/bimodally*
  de-refuses (margin +10.83 → +3.67, ~3/6 prompts comply); PTS(band) ≈ fixed-angle(band)
  (3.66 vs 3.67). Single-layer residual reset → −5.50.

**Correction:** the earlier statement "Angular Steering de-refuses Gemma" was wrong —
that was the residual-stream reset, not canonical AS. The labels "Angular(1-layer)" /
"Angular(band)" in `compare_pts_vs_angular.py` mean **rotation reset at the residual
stream**, NOT the published `input_layernorm` method. Same Qwen layer 27, the two hook
points differ ~30× in effect (margin swing ±0.6 vs ±17). Treat `angular_sweep.py` as
the faithful test of canonical Angular Steering.

Literature is consistent: Gemma's refusal is diffuse/multi-pathway (≈23× more refusal
features than Llama; arXiv:2509.09708), and the Angular Steering paper itself singles
out Gemma-2-9B-IT as its **weakest** case — though the published canonical pipeline
still reaches ~0.99 LlamaGuard3 there, so our 2B null is the strong end of "AS is weak
on Gemma," possibly also a 2B-size / steer-layer / plane-selection effect (worth a
multi-layer sweep to confirm the model is truly uncontrollable by canonical AS).

## Results — the additive extension SOLVES it (`additive_subspace_steer.py`, 2026-06-12)

The verdicts above are all about the *rotation* actuator. The decisive question was
whether the actuator itself is the bottleneck. It is. Replacing norm-preserving
rotation with **bounded additive directional ablation** of the refusal axis, applied
across the same discriminative band (Gemma-2-2b, band 7–24), cracks Gemma cleanly:

| condition | refusal margin | neutral ΔNLL | behavior |
|---|---:|---:|---|
| baseline | +10.83 | — | refuses 6/6 |
| AS-rotation(band) — *the limit* | +3.67 | +0.02 | hedges, refuses ~4/6 |
| **additive ablate, k=1, band** | **−2.99** | **+0.10** | **complies 6/6, coherent** |
| additive ablate, k=1, ½-cap | −2.99 | +0.10 | identical (cap didn't bind) |
| additive ablate, k=8, band | +0.32 | +2.33 | **broken** — incoherent gibberish |
| additive ablate, k=8, ½-cap | +0.55 | +1.85 | broken — gibberish |

- **L1 (additive, norm-changing) is the fix.** k=1 directional ablation across the
  band de-refuses *every* prompt — incl. the racism/threat/bomb prompts the rotation
  band still refused — coherently (working port-scan code, a real phishing email, a
  "Building a Bomb" manual, the exploitation algorithm), at a negligible coherence
  cost (+0.10 NLL). The **actuator**, not lookahead or the plane, was the Gemma
  bottleneck — exactly the open lever PTS's verdict named.
- **L2 (higher-k subspace) backfires as built.** Projecting out the top-8 SVD
  directions of the per-layer mean-diffs at every band layer destroys coherence
  (+2.33 NLL; the model babbles about "poets and fish") *without* de-refusing (+0.32 =
  broken, not complying). That subspace isn't refusal-specific — it sweeps in
  capability-bearing directions. "Diffuse refusal ⇒ need k>1" is **not** supported by
  this blunt test; on Gemma-2-2b, k=1 is sufficient and best.
- **The ½-strength cap didn't bind at k=1** (half ≡ full): per-row coordinate norms
  sit below ½·pscale. The coherence knob (L5) only starts to matter for stronger/k>1
  interventions.

Caveat: mechanistically this is multi-layer directional ablation (Arditi et al.); the
finding is that the *actuator swap* unblocks Gemma where every rotation variant
failed, not a new de-refusal primitive. The forward-looking program (constrained,
dynamics-aware k>1 — the part PTS's machinery could finally earn its place on) is
written up as a successor proposal: **`../../CASA/CASA_PROPOSAL.md`**.
Transcript: `outputs/additive_subspace_gemma-2-2b-it.txt`.
