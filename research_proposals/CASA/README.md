# CASA — Constrained Additive Subspace Ablation

The **additive salvage of PTS**: replace Angular Steering's norm-preserving rotation
with bounded *additive* directional ablation across a layer band. On Gemma-2-2b this
flips the first-token refusal margin **+10.83 → −2.99** with coherent compliance on
all 6 harmful prompts at a negligible coherence tax (+0.10 neutral-corpus NLL) —
where the rotation band only reaches +3.67 (still hedging/refusing). **It solves the
Angular-Steering limit on Gemma.**

See `CASA_PROPOSAL.md` for the full direction (four levers, kill-fast experiment
plan, novelty boundary).

## Status (2026-06-12)

- **L1 (additive actuator): validated.** k=1 ablation across the discriminative band
  de-refuses Gemma cleanly and coherently. *The actuator was the bottleneck.*
- **L2 (k-dim subspace): negative as built.** k=8 (top-SVD of per-layer mean-diffs)
  destroys coherence (+2.33 NLL, gibberish) without de-refusing — the subspace isn't
  refusal-specific. Open: a clean refusal subspace.
- **L5 (bounded-‖u‖ coherence knob) + MPC distribution: untested at the binding
  regime.** The central open question — do they buy coherent k>1 beyond blunt k=1?

## Prototype

The working experiment is `../PTS/verify/additive_subspace_steer.py` (reuses the
shared `pytorch_pure/` lib + PTS's plane/hook machinery). Transcript:
`../PTS/verify/outputs/additive_subspace_gemma-2-2b-it.txt`.

```bash
cd research_proposals/PTS/verify
../../../.venv/bin/python additive_subspace_steer.py --model google/gemma-2-2b-it
```

Graduating it into first-class `casa_*.py` modules (plant generalized to k dims,
bounded-u MPC over the band) is Experiment 3 of the proposal.

## ⚠️ This is de-refusal (jailbreak) research

The harmful generations are the *measured experimental outcome* (the de-refusal we
are quantifying), not a product. Authorized safety/steering research; treat outputs
as data.
