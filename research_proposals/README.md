# Research Proposals — Control-Theoretic Angular Steering

Control-theoretic recastings of **Angular Steering** (Vu & Nguyen, NeurIPS 2025), which
steers LLM behavior by rotating the in-plane component of activations inside a 2D plane
`{b1, b2}`. Each proposal has its own directory: the proposal markdown + its prototype
code + outputs.

## Layout & how to run

- **Shared library** lives in `../pytorch_pure/` (`utils.py`, `phase_portrait.py`,
  `observables.py`) — imported by every proposal *and* by the base pure-PyTorch pipeline.
- Proposal scripts add it to `sys.path` via a small shim at the top of each `*.py`
  (`sys.path.insert(0, <repo>/pytorch_pure)`); notebooks do the same in their first cell.
- **Run a proposal's scripts/notebooks from inside its own directory.**

## Proposals

| Dir | Proposal | Control paradigm | Status |
|-----|----------|------------------|--------|
| `SO2/` | Angular Steering as Control on SO(2) | energy/Lyapunov **state-feedback over depth** (pendulum: φ, ω, separatrix) | active — phase-portrait experiments implemented (`phase_portrait.ipynb`, `steering_validation.ipynb`) |
| `PTS/` | Predictive Trajectory Steering | **predictive MPC** tracking a 2D reference trajectory | **verified negative (2026-06-12)** — implemented & self-tested (`pts_*.py`, dependency-free QP/MPC), but end-to-end **PTS ≈ naive fixed-angle** (no behavioural gain; Qwen+Gemma). Durable spinoff: the 2×2 plant. See `PTS/PTS_README.md` **Verdict** + `PTS/verify/` |
| `OAS/` | Observer-Based, Soft-Landing Angular Steering | depth-domain **LQG** (Kalman observer + terminal-target LQR) | active — implemented & self-tested (`oas_*.py`: `oas_lqr`, `oas_observer`, `oas_controller`, `oas_offline`, `oas_prototype`, `oas_plot`); offline validated, prototype smoke-tested. See `OAS/README.md` + `OAS/OAS_SPEC.md` |
| `CASA/` | Constrained Additive Subspace Ablation | **norm-CHANGING** bounded ablation of a refusal **concept cone** over a band (+ k×k plant/MPC) | active — **trained to convergence + StrongREJECT-judged (2026-06-12)**: the additive salvage of PTS. (L1) additive band-ablation de-refuses Gemma where every rotation variant fails. (L2) the blunt SVD subspace is gibberish at k≥2 (StrongREJECT≈0); a retain-loss-trained **concept cone** (Geometry-of-Refusal, ICML 2025) at k=4 is a **strong, coherent jailbreak** (StrongREJECT **0.68**, neutral ΔNLL **−0.72**) — L2 rescued; bare k>1 ties best k=1. (MPC) k-dim cone plant R²≈0.999; bounded-u **additive** MPC gives the **best operating point of all** (StrongREJECT **0.76** at −0.41 tax) — the PTS apparatus, inert on rotation, load-bearing on the additive actuator. Modules `casa_{actuator,cone,control,judge,experiment,plot}.py`. See `CASA/CASA_PROPOSAL.md` + `CASA/CASA_RESULTS.md` |
| `CLAS/` | Closed-Loop Angular Steering | per-token **output-feedback** thermostat | **DROPPED 2026-06-09** — see `CLAS/README.md` |

## Shared empirical facts (Qwen2.5-3B-Instruct)

- Auto-selected steer layer ≈ **27 of 36**. The **refusal** direction has strong authority
  (`G(θ)` range ~18, flips behavior) **at the residual-stream hook**; subtle/sustained attributes
  (e.g. sentiment) have negligible authority — this is what dropped CLAS.
- **Two non-interchangeable hook points (verified 2026-06-11, `PTS/verify/`):** (a) *canonical
  Angular Steering* hooks `model.layers.{L}.input_layernorm` output (single layer) and is **nearly
  inert** — angle sweep moves the refusal margin only ±0.6 on Qwen, ±0.13 on Gemma, generations
  still refuse (Gemma: inert from *all* 25 layers); (b) the *residual-stream reset* (`model.layers.{k}`
  output — what CLAS/PTS/OAS use) is what actually flips behavior. The "`G(θ)` range ~18" above is (b).
- Per-layer in-plane dynamics are well-modeled by a **2×2 affine** map `c_{k+1}=A_k c_k+b_k`
  (held-out R²≈0.999), validated by PTS — the plant OAS's LQG reuses.
- The angular actuator is a **norm-preserving absolute-angle reset** (sets the in-plane angle,
  drops magnitude) — central caveat across PTS/OAS.
- **The actuator, not the planner, is the Gemma bottleneck (verified 2026-06-12, `CASA/`).**
  Every rotation variant (canonical AS, residual band, PTS adaptive angle, OAS soft-landing)
  caps out on Gemma-2-2b — best margin +3.67, still hedging. A **norm-CHANGING additive**
  actuator — bounded directional ablation of the refusal axis (k=1) across the discriminative
  band — flips it to **−2.99 with coherent compliance** at +0.10 NLL coherence tax. So Gemma
  *is* steerable; AS just couldn't reach it because rotation collapses the plan to one DoF.
  (k=8 blunt subspace ablation backfires — destroys coherence; the subspace isn't
  refusal-specific.) This is the `CASA/` direction, and the open lever PTS's verdict named.
