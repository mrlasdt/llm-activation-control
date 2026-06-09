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
| `PTS/` | Predictive Trajectory Steering | **predictive MPC** tracking a 2D reference trajectory | active — implemented & self-tested (`pts_*.py`, dependency-free QP/MPC) |
| `OAS/` | Observer-Based, Soft-Landing Angular Steering | depth-domain **LQG** (Kalman observer + terminal-target LQR) | active — proposal only; code TBD |
| `CLAS/` | Closed-Loop Angular Steering | per-token **output-feedback** thermostat | **DROPPED 2026-06-09** — see `CLAS/README.md` |

## Shared empirical facts (Qwen2.5-3B-Instruct)

- Auto-selected steer layer ≈ **27 of 36**. The **refusal** direction has strong angular-steering
  authority (`G(θ)` range ~18, flips behavior); subtle/sustained attributes (e.g. sentiment)
  have negligible authority — this is what dropped CLAS.
- Per-layer in-plane dynamics are well-modeled by a **2×2 affine** map `c_{k+1}=A_k c_k+b_k`
  (held-out R²≈0.999), validated by PTS — the plant OAS's LQG reuses.
- The angular actuator is a **norm-preserving absolute-angle reset** (sets the in-plane angle,
  drops magnitude) — central caveat across PTS/OAS.
