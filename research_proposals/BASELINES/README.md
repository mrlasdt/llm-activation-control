# BASELINES — A-LQR & PID-AcT, and is MPC actually better?

Faithful reimplementations of the two control-theoretic activation-steering baselines, and a
**fair P → PID → LQR → MPC head-to-head** on CASA's refusal-cone plant.

- **A-LQR** — *Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal
  Control* (Skifstad/Yang/Chou).
- **PID-AcT / PID Steering** — *Activation Steering with a Feedback Controller* (ICLR 2026).

## Headline

The user's instinct ("predictive control always beats PID/LQR") is **not supported in this regime**.
At matched control effort on the `k=4` refusal cone (plant R²=0.9994), Gemma-2-2b-it, AdvBench,
StrongREJECT: **P ≈ LQR ≈ MPC ≈ 0.72**, tied within ~1 SE. The dominant lever is the **effort
bound**, not the law; MPC's only edge is marginally better coherence; PID's integral and LQR's
feedforward add cost without benefit (LQR+ff over-steers). Theory-consistent: near-identity dynamics
⇒ lookahead inert; unconstrained ⇒ MPC=LQR (proved in `casa_baselines._selftest_mpc_equals_lqr`).
See **`BASELINES_RESULTS.md`** for the full table, analysis, and proposal.

## Modules

| File | What |
|------|------|
| `BASELINES_PROVENANCE.md` | Exact configs pulled from the authors' repos (`file:line` cited) |
| `BASELINES_RESULTS.md` | **Authoritative**: head-to-head result, control-theory reading, proposal, **deferred TODO** |
| `alqr_native.py` | Native A-LQR (full-d JVP Jacobians + B=I Riccati + LFS + rank-1 control). Self-tests pass. |
| `pid_native.py` | Native PID Steering (diff-in-means, cumsum integral, ActAdd/DirAblate) + Fig-3 diagnostic. Self-tests pass. |
| `repro_jailbreak.py` | Jailbreak reproduction driver (written; run is deferred — see TODO) |
| `plot_ladder.py` | Head-to-head Pareto figure (`outputs/ladder_pareto.png`) |

The cone-restricted controllers used for the head-to-head live in `../CASA/casa_baselines.py`
(`ConeP/ConePID/ConeLQR/RecordingController`); the ladder is driven by
`../CASA/casa_experiment.py --ladder`.

## Run

```bash
# control-law self-tests (no model, incl. MPC≡LQR cross-check):
../../.venv/bin/python ../CASA/casa_baselines.py
../../.venv/bin/python alqr_native.py && ../../.venv/bin/python pid_native.py
# the head-to-head (≈30 min, A10G):
cd ../CASA && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python casa_experiment.py --full --ladder --max-new-tokens 256
cd ../BASELINES && ../../.venv/bin/python plot_ladder.py
```

## Status

✅ configs extracted · ✅ native impls + cone controllers built & unit-tested · ✅ head-to-head run +
figure · ⏭ paper-number reproductions (jailbreak/toxicity/truthfulness) + Fig-3/Fig-5 diagnostics
**deferred** (code ready; see `BASELINES_RESULTS.md §6`).

## ⚠️ De-refusal (jailbreak) research

Harmful generations are the *measured outcome* (the de-refusal we quantify), not a product.
Authorized safety/steering research; genNLL exists so gibberish is never mistaken for a jailbreak.
