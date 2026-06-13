# CALM — Coherence-Aware Lookahead MPC

Making MPC earn its keep on the steering-**strength ↔ fluency/coherence** trade-off — the
trade-off the BASELINES head-to-head showed the control *law* cannot exploit (P ≈ LQR ≈ MPC;
the bound is the only lever) because layer-domain refusal **saturates** (a hump, not a frontier).

CALM puts coherence **inside** the controller. Existing fixes for the off-manifold/coherence
problem are heuristic scalar intensity dials ([IDS](https://arxiv.org/abs/2510.13285),
[Dynamic Activation Composition](https://arxiv.org/abs/2406.17563)); CALM instead adds an
**on-manifold density penalty** to the MPC cost so its *constrained* optimization can trade
de-refusal against staying on the harmless activation manifold:

```
J = Σ_l  ‖s_l−ref_l‖²_Q  +  κ·(s_l−μ_l)ᵀΣ_l⁻¹(s_l−μ_l)  +  ‖u_l‖²_R    s.t. ‖u_l‖≤u_max
            suppress refusal     stay in the harmless density           effort       (s_l=c_l+u_l)
```

The density term is **quadratic** → it folds into CASA/PTS's condensed QP, solved by the SAME
FISTA. κ=0 ≡ `casa_control.ConeMPC`. κ traces the strength↔coherence frontier. It encodes what
P/LQR/clip cannot: local anisotropy — spend control freely along robust (high-variance) cone
directions, gently along brittle (low-variance) ones.

See `../FINDINGS_SYNTHESIS.md` for why this is the next direction, and `../BASELINES/BASELINES_RESULTS.md`
for the head-to-head that motivated it.

## Status

- ✅ **`calm_mpc.py`** — `CoherenceMPC` + `fit_cone_density` + `mahalanobis_trajectory`. Self-tests
  pass: κ=0 ≡ ConeMPC; augmented cost matches brute force to 1e-8; κ pulls toward the manifold;
  density fit recovers anisotropy + normalization.
- ✅ **`--frontier` wiring** in `../CASA/casa_experiment.py` — sweeps {P, MPC, cMPC(κ)} over a wide
  strength grid on the SAME cone plant; logs realized effort + the cone-space Mahalanobis coherence
  surrogate; reuses `--load-subspaces` (no retraining).
- ✅ **`plot_frontier.py`** — strength↔coherence Pareto + matched-effort view + surrogate↔genNLL
  validation scatter.
- ✅ **Phase 1+2 run done (2026-06-13)** — **honest null + sharp diagnosis** (`CALM_RESULTS.md`,
  `outputs/calm_frontier_gemma-2-2b-it.png`). cMPC does **not** dominate the frontier: P, MPC, and
  cMPC@κ∈{0.5,2,8} all lie on the SAME strength↔coherence Pareto (srScore spread ≤0.012 ≪ 1 SE). Why:
  the cone-space density cMPC optimizes is **not a faithful coherence proxy** — surrogate↔genNLL
  **r=0.20**; genNLL is set by the *bound*, not cone-space off-manifold distance. Coherence lives in
  the ambient residual / next-token distribution, not the k=4 cone. The controller works in cone space
  (surrogate ↓ monotonically with κ: 200→172→139→110) — it's optimizing the wrong space.
- ✅ **Phase 3 done (2026-06-13)** — token-domain sustained control (`calm_token.py`): a verified
  batched dual-cache Gemma-2 decode loop (matches `model.generate` 16/16, KL=0 under identity) with a
  **KL-thermostat** that modulates per-token additive-ablation strength on the **ambient KL-to-unsteered**
  signal. **Third null:** dynamic ties static (best Δsr +0.018, at *higher* genNLL — no dominance), even
  though the feedback genuinely modulates (α_std 18–63). **But the ambient KL IS a faithful coherence
  proxy: r(meanKL,genNLL)=0.976** vs the cone-space surrogate's 0.20 — validating the Phase-1+2
  diagnosis. Per the pre-registered rule (no dynamic win), **token-MPC was gated out, not built.**
- 🧱 **Bottom line:** across PTS (lookahead) → CALM P1+2 (cone-coherence cost) → CALM P3 (token-domain
  KL feedback), **no control sophistication beats a well-chosen fixed magnitude bound for cone-based
  de-refusal** — the lever is the actuator + the bound, not the controller. See `CALM_RESULTS.md §5`.

## Modules

| File | What |
|------|------|
| `calm_mpc.py` | **Phase 2.** `CoherenceMPC` (ConeMPC + anisotropic on-manifold density cost), `fit_cone_density`, `mahalanobis_trajectory`. Reuses `casa_control.{build_condensed_qp,solve_qp}` + `pts_mpc.assemble_q` unchanged. Self-tested. |
| `plot_frontier.py` | **Phase 2.** Renders `outputs/calm_frontier_<model>.png` from `../CASA/outputs/casa_frontier_<model>.json`. |
| `calm_token.py` | **Phase 3.** Verified batched dual-cache (steered+clean) Gemma-2 decode loop + per-token KL-to-unsteered + live-mutable actuator strength + controller ladder {StaticHold, TokenThermostat(bang/prop)}. `--selfcheck` gates (vs `model.generate`), `--run` experiment. (TokenMPC gated out.) |
| `plot_token.py` | **Phase 3.** Renders `outputs/calm_token_<model>.png` (dynamic-vs-static frontier + α_t modulation + α_std). |

## Run

```bash
# control-math self-tests (no model):
../../.venv/bin/python calm_mpc.py
# the frontier (reuses the saved k=4 cone; ~30–45 min A10G):
cd ../CASA && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python casa_experiment.py --full --frontier --load-subspaces \
  --max-new-tokens 256 --frontier-fracs 1e9 2.0 1.0 0.5 0.25 --kappas 0.5 2.0 8.0
cd ../CALM && ../../.venv/bin/python plot_frontier.py
```

## ⚠️ De-refusal (jailbreak) research

Harmful generations are the *measured outcome* we quantify (StrongREJECT), not a product.
Authorized safety/steering research on an open-weights model; genNLL guards against scoring gibberish.
