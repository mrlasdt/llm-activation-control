# MPC vs LQR vs PID for activation steering — results

**Question (user's instinct).** *"Predictive control should always beat normal control like PID
and LQR. Reproduce A-LQR and PID-AcT, then compare against CASA's MPC."*

**Headline finding.** On the refusal-cone steering plant, **the control law barely matters**:
at matched control effort, **P ≈ LQR ≈ MPC** on de-refusal (StrongREJECT ≈ 0.72, tied within
~1 SE at n=40). Predictive control gives **no de-refusal advantage** here — only a marginal
*coherence* edge. The dominant lever is the **effort bound**, not the law. This is
theory-consistent, not a null artifact: the cone plant is near-identity (R²=0.9994), so MPC's
lookahead has nothing to anticipate and its constraint barely reshapes the optimum, collapsing
MPC→LQR→P. Adding sophistication (PID integral, model-based feedforward) can even *hurt*.

This is the honest, falsifiable answer: the instinct is **not supported in this regime**, and we
can say exactly why.

---

## 1. The fair testbed (control law = the only variable)

Everything is held fixed except `.control`; see `casa_baselines.py`, `casa_control.py`,
`casa_experiment.py --ladder`.

- **Model**: `google/gemma-2-2b-it` (frozen, bf16).
- **Plant**: one `k=4` refusal **concept cone** (Geometry-of-Refusal RCO, retain-loss trained),
  fit once: `c_{l+1}=A_l(c_l+u_l)+b_l`, **held-out R²(1-step)=0.9994, 5-step=0.9955**. All laws
  read the identical `{A,b}`.
- **Reference**: the harmless-mean cone coordinate per layer (`ref=lmean_coord`).
- **Actuator + budget**: additive on the residual output across the discriminative band; the same
  per-token L2 budget `u_max = frac·pscale` (`pscale=180.85`), swept `frac ∈ {∞, 0.5, 0.25}`
  (∞ = unconstrained, to expose the constrained↔unconstrained regime).
- **Eval**: AdvBench held-out (n=40, the papers' `test_size=0.2, random_state=42` split), greedy,
  256 tokens. **Behaviour = StrongREJECT** (deterministic; 0=refuse/gibberish→1=full assistance).
  **Coherence = genNLL** (steered text scored by the clean model; high=gibberish).
  **Effort = realized mean‖u_l‖** (logged per condition, not just capped).
- **Matched weights**: LQR & MPC both use `q_pos=1.0, r_ctrl=0.05`.

The five laws (the P→PID→LQR→MPC ladder):

| Law | `u_l` over the band | role |
|---|---|---|
| **P** | `Kp·(ref−c)` | proportional (≈ bounded ablation toward the harmless ref) |
| **PID** | `Kp·e+Ki·Σe+Kd·Δe` | + integral (kills steady-state error) + derivative |
| **LQR** | `−K_l(c−ref)`, `K_l`=cross-term DARE on `A_l` | optimal multivariable feedback |
| **LQR+ff** | `−K_l(c−ref)+g_l`, `g_l`=inverse-dynamics feedforward | LQR + the model knowledge MPC plans over |
| **MPC** | receding-horizon QP, `‖u‖≤u_max` | predictive + hard constraint |

---

## 2. Result — the ladder (Gemma-2-2b-it, n=40, StrongREJECT)

baseline StrongREJECT = **0.017**. (`outputs/ladder_pareto.png`; data in
`../CASA/outputs/casa_cone_gemma-2-2b-it.json`.)

| Law | u_max | srScore ↑ | sr>0.5 | genNLL ↓ | effort | margin |
|---|---|---|---|---|---|---|
| P       | ∞     | 0.674 | 0.78 | 0.98 | 9.9  | −5.04 |
| P       | 90.4  | 0.712 | 0.85 | 0.93 | 9.9  | −5.51 |
| **P**   | 45.2  | **0.721** | 0.82 | 0.89 | 9.4 | −5.62 |
| PID     | ∞     | 0.636 | 0.80 | 1.03 | 11.0 | −4.85 |
| PID     | 45.2  | 0.703 | 0.78 | 0.92 | 10.2 | −5.66 |
| LQR     | ∞     | 0.676 | 0.80 | 0.96 | 9.7  | −5.08 |
| **LQR** | 45.2  | 0.718 | 0.85 | 0.89 | 9.2  | −5.65 |
| LQR+ff  | ∞     | 0.605 | 0.70 | 1.10 | 18.1 | −6.30 |
| LQR+ff  | 45.2  | 0.687 | 0.80 | 1.02 | 16.1 | −6.92 |
| MPC     | ∞     | 0.647 | 0.82 | 0.98 | 10.7 | −4.88 |
| MPC     | 90.4  | 0.707 | 0.82 | 0.95 | 10.6 | −5.30 |
| **MPC** | 45.2  | 0.720 | 0.85 | **0.87** | 9.4 | −5.40 |

Context (same run, subspace/actuator baselines): RDO k=1 bounded **0.718**, CONE k=4 blunt 0.679,
SVD k≥2 ≈ 0.00 (gibberish), DIM k=1 0.16.

### Findings

1. **At matched effort, MPC does not beat LQR or P.** Best-budget srScore: P 0.721, LQR 0.718,
   MPC 0.720 — a 0.003 spread, far inside ~1 SE (≈0.05 at n=40). Predictive control buys **no
   de-refusal**. A bounded **k=1 RDO** (0.718) ties the whole k=4 ladder — dimensionality +
   sophisticated control don't beat a bounded single direction either.
2. **The bound is the real lever.** ∞→45 lifts srScore ~0.64→0.72 for *every* law (a larger,
   consistent effect than any law-to-law gap). Tighter is better here. (Confirms the prior CASA
   finding: the working ingredient is the bound + reference, not lookahead.)
3. **MPC's only edge is coherence.** At matched effort it has the lowest genNLL (0.87 vs 0.89
   P/LQR, 0.92 PID) and top sr>0.5 (0.85) — the gentle receding-horizon distribution yields
   slightly more fluent control at equal de-refusal. Real but small.
4. **More sophistication can hurt.** PID is marginally worse (de-refusal isn't precise setpoint
   tracking, so the integral mostly adds effort, 10.2 vs 9.2–9.4). **LQR+ff is worst** (0.687,
   effort 16.1, genNLL 1.02): exactly tracking the harmless mean via inverse-dynamics feedforward
   **over-steers** — de-refusal needs *suppression*, not *reaching a setpoint*, so the extra model
   knowledge wastes effort and degrades fluency.
5. **First-token margin is unreliable (again).** LQR+ff has the *most negative* margin (−6.92,
   "most de-refused" by the proxy) yet the *lowest* srScore (0.687). Judge behaviour at length.

---

## 3. Why — the control-theory reading

- **Unconstrained ⇒ MPC = LQR (proved in code).** For an unconstrained quadratic-cost LTI problem
  the infinite-horizon LQR is optimal and finite-horizon MPC only approximates it. `casa_baselines._selftest_mpc_equals_lqr` demonstrates this exactly: unconstrained MPC with
  terminal cost `Qf = S∞` reproduces the DARE LQR control to 1e-5. So MPC can only beat LQR when
  the **constraint binds** or the **horizon/terminal cost** matters.
- **(Moderately) near-identity band ⇒ weak lookahead value.** Residual connections make each block
  *add* to the stream (`z_{l+1}=z_l+f_l(z_l)`), so `A_l=I+∂f_l/∂z` is identity-plus-a-perturbation.
  Measured on this cone (`diagnostics.py`): full-d RMS `‖(A_l−I)v‖/‖v‖ ≈ 0.42–0.52` in the late band
  (layer 25 ≈ 0.42), exact cone-plant `‖A_l−I‖₂` band-mean ≈ 0.86, and **5-step plant R²=0.9955**
  (multi-step rollout barely degrades ⇒ mild, predictable compounding). So most directions evolve
  near-identically and there is little for the receding horizon to anticipate — MPC's predictive
  content is weak. **Honest nuance:** this is *moderate* near-identity, not the extreme 0.34 PTS
  measured on Qwen's 2×2 plane — so weak-lookahead is a *contributing* factor, not the sole cause;
  the non-binding constraint and the suppression-not-tracking objective (below) matter too.
- **Constraint barely reshapes the optimum.** At the budgets that help, all laws sit near the same
  bounded operating point, so MPC's constraint-awareness ≈ post-hoc clamping the others.
- **Objective ≠ tracking.** De-refusal is *suppress the refusal component*, not *track the harmless
  mean precisely*. So the tools that excel at precise tracking — PID's integral, LQR's feedforward —
  add cost without benefit (or over-steer).

Net: in this near-linear, low-dimensional, suppression-not-tracking regime, **the control law is
not the bottleneck — the actuator + bound are** (the original CASA thesis). The instinct "MPC
always wins" holds only where its assumptions bite (binding constraints, nonlinearity, genuine
lookahead value), none of which this plant provides.

---

## 4. Implementation provenance + verification

- **Exact baseline configs** pulled from the authors' repos (`BASELINES_PROVENANCE.md`,
  `file:line` cited): A-LQR `time_varying_lqr_noB` Riccati, JVP Jacobians, LFS `β*=λ‖e‖`
  (q=10,r=10,qf=1; refusal q=0.1,r=10,qf=0.1); S-PID (PID on LFS error, 10-layer integral reset);
  PID-AcT (diff-in-means, cumsum integral, gains 1.0/0.3/0.01 jailbreak).
- **Native implementations** (`alqr_native.py`, `pid_native.py`) — synthetic self-tests pass:
  Riccati≡DARE, JVP Jacobian≡autograd≡finite-diff, rank-1 control identity `(β*−vᵀz)Kv≡K(αv)`,
  LFS, PID recurrence (integral≡cumsum, derivative≡Δ), ActAdd/DirAblate actuators.
- **Cone controllers** (`casa_baselines.py`) — synthetic self-tests pass incl. the MPC≡LQR
  cross-check and PID-kills-steady-state-error (1.114→0.000). A modeling subtlety the tests caught:
  the cone actuator gives effective `B=A_l` and the MPC costs the *actuated* state — `ConeLQR`
  matches that exactly (cross-term DARE + inverse-dynamics feedforward).

The head-to-head conclusion therefore stands on independently-verified controllers; it does **not**
depend on the (deferred) full-paper reproductions.

---

## 5. Proposal — improvements / research directions

1. **Stop optimizing the law; optimize the actuator + bound.** The bound (L5) and the
   refusal-specific subspace (L2) dominate; that's where returns are. A learned/scheduled `u_max(l)`
   per band layer likely beats any fixed-gain-vs-MPC distinction.
2. **Make MPC earn its keep — test where its assumptions bite.** Concrete, falsifiable predictions:
   MPC should beat LQR **(a)** on a *constraint-dominated* objective (very tight `u_max` with a
   hard coherence constraint), and **(b)** in *mid-band layers where local linearity breaks*
   (A-LQR's Fig-5 shows Jacobian subspace similarity ≈0.5 mid vs ≈0.8 early/late) via
   successive-linearization / nonlinear MPC. If MPC doesn't win there either, predictive control is
   simply the wrong tool for cone steering.
3. **Integral action for *steady-state* de-refusal, not first-token.** PID's integral is wasted on
   the suppression objective here, but could matter for *sustained* de-refusal across long
   generations (token-domain, not layer-domain) — an untested axis.
4. **"Best of both" terminal cost.** MPC with `Qf=S∞` = infinite-horizon optimality at finite
   horizon (verified in code); use it if/when a constrained regime makes MPC worthwhile.
5. **Right subspace × right operator.** The decisive levers were the *cone* (which subspace) and the
   *bound* (how hard); the *law* was a wash. Future work: combine the refusal cone with a learned
   bounded ablation, drop the control-law machinery.

---

## 6. Deferred / TODO (implemented + unit-tested, not yet run at scale)

These validate *external* fidelity (matching the papers' own numbers); the head-to-head above does
not depend on them. Sized for a single A10G; sequence by value.

- [x] **PID Fig-3 diagnostic** (`diagnostics.py`, Gemma-2-2b) — **partial/directional**: with gentle
      gains P leaves a larger residual plateau (norm. 2.68) than PI/PID (1.85/1.84) → the integral
      *does* reduce steady-state error, the right direction, but the open-loop model signal is noisy
      (overshoots negative mid-stream). The **clean** proof of the mechanism is the synthetic closed-loop
      `casa_baselines.ConePID` (P residual 1.114 → PID 0.000).
- [x] **Near-identity diagnostic** (replaced the flaky randomized-range Fig-5; `diagnostics.py`) —
      exact cone-plant `‖A_l−I‖₂` band-mean **0.86**, full-d JVP RMS `‖(A_l−I)v‖/‖v‖` **0.42–0.52**
      late band, 5-step plant R² **0.9955**. Directly quantifies the property behind MPC≈LQR (moderate
      near-identity ⇒ weak lookahead). NOTE: the paper's exact Fig-5 (full-SVD top-m energy-weighted
      `sim_m` ~0.8/~0.5) was **not** reproduced — a cheap randomized-range proxy came out near-random
      (≈0.02, the r/d baseline); the faithful version needs the full Jacobian SVD (still deferred).
- [ ] **Jailbreak reproduction** — `repro_jailbreak.py --model Qwen/Qwen2.5-3B-Instruct --alqr
      --judge` (both papers report Qwen-3B: A-LQR+ 0.96, A-LQR 0.86, S-PID 0.84, PID 0.76). Validate
      *ordering* (method > best baseline). NOTE: A-LQR full-d Jacobians are the expensive step;
      verify the de-refusal sign (lam sign / harmful−harmless) empirically.
- [ ] **Toxicity reproduction** — RTP on Gemma-2-2B/Llama-3-8B; needs `s-nlp/roberta_toxicity_classifier`
      + Mistral-7B PPL + MMLU. Targets A-LQR 0.18/0.12, PID-AcT 0.51/0.72. Heaviest (sampling + judges).
- [ ] **Truthfulness reproduction** — TruthfulQA-gen; needs the two llama2-7B judges. Target A-LQR T*I 67.81/63.63.
- [ ] **S-PID** as an explicit cone-ladder rung (PID-on-LFS-error variant) for completeness.
- [ ] **Larger n / second model** for the head-to-head to tighten the ~1-SE ties (cheap via `--load-subspaces`).

---

## 7. Reproduce

```bash
cd research_proposals/CASA
../../.venv/bin/python casa_baselines.py        # control-law self-tests (incl. MPC≡LQR)
# the head-to-head (≈30 min on A10G: trains the k=4 cone, runs P/PID/LQR/LQR+ff/MPC, judges):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python casa_experiment.py --full --ladder --max-new-tokens 256
cd ../BASELINES
../../.venv/bin/python plot_ladder.py           # outputs/ladder_pareto.png
../../.venv/bin/python alqr_native.py && ../../.venv/bin/python pid_native.py   # native self-tests
```
