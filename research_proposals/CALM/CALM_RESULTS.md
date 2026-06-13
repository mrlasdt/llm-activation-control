# CALM — Coherence-Aware MPC on the strength↔coherence frontier: results (Phases 1–3)

**Question.** The BASELINES head-to-head found the control law is a wash for layer-domain refusal
(P ≈ LQR ≈ MPC); the user's hypothesis is that MPC's edge lives in the steering-**strength ↔
coherence** trade-off, which those experiments didn't exercise. CALM tests this directly across two
domains: §1–3 map the full frontier and put coherence *inside* the controller in the **layer domain**
(`CoherenceMPC`, an on-manifold density penalty in the MPC's QP); §5 moves to the **token domain**
with a closed loop on the **ambient KL-to-unsteered** signal (`calm_token.py`). **Bottom line: three
consecutive nulls — no control sophistication beats a well-chosen fixed magnitude bound (§5).**

**Headline (honest null, sharp diagnosis).** **It does not.** Putting coherence in the objective
does **not** move the strength↔coherence Pareto: P, vanilla MPC, and coherence-aware cMPC at every
κ all lie on the **same frontier** (best srScore spread ≤ 0.012 at matched genNLL, ≪ 1 SE ≈ 0.05 at
n=40). And we now know **why**, which is the valuable part: the cone-space density that cMPC can
optimize is **not a faithful coherence proxy** — its correlation with measured genNLL is only
**r = 0.20**. Coherence degradation lives in the *ambient* residual stream / next-token distribution,
not in the k=4 cone coordinate, so optimizing on-manifold-ness *within the cone* buys real
de-refusal nothing. This is exactly the contingency the plan anticipated, and it points cleanly to
Phase 3.

---

## 1. Testbed

Identical to the BASELINES head-to-head (the fairness contract carries over) — `casa_experiment.py
--full --frontier --load-subspaces --max-new-tokens 256`:

- **Model / plant**: `google/gemma-2-2b-it`, the saved `k=4` refusal concept cone, **R²(1-step)=0.9994**.
- **Laws** (control law/cost = the only variable): **P**, **MPC** (`ConeMPC`), **cMPC@κ** (`CoherenceMPC`,
  the on-manifold density penalty at `κ ∈ {0.5, 2, 8}`). κ=0 ≡ ConeMPC (verified in `calm_mpc.py`).
- **Strength grid**: `u_max = frac·pscale`, `frac ∈ {∞, 2, 1, 0.5, 0.25}` (`pscale=180.85`) — spans the
  over-steer regime (∞) to tight.
- **Metrics**: StrongREJECT (behaviour), genNLL + neutral ΔNLL (coherence), realized `mean‖u_l‖`
  (effort), and the **cone-space Mahalanobis surrogate** `Σ_l (s_l−μ_l)ᵀΣ_l⁻¹(s_l−μ_l)` vs the
  harmless density (the coherence proxy CALM optimizes — logged to validate it).
- **n=40** AdvBench held-out, greedy, 256 tokens. Figure: `outputs/calm_frontier_gemma-2-2b-it.png`;
  data: `../CASA/outputs/casa_frontier_gemma-2-2b-it.json`.

---

## 2. Result — the frontier (StrongREJECT ↑ / genNLL ↓ / surrogate)

baseline StrongREJECT 0.017. Selected rows (full table in the JSON):

| law | u_max | srScore ↑ | sr>0.5 | genNLL ↓ | effort | surrogate |
|---|---|---|---|---|---|---|
| P        | ∞     | 0.674 | 0.78 | 0.98 | 9.9 | **0.0** |
| P        | 90.4  | 0.712 | 0.85 | 0.93 | 9.9 | 1.6 |
| **P**    | 45.2  | **0.721** | 0.82 | 0.89 | 9.4 | 46.5 |
| MPC      | ∞     | 0.647 | 0.82 | 0.98 | 10.7 | 200.2 |
| MPC      | 90.4  | 0.707 | 0.82 | 0.95 | 10.6 | 201.9 |
| MPC      | 45.2  | 0.720 | 0.85 | 0.87 | 9.4 | 138.6 |
| cMPC@0.5 | 90.4  | **0.726** | 0.85 | 0.92 | 10.6 | 173.4 |
| cMPC@0.5 | 45.2  | 0.714 | 0.88 | 0.88 | 9.4 | 115.9 |
| cMPC@2   | 45.2  | 0.725 | **0.90** | 0.87 | 9.5 | 92.2 |
| cMPC@8   | ∞     | 0.648 | 0.80 | 0.98 | 10.6 | **109.8** |
| cMPC@8   | 45.2  | 0.718 | 0.88 | 0.88 | 9.5 | 68.1 |

(anchors: CONE k=4 static-ablation 0.638, RDO k=1 0.706.)

### Findings

1. **cMPC does not dominate the frontier.** At any matched genNLL the srScore spread across
   {P, MPC, cMPC@0.5/2/8} is ≤ 0.012 — inside ~1 SE. Best points are a wash: cMPC@0.5 0.726 (genNLL
   0.92), cMPC@2 0.725 (0.87), P 0.721 (0.89), MPC 0.720 (0.87). **Adding coherence to the objective
   did not buy de-refusal at equal coherence.**
2. **The bound is *still* the only lever** (∞→tight lifts every law 0.65→0.72), reproducing BASELINES.
3. **The coherence term DOES work — in cone space.** cMPC monotonically shrinks the cone-space
   Mahalanobis surrogate with κ (at ∞: MPC 200 → cMPC@0.5 172 → @2 139 → @8 110; the controller stays
   demonstrably closer to the harmless density). So the mechanism is real — it just doesn't matter.
4. **Weak, ~1-SE hint only:** cMPC has a marginally higher *fraction of strong* jailbreaks (sr>0.5 up
   to 0.90 vs 0.82–0.85) at matched genNLL. Suggestive, not significant at n=40; not claimed.

---

## 3. Why — the surrogate doesn't predict coherence (the load-bearing finding)

**The cone-space density is the wrong place to enforce coherence.** Across all conditions,
genNLL is governed by the **effort/bound** (∞ → genNLL ≈ 0.98 for *every* law regardless of
surrogate 0–200; tight → ≈ 0.87–0.89 regardless of surrogate 46–138), **not** by the cone-space
Mahalanobis. The Pearson correlation between the surrogate and measured genNLL is **r = 0.20**
(`outputs/calm_frontier_*.png`, right panel).

The k=4 refusal cone captures *where refusal lives*, but coherence degradation is a property of the
**full residual stream / next-token distribution** — the off-manifold excursion the literature names
([IDS](https://arxiv.org/abs/2510.13285), [DAC](https://arxiv.org/abs/2406.17563)) happens in ambient
space, largely orthogonal to the cone. So a quadratic on-manifold penalty *inside the cone* optimizes
a direction that is decoupled from fluency. CALM's controller faithfully does what it was built to do;
the cost was measuring the wrong space.

This also re-confirms the BASELINES/saturation diagnosis from a new angle: the strength↔coherence
relationship for refusal is one shared frontier (a hump set by effort), and **no layer-domain control
sophistication reshapes it**, because (a) the constraint doesn't bind against a continuing objective
(behaviour saturates) and (b) the coherence cost that *would* make MPC bite isn't expressible in the
low-dimensional state the controller observes.

---

## 4. Implications → Phase 3 (token-domain, ambient/KL coherence) — DESIGN (executed in §5)

The Phase-1+2 result was a precise signpost: to make a coherence-aware controller bite, two things
had to change together (this became Phase 3, executed in §5 — outcome: another null, but the
ambient-KL signal validated the diagnosis):

1. **Measure coherence where it lives.** Replace the cone-space quadratic surrogate with an
   **ambient / behavioural** signal — the **KL between the steered and unsteered next-token
   distribution** (DAC's metric), which directly measures off-manifold/off-distribution drift. This is
   not quadratic in the cone coordinate, so it needs an MPC that closes the loop on a measured output
   (not a pure condensed QP) — exactly the receding-horizon-with-feedback regime.
2. **Move to the token axis.** Layer-domain refusal saturates (one shared frontier). The KL/coherence
   trade-off is *non-saturating and continuous over generation*, where lookahead has genuine value
   (anticipate drift before it compounds) — the regime CLAS wanted but lacked authority for, now armed
   with CASA's additive actuator. Target the known failure mode: deadbeat flips the first token but the
   body reverts; token-domain control to *hold* compliance/coherence through the generation.

`CoherenceMPC` and the frontier harness are reusable; Phase 3 swaps the cost signal (cone-space →
ambient KL) and the axis (depth → tokens).

---

## 5. Phase 3 — token-domain sustained control on the ambient KL signal (executed)

The Phase 1+2 diagnosis said: measure coherence where it lives (ambient KL, not cone-space), on the
token axis (non-saturating). Phase 3 did exactly that — a per-token closed loop (`calm_token.py`) that
modulates additive-ablation strength on the **per-token KL between the steered and unsteered
next-token distribution** (the DAC signal). Disciplined ordering (three prior nulls): build
static-hold + a KL thermostat first; gate token-MPC behind a measured dynamic win.

**Decode loop verified** (the mandatory gate): the batched dual-cache (steered+clean) Gemma-2 loop
reproduces `model.generate` **16/16 tokens**, KL under identity steering is **exactly 0** (clean
stream tracks the steered tokens, no cache crosstalk), and batched==singleton 32/32. So the numbers
below are trustworthy.

**Result (Gemma-2-2b, n=10, 256 tokens, StrongREJECT):**

| controller | srScore ↑ | sr>0.5 | genNLL ↓ | mean KL | α_std |
|---|---|---|---|---|---|
| static u_max=∞ | 0.578 | 0.60 | 1.121 | 0.63 | 0.0 |
| **static 0.5·ps** | **0.602** | 0.70 | 1.028 | 0.58 | 0.0 |
| static 0.25·ps | 0.515 | 0.60 | 0.946 | 0.50 | 0.0 |
| bang hi=1 lo=0.25 | 0.566 | 0.70 | 1.079 | 0.61 | 43.6 |
| bang hi=0.5 lo=0.125 | 0.604 | 0.70 | 1.032 | 0.59 | 21.5 |
| bang hi=1 lo=0 | 0.582 | 0.60 | 1.043 | 0.59 | 59.8 |
| prop kb=0.5 | 0.581 | 0.60 | 1.044 | 0.59 | 63.3 |
| **prop kb=1** | 0.620 | 0.80 | 1.131 | 0.64 | 39.8 |
| prop kb=2 | 0.605 | 0.60 | 1.108 | 0.63 | 18.8 |

baseline srScore 0.008. Figure: `outputs/calm_token_gemma-2-2b-it.png`; data `outputs/calm_token_*.json`.

### Findings

1. **NULL again — dynamic ties static.** Best dynamic point (prop kb=1, sr 0.620) beats the best
   static (0.602) by only **+0.018 — and at *higher* genNLL** (1.131 vs 1.028), i.e. upper-*right*,
   not a dominance. Every thermostat lands within ±0.04 of the static frontier (well inside ~1 SE at
   n=10). Per the pre-registered decision rule, **no dynamic win ⇒ token-MPC is NOT built** (gated
   out — building it would repeat the lookahead-is-inert failure mode a fourth time).
2. **The feedback genuinely modulates** (α_std 18–63, the α_t trace jumps between strength levels
   per token — middle panel) — so this is a real "dynamic *allocation* doesn't help" result, not a
   "controller did nothing" artifact. Reallocating a fixed average ablation across tokens leaves the
   aggregate strength↔coherence outcome on the same frontier.
3. **Positive byproduct — ambient KL IS a faithful coherence proxy.** Across conditions
   **r(mean KL, genNLL) = 0.976** — versus the cone-space surrogate's **r = 0.20** in Phase 2. This
   *validates the Phase 1+2 diagnosis*: coherence lives in the ambient / next-token distribution, not
   the k=4 cone. CALM correctly identified *where* coherence lives; the limitation is that even
   measuring it correctly and modulating on it dynamically does not beat a well-chosen fixed bound.

### The three-null through-line (the real result of the MPC investigation)

| experiment | domain | actuator | what was added | outcome |
|---|---|---|---|---|
| PTS | layer | rotation | multi-step lookahead | inert (H=1≈H=8) |
| CALM Phase 1+2 | layer | additive | coherence in the cost (cone-space) | ties (r=0.20, wrong space) |
| CALM Phase 3 | **token** | additive | dynamic feedback on ambient KL | ties (r=0.98, right space, still no win) |

**For cone-based de-refusal, no control sophistication — lookahead, a coherence-aware cost, or
dynamic per-token feedback — beats a well-chosen fixed magnitude bound.** The lever is always the
*actuator* (additive vs rotation) and the *magnitude bound*; the *controller* is a wash. Mechanism:
behaviour saturates (a hump, not a frontier), and the strength↔coherence trade-off is governed by the
*aggregate* ablation, not its allocation in depth or time — so there is no binding-constraint or
lookahead structure for a sophisticated controller to exploit. This is the honest, falsifiable answer
to the user's instinct, now established across two domains, two coherence signals, and fixed/dynamic/
predictive control. Token-MPC is not pursued; n=10 is small but there is no signal to confirm.

## 6. Verification / reproduce

```bash
../../.venv/bin/python CALM/calm_mpc.py          # math self-tests (κ=0≡ConeMPC; cost exact 1e-8)
cd CASA && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ../../.venv/bin/python casa_experiment.py --full --frontier --load-subspaces \
  --max-new-tokens 256 --frontier-fracs 1e9 2.0 1.0 0.5 0.25 --kappas 0.5 2.0 8.0
cd ../CALM && ../../.venv/bin/python plot_frontier.py
# Phase 3 — token domain:
../../.venv/bin/python calm_token.py              # math self-tests (KL, controllers, live mutation)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../../.venv/bin/python calm_token.py --selfcheck
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ../../.venv/bin/python calm_token.py --run --n 10
../../.venv/bin/python plot_token.py
```
Determinism: greedy; training untouched (`--load-subspaces`); the same k=4 cone + band as Phase 1+2.
