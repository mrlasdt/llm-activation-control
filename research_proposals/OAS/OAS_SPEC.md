# OAS implementation spec (the contract every module is built against)

This is the *engineering* contract for implementing `research_proposal_OAS.md`
(Observer-Based, Soft-Landing Angular Steering = depth-domain LQG). It fixes the
math, the actuator convention, the module interfaces, and the per-module
self-tests so the pieces compose. Mirrors the PTS code pattern
(`../PTS/pts_{dynamics,mpc,controller,offline,prototype,plot}.py`).

Everything is pure-numpy on the 2D plane coordinates except the two drivers, which
touch the model. Shared library is in `../../pytorch_pure/` and reached via the
sys.path shim already used by PTS:

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "pytorch_pure"))
```

OAS is **self-contained** (own hook copy), like CLAS and PTS — it imports only
from the shared lib (`utils`, `phase_portrait`, `observables`, `pts_dynamics`) and
its own `oas_*` modules, never from `../PTS` or `../CLAS`. `pts_dynamics` is treated
as shared (it is pure-numpy plant-fitting; OAS reuses `fit_layer_dynamics`,
`validate_dynamics`). If a reviewer objects to importing across proposal dirs,
copy the two needed functions; default is to import `pts_dynamics` via the shim
by also adding `../PTS` to sys.path. **Decision: add both `pytorch_pure` and the
sibling `PTS` dir to sys.path** so `from pts_dynamics import ...` works; document it.

---

## 0. Notation and the actuator convention (MUST match PTS exactly)

Plane coordinate of the layer-`k` residual-stream output: `c_k ∈ R^2`
(`c_k = (h_k·b1, h_k·b2)`). Fitted plant (PTS, held-out R²≈0.999):

```
c_{k+1} = A_k c_k + b_k          A_k ∈ R^{2x2}, b_k ∈ R^2     (autonomous)
```

The **actuator** is the norm-preserving SO(2) reset on layer `k`'s output: it
keeps `‖c_k‖` and sets the *angle*. We model control additively and then realize
only the angle (the "norm-preserving tax", Exp 5):

- **state** `x_k` = the *natural incoming* coordinate at layer `k` (what the plant
  delivers, before actuation).
- **decision** `s_k` = the *actuated* coordinate (what flows downstream).
  `s_k = x_k + u_k`, control `u_k = s_k − x_k`.
- **dynamics** `x_{k+1} = A_k s_k + b_k = A_k(x_k + u_k) + b_k`.
- **physical realization**: the hook sets `θ_k = atan2(s_k[1], s_k[0])` and keeps
  `‖x_k‖` (norm-preserving). The dropped magnitude `‖s_k‖ vs ‖x_k‖` is the Exp-5 tax.

This is *identical* to PTS's `rollout`/`simulate_closed_loop` convention
(`s = c0 + u_0`, `s_{k} = A_{k-1} s_{k-1} + b_{k-1} + u_k`). Reuse
`pts_dynamics.rollout` semantics for any open-loop simulation.

Deadbeat reset (the existing Angular Steering, and the Exp-3 baseline) = the
degenerate corner: a single actuated layer, `R→0`, `Q→∞`, no observer; `s_k` is
slammed to the target angle ignoring `x_k`.

---

## 1. `oas_lqr.py` — finite-horizon LQ tracking with a terminal cost (Idea 1)

### 1.1 Problem
Actuated band layers `k0..kT` (`kT` = terminal layer, the late layer the angle
must land at). Per-layer references `r_k ∈ R^2`; the **terminal target** is
`r_{kT} = c_target` (a point at the target angle with the natural magnitude). Cost:

```
J = Σ_{k=k0}^{kT} [ (s_k − r_k)' Q_k (s_k − r_k) + (s_k − x_k)' R (s_k − x_k) ]
    subject to   x_{k+1} = A_k s_k + b_k ,   s_k = x_k + u_k
```

Soft-landing config: `Q_k = 0` (or tiny) for `k < kT`, `Q_{kT} = Q_term` (large),
`R = ρ I` (ρ>0). Deadbeat config: single layer `kT`, `Q_term` huge, `ρ→0`.

### 1.2 Backward recursion (VERIFIED — implement these formulas exactly)
Value function `V_k(x) = x' S_k x − 2 v_k' x + const`. Sentinel beyond the band:
`S_{kT+1} = 0`, `v_{kT+1} = 0`. For `k = kT, kT-1, …, k0`:

```
M_k = Q_k + R + A_k' S_{k+1} A_k                       # 2x2 SPD
e_k = Q_k r_k + A_k' (v_{k+1} − S_{k+1} b_k)           # 2-vector
F_k = M_k^{-1} R                                       # feedback gain on x_k
g_k = M_k^{-1} e_k                                     # feedforward
s_k*(x) = F_k x + g_k          ( u_k = (F_k − I) x + g_k )
S_k = R − R M_k^{-1} R                                 # symmetric PSD
v_k = R g_k
```

Checks the implementer MUST confirm in the self-test:
- **Terminal layer** (`S_{kT+1}=0`): `s* = (Q_term+R)^{-1}(R x + Q_term r)` — a
  weighted average of natural `x` and target `r`. `R→0 ⇒ s*→r` (deadbeat/slam);
  `R→∞ ⇒ s*→x` (no move).
- **Deadbeat single layer, R→0, A invertible**: `s* → A_k^{-1}(r_{kT} − b_k)` so
  `x_{k+1} = A_k s* + b_k = r_{kT}` exactly (lands the *next* coordinate on target).
- **Condensed-QP gold cross-check**: stack the whole horizon decision
  `U = [s_{k0};…;s_{kT}]`, write `J(U)` as a single convex quadratic
  `½U'PU + q'U + c` (states eliminated via `x_{k+1}=A_k s_k + b_k`), solve
  `U* = −P^{-1} q`, roll the Riccati policy `s_k = F_k x_k + g_k` forward through
  the SAME dynamics, and assert the two `s`-trajectories agree to < 1e-8 and the
  costs agree to < 1e-8. (This is the analogue of `pts_mpc` test [1]/[7].)

### 1.3 Public API
```python
def finite_horizon_lqr(A, b, band, r, R, Q_stage, Q_term, *, terminal_layer=None)
    # A:(L-1,2,2) b:(L-1,2) full fitted plant; band: list of actuated layer indices
    #   (ascending, contiguous); r:(L,2) per-layer reference (r[terminal_layer]=target)
    # R:(2,2) effort; Q_stage:(2,2) small intermediate; Q_term:(2,2) terminal
    # terminal_layer defaults to band[-1].
    # returns {"F": {k:(2,2)}, "g": {k:(2,)}, "S": {k:(2,2)}, "v": {k:(2,)},
    #          "band": band, "terminal_layer": kT}

def lqr_rollout(A, b, gains, x0, band)          # roll s_k=F_k x_k+g_k forward, return
    # states x_k (natural) and s_k (actuated) and u_k over the band. (numpy, no model)

def total_effort(u_seq)                         # Σ_k ‖u_k‖  (the off-manifold push)
```
Provide a `deadbeat_gains(A,b,kT,target,...)` convenience returning the R→0 corner,
AND note that the *physical* deadbeat baseline in the drivers is simply the
fixed-target-angle reset at one layer (so the driver can use the angle directly).

### 1.4 Self-test (`if __name__ == "__main__"`)
Synthetic stable affine plant. Assert: (a) condensed-QP gold cross-check < 1e-8;
(b) deadbeat limit lands terminal coordinate on target < 1e-6; (c) **monotone
trade-off**: sweeping ρ up ⇒ `total_effort` ↓ and terminal angle error ↑
(monotone), printing the frontier; (d) soft-landing (band of layers, ρ>0) reaches
the same terminal angle as the single-layer deadbeat at strictly lower
`max_k ‖u_k‖` (smaller per-layer push). Print `[oas_lqr self-test] OK`.

---

## 2. `oas_observer.py` — Kalman filter / EKF + observability (Idea 2)

### 2.1 Linear-Gaussian model
```
x_{k+1} = A_k x_k + B_k u_k + b_k + w_k,   w_k ~ N(0, W)     (process; B_k = A_k here)
z_k     = H_k x_k + v_k,                    v_k ~ N(0, V)     (measurement)
```
Standard recursion (implement exactly):
```
predict:  x̂⁻ = A_{k-1} x̂_{k-1} + B u_{k-1} + b_{k-1};   P⁻ = A_{k-1} P_{k-1} A_{k-1}' + W
update:   S  = H P⁻ H' + V;  K = P⁻ H' S^{-1}
          x̂  = x̂⁻ + K (z_k − H x̂⁻);  P = (I − K H) P⁻   (use Joseph form for stability)
```

### 2.2 Noise estimation from the fitted plant (reuse PTS residuals)
`estimate_process_noise(A, b, c_traj)`: from `(N,L,2)` coordinate paths, compute
one-step residuals `ε_k = c_{k+1} − (A_k c_k + b_k)` and return pooled (and
per-layer) covariance `W` — this is the ≈14% PTS residual, now as a covariance.
`V` is supplied (measurement-noise model) or estimated from per-sample spread.

### 2.3 Observability
`observability_matrix(A, H, n)` and `is_observable(A,H)` (rank of `[H;HA;…;HA^{n-1}]`)
for a representative time-invariant `A` (e.g. band-mean), plus the observability
**Gramian** condition number as a continuous degree-of-observability. Document the
time-varying caveat.

### 2.4 EKF on the circle
`ekf_angle(...)`: state `[φ, ω]`, process `φ_{k+1}=φ_k+ω_k`, `ω_{k+1}=ω_k` (+W),
scalar measurement `z = φ` with **wrapped innovation** `atan2(sin(z−Hx̂⁻), cos(...))`.
H=[1,0]. This is the SO(2) variant for Exp 5 / circle-nonlinearity risk.

### 2.5 Public API
```python
class KalmanFilter:  # time-varying-capable; .reset(x0,P0); .step(z, A,b,H,W,V,u=0)->x̂,P
def kalman_filter_seq(A,b,H,W,V,z_seq,x0,P0,u_seq=None)   # batch a whole trajectory
def estimate_process_noise(A,b,c_traj)                    # -> W (2,2), W_per_layer
def observability_matrix(A,H,n=2); def is_observable(A,H)
def observability_gramian(A,H,n); 
def ekf_angle(...)                                        # -> phi_hat, omega_hat, P
```

### 2.6 Self-test
(a) Synthetic LG system: filtered MSE < raw-measurement MSE; steady-state gain
matches the algebraic (DARE) filter gain. (b) **Partial observability**: `dim x=2`,
scalar `z` (observable pair) — fusing the sequence recovers `x` (low MSE) where a
single measurement cannot; and an *unobservable* pair is flagged by
`is_observable=False`. (c) Degeneracy: `V→0 ⇒ x̂→z`; `W→0 ⇒ x̂→model prediction`.
(d) EKF tracks a wrapping ramp angle without divergence. Print `[oas_observer self-test] OK`.

---

## 3. `oas_controller.py` — glue (LQR / LQG / deadbeat in the loop)

Own copy of the mutable-angle reset (identical semantics to
`pts_controller._apply_reset` / `clas_controller.make_clas_hook`). Mirror PTS's
`PTSState` / `make_pts_layer_hook` / `attach_pts_hooks` / policy pattern.

```python
class OASState:           # .policy, .enabled, .record, .log  (per forward pass)
def _apply_reset(hidden,b1,b2,theta)                 # norm-preserving SO(2) reset (copy)
def make_oas_layer_hook(layer_idx,b1,b2,state)       # tuple-aware; calls state.policy
def attach_oas_hooks(module_dict,layers,b1,b2,state)

# policies: __call__(layer_idx, coords:(M,2) torch) -> (theta:(M,) torch, ctrl|None)
class PolicyNone                                     # read-only / keep angle
class PolicyDeadbeat(target_angle, layer)            # single-layer slam (Angular Steering)
class PolicyMultiAngle(target_angle, layers)         # fixed angle at a BAND (Exp-1 lever)
class PolicySoftLandingLQR(gains, ref)               # s=F_k c + g_k ; theta=angle(s)
class PolicyLQG(gains, ref, kalman_cfg)              # filter c per-layer, then LQR on x̂;
                                                     #   .reset() at band start
def build_softlanding(A,b,band,ref,target,R_rho,Q_term,Q_stage=0,terminal_layer=None)
    # -> gains dict from oas_lqr.finite_horizon_lqr (the offline solve)
```
`PolicyLQG` runs `oas_observer.KalmanFilter` sequentially across band layers within
one forward pass (hooks fire in layer order); it MUST reset filter state per
forward pass (call `.reset()` from the driver before each prefill/decode-step, and
guard by detecting `layer_idx == band[0]`). The actuated angle is
`angle(F_k x̂_k + g_k)`.

### 3.1 Self-test
No model. On a tiny synthetic plane: (a) `_apply_reset` is norm-preserving on the
plane and angle-exact (copy PTS test). (b) `PolicySoftLandingLQR` reproduces
`oas_lqr.lqr_rollout` angles when the actuator is additive (angle-only introduces
the known tax). (c) `PolicyLQG` with `V→0` equals `PolicySoftLandingLQR`. Print OK.

---

## 4. `oas_offline.py` — model-free experiments on saved trajectories

Runs from `../SO2/phase_portrait_output/Qwen2.5-3B-Instruct/trajectories.npz`
(the same file PTS uses). `--traj` arg, default to that path. Reuses
`pts_dynamics.fit_layer_dynamics` / `validate_dynamics`. Computes the steering band
exactly as `pts_offline.py` does (angle-separation > 0.5 rad around the peak). Then:

- **Plant + noise**: fit `{A_k,b_k}` on a train split, validate held-out (reuse),
  `W = estimate_process_noise(...)`.
- **Exp 2 (geometric, offline part)**: enforce target angle at a *single* layer,
  sweep that layer across the band, roll the fitted plant to the band end, and
  report the **terminal geometric angle error** vs enforcement layer (does early
  enforcement wash out under the natural dynamics? → motivates the *terminal*
  objective and picks `kT`). Save the curve.
- **Exp 3 (offline part — the headline maths)**: soft-landing LQR vs single-layer
  deadbeat at **matched terminal angle**. Metrics: total effort `Σ‖u_k‖` and
  `max_k‖u_k‖` (off-manifold-push proxy for coherence cost), and **robustness** =
  terminal-angle variance under a perturbed plant `Â_k = A_k + δ·N(0,1)` (Monte
  Carlo). Show soft-landing achieves matched terminal angle at lower push and lower
  sensitivity, and **correlate the soft-landing advantage with `‖A_k − I‖₂` across
  depth bands** (predict: largest in rotation-dominated bands; smallest in the
  near-identity late band — the honest PTS hedge). Sweep ρ to trace the
  effort-vs-terminal-error frontier (deadbeat = the ρ→0 endpoint).
- **Exp 4 (offline part)**: observability of `(A_band, H)`; compare behavioral-state
  (here: angle) estimate from **Kalman-fused** vs **best single-layer** vs
  **ad-hoc EMA** under injected measurement noise `V` and process noise `W`;
  compare **LQG** (control on `x̂`) vs **LQR-on-raw** at matched effort under
  injected noise (LQG should win as noise grows; tie when noise→0 — report honestly).

Save `oas_offline_results.npz` (everything the plot needs). Print a clear summary.
This driver MUST run to completion model-free (it will be executed in CI of the
build) — keep it fast (< ~30 s).

## 5. `oas_prototype.py` — model in the loop (Qwen2.5-3B-Instruct, A10G)

Mirror `pts_prototype.py` structure (config block, plane+plant on the real
residual stream, `prefill_with_policy`, `baseline_margin`, `generate`). Add a
`--smoke` flag (tiny N, MAX_NEW_TOKENS small) so the build can dry-run it without a
full run. Experiments:

- **Exp 1 — AUTHORITY GATE (run first, headline kill-switch).** Sweep steered band
  **width** 1→many layers (single layer, then growing contiguous bands around the
  steer layer). For each width, apply the multi-layer angle (use `PolicyMultiAngle`
  at the refusal-suppressing target, and separately the soft-landing LQR) and
  measure the **realized behavioral range** (`G`-sweep of the refusal/compliance
  margin from `observables.make_margin_fn`) for (a) **refusal** [known strong] and
  (b) a **sustained attribute** [sentiment, via generation-eliciting contrastive
  instructions like CLAS's drift test — "write upbeat…" vs "write bleak…"]. Report:
  does multi-layer materially raise authority vs single-layer? PASS/FAIL exactly as
  the proposal states. This is the lever CLAS left open.
- **Exp 2 — enforcement-layer sweep (behavioral).** Single-layer deadbeat at the
  target angle, sweep the enforcement layer across depth, measure the behavioral
  margin → behavioral-effect-vs-enforcement-layer curve. Confirms/where-the-angle-
  must-land; cross-check against the offline geometric curve.
- **Exp 3 — soft-landing vs deadbeat (coherence/robustness).** At **matched
  terminal behavioral effect** (margin), compare single-layer deadbeat vs the
  soft-landing LQR band on **coherence** = KL(steered‖unsteered) of the next-token
  distribution (+ a short-generation perplexity proxy) and **robustness** = margin
  variance under prompt paraphrase / a perturbed plant. Plus qualitative
  generations. Predict soft-landing matches effect at lower coherence cost, gap
  largest off the near-identity band.
- **Exp 4 — observer value (in-model, light).** Inject measurement noise into the
  per-layer readout; show LQG (filtered) tracks the target angle more stably than
  LQR-on-raw, and the filtered estimate is smoother (the principled EMA). Mostly a
  confirmation of the offline result on the real model.

Save `oas_prototype_results.npz` + write `oas_prototype.log`.

## 6. `oas_plot.py` — figures (mirror `pts_plot.py`)
`plot_offline()` from `oas_offline_results.npz` and `plot_prototype()` from
`oas_prototype_results.npz`; `matplotlib.use("Agg")`; guard missing files. One
multi-panel figure each: (offline) Exp-2 enforcement-layer curve, Exp-3
effort-vs-terminal-error frontier + soft-vs-deadbeat at matched angle + advantage
vs `‖A_k−I‖`, Exp-4 Kalman-vs-single-vs-EMA + LQG-vs-LQR-under-noise; (prototype)
Exp-1 authority-vs-bandwidth for refusal & sentiment (the gate), Exp-2 behavioral
enforcement-layer curve, Exp-3 coherence-vs-effect Pareto.

---

## 7. Build acceptance criteria
- `python oas_lqr.py`, `python oas_observer.py`, `python oas_controller.py` all
  print `... OK` (self-tests green).
- `python oas_offline.py` runs to completion on the saved trajectories and writes
  `oas_offline_results.npz`; `python oas_plot.py` produces `oas_offline.png`.
- `python oas_prototype.py --smoke` imports and runs end-to-end on tiny N.
- No imports from `../CLAS` or `../PTS` except `pts_dynamics` (documented shim).
- Control-theory correctness reviewed against §1.2 (Riccati) and §2.1 (Kalman).
