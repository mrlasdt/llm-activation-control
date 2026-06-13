# Baseline provenance — exact configs pulled from the authors' code

Phase-0 deliverable. The two papers do **not** tabulate their controller hyperparameters; these
were extracted directly from the authors' public repos (cloned read-only to `/tmp`). Every value
below carries a `file:line` reference into the clone so the reproduction is auditable.

- **A-LQR** — `github.com/trustworthyrobotics/lqr-activation-steering` → `/tmp/lqr-activation-steering`
- **PID-AcT / PID Steering** — `github.com/dungnvnus/pid-steering` → `/tmp/pid-steering`
  (the LM half is a fork of *this* project's `llm-activation-control`; toxicity uses Apple's Mean-AcT.)

---

## 1. A-LQR (Activation-LQR) — `steer/steering.py`, `steer/lqr_utils.py`

### Riccati / gains (B = I) — `lqr_utils.py:124 time_varying_lqr_noB`
Backward recursion from `S_T = Qf`, for `t = T-1..0`:
```
P = S[t+1] + R_t                 # (BᵀS B + R) with B=I
F = S[t+1] @ A_t                 # (BᵀS A) with B=I
G = Q_t + A_tᵀ S[t+1] A_t
K[t] = P⁻¹ F                     # feedback gain (d×d)
S[t] = G − Fᵀ P⁻¹ F
```
Mem-efficient variant keeps `S` on CPU (`lqr_utils.py:162`). `time_varying_lqr` (`:201`) is the
general-B form (unused for A-LQR since B=I).

### Cost matrices — `steering.py:30` `LQRSteering.__init__`
`Q = q·I` (per layer, ×T), `R = r·I`, `Qf = qf·I`. **Constructor defaults `q=10, r=10, qf=1`.**
`T = len(model.model.layers)`, `n = m = hidden_size`.

### LFS setpoint + control law — `steering.py:194 hook_setpoint_tracking` (Mode.SETPOINT, = `track_setpoint` `:398`)
- `v_k = e_k/‖e_k‖`, `β*_k = λ·‖e_k‖` (`betas[i] = lmbda * nrm`, `steering.py:436`). **λ default 1**, swept 0.5–2.5.
- Per layer (last token): read `x = input[0][:,-1,:]`; `α = β*_k − v_kᵀx`; `e = α·v_k`; `u = K_k @ e`;
  **add to the layer OUTPUT** `output[0][...,-1,:] += u`. (Reads block *input*, injects on block *output*.)
- `ALL_TOKENS=True` → A-LQR+ (apply to every token position; `steering.py:197`).
- Multi-concept: sum per-concept `u` (`steering.py:261 hook_multisteer_tracking`).

### Jacobian — `lqr_utils.py:32 linearize` (full, `autograd.functional.jacobian`, vectorized) / `:67 linearize_jvp_streamed_gpu` (column-by-column JVP into a preallocated GPU tensor, the memory-frugal path).
Nominal: linearize each block about the **collected D+ activation trajectory** `X[0]` (the mean/positive
nominal), `ū=0`. Block wrapped with rotary pos-emb via `tf_block_wrapper` (`:256`); control injected by
`transformerBlockControl` (`:241`): `x_next[...,-1,:] += u`.

### Decoding — `track_setpoint` (`steering.py:438`)
Greedy (`do_sample=False`) for **refusal**; otherwise sampling `top_p=0.3, repetition_penalty=1.2,
temperature=temp`, `use_cache=True`. `max_new_tokens`: 50 (refusal/truth), 100 (toxicity).

### Per-task overrides (best-params dicts read at run time)
- **Refusal** — `steer/refusal/test_ref.py:51 run_trials_lfs` defaults: `q=0.1, r=10, qf=0.1, λ=1, k=50,
  do_sample=False`. AdvBench `harmful_behaviors.csv`, `train_test_split(test_size=0.2, random_state=42)`
  → 104-prompt eval (`ref_asr_script.py:11`). Models `{gemma2b, gemma9b, qwen3b, qwen14b, llama8b,
  llama3b}` (`ref_asr_script.py:64`), loaded `quant=True`. Best `Q/R/Qf/λ` per model loaded from a saved
  dict (`ref_asr_script.py:257`). Contrastive dir = best-mean-cosine layer, replicated (`get_best_dir`).
- **Toxicity** general default `q=r=10, qf=1`; tables report best-over-λ.

---

## 2. S-PID (the A-LQR paper's own PID baseline) — `steer/PIDsteering.py`

PID on the **LFS feature error** (same `e = α·v_k`, `α = β*_k − v_kᵀx` as A-LQR), NOT using K:
```
e_sum += e
if layer_idx % 10 == 0: e_sum *= 0      # anti-windup: reset integral every 10 layers (PIDsteering.py:174)
u = Kp·e + Ki·e_sum + Kd·(e − e_prev)    # PIDsteering.py:177
e_prev = e
output[0][...,-1,:] += u
```
Constructor defaults `kp=10, ki=10, kd=1` (`:22`). **Refusal gains** `kp=0.5, ki=0.1, kd=0.1`
(`test_ref.py:160 run_trials_pid`), λ=1, k=50, greedy. Pure-P variant `track_setpoint_actadd`
(`:286`, Ki=Kd=0, select layers) = "ActAddLFS".

---

## 3. PID-AcT / PID Steering (Nguyen et al.) — two implementations

### 3a. Toxicity — `pid-steering/Mean-AcT/act/hooks/transport.py:274 GaussianOTPIDHook` (alias `OnlyMeanPIDHook`, registered `mean_ot_pid`)
- Error = diff-in-means `diff = mu2 − mu1` (`fit`, `:412`).
- **Integral across layers** at load time (`load_state_dict`, `:382`): loads every prior layer's saved
  `diff` and sets `diff_m = diff + 0.005·(mean(prior_diffs) + diff)` → **Ki ≈ 0.005** (cumulative).
- Actuator (onlymean, `:484`): `z_ot = z + 0.7·diff_m` then `strength·z_ot + (1−strength)·z`
  → additive with **strength 0.7** on diff_m. Sequential Mean-AcT pipeline.

### 3b. Jailbreak — `pid-steering/llm-activation-control/llama_many_layers.py` (transformer_lens)
- `ref_dir` = **normed** diff-in-means (normalize acts, mean over batch, harmful−harmless), last token
  (`:510-519`).
- `der_comp = ref_dir_set − roll(ref_dir_set,1)` (`:524`); `int_comp = cumsum(ref_dir_set, dim=0)` (`:526`).
- PID direction (`:536`): `v = p_coe·ref_dir + i_coe·int_comp[l] + d_coe·der_comp[l] + randn·noise + 0.1`,
  then row-normalized and used as the **directional-ablation** direction.
- **Gains `p_coe=1.0, i_coe=0.3, d_coe=0.01, noise_prob=1.0`** (`:50-53`), `momentum_mode=True`,
  `N_INST_TRAIN=512`. AdvBench harmful + Alpaca harmless.

### 3c. Paper-stated regimes (Fig 6, for cross-check)
`Kp=1`; `Ki∈[0.05,0.10]` (fastest `Ki=0.056` on Gemma-2-9b-it; stability `(-0.23,0.23)` at Kp=1; Gemma-2-2B
`(-0.1355,0.1355)`); `Kd∈[0.01,0.05]` (Kd 0.0→0.01 raises ASR 76.61→78.53). Euler discretization, h=1,
integral updated *after* emitting `u(k)`, `r(1)=0`.

---

## 4. What we reuse vs. re-derive
- **Reuse the A-LQR math verbatim**: `time_varying_lqr_noB` Riccati, `linearize_jvp_streamed_gpu` JVP,
  `transformerBlockControl` additive last-token actuator, the LFS `β*_k=λ‖e_k‖` + `u=K(β*−vᵀx)v` law,
  and S-PID's PID-on-LFS-error with the every-10-layer integral reset.
- **Reuse the PID-AcT law**: diff-in-means error + `cumsum` integral + `Δ` derivative across layers,
  gains `(1.0, 0.3, 0.01)` jailbreak / `(1, ~0.005–0.056, ~0.01)` toxicity, additive (ActAdd) or
  directional-ablation actuator.
- **Our own clean reimplementation** (`alqr_native.py`, `pid_native.py`) targets the same equations on
  HF models via `pytorch_pure/utils.add_hooks`, cross-checked against these references; the cone-restricted
  `casa_baselines.py` ports the same laws onto CASA's k-dim cone plant for the matched head-to-head.

## 5. Gaps / fallbacks
- A-LQR per-model best `Q/R/Qf/λ` come from saved dicts not in the repo → we sweep `λ∈{0.5,1,1.5,2,2.5}`
  with refusal defaults `q=0.1, r=10, qf=0.1` and report best-over-λ per the paper's selection rule.
- PID-AcT exact per-table gains beyond the above are not all pinned → use the extracted jailbreak/toxicity
  values + paper Fig-6 ranges.
- Both repos use 4-bit (`quant=True`) for ≥9B models; we match where memory requires it (flag in results).
