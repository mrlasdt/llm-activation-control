# Predictive Trajectory Steering: Model Predictive Control in the Angular Steering Plane

## 1. Problem Statement

Activation steering methods control LLM behavior by perturbing hidden representations at inference time. Current approaches face two fundamental limitations:

1. **The setpoint problem**: Methods either have no target (open-loop addition/rotation) or use ad-hoc scalar targets (A-LQR's beta*_k = lambda * mu_k). No principled theory defines WHERE to steer activations.

2. **The reactivity problem**: Even closed-loop methods (PID, LQR) are reactive — they correct deviations after they occur. No method anticipates how current steering will affect downstream layers or future tokens.

We propose **Predictive Trajectory Steering (PTS)**, which addresses both problems by:
- Defining the steering target as a **reference trajectory** through the 2D steering plane (not a scalar setpoint)
- Using **Model Predictive Control (MPC)** to track this trajectory with lookahead and constraint satisfaction

---

## 2. Background & Positioning

### 2.1 Control-Theoretic Steering Landscape

| Method | Target Definition | Control Type | Model-Based? | Constraints? |
|--------|------------------|-------------|-------------|-------------|
| ActAdd / Ablation | None (open-loop) | Proportional only | No | No |
| Angular Steering (ours) | Fixed angle | Open-loop rotation | No | No |
| PID-AcT (ICLR'26) | Offline error signal | PID across layers | No | No |
| ODESteer (ICLR'26) | Safe region (barrier) | ODE integration | No | Implicit (barrier) |
| A-LQR (2026) | Scalar setpoint per layer | LQR across layers | Yes (Jacobians) | No |
| **PTS (proposed)** | **2D trajectory** | **MPC across layers** | **Yes (2x2 model)** | **Yes (explicit)** |

### 2.2 Key Enabling Results

**Local linearity** (A-LQR, April 2026): Transformer blocks are well-approximated by locally-linear models. Layer-wise Jacobians at different reachable activations within the same layer are highly correlated.

**Low-order linear depth dynamics** (Nettasinghe & Joseph, March 2026, arXiv:2603.12541): A 32-dimensional linear surrogate reproduces the layerwise sensitivity profile of GPT-2-large with near-perfect agreement. Agreement improves with model scale. This proves cheap dynamics models of transformer depth evolution are accurate.

**Activation trajectories** (Fernando et al., 2025-2026): Residual stream activations follow curved trajectories with attractor-like dynamics. Spectral analysis reveals a monotonic spectral gradient through depth — from rotation-dominated early layers to near-symmetric late layers.

**LLM reasoning as trajectories** (Microsoft, April 2026, arXiv:2604.05655): Correct and incorrect solutions diverge systematically in trajectory space (ROC-AUC 0.87). Step-wise mean trajectories serve as effective reference signals.

**Non-surjectivity** (Mishra et al., April 2026, arXiv:2604.09839): Additive steering pushes activations off the reachable manifold. Rotation-based steering (Angular Steering) partially mitigates this by preserving activation norms.

### 2.3 What This Proposal Adds

The **separation principle** from classical control theory: cleanly separate **reference generation** (what to track) from **controller design** (how to track it). In current LLM steering, these are conflated — the same contrastive data defines both the direction and the intervention. PTS decouples them:

- **Reference generation**: Extract a 2D trajectory from contrastive data (Section 4)
- **Controller design**: MPC tracks this trajectory optimally (Section 5)

---

## 3. The Angular Steering Plane (Review)

From Angular Steering (Vu & Nguyen, NeurIPS 2025), we define a 2D steering plane:

1. **Feature direction** d_feat: extracted via difference-in-means from contrastive data (harmful vs harmless activations), selected by max average cosine similarity across layers
2. **Second axis** d_PC0: first principal component of candidate directions across layers
3. **Orthonormal basis**: {b1, b2} via Gram-Schmidt on {d_feat, d_PC0}
4. **Projection matrix**: P = b1 b1^T + b2 b2^T

Any activation h can be decomposed:
```
h = proj_P(h) + proj_Q(h)
```
where proj_P(h) is the 2D component in the steering plane and proj_Q(h) is the (d-2)-dimensional orthogonal complement (left unchanged by steering).

The 2D coordinates of h in the steering plane:
```
c(h) = [b1^T h, b2^T h]^T ∈ R^2
```

Angular Steering rotates c(h) to a target angle theta while preserving ||proj_P(h)||.

---

## 4. Reference Trajectory Extraction

### 4.1 Layer-wise 2D Trajectory

For a given input prompt, as it passes through layers k = 1, ..., L, the 2D coordinates trace a path:
```
tau(prompt) = {c_k}_{k=1}^{L} = {[b1^T z_k, b2^T z_k]^T}_{k=1}^{L}
```
where z_k is the activation at layer k (after normalization, before attention or MLP).

### 4.2 Contrastive Trajectory Sets

From the calibration datasets D_harmful and D_harmless:

**Harmful trajectories**: tau_harmful^(i) for each prompt in D_harmful
**Harmless trajectories**: tau_harmless^(i) for each prompt in D_harmless

Compute statistics:
```
tau_harmful_mean = {mean_i(c_k^harmful(i))}_{k=1}^{L}     # mean harmful trajectory
tau_harmless_mean = {mean_i(c_k^harmless(i))}_{k=1}^{L}   # mean harmless trajectory
Sigma_k = Cov(c_k)                                          # per-layer covariance
```

### 4.3 Reference Trajectory Definition

The reference trajectory tau* defines WHERE we want the steered activation to go at each layer. Several options:

**Option A: Target trajectory (direct)**
```
tau*_k = tau_harmless_mean_k    (steer toward harmless behavior)
```

**Option B: Offset trajectory (relative)**
```
tau*_k = tau_harmful_mean_k + lambda * (tau_harmless_mean_k - tau_harmful_mean_k)
```
where lambda ∈ [0, 1] controls interpolation. This generalizes A-LQR's LFS: their scalar setpoint beta*_k = lambda * mu_k is the projection of this onto b1 only. We use both dimensions.

**Option C: Tube trajectory (region)**
```
tau*_k = {c ∈ R^2 : (c - tau_harmless_mean_k)^T Sigma_k^{-1} (c - tau_harmless_mean_k) <= chi^2_alpha}
```
This defines an ellipsoidal tube around the harmless mean trajectory. MPC steers activations into this tube rather than to an exact point, allowing more flexibility.

**Option D: Adaptive trajectory (input-dependent)**
For each input prompt, compute its current 2D trajectory, then define the reference as the closest point in the harmless trajectory distribution. This adapts the reference to the specific input.

### 4.4 Connection to Existing Setpoints

| Method | Target | Dimension | Adaptive? |
|--------|--------|-----------|-----------|
| A-LQR LFS | beta*_k = lambda * mu_k | 1D (scalar projection onto v_k) | No (fixed lambda) |
| Angular Steering | Fixed angle theta | 1D (angle in 2D plane) | No (fixed angle) |
| PTS Option A | tau_harmless_mean_k | 2D (point in plane) | No (fixed trajectory) |
| PTS Option B | Interpolated trajectory | 2D (point in plane) | Partial (lambda) |
| PTS Option C | Ellipsoidal tube | 2D (region in plane) | No (fixed tube) |
| PTS Option D | Nearest harmless trajectory | 2D (point in plane) | Yes (input-dependent) |

---

## 5. Model Predictive Control Formulation

### 5.1 Dynamics Model in the Steering Plane

At each layer k, the 2D coordinates evolve as:
```
c_{k+1} = f_k(c_k, u_k)
```
where:
- c_k ∈ R^2: current 2D coordinates
- u_k ∈ R^2: steering perturbation in the plane
- f_k: dynamics of the k-th transformer block projected onto the plane

**Linear approximation** (justified by local linearity results):
```
c_{k+1} ≈ A_k c_k + B_k u_k
```
where A_k ∈ R^{2x2} and B_k ∈ R^{2x2}.

**Fitting A_k and B_k**: Two options:

**(a) From contrastive data (cheap, offline)**:
- Run N prompts through the model, record {c_k^(i)} at each layer
- Fit A_k via least squares: A_k = argmin_A sum_i ||c_{k+1}^(i) - A c_k^(i)||^2
- Set B_k = I (identity) since we directly add perturbations in the plane

**(b) From Jacobians (more accurate, more expensive)**:
- Compute the full Jacobian J_k = d(phi_k)/dz at the mean activation
- Project: A_k = [b1, b2]^T J_k [b1, b2]
- This is only 2 vector-Jacobian products (much cheaper than A-LQR's full d×d Jacobian)

**Key insight**: The linear depth dynamics paper (arXiv:2603.12541) shows that option (a) works well. We do NOT need to compute full Jacobians. A simple least-squares fit from ~500 forward passes gives an accurate 2x2 model per layer.

### 5.2 MPC Optimization (Layer-Horizon)

At each layer k during steered inference, solve:

```
min_{u_k, ..., u_{k+H-1}} sum_{j=0}^{H-1} [ (c_{k+j} - tau*_{k+j})^T Q (c_{k+j} - tau*_{k+j}) + u_{k+j}^T R u_{k+j} ]
                         + (c_{k+H} - tau*_{k+H})^T Q_f (c_{k+H} - tau*_{k+H})

subject to:
    c_{k+j+1} = A_{k+j} c_{k+j} + B_{k+j} u_{k+j}     (dynamics)
    ||u_{k+j}||_2 <= u_max                                (steering magnitude bound)
    c_{k+j} ∈ C_safe                                      (optional: stay in safe region)
```

where:
- H: prediction horizon (number of layers to look ahead, e.g., 5-10)
- Q ∈ R^{2x2}: state tracking cost (penalizes deviation from reference)
- R ∈ R^{2x2}: control cost (penalizes large perturbations)
- Q_f: terminal cost
- u_max: maximum steering perturbation magnitude
- C_safe: optional safe region constraint (e.g., the tube from Option C)

**Apply only u_k** (the first control action), advance to layer k+1, re-measure c_{k+1}, and re-solve.

### 5.3 Computational Cost

This is a tiny QP:
- **Decision variables**: 2H (e.g., 2×10 = 20 for H=10)
- **Constraints**: 2H (magnitude bounds) + optional region constraints
- **Solve time**: <100 microseconds on CPU with OSQP, or precomputed as explicit MPC (lookup table)

For comparison, a single transformer layer forward pass takes 1-5ms on GPU. The MPC overhead is **<2% of total inference time**.

For extreme efficiency, **explicit MPC** precomputes the solution offline as a piecewise-affine function of the current state c_k. At inference, it's a single lookup + 2x2 matrix multiply — effectively zero cost.

### 5.4 Connection to Angular Steering

The MPC output u_k ∈ R^2 in the steering plane can be converted to an equivalent rotation angle:
```
theta_k = atan2(c_k[2] + u_k[2], c_k[1] + u_k[1])
```

This means PTS produces a **per-layer, per-token adaptive angle** for Angular Steering. The implementation just modifies the target angle at each layer rather than using a fixed global angle.

### 5.5 Relation to A-LQR

PTS and A-LQR solve related but different problems:

| Aspect | A-LQR | PTS (proposed) |
|--------|-------|----------------|
| State space | Full d-dimensional | 2D steering plane |
| Target | 1D scalar beta*_k | 2D trajectory point tau*_k |
| Dynamics model | d×d Jacobian per layer | 2×2 matrix per layer |
| Controller | LQR (unconstrained) | MPC (constrained) |
| Offline cost | Jacobian computation (VRAM-intensive) | Forward passes only (cheap) |
| Online cost | d×d matrix multiply per layer | 2D QP solve per layer |
| Constraints | None (quadratic penalty only) | Explicit (magnitude, region) |
| Storage | K_k ∈ R^{d×d} per layer (~2-6 GB) | A_k ∈ R^{2×2} per layer (~256 bytes) |

PTS trades representational richness (d-dim vs 2D) for computational efficiency and constraint handling. The 2D projection is justified because Angular Steering already shows that the steering plane captures the behaviorally relevant variation.

---

## 6. Extension: Token-Horizon MPC

### 6.1 Motivation

Layer-horizon MPC optimizes within a single forward pass. But LLMs generate tokens autoregressively — the activation at token t+1 depends on which token was sampled at step t. Token-horizon MPC extends the prediction across generation steps.

### 6.2 Token-Level Dynamics

At each generation step t, the 2D coordinates at the critical layer evolve:
```
c^(t+1) = g(c^(t), token_t)
```

where token_t is the generated token (stochastic). Two approaches to model g:

**(a) Empirical average**: Run the model on many prompts, record how c^(t) evolves across tokens, fit a linear model:
```
c^(t+1) ≈ A_token c^(t) + w_t
```
where w_t captures the token-dependent noise.

**(b) Rollout-based**: At each generation step, perform a short speculative rollout of k tokens, observe the resulting 2D trajectory, and optimize steering to keep it on track. This leverages speculative decoding infrastructure.

### 6.3 Two-Level MPC

Combine layer-horizon and token-horizon:
- **Outer loop** (token level): Every N tokens, re-evaluate the 2D trajectory and adjust the reference trajectory or MPC parameters
- **Inner loop** (layer level): At each layer within each token's forward pass, run the 2D MPC

This is analogous to hierarchical MPC in robotics (slow outer planner + fast inner controller).

---

## 7. Experimental Plan

### 7.1 Models
- Primary: Qwen2.5-3B-Instruct, Qwen2.5-7B-Instruct (same as Angular Steering paper)
- Secondary: Llama-3.1-8B-Instruct, Gemma-2-9B-IT (for generalization)

### 7.2 Tasks
- **Refusal/Jailbreaking** (primary): AdvBench eval split (104 prompts), measured by ASR (HarmBench), refusal score (substring matching), LlamaGuard3
- **Toxicity** (secondary): RTP dataset (1000 prompts), measured by RoBERTa toxicity classifier + Dist-1/2/3
- **Truthfulness** (secondary): TruthfulQA generation split, measured by T*I score

### 7.3 Baselines
1. **No steering** (original model)
2. **Angular Steering** (fixed angle, our NeurIPS paper)
3. **PID-AcT** (layer-wise PID feedback)
4. **A-LQR** (LQR with Jacobians and LFS setpoint)
5. **ODESteer** (barrier function ODE)

### 7.4 Experiments

**Experiment 1: Reference trajectory analysis**
- Extract 2D trajectories for harmful and harmless prompts across all models
- Visualize trajectory separation, curvature, and variance across layers
- Compare trajectory-level separation with scalar projection separation (Fig. 4 of Angular Steering paper)
- Deliverable: Visualization showing that 2D trajectories carry more information than 1D projections

**Experiment 2: Dynamics model validation**
- Fit 2x2 linear dynamics models A_k from contrastive data
- Compare prediction accuracy: 1-step, 5-step, and full L-step rollouts
- Compare with A-LQR's Jacobian-based linearization (projected to 2D)
- Deliverable: Table showing prediction error vs model complexity

**Experiment 3: Layer-horizon MPC vs baselines**
- Run PTS with layer-horizon MPC on refusal steering
- Compare with Angular Steering (fixed angle), PID (reactive feedback), A-LQR (scalar setpoint)
- Sweep Q/R ratio to explore the steering-strength vs capability tradeoff
- Deliverable: Pareto curves of steering effectiveness vs model capability (PPL, TinyBenchmarks)

**Experiment 4: Constraint satisfaction**
- Add explicit constraints to MPC: ||u_k|| <= u_max, and c_k ∈ safe_tube
- Show that constrained MPC avoids the coherence degradation seen in smaller models with non-adaptive Angular Steering
- Compare perplexity stability across the full angle sweep
- Deliverable: Perplexity plots showing bounded degradation under PTS vs unbounded under baselines

**Experiment 5: Trajectory options comparison**
- Compare Options A/B/C/D for reference trajectory definition
- Analyze: which option gives the best steering with minimal capability loss?
- Deliverable: Ablation table across trajectory options

**Experiment 6: Token-horizon extension**
- Implement outer-loop token-level MPC (re-adjust steering every N tokens)
- Test on long-form generation where models tend to drift
- Compare with fixed-angle and PID approaches
- Deliverable: Token-position analysis showing PTS maintains behavioral consistency over long sequences

**Experiment 7: Efficiency benchmarks**
- Measure tokens/sec overhead of PTS vs baselines
- Compare VRAM requirements: PTS (2x2 matrices) vs A-LQR (d×d K matrices)
- Deliverable: Efficiency table proving PTS is production-viable

### 7.5 Ablations
- Prediction horizon H: how many layers ahead should MPC plan?
- Dynamics model complexity: constant A vs per-layer A_k vs nonlinear
- Reference trajectory: mean vs median vs mode vs input-adaptive
- With/without constraints: does constraint satisfaction meaningfully help?

---

## 8. Theoretical Contributions

### 8.1 Trajectory Tracking Error Bound

Extend A-LQR's Theorem 4.2 to the 2D case. Since our dynamics are 2x2, the bound is tighter and more interpretable:

```
||c_k - tau*_k||_2 <= rho^k ||c_1 - tau*_1||_2 + sum_{j=1}^{k-1} rho^{k-j} ||w_j||_2
```

where rho = max_k ||A_k - B_k K_k||_2 (spectral radius of closed-loop dynamics) and w_j is the linearization error at layer j.

If rho < 1 (the MPC stabilizes the closed-loop), the tracking error is bounded and contracts geometrically.

### 8.2 Constraint Satisfaction Guarantee

MPC with constraint ||u_k|| <= u_max guarantees that the steering perturbation never exceeds a known bound. This directly addresses the coherence degradation problem: by bounding the perturbation, we limit how far off-manifold the steered activation can go.

### 8.3 Separation Principle

Formally establish that PTS implements the separation principle for activation steering:
- Reference trajectory tau* is computed independently of the controller
- MPC controller is designed independently of the reference
- Performance decomposes: total error = reference error + tracking error

This enables independent optimization of each component and provides a framework for future improvements to either part.

### 8.4 Angular Steering Plane Sufficiency

Prove that for rotation-based steering, the 2D plane captures the behaviorally relevant dynamics:
- The projection onto {b1, b2} maximizes the variance explained in the harmful/harmless distinction
- The orthogonal complement Q is unchanged by steering (by construction)
- Therefore, the 2D dynamics model is sufficient for predicting steering effects

---

## 9. Implementation Plan

### Phase 1: Trajectory Extraction & Analysis [Weeks 1-2]
- Extend `pytorch_pure/extract_directions.py` to record full 2D trajectories
- Implement trajectory visualization in the steering plane
- Produce Figure 1 of the paper (2D trajectory plots)

### Phase 2: Dynamics Model Fitting [Weeks 2-3]
- Fit 2x2 A_k matrices from contrastive forward passes
- Validate prediction accuracy across layers and models
- Compare with Jacobian-based linearization

### Phase 3: MPC Implementation [Weeks 3-5]
- Implement 2D MPC using OSQP (or explicit MPC for maximum speed)
- Integrate with Angular Steering's hook system in `angular_steering.py`
- Convert MPC output to per-layer rotation angles

### Phase 4: Evaluation [Weeks 5-8]
- Run all experiments from Section 7
- Generate paper figures and tables

### Phase 5: Token-Horizon Extension [Weeks 8-10]
- Implement outer-loop token-level MPC
- Evaluate on long-form generation tasks

### Phase 6: Writing [Weeks 10-14]
- Draft paper with the narrative: setpoint problem + predictive control -> PTS
- Target venue: ICML 2027, NeurIPS 2026, or ICLR 2027

---

## 10. Key Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `angular_steering.py` | Modify | Add per-layer adaptive angle from MPC output |
| `trajectory_extraction.py` | Create | Extract and store 2D trajectories from contrastive data |
| `dynamics_model.py` | Create | Fit and store 2x2 linear dynamics models per layer |
| `mpc_controller.py` | Create | MPC solver (OSQP wrapper or explicit MPC) |
| `pts_steering.py` | Create | Main PTS pipeline: trajectory + dynamics + MPC -> steering hooks |
| `generate_responses.py` | Modify | Support per-layer adaptive angles and token-level re-planning |
| `evaluate_jailbreak.py` | Reuse | Existing evaluation pipeline |
| `visualization/` | Extend | 2D trajectory plots, MPC solution visualization |

---

## 11. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 2D plane insufficient for some behaviors | Medium | Fall back to higher-D subspace (3D-5D) with same MPC framework |
| Linear dynamics model inaccurate | Low | Supported by existing literature; can add nonlinear correction |
| MPC overhead too high | Very Low | 2D QP is trivially small; explicit MPC eliminates runtime cost |
| Reference trajectory not discriminative enough | Medium | Try Options C/D; combine with adaptive/input-dependent targets |
| A-LQR's full-dimensional approach dominates | Medium | Our contribution is efficiency + constraints, not raw performance |

---

## 12. Expected Contributions

1. **First trajectory-based setpoint definition** for activation steering (replacing ad-hoc scalar setpoints)
2. **First application of MPC** to LLM activation steering (with explicit constraint handling)
3. **Orders-of-magnitude efficiency gain** over A-LQR (2x2 vs d×d, no Jacobians, ~256 bytes vs ~2-6 GB storage)
4. **Formal separation principle** for activation steering (decoupling reference generation from controller design)
5. **Constraint satisfaction guarantees** preventing coherence degradation (bounded perturbations)
6. **Unified view** connecting Angular Steering's rotation to optimal trajectory tracking

---

## 13. References

### Papers in Repository
- Vu & Nguyen (2025). Angular Steering: Behavior Control via Rotation in Activation Space. NeurIPS 2025.
- Nguyen et al. (2025). Activation Steering with a Feedback Controller. ICLR 2026. arXiv:2510.04309.
- Skifstad et al. (2026). Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control. arXiv:2604.19018.

### Key External References
- Nettasinghe & Joseph (2026). As Language Models Scale, Low-order Linear Depth Dynamics Emerge. arXiv:2603.12541.
- Zhao et al. (2026). ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment. ICLR 2026.
- Mishra et al. (2026). Steered LLM Activations are Non-Surjective. arXiv:2604.09839.
- Fernando & Guitchounts (2026). Dynamics of the Transformer Residual Stream. arXiv:2605.14258.
- Microsoft (2026). LLM Reasoning as Trajectories. arXiv:2604.05655.
- Zhang et al. (2026). Truth as a Trajectory. arXiv:2603.01326.
- Tran et al. (2026). BarrierSteer: LLM Safety via Learning Barrier Steering. arXiv:2602.20102.
- NVIDIA (2026). TMPC: Test-Time Alignment for LLMs via Textual Model Predictive Control. ICLR 2026.
- SD-squared (2026). Steering Pretrained Drafters during Speculative Decoding. AAAI 2026. arXiv:2511.09844.
- ECCV (2024). An Optimal Control View of LoRA and Binary Controller Design for Vision Transformers.
