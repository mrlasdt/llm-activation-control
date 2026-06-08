# Angular Steering as Control on SO(2): Phase Portrait Analysis and Energy-Based Stabilization of LLM Behavior

## 1. Problem Statement

Angular Steering (Vu & Nguyen, NeurIPS 2025) controls LLM behavior by rotating activations within a 2D subspace. Despite its geometric nature, the method lacks a formal dynamical systems analysis:

1. **No characterization of natural dynamics**: How does the activation angle evolve across layers without intervention? Does it converge, oscillate, or drift?
2. **No stability guarantees**: When does a fixed-angle rotation preserve model capabilities? When does it break coherence?
3. **No principled controller design**: The rotation angle is chosen heuristically. What is the optimal angle at each layer?

We observe that Angular Steering naturally operates on **SO(2)** — the Lie group of planar rotations. This is the same mathematical structure underlying the simple pendulum. We propose to formalize this connection, characterize the activation dynamics via phase portrait analysis, and design energy-based controllers with provable stability guarantees.

---

## 2. The Core Analogy

### 2.1 The Simple Pendulum

A simple pendulum with angle theta from the downward vertical, angular velocity omega, and control torque u has dynamics:

```
d(theta)/dt = omega
d(omega)/dt = -(g/L) sin(theta) + u/(mL^2)
```

Key features:
- **Two equilibria**: stable at theta=0 (hanging), unstable at theta=pi (upright)
- **Phase portrait**: closed orbits (oscillation) near stable equilibrium, open orbits (rotation) beyond the separatrix
- **Energy**: E = (1/2) mL^2 omega^2 - mgL cos(theta)
- **Separatrix**: the curve E = mgL that separates oscillatory from rotational motion
- **Control problems**: stabilization (keep upright), swing-up (move from hanging to upright)

### 2.2 Angular Steering as a Dynamical System on SO(2)

In Angular Steering, at each layer k, the activation h_k has a projection onto the steering plane with basis {b1, b2}. Define:

```
phi_k = atan2(b2^T h_k, b1^T h_k)    — the activation angle at layer k
r_k = ||proj_P(h_k)||                  — the activation magnitude in the plane
```

As the activation passes through layers k = 1, ..., L (without steering):

```
phi_{k+1} = F_k(phi_k, r_k)           — natural angular dynamics
```

Define the angular velocity:

```
omega_k = phi_{k+1} - phi_k            — angular change per layer
```

The state (phi_k, omega_k) lives on **SO(2) x R** — the same state space as the pendulum.

### 2.3 The Mapping

| Pendulum | Angular Steering | Interpretation |
|----------|-----------------|----------------|
| Angle theta | phi_k (activation angle in steering plane) | How aligned is the activation with the feature direction |
| Angular velocity omega | omega_k = phi_{k+1} - phi_k | How fast the alignment is changing across layers |
| Time t | Layer index k | Depth through the transformer |
| Gravity g | Natural restoring dynamics of the model | Trained behavior pulls activations toward learned representations |
| Damping | Contractive dynamics (spectral decay) | Perturbations decay across layers due to low-rank bottleneck |
| Torque u | Rotation applied by Angular Steering | The behavioral intervention |
| Stable equilibrium (theta=0) | Aligned behavior (e.g., refusal angle) | The model's trained default behavior |
| Unstable equilibrium (theta=pi) | Opposite behavior (e.g., compliance) | The behavior we want to steer toward (or prevent) |
| Separatrix | Behavioral decision boundary | The critical trajectory separating safe from unsafe behavior |
| Swing-up | Jailbreaking | Moving from trained (refusal) to opposite (compliance) |
| Stabilization | Alignment maintenance | Keeping the model at the desired behavior |
| Energy E | Behavioral energy V(phi, omega) | A scalar characterizing how far from the target the system is |

### 2.4 Where the Analogy Holds and Where It Breaks

**Holds well:**
- The state space is genuinely SO(2) x R — angles that wrap around 2pi
- The dynamics are discrete-time — layers are discrete steps (like a sampled pendulum)
- There is a "restoring force" — models tend to self-correct toward trained behavior
- Phase portraits are well-defined and empirically computable

**Requires adaptation:**
- **No conservation law**: Transformer dynamics are contractive (damped), not conservative. This is a *damped* pendulum, not an ideal one. Energy-based methods must account for dissipation.
- **Layer-varying dynamics**: Each layer has different weights, so the "gravity" and "damping" change at each step. This is a pendulum with time-varying parameters.
- **Input-dependent**: The dynamics change with the input prompt. This is a pendulum where gravity depends on external conditions.
- **Coarse discretization**: Only 32-80 layers means the continuous-time approximation is rough.

**Key insight**: The differences do not invalidate the framework — they make it a **damped, time-varying, discrete pendulum**, which is a well-studied system in control theory. The SO(2) geometry and energy-based analysis carry through regardless.

---

## 3. Phase Portrait Analysis of Activation Dynamics

### 3.1 Extracting the Phase Portrait

For a given model and steering plane {b1, b2}:

1. Run N prompts (both harmful and harmless) through the model without steering
2. At each layer k, record phi_k = atan2(b2^T h_k, b1^T h_k) and r_k = ||proj_P(h_k)||
3. Compute omega_k = phi_{k+1} - phi_k (unwrapped to handle the 2pi boundary)
4. Plot the phase portrait: (phi_k, omega_k) for all prompts and all layers, color-coded by layer depth and prompt type

### 3.2 What We Expect to See

Based on existing observations (Angular Steering Fig. 4, Fig. 6):

**For harmless prompts:**
- phi_k should cluster around the "harmless" angle (negative projection on refusal direction)
- omega_k should be small (the angle is stable — the model is confident in its behavior)
- Phase portrait: tight cluster near a stable fixed point

**For harmful prompts (model refuses):**
- phi_k should cluster around the "refusal" angle (positive projection on refusal direction)
- omega_k should be small in later layers (refusal behavior stabilizes)
- Phase portrait: trajectories converge to a different fixed point

**For harmful prompts (model complies — jailbroken):**
- phi_k should transition from refusal to compliance across layers
- omega_k should be large during the transition
- Phase portrait: trajectories cross the region between the two fixed points

**The separatrix hypothesis:**
If the phase portrait shows a curve separating the basins of attraction of the refusal and compliance fixed points, this is the **behavioral separatrix**. Prompts whose phase trajectories cross this curve lead to behavioral transitions. The separatrix would be a natural decision boundary for safety classifiers.

### 3.3 Empirical Predictions

From the spectral geometry literature (Fernando & Guitchounts, 2026):
- Early layers are rotation-dominated (non-normal Jacobians) → expect large omega_k, diverse trajectories
- Late layers are near-symmetric → expect small omega_k, convergence to fixed points
- A cumulative low-rank bottleneck funnels perturbations → expect contractive dynamics overall

This predicts a phase portrait that looks like a **damped pendulum**: wide orbits in early layers that spiral inward toward fixed points in later layers.

### 3.4 Connection to Existing Observations

Angular Steering's Fig. 4 already shows that the scalar projection (= cos(phi_k)) of harmful and harmless activations diverge across layers. The phase portrait extends this by adding the derivative information (omega_k), providing a richer characterization of the dynamics.

The A-LQR paper's tracking error bound (Theorem 4.2) can be reinterpreted as bounding the phase trajectory — the closed-loop error ||delta_z_k|| corresponds to the distance from the target in phase space.

---

## 4. Energy-Based Control Framework

### 4.1 Defining Behavioral Energy

By analogy with the pendulum energy E = (1/2) I omega^2 - mgL cos(theta), define:

```
V(phi_k, omega_k) = (1/2) J omega_k^2 + U(phi_k)
```

where:
- **Kinetic term** (1/2) J omega_k^2: penalizes rapid angular changes across layers. Large omega_k means the behavior is unstable/transitioning. J is an inertia-like parameter.
- **Potential term** U(phi_k): penalizes deviation from the target angle phi*. Natural choice:
  ```
  U(phi_k) = kappa * (1 - cos(phi_k - phi*))
  ```
  This is periodic (respects the SO(2) topology), smooth, and has a unique minimum at phi_k = phi* within [-pi, pi]. kappa is a stiffness-like parameter.

### 4.2 Properties of the Behavioral Energy

- V >= 0, with V = 0 iff phi_k = phi* and omega_k = 0 (on target, stable)
- V is periodic in phi_k (handles angle wrapping naturally)
- V = kappa corresponds to the separatrix energy (the boundary between "staying near target" and "transitioning to opposite behavior")
- V > 2*kappa means the system is in the "rotational" regime (behavior has fully transitioned)

### 4.3 Energy-Based Controller Design

**Goal**: Design a control input u_k (the rotation to apply at layer k) such that V decreases monotonically across layers.

**Lyapunov approach**: Choose u_k such that:

```
Delta V = V(phi_{k+1}, omega_{k+1}) - V(phi_k, omega_k) <= -gamma * V(phi_k, omega_k)
```

for some decay rate gamma > 0. This guarantees exponential convergence to the target.

**Derivation**: The controlled dynamics are:

```
phi_{k+1} = F_k(phi_k) + u_k
omega_{k+1} = phi_{k+1} - phi_k = F_k(phi_k) + u_k - phi_k
```

where F_k is the natural layer dynamics (unknown but observable). The energy change is:

```
Delta V = (1/2) J (omega_{k+1}^2 - omega_k^2) + kappa (cos(phi_k - phi*) - cos(phi_{k+1} - phi*))
```

Setting d(Delta V)/d(u_k) = 0 and solving gives the energy-optimal control:

```
u_k* = phi* - F_k(phi_k) + damping_term(omega_k)
```

where the damping term dissipates kinetic energy:

```
damping_term(omega_k) = -c * omega_k    (linear damping, c > 0)
```

This yields the **energy-damping controller**:

```
u_k* = (phi* - F_k(phi_k)) - c * omega_k
```

**Interpretation:**
- First term (phi* - F_k(phi_k)): corrects the natural drift — steers toward the target
- Second term (-c * omega_k): damps oscillations — prevents overshooting

**Connection to PID**: This is equivalent to a PD controller (proportional + derivative) in the angular domain, but derived from energy principles rather than heuristic tuning. The PID steering paper's P and D terms correspond directly.

### 4.4 Estimating F_k (Natural Dynamics)

The natural layer dynamics F_k(phi_k) — how the angle evolves through layer k without steering — can be estimated:

**Option A: Online estimation**
At each layer k during steered inference, first observe the pre-steering activation to get phi_k, compute F_k(phi_k) from the layer's output, then apply u_k.

**Option B: Offline calibration**
Run a set of prompts through the model without steering, record (phi_k, phi_{k+1}) pairs at each layer, and fit a simple model:
```
F_k(phi) ≈ a_k * phi + b_k    (linear)
F_k(phi) ≈ phi + c_k * sin(phi - d_k)    (pendulum-like)
```

**Option C: Assume identity dynamics**
If F_k ≈ identity (angle doesn't change much per layer), then u_k* ≈ (phi* - phi_k) - c * omega_k, which is a pure PD controller on the angle.

### 4.5 Stability Guarantee

**Theorem (informal)**: If the energy-damping controller is applied at every layer with c > 0, and the natural dynamics F_k satisfy a Lipschitz condition, then:

```
V(phi_K, omega_K) <= (1 - gamma)^K * V(phi_1, omega_1)
```

for some gamma > 0 depending on c, kappa, J, and the Lipschitz constants.

This means the behavioral energy decreases exponentially across layers, guaranteeing convergence to the target behavior.

**Comparison with A-LQR's Theorem 4.2**: A-LQR bounds the tracking error in full d-dimensional space, requiring local linearity. Our bound operates in 1D (the angle) and uses energy monotonicity rather than linearization, making it simpler and potentially tighter.

---

## 5. SO(2) Formalism for Angular Steering

### 5.1 Why SO(2) Matters

Current Angular Steering uses atan2 to compute angles and applies rotation matrices. This works but has subtle issues:

- **Wrapping discontinuity**: The angle jumps from +pi to -pi, causing numerical issues in error computation
- **Euclidean error metrics**: Computing phi_k - phi* as a scalar difference is wrong when the angles are near +/-pi
- **Non-geodesic interpolation**: Linear interpolation between angles doesn't follow the shortest path on the circle

### 5.2 Lie-Algebraic Control

The proper framework is control on the Lie group SO(2):

**State**: g_k ∈ SO(2), represented as a 2x2 rotation matrix
```
g_k = [cos(phi_k), -sin(phi_k); sin(phi_k), cos(phi_k)]
```

**Error**: Computed in the Lie algebra so(2) (tangent space):
```
e_k = Log(g_target^{-1} g_k) = phi_k - phi* (mod 2pi, wrapped to [-pi, pi])
```

**Dynamics**: g_{k+1} = g_k * Exp(omega_k) * Exp(u_k), where Exp and Log are the SO(2) exponential and logarithm maps.

**Cost function** (for LQR on SO(2)):
```
J = sum_k [ q * ||Log(g_target^{-1} g_k)||^2 + r * ||u_k||^2 ]
```

This is geometrically correct: the cost penalizes the geodesic distance on SO(2), not the Euclidean distance.

### 5.3 Practical Impact

For most angles, the Euclidean and geodesic metrics agree. The SO(2) formalism matters when:
- Steering by large angles (> 90°) where wrapping artifacts occur
- Computing error signals for PID/LQR controllers near the +/-pi boundary
- Interpolating between different steering angles (e.g., for smooth behavioral transitions)

The implementation change is minimal: replace scalar angle differences with the wrapped difference `atan2(sin(phi_k - phi*), cos(phi_k - phi*))`.

---

## 6. Controller Design Catalogue

The pendulum analogy provides a catalogue of well-studied controllers, each applicable to a different steering scenario:

### 6.1 Stabilization (Maintaining Alignment)

**Goal**: Keep phi_k near phi* (e.g., keep the model refusing harmful prompts).

**Controller**: Linearized LQR on SO(2)

Near the equilibrium, sin(phi - phi*) ≈ phi - phi*, and the dynamics become linear:
```
delta_phi_{k+1} = a_k * delta_phi_k + u_k
```

The LQR gain K_k minimizes the quadratic cost and stabilizes the equilibrium. This is a principled version of the adaptive mode in Angular Steering.

**When to use**: Normal operation — maintaining trained behavior under potentially adversarial inputs.

### 6.2 Swing-Up (Behavioral Transition)

**Goal**: Move from phi_k near phi_refusal to phi_k near phi_comply (e.g., jailbreaking for red-teaming).

**Controller**: Astrom-Furuta energy pumping

```
u_k = -k_e * sign(omega_k * cos(phi_k - phi_target)) * (V(phi_k, omega_k) - V_target)
```

This pumps energy into the system when the angular velocity is aligned with the desired direction, and removes energy when misaligned. The parameter V_target is the energy of the target equilibrium.

**When to use**: Red-teaming, studying jailbreak mechanics, understanding the energy barrier between refusal and compliance.

### 6.3 Hybrid Controller (Swing-Up + Stabilization)

**Goal**: Transition between behaviors and then stabilize.

**Controller**: Switch between energy pumping and LQR based on proximity:
```
if |phi_k - phi_target| > epsilon:
    u_k = energy_pumping(phi_k, omega_k)     # swing up
else:
    u_k = -K * [phi_k - phi_target; omega_k]   # stabilize (LQR)
```

**When to use**: Controlled behavioral transitions that need to be smooth and stable.

### 6.4 Energy-Damping Controller (From Section 4.3)

**Goal**: Steer toward a target while preserving stability.

**Controller**: u_k = (phi* - F_k(phi_k)) - c * omega_k

**When to use**: General-purpose steering with stability guarantees. The default choice.

### 6.5 Barrier-Constrained Controller

**Goal**: Keep the behavioral energy below the separatrix (prevent behavioral transitions).

**Controller**: If V(phi_k, omega_k) approaches V_separatrix, apply corrective torque:
```
u_k = -k_barrier * max(0, V(phi_k, omega_k) - V_safe) * dV/d(phi_k)
```

This is the pendulum analogue of Control Barrier Functions (BarrierSteer, 2026).

**When to use**: Safety-critical deployment — prevent the model from ever crossing the behavioral separatrix.

---

## 7. Experimental Plan

### Experiment 1: Phase Portrait Extraction [Primary — validates the entire proposal]

**Method:**
1. For each model (Qwen2.5-{3B,7B,14B}, Llama-3.1-8B, Gemma-2-9B):
   - Run D_harmful (416 prompts) and D_harmless (512 prompts) through the model
   - At each layer k, compute phi_k and omega_k = phi_{k+1} - phi_k
   - Plot the phase portrait (phi_k, omega_k), color-coded by:
     - Layer depth (expect: wide orbits early, tight clusters late)
     - Prompt type (expect: two distinct basins of attraction)

2. Identify fixed points, basins of attraction, and candidate separatrices
3. Compare with the theoretical pendulum phase portrait

**Deliverable:** Figure showing phase portraits across models. If the separatrix structure exists, this is the paper's main result.

**GPU requirement:** Single forward pass per prompt, no steering needed. ~15 minutes on a single GPU.

### Experiment 2: Natural Dynamics Characterization

**Method:**
1. From the phase portrait data, fit the natural dynamics model:
   ```
   phi_{k+1} = F_k(phi_k, omega_k)
   ```
2. Test candidate models:
   - Linear: phi_{k+1} = a_k phi_k + b_k
   - Pendulum-like: phi_{k+1} = phi_k + omega_k + c_k sin(phi_k - d_k)
   - Damped pendulum: phi_{k+1} = phi_k + omega_k, omega_{k+1} = alpha_k omega_k - beta_k sin(phi_k - gamma_k)
3. Compare R^2 and prediction error across models

**Deliverable:** Table showing which dynamics model best fits the empirical data. If the damped pendulum model fits well, the analogy is strongly validated.

### Experiment 3: Energy Landscape Mapping

**Method:**
1. Compute V(phi_k, omega_k) = (1/2) J omega_k^2 + kappa (1 - cos(phi_k - phi*)) for all prompts and layers
2. Plot energy evolution across layers for harmful vs harmless prompts
3. Identify the energy level of the separatrix (if it exists)
4. Test: does the energy reliably distinguish harmful from harmless prompts?

**Deliverable:** Energy landscape figure + separatrix energy level + ROC curve for energy-based safety classification.

### Experiment 4: Energy-Damping Controller vs Baselines

**Method:**
1. Implement the energy-damping controller: u_k = (phi* - F_k(phi_k)) - c * omega_k
2. Compare with:
   - Angular Steering (fixed angle)
   - PID-AcT (layer-wise PID)
   - A-LQR (scalar setpoint LQR)
3. Evaluate on refusal steering (AdvBench) and general capability (TinyBenchmarks, perplexity)
4. Sweep c (damping coefficient) to explore the stability-effectiveness tradeoff

**Deliverable:** Comparison table + Pareto curves (steering effectiveness vs capability preservation).

### Experiment 5: Separatrix-Based Safety Classifier

**Method:**
1. From the phase portrait, estimate the separatrix (boundary between safe and unsafe basins)
2. At inference time, monitor (phi_k, omega_k) and flag when the trajectory approaches the separatrix
3. Compare detection accuracy with:
   - LlamaGuard3 (post-hoc classifier)
   - The Geometry of Harmful Intent (angular deviation, arXiv:2603.27412, reports AUROC >= 0.937)
   - Probing classifiers on raw activations

**Deliverable:** AUROC comparison showing whether phase-space monitoring adds value over existing detectors.

### Experiment 6: Swing-Up Analysis (Jailbreaking Mechanics)

**Method:**
1. Apply Angular Steering at various angles, recording the full phase trajectory
2. Identify the critical angle at which the trajectory crosses the separatrix (= minimum jailbreaking strength)
3. Compare with the Astrom-Furuta critical acceleration ratio: if max_torque/gravity > 2, one-swing sufficiency
4. Test the energy-pumping controller for controlled behavioral transitions

**Deliverable:** Analysis of the "energy barrier" between refusal and compliance, quantified in terms of the pendulum analogy.

### Experiment 7: SO(2) vs Euclidean Error Metrics

**Method:**
1. Implement SO(2)-correct error computation (wrapped angular difference via atan2)
2. Compare with Euclidean error (simple scalar difference) in the PID and energy-damping controllers
3. Measure: does the SO(2) formulation improve stability at large steering angles?

**Deliverable:** Ablation showing the practical impact of geometric correctness.

---

## 8. Theoretical Contributions

### 8.1 Phase Portrait Theorem

**Theorem (informal):** For a transformer with L layers and a steering plane P, the activation angle phi_k and angular velocity omega_k define a discrete dynamical system on SO(2) x R. Under mild conditions on the layer-wise dynamics, this system has:
- At least two fixed points corresponding to the contrastive behaviors
- A separatrix curve in the (phi, omega) plane separating their basins of attraction
- Contractive dynamics in late layers (eigenvalues inside the unit circle)

### 8.2 Energy Monotonicity Theorem

**Theorem:** Under the energy-damping controller u_k = (phi* - F_k(phi_k)) - c * omega_k with c > 0, if the natural dynamics satisfy ||F_k(phi) - phi|| <= L_k for all phi, then:

```
V_{k+1} <= (1 - gamma(c, L_k)) * V_k
```

where gamma > 0 for sufficiently large c. This guarantees exponential convergence of the activation angle to the target.

### 8.3 Separatrix Energy Bound

**Theorem:** The energy of the separatrix V_sep provides a safety certificate: if V(phi_k, omega_k) < V_sep for all layers k, then the activation trajectory remains in the basin of attraction of the target behavior.

### 8.4 Connection to Existing Methods

**Proposition:** Angular Steering with fixed angle theta is equivalent to the energy-damping controller with:
- F_k = identity (no natural dynamics correction)
- c = 0 (no damping)
- u_k = theta - phi_k (pure proportional control)

PID-AcT adds integral and derivative terms but in the linear (Euclidean) domain. The energy-damping controller operates in the SO(2) domain with explicit energy guarantees.

---

## 9. Implementation Plan

### Phase 1: Phase Portrait Extraction [Week 1-2]
- Extend `pytorch_pure/extract_directions.py` to compute phi_k and omega_k at each layer
- Create `phase_portrait.py` for visualization
- Run on all 6 models
- **Decision point:** If phase portraits show separatrix structure, continue. If not, pivot.

### Phase 2: Dynamics Fitting [Week 2-3]
- Implement dynamics model candidates (linear, pendulum, damped pendulum)
- Fit from phase portrait data
- Validate prediction accuracy

### Phase 3: Energy Analysis [Week 3-4]
- Implement behavioral energy V(phi, omega)
- Map energy landscapes
- Test energy-based classification

### Phase 4: Controller Implementation [Week 4-6]
- Implement energy-damping controller in `angular_steering.py`
- Implement SO(2) error computation
- Integrate with vLLM hooks

### Phase 5: Evaluation [Week 6-8]
- Run all experiments from Section 7
- Generate paper figures

### Phase 6: Writing [Week 8-12]
- Target venue: ICML 2027 or ICLR 2027

---

## 10. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `phase_portrait.py` | Create | Extract and visualize (phi_k, omega_k) phase portraits |
| `dynamics_model.py` | Create | Fit natural dynamics F_k from empirical data |
| `energy_controller.py` | Create | Energy-damping controller with SO(2) error computation |
| `angular_steering.py` | Modify | Add per-layer adaptive angle from energy controller |
| `pytorch_pure/extract_directions.py` | Modify | Add angular trajectory extraction |
| `visualization/` | Extend | Phase portraits, energy landscapes, separatrix plots |

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Phase portraits show no separatrix | Medium | Fatal | Run Experiment 1 first (2 weeks). Pivot if fails. |
| Dynamics are too noisy for pendulum model | Medium | Moderate | Use robust fitting, average over prompts, try nonlinear models |
| Energy-damping controller doesn't outperform PID | Medium | Moderate | The contribution would shift to analysis (phase portraits, energy landscapes) rather than a new controller |
| SO(2) formalism doesn't improve over Euclidean | High | Low | Minor contribution; the phase portrait analysis is the main result |
| Reviewers reject the pendulum analogy as superficial | Medium | Moderate | Frame as "control on SO(2)" not "pendulum"; emphasize the formal results |

---

## 12. Why This Works as a Paper

### Narrative arc:
1. Angular Steering operates on SO(2) but treats it as Euclidean → missed structure
2. We extract phase portraits revealing pendulum-like dynamics → the structure exists
3. We define behavioral energy with a separatrix → safety has a geometric meaning
4. We design an energy-damping controller with stability guarantees → principled steering
5. The controller outperforms heuristic methods → practical value

### Novelty:
- **First phase portrait analysis** of LLM activation dynamics in the steering plane
- **First energy-based stability guarantee** for activation steering
- **First SO(2)-correct controller** for angular steering
- **Separatrix as safety boundary** — a new geometric interpretation of alignment

### Practical value:
- The energy-damping controller is as cheap as PID (one scalar computation per layer)
- No Jacobians needed (unlike A-LQR)
- No offline training needed (unlike BarrierSteer)
- The separatrix provides a real-time safety monitor at zero cost

---

## 13. References

### Core Papers
- Vu & Nguyen (2025). Angular Steering: Behavior Control via Rotation in Activation Space. NeurIPS 2025.
- Nguyen et al. (2025). Activation Steering with a Feedback Controller. ICLR 2026. arXiv:2510.04309.
- Skifstad et al. (2026). Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control. arXiv:2604.19018.

### Dynamical Systems / Spectral Analysis
- Fernando & Guitchounts (2026). Dynamics of the Transformer Residual Stream. arXiv:2605.14258.
- Fernando et al. (2025). Transformer Dynamics: A Neuroscientific Approach. arXiv:2502.12131.
- Nettasinghe & Joseph (2026). As Language Models Scale, Low-order Linear Depth Dynamics Emerge. arXiv:2603.12541.

### Control on SO(2) / Pendulum Control
- Torgesen. Optimal Linear Control on the SO(2) Manifold Using Lie Algebras.
- Astrom & Furuta (2000). Swinging Up a Pendulum by Energy Control. Automatica.
- MPC on differentiable manifolds. arXiv:2106.15233.

### Safety / Barrier Methods
- Zhao et al. (2026). ODESteer: A Unified ODE-Based Steering Framework. ICLR 2026.
- Tran et al. (2026). BarrierSteer: LLM Safety via Learning Barrier Steering. arXiv:2602.20102.
- Mishra et al. (2026). Steered LLM Activations are Non-Surjective. arXiv:2604.09839.

### Trajectory Analysis
- Microsoft (2026). LLM Reasoning as Trajectories. arXiv:2604.05655.
- Zhang et al. (2026). Truth as a Trajectory. arXiv:2603.01326.
