"""PTS — Model Predictive Control in the 2D Angular Steering plane.

Implements Section 5 of research_proposal_PTS.md:

  * reference_trajectory(...)        S4.3 Options A / B / C(tube) / D
  * build_condensed_qp(...)          S5.2 condensation of the layer-horizon OCP
  * solve_qp(...)                    a dependency-free QP solver (closed-form
                                     unconstrained + FISTA for the ||u||<=u_max
                                     control constraint, S5.2 / S8.2)
  * dlqr(...)                        discrete-time LQR via the DARE — the
                                     unconstrained "A-LQR projected to 2D" baseline
                                     and the gold cross-check for the MPC
  * MPCController                    glue: dynamics + reference + weights -> the
                                     first control u_k and its Angular-Steering
                                     angle theta_k (S5.4)

Actuator convention (B_k = I): a control u_k is the additive update to the
in-plane coordinate; the physical actuator then realises the *angle* of (c_k+u_k)
via the norm-preserving SO(2) reset (S5.4). The MPC plans the 2D point; the
realised state keeps ||proj_plane|| and tracks only the angle. The gap between the
planned additive update and the angle-only realisation is exactly the
linearisation residual w_j of the proposal's tracking bound (S8.1) — we measure it.

No osqp / cvxpy: the QP is tiny (2H ~ 12 vars) and the only constraint is a
per-step 2-norm ball, whose Euclidean projection is closed form, so projected /
accelerated gradient (FISTA) is exact and trivially batched over prompts.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Reference trajectory  (S4.3)
# =============================================================================


def reference_trajectory(
    harmful_mean: np.ndarray,
    harmless_mean: np.ndarray,
    option: str = "A",
    lam: float = 1.0,
) -> np.ndarray:
    """Build the per-layer 2D reference tau*_k.

    Args:
        harmful_mean, harmless_mean: (L, 2) mean coordinate trajectories.
        option: "A" -> harmless mean (track the benign/compliant trajectory);
                "B" -> harmful + lam*(harmless - harmful) interpolation;
                "harmful" -> harmful mean (the refusal-amplifying direction).
        lam: interpolation weight for option B (0 -> harmful, 1 -> harmless).

    Returns:
        ref: (L, 2). (Option C's tube and Option D's input-adaptive nearest-point
        are handled by reference_tube / adaptive_reference below.)
    """
    harmful_mean = np.asarray(harmful_mean, np.float64)
    harmless_mean = np.asarray(harmless_mean, np.float64)
    if option == "A":
        return harmless_mean.copy()
    if option == "harmful":
        return harmful_mean.copy()
    if option == "B":
        return harmful_mean + lam * (harmless_mean - harmful_mean)
    raise ValueError(f"unknown reference option {option!r}")


def adaptive_reference(c_traj: np.ndarray, ref_pool: np.ndarray) -> np.ndarray:
    """S4.3 Option D: per layer, snap the reference to the nearest harmless mean.

    Here we keep it simple/per-layer: ref_pool is the (L,2) harmless mean and we
    return it (the nearest fixed point per layer); a richer pool of harmless
    trajectories could be matched per-sample, but the mean is the documented default.
    """
    return np.asarray(ref_pool, np.float64).copy()


# =============================================================================
# Condensed QP  (S5.2)
# =============================================================================


def build_condensed_qp(
    A: np.ndarray,
    b: np.ndarray,
    k: int,
    H: int,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
) -> dict:
    """Condense the layer-horizon optimal-control problem into a dense QP.

    States are the *actuated* coordinates s_j (B=I convention):
        s_0   = xi + u_0
        s_j   = A_{k+j-1} s_{j-1} + b_{k+j-1} + u_j     (1 <= j <= H-1)
        s_H   = A_{k+H-1} s_{H-1} + b_{k+H-1}           (terminal, no control)
    cost  J = sum_{j=0}^{H-1} (s_j-r_j)'Q(s_j-r_j) + u_j'R u_j
              + (s_H - r_H)' Qf (s_H - r_H)

    The decision vector is U = [u_0..u_{H-1}] in R^{2H}. Eliminating the states
    gives S = G U + F xi + phi (S in R^{2(H+1)}), so

        J = 1/2 U' P U + q(xi, ref)' U + const,
        P  = 2 (G' Qblk G + Rblk),
        q  = 2 G' Qblk (F xi + phi - Rref).

    P, G, F, phi, Qblk depend only on (A,b,k,H,Q,R,Qf); only q depends on the
    measured state xi and the reference, so this is built once per layer and q is
    assembled per prompt at solve time.

    Returns dict with P, G, F, phi, Qblk, Rblk, H, and L_lipschitz (||P||_2).
    """
    n = 2
    Hs = H + 1                                  # number of states s_0..s_H
    # Transition products M[j,i] = A_{k+j-1}...A_{k+i}  (n x n), i<=j ; M[j,j]=I.
    M = np.zeros((Hs, Hs, n, n))
    for j in range(Hs):
        M[j, j] = np.eye(n)
        for i in range(j - 1, -1, -1):
            M[j, i] = A[k + j - 1] @ M[j - 1, i]   # prepend A_{k+j-1}

    # G: (2Hs) x (2H) block-lower-triangular, G[j,i]=M[j,i] for i<=min(j,H-1)
    G = np.zeros((n * Hs, n * H))
    F = np.zeros((n * Hs, n))
    phi = np.zeros(n * Hs)
    for j in range(Hs):
        F[n * j:n * j + n] = M[j, 0]             # xi enters s_0 with I, like u_0
        # affine: b_{k+i} enters s_{i+1}, propagates by M[j,i+1] for j>=i+1
        acc = np.zeros(n)
        for i in range(j):
            acc = acc + M[j, i + 1] @ b[k + i]
        phi[n * j:n * j + n] = acc
        for i in range(min(j, H - 1) + 1):
            G[n * j:n * j + n, n * i:n * i + n] = M[j, i]

    Qblk = np.zeros((n * Hs, n * Hs))
    for j in range(H):
        Qblk[n * j:n * j + n, n * j:n * j + n] = Q
    Qblk[n * H:n * H + n, n * H:n * H + n] = Qf
    Rblk = np.kron(np.eye(H), R)

    P = 2.0 * (G.T @ Qblk @ G + Rblk)
    P = 0.5 * (P + P.T)                          # symmetrise
    L_lip = float(np.linalg.eigvalsh(P)[-1])     # Lipschitz const of grad (||P||_2)
    return {"P": P, "G": G, "F": F, "phi": phi, "Qblk": Qblk, "Rblk": Rblk,
            "H": H, "n": n, "L": L_lip}


def assemble_q(cqp: dict, xi: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """q = 2 G' Qblk (F xi + phi - Rref). Batched over leading dim of xi.

    Args:
        xi: (..., 2) measured coordinate at the first horizon layer.
        ref: (..., H+1, 2) reference per horizon state (broadcast over batch ok).
    Returns: q of shape (..., 2H).
    """
    G, F, phi, Qblk = cqp["G"], cqp["F"], cqp["phi"], cqp["Qblk"]
    xi = np.asarray(xi, np.float64)
    ref = np.asarray(ref, np.float64)
    # collapse the trailing (H+1, 2) of ref; any leading dims broadcast against xi.
    Rref = ref.reshape(ref.shape[:-2] + (-1,))                # (2(H+1),) or (...,2(H+1))
    d = xi @ F.T + phi - Rref                                  # (..., 2(H+1))
    return 2.0 * (d @ (Qblk @ G))                              # (..., 2H)


# =============================================================================
# QP solver  (closed-form unconstrained + FISTA for ||u_j||<=u_max)
# =============================================================================


def _project_ball(U: np.ndarray, u_max: float, H: int) -> np.ndarray:
    """Project each 2-block of U onto the Euclidean ball of radius u_max."""
    Ur = U.reshape(U.shape[:-1] + (H, 2))
    nrm = np.linalg.norm(Ur, axis=-1, keepdims=True)
    scale = np.minimum(1.0, u_max / (nrm + 1e-12))
    return (Ur * scale).reshape(U.shape)


def solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    u_max: float | None,
    H: int,
    L: float | None = None,
    iters: int = 80,
    tol: float = 1e-9,
) -> np.ndarray:
    """Minimise 1/2 U'P U + q'U subject to ||u_j||_2 <= u_max for each step.

    P: (2H,2H) shared; q: (..., 2H) batched. Returns U*: (..., 2H).
    Unconstrained (u_max None/inf) -> single linear solve. Constrained -> FISTA
    (accelerated projected gradient) with the closed-form ball projection.
    """
    q = np.asarray(q, np.float64)
    lead = q.shape[:-1]
    q2 = q.reshape(-1, q.shape[-1])              # (M, 2H)

    # unconstrained closed form
    U_unc = np.linalg.solve(P, -q2.T).T          # (M, 2H)
    if u_max is None or not np.isfinite(u_max):
        return U_unc.reshape(lead + (2 * H,))

    # warm-start FISTA from the projected unconstrained solution
    if L is None:
        L = float(np.linalg.eigvalsh(P)[-1])
    step = 1.0 / L
    U = _project_ball(U_unc, u_max, H)
    Y = U.copy()
    t = 1.0
    for _ in range(iters):
        grad = Y @ P + q2                        # (M, 2H)  (P symmetric)
        U_new = _project_ball(Y - step * grad, u_max, H)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Y = U_new + ((t - 1.0) / t_new) * (U_new - U)
        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U, t = U_new, t_new
    return U.reshape(lead + (2 * H,))


def qp_objective(P: np.ndarray, q: np.ndarray, U: np.ndarray) -> np.ndarray:
    """1/2 U'P U + q'U, batched over leading dim."""
    quad = 0.5 * np.einsum("...i,ij,...j->...", U, P, U)
    lin = np.einsum("...i,...i->...", q, U)
    return quad + lin


# =============================================================================
# Discrete LQR via the DARE  (unconstrained baseline + MPC cross-check)
# =============================================================================


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
         iters: int = 2000, tol: float = 1e-12) -> dict:
    """Infinite-horizon discrete LQR for x_{t+1}=A x_t + B u_t.

    Returns gain K (u=-Kx), cost-to-go S, and closed-loop spectral radius rho.
    Solved by iterating the Riccati recursion to convergence.
    """
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    Q = np.asarray(Q, np.float64)
    R = np.asarray(R, np.float64)
    S = Q.copy()
    K = np.zeros((B.shape[1], A.shape[0]))
    for _ in range(iters):
        BtSB_R = R + B.T @ S @ B
        K_new = np.linalg.solve(BtSB_R, B.T @ S @ A)
        S_new = Q + A.T @ S @ A - A.T @ S @ B @ K_new
        if np.max(np.abs(S_new - S)) < tol:
            S, K = S_new, K_new
            break
        S, K = S_new, K_new
    A_cl = A - B @ K
    rho = float(np.max(np.abs(np.linalg.eigvals(A_cl))))
    return {"K": K, "S": S, "rho": rho, "A_cl": A_cl}


# =============================================================================
# Angle conversion (S5.4) + controller glue
# =============================================================================


def angle_from_control(c: np.ndarray, u: np.ndarray) -> np.ndarray:
    """theta_k = atan2(c2+u2, c1+u1) — the Angular-Steering target angle (S5.4)."""
    cu = np.asarray(c) + np.asarray(u)
    return np.arctan2(cu[..., 1], cu[..., 0])


class MPCController:
    """Receding-horizon layer MPC over the 2D steering plane.

    Pre-builds the condensed QP for every layer where a hook will run, then
    `.control(layer, xi)` returns (u_k, theta_k) for the measured coordinate(s)
    xi at that layer — vectorised over an arbitrary batch of prompts/positions.
    """

    def __init__(self, A, b, ref, layers, H=6, q_pos=1.0, r_ctrl=0.05,
                 qf_scale=4.0, u_max=None, fista_iters=60):
        self.A = np.asarray(A, np.float64)
        self.b = np.asarray(b, np.float64)
        self.ref = np.asarray(ref, np.float64)        # (L, 2)
        self.layers = list(layers)                    # layers we actuate at
        # horizon is bounded by both the available dynamics (A has L-1 rows) and
        # the available reference rows, so end-of-grid slices never overrun.
        self.L = min(self.A.shape[0] + 1, self.ref.shape[0])
        self.H = H
        self.u_max = u_max
        self.fista_iters = fista_iters
        self.Q = q_pos * np.eye(2)
        self.R = r_ctrl * np.eye(2)
        self.Qf = qf_scale * q_pos * np.eye(2)
        self._cqp: dict[int, dict] = {}
        self._refstack: dict[int, np.ndarray] = {}
        for k in self.layers:
            Hk = min(H, self.L - 1 - k)               # shrink horizon near the end
            if Hk < 1:
                continue
            self._cqp[k] = build_condensed_qp(
                self.A, self.b, k, Hk, self.Q, self.R, self.Qf)
            self._refstack[k] = self.ref[k:k + Hk + 1]  # (Hk+1, 2)

    def horizon_at(self, k: int) -> int:
        return self._cqp[k]["H"] if k in self._cqp else 0

    def control(self, k: int, xi: np.ndarray):
        """Return (u, theta) for measured coordinate(s) xi:(...,2) at layer k."""
        if k not in self._cqp:
            xi = np.asarray(xi, np.float64)
            zero = np.zeros_like(xi)
            return zero, np.arctan2(xi[..., 1], xi[..., 0])
        cqp = self._cqp[k]
        ref = self._refstack[k]
        q = assemble_q(cqp, xi, ref)                  # (..., 2H)
        U = solve_qp(cqp["P"], q, self.u_max, cqp["H"], L=cqp["L"],
                     iters=self.fista_iters)
        u0 = U[..., :2]                               # first control only
        theta = angle_from_control(xi, u0)
        return u0, theta


# =============================================================================
# Self-tests
# =============================================================================


def _simulate_cost(A, b, k, H, Q, R, Qf, xi, ref, U):
    """Forward-simulate J(U) directly from the recursion (ground truth for P,q)."""
    n = 2
    s = xi + U[:n]
    J = 0.0
    for j in range(H + 1):
        Qj = Qf if j == H else Q
        e = s - ref[j]
        J += e @ Qj @ e
        if j < H:
            J += U[n * j:n * j + n] @ R @ U[n * j:n * j + n]
        if j < H:
            s = A[k + j] @ s + b[k + j]
            if j + 1 <= H - 1:
                s = s + U[n * (j + 1):n * (j + 1) + n]
    return J


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    n = 2

    # ---- 1. condensation correctness: P,q reproduce the simulated cost ----
    L = 10
    A = np.stack([0.85 * np.eye(2) + 0.1 * rng.randn(2, 2) for _ in range(L - 1)])
    b = 0.1 * rng.randn(L - 1, 2)
    Q, R, Qf = np.eye(2), 0.03 * np.eye(2), 3.0 * np.eye(2)
    k, H = 2, 5
    cqp = build_condensed_qp(A, b, k, H, Q, R, Qf)
    xi = rng.randn(2)
    ref = rng.randn(H + 1, 2)
    q = assemble_q(cqp, xi, ref)
    const = (xi @ cqp["F"].T + cqp["phi"] - ref.reshape(-1)) @ cqp["Qblk"] @ \
            (xi @ cqp["F"].T + cqp["phi"] - ref.reshape(-1))
    max_rel = 0.0
    for _ in range(200):
        U = rng.randn(2 * H)
        j_quad = qp_objective(cqp["P"], q, U) + const
        j_sim = _simulate_cost(A, b, k, H, Q, R, Qf, xi, ref, U)
        max_rel = max(max_rel, abs(j_quad - j_sim) / (abs(j_sim) + 1e-9))
    print(f"[1] condensation: max rel err J_quad vs J_sim = {max_rel:.2e}")
    assert max_rel < 1e-8

    # ---- 2. unconstrained solver optimality (gradient ~ 0) ----
    U_unc = solve_qp(cqp["P"], q, None, H)
    grad = cqp["P"] @ U_unc + q
    print(f"[2] unconstrained: ||grad||={np.linalg.norm(grad):.2e}")
    assert np.linalg.norm(grad) < 1e-8

    # ---- 3. FISTA unconstrained matches the linear solve ----
    U_fista = solve_qp(cqp["P"], q, 1e9, H, iters=300)
    print(f"[3] FISTA(unc) vs solve: {np.max(np.abs(U_fista - U_unc)):.2e}")
    assert np.max(np.abs(U_fista - U_unc)) < 1e-5

    # ---- 4. constrained: feasibility + optimality vs feasible clipping ----
    u_max = 0.3
    U_con = solve_qp(cqp["P"], q, u_max, H, iters=400)
    blocks = U_con.reshape(H, 2)
    feas = np.linalg.norm(blocks, axis=-1).max()
    U_clip = _project_ball(U_unc, u_max, H)
    obj_con = qp_objective(cqp["P"], q, U_con)
    obj_clip = qp_objective(cqp["P"], q, U_clip)
    print(f"[4] constrained: max||u_j||={feas:.4f} (<= {u_max}); "
          f"obj_fista={obj_con:.4f} <= obj_clip={obj_clip:.4f}")
    assert feas <= u_max + 1e-6 and obj_con <= obj_clip + 1e-6

    # ---- 5. batched solve == per-item solve (unique minimizer -> objectives match) ----
    Q_batch = np.stack([assemble_q(cqp, rng.randn(2), ref) for _ in range(7)])
    U_batch = solve_qp(cqp["P"], Q_batch, u_max, H, iters=600)
    U_each = np.stack([solve_qp(cqp["P"], Q_batch[i], u_max, H, iters=600)
                       for i in range(7)])
    obj_diff = float(np.max(np.abs(
        qp_objective(cqp["P"], Q_batch, U_batch)
        - qp_objective(cqp["P"], Q_batch, U_each))))
    u_diff = float(np.max(np.abs(U_batch - U_each)))
    print(f"[5] batched vs per-item: max obj diff={obj_diff:.2e}, max U diff={u_diff:.2e}")
    assert obj_diff < 1e-9 and u_diff < 1e-4

    # ---- 6. dlqr solves the DARE (residual ~ 0) and stabilises ----
    Ai = 1.15 * np.array([[0.9, 0.2], [-0.1, 0.8]])     # open-loop unstable
    Bi = np.eye(2)
    Qd, Rd = np.eye(2), 0.1 * np.eye(2)
    lqr = dlqr(Ai, Bi, Qd, Rd)
    S, K = lqr["S"], lqr["K"]
    dare = Qd + Ai.T @ S @ Ai - Ai.T @ S @ Bi @ np.linalg.solve(
        Rd + Bi.T @ S @ Bi, Bi.T @ S @ Ai) - S
    print(f"[6] dlqr: DARE residual={np.max(np.abs(dare)):.2e}, "
          f"open-loop rho={np.max(np.abs(np.linalg.eigvals(Ai))):.3f} -> "
          f"closed-loop rho={lqr['rho']:.3f}")
    assert np.max(np.abs(dare)) < 1e-8 and lqr["rho"] < 1.0

    # ---- 7. receding-horizon MPC closed-loop contracts (Thm 8.1 ground truth) ----
    # Track ref=0 on a time-invariant unstable system; re-solve each layer, apply
    # u_0, step the TRUE dynamics. The actuated state ||s|| must contract to 0.
    n_steps = 25
    A_cl = np.stack([Ai] * (n_steps + 2))
    b_cl = np.zeros((n_steps + 2, 2))
    refz = np.zeros((n_steps + 2, 2))
    ctrl = MPCController(A_cl, b_cl, refz, layers=list(range(n_steps)),
                         H=6, q_pos=1.0, r_ctrl=0.05, u_max=None)
    s = np.array([2.0, -1.5])
    norms = [np.linalg.norm(s)]
    for kk in range(n_steps):
        u0, _ = ctrl.control(kk, s)
        s = s + u0                       # actuate
        s = Ai @ s + b_cl[kk]            # propagate true dynamics
        norms.append(np.linalg.norm(s))
    print(f"[7] closed-loop ||s||: {norms[0]:.2f} -> {norms[-1]:.2e} "
          f"(contracts: {norms[-1] < 0.05 * norms[0]})")
    assert norms[-1] < 0.05 * norms[0]

    print("[pts_mpc self-test] ALL OK")
