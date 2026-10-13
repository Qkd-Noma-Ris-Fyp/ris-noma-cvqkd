"""
ris_optimizer.py
=================
Chooses the RIS reflection phase-shift vector theta = [e^{j*theta_1}, ..., e^{j*theta_N}].

Two methods are provided:

1. `phase_alignment_heuristic`  (always available, no extra dependencies)
   Aligns each RIS element's phase to coherently add the BS->RIS->user
   path for a *target* user (or the sum of a weighted set of users).
   This is the standard low-complexity baseline used in most RIS papers
   before moving to a joint optimization.

2. `optimize_sum_rate_cvxpy`  (optional, needs `pip install cvxpy`)
   A placeholder for the joint active/passive beamforming problem your
   proposal mentions (SCA / SDP-based optimization). It solves a
   semidefinite-relaxation (SDR) of sum-rate maximization for a SINGLE
   representative user direction as a starting point -- extend this to
   the full multi-user weighted sum-rate objective once your system
   model is finalized with your guide.
"""

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


def phase_alignment_heuristic(H1, h2, hd=None, beamformer=None):
    """Pick RIS phases that maximize the coherent combination for one user.

    theta_n is chosen so that each reflected path arrives in phase with
    the (optional) direct path, maximizing |sum of paths|.

    Parameters
    ----------
    H1 : (Nr, Nt) BS->RIS channel
    h2 : (Nr,) RIS->user channel
    hd : (Nt,) direct BS->user channel, optional
    beamformer : (Nt,) transmit beamformer, defaults to equal-gain

    Returns
    -------
    theta : (Nr,) complex array, unit modulus
    """
    Nt = H1.shape[1]
    if beamformer is None:
        beamformer = np.ones(Nt) / np.sqrt(Nt)

    # Effective per-element cascaded gain BEFORE applying RIS phase:
    # cascaded_n = h2_n^* * (H1 @ beamformer)_n
    bs_to_ris_combined = H1 @ beamformer          # (Nr,)
    cascaded = np.conj(h2) * bs_to_ris_combined   # (Nr,)

    direct_phase = 0.0
    if hd is not None:
        direct_term = hd @ beamformer
        direct_phase = np.angle(direct_term)

    # Choose theta_n to cancel the cascaded path's own phase and align
    # it with the direct path's phase (constructive combination).
    theta_angles = direct_phase - np.angle(cascaded)
    theta = np.exp(1j * theta_angles)
    return theta


def optimize_sum_rate_cvxpy(H1, h2, hd=None, beamformer=None, max_iters=50):
    """SDR-based passive beamforming for a single user's SNR maximization.

    This is intentionally a SIMPLE starting point (single-user SDR),
    not the full joint multi-user SCA/SDP problem from advanced RIS-NOMA
    papers. It demonstrates the CVXPY workflow your proposal names, so
    you can extend the objective (e.g., weighted sum-rate, per-user QoS
    constraints) once the system model is finalized with your guide.
    """
    if not _HAS_CVXPY:
        raise ImportError(
            "cvxpy is not installed in this environment. Run "
            "'pip install cvxpy' locally, then re-run this function."
        )

    Nt = H1.shape[1]
    Nr = H1.shape[0]
    if beamformer is None:
        beamformer = np.ones(Nt) / np.sqrt(Nt)

    bs_to_ris_combined = H1 @ beamformer            # (Nr,)
    a = np.conj(h2) * bs_to_ris_combined            # (Nr,) cascaded coefficient
    d = 0.0 if hd is None else hd @ beamformer      # direct-path scalar

    # We want to maximize |a^T theta + d|^2 subject to |theta_n| = 1.
    # Standard SDR lifting: v = [theta; 1], R = v v^H (rank-1, PSD),
    # objective becomes trace(C R) with C built from [a; d].
    c = np.concatenate([a, [d]])
    C = np.outer(c, np.conj(c))

    R = cp.Variable((Nr + 1, Nr + 1), hermitian=True)
    constraints = [R >> 0]
    for n in range(Nr + 1):
        constraints.append(R[n, n] == 1)

    objective = cp.Maximize(cp.real(cp.trace(C @ R)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, max_iters=max_iters)

    # Gaussian randomization to extract a feasible rank-1 solution
    eigvals, eigvecs = np.linalg.eigh(R.value)
    eigvals = np.clip(eigvals, 0, None)
    best_theta, best_val = None, -np.inf
    for _ in range(200):
        z = np.random.randn(Nr + 1) + 1j * np.random.randn(Nr + 1)
        v = eigvecs @ (np.sqrt(eigvals) * z)
        v = v / v[-1]                # normalize so last entry is 1
        theta_candidate = np.exp(1j * np.angle(v[:Nr]))
        val = np.abs(a @ theta_candidate + d) ** 2
        if val > best_val:
            best_val, best_theta = val, theta_candidate
    return best_theta


def coordinate_ascent_weighted_sum(H1, H2_list, Hd_list, weights=None, beamformer=None,
                                    n_outer=15, grid_size=180, theta_init=None, seed=None):
    """Multi-user RIS phase optimization via coordinate ascent (no external solver).

    For each RIS element n, holding every other element's phase fixed, the
    weighted sum objective  f(theta) = sum_k w_k |g_k(theta)|^2  is evaluated
    over a fine grid of candidate phases for element n, and the best one is
    kept. Sweeping all N elements this way is GUARANTEED not to decrease the
    objective each outer iteration (each per-element step is exact-optimal
    over the grid), so it converges monotonically to a local optimum -- a
    standard, citable technique in the RIS literature (e.g. Wu & Zhang,
    "Intelligent Reflecting Surface Enhanced Wireless Network", 2019) that
    needs no semidefinite solver.

    Parameters
    ----------
    H1 : (Nr, Nt) BS->RIS channel
    H2_list : list of K arrays, each (Nr,) -- RIS->user_k channel
    Hd_list : list of K arrays, each (Nt,) -- direct BS->user_k channel
    weights : (K,) per-user objective weights, defaults to equal weights
    beamformer : (Nt,) transmit beamformer, defaults to equal-gain
    n_outer : number of full sweeps over all N elements
    grid_size : number of candidate phases tested per element per sweep
    theta_init : (Nr,) initial phase vector; defaults to all-ones (RIS "off")

    Returns
    -------
    theta : (Nr,) complex array, unit modulus -- the optimized phases
    history : (n_outer+1,) array of the objective value after each sweep
              (history[0] is the objective at theta_init) -- plot this to
              show/prove convergence.
    """
    Nt = H1.shape[1]
    Nr = H1.shape[0]
    K = len(H2_list)
    if beamformer is None:
        beamformer = np.ones(Nt) / np.sqrt(Nt)
    if weights is None:
        weights = np.ones(K)
    weights = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(seed)

    bs_to_ris_combined = H1 @ beamformer  # (Nr,)
    a = np.stack([np.conj(H2_list[k]) * bs_to_ris_combined for k in range(K)])  # (K, Nr)
    d = np.array([Hd_list[k] @ beamformer for k in range(K)])  # (K,)

    theta = np.ones(Nr, dtype=complex) if theta_init is None else theta_init.copy()
    g = a @ theta + d  # (K,)

    def objective(g_vec):
        return float(np.sum(weights * np.abs(g_vec) ** 2))

    history = [objective(g)]
    grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    exp_grid = np.exp(1j * grid)  # (G,)

    for _ in range(n_outer):
        order = rng.permutation(Nr)
        for n in order:
            B = g - a[:, n] * theta[n]                       # (K,) contribution of all OTHER elements
            g_cand = B[:, None] + a[:, n][:, None] * exp_grid[None, :]  # (K, G)
            obj_vals = (weights[:, None] * np.abs(g_cand) ** 2).sum(axis=0)  # (G,)
            best = np.argmax(obj_vals)
            theta[n] = exp_grid[best]
            g = B + a[:, n] * theta[n]
        history.append(objective(g))

    return theta, np.array(history)


if __name__ == "__main__":
    from channel.channel_model import RISChannelModel

    model = RISChannelModel(num_bs_antennas=4, num_ris_elements=16, num_users=1, seed=2)
    bf = np.ones(model.Nt) / np.sqrt(model.Nt)

    theta_off = np.ones(model.Nr, dtype=complex)
    theta_heur = phase_alignment_heuristic(model.H1, model.H2[0], model.Hd[0], bf)

    g_off = model.effective_channel(0, theta_off, bf)
    g_heur = model.effective_channel(0, theta_heur, bf)
    print(f"RIS off      : |g|^2 = {abs(g_off) ** 2:.3e}")
    print(f"RIS heuristic: |g|^2 = {abs(g_heur) ** 2:.3e}  "
          f"(gain x{abs(g_heur) ** 2 / abs(g_off) ** 2:.1f})")

    if _HAS_CVXPY:
        theta_sdr = optimize_sum_rate_cvxpy(model.H1, model.H2[0], model.Hd[0], bf)
        g_sdr = model.effective_channel(0, theta_sdr, bf)
        print(f"RIS SDR      : |g|^2 = {abs(g_sdr) ** 2:.3e}")
    else:
        print("cvxpy not installed -- skipping SDR comparison (see README).")
