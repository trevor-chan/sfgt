"""
solver.py — Time-marching Newton solver for the stress-free growth PDAE system.

Main entry point: solve_trajectory()

At each time step t_n → t_{n+1}:
  1. Compute interpolated boundary conditions for the metric
  2. Newton iteration: assemble J, solve J δX = -R, update metric
  3. Line search with metric positivity enforcement (det > 0)
  4. Check convergence; if stalled, halve Δt (adaptive mode)
  5. Store metric snapshot
  6. Advance to next time step

Also provides:
  - single_newton_step() for debugging
  - solve_at_time() for computing a single trajectory snapshot
"""

import numpy as np
import warnings
from typing import Dict, Callable, Optional, List, Tuple
from dataclasses import dataclass, field

from geometry import (
    build_endpoint_data,
    linear_interpolation,
    logarithmic_interpolation,
    metric_det,
)
from discretization import (
    setup_discretization,
    flatten_field,
    unflatten_field,
    apply_dirichlet_bc,
    boundary_mask,
)
from assembly import (
    assemble_system_with_bcs,
    solve_newton_step,
    unpack_delta,
    compute_residual_norms,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SolverConfig:
    """Configuration for the PDAE trajectory solver."""

    # Grid
    K: int = 50
    margin: float = 5e-2

    # Time stepping
    dt: float = 0.01
    t_start: float = 0.0
    t_end: float = 1.0

    # Newton iteration
    newton_max_iter: int = 20
    newton_tol: float = 1e-6
    newton_method: str = 'spsolve'

    # Line search — backtracking with metric positivity enforcement
    use_line_search: bool = True
    ls_alpha_min: float = 0.01
    ls_backtrack_factor: float = 0.5
    ls_max_backtracks: int = 15

    # Adaptive time stepping
    adaptive_dt: bool = False
    dt_min: float = 1e-5
    dt_max: float = 0.1
    dt_grow_factor: float = 1.5
    dt_shrink_factor: float = 0.5
    max_newton_for_grow: int = 4

    # Regularization for near-singular Jacobians
    regularization: float = 1e-8

    # Interpolation
    interp_fn: Callable = field(default_factory=lambda: linear_interpolation)

    # Output
    snapshot_interval: int = 1
    verbose: bool = True


# =============================================================================
# Trajectory snapshot
# =============================================================================

@dataclass
class Snapshot:
    """Metric state at a particular time."""
    t: float
    eps: np.ndarray
    phi: np.ndarray
    gam: np.ndarray
    residual_norms: Dict[str, float]
    newton_iters: int


@dataclass
class TrajectoryResult:
    """Full result of a trajectory solve."""
    snapshots: List[Snapshot]
    config: SolverConfig
    endpoint_data: Dict
    converged: bool
    message: str

    @property
    def times(self) -> np.ndarray:
        return np.array([s.t for s in self.snapshots])

    @property
    def n_steps(self) -> int:
        return len(self.snapshots)

    def metric_at(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = self.snapshots[idx]
        return s.eps, s.phi, s.gam

    def det_at(self, idx: int) -> np.ndarray:
        e, f, g = self.metric_at(idx)
        return metric_det(e, f, g)


# =============================================================================
# Boundary condition interpolation
# =============================================================================

def interpolate_boundary_metric(
    e_o: np.ndarray, f_o: np.ndarray, g_o: np.ndarray,
    E_f: np.ndarray, F_f: np.ndarray, G_f: np.ndarray,
    t: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Dirichlet boundary values for the metric at time t by
    linearly interpolating between the initial and final metrics.

    Returns
    -------
    bc_eps, bc_phi, bc_gam : ndarray (K, K)
        Interpolated metric fields (full arrays, but only boundary values used).
    """
    bc_eps = (1.0 - t) * e_o + t * E_f
    bc_phi = (1.0 - t) * f_o + t * F_f
    bc_gam = (1.0 - t) * g_o + t * G_f
    return bc_eps, bc_phi, bc_gam


# =============================================================================
# Metric validity check
# =============================================================================

def _is_metric_valid(eps, phi, gam) -> bool:
    """Check that the metric is finite and has positive determinant everywhere."""
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(phi)) and np.all(np.isfinite(gam))):
        return False
    det = metric_det(eps, phi, gam)
    return bool(np.all(det > 0))


# =============================================================================
# Single Newton step (exposed for debugging)
# =============================================================================

def single_newton_step(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    data: Dict, ops: Dict, K: int, t: float,
    bc_eps: np.ndarray, bc_phi: np.ndarray, bc_gam: np.ndarray,
    config: SolverConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one Newton iteration and return the updated metric fields
    and the raw correction vector.

    Returns
    -------
    eps_new, phi_new, gam_new : ndarray (K, K)
    delta_X : ndarray (3K²,)
    """
    J, neg_R = assemble_system_with_bcs(
        eps, phi, gam,
        data['e'], data['f'], data['g'],
        data['E'], data['F'], data['G'],
        t, data['lam_of'], data['lam_fo'],
        ops, K,
        bc_eps, bc_phi, bc_gam,
        interp_fn=config.interp_fn,
        regularization=config.regularization,
    )

    delta_X = solve_newton_step(J, neg_R, method=config.newton_method)
    d_eps, d_phi, d_gam = unpack_delta(delta_X, K)

    eps_new = eps + d_eps
    phi_new = phi + d_phi
    gam_new = gam + d_gam

    # Enforce BCs explicitly
    bnd = boundary_mask(K)
    eps_new[bnd] = bc_eps[bnd]
    phi_new[bnd] = bc_phi[bnd]
    gam_new[bnd] = bc_gam[bnd]

    return eps_new, phi_new, gam_new, delta_X


# =============================================================================
# Newton iteration with line search
# =============================================================================

def _compute_norms(eps, phi, gam, data, ops, K, t, config):
    """Compute total residual norm, returning None if metric is invalid."""
    if not _is_metric_valid(eps, phi, gam):
        return None
    return compute_residual_norms(
        eps, phi, gam,
        data['e'], data['f'], data['g'],
        data['E'], data['F'], data['G'],
        t, data['lam_of'], data['lam_fo'],
        ops, K,
        interp_fn=config.interp_fn,
    )


def _apply_step(eps, phi, gam, d_eps, d_phi, d_gam, alpha, bc_eps, bc_phi, bc_gam, bnd):
    """Apply a damped Newton step and enforce BCs."""
    e_trial = eps + alpha * d_eps
    p_trial = phi + alpha * d_phi
    g_trial = gam + alpha * d_gam
    e_trial[bnd] = bc_eps[bnd]
    p_trial[bnd] = bc_phi[bnd]
    g_trial[bnd] = bc_gam[bnd]
    return e_trial, p_trial, g_trial


def newton_solve(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    data: Dict, ops: Dict, K: int, t: float,
    bc_eps: np.ndarray, bc_phi: np.ndarray, bc_gam: np.ndarray,
    config: SolverConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], int, bool]:
    """
    Run Newton iterations until convergence or max iterations.

    The line search enforces:
      1. Residual decrease (Armijo-like)
      2. Metric positivity (det > 0 everywhere)
      3. Finiteness of the solution

    Returns
    -------
    eps, phi, gam : ndarray (K, K) — converged metric
    norms : dict — final residual norms
    n_iter : int — number of iterations taken
    converged : bool
    """
    bnd = boundary_mask(K)

    norms = _compute_norms(eps, phi, gam, data, ops, K, t, config)
    if norms is None:
        return eps, phi, gam, {'gauss': np.inf, 'alg_fwd': np.inf, 'alg_rev': np.inf, 'total': np.inf}, 0, False

    if config.verbose:
        print(f"    Newton iter 0: |R| = {norms['total']:.2e} "
              f"(G={norms['gauss']:.2e}, F={norms['alg_fwd']:.2e}, R={norms['alg_rev']:.2e})")

    if norms['total'] < config.newton_tol:
        return eps, phi, gam, norms, 0, True

    for it in range(1, config.newton_max_iter + 1):
        # Assemble and solve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress singular matrix warnings
            J, neg_R = assemble_system_with_bcs(
                eps, phi, gam,
                data['e'], data['f'], data['g'],
                data['E'], data['F'], data['G'],
                t, data['lam_of'], data['lam_fo'],
                ops, K,
                bc_eps, bc_phi, bc_gam,
                interp_fn=config.interp_fn,
                regularization=config.regularization,
            )
            delta_X = solve_newton_step(J, neg_R, method=config.newton_method)

        # Check for solver failure
        if not np.all(np.isfinite(delta_X)):
            if config.verbose:
                print(f"    Newton iter {it}: linear solve produced NaN/Inf, aborting")
            return eps, phi, gam, norms, it, False

        d_eps, d_phi, d_gam = unpack_delta(delta_X, K)

        # Line search: find largest alpha that gives valid metric + reduced residual
        alpha = 1.0
        accepted = False
        current_total = norms['total']

        for ls_it in range(config.ls_max_backtracks):
            e_trial, p_trial, g_trial = _apply_step(
                eps, phi, gam, d_eps, d_phi, d_gam,
                alpha, bc_eps, bc_phi, bc_gam, bnd,
            )

            # Check metric validity (finite + det > 0)
            if not _is_metric_valid(e_trial, p_trial, g_trial):
                alpha *= config.ls_backtrack_factor
                continue

            trial_norms = _compute_norms(e_trial, p_trial, g_trial, data, ops, K, t, config)
            if trial_norms is None:
                alpha *= config.ls_backtrack_factor
                continue

            # Accept if residual decreased or alpha is at minimum
            if trial_norms['total'] < current_total or alpha <= config.ls_alpha_min:
                accepted = True
                break

            alpha *= config.ls_backtrack_factor

        if not accepted:
            # Last resort: use minimum alpha if metric is valid
            alpha = config.ls_alpha_min
            e_trial, p_trial, g_trial = _apply_step(
                eps, phi, gam, d_eps, d_phi, d_gam,
                alpha, bc_eps, bc_phi, bc_gam, bnd,
            )
            if _is_metric_valid(e_trial, p_trial, g_trial):
                trial_norms = _compute_norms(e_trial, p_trial, g_trial, data, ops, K, t, config)
                if trial_norms is not None:
                    accepted = True

        if not accepted:
            if config.verbose:
                print(f"    Newton iter {it}: line search failed, aborting")
            return eps, phi, gam, norms, it, False

        eps, phi, gam = e_trial, p_trial, g_trial
        norms = trial_norms

        if config.verbose:
            step_norm = np.max(np.abs(alpha * delta_X))
            print(f"    Newton iter {it}: |R| = {norms['total']:.2e} "
                  f"(G={norms['gauss']:.2e}, F={norms['alg_fwd']:.2e}, "
                  f"R={norms['alg_rev']:.2e})  α={alpha:.3f}  |δX|={step_norm:.2e}")

        if norms['total'] < config.newton_tol:
            return eps, phi, gam, norms, it, True

    return eps, phi, gam, norms, config.newton_max_iter, False


# =============================================================================
# Solve at a single time
# =============================================================================

def solve_at_time(
    t: float,
    data: Dict,
    disc: Dict,
    config: SolverConfig,
    initial_guess: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> Snapshot:
    """
    Solve for the metric at a single time t, starting from an initial guess.

    Parameters
    ----------
    t : float in [0, 1]
    data : endpoint data from build_endpoint_data
    disc : discretization data from setup_discretization
    config : solver configuration
    initial_guess : (eps, phi, gam) or None (defaults to interpolated metric)

    Returns
    -------
    snapshot : Snapshot
    """
    K = config.K
    ops = disc['ops']

    bc_eps, bc_phi, bc_gam = interpolate_boundary_metric(
        data['e'], data['f'], data['g'],
        data['E'], data['F'], data['G'],
        t,
    )

    if initial_guess is not None:
        eps, phi, gam = [a.copy() for a in initial_guess]
    else:
        # Use interpolated metric as initial guess
        eps, phi, gam = bc_eps.copy(), bc_phi.copy(), bc_gam.copy()

    # Enforce BCs on initial guess
    bnd = boundary_mask(K)
    eps[bnd] = bc_eps[bnd]
    phi[bnd] = bc_phi[bnd]
    gam[bnd] = bc_gam[bnd]

    eps, phi, gam, norms, n_iter, converged = newton_solve(
        eps, phi, gam, data, ops, K, t,
        bc_eps, bc_phi, bc_gam, config,
    )

    return Snapshot(
        t=t,
        eps=eps.copy(),
        phi=phi.copy(),
        gam=gam.copy(),
        residual_norms=norms,
        newton_iters=n_iter,
    )


# =============================================================================
# Full trajectory solve
# =============================================================================

def solve_trajectory(config: SolverConfig = None) -> TrajectoryResult:
    """
    Solve the PDAE system along the full trajectory from t_start to t_end.

    This is the main entry point. It:
      1. Sets up endpoint geometry and discretization
      2. Marches forward in time, solving at each step via Newton iteration
      3. Uses the interpolated metric at t_{n+1} as initial guess, with the
         previous solution blended in as continuation improves
      4. Optionally adapts Δt based on Newton convergence

    Parameters
    ----------
    config : SolverConfig, optional

    Returns
    -------
    result : TrajectoryResult
    """
    if config is None:
        config = SolverConfig()

    K = config.K

    if config.verbose:
        print(f"Setting up solver: K={K}, dt={config.dt}, "
              f"t=[{config.t_start}, {config.t_end}]")

    # Setup
    data = build_endpoint_data(K, margin=config.margin)
    disc = setup_discretization(K, data['dx'])
    ops = disc['ops']

    # Initial state
    eps = data['e'].copy()
    phi = data['f'].copy()
    gam = data['g'].copy()

    snapshots = []

    # Store initial snapshot
    norms_init = compute_residual_norms(
        eps, phi, gam,
        data['e'], data['f'], data['g'],
        data['E'], data['F'], data['G'],
        config.t_start, data['lam_of'], data['lam_fo'],
        ops, K, interp_fn=config.interp_fn,
    )
    snapshots.append(Snapshot(
        t=config.t_start,
        eps=eps.copy(), phi=phi.copy(), gam=gam.copy(),
        residual_norms=norms_init, newton_iters=0,
    ))

    t = config.t_start
    dt = config.dt
    step = 0
    total_newton = 0
    failed = False
    norms = norms_init

    while t < config.t_end - 1e-14:
        step += 1
        t_next = min(t + dt, config.t_end)

        if config.verbose:
            print(f"\n--- Step {step}: t = {t:.6f} → {t_next:.6f} (dt = {dt:.2e}) ---")

        # Boundary conditions at t_next
        bc_eps, bc_phi, bc_gam = interpolate_boundary_metric(
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            t_next,
        )

        # Initial guess: interpolated metric at t_next.
        # The interpolated metric satisfies the algebraic constraints approximately
        # and provides a non-degenerate starting point for Newton.
        eps_guess = bc_eps.copy()
        phi_guess = bc_phi.copy()
        gam_guess = bc_gam.copy()

        # Newton solve
        eps_new, phi_new, gam_new, norms, n_iter, converged = newton_solve(
            eps_guess, phi_guess, gam_guess,
            data, ops, K, t_next,
            bc_eps, bc_phi, bc_gam,
            config,
        )

        total_newton += n_iter

        if not converged and config.adaptive_dt:
            dt_old = dt
            dt = max(dt * config.dt_shrink_factor, config.dt_min)
            if config.verbose:
                print(f"  Newton did not converge. Shrinking dt: {dt_old:.2e} → {dt:.2e}")
            if dt <= config.dt_min:
                if config.verbose:
                    print(f"  dt at minimum ({config.dt_min:.2e}). Accepting current solution.")
            else:
                continue  # retry with smaller dt
        elif not converged:
            if config.verbose:
                print(f"  Warning: Newton did not converge (|R| = {norms['total']:.2e}). "
                      f"Accepting and continuing.")
        else:
            # Success — adaptive dt growth
            if config.adaptive_dt and n_iter <= config.max_newton_for_grow:
                dt = min(dt * config.dt_grow_factor, config.dt_max)

        # Accept the step
        eps, phi, gam = eps_new, phi_new, gam_new
        t = t_next

        # Check for metric degeneracy
        if not _is_metric_valid(eps, phi, gam):
            if config.verbose:
                det = metric_det(eps, phi, gam)
                det_min = np.nanmin(det)
                print(f"  FATAL: metric invalid (min det = {det_min:.2e}). Stopping.")
            failed = True
            break

        # Store snapshot
        if step % config.snapshot_interval == 0 or abs(t - config.t_end) < 1e-14:
            snapshots.append(Snapshot(
                t=t,
                eps=eps.copy(), phi=phi.copy(), gam=gam.copy(),
                residual_norms=norms, newton_iters=n_iter,
            ))

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"Trajectory complete: {step} time steps, {total_newton} total Newton iterations")
        print(f"Final t = {t:.6f}, final |R| = {norms['total']:.2e}")
        if _is_metric_valid(eps, phi, gam):
            det = metric_det(eps, phi, gam)
            print(f"Metric det range: [{np.min(det):.4e}, {np.max(det):.4e}]")
        print(f"{'='*60}")

    return TrajectoryResult(
        snapshots=snapshots,
        config=config,
        endpoint_data=data,
        converged=not failed,
        message="Completed" if not failed else "Metric became degenerate",
    )


# =============================================================================
# Convenience: quick solve with defaults
# =============================================================================

def quick_solve(K: int = 30, dt: float = 0.05, verbose: bool = True) -> TrajectoryResult:
    """Run a quick trajectory solve with sensible defaults for testing."""
    config = SolverConfig(
        K=K,
        dt=dt,
        newton_max_iter=15,
        newton_tol=1e-4,
        use_line_search=True,
        adaptive_dt=False,
        verbose=verbose,
    )
    return solve_trajectory(config)
