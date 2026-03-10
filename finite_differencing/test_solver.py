"""
test_solver.py — Tests for the time-marching PDAE solver.

Verifies:
  1. Boundary condition interpolation
  2. Single Newton step mechanics
  3. solve_at_time at endpoints
  4. Short trajectory solve (basic properties)
  5. Metric determinant stays positive along trajectory
  6. Snapshot storage and TrajectoryResult interface
  7. Adaptive time stepping behavior

Note: trajectory tests use margin=0.15 to avoid the near-singular corners
of the elliptic grid mapping, where lam_fo (reverse strain eigenvalue)
blows up as det(a_f) → 0.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geometry import build_endpoint_data, metric_det
from discretization import setup_discretization, boundary_mask
from solver import (
    SolverConfig,
    Snapshot,
    TrajectoryResult,
    interpolate_boundary_metric,
    single_newton_step,
    newton_solve,
    solve_at_time,
    solve_trajectory,
    quick_solve,
    _is_metric_valid,
)


# ---------------------------------------------------------------------------
# Fixtures and shared config
# ---------------------------------------------------------------------------

K_TEST = 15
MARGIN_UNIT = 0.05   # for unit tests that don't run trajectories
MARGIN_TRAJ = 0.15   # larger margin avoids corner singularity for trajectories


@pytest.fixture
def small_config():
    """Config for single-step / unit tests."""
    return SolverConfig(
        K=K_TEST,
        margin=MARGIN_TRAJ,
        dt=0.05,
        newton_max_iter=15,
        newton_tol=1e-3,
        use_line_search=True,
        adaptive_dt=False,
        verbose=False,
    )


@pytest.fixture
def data_and_disc():
    """Endpoint data and discretization with trajectory-safe margin."""
    data = build_endpoint_data(K_TEST, margin=MARGIN_TRAJ)
    disc = setup_discretization(K_TEST, data['dx'])
    return data, disc


# ===========================================================================
# 1. Boundary condition interpolation
# ===========================================================================

class TestBCInterpolation:

    def test_bc_at_t0(self):
        data = build_endpoint_data(K_TEST, margin=MARGIN_UNIT)
        bc_e, bc_f, bc_g = interpolate_boundary_metric(
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'], 0.0)
        np.testing.assert_allclose(bc_e, data['e'])
        np.testing.assert_allclose(bc_f, data['f'])
        np.testing.assert_allclose(bc_g, data['g'])

    def test_bc_at_t1(self):
        data = build_endpoint_data(K_TEST, margin=MARGIN_UNIT)
        bc_e, bc_f, bc_g = interpolate_boundary_metric(
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'], 1.0)
        np.testing.assert_allclose(bc_e, data['E'])
        np.testing.assert_allclose(bc_f, data['F'])
        np.testing.assert_allclose(bc_g, data['G'])

    def test_bc_at_midpoint(self):
        data = build_endpoint_data(K_TEST, margin=MARGIN_UNIT)
        bc_e, bc_f, bc_g = interpolate_boundary_metric(
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'], 0.5)
        np.testing.assert_allclose(bc_e, 0.5 * (data['e'] + data['E']))

    def test_bc_metric_stays_spd(self):
        data = build_endpoint_data(K_TEST, margin=MARGIN_UNIT)
        for t in np.linspace(0, 1, 11):
            bc_e, bc_f, bc_g = interpolate_boundary_metric(
                data['e'], data['f'], data['g'],
                data['E'], data['F'], data['G'], t)
            det = metric_det(bc_e, bc_f, bc_g)
            assert np.all(det > 0), f"SPD violation at t={t}"


# ===========================================================================
# 2. Single Newton step
# ===========================================================================

class TestSingleNewtonStep:

    def test_returns_finite(self, data_and_disc, small_config):
        data, disc = data_and_disc
        K = K_TEST
        t = 0.1

        bc_eps, bc_phi, bc_gam = interpolate_boundary_metric(
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'], t)

        eps_new, phi_new, gam_new, delta_X = single_newton_step(
            bc_eps.copy(), bc_phi.copy(), bc_gam.copy(),
            data, disc['ops'], K, t,
            bc_eps, bc_phi, bc_gam, small_config)

        assert np.all(np.isfinite(eps_new))
        assert np.all(np.isfinite(phi_new))
        assert np.all(np.isfinite(gam_new))

    def test_bc_enforced(self, data_and_disc, small_config):
        data, disc = data_and_disc
        K = K_TEST
        t = 0.3

        bc_eps, bc_phi, bc_gam = interpolate_boundary_metric(
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'], t)

        eps_new, phi_new, gam_new, _ = single_newton_step(
            bc_eps.copy(), bc_phi.copy(), bc_gam.copy(),
            data, disc['ops'], K, t,
            bc_eps, bc_phi, bc_gam, small_config)

        bnd = boundary_mask(K)
        np.testing.assert_allclose(eps_new[bnd], bc_eps[bnd], atol=1e-12)
        np.testing.assert_allclose(phi_new[bnd], bc_phi[bnd], atol=1e-12)
        np.testing.assert_allclose(gam_new[bnd], bc_gam[bnd], atol=1e-12)


# ===========================================================================
# 3. solve_at_time
# ===========================================================================

class TestSolveAtTime:

    def test_at_t0(self, data_and_disc, small_config):
        data, disc = data_and_disc
        snap = solve_at_time(0.0, data, disc, small_config)
        np.testing.assert_allclose(snap.eps, data['e'], atol=0.1)
        np.testing.assert_allclose(snap.gam, data['g'], atol=0.1)

    def test_returns_snapshot(self, data_and_disc, small_config):
        data, disc = data_and_disc
        snap = solve_at_time(0.5, data, disc, small_config)
        assert isinstance(snap, Snapshot)
        assert snap.t == 0.5
        assert snap.eps.shape == (K_TEST, K_TEST)

    def test_initial_guess_used(self, data_and_disc, small_config):
        data, disc = data_and_disc
        t = 0.5
        snap1 = solve_at_time(t, data, disc, small_config)
        snap2 = solve_at_time(t, data, disc, small_config,
                              initial_guess=(snap1.eps, snap1.phi, snap1.gam))
        assert snap2.newton_iters <= snap1.newton_iters


# ===========================================================================
# 4. Short trajectory solve
# ===========================================================================

class TestTrajectory:

    def _traj_config(self, t_end=0.2, dt=0.05):
        return SolverConfig(
            K=K_TEST, margin=MARGIN_TRAJ,
            dt=dt, t_end=t_end,
            newton_max_iter=15, newton_tol=1e-3,
            use_line_search=True, adaptive_dt=False,
            verbose=False,
        )

    def test_trajectory_runs(self):
        result = solve_trajectory(self._traj_config(t_end=0.15, dt=0.05))
        assert isinstance(result, TrajectoryResult)
        assert result.n_steps >= 2  # initial + at least one step

    def test_trajectory_times_monotonic(self):
        result = solve_trajectory(self._traj_config(t_end=0.15, dt=0.05))
        times = result.times
        assert np.all(np.diff(times) > 0)

    def test_trajectory_endpoints(self):
        result = solve_trajectory(self._traj_config(t_end=0.1, dt=0.05))
        assert abs(result.snapshots[0].t - 0.0) < 1e-14
        assert abs(result.snapshots[-1].t - 0.1) < 1e-12

    def test_metric_det_positive(self):
        result = solve_trajectory(self._traj_config(t_end=0.15, dt=0.05))
        for snap in result.snapshots:
            assert _is_metric_valid(snap.eps, snap.phi, snap.gam), \
                f"Invalid metric at t={snap.t}"

    def test_trajectory_result_interface(self):
        result = solve_trajectory(self._traj_config(t_end=0.1, dt=0.05))
        eps, phi, gam = result.metric_at(0)
        assert eps.shape == (K_TEST, K_TEST)
        det = result.det_at(0)
        assert det.shape == (K_TEST, K_TEST)


# ===========================================================================
# 5. Adaptive time stepping
# ===========================================================================

class TestAdaptiveDt:

    def test_adaptive_runs(self):
        config = SolverConfig(
            K=K_TEST, margin=MARGIN_TRAJ,
            dt=0.05, t_end=0.15,
            newton_max_iter=15, newton_tol=1e-3,
            use_line_search=True,
            adaptive_dt=True, dt_min=0.01, dt_max=0.1,
            verbose=False,
        )
        result = solve_trajectory(config)
        assert result.n_steps >= 2


# ===========================================================================
# 6. Config defaults
# ===========================================================================

class TestConfig:

    def test_default_config(self):
        config = SolverConfig()
        assert config.K == 50
        assert config.dt == 0.01
        assert config.newton_max_iter == 20

    def test_custom_config(self):
        config = SolverConfig(K=100, dt=0.005)
        assert config.K == 100
        assert config.dt == 0.005


# ===========================================================================
# 7. Metric validity helper
# ===========================================================================

class TestMetricValidity:

    def test_identity_valid(self):
        K = 10
        assert _is_metric_valid(np.ones((K, K)), np.zeros((K, K)), np.ones((K, K)))

    def test_nan_invalid(self):
        K = 10
        e = np.ones((K, K))
        e[5, 5] = np.nan
        assert not _is_metric_valid(e, np.zeros((K, K)), np.ones((K, K)))

    def test_negative_det_invalid(self):
        K = 10
        # det = e*g - f^2 = 1*1 - 2^2 = -3 < 0
        assert not _is_metric_valid(
            np.ones((K, K)), 2.0 * np.ones((K, K)), np.ones((K, K)))


# ===========================================================================
# 8. Quick solve
# ===========================================================================

class TestQuickSolve:

    def test_quick_solve_runs(self):
        config = SolverConfig(
            K=12, margin=0.15,
            dt=0.05, t_end=0.1,
            newton_max_iter=10, newton_tol=1e-2,
            use_line_search=True, verbose=False,
        )
        result = solve_trajectory(config)
        assert result.n_steps >= 2


# ===========================================================================
# Run
# ===========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
