"""
test_visualization.py — Tests for the visualization module.

Verifies:
  1. Surface reconstruction recovers known endpoints (square, circle)
  2. Reconstruction preserves metric (round-trip consistency)
  3. Plotting functions execute without errors
  4. generate_all_plots produces expected files
"""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geometry import (
    build_endpoint_data, metric_det, metric_circle,
    square_to_circle,
)
from discretization import setup_discretization
from solver import SolverConfig, solve_trajectory, Snapshot
from visualization import (
    reconstruct_surface,
    reconstruct_endpoint_surfaces,
    plot_metric_fields,
    plot_morphing_mesh,
    plot_convergence,
    plot_det_evolution,
    plot_trajectory_summary,
    generate_all_plots,
    plot_endpoint_comparison,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

K_TEST = 15
MARGIN = 0.15


@pytest.fixture
def endpoint_data():
    return build_endpoint_data(K_TEST, margin=MARGIN)


@pytest.fixture
def short_result():
    """Run a short trajectory for visualization tests."""
    config = SolverConfig(
        K=K_TEST, margin=MARGIN,
        dt=0.05, t_end=0.15,
        newton_max_iter=15, newton_tol=1e-3,
        use_line_search=True, verbose=False,
    )
    return solve_trajectory(config)


# ===========================================================================
# 1. Surface reconstruction
# ===========================================================================

class TestReconstruction:

    def test_square_shape(self, endpoint_data):
        """Identity metric → rectangular grid (up to centering)."""
        K = K_TEST
        L = 1.0 - 2 * MARGIN
        du = L / (K - 1)
        e, f, g = endpoint_data['e'], endpoint_data['f'], endpoint_data['g']

        X, Y = reconstruct_surface(e, f, g, du, du)

        assert X.shape == (K, K)
        assert Y.shape == (K, K)

        # For identity metric: X should increase along axis 0, Y along axis 1
        # Check monotonicity
        for j in range(K):
            assert np.all(np.diff(X[:, j]) > 0), "X not monotone along u"
        for i in range(K):
            assert np.all(np.diff(Y[i, :]) > 0), "Y not monotone along v"

    def test_square_aspect_ratio(self, endpoint_data):
        """Identity metric should give a square domain (aspect ≈ 1)."""
        K = K_TEST
        L = 1.0 - 2 * MARGIN
        du = L / (K - 1)
        e, f, g = endpoint_data['e'], endpoint_data['f'], endpoint_data['g']

        X, Y = reconstruct_surface(e, f, g, du, du)

        dx = X.max() - X.min()
        dy = Y.max() - Y.min()
        assert abs(dx / dy - 1.0) < 0.01, f"Aspect ratio: {dx/dy}"

    def test_circle_is_round(self, endpoint_data):
        """
        Circle metric should produce an approximately round shape.
        Check that boundary points lie approximately on a circle.
        """
        K = K_TEST
        L = 1.0 - 2 * MARGIN
        du = L / (K - 1)
        E, F, G = endpoint_data['E'], endpoint_data['F'], endpoint_data['G']

        X, Y = reconstruct_surface(E, F, G, du, du)

        # Boundary points
        bnd_x = np.concatenate([X[0, :], X[-1, :], X[:, 0], X[:, -1]])
        bnd_y = np.concatenate([Y[0, :], Y[-1, :], Y[:, 0], Y[:, -1]])
        r = np.sqrt(bnd_x**2 + bnd_y**2)

        # Radius should be approximately uniform (within ~30% given FD integration errors)
        r_std = np.std(r) / np.mean(r)
        print(f"Circle boundary radius CV: {r_std:.4f}")
        assert r_std < 0.5, f"Boundary too non-circular: CV = {r_std}"

    def test_reconstruction_preserves_area(self, endpoint_data):
        """
        The reconstructed surface should have approximately the correct area
        (∫√det(a) du dv).
        """
        K = K_TEST
        L = 1.0 - 2 * MARGIN
        du = L / (K - 1)
        e, f, g = endpoint_data['e'], endpoint_data['f'], endpoint_data['g']

        X, Y = reconstruct_surface(e, f, g, du, du)

        # Numerical area from grid: sum of parallelogram areas
        area = 0.0
        for i in range(K - 1):
            for j in range(K - 1):
                # Cross product of edge vectors
                dx_u = X[i+1, j] - X[i, j]
                dy_u = Y[i+1, j] - Y[i, j]
                dx_v = X[i, j+1] - X[i, j]
                dy_v = Y[i, j+1] - Y[i, j]
                area += abs(dx_u * dy_v - dx_v * dy_u)

        # For identity metric, area should be L²
        expected = L**2
        rel_err = abs(area - expected) / expected
        print(f"Square area: {area:.4f} vs expected {expected:.4f} (rel err: {rel_err:.4f})")
        assert rel_err < 0.05

    def test_endpoint_surfaces(self, endpoint_data):
        """reconstruct_endpoint_surfaces returns correct shapes."""
        surfaces = reconstruct_endpoint_surfaces(endpoint_data, K_TEST, MARGIN)
        assert 'square' in surfaces
        assert 'circle' in surfaces
        assert surfaces['square'][0].shape == (K_TEST, K_TEST)
        assert surfaces['circle'][0].shape == (K_TEST, K_TEST)


# ===========================================================================
# 2. Plotting functions don't crash
# ===========================================================================

class TestPlotMetricFields:

    def test_plot_metric_fields(self, endpoint_data):
        fig = plot_metric_fields(
            endpoint_data['E'], endpoint_data['F'], endpoint_data['G'], 1.0)
        assert fig is not None
        plt.close(fig)

    def test_plot_metric_fields_save(self, endpoint_data, tmp_path):
        path = str(tmp_path / 'metric.png')
        fig = plot_metric_fields(
            endpoint_data['E'], endpoint_data['F'], endpoint_data['G'],
            1.0, save_path=path)
        plt.close(fig)
        assert os.path.exists(path)


class TestPlotMorphingMesh:

    def test_plot_morphing(self, short_result):
        fig = plot_morphing_mesh(
            short_result.snapshots, short_result.endpoint_data,
            margin=MARGIN)
        assert fig is not None
        plt.close(fig)


class TestPlotConvergence:

    def test_plot_convergence(self, short_result):
        fig = plot_convergence(short_result)
        assert fig is not None
        plt.close(fig)


class TestPlotDetEvolution:

    def test_plot_det(self, short_result):
        fig = plot_det_evolution(short_result)
        assert fig is not None
        plt.close(fig)


class TestPlotSummary:

    def test_summary(self, short_result):
        fig = plot_trajectory_summary(short_result)
        assert fig is not None
        plt.close(fig)

    def test_summary_save(self, short_result, tmp_path):
        path = str(tmp_path / 'summary.png')
        fig = plot_trajectory_summary(short_result, save_path=path)
        plt.close(fig)
        assert os.path.exists(path)


# ===========================================================================
# 3. generate_all_plots
# ===========================================================================

class TestGenerateAllPlots:

    def test_generates_files(self, short_result, tmp_path):
        paths = generate_all_plots(
            short_result,
            output_dir=str(tmp_path),
            prefix='test',
        )
        assert len(paths) >= 4  # summary, morphing, convergence, det + metric snapshots
        for p in paths:
            assert os.path.exists(p), f"Missing: {p}"


# ===========================================================================
# Run
# ===========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
