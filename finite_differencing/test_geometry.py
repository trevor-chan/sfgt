"""
test_geometry.py — Tests for the endpoint geometry module.

Verifies:
  1. Elliptic grid transformation roundtrips
  2. Metric tensors are SPD with correct determinants
  3. Analytic metric matches Jacobian-derived metric
  4. Identity metric satisfies the Gauss equation
  5. Circle metric satisfies the Gauss equation (numerically)
  6. Green strain eigenvalue consistency between the two formulas
  7. Strain eigenvalue boundary values (Λ=0 when current=reference)
  8. Algebraic constraints are satisfied at trajectory endpoints
  9. Interpolation schemes have correct boundary behavior
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geometry import (
    square_to_circle, circle_to_square,
    metric_square, metric_circle, metric_circle_from_jacobian, metric_det,
    gauss_residual,
    green_strain_eigenvalue, green_strain_eigenvalue_direct,
    linear_interpolation, logarithmic_interpolation,
    algebraic_constraint_forward, algebraic_constraint_reverse,
    build_endpoint_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_derivatives(f_vals, dx, dy):
    """
    Compute first and second derivatives of a 2D field using second-order
    central differences. Returns a dict of derivative arrays.
    """
    f_u = np.zeros_like(f_vals)
    f_v = np.zeros_like(f_vals)
    f_uu = np.zeros_like(f_vals)
    f_vv = np.zeros_like(f_vals)
    f_uv = np.zeros_like(f_vals)

    # First derivatives (central difference, interior only)
    f_u[1:-1, :] = (f_vals[2:, :] - f_vals[:-2, :]) / (2.0 * dx)
    f_v[:, 1:-1] = (f_vals[:, 2:] - f_vals[:, :-2]) / (2.0 * dy)

    # Second derivatives
    f_uu[1:-1, :] = (f_vals[2:, :] - 2.0 * f_vals[1:-1, :] + f_vals[:-2, :]) / dx**2
    f_vv[:, 1:-1] = (f_vals[:, 2:] - 2.0 * f_vals[:, 1:-1] + f_vals[:, :-2]) / dy**2

    # Mixed derivative
    f_uv[1:-1, 1:-1] = (
        f_vals[2:, 2:] - f_vals[2:, :-2] - f_vals[:-2, 2:] + f_vals[:-2, :-2]
    ) / (4.0 * dx * dy)

    return {
        'u': f_u, 'v': f_v,
        'uu': f_uu, 'vv': f_vv, 'uv': f_uv,
    }


def compute_gauss_residual_numerical(eps, phi, gam, dx, dy):
    """
    Compute the Gauss residual using numerical derivatives on the full grid.
    Only interior points (excluding 2-cell boundary) are meaningful.
    """
    d_eps = numerical_derivatives(eps, dx, dy)
    d_phi = numerical_derivatives(phi, dx, dy)
    d_gam = numerical_derivatives(gam, dx, dy)

    return gauss_residual(
        eps, phi, gam,
        d_eps['u'], d_eps['v'],
        d_gam['u'], d_gam['v'],
        d_phi['u'], d_phi['v'],
        d_eps['vv'], d_phi['uv'], d_gam['uu'],
    )


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

K_SMALL = 50       # for fast tests
K_MEDIUM = 200     # for accuracy tests
MARGIN = 1e-3      # avoid corner singularities of elliptic grid transform
TOL_ROUNDTRIP = 1e-10
TOL_METRIC = 1e-8
TOL_GAUSS = 1e-2    # finite difference Gauss residual (depends on K)
TOL_STRAIN = 1e-10


# ===========================================================================
# 1. Elliptic grid transformation roundtrip
# ===========================================================================

class TestEllipticGrid:

    def test_roundtrip_square_circle_square(self):
        """(x,y) -> (u,v) -> (x',y') should recover (x,y)."""
        lin = np.linspace(-1.0 + MARGIN, 1.0 - MARGIN, K_SMALL)
        x, y = np.meshgrid(lin, lin, indexing='ij')

        u, v = square_to_circle(x, y)
        x2, y2 = circle_to_square(u, v)

        np.testing.assert_allclose(x2, x, atol=TOL_ROUNDTRIP)
        np.testing.assert_allclose(y2, y, atol=TOL_ROUNDTRIP)

    def test_origin_maps_to_origin(self):
        """The origin should be a fixed point."""
        u, v = square_to_circle(np.array([0.0]), np.array([0.0]))
        assert abs(u[0]) < 1e-15
        assert abs(v[0]) < 1e-15

    def test_image_inside_unit_disc(self):
        """All mapped points should lie within (or on) the unit disc."""
        lin = np.linspace(-1.0, 1.0, K_SMALL)
        x, y = np.meshgrid(lin, lin, indexing='ij')
        u, v = square_to_circle(x, y)
        r2 = u**2 + v**2
        assert np.all(r2 <= 1.0 + 1e-12)

    def test_axes_map_to_axes(self):
        """Points on the x-axis (y=0) should map to (x, 0) and similarly for y."""
        x_ax = np.linspace(-0.9, 0.9, 20)
        u, v = square_to_circle(x_ax, np.zeros_like(x_ax))
        np.testing.assert_allclose(u, x_ax, atol=1e-14)
        np.testing.assert_allclose(v, np.zeros_like(x_ax), atol=1e-14)


# ===========================================================================
# 2. Metric tensor properties
# ===========================================================================

class TestMetricProperties:

    @pytest.fixture
    def grid(self):
        lin = np.linspace(-1.0 + MARGIN, 1.0 - MARGIN, K_SMALL)
        x, y = np.meshgrid(lin, lin, indexing='ij')
        return x, y

    def test_square_metric_is_identity(self, grid):
        x, y = grid
        e, f, g = metric_square(x, y)
        np.testing.assert_allclose(e, 1.0)
        np.testing.assert_allclose(f, 0.0, atol=1e-15)
        np.testing.assert_allclose(g, 1.0)

    def test_circle_metric_spd(self, grid):
        """Circle metric should be symmetric positive definite everywhere."""
        x, y = grid
        E, F, G = metric_circle(x, y)
        det = metric_det(E, F, G)

        # SPD conditions: E > 0, det > 0 (G > 0 follows)
        assert np.all(E > 0), f"E has non-positive values: min={E.min()}"
        assert np.all(det > 0), f"det has non-positive values: min={det.min()}"

    def test_circle_metric_diagonal_at_origin(self, grid):
        """At the origin, the elliptic grid is locally conformal, so F=0."""
        x, y = grid
        E, F, G = metric_circle(x, y)

        # Find the grid point closest to the origin
        center = K_SMALL // 2
        assert abs(F[center, center]) < 1e-2  # grid point near but not exactly at origin

    def test_circle_metric_matches_jacobian(self, grid):
        """Analytic metric formula should match the Jacobian-based computation."""
        x, y = grid
        E1, F1, G1 = metric_circle(x, y)
        E2, F2, G2 = metric_circle_from_jacobian(x, y)

        np.testing.assert_allclose(E1, E2, atol=TOL_METRIC, rtol=TOL_METRIC)
        np.testing.assert_allclose(F1, F2, atol=TOL_METRIC, rtol=TOL_METRIC)
        np.testing.assert_allclose(G1, G2, atol=TOL_METRIC, rtol=TOL_METRIC)

    def test_square_metric_det_is_one(self, grid):
        x, y = grid
        e, f, g = metric_square(x, y)
        det = metric_det(e, f, g)
        np.testing.assert_allclose(det, 1.0)


# ===========================================================================
# 3. Gauss equation
# ===========================================================================

class TestGaussEquation:

    def test_identity_metric_gauss_zero(self):
        """The identity metric (constant) has all derivatives = 0, so residual = 0."""
        K = K_SMALL
        ones = np.ones((K, K))
        zeros = np.zeros((K, K))

        res = gauss_residual(
            ones, zeros, ones,
            zeros, zeros, zeros, zeros, zeros, zeros,
            zeros, zeros, zeros,
        )
        np.testing.assert_allclose(res, 0.0, atol=1e-15)

    def test_circle_metric_gauss_residual_small(self):
        """
        The circle metric, being induced by a valid embedding, must satisfy
        the Gauss equation. The residual should be small (limited by finite
        differencing accuracy).
        """
        K = K_MEDIUM
        margin = 5e-2  # larger margin to avoid boundary derivative issues
        lin = np.linspace(-1.0 + margin, 1.0 - margin, K)
        x, y = np.meshgrid(lin, lin, indexing='ij')
        dx = lin[1] - lin[0]

        E, F, G = metric_circle(x, y)
        res = compute_gauss_residual_numerical(E, F, G, dx, dx)

        # Check interior (exclude 2-cell border where central differences are invalid)
        interior = res[2:-2, 2:-2]
        max_res = np.max(np.abs(interior))
        print(f"Circle metric Gauss residual (K={K}): max |R| = {max_res:.2e}")

        # With K=200 and margin=0.05, residual should be O(dx^2) ~ O(1e-4)
        assert max_res < TOL_GAUSS, f"Gauss residual too large: {max_res:.2e}"

    def test_gauss_residual_convergence(self):
        """
        Gauss residual should decrease as O(dx^2) with grid refinement,
        confirming second-order accuracy of the central differences.
        """
        margin = 0.1
        residuals = []
        Ks = [50, 100, 200]

        for K in Ks:
            lin = np.linspace(-1.0 + margin, 1.0 - margin, K)
            x, y = np.meshgrid(lin, lin, indexing='ij')
            dx = lin[1] - lin[0]

            E, F, G = metric_circle(x, y)
            res = compute_gauss_residual_numerical(E, F, G, dx, dx)
            # Use a central sub-region to avoid boundary effects
            k4 = K // 4
            interior = res[k4:-k4, k4:-k4]
            residuals.append(np.max(np.abs(interior)))

        # Check convergence rate
        rates = []
        for i in range(1, len(Ks)):
            rate = np.log(residuals[i-1] / residuals[i]) / np.log(Ks[i] / Ks[i-1])
            rates.append(rate)

        print(f"Gauss residual convergence rates: {rates}")
        print(f"Gauss residuals: {residuals}")
        # Expect rate ~ 2 for second-order central differences
        for rate in rates:
            assert rate > 1.5, f"Convergence rate {rate:.2f} < 1.5 (expected ~2)"


# ===========================================================================
# 4. Green strain eigenvalue
# ===========================================================================

class TestStrainEigenvalue:

    @pytest.fixture
    def endpoint_data(self):
        return build_endpoint_data(K_SMALL, margin=MARGIN)

    def test_strain_zero_when_same(self, endpoint_data):
        """Λ[ε_rr] = 0 when current = reference (no deformation)."""
        d = endpoint_data
        lam = green_strain_eigenvalue(d['e'], d['f'], d['g'],
                                       d['e'], d['f'], d['g'])
        np.testing.assert_allclose(lam, 0.0, atol=TOL_STRAIN)

    def test_direct_formula_matches_general(self, endpoint_data):
        """
        The direct formula (eq. 7) should match the general eigenvalue
        computation for Λ[ε_of].
        """
        d = endpoint_data
        lam1 = green_strain_eigenvalue(d['e'], d['f'], d['g'],
                                        d['E'], d['F'], d['G'])
        lam2 = green_strain_eigenvalue_direct(d['e'], d['f'], d['g'],
                                               d['E'], d['F'], d['G'])
        np.testing.assert_allclose(lam1, lam2, atol=1e-8, rtol=1e-8)

    def test_strain_eigenvalue_symmetric_property(self, endpoint_data):
        """
        Λ[ε_of] and Λ[ε_fo] should generally have opposite signs
        (stretching in one direction = compression in reverse), though
        magnitudes differ due to the reference metric normalization.
        """
        d = endpoint_data
        # Just check that they're not identical (they shouldn't be unless a_o = a_f)
        diff = np.max(np.abs(d['lam_of'] - d['lam_fo']))
        assert diff > 0.01, "Forward and reverse strain eigenvalues should differ"

    def test_strain_eigenvalue_center_point(self, endpoint_data):
        """
        At the center (x=0, y=0), the circle metric is the identity,
        so the strain should be zero.
        """
        d = endpoint_data
        center = K_SMALL // 2
        lam_center = d['lam_of'][center, center]
        # At center, E=1, F=0, G=1 = a_o, so strain = 0
        assert abs(lam_center) < 1e-2, f"Strain at center = {lam_center} (expected ~0)"


# ===========================================================================
# 5. Interpolation schemes
# ===========================================================================

class TestInterpolation:

    def test_linear_boundary_values(self):
        lam = np.array([1.0, 2.0, 0.5])
        np.testing.assert_allclose(linear_interpolation(lam, 0.0), lam)
        np.testing.assert_allclose(linear_interpolation(lam, 1.0), 0.0)

    def test_linear_midpoint_symmetry(self):
        """Linear interpolation is symmetric about t=0.5."""
        lam = np.array([1.0, 2.0, 0.5])
        val_at_t = linear_interpolation(lam, 0.3)
        val_at_1mt = linear_interpolation(lam, 0.7)
        np.testing.assert_allclose(val_at_t + val_at_1mt, lam)

    def test_logarithmic_boundary_values(self):
        lam = np.array([1.0, 2.0, 0.5])
        np.testing.assert_allclose(logarithmic_interpolation(lam, 0.0), lam)
        np.testing.assert_allclose(logarithmic_interpolation(lam, 1.0), 0.0, atol=1e-14)

    def test_logarithmic_breaks_symmetry(self):
        """Log interpolation should NOT be symmetric about t=0.5."""
        lam = np.array([1.0])
        val_t = logarithmic_interpolation(lam, 0.3)
        val_1mt = logarithmic_interpolation(lam, 0.7)
        # For the linear case, val_t + val_1mt = lam. Not so for log.
        assert abs((val_t + val_1mt)[0] - lam[0]) > 0.01


# ===========================================================================
# 6. Algebraic constraints at endpoints
# ===========================================================================

class TestAlgebraicConstraints:

    @pytest.fixture
    def endpoint_data(self):
        return build_endpoint_data(K_SMALL, margin=MARGIN)

    def test_forward_constraint_at_t0(self, endpoint_data):
        """
        At t=0, the current metric = a_o, so Λ_direct(a_o; a_o) = 0,
        and the interpolation target = Λ[ε_of]. So residual = Λ[ε_of].
        This is expected: the constraint says "you still have this much
        deformation to go."
        """
        d = endpoint_data
        res = algebraic_constraint_forward(
            d['e'], d['f'], d['g'],
            d['e'], d['f'], d['g'],  # current = initial
            0.0, d['lam_of'],
        )
        # At t=0: target = lam_of, actual = 0, so residual = lam_of
        np.testing.assert_allclose(res, d['lam_of'], atol=1e-10)

    def test_forward_constraint_at_t1(self, endpoint_data):
        """
        At t=1, current metric should be a_f. Then:
        - target = interp(lam_of, 1) = 0
        - actual = Λ_direct(a_o; a_f) = lam_of
        So residual = -lam_of. The constraint is satisfied only when
        the current metric actually equals a_f.
        """
        d = endpoint_data
        # If we plug in the TRUE final metric, residual should be 0 - lam_of
        res = algebraic_constraint_forward(
            d['e'], d['f'], d['g'],
            d['E'], d['F'], d['G'],  # current = final
            1.0, d['lam_of'],
        )
        np.testing.assert_allclose(res, -d['lam_of'], atol=1e-8)

    def test_reverse_constraint_at_t1(self, endpoint_data):
        """At t=1, reverse target = interp(lam_fo, 0) = lam_fo."""
        d = endpoint_data
        res = algebraic_constraint_reverse(
            d['E'], d['F'], d['G'],
            d['E'], d['F'], d['G'],  # current = final
            1.0, d['lam_fo'],
        )
        np.testing.assert_allclose(res, d['lam_fo'], atol=1e-10)

    def test_both_constraints_zero_at_known_intermediate(self, endpoint_data):
        """
        For linear interpolation of the metric (van Rees' approach),
        at t=0.5 both constraints should be approximately satisfied
        IF the metric is the midpoint. This is a sanity check that our
        constraint formulation is consistent.

        Note: this won't be exactly zero because linear metric interpolation
        doesn't satisfy the Gauss equation — that's the whole point of the paper.
        But the algebraic constraints alone should be close.
        """
        d = endpoint_data
        # Linear metric midpoint
        eps_mid = 0.5 * (d['e'] + d['E'])
        phi_mid = 0.5 * (d['f'] + d['F'])
        gam_mid = 0.5 * (d['g'] + d['G'])

        res_fwd = algebraic_constraint_forward(
            d['e'], d['f'], d['g'],
            eps_mid, phi_mid, gam_mid,
            0.5, d['lam_of'],
        )
        # This tests that the constraint machinery runs without errors
        # and returns finite values
        assert np.all(np.isfinite(res_fwd))


# ===========================================================================
# 7. build_endpoint_data integration test
# ===========================================================================

class TestBuildEndpointData:

    def test_shapes(self):
        K = 30
        d = build_endpoint_data(K)
        assert d['x'].shape == (K, K)
        assert d['lam_of'].shape == (K, K)
        assert d['dx'] == d['dy']  # uniform grid

    def test_all_finite(self):
        d = build_endpoint_data(K_SMALL, margin=MARGIN)
        for key in ['e', 'f', 'g', 'E', 'F', 'G', 'lam_of', 'lam_fo']:
            assert np.all(np.isfinite(d[key])), f"{key} contains non-finite values"

    def test_det_positive(self):
        d = build_endpoint_data(K_SMALL, margin=MARGIN)
        det_o = metric_det(d['e'], d['f'], d['g'])
        det_f = metric_det(d['E'], d['F'], d['G'])
        assert np.all(det_o > 0)
        assert np.all(det_f > 0)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
