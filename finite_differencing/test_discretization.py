"""
test_discretization.py — Tests for the spatial discretization module.

Verifies:
  1. Index conversion roundtrips
  2. Boundary/interior masks have correct counts
  3. Difference operators match analytic derivatives on known functions
  4. Convergence rates of difference operators
  5. Kronecker (fast) and loop-based operators agree
  6. BC projection extracts correct DOFs
  7. Interior-restricted operators give consistent results
  8. BC RHS contribution correctly accounts for boundary values
  9. Integration with geometry module (Gauss residual via sparse ops)
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from discretization import (
    ij_to_flat, flat_to_ij, flatten_field, unflatten_field,
    boundary_mask, interior_mask, interior_indices, boundary_indices,
    build_difference_operators, build_difference_operators_fast,
    apply_operator, compute_all_derivatives,
    apply_dirichlet_bc, build_bc_projection,
    restrict_operator_to_interior, bc_rhs_contribution, make_bc_vector,
    setup_discretization,
)
from geometry import metric_circle, gauss_residual


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

K_SMALL = 20
K_MEDIUM = 80
TOL_DERIV = None  # set per-test based on expected order


# ===========================================================================
# 1. Index utilities
# ===========================================================================

class TestIndexUtilities:

    def test_ij_flat_roundtrip(self):
        K = 15
        for i in range(K):
            for j in range(K):
                idx = ij_to_flat(i, j, K)
                i2, j2 = flat_to_ij(idx, K)
                assert (i2, j2) == (i, j)

    def test_flatten_unflatten_roundtrip(self):
        K = 10
        field = np.random.rand(K, K)
        recovered = unflatten_field(flatten_field(field), K)
        np.testing.assert_array_equal(field, recovered)

    def test_flat_index_range(self):
        K = 12
        indices = [ij_to_flat(i, j, K) for i in range(K) for j in range(K)]
        assert min(indices) == 0
        assert max(indices) == K * K - 1
        assert len(set(indices)) == K * K  # all unique


# ===========================================================================
# 2. Boundary / interior masks
# ===========================================================================

class TestMasks:

    def test_boundary_count(self):
        """Boundary of K×K grid has 4K - 4 points."""
        K = 25
        bnd = boundary_mask(K)
        assert bnd.sum() == 4 * K - 4

    def test_interior_count(self):
        """Interior has (K-2)^2 points."""
        K = 25
        interior = interior_mask(K)
        assert interior.sum() == (K - 2) ** 2

    def test_masks_complement(self):
        K = 15
        bnd = boundary_mask(K)
        interior = interior_mask(K)
        assert np.all(bnd | interior)
        assert not np.any(bnd & interior)

    def test_boundary_indices_match_mask(self):
        K = 10
        bnd_mask = boundary_mask(K)
        bnd_idx = boundary_indices(K)
        mask_from_idx = np.zeros(K * K, dtype=bool)
        mask_from_idx[bnd_idx] = True
        np.testing.assert_array_equal(mask_from_idx, bnd_mask.ravel())


# ===========================================================================
# 3. Difference operators on known analytic functions
# ===========================================================================

def _make_test_grid(K, margin=0.0):
    """Build a grid on [-1+m, 1-m] and return x, y, dx."""
    lin = np.linspace(-1.0 + margin, 1.0 - margin, K)
    x, y = np.meshgrid(lin, lin, indexing='ij')
    dx = lin[1] - lin[0]
    return x, y, dx


class TestDifferenceOperators:

    @pytest.fixture(params=['loop', 'fast'])
    def ops_and_grid(self, request):
        """Provide operators built both ways for comparison."""
        K = K_MEDIUM
        x, y, dx = _make_test_grid(K)
        if request.param == 'loop':
            ops = build_difference_operators(K, dx)
        else:
            ops = build_difference_operators_fast(K, dx)
        return ops, x, y, dx, K

    def test_D_u_on_polynomial(self, ops_and_grid):
        """D_u (x^2 y) = 2xy."""
        ops, x, y, dx, K = ops_and_grid
        f = x**2 * y
        analytic = 2.0 * x * y
        numerical = apply_operator(ops['D_u'], f, K)

        # Check interior (central diff is exact for degree ≤ 2 polynomials
        # on uniform grids — it should be exact to machine precision)
        interior = slice(1, -1), slice(1, -1)
        np.testing.assert_allclose(numerical[interior], analytic[interior], atol=1e-10)

    def test_D_v_on_polynomial(self, ops_and_grid):
        """D_v (x y^2) = 2xy. Exact for central differences on quadratic."""
        ops, x, y, dx, K = ops_and_grid
        f = x * y**2
        analytic = 2.0 * x * y
        numerical = apply_operator(ops['D_v'], f, K)

        interior = slice(1, -1), slice(1, -1)
        np.testing.assert_allclose(numerical[interior], analytic[interior], atol=1e-10)

    def test_D_uu_on_polynomial(self, ops_and_grid):
        """D_uu (x^3) = 6x. Central diff is exact for cubic? No — it's O(dx^2).
        Use quadratic: D_uu(x^2 y) = 2y, which is exact."""
        ops, x, y, dx, K = ops_and_grid
        f = x**2 * y
        analytic = 2.0 * y
        numerical = apply_operator(ops['D_uu'], f, K)

        interior = slice(1, -1), slice(1, -1)
        np.testing.assert_allclose(numerical[interior], analytic[interior], atol=1e-10)

    def test_D_vv_on_polynomial(self, ops_and_grid):
        """D_vv (x y^2) = 2x."""
        ops, x, y, dx, K = ops_and_grid
        f = x * y**2
        analytic = 2.0 * x
        numerical = apply_operator(ops['D_vv'], f, K)

        interior = slice(1, -1), slice(1, -1)
        np.testing.assert_allclose(numerical[interior], analytic[interior], atol=1e-10)

    def test_D_uv_on_polynomial(self, ops_and_grid):
        """D_uv (x^2 y^2) = 4xy."""
        ops, x, y, dx, K = ops_and_grid
        f = x**2 * y**2
        analytic = 4.0 * x * y
        numerical = apply_operator(ops['D_uv'], f, K)

        # Mixed derivative needs 2-cell border
        interior = slice(2, -2), slice(2, -2)
        np.testing.assert_allclose(numerical[interior], analytic[interior], atol=1e-10)

    def test_D_u_on_sin(self, ops_and_grid):
        """D_u sin(πx) cos(πy) = π cos(πx) cos(πy), to O(dx^2)."""
        ops, x, y, dx, K = ops_and_grid
        f = np.sin(np.pi * x) * np.cos(np.pi * y)
        analytic = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        numerical = apply_operator(ops['D_u'], f, K)

        interior = slice(1, -1), slice(1, -1)
        err = np.max(np.abs(numerical[interior] - analytic[interior]))
        # For K=80, dx ~ 0.025, O(dx^2) ~ 6e-4; error should be in that ballpark
        assert err < 0.01, f"D_u sin error = {err:.2e}"

    def test_D_vv_on_sin(self, ops_and_grid):
        """D_vv sin(πx) sin(πy) = -π^2 sin(πx) sin(πy)."""
        ops, x, y, dx, K = ops_and_grid
        f = np.sin(np.pi * x) * np.sin(np.pi * y)
        analytic = -np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        numerical = apply_operator(ops['D_vv'], f, K)

        interior = slice(1, -1), slice(1, -1)
        err = np.max(np.abs(numerical[interior] - analytic[interior]))
        assert err < 0.05, f"D_vv sin error = {err:.2e}"


# ===========================================================================
# 4. Convergence rate of difference operators
# ===========================================================================

class TestConvergenceRates:

    def _measure_error(self, K, which_op, f_func, df_func):
        """Compute max interior error for a given operator on a test function."""
        x, y, dx = _make_test_grid(K)
        ops = build_difference_operators_fast(K, dx)

        f = f_func(x, y)
        analytic = df_func(x, y)
        numerical = apply_operator(ops[which_op], f, K)

        # Use generous interior slice to avoid boundary artifacts
        s = K // 4
        interior = slice(s, K - s), slice(s, K - s)
        return np.max(np.abs(numerical[interior] - analytic[interior]))

    @pytest.mark.parametrize("op_key,f,df", [
        ('D_u',
         lambda x, y: np.sin(2 * np.pi * x) * np.cos(np.pi * y),
         lambda x, y: 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(np.pi * y)),
        ('D_vv',
         lambda x, y: np.cos(np.pi * x) * np.sin(2 * np.pi * y),
         lambda x, y: -4 * np.pi**2 * np.cos(np.pi * x) * np.sin(2 * np.pi * y)),
        ('D_uv',
         lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
         lambda x, y: np.pi**2 * np.cos(np.pi * x) * np.cos(np.pi * y)),
    ])
    def test_second_order_convergence(self, op_key, f, df):
        """All operators should converge at O(dx^2)."""
        Ks = [40, 80, 160]
        errors = [self._measure_error(K, op_key, f, df) for K in Ks]

        rates = []
        for i in range(1, len(Ks)):
            if errors[i] > 0 and errors[i-1] > 0:
                rate = np.log(errors[i-1] / errors[i]) / np.log(Ks[i] / Ks[i-1])
                rates.append(rate)

        print(f"  {op_key}: errors={[f'{e:.2e}' for e in errors]}, rates={[f'{r:.2f}' for r in rates]}")
        for rate in rates:
            assert rate > 1.7, f"{op_key} convergence rate {rate:.2f} < 1.7 (expected ~2)"


# ===========================================================================
# 5. Fast vs loop operators agree
# ===========================================================================

class TestFastVsLoop:

    def test_operators_agree(self):
        """Kronecker-product and explicit-loop operators should be identical."""
        K = K_SMALL
        x, y, dx = _make_test_grid(K)
        ops_loop = build_difference_operators(K, dx)
        ops_fast = build_difference_operators_fast(K, dx)

        f = np.sin(np.pi * x) * np.cos(0.5 * np.pi * y)

        for key in ['D_u', 'D_v', 'D_uu', 'D_vv', 'D_uv']:
            r_loop = apply_operator(ops_loop[key], f, K)
            r_fast = apply_operator(ops_fast[key], f, K)
            np.testing.assert_allclose(
                r_loop, r_fast, atol=1e-13,
                err_msg=f"Operators disagree for {key}"
            )


# ===========================================================================
# 6. BC projection
# ===========================================================================

class TestBCProjection:

    def test_projection_shapes(self):
        K = 10
        P_int, P_bnd = build_bc_projection(K)
        n_int = (K - 2) ** 2
        n_bnd = 4 * K - 4
        assert P_int.shape == (n_int, K * K)
        assert P_bnd.shape == (n_bnd, K * K)

    def test_projection_extracts_correct_values(self):
        K = 8
        P_int, P_bnd = build_bc_projection(K)

        field = np.arange(K * K, dtype=float)
        int_vals = P_int @ field
        bnd_vals = P_bnd @ field

        # Verify values match
        int_idx = interior_indices(K)
        bnd_idx = boundary_indices(K)
        np.testing.assert_array_equal(int_vals, field[int_idx])
        np.testing.assert_array_equal(bnd_vals, field[bnd_idx])

    def test_projection_reconstruction(self):
        """P_int^T @ P_int + P_bnd^T @ P_bnd = I (orthogonal decomposition)."""
        K = 8
        P_int, P_bnd = build_bc_projection(K)
        N = K * K
        reconstructed = P_int.T @ P_int + P_bnd.T @ P_bnd
        np.testing.assert_allclose(reconstructed.toarray(), np.eye(N), atol=1e-14)


# ===========================================================================
# 7. Interior-restricted operators
# ===========================================================================

class TestInteriorRestriction:

    def test_restricted_shape(self):
        K = 10
        disc = setup_discretization(K, du=0.1)
        D_u_int = restrict_operator_to_interior(disc['ops']['D_u'], K, disc['P_int'])
        n_int = disc['n_int']
        assert D_u_int.shape == (n_int, n_int)

    def test_restricted_derivative_matches(self):
        """Restricted operator applied to interior DOFs should match
        full operator applied to the full field (at interior points)."""
        K = 20
        x, y, dx = _make_test_grid(K)
        disc = setup_discretization(K, dx)
        P_int = disc['P_int']

        f = x**2 * y  # D_u f = 2xy
        f_flat = flatten_field(f)
        f_int = P_int @ f_flat

        # Full operator result at interior
        full_result = disc['ops']['D_u'] @ f_flat
        full_at_int = P_int @ full_result

        # Restricted operator (ignoring BC contributions)
        D_u_int = restrict_operator_to_interior(disc['ops']['D_u'], K, P_int)
        restricted_result = D_u_int @ f_int

        # For points not adjacent to boundary, these should match
        # For boundary-adjacent interior points, the restricted op misses BC contributions
        # But for x^2 y, the boundary values are part of the polynomial, so
        # we also need to add the BC contribution
        bc_vec = make_bc_vector(f, K)
        bc_contrib = bc_rhs_contribution(disc['ops']['D_u'], bc_vec, K, P_int)

        total = restricted_result - bc_contrib  # note: bc_rhs_contribution returns -(P_int @ op @ bc_vec)
        np.testing.assert_allclose(total, full_at_int, atol=1e-12)


# ===========================================================================
# 8. BC RHS contribution
# ===========================================================================

class TestBCContribution:

    def test_bc_contribution_for_known_field(self):
        """
        For a field where we know the full solution, splitting into
        interior solve + BC contribution should recover the correct derivative.
        """
        K = 30
        x, y, dx = _make_test_grid(K)
        disc = setup_discretization(K, dx)
        P_int = disc['P_int']

        # f = x^2 + y^2, D_uu f = 2 everywhere
        f = x**2 + y**2
        analytic = 2.0 * np.ones((K, K))

        # Full operator
        full_result = apply_operator(disc['ops']['D_uu'], f, K)

        # Decomposed: restricted op on interior + BC contribution
        D_uu_int = restrict_operator_to_interior(disc['ops']['D_uu'], K, P_int)
        f_int = P_int @ flatten_field(f)
        bc_vec = make_bc_vector(f, K)
        bc_contrib = bc_rhs_contribution(disc['ops']['D_uu'], bc_vec, K, P_int)

        interior_result = D_uu_int @ f_int - bc_contrib
        full_at_int = P_int @ flatten_field(full_result)

        np.testing.assert_allclose(interior_result, full_at_int, atol=1e-12)


# ===========================================================================
# 9. Integration: Gauss residual via sparse operators
# ===========================================================================

class TestGaussWithSparseOps:

    def test_circle_metric_gauss_via_sparse(self):
        """
        Verify the circle metric satisfies the Gauss equation when derivatives
        are computed via sparse operators, matching the geometry module's test.
        """
        K = 200
        margin = 5e-2
        lin = np.linspace(-1.0 + margin, 1.0 - margin, K)
        x, y = np.meshgrid(lin, lin, indexing='ij')
        dx = lin[1] - lin[0]

        E, F, G = metric_circle(x, y)
        ops = build_difference_operators_fast(K, dx)

        d_E = compute_all_derivatives(ops, E, K)
        d_F = compute_all_derivatives(ops, F, K)
        d_G = compute_all_derivatives(ops, G, K)

        res = gauss_residual(
            E, F, G,
            d_E['u'], d_E['v'],
            d_G['u'], d_G['v'],
            d_F['u'], d_F['v'],
            d_E['vv'], d_F['uv'], d_G['uu'],
        )

        # Interior residual should be small
        s = K // 8
        interior = res[s:-s, s:-s]
        max_res = np.max(np.abs(interior))
        print(f"Gauss residual via sparse ops (K={K}): max |R| = {max_res:.2e}")
        assert max_res < 0.01


# ===========================================================================
# 10. setup_discretization integration
# ===========================================================================

class TestSetupDiscretization:

    def test_returned_keys(self):
        disc = setup_discretization(K_SMALL, du=0.1)
        expected_keys = {
            'K', 'du', 'dv', 'ops', 'P_int', 'P_bnd',
            'int_idx', 'bnd_idx', 'n_int', 'n_bnd',
            'int_mask', 'bnd_mask',
        }
        assert set(disc.keys()) == expected_keys

    def test_consistency(self):
        K = 15
        disc = setup_discretization(K, du=0.2)
        assert disc['n_int'] == (K - 2) ** 2
        assert disc['n_bnd'] == 4 * K - 4
        assert disc['n_int'] + disc['n_bnd'] == K * K
        assert disc['P_int'].shape == (disc['n_int'], K * K)


# ===========================================================================
# 11. Dirichlet BC application
# ===========================================================================

class TestDirichletBC:

    def test_overwrites_boundary_only(self):
        K = 10
        field = np.ones((K, K))
        bc = np.zeros((K, K))
        result = apply_dirichlet_bc(field.copy(), bc, K)

        # Interior should be unchanged
        assert np.all(result[1:-1, 1:-1] == 1.0)
        # Boundary should be overwritten
        bnd = boundary_mask(K)
        assert np.all(result[bnd] == 0.0)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
