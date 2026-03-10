"""
test_assembly.py — Tests for the sparse system assembly module.

Verifies:
  1. Gauss Jacobian coefficients match finite-difference Jacobian
  2. Strain eigenvalue Jacobian matches finite differences
  3. Full system has correct shape and structure
  4. BC enforcement produces identity rows at boundary DOFs
  5. Residuals are zero at known valid metrics (endpoints)
  6. Newton step from a_o at t=0 produces zero correction at BCs
  7. Assembled Jacobian acts correctly via directional derivative test
"""

import numpy as np
import scipy.sparse as sp
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from geometry import (
    metric_square, metric_circle, metric_det,
    gauss_residual, green_strain_eigenvalue_direct,
    build_endpoint_data, linear_interpolation,
)
from discretization import (
    build_difference_operators_fast, compute_all_derivatives,
    flatten_field, unflatten_field, boundary_indices, interior_indices,
    boundary_mask, setup_discretization,
)
from assembly import (
    gauss_jacobian_coefficients,
    assemble_gauss_jacobian,
    strain_eigenvalue_and_jacobian,
    assemble_algebraic_jacobian,
    evaluate_gauss_residual,
    assemble_system,
    assemble_system_with_bcs,
    solve_newton_step,
    unpack_delta,
    compute_residual_norms,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

K_TEST = 20
MARGIN = 0.05
FD_DELTA = 1e-6


@pytest.fixture
def setup():
    """Provide grid, endpoint data, and operators for a small test grid."""
    data = build_endpoint_data(K_TEST, margin=MARGIN)
    disc = setup_discretization(K_TEST, data['dx'])
    return data, disc


def _interior_flat_mask(K, border=3):
    """Boolean mask of length K² that is True at interior points away from border."""
    mask = np.zeros(K * K, dtype=bool)
    for i in range(border, K - border):
        for j in range(border, K - border):
            mask[i * K + j] = True
    return mask


# ===========================================================================
# 1. Gauss Jacobian via finite differences
# ===========================================================================

class TestGaussJacobian:

    def _gauss_residual_from_fields(self, eps, phi, gam, ops, K):
        res, _, _, _ = evaluate_gauss_residual(eps, phi, gam, ops, K)
        return res

    @pytest.mark.parametrize("field_idx,seed", [(0, 42), (1, 43), (2, 44)])
    def test_gauss_jacobian_fd(self, setup, field_idx, seed):
        """
        Verify each block of the Gauss Jacobian against finite differences.
        field_idx: 0=ε, 1=φ, 2=γ
        """
        data, disc = setup
        K = K_TEST
        ops = disc['ops']
        N = K * K

        eps, phi, gam = data['E'].copy(), data['F'].copy(), data['G'].copy()

        d_eps = compute_all_derivatives(ops, eps, K)
        d_phi = compute_all_derivatives(ops, phi, K)
        d_gam = compute_all_derivatives(ops, gam, K)
        coeffs = gauss_jacobian_coefficients(eps, phi, gam, d_eps, d_phi, d_gam)
        J_gauss = assemble_gauss_jacobian(coeffs, ops, K)

        np.random.seed(seed)
        delta = np.random.randn(K, K) * 0.01

        # Build direction vector with perturbation only in the target field
        dX = np.zeros(3 * N)
        dX[field_idx * N:(field_idx + 1) * N] = flatten_field(delta)
        J_dir = J_gauss @ dX

        # Finite difference
        h = FD_DELTA
        fields = [eps.copy(), phi.copy(), gam.copy()]
        fields[field_idx] = fields[field_idx] + h * delta
        R_plus = self._gauss_residual_from_fields(*fields, ops, K)
        fields[field_idx] = fields[field_idx] - 2 * h * delta
        R_minus = self._gauss_residual_from_fields(*fields, ops, K)
        fd_dir = flatten_field((R_plus - R_minus) / (2.0 * h))

        mask = _interior_flat_mask(K, border=3)
        err = np.max(np.abs(J_dir[mask] - fd_dir[mask]))
        scale = np.max(np.abs(fd_dir[mask])) + 1e-30
        rel_err = err / scale

        names = ['eps', 'phi', 'gam']
        print(f"Gauss J_{names[field_idx]}: abs err = {err:.2e}, rel = {rel_err:.2e}")
        assert rel_err < 1e-3 or err < 1e-6


# ===========================================================================
# 2. Strain eigenvalue Jacobian via finite differences
# ===========================================================================

class TestStrainJacobian:

    def test_strain_jacobian_fd(self, setup):
        """Verify strain eigenvalue derivatives against centered FD."""
        data, _ = setup
        e, f, g = data['e'], data['f'], data['g']

        eps = 0.5 * (data['e'] + data['E'])
        phi = 0.5 * (data['f'] + data['F'])
        gam = 0.5 * (data['g'] + data['G'])

        lam, dl_de, dl_dp, dl_dg = strain_eigenvalue_and_jacobian(e, f, g, eps, phi, gam)

        h = FD_DELTA
        for field_name, dl_analytic, perturb in [
            ('eps', dl_de, (h, 0, 0)),
            ('phi', dl_dp, (0, h, 0)),
            ('gam', dl_dg, (0, 0, h)),
        ]:
            lam_p = green_strain_eigenvalue_direct(
                e, f, g, eps + perturb[0], phi + perturb[1], gam + perturb[2])
            lam_m = green_strain_eigenvalue_direct(
                e, f, g, eps - perturb[0], phi - perturb[1], gam - perturb[2])
            fd = (lam_p - lam_m) / (2.0 * h)
            np.testing.assert_allclose(
                dl_analytic, fd, atol=1e-4, rtol=1e-4,
                err_msg=f"Strain Jacobian mismatch for {field_name}",
            )

    def test_strain_jacobian_at_identity(self, setup):
        """When current = reference, eigenvalue is 0 and derivatives are finite."""
        data, _ = setup
        e, f, g = data['e'], data['f'], data['g']
        lam, dl_de, dl_dp, dl_dg = strain_eigenvalue_and_jacobian(e, f, g, e, f, g)
        np.testing.assert_allclose(lam, 0.0, atol=1e-12)
        assert np.all(np.isfinite(dl_de))
        assert np.all(np.isfinite(dl_dp))
        assert np.all(np.isfinite(dl_dg))


# ===========================================================================
# 3. System shape and structure
# ===========================================================================

class TestSystemStructure:

    def test_system_shape(self, setup):
        data, disc = setup
        K = K_TEST
        N = K * K

        J, neg_R = assemble_system(
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.0, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
        )
        assert J.shape == (3 * N, 3 * N)
        assert neg_R.shape == (3 * N,)

    def test_system_sparse(self, setup):
        data, disc = setup
        K = K_TEST
        N = K * K

        J, _ = assemble_system(
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.0, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
        )
        density = J.nnz / (3 * N) ** 2
        print(f"System density: {density:.4f} ({J.nnz} nnz of {(3*N)**2})")
        assert density < 0.1

    def test_algebraic_rows_are_diagonal(self, setup):
        """Algebraic constraint rows should have at most 3 nnz each."""
        data, disc = setup
        K = K_TEST
        N = K * K

        J, _ = assemble_system(
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.5, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
        )
        J_alg = J[N:, :]
        nnz_per_row = np.diff(J_alg.indptr)
        assert np.all(nnz_per_row <= 3)


# ===========================================================================
# 4. Boundary condition enforcement
# ===========================================================================

class TestBCEnforcement:

    def test_bc_rows_are_identity(self, setup):
        data, disc = setup
        K = K_TEST
        N = K * K

        J, neg_R = assemble_system_with_bcs(
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.5, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
            data['e'], data['f'], data['g'],
        )

        bnd_idx = boundary_indices(K)
        for block in range(3):
            offset = block * N
            for idx in bnd_idx:
                row = offset + idx
                row_data = J[row, :].toarray().ravel()
                assert abs(row_data[offset + idx] - 1.0) < 1e-14
                row_data[offset + idx] = 0.0
                assert np.max(np.abs(row_data)) < 1e-14
                assert abs(neg_R[row]) < 1e-14

    def test_bc_preserves_interior_rows(self, setup):
        data, disc = setup
        K = K_TEST
        N = K * K

        common_args = (
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.5, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
        )

        J_orig, _ = assemble_system(*common_args)
        J_bc, _ = assemble_system_with_bcs(
            *common_args,
            data['e'], data['f'], data['g'],
        )

        int_idx = interior_indices(K)
        for idx in int_idx[:10]:
            row_orig = J_orig[idx, :].toarray().ravel()
            row_bc = J_bc[idx, :].toarray().ravel()
            np.testing.assert_allclose(row_orig, row_bc, atol=1e-14)


# ===========================================================================
# 5. Residuals at endpoints
# ===========================================================================

class TestEndpointResiduals:

    def test_gauss_residual_zero_for_identity(self, setup):
        data, disc = setup
        K = K_TEST

        norms = compute_residual_norms(
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.0, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
        )
        assert norms['gauss'] < 1e-12

    def test_gauss_residual_small_for_circle(self, setup):
        data, disc = setup
        K = K_TEST

        norms = compute_residual_norms(
            data['E'], data['F'], data['G'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            1.0, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
        )
        assert norms['gauss'] < 1.0  # limited by K=20 FD accuracy


# ===========================================================================
# 6. Newton step sanity
# ===========================================================================

class TestNewtonStep:

    def test_boundary_corrections_zero(self, setup):
        data, disc = setup
        K = K_TEST

        J, neg_R = assemble_system_with_bcs(
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.0, data['lam_of'], data['lam_fo'],
            disc['ops'], K,
            data['e'], data['f'], data['g'],
        )

        delta_X = solve_newton_step(J, neg_R)
        d_eps, d_phi, d_gam = unpack_delta(delta_X, K)

        assert np.all(np.isfinite(d_eps))
        assert np.all(np.isfinite(d_phi))
        assert np.all(np.isfinite(d_gam))

        bnd = boundary_mask(K)
        np.testing.assert_allclose(d_eps[bnd], 0.0, atol=1e-10)
        np.testing.assert_allclose(d_phi[bnd], 0.0, atol=1e-10)
        np.testing.assert_allclose(d_gam[bnd], 0.0, atol=1e-10)

    def test_unpack_delta_roundtrip(self):
        K = K_TEST
        N = K * K
        vec = np.arange(3 * N, dtype=float)
        a, b, c = unpack_delta(vec, K)
        assert a.shape == (K, K)
        np.testing.assert_array_equal(flatten_field(a), vec[:N])
        np.testing.assert_array_equal(flatten_field(b), vec[N:2*N])
        np.testing.assert_array_equal(flatten_field(c), vec[2*N:3*N])


# ===========================================================================
# 7. Full Jacobian directional derivative
# ===========================================================================

class TestFullJacobianFD:

    def test_full_jacobian_directional_derivative(self, setup):
        """
        J @ δX ≈ (R(X + h δX) - R(X - h δX)) / (2h) at a non-trivial point.
        """
        data, disc = setup
        K = K_TEST
        N = K * K
        ops = disc['ops']
        t = 0.3

        np.random.seed(99)
        eps = 0.7 * data['e'] + 0.3 * data['E'] + 0.001 * np.random.randn(K, K)
        phi = 0.7 * data['f'] + 0.3 * data['F'] + 0.001 * np.random.randn(K, K)
        gam = 0.7 * data['g'] + 0.3 * data['G'] + 0.001 * np.random.randn(K, K)

        J, neg_R = assemble_system(
            eps, phi, gam,
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            t, data['lam_of'], data['lam_fo'],
            ops, K,
        )

        dX = np.random.randn(3 * N) * 0.001
        d_eps = unflatten_field(dX[:N], K)
        d_phi = unflatten_field(dX[N:2*N], K)
        d_gam = unflatten_field(dX[2*N:], K)

        Jd = J @ dX

        h = 1e-6

        def eval_residual(e_, p_, g_):
            R_g, _, _, _ = evaluate_gauss_residual(e_, p_, g_, ops, K)
            lam_f = green_strain_eigenvalue_direct(
                data['e'], data['f'], data['g'], e_, p_, g_)
            R1 = linear_interpolation(data['lam_of'], t) - lam_f
            lam_r = green_strain_eigenvalue_direct(
                data['E'], data['F'], data['G'], e_, p_, g_)
            R2 = linear_interpolation(data['lam_fo'], 1.0 - t) - lam_r
            return np.concatenate([
                flatten_field(R_g), flatten_field(R1), flatten_field(R2)
            ])

        R_plus = eval_residual(eps + h * d_eps, phi + h * d_phi, gam + h * d_gam)
        R_minus = eval_residual(eps - h * d_eps, phi - h * d_phi, gam - h * d_gam)
        fd_dir = (R_plus - R_minus) / (2.0 * h)

        gauss_mask = _interior_flat_mask(K, border=3)
        full_mask = np.concatenate([gauss_mask, np.ones(2 * N, dtype=bool)])

        err = np.max(np.abs(Jd[full_mask] - fd_dir[full_mask]))
        scale = np.max(np.abs(fd_dir[full_mask])) + 1e-30
        rel_err = err / scale

        print(f"Full Jacobian directional deriv: abs err = {err:.2e}, rel = {rel_err:.2e}")
        assert rel_err < 5e-3 or err < 1e-5


# ===========================================================================
# 8. Solver method options
# ===========================================================================

class TestSolverMethods:

    def _build_test_system(self, setup):
        data, disc = setup
        return assemble_system_with_bcs(
            data['e'], data['f'], data['g'],
            data['e'], data['f'], data['g'],
            data['E'], data['F'], data['G'],
            0.5, data['lam_of'], data['lam_fo'],
            disc['ops'], K_TEST,
            data['e'], data['f'], data['g'],
        )

    def test_spsolve_runs(self, setup):
        J, neg_R = self._build_test_system(setup)
        delta_X = solve_newton_step(J, neg_R, method='spsolve')
        assert np.all(np.isfinite(delta_X))

    def test_gmres_runs(self, setup):
        J, neg_R = self._build_test_system(setup)
        delta_X = solve_newton_step(J, neg_R, method='gmres')
        assert np.all(np.isfinite(delta_X))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            solve_newton_step(sp.eye(4), np.zeros(4), method='magic')


# ===========================================================================
# Run
# ===========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
