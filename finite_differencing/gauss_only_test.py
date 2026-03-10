"""
Simple test: interpolate metrics from square to circle while
satisfying only the Gauss equation (K=0). No algebraic constraints.

Since we have N Gauss equations for 3N unknowns, the system is
underdetermined. We use minimum-norm Newton: at each step, find
the smallest correction that reduces the Gauss residual.

    J_gauss @ dX = -R_gauss    (N x 3N, underdetermined)
    dX = J^T (J J^T)^{-1} (-R)  (minimum-norm solution)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from geometry import build_endpoint_data, gauss_residual
from discretization import setup_discretization, boundary_indices, flatten_field, unflatten_field
from assembly import (
    evaluate_gauss_residual, gauss_jacobian_coefficients,
    assemble_gauss_jacobian, unpack_delta,
)
from solver import interpolate_boundary_metric


def solve_gauss_only(K=25, n_time=21, max_newton=30, tol=1e-6, margin=0.05, verbose=True):
    """
    March from t=0 to t=1, at each t solving only Gauss = 0
    starting from the linearly interpolated metric.
    """
    data = build_endpoint_data(K, margin=margin)
    disc = setup_discretization(K, data['dx'])
    ops = disc['ops']
    N = K * K
    bnd = boundary_indices(K)
    bnd_set = set(bnd)
    int_idx = sorted(set(range(N)) - bnd_set)
    n_int = len(int_idx)

    times = np.linspace(0, 1, n_time)
    results = []

    for ti, t in enumerate(times):
        # Initial guess: linearly interpolated metric
        eps = (1 - t) * data['e'] + t * data['E']
        phi = (1 - t) * data['f'] + t * data['F']
        gam = (1 - t) * data['g'] + t * data['G']

        # BCs (same as initial guess on boundary)
        bc_eps, bc_phi, bc_gam = eps.copy(), phi.copy(), gam.copy()

        # Evaluate initial Gauss residual
        R0, _, _, _ = evaluate_gauss_residual(eps, phi, gam, ops, K)
        r0 = np.max(np.abs(R0[2:-2, 2:-2]))  # interior only

        converged = False
        for it in range(max_newton):
            # Evaluate residual and build Gauss Jacobian
            R, d_eps, d_phi, d_gam = evaluate_gauss_residual(eps, phi, gam, ops, K)
            coeffs = gauss_jacobian_coefficients(eps, phi, gam, d_eps, d_phi, d_gam)
            J_gauss = assemble_gauss_jacobian(coeffs, ops, K)  # N x 3N

            # Zero out boundary rows in J and R (BCs are fixed)
            R_flat = flatten_field(R)
            for b in bnd:
                R_flat[b] = 0.0

            J_gauss = J_gauss.tolil()
            for b in bnd:
                J_gauss[b, :] = 0.0
            J_gauss = J_gauss.tocsr()

            # Also zero out columns corresponding to boundary DOFs
            # (we don't want to perturb boundary values)
            J_gauss = J_gauss.tocsc()
            for block in range(3):
                for b in bnd:
                    J_gauss[:, block * N + b] = 0.0
            J_gauss = J_gauss.tocsr()

            r_cur = np.max(np.abs(R_flat))
            if r_cur < tol:
                converged = True
                break

            # Minimum-norm solve: dX = J^T (J J^T + mu I)^{-1} (-R)
            # Add small regularization to J J^T for numerical stability
            mu = 1e-10 * r_cur
            JJt = J_gauss @ J_gauss.T + mu * sp.eye(N, format='csr')
            z = spla.spsolve(JJt, -R_flat)
            dX = J_gauss.T @ z

            d_e = unflatten_field(dX[:N], K)
            d_p = unflatten_field(dX[N:2*N], K)
            d_g = unflatten_field(dX[2*N:], K)

            # Line search
            best_alpha, best_r = 0.0, r_cur
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                e_try = eps + alpha * d_e
                p_try = phi + alpha * d_p
                g_try = gam + alpha * d_g

                # Enforce BCs
                for b in bnd:
                    i, j = b // K, b % K
                    e_try[i, j] = bc_eps[i, j]
                    p_try[i, j] = bc_phi[i, j]
                    g_try[i, j] = bc_gam[i, j]

                # Check metric positivity
                det = e_try * g_try - p_try**2
                if np.any(det <= 0):
                    continue

                R_try, _, _, _ = evaluate_gauss_residual(e_try, p_try, g_try, ops, K)
                R_try_flat = flatten_field(R_try)
                for b in bnd:
                    R_try_flat[b] = 0.0
                r_try = np.max(np.abs(R_try_flat))

                if r_try < best_r:
                    best_r = r_try
                    best_alpha = alpha

            if best_alpha == 0:
                break

            eps = eps + best_alpha * d_e
            phi = phi + best_alpha * d_p
            gam = gam + best_alpha * d_g
            for b in bnd:
                i, j = b // K, b % K
                eps[i, j] = bc_eps[i, j]
                phi[i, j] = bc_phi[i, j]
                gam[i, j] = bc_gam[i, j]

        # Final residual
        R_final, _, _, _ = evaluate_gauss_residual(eps, phi, gam, ops, K)
        r_final = np.max(np.abs(flatten_field(R_final)[list(set(range(N)) - bnd_set)]))
        det = eps * gam - phi**2

        results.append({
            't': t,
            'eps': eps.copy(), 'phi': phi.copy(), 'gam': gam.copy(),
            'gauss_init': r0,
            'gauss_final': r_final,
            'newton_iters': it + 1,
            'converged': converged,
            'det_min': det.min(),
            'det_max': det.max(),
        })

        if verbose:
            tag = "CONV" if converged else f"{it+1:3d}it"
            print(f"  t={t:.3f}: {tag}  Gauss: {r0:.3e} -> {r_final:.3e}  "
                  f"det=[{det.min():.4f}, {det.max():.4f}]")

    return results


def to_trajectory_result(results, K=25, margin=0.15):
    """Convert Gauss-only results to a TrajectoryResult for visualization."""
    from solver import SolverConfig, Snapshot, TrajectoryResult

    config = SolverConfig(K=K, margin=margin)
    data = build_endpoint_data(K, margin=margin)

    snapshots = []
    for r in results:
        snap = Snapshot(
            t=r['t'],
            eps=r['eps'],
            phi=r['phi'],
            gam=r['gam'],
            residual_norms={
                'gauss': r['gauss_final'],
                'alg_fwd': 0.0,
                'alg_rev': 0.0,
                'total': r['gauss_final'],
            },
            newton_iters=r['newton_iters'],
        )
        snapshots.append(snap)

    all_converged = all(r['converged'] for r in results)
    return TrajectoryResult(
        snapshots=snapshots,
        config=config,
        endpoint_data=data,
        converged=all_converged,
        message="Gauss-only solve (no algebraic constraints)",
    )


if __name__ == '__main__':
    print("=== Gauss-only solve, K=25 ===\n")
    K = 25
    results = solve_gauss_only(K=K, n_time=21, max_newton=30, tol=1e-6, margin=0.05)

    print(f"\n--- Summary ---")
    conv = sum(r['converged'] for r in results)
    gauss_finals = [r['gauss_final'] for r in results]
    print(f"  Converged: {conv}/{len(results)}")
    print(f"  Gauss final: max={max(gauss_finals):.4e}, mean={np.mean(gauss_finals):.4e}")
    print(f"  det min: {min(r['det_min'] for r in results):.4f}")

    # Generate plots
    traj = to_trajectory_result(results, K=K)

    from visualization import generate_all_plots
    output_dir = './plots'
    paths = generate_all_plots(traj, output_dir=output_dir, prefix='gauss_only')
    print(f"\nPlots saved:")
    for p in paths:
        print(f"  {p}")