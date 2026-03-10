"""
gauss_only_fixed_tensor.py — Gauss-only solve with selectable fixed components.

Allows fixing any combination of (ε, φ, γ) to their linearly interpolated
values while solving the Gauss equation using the remaining free components.

Examples:
    # All free (same as gauss_only_test.py)
    solve_gauss_fixed(free='epg')

    # Fix ε, solve with φ and γ
    solve_gauss_fixed(free='pg')

    # Fix φ and γ, solve with ε only
    solve_gauss_fixed(free='e')
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from geometry import build_endpoint_data, gauss_residual
from discretization import (
    setup_discretization, boundary_indices,
    flatten_field, unflatten_field,
)
from assembly import (
    evaluate_gauss_residual, gauss_jacobian_coefficients,
    assemble_gauss_jacobian, unpack_delta,
)


def solve_gauss_fixed(K=25, n_time=21, free='epg', max_newton=30, tol=1e-6,
                      margin=0.05, verbose=True):
    """
    Solve the Gauss equation starting from the linearly interpolated metric,
    allowing only the components listed in `free` to vary.

    Parameters
    ----------
    K : int
        Grid size.
    n_time : int
        Number of time steps (uniformly spaced in [0, 1]).
    free : str
        Which metric components are free to vary. Any combination of:
        'e' (ε), 'p' (φ), 'g' (γ). E.g. 'epg' = all free, 'eg' = fix φ.
    max_newton : int
        Maximum Newton iterations per time step.
    tol : float
        Convergence tolerance on the Gauss residual (l-inf, interior).
    margin : float
        Grid margin parameter.
    verbose : bool
        Print per-step diagnostics.

    Returns
    -------
    results : list of dict
        Per-time-step results with metrics, residuals, convergence info.
    """
    free = free.lower()
    free_e = 'e' in free
    free_p = 'p' in free
    free_g = 'g' in free
    n_free = sum([free_e, free_p, free_g])

    if n_free == 0:
        raise ValueError("At least one component must be free.")

    label = []
    if free_e: label.append('ε')
    if free_p: label.append('φ')
    if free_g: label.append('γ')
    label = ', '.join(label)

    if verbose:
        print(f"Free components: {label} ({n_free}/3)")
        print(f"Fixed components: "
              f"{', '.join(c for c, f in [('ε', free_e), ('φ', free_p), ('γ', free_g)] if not f) or 'none'}\n")

    data = build_endpoint_data(K, margin=margin)
    disc = setup_discretization(K, data['dx'])
    ops = disc['ops']
    N = K * K
    bnd = boundary_indices(K)
    bnd_set = set(bnd)
    int_idx = sorted(set(range(N)) - bnd_set)

    times = np.linspace(0, 1, n_time)
    results = []

    for ti, t in enumerate(times):
        # Initial guess: linearly interpolated metric
        eps = (1 - t) * data['e'] + t * data['E']
        phi = (1 - t) * data['f'] + t * data['F']
        gam = (1 - t) * data['g'] + t * data['G']

        bc_eps, bc_phi, bc_gam = eps.copy(), phi.copy(), gam.copy()

        # Initial residual
        R0, _, _, _ = evaluate_gauss_residual(eps, phi, gam, ops, K)
        R0_flat = flatten_field(R0)
        for b in bnd:
            R0_flat[b] = 0.0
        r0 = np.max(np.abs(R0_flat))

        converged = False
        for it in range(max_newton):
            # Evaluate residual and Jacobian
            R, d_eps, d_phi, d_gam = evaluate_gauss_residual(eps, phi, gam, ops, K)
            coeffs = gauss_jacobian_coefficients(eps, phi, gam, d_eps, d_phi, d_gam)
            J_full = assemble_gauss_jacobian(coeffs, ops, K)  # N x 3N

            R_flat = flatten_field(R)
            for b in bnd:
                R_flat[b] = 0.0

            r_cur = np.max(np.abs(R_flat))
            if r_cur < tol:
                converged = True
                break

            # Extract columns for free components only
            col_blocks = []
            if free_e: col_blocks.append(J_full[:, :N])
            if free_p: col_blocks.append(J_full[:, N:2*N])
            if free_g: col_blocks.append(J_full[:, 2*N:])
            J_free = sp.hstack(col_blocks, format='csr')  # N x (n_free * N)

            # Zero out boundary rows in J and R
            J_free = J_free.tolil()
            for b in bnd:
                J_free[b, :] = 0.0
            J_free = J_free.tocsr()

            # Zero out columns for boundary DOFs in each free block
            J_free = J_free.tocsc()
            for block_idx in range(n_free):
                for b in bnd:
                    J_free[:, block_idx * N + b] = 0.0
            J_free = J_free.tocsr()

            # Minimum-norm solve: dX = J^T (J J^T + mu I)^{-1} (-R)
            mu = 1e-10 * max(r_cur, 1e-12)
            JJt = J_free @ J_free.T + mu * sp.eye(N, format='csr')
            z = spla.spsolve(JJt, -R_flat)
            dX_free = J_free.T @ z

            # Unpack into per-component corrections
            d_e = np.zeros((K, K))
            d_p = np.zeros((K, K))
            d_g = np.zeros((K, K))

            idx = 0
            if free_e:
                d_e = unflatten_field(dX_free[idx*N:(idx+1)*N], K)
                idx += 1
            if free_p:
                d_p = unflatten_field(dX_free[idx*N:(idx+1)*N], K)
                idx += 1
            if free_g:
                d_g = unflatten_field(dX_free[idx*N:(idx+1)*N], K)
                idx += 1

            # Line search
            best_alpha, best_r = 0.0, r_cur
            for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
                e_try = eps + alpha * d_e
                p_try = phi + alpha * d_p
                g_try = gam + alpha * d_g

                for b in bnd:
                    i, j = b // K, b % K
                    e_try[i, j] = bc_eps[i, j]
                    p_try[i, j] = bc_phi[i, j]
                    g_try[i, j] = bc_gam[i, j]

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
        R_final_flat = flatten_field(R_final)
        for b in bnd:
            R_final_flat[b] = 0.0
        r_final = np.max(np.abs(R_final_flat))
        det = eps * gam - phi**2

        results.append({
            't': t,
            'eps': eps.copy(), 'phi': phi.copy(), 'gam': gam.copy(),
            'gauss_init': r0,
            'gauss_final': r_final,
            'newton_iters': it + 1,
            'converged': r_final < tol,
            'det_min': det.min(),
            'det_max': det.max(),
        })

        if verbose:
            tag = "CONV" if r_final < tol else f"{it+1:3d}it"
            print(f"  t={t:.3f}: {tag}  Gauss: {r0:.3e} -> {r_final:.3e}  "
                  f"det=[{det.min():.4f}, {det.max():.4f}]")

    return results


def to_trajectory_result(results, K=25, margin=0.15):
    """Convert results to a TrajectoryResult for visualization."""
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
        message="Gauss-only solve (fixed tensor test)",
    )


def run_all_combinations(K=25, n_time=11, max_newton=30, tol=1e-6):
    """Run all 7 non-empty combinations of free components."""
    combos = ['e', 'p', 'g', 'ep', 'eg', 'pg', 'epg']
    labels = {
        'e': 'ε only', 'p': 'φ only', 'g': 'γ only',
        'ep': 'ε+φ', 'eg': 'ε+γ', 'pg': 'φ+γ',
        'epg': 'all free',
    }

    print(f"{'Combo':<10} {'Label':<10} {'Conv':>5} {'Max Gauss':>12} {'Mean Gauss':>12}")
    print("-" * 55)

    all_results = {}
    for combo in combos:
        results = solve_gauss_fixed(
            K=K, n_time=n_time, free=combo,
            max_newton=max_newton, tol=tol, verbose=False,
        )
        all_results[combo] = results

        conv = sum(r['converged'] for r in results)
        gauss = [r['gauss_final'] for r in results]
        print(f"{combo:<10} {labels[combo]:<10} {conv:>3}/{len(results)}"
              f"  {max(gauss):>12.4e}  {np.mean(gauss):>12.4e}")

    return all_results


if __name__ == '__main__':
    print("=== Testing all combinations of fixed/free components ===\n")
    all_results = run_all_combinations(K=25, n_time=21, max_newton=30, tol=1e-6)

    # Generate plots for each combination
    from visualization import generate_all_plots
    output_dir = './plots'

    for combo, results in all_results.items():
        traj = to_trajectory_result(results, K=25)
        paths = generate_all_plots(traj, output_dir=output_dir,
                                   prefix=f'gauss_fixed_{combo}')
        n_conv = sum(r['converged'] for r in results)
        print(f"\n  {combo}: {len(paths)} plots saved, {n_conv}/{len(results)} converged")