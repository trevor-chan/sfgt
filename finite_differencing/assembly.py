"""
assembly.py — Sparse system assembly for the stress-free growth PDAE solver.

Assembles the linearized 3K² × 3K² system at each time step:

    [ J_gauss ]           [ -R_gauss ]
    [ J_alg1  ] @ δX   =  [ -R_alg1  ]
    [ J_alg2  ]           [ -R_alg2  ]

where δX = [δε, δφ, δγ] (flattened, length 3K²) are perturbations to the
current metric, and the rows come from:
  - J_gauss : linearized Gauss equation (eq. 10, corrected Brioschi form)
  - J_alg1  : linearized forward strain constraint (eq. 11)
  - J_alg2  : linearized reverse strain constraint (eq. 12)

The Gauss Jacobian is constructed as:
    J_gauss = [J_ε | J_φ | J_γ]
where each block is assembled from pointwise coefficients × sparse difference
operators (chain rule through the spatial derivatives).
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple, Callable

from geometry import (
    metric_det,
    gauss_residual,
    green_strain_eigenvalue_direct,
    linear_interpolation,
)
from discretization import (
    flatten_field, unflatten_field,
    compute_all_derivatives,
    apply_operator,
)


# =============================================================================
# Gauss equation Jacobian coefficients
# =============================================================================

def gauss_jacobian_coefficients(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    d_eps: Dict[str, np.ndarray],
    d_phi: Dict[str, np.ndarray],
    d_gam: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute the pointwise partial derivatives of the (corrected) Gauss residual
    R with respect to each field value and each spatial derivative.

    The corrected Gauss equation (Brioschi form, K=0) is:

    R = (-2ε_vv + 4φ_uv - 2γ_uu)(εγ - φ²)
        - ε_u (2φ_v γ - γ_u γ - φ γ_v)
        + (2φ_u - ε_v)(2φ_v φ - γ_u φ - ε γ_v)
        + ε_v² γ - 2 ε_v γ_u φ + γ_u² ε

    Parameters
    ----------
    eps, phi, gam : ndarray (K, K)
        Current metric field values.
    d_eps, d_phi, d_gam : dict
        Spatial derivatives, each with keys 'u', 'v', 'uu', 'vv', 'uv'.

    Returns
    -------
    coeffs : dict
        Keys are 'dR_deps', 'dR_dphi', 'dR_dgam' (field value partials),
        'dR_deps_u', 'dR_deps_v', 'dR_dphi_u', 'dR_dphi_v',
        'dR_dgam_u', 'dR_dgam_v' (first derivative partials),
        'dR_deps_vv', 'dR_dphi_uv', 'dR_dgam_uu' (second derivative partials).
        Each value is an ndarray (K, K).
    """
    eps_u = d_eps['u']
    eps_v = d_eps['v']
    phi_u = d_phi['u']
    phi_v = d_phi['v']
    gam_u = d_gam['u']
    gam_v = d_gam['v']
    eps_vv = d_eps['vv']
    phi_uv = d_phi['uv']
    gam_uu = d_gam['uu']

    D = eps * gam - phi**2  # metric determinant
    S = -2.0 * eps_vv + 4.0 * phi_uv - 2.0 * gam_uu  # second-derivative combo

    # Bracket terms that appear repeatedly
    A = 2.0 * phi_v * gam - gam_u * gam - phi * gam_v    # in T2
    B = 2.0 * phi_v * phi - gam_u * phi - eps * gam_v     # in T3
    C = 2.0 * phi_u - eps_v                                # coefficient in T3

    # --- Partials w.r.t. field values ---

    # ∂R/∂ε
    dR_deps = S * gam - C * gam_v + gam_u**2

    # ∂R/∂φ
    dR_dphi = -2.0 * S * phi + eps_u * gam_v + C * (2.0 * phi_v - gam_u) - 2.0 * eps_v * gam_u

    # ∂R/∂γ
    dR_dgam = S * eps - eps_u * (2.0 * phi_v - gam_u) + eps_v**2

    # --- Partials w.r.t. first derivatives ---

    # ∂R/∂ε_u
    dR_deps_u = -A

    # ∂R/∂ε_v
    dR_deps_v = -B + 2.0 * eps_v * gam - 2.0 * gam_u * phi

    # ∂R/∂φ_u
    dR_dphi_u = 2.0 * B

    # ∂R/∂φ_v
    dR_dphi_v = -2.0 * eps_u * gam + 2.0 * C * phi

    # ∂R/∂γ_u
    dR_dgam_u = eps_u * gam - C * phi - 2.0 * eps_v * phi + 2.0 * gam_u * eps

    # ∂R/∂γ_v
    dR_dgam_v = eps_u * phi - C * eps

    # --- Partials w.r.t. second derivatives ---

    # ∂R/∂ε_vv
    dR_deps_vv = -2.0 * D

    # ∂R/∂φ_uv
    dR_dphi_uv = 4.0 * D

    # ∂R/∂γ_uu
    dR_dgam_uu = -2.0 * D

    return {
        'dR_deps': dR_deps, 'dR_dphi': dR_dphi, 'dR_dgam': dR_dgam,
        'dR_deps_u': dR_deps_u, 'dR_deps_v': dR_deps_v,
        'dR_dphi_u': dR_dphi_u, 'dR_dphi_v': dR_dphi_v,
        'dR_dgam_u': dR_dgam_u, 'dR_dgam_v': dR_dgam_v,
        'dR_deps_vv': dR_deps_vv, 'dR_dphi_uv': dR_dphi_uv, 'dR_dgam_uu': dR_dgam_uu,
    }


def _diag(arr: np.ndarray) -> sp.csr_matrix:
    """Build a sparse diagonal matrix from a flattened 2D array."""
    return sp.diags(arr.ravel(), 0, format='csr')


def assemble_gauss_jacobian(
    coeffs: Dict[str, np.ndarray],
    ops: Dict[str, sp.csr_matrix],
    K: int,
) -> sp.csr_matrix:
    """
    Assemble the K² × 3K² Gauss Jacobian from pointwise coefficients
    and sparse difference operators.

    The Jacobian has block structure [J_ε | J_φ | J_γ] where:

        J_ε = diag(∂R/∂ε) + diag(∂R/∂ε_u)·D_u + diag(∂R/∂ε_v)·D_v + diag(∂R/∂ε_vv)·D_vv
        J_φ = diag(∂R/∂φ) + diag(∂R/∂φ_u)·D_u + diag(∂R/∂φ_v)·D_v + diag(∂R/∂φ_uv)·D_uv
        J_γ = diag(∂R/∂γ) + diag(∂R/∂γ_u)·D_u + diag(∂R/∂γ_v)·D_v + diag(∂R/∂γ_uu)·D_uu

    Parameters
    ----------
    coeffs : dict from gauss_jacobian_coefficients
    ops : dict of sparse difference operators
    K : int

    Returns
    -------
    J_gauss : sparse (K², 3K²)
    """
    D_u = ops['D_u']
    D_v = ops['D_v']
    D_uu = ops['D_uu']
    D_vv = ops['D_vv']
    D_uv = ops['D_uv']

    # Block for δε
    J_eps = (
        _diag(coeffs['dR_deps'])
        + _diag(coeffs['dR_deps_u']) @ D_u
        + _diag(coeffs['dR_deps_v']) @ D_v
        + _diag(coeffs['dR_deps_vv']) @ D_vv
    )

    # Block for δφ
    J_phi = (
        _diag(coeffs['dR_dphi'])
        + _diag(coeffs['dR_dphi_u']) @ D_u
        + _diag(coeffs['dR_dphi_v']) @ D_v
        + _diag(coeffs['dR_dphi_uv']) @ D_uv
    )

    # Block for δγ
    J_gam = (
        _diag(coeffs['dR_dgam'])
        + _diag(coeffs['dR_dgam_u']) @ D_u
        + _diag(coeffs['dR_dgam_v']) @ D_v
        + _diag(coeffs['dR_dgam_uu']) @ D_uu
    )

    return sp.hstack([J_eps, J_phi, J_gam], format='csr')


# =============================================================================
# Algebraic constraint Jacobians
# =============================================================================

def strain_eigenvalue_and_jacobian(
    e_r: np.ndarray, f_r: np.ndarray, g_r: np.ndarray,
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the dominant eigenvalue Λ of the Green strain tensor and its
    partial derivatives with respect to (ε, φ, γ).

    S = (1/2) a_r^{-1} (a_c - a_r)

    The eigenvalues are λ± = (tr_S ± √(tr_S² - 4 det_S)) / 2.
    We select Λ = max(|λ+|, |λ-|) and return its derivatives.

    Parameters
    ----------
    e_r, f_r, g_r : ndarray
        Reference metric components.
    eps, phi, gam : ndarray
        Current metric components.

    Returns
    -------
    lam : ndarray — dominant eigenvalue
    dlam_deps, dlam_dphi, dlam_dgam : ndarray — partial derivatives
    """
    det_r = metric_det(e_r, f_r, g_r)

    # Components of M = a_r^{-1} (a_c - a_r) * det_r
    de = eps - e_r
    df = phi - f_r
    dg = gam - g_r

    M11 = g_r * de - f_r * df
    M12 = g_r * df - f_r * dg
    M21 = -f_r * de + e_r * df
    M22 = -f_r * df + e_r * dg

    # Trace and det of M (before dividing by det_r)
    tr_M = M11 + M22
    det_M = M11 * M22 - M12 * M21

    # Discriminant: disc = tr_M² - 4 det_M
    disc = tr_M**2 - 4.0 * det_M
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)

    # Eigenvalues of S = M / (2 det_r):  λ± = (tr_M ± √disc) / (4 det_r)
    inv_4det = 1.0 / (4.0 * det_r)
    lam_plus = (tr_M + sqrt_disc) * inv_4det
    lam_minus = (tr_M - sqrt_disc) * inv_4det

    # Select dominant eigenvalue
    pick_plus = np.abs(lam_plus) >= np.abs(lam_minus)
    sign = np.where(pick_plus, 1.0, -1.0)
    lam = np.where(pick_plus, lam_plus, lam_minus)

    # --- Derivatives of M entries w.r.t. (ε, φ, γ) ---
    # ∂M11/∂ε = g_r,  ∂M11/∂φ = -f_r,  ∂M11/∂γ = 0
    # ∂M12/∂ε = 0,    ∂M12/∂φ = g_r,   ∂M12/∂γ = -f_r
    # ∂M21/∂ε = -f_r, ∂M21/∂φ = e_r,   ∂M21/∂γ = 0
    # ∂M22/∂ε = 0,    ∂M22/∂φ = -f_r,  ∂M22/∂γ = e_r

    # ∂tr_M/∂· = ∂M11/∂· + ∂M22/∂·
    dtr_deps = g_r
    dtr_dphi = -2.0 * f_r
    dtr_dgam = e_r

    # ∂det_M/∂· via product rule on M11*M22 - M12*M21
    ddet_deps = g_r * M22 + f_r * M12
    ddet_dphi = -f_r * (M11 + M22) - g_r * M21 - e_r * M12
    ddet_dgam = e_r * M11 + f_r * M21

    # ∂disc/∂· = 2 tr_M · ∂tr_M/∂· - 4 · ∂det_M/∂·
    ddisc_deps = 2.0 * tr_M * dtr_deps - 4.0 * ddet_deps
    ddisc_dphi = 2.0 * tr_M * dtr_dphi - 4.0 * ddet_dphi
    ddisc_dgam = 2.0 * tr_M * dtr_dgam - 4.0 * ddet_dgam

    # ∂√disc/∂· = ∂disc/∂· / (2√disc), guarded
    safe_sqrt = np.where(sqrt_disc > 1e-30, sqrt_disc, 1e-30)
    dsqrt_deps = ddisc_deps / (2.0 * safe_sqrt)
    dsqrt_dphi = ddisc_dphi / (2.0 * safe_sqrt)
    dsqrt_dgam = ddisc_dgam / (2.0 * safe_sqrt)

    # ∂λ_selected/∂· = (∂tr_M/∂· + sign · ∂√disc/∂·) / (4 det_r)
    dlam_deps = (dtr_deps + sign * dsqrt_deps) * inv_4det
    dlam_dphi = (dtr_dphi + sign * dsqrt_dphi) * inv_4det
    dlam_dgam = (dtr_dgam + sign * dsqrt_dgam) * inv_4det

    return lam, dlam_deps, dlam_dphi, dlam_dgam


def assemble_algebraic_jacobian(
    dlam_deps: np.ndarray,
    dlam_dphi: np.ndarray,
    dlam_dgam: np.ndarray,
    K: int,
) -> sp.csr_matrix:
    """
    Assemble a K² × 3K² Jacobian for one algebraic constraint.

    Since the constraint is pointwise (no spatial coupling), the Jacobian
    is block-diagonal: [diag(-∂Λ/∂ε) | diag(-∂Λ/∂φ) | diag(-∂Λ/∂γ)]

    The negative sign is because the constraint is target - Λ = 0,
    so ∂(target - Λ)/∂x = -∂Λ/∂x.
    """
    return sp.hstack([
        _diag(-dlam_deps),
        _diag(-dlam_dphi),
        _diag(-dlam_dgam),
    ], format='csr')


# =============================================================================
# Gauss residual evaluation (using sparse operators)
# =============================================================================

def evaluate_gauss_residual(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    ops: Dict[str, sp.csr_matrix],
    K: int,
) -> Tuple[np.ndarray, Dict, Dict, Dict]:
    """
    Evaluate the Gauss residual and return all derivative fields.

    Returns
    -------
    residual : ndarray (K, K)
    d_eps, d_phi, d_gam : dict of derivative arrays
    """
    d_eps = compute_all_derivatives(ops, eps, K)
    d_phi = compute_all_derivatives(ops, phi, K)
    d_gam = compute_all_derivatives(ops, gam, K)

    res = gauss_residual(
        eps, phi, gam,
        d_eps['u'], d_eps['v'],
        d_gam['u'], d_gam['v'],
        d_phi['u'], d_phi['v'],
        d_eps['vv'], d_phi['uv'], d_gam['uu'],
    )

    return res, d_eps, d_phi, d_gam


# =============================================================================
# Full system assembly
# =============================================================================

def assemble_system(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    e_o: np.ndarray, f_o: np.ndarray, g_o: np.ndarray,
    E_f: np.ndarray, F_f: np.ndarray, G_f: np.ndarray,
    t: float,
    lam_of: np.ndarray,
    lam_fo: np.ndarray,
    ops: Dict[str, sp.csr_matrix],
    K: int,
    interp_fn: Callable = linear_interpolation,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Assemble the full 3K² × 3K² Newton system for one iteration.

    The system is:
        J @ δX = -R

    where δX = [δε, δφ, δγ] (flattened) and R = [R_gauss, R_alg1, R_alg2].

    Parameters
    ----------
    eps, phi, gam : ndarray (K, K)
        Current metric components.
    e_o, f_o, g_o : ndarray (K, K)
        Initial surface metric.
    E_f, F_f, G_f : ndarray (K, K)
        Final surface metric.
    t : float
        Current time in [0, 1].
    lam_of, lam_fo : ndarray (K, K)
        Precomputed strain eigenvalue fields.
    ops : dict of sparse operators.
    K : int
    interp_fn : callable
        Interpolation function for the strain targets.

    Returns
    -------
    J : sparse (3K², 3K²)
        Jacobian matrix.
    neg_R : ndarray (3K²,)
        Negative residual vector.
    """
    N = K * K

    # --- Gauss equation ---
    R_gauss, d_eps, d_phi, d_gam = evaluate_gauss_residual(eps, phi, gam, ops, K)
    coeffs = gauss_jacobian_coefficients(eps, phi, gam, d_eps, d_phi, d_gam)
    J_gauss = assemble_gauss_jacobian(coeffs, ops, K)

    # --- Forward algebraic constraint (eq. 11) ---
    lam_fwd, dl_deps_f, dl_dphi_f, dl_dgam_f = strain_eigenvalue_and_jacobian(
        e_o, f_o, g_o, eps, phi, gam
    )
    target_fwd = interp_fn(lam_of, t)
    R_alg1 = target_fwd - lam_fwd
    J_alg1 = assemble_algebraic_jacobian(dl_deps_f, dl_dphi_f, dl_dgam_f, K)

    # --- Reverse algebraic constraint (eq. 12) ---
    lam_rev, dl_deps_r, dl_dphi_r, dl_dgam_r = strain_eigenvalue_and_jacobian(
        E_f, F_f, G_f, eps, phi, gam
    )
    target_rev = interp_fn(lam_fo, 1.0 - t)
    R_alg2 = target_rev - lam_rev
    J_alg2 = assemble_algebraic_jacobian(dl_deps_r, dl_dphi_r, dl_dgam_r, K)

    # --- Stack into full system ---
    J = sp.vstack([J_gauss, J_alg1, J_alg2], format='csr')
    neg_R = np.concatenate([
        -flatten_field(R_gauss),
        -flatten_field(R_alg1),
        -flatten_field(R_alg2),
    ])

    return J, neg_R


def assemble_system_with_bcs(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    e_o: np.ndarray, f_o: np.ndarray, g_o: np.ndarray,
    E_f: np.ndarray, F_f: np.ndarray, G_f: np.ndarray,
    t: float,
    lam_of: np.ndarray,
    lam_fo: np.ndarray,
    ops: Dict[str, sp.csr_matrix],
    K: int,
    bc_eps: np.ndarray,
    bc_phi: np.ndarray,
    bc_gam: np.ndarray,
    interp_fn: Callable = linear_interpolation,
    regularization: float = 1e-8,
) -> Tuple[sp.csr_matrix, np.ndarray]:
    """
    Assemble the full system with Dirichlet boundary conditions enforced.

    Boundary DOFs are pinned by replacing their rows in the Jacobian with
    identity rows and setting the corresponding RHS to zero (i.e., δX = 0
    at boundary points, keeping them at their prescribed values).

    A small Tikhonov regularization term is added to the diagonal to handle
    near-singular Jacobians. This arises structurally because the forward
    algebraic constraint's sensitivity to φ vanishes when the initial metric's
    off-diagonal f=0 (as for the square), and similarly along symmetry axes
    of the grid. The regularization biases φ perturbations toward zero at
    these degenerate points, which is physically appropriate.

    Parameters
    ----------
    bc_eps, bc_phi, bc_gam : ndarray (K, K)
        Boundary condition fields (only boundary values are used).
    regularization : float
        Tikhonov regularization parameter added to diagonal.
    (other parameters same as assemble_system)

    Returns
    -------
    J : sparse (3K², 3K²)
    neg_R : ndarray (3K²,)
    """
    N = K * K

    # First assemble the unconstrained system
    J, neg_R = assemble_system(
        eps, phi, gam,
        e_o, f_o, g_o, E_f, F_f, G_f,
        t, lam_of, lam_fo, ops, K, interp_fn,
    )

    # Add Tikhonov regularization to diagonal
    if regularization > 0:
        J = J + regularization * sp.eye(3 * N, format='csr')

    # Build boundary index sets for the three blocks
    from discretization import boundary_indices
    bnd_idx = boundary_indices(K)

    # Convert to LIL for efficient row modification
    J = J.tolil()

    for block in range(3):
        offset = block * N
        for idx in bnd_idx:
            row = offset + idx
            J[row, :] = 0
            J[row, offset + idx] = 1.0
            neg_R[row] = 0.0

    return J.tocsr(), neg_R


# =============================================================================
# Solve one Newton step
# =============================================================================

def solve_newton_step(
    J: sp.csr_matrix,
    neg_R: np.ndarray,
    method: str = 'spsolve',
) -> np.ndarray:
    """
    Solve the linear system J @ δX = neg_R.

    Parameters
    ----------
    J : sparse (3K², 3K²)
    neg_R : ndarray (3K²,)
    method : str
        'spsolve' for direct solve (good for small-medium K),
        'gmres' or 'bicgstab' for iterative (better for large K).

    Returns
    -------
    delta_X : ndarray (3K²,)
    """
    if method == 'spsolve':
        from scipy.sparse.linalg import spsolve
        return spsolve(J, neg_R)
    elif method == 'gmres':
        from scipy.sparse.linalg import gmres, spilu, LinearOperator
        ilu = spilu(J.tocsc())
        M = LinearOperator(J.shape, matvec=ilu.solve)
        delta_X, info = gmres(J, neg_R, M=M, atol=1e-10, maxiter=500)
        if info != 0:
            print(f"Warning: GMRES did not converge (info={info})")
        return delta_X
    elif method == 'bicgstab':
        from scipy.sparse.linalg import bicgstab, spilu, LinearOperator
        ilu = spilu(J.tocsc())
        M = LinearOperator(J.shape, matvec=ilu.solve)
        delta_X, info = bicgstab(J, neg_R, M=M, atol=1e-10, maxiter=500)
        if info != 0:
            print(f"Warning: BiCGSTAB did not converge (info={info})")
        return delta_X
    else:
        raise ValueError(f"Unknown method: {method}")


def unpack_delta(delta_X: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack a (3K²,) solution vector into three (K, K) perturbation fields.

    Returns
    -------
    d_eps, d_phi, d_gam : ndarray (K, K)
    """
    N = K * K
    return (
        unflatten_field(delta_X[0:N], K),
        unflatten_field(delta_X[N:2*N], K),
        unflatten_field(delta_X[2*N:3*N], K),
    )


# =============================================================================
# Residual norm computation (for convergence monitoring)
# =============================================================================

def compute_residual_norms(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    e_o: np.ndarray, f_o: np.ndarray, g_o: np.ndarray,
    E_f: np.ndarray, F_f: np.ndarray, G_f: np.ndarray,
    t: float,
    lam_of: np.ndarray,
    lam_fo: np.ndarray,
    ops: Dict[str, sp.csr_matrix],
    K: int,
    interp_fn: Callable = linear_interpolation,
) -> Dict[str, float]:
    """
    Evaluate all three residuals and return their norms.

    Returns
    -------
    norms : dict with keys 'gauss', 'alg_fwd', 'alg_rev', 'total'
        Each is the L-infinity norm of the corresponding residual
        (interior points only for Gauss).
    """
    R_gauss, _, _, _ = evaluate_gauss_residual(eps, phi, gam, ops, K)

    lam_fwd = green_strain_eigenvalue_direct(e_o, f_o, g_o, eps, phi, gam)
    target_fwd = interp_fn(lam_of, t)
    R_alg1 = target_fwd - lam_fwd

    lam_rev = green_strain_eigenvalue_direct(E_f, F_f, G_f, eps, phi, gam)
    target_rev = interp_fn(lam_fo, 1.0 - t)
    R_alg2 = target_rev - lam_rev

    # Interior Gauss norm (boundary values are not meaningful)
    gauss_int = R_gauss[2:-2, 2:-2]
    norm_gauss = np.max(np.abs(gauss_int)) if gauss_int.size > 0 else 0.0
    norm_alg1 = np.max(np.abs(R_alg1))
    norm_alg2 = np.max(np.abs(R_alg2))

    return {
        'gauss': norm_gauss,
        'alg_fwd': norm_alg1,
        'alg_rev': norm_alg2,
        'total': max(norm_gauss, norm_alg1, norm_alg2),
    }
