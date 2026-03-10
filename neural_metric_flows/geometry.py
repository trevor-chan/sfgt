"""
geometry.py — Differential geometry computations via torch autograd.

All quantities computed from the fundamental forms and their derivatives
with respect to parameter coordinates (u, v). No finite differences.
"""

import torch


# =============================================================================
# Autograd helpers
# =============================================================================

def grad_scalar(y, x, create_graph=True):
    """
    Compute gradient of scalar field y w.r.t. input x.

    Parameters
    ----------
    y : Tensor (N,) or (N, 1)
        Scalar field values at N points.
    x : Tensor (N, d)
        Input coordinates (must have requires_grad=True).

    Returns
    -------
    grad : Tensor (N, d)
    """
    if y.dim() == 1:
        y = y.unsqueeze(-1)
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


def partial(y, x, idx, create_graph=True):
    """Partial derivative dy/dx_idx."""
    g = grad_scalar(y, x, create_graph=create_graph)
    return g[:, idx]


def partial_u(y, uv, create_graph=True):
    """dy/du where uv[:, 0] = u."""
    return partial(y, uv, 0, create_graph)


def partial_v(y, uv, create_graph=True):
    """dy/dv where uv[:, 1] = v."""
    return partial(y, uv, 1, create_graph)


def partial_t(y, tuv, create_graph=True):
    """dy/dt where tuv[:, 0] = t."""
    return partial(y, tuv, 0, create_graph)


# =============================================================================
# Christoffel symbols from first fundamental form
# =============================================================================

def christoffel_symbols(E, F, G, uv):
    """
    Christoffel symbols of the second kind from (E, F, G).

    Parameters
    ----------
    E, F, G : Tensor (N,)
    uv : Tensor (N, 2) with requires_grad=True

    Returns
    -------
    dict with keys 'G111', 'G112', 'G122', 'G211', 'G212', 'G222'
    """
    E_u = partial_u(E, uv)
    E_v = partial_v(E, uv)
    F_u = partial_u(F, uv)
    F_v = partial_v(F, uv)
    G_u = partial_u(G, uv)
    G_v = partial_v(G, uv)

    det = E * G - F * F
    inv_det = 1.0 / (det + 1e-12)

    G111 = 0.5 * inv_det * (G * E_u - 2 * F * F_u + F * E_v)
    G112 = 0.5 * inv_det * (G * E_v - F * G_u)
    G122 = 0.5 * inv_det * (2 * G * F_v - G * G_u - F * G_v)

    G211 = 0.5 * inv_det * (2 * E * F_u - E * E_v - F * E_u)
    G212 = 0.5 * inv_det * (E * G_u - F * E_v)
    G222 = 0.5 * inv_det * (E * G_v - 2 * F * F_v + F * G_u)

    return {
        'G111': G111, 'G112': G112, 'G122': G122,
        'G211': G211, 'G212': G212, 'G222': G222,
    }


# =============================================================================
# Gaussian curvature via Brioschi formula
# =============================================================================

def gaussian_curvature_brioschi(E, F, G, uv):
    """
    Gaussian curvature K from first fundamental form only (Theorema Egregium).

    Uses the Brioschi formula with two 3x3 determinants.

    Parameters
    ----------
    E, F, G : Tensor (N,)
    uv : Tensor (N, 2)

    Returns
    -------
    K : Tensor (N,)
    """
    E_u = partial_u(E, uv)
    E_v = partial_v(E, uv)
    F_u = partial_u(F, uv)
    F_v = partial_v(F, uv)
    G_u = partial_u(G, uv)
    G_v = partial_v(G, uv)

    E_vv = partial_v(E_v, uv)
    G_uu = partial_u(G_u, uv)
    F_uv = partial_v(F_u, uv)

    det_a = E * G - F * F

    # First determinant
    a11 = -0.5 * E_vv + F_uv - 0.5 * G_uu
    a12 = 0.5 * E_u
    a13 = F_u - 0.5 * E_v
    a21 = F_v - 0.5 * G_u
    a22 = E
    a23 = F
    a31 = 0.5 * G_v
    a32 = F
    a33 = G

    det1 = (a11 * (a22 * a33 - a23 * a32)
            - a12 * (a21 * a33 - a23 * a31)
            + a13 * (a21 * a32 - a22 * a31))

    # Second determinant
    b12 = 0.5 * E_v
    b13 = 0.5 * G_u

    det2 = (-b12 * (b12 * G - b13 * F)
            + b13 * (b12 * F - b13 * E))

    K = (det1 - det2) / (det_a * det_a + 1e-12)
    return K


# =============================================================================
# Gauss equation residual (3D: extrinsic = intrinsic)
# =============================================================================

def gauss_residual_3d(E, F, G, L, M, N, uv):
    """
    Gauss equation: K_extrinsic - K_intrinsic = 0.

    Returns
    -------
    residual : Tensor (N,)
    """
    det_a = E * G - F * F
    K_ext = (L * N - M * M) / (det_a + 1e-12)
    K_int = gaussian_curvature_brioschi(E, F, G, uv)
    return K_ext - K_int


# =============================================================================
# Codazzi-Mainardi residuals (3D)
# =============================================================================

def codazzi_residuals(E, F, G, L, M, N, uv):
    """
    Codazzi-Mainardi equations.

    Returns
    -------
    R1, R2 : Tensor (N,)
    """
    Gamma = christoffel_symbols(E, F, G, uv)

    L_v = partial_v(L, uv)
    M_u = partial_u(M, uv)
    M_v = partial_v(M, uv)
    N_u = partial_u(N, uv)

    lhs1 = L_v - M_u
    rhs1 = (L * Gamma['G112']
            + M * (Gamma['G212'] - Gamma['G111'])
            - N * Gamma['G211'])
    R1 = lhs1 - rhs1

    lhs2 = M_v - N_u
    rhs2 = (L * Gamma['G122']
            + M * (Gamma['G222'] - Gamma['G112'])
            - N * Gamma['G212'])
    R2 = lhs2 - rhs2

    return R1, R2


# =============================================================================
# Curvatures from both fundamental forms
# =============================================================================

def gaussian_curvature(E, F, G, L, M, N):
    """K = (LN - M²) / (EG - F²)."""
    return (L * N - M * M) / (E * G - F * F + 1e-12)


def mean_curvature(E, F, G, L, M, N):
    """H = (EN - 2FM + GL) / (2(EG - F²))."""
    return (E * N - 2 * F * M + G * L) / (2 * (E * G - F * F) + 1e-12)


# =============================================================================
# Green strain tensor
# =============================================================================

def green_strain_invariants(E, F, G, E0, F0, G0):
    """
    Trace and determinant of Green strain S = (1/2) a₀⁻¹(a - a₀).

    Returns
    -------
    tr_S, det_S : Tensor (N,)
    """
    det0 = E0 * G0 - F0 * F0
    dE, dF, dG = E - E0, F - F0, G - G0

    M11 = G0 * dE - F0 * dF
    M12 = G0 * dF - F0 * dG
    M21 = -F0 * dE + E0 * dF
    M22 = -F0 * dF + E0 * dG

    tr_S = (M11 + M22) / (2 * det0 + 1e-12)
    det_S = (M11 * M22 - M12 * M21) / (4 * det0 * det0 + 1e-12)

    return tr_S, det_S
