"""
reconstruction.py — Surface embedding recovery from fundamental forms.

Given the first and second fundamental forms (E, F, G, L, M, N), reconstructs
the surface embedding r(u,v) = (x, y, z) using Gauss-Weingarten frame integration.

This handles both:
- 3D surfaces (arbitrary L, M, N)
- 2D/flat surfaces (L = M = N = 0, surface stays in-plane)
"""

import numpy as np
from typing import Tuple


# =============================================================================
# Finite difference helpers
# =============================================================================

def _fd_gradient(f: np.ndarray, h: float, axis: int) -> np.ndarray:
    """
    Compute gradient using finite differences.

    4th-order central differences in interior, 2nd-order at boundaries.

    Parameters
    ----------
    f : ndarray (K, K)
        Scalar field.
    h : float
        Grid spacing.
    axis : int
        Axis along which to differentiate (0 or 1).

    Returns
    -------
    grad : ndarray (K, K)
        Gradient field.
    """
    if axis == 1:
        return _fd_gradient(f.T, h, 0).T

    g = np.zeros_like(f)
    K = f.shape[0]

    # Interior: 4th-order central
    if K > 4:
        g[2:-2] = (-f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[:-4]) / (12*h)

    # Near-boundary: 2nd-order central
    if K > 2:
        g[1] = (f[2] - f[0]) / (2*h)
        g[-2] = (f[-1] - f[-3]) / (2*h)

    # Boundary: 2nd-order one-sided
    if K > 2:
        g[0] = (-3*f[0] + 4*f[1] - f[2]) / (2*h)
        g[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2*h)
    elif K == 2:
        g[0] = (f[1] - f[0]) / h
        g[1] = (f[1] - f[0]) / h

    return g


# =============================================================================
# Christoffel symbols
# =============================================================================

def compute_christoffel_symbols(
    E: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    du: float,
    dv: float,
) -> dict:
    """
    Compute Christoffel symbols of the second kind from first fundamental form.

    Parameters
    ----------
    E, F, G : ndarray (K, K)
        First fundamental form components.
    du, dv : float
        Grid spacing in u and v directions.

    Returns
    -------
    Gamma : dict
        Dictionary with keys 'G111', 'G112', 'G122', 'G211', 'G212', 'G222'.
    """
    E_u = _fd_gradient(E, du, axis=0)
    E_v = _fd_gradient(E, dv, axis=1)
    F_u = _fd_gradient(F, du, axis=0)
    F_v = _fd_gradient(F, dv, axis=1)
    G_u = _fd_gradient(G, du, axis=0)
    G_v = _fd_gradient(G, dv, axis=1)

    det = E * G - F**2
    det = np.maximum(det, 1e-30)
    inv2det = 0.5 / det

    return {
        'G111': inv2det * (G * E_u - 2 * F * F_u + F * E_v),
        'G211': inv2det * (2 * E * F_u - E * E_v - F * E_u),
        'G112': inv2det * (G * E_v - F * G_u),
        'G212': inv2det * (E * G_u - F * E_v),
        'G122': inv2det * (2 * G * F_v - G * G_u - F * G_v),
        'G222': inv2det * (E * G_v - 2 * F * F_v + F * G_u),
    }


# =============================================================================
# Surface reconstruction via Gauss-Weingarten frame integration
# =============================================================================

def reconstruct_surface(
    E: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    L: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    du: float,
    dv: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct surface embedding r(u,v) = (x, y, z) from fundamental forms.

    Uses Gauss-Weingarten frame integration. The frame {r_u, r_v, n} evolves
    according to the Gauss equations with Christoffel symbols and curvature
    terms.

    For flat surfaces (L = M = N = 0), the surface stays in the xy-plane.

    Parameters
    ----------
    E, F, G : ndarray (K, K)
        First fundamental form components.
    L, M, N : ndarray (K, K)
        Second fundamental form components.
    du, dv : float
        Grid spacing.

    Returns
    -------
    X, Y, Z : ndarray (K, K)
        Surface coordinates in 3D (centered at origin).
    """
    K = E.shape[0]

    # Christoffel symbols
    Gamma = compute_christoffel_symbols(E, F, G, du, dv)

    # Initialize frame at (0, 0)
    # Position r, tangent vectors r_u, r_v (each 3D vectors at each grid point)
    r = np.zeros((K, K, 3))      # position
    r_u = np.zeros((K, K, 3))    # ∂r/∂u
    r_v = np.zeros((K, K, 3))    # ∂r/∂v

    # Initial frame via Cholesky-like decomposition
    # r_u · r_u = E, r_u · r_v = F, r_v · r_v = G
    sqrt_E = np.sqrt(np.maximum(E[0, 0], 1e-10))
    det = E[0, 0] * G[0, 0] - F[0, 0]**2
    sqrt_det_over_E = np.sqrt(np.maximum(det, 1e-10)) / sqrt_E

    # Initial frame in xy-plane (z=0)
    # As we integrate, curvature terms (L, M, N) lift it into 3D
    r[0, 0] = np.array([0.0, 0.0, 0.0])
    r_u[0, 0] = np.array([sqrt_E, 0.0, 0.0])
    r_v[0, 0] = np.array([F[0, 0] / sqrt_E, sqrt_det_over_E, 0.0])

    def compute_normal(ru: np.ndarray, rv: np.ndarray) -> np.ndarray:
        """Compute unit normal n = (r_u × r_v) / |r_u × r_v|."""
        cross = np.cross(ru, rv)
        norm = np.linalg.norm(cross)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return cross / norm

    def gauss_rhs_u(ru, rv, Gamma_pt, L_pt, M_pt):
        """RHS of Gauss equation for ∂r_u/∂u and ∂r_v/∂u."""
        n = compute_normal(ru, rv)
        dr_u_du = Gamma_pt['G111'] * ru + Gamma_pt['G211'] * rv + L_pt * n
        dr_v_du = Gamma_pt['G112'] * ru + Gamma_pt['G212'] * rv + M_pt * n
        return dr_u_du, dr_v_du

    def gauss_rhs_v(ru, rv, Gamma_pt, M_pt, N_pt):
        """RHS of Gauss equation for ∂r_u/∂v and ∂r_v/∂v."""
        n = compute_normal(ru, rv)
        dr_u_dv = Gamma_pt['G112'] * ru + Gamma_pt['G212'] * rv + M_pt * n
        dr_v_dv = Gamma_pt['G122'] * ru + Gamma_pt['G222'] * rv + N_pt * n
        return dr_u_dv, dr_v_dv

    def get_gamma_at(i, j):
        """Extract Christoffel symbols at point (i, j)."""
        return {k: v[i, j] for k, v in Gamma.items()}

    # Propagate along left edge (i=0, varying j in v-direction)
    for j in range(1, K):
        # Previous state
        r_prev = r[0, j-1]
        ru_prev = r_u[0, j-1]
        rv_prev = r_v[0, j-1]
        Gamma_prev = get_gamma_at(0, j-1)
        M_prev, N_prev = M[0, j-1], N[0, j-1]

        # RHS at previous point
        dru_dv_prev, drv_dv_prev = gauss_rhs_v(ru_prev, rv_prev, Gamma_prev, M_prev, N_prev)

        # Euler predictor for frame
        ru_pred = ru_prev + dv * dru_dv_prev
        rv_pred = rv_prev + dv * drv_dv_prev

        # RHS at predicted point
        Gamma_curr = get_gamma_at(0, j)
        M_curr, N_curr = M[0, j], N[0, j]
        dru_dv_curr, drv_dv_curr = gauss_rhs_v(ru_pred, rv_pred, Gamma_curr, M_curr, N_curr)

        # Trapezoidal corrector for frame
        r_u[0, j] = ru_prev + 0.5 * dv * (dru_dv_prev + dru_dv_curr)
        r_v[0, j] = rv_prev + 0.5 * dv * (drv_dv_prev + drv_dv_curr)

        # Position: integrate r_v
        r[0, j] = r_prev + 0.5 * dv * (rv_prev + r_v[0, j])

    # Propagate across each row (varying i in u-direction)
    for j in range(K):
        for i in range(1, K):
            # Previous state
            r_prev = r[i-1, j]
            ru_prev = r_u[i-1, j]
            rv_prev = r_v[i-1, j]
            Gamma_prev = get_gamma_at(i-1, j)
            L_prev, M_prev = L[i-1, j], M[i-1, j]

            # RHS at previous point
            dru_du_prev, drv_du_prev = gauss_rhs_u(ru_prev, rv_prev, Gamma_prev, L_prev, M_prev)

            # Euler predictor for frame
            ru_pred = ru_prev + du * dru_du_prev
            rv_pred = rv_prev + du * drv_du_prev

            # RHS at predicted point
            Gamma_curr = get_gamma_at(i, j)
            L_curr, M_curr = L[i, j], M[i, j]
            dru_du_curr, drv_du_curr = gauss_rhs_u(ru_pred, rv_pred, Gamma_curr, L_curr, M_curr)

            # Trapezoidal corrector for frame
            r_u[i, j] = ru_prev + 0.5 * du * (dru_du_prev + dru_du_curr)
            r_v[i, j] = rv_prev + 0.5 * du * (drv_du_prev + drv_du_curr)

            # Position: integrate r_u
            r[i, j] = r_prev + 0.5 * du * (ru_prev + r_u[i, j])

    X = r[:, :, 0]
    Y = r[:, :, 1]
    Z = r[:, :, 2]

    # Center the surface
    X = X - X.mean()
    Y = Y - Y.mean()
    Z = Z - Z.mean()

    return X, Y, Z


# =============================================================================
# Convenience functions
# =============================================================================

def reconstruct_flat_surface(
    E: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    du: float,
    dv: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a flat surface (2D embedding) from first fundamental form only.

    Convenience wrapper that sets L = M = N = 0.

    Parameters
    ----------
    E, F, G : ndarray (K, K)
        First fundamental form components.
    du, dv : float
        Grid spacing.

    Returns
    -------
    X, Y : ndarray (K, K)
        Surface coordinates in 2D (Z ≈ 0).
    """
    K = E.shape[0]
    zeros = np.zeros((K, K))
    X, Y, Z = reconstruct_surface(E, F, G, zeros, zeros, zeros, du, dv)
    return X, Y


def reconstruct_from_model(
    model,
    t_val: float,
    K_grid: int = 25,
    margin: float = 0.15,
    device: str = 'cpu',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Reconstruct surface from a trained model at a specific time.

    Parameters
    ----------
    model : FundamentalFormNet
        Trained neural metric model.
    t_val : float
        Time value in [0, 1].
    K_grid : int
        Grid resolution.
    margin : float
        Domain margin.
    device : str
        Torch device.

    Returns
    -------
    X, Y, Z : ndarray (K_grid, K_grid)
        Reconstructed surface coordinates.
    metrics : dict
        Dictionary with E, F, G, L, M, N, K, H arrays.
    """
    import torch

    lo = -1.0 + margin
    hi = 1.0 - margin
    L_domain = hi - lo
    du = L_domain / (K_grid - 1)

    # Create grid
    uv_1d = torch.linspace(lo, hi, K_grid, device=device)
    U, V = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    uv_flat = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)

    N_pts = uv_flat.shape[0]
    t_col = torch.full((N_pts, 1), t_val, device=device)
    tuv = torch.cat([t_col, uv_flat], dim=1)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        E, F, G, L, M, N = model(tuv)

    # Reshape to grid
    E_np = E.reshape(K_grid, K_grid).cpu().numpy()
    F_np = F.reshape(K_grid, K_grid).cpu().numpy()
    G_np = G.reshape(K_grid, K_grid).cpu().numpy()
    L_np = L.reshape(K_grid, K_grid).cpu().numpy()
    M_np = M.reshape(K_grid, K_grid).cpu().numpy()
    N_np = N.reshape(K_grid, K_grid).cpu().numpy()

    # Compute curvatures
    det_a = E_np * G_np - F_np**2
    K_gauss = (L_np * N_np - M_np**2) / (det_a + 1e-12)
    H_mean = (E_np * N_np + G_np * L_np - 2 * F_np * M_np) / (2 * det_a + 1e-12)

    # Reconstruct surface
    X, Y, Z = reconstruct_surface(E_np, F_np, G_np, L_np, M_np, N_np, du, du)

    metrics = {
        'E': E_np, 'F': F_np, 'G': G_np,
        'L': L_np, 'M': M_np, 'N': N_np,
        'K': K_gauss, 'H': H_mean,
        'det': det_a,
    }

    return X, Y, Z, metrics
