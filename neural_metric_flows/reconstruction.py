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
    K_u, K_v = E.shape

    # Christoffel symbols
    Gamma = compute_christoffel_symbols(E, F, G, du, dv)

    # Initialize frame at (0, 0)
    # Position r, tangent vectors r_u, r_v (each 3D vectors at each grid point)
    r = np.zeros((K_u, K_v, 3))      # position
    r_u = np.zeros((K_u, K_v, 3))    # ∂r/∂u
    r_v = np.zeros((K_u, K_v, 3))    # ∂r/∂v

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
    for j in range(1, K_v):
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
    for j in range(K_v):
        for i in range(1, K_u):
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
    K_u, K_v = E.shape
    zeros = np.zeros((K_u, K_v))
    X, Y, Z = reconstruct_surface(E, F, G, zeros, zeros, zeros, du, dv)
    return X, Y


def enforce_periodic_closure(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    periodic_u: bool = False,
    periodic_v: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Distribute seam drift across a reconstructed surface to close periodic loops.

    This is a visualization-time correction for cylindrical/toroidal charts.
    The underlying metric may already be periodic, but open-boundary frame
    integration accumulates drift around a closed parameter cycle.
    """
    coords = np.stack([X, Y, Z], axis=-1).copy()
    K_u, K_v, _ = coords.shape

    if periodic_v and K_v > 1:
        drift_v = coords[:, -1, :] - coords[:, 0, :]
        weights_v = np.linspace(0.0, 1.0, K_v)[None, :, None]
        coords -= weights_v * drift_v[:, None, :]

    if periodic_u and K_u > 1:
        drift_u = coords[-1, :, :] - coords[0, :, :]
        weights_u = np.linspace(0.0, 1.0, K_u)[:, None, None]
        coords -= weights_u * drift_u[None, :, :]

    return coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]


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

    # Evaluate model. Do not use `no_grad()` here because endpoint-conditioned
    # models may compute analytic derivatives internally.
    model.eval()
    E, F, G, L, M, N = model(tuv)

    # Reshape to grid
    E_np = E.reshape(K_grid, K_grid).detach().cpu().numpy()
    F_np = F.reshape(K_grid, K_grid).detach().cpu().numpy()
    G_np = G.reshape(K_grid, K_grid).detach().cpu().numpy()
    L_np = L.reshape(K_grid, K_grid).detach().cpu().numpy()
    M_np = M.reshape(K_grid, K_grid).detach().cpu().numpy()
    N_np = N.reshape(K_grid, K_grid).detach().cpu().numpy()

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


# =============================================================================
# Spherical coordinate reconstruction
# =============================================================================

def reconstruct_surface_spherical(
    E: np.ndarray,
    F: np.ndarray,
    G: np.ndarray,
    L: np.ndarray,
    M: np.ndarray,
    N: np.ndarray,
    d_theta: float,
    d_phi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct surface embedding from fundamental forms in spherical coordinates.

    Uses Gauss-Weingarten frame integration in (θ, φ) coordinates.

    Parameters
    ----------
    E, F, G : ndarray (K_theta, K_phi)
        First fundamental form components.
    L, M, N : ndarray (K_theta, K_phi)
        Second fundamental form components.
    d_theta, d_phi : float
        Grid spacing in θ and φ.

    Returns
    -------
    X, Y, Z : ndarray (K_theta, K_phi)
        Surface coordinates in 3D.
    """
    K_theta, K_phi = E.shape

    # Christoffel symbols
    Gamma = compute_christoffel_symbols(E, F, G, d_theta, d_phi)

    # Initialize frame
    r = np.zeros((K_theta, K_phi, 3))
    r_theta = np.zeros((K_theta, K_phi, 3))
    r_phi = np.zeros((K_theta, K_phi, 3))

    # Initial frame at (0, 0) - corresponds to near north pole
    # For sphere-like surfaces, start with appropriate orientation
    sqrt_E = np.sqrt(np.maximum(E[0, 0], 1e-10))
    det = E[0, 0] * G[0, 0] - F[0, 0]**2
    sqrt_det_over_E = np.sqrt(np.maximum(det, 1e-10)) / sqrt_E

    # Initial tangent frame
    r[0, 0] = np.array([0.0, 0.0, 1.0])  # Start near north pole
    r_theta[0, 0] = np.array([sqrt_E, 0.0, 0.0])
    r_phi[0, 0] = np.array([F[0, 0] / sqrt_E, sqrt_det_over_E, 0.0])

    def compute_normal(r_t: np.ndarray, r_p: np.ndarray) -> np.ndarray:
        cross = np.cross(r_t, r_p)
        norm = np.linalg.norm(cross)
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0])
        return cross / norm

    def gauss_rhs_theta(r_t, r_p, Gamma_pt, L_pt, M_pt):
        n = compute_normal(r_t, r_p)
        dr_t_dt = Gamma_pt['G111'] * r_t + Gamma_pt['G211'] * r_p + L_pt * n
        dr_p_dt = Gamma_pt['G112'] * r_t + Gamma_pt['G212'] * r_p + M_pt * n
        return dr_t_dt, dr_p_dt

    def gauss_rhs_phi(r_t, r_p, Gamma_pt, M_pt, N_pt):
        n = compute_normal(r_t, r_p)
        dr_t_dp = Gamma_pt['G112'] * r_t + Gamma_pt['G212'] * r_p + M_pt * n
        dr_p_dp = Gamma_pt['G122'] * r_t + Gamma_pt['G222'] * r_p + N_pt * n
        return dr_t_dp, dr_p_dp

    def get_gamma_at(i, j):
        return {k: v[i, j] for k, v in Gamma.items()}

    # Propagate along first row (φ direction)
    for j in range(1, K_phi):
        r_prev = r[0, j-1]
        rt_prev = r_theta[0, j-1]
        rp_prev = r_phi[0, j-1]
        Gamma_prev = get_gamma_at(0, j-1)
        M_prev, N_prev = M[0, j-1], N[0, j-1]

        drt_dp_prev, drp_dp_prev = gauss_rhs_phi(rt_prev, rp_prev, Gamma_prev, M_prev, N_prev)

        rt_pred = rt_prev + d_phi * drt_dp_prev
        rp_pred = rp_prev + d_phi * drp_dp_prev

        Gamma_curr = get_gamma_at(0, j)
        M_curr, N_curr = M[0, j], N[0, j]
        drt_dp_curr, drp_dp_curr = gauss_rhs_phi(rt_pred, rp_pred, Gamma_curr, M_curr, N_curr)

        r_theta[0, j] = rt_prev + 0.5 * d_phi * (drt_dp_prev + drt_dp_curr)
        r_phi[0, j] = rp_prev + 0.5 * d_phi * (drp_dp_prev + drp_dp_curr)
        r[0, j] = r_prev + 0.5 * d_phi * (rp_prev + r_phi[0, j])

    # Propagate along θ direction for each φ
    for j in range(K_phi):
        for i in range(1, K_theta):
            r_prev = r[i-1, j]
            rt_prev = r_theta[i-1, j]
            rp_prev = r_phi[i-1, j]
            Gamma_prev = get_gamma_at(i-1, j)
            L_prev, M_prev = L[i-1, j], M[i-1, j]

            drt_dt_prev, drp_dt_prev = gauss_rhs_theta(rt_prev, rp_prev, Gamma_prev, L_prev, M_prev)

            rt_pred = rt_prev + d_theta * drt_dt_prev
            rp_pred = rp_prev + d_theta * drp_dt_prev

            Gamma_curr = get_gamma_at(i, j)
            L_curr, M_curr = L[i, j], M[i, j]
            drt_dt_curr, drp_dt_curr = gauss_rhs_theta(rt_pred, rp_pred, Gamma_curr, L_curr, M_curr)

            r_theta[i, j] = rt_prev + 0.5 * d_theta * (drt_dt_prev + drt_dt_curr)
            r_phi[i, j] = rp_prev + 0.5 * d_theta * (drp_dt_prev + drp_dt_curr)
            r[i, j] = r_prev + 0.5 * d_theta * (rt_prev + r_theta[i, j])

    X = r[:, :, 0]
    Y = r[:, :, 1]
    Z = r[:, :, 2]

    return X, Y, Z


def reconstruct_from_model_spherical(
    model,
    t_val: float,
    K_grid: int = None,
    K_theta: int = 30,
    K_phi: int = 60,
    theta_range: Tuple[float, float] = (0.1, 3.04),
    phi_range: Tuple[float, float] = (0.0, 2 * np.pi),
    device: str = 'cpu',
    return_metrics: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Reconstruct surface from a trained spherical model at a specific time.

    Parameters
    ----------
    model : FundamentalFormNet
        Trained neural metric model with spherical topology.
    t_val : float
        Time value in [0, 1].
    K_theta : int
        Grid resolution in θ.
    K_phi : int
        Grid resolution in φ.
    theta_range : tuple
        (θ_min, θ_max) range.
    device : str
        Torch device.

    Returns
    -------
    X, Y, Z : ndarray (K_theta, K_phi)
        Reconstructed surface coordinates.
    metrics : dict
        Dictionary with fundamental forms and curvatures.
    """
    import torch
    import math

    if K_grid is not None:
        K_theta = K_grid
        K_phi = K_grid

    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range
    d_theta = (theta_max - theta_min) / (K_theta - 1)
    d_phi = (phi_max - phi_min) / (K_phi - 1) if K_phi > 1 else 0.0

    # Create grid
    theta_1d = torch.linspace(theta_min, theta_max, K_theta, device=device)
    phi_1d = torch.linspace(phi_min, phi_max, K_phi, device=device)
    Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_phi_flat = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)

    N_pts = theta_phi_flat.shape[0]
    t_col = torch.full((N_pts, 1), t_val, device=device)
    t_theta_phi = torch.cat([t_col, theta_phi_flat], dim=1)

    # Evaluate model. Do not use `no_grad()` here because endpoint-conditioned
    # models may compute analytic derivatives internally.
    model.eval()
    E, F, G, L, M, N_coef = model(t_theta_phi)

    # Reshape to grid
    E_np = E.reshape(K_theta, K_phi).detach().cpu().numpy()
    F_np = F.reshape(K_theta, K_phi).detach().cpu().numpy()
    G_np = G.reshape(K_theta, K_phi).detach().cpu().numpy()
    L_np = L.reshape(K_theta, K_phi).detach().cpu().numpy()
    M_np = M.reshape(K_theta, K_phi).detach().cpu().numpy()
    N_np = N_coef.reshape(K_theta, K_phi).detach().cpu().numpy()

    # Compute curvatures
    det_a = E_np * G_np - F_np**2
    K_gauss = (L_np * N_np - M_np**2) / (det_a + 1e-12)
    H_mean = (E_np * N_np + G_np * L_np - 2 * F_np * M_np) / (2 * det_a + 1e-12)

    # Reconstruct surface
    X, Y, Z = reconstruct_surface_spherical(E_np, F_np, G_np, L_np, M_np, N_np, d_theta, d_phi)

    metrics = {
        'E': E_np, 'F': F_np, 'G': G_np,
        'L': L_np, 'M': M_np, 'N': N_np,
        'K': K_gauss, 'H': H_mean,
        'det': det_a,
        'Theta': Theta.cpu().numpy(),
        'Phi': Phi.cpu().numpy(),
    }

    if return_metrics:
        return X, Y, Z, metrics
    return X, Y, Z
