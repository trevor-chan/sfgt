"""
geometry.py — Endpoint geometry for the stress-free growth PDAE solver.

Defines:
  - Elliptic grid transformation (square <-> circle)
  - First fundamental forms (metric tensors) for both endpoints
  - Green strain tensor and its dominant eigenvalue Λ
  - Gauss equation residual for flat surfaces (eq. 10)
  - Interpolation schemes (linear, logarithmic)
"""

import numpy as np
from typing import Tuple, Callable


# =============================================================================
# Elliptic grid transformation
# =============================================================================

def square_to_circle(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elliptic grid mapping from the unit square [-1,1]^2 to the unit disc.

        u = x * sqrt(1 - y^2/2)
        v = y * sqrt(1 - x^2/2)

    Parameters
    ----------
    x, y : ndarray
        Coordinates in [-1, 1] x [-1, 1].

    Returns
    -------
    u, v : ndarray
        Coordinates in the unit disc.
    """
    u = x * np.sqrt(1.0 - 0.5 * y**2)
    v = y * np.sqrt(1.0 - 0.5 * x**2)
    return u, v


def circle_to_square(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of the elliptic grid mapping (disc -> square).

    Uses the analytic inverse involving sqrt(2) and complementary terms.
    Valid for points strictly inside the disc; boundary requires care.
    """
    u2 = u**2
    v2 = v**2
    sqrt2 = np.sqrt(2.0)

    common = u2 - v2
    inner_x = 2.0 + common + sqrt2 * 2.0 * u
    inner_y = 2.0 - common + sqrt2 * 2.0 * v
    inner_x2 = 2.0 + common - sqrt2 * 2.0 * u
    inner_y2 = 2.0 - common - sqrt2 * 2.0 * v

    # Clamp to avoid negative sqrt arguments at boundaries
    inner_x = np.maximum(inner_x, 0.0)
    inner_y = np.maximum(inner_y, 0.0)
    inner_x2 = np.maximum(inner_x2, 0.0)
    inner_y2 = np.maximum(inner_y2, 0.0)

    x = 0.5 * (np.sqrt(inner_x) - np.sqrt(inner_x2))
    y = 0.5 * (np.sqrt(inner_y) - np.sqrt(inner_y2))
    return x, y


# =============================================================================
# Metric tensors (first fundamental forms)
# =============================================================================

def metric_square(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Metric tensor for the unit square in its own (x, y) coordinates.

    a_sq = I  (identity), so E=1, F=0, G=1 everywhere.

    Returns
    -------
    e, f, g : ndarray
        Components of the metric tensor, each with the same shape as x.
    """
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)
    return ones, zeros, ones


def metric_circle(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Metric tensor for the unit disc expressed in the square's (x, y) coordinates,
    induced by the elliptic grid transformation.

    Computed by differentiating (u, v) = square_to_circle(x, y) with respect to
    (x, y) and forming a = J^T J where J is the Jacobian.

    The analytic result is:
        E = 1 - (x^2 - 1) y^2 / (x^2 - 2)
        F = -x y
        G = 1 + x^2 (-1 + 1/(2 - y^2))

    Returns
    -------
    E, F, G : ndarray
        Components of the metric tensor for the circle.
    """
    x2 = x**2
    y2 = y**2

    E = 1.0 - (x2 - 1.0) * y2 / (x2 - 2.0)
    F = -x * y
    G = 1.0 + x2 * (-1.0 + 1.0 / (2.0 - y2))

    return E, F, G


def metric_circle_from_jacobian(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the circle metric via explicit Jacobian construction (for validation).

    J = [[du/dx, du/dy],
         [dv/dx, dv/dy]]

    a = J^T J, so E = J[:,0]·J[:,0], F = J[:,0]·J[:,1], G = J[:,1]·J[:,1].
    """
    x2 = x**2
    y2 = y**2

    # u = x * sqrt(1 - y^2/2)
    sqrt_term_u = np.sqrt(1.0 - 0.5 * y2)
    du_dx = sqrt_term_u
    du_dy = x * (-y) / (2.0 * sqrt_term_u + 1e-300)  # avoid /0

    # v = y * sqrt(1 - x^2/2)
    sqrt_term_v = np.sqrt(1.0 - 0.5 * x2)
    dv_dx = y * (-x) / (2.0 * sqrt_term_v + 1e-300)
    dv_dy = sqrt_term_v

    E = du_dx**2 + dv_dx**2
    F = du_dx * du_dy + dv_dx * dv_dy
    G = du_dy**2 + dv_dy**2

    return E, F, G


def metric_det(e: np.ndarray, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Determinant of a 2x2 metric tensor: det(a) = eg - f^2."""
    return e * g - f**2


# =============================================================================
# Gauss equation for flat surfaces (eq. 10)
# =============================================================================

def gauss_residual(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    eps_u: np.ndarray, eps_v: np.ndarray,
    gam_u: np.ndarray, gam_v: np.ndarray,
    phi_u: np.ndarray, phi_v: np.ndarray,
    eps_vv: np.ndarray, phi_uv: np.ndarray, gam_uu: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the Gauss equation residual for flat surfaces (eq. 10 / eq. 4).

    For a flat surface with metric (ε, φ, γ), the Gauss equation reads:

    0 = (-2ε_vv + 4φ_uv - 2γ_uu)(εγ - φ²)
        - ε_u (2φ_v γ - γ_u γ - γ_v φ)
        + (2φ_u - ε_v)(2φ_v φ - γ_u φ - γ_v ε)
        + ε_v (ε_v φ - γ_u ε)
        - γ_u (ε_v φ - γ_u ε)

    Parameters
    ----------
    eps, phi, gam : ndarray
        Metric components ε, φ, γ at each grid point.
    eps_u, eps_v, ..., gam_uu : ndarray
        First and second partial derivatives of the metric components.

    Returns
    -------
    residual : ndarray
        Gauss equation residual at each grid point (should be ~0 for valid metrics).
    """
    det_a = eps * gam - phi**2

    # Derived from the Brioschi formula for Gaussian curvature set to K=0.
    # K*(EG-F^2)^2 = det1 - det2 = 0, multiplied through by 2.
    #
    # Note: the paper's eq. (4) has an error in the last terms.
    # The correct form (from Brioschi) has eps_v^2 * gam, not eps_v^2 * phi.
    term1 = (-2.0 * eps_vv + 4.0 * phi_uv - 2.0 * gam_uu) * det_a
    term2 = -eps_u * (2.0 * phi_v * gam - gam_u * gam - phi * gam_v)
    term3 = (2.0 * phi_u - eps_v) * (2.0 * phi_v * phi - gam_u * phi - eps * gam_v)
    term4 = eps_v**2 * gam - 2.0 * eps_v * gam_u * phi + gam_u**2 * eps

    return term1 + term2 + term3 + term4


# =============================================================================
# Green strain tensor and dominant eigenvalue Λ
# =============================================================================

def green_strain_eigenvalue(
    e_r: np.ndarray, f_r: np.ndarray, g_r: np.ndarray,
    e_c: np.ndarray, f_c: np.ndarray, g_c: np.ndarray,
) -> np.ndarray:
    """
    Compute Λ[ε_rc], the largest-magnitude eigenvalue of the Green strain tensor
    ε_rc = (1/2) a_r^{-1} (a_c - a_r), evaluated pointwise.

    This follows from solving the characteristic polynomial of the 2x2 strain
    tensor analytically.

    Parameters
    ----------
    e_r, f_r, g_r : ndarray
        Metric components of the reference surface.
    e_c, f_c, g_c : ndarray
        Metric components of the current surface.

    Returns
    -------
    lam : ndarray
        Dominant eigenvalue Λ at each grid point.
    """
    det_r = metric_det(e_r, f_r, g_r)

    # Components of a_r^{-1} (a_c - a_r)  (before the 1/2 factor)
    # a_r^{-1} = (1/det_r) * [[g_r, -f_r], [-f_r, e_r]]
    # a_c - a_r = [[e_c - e_r, f_c - f_r], [f_c - f_r, g_c - g_r]]
    de = e_c - e_r
    df = f_c - f_r
    dg = g_c - g_r

    # M = a_r^{-1} @ (a_c - a_r), unnormalized by det_r
    M11 = g_r * de - f_r * df
    M12 = g_r * df - f_r * dg
    M21 = -f_r * de + e_r * df
    M22 = -f_r * df + e_r * dg

    # Eigenvalues of (1/2) M / det_r via trace and determinant
    tr = (M11 + M22) / det_r  # trace of M/det_r
    det_M = (M11 * M22 - M12 * M21) / det_r**2  # det of M/det_r

    # Eigenvalues of the strain tensor (with the 1/2):
    #   λ = (1/2) * (tr/2 ± sqrt((tr/2)^2 - det_M))
    # But we want Λ = eigenvalue of largest absolute value of (1/2)(M/det_r)
    half_tr = 0.5 * tr
    disc = half_tr**2 - det_M
    disc = np.maximum(disc, 0.0)  # numerical safety
    sqrt_disc = np.sqrt(disc)

    lam1 = 0.5 * (half_tr + sqrt_disc)
    lam2 = 0.5 * (half_tr - sqrt_disc)

    # Select eigenvalue with largest absolute value
    lam = np.where(np.abs(lam1) >= np.abs(lam2), lam1, lam2)
    return lam


def green_strain_eigenvalue_direct(
    e_r: np.ndarray, f_r: np.ndarray, g_r: np.ndarray,
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
) -> np.ndarray:
    """
    Compute Λ using explicit formulas for the strain tensor eigenvalue,
    expressed directly in terms of the reference metric (e,f,g) and current
    metric (ε, φ, γ).

    The strain tensor is S = (1/2) a_r^{-1} (a_c - a_r), which has entries:

        M = a_r^{-1} (a_c - a_r) * det(a_r)  (unnormalized)
        M11 = g_r*(ε - e_r) - f_r*(φ - f_r)
        M12 = g_r*(φ - f_r) - f_r*(γ - g_r)
        M21 = -f_r*(ε - e_r) + e_r*(φ - f_r)
        M22 = -f_r*(φ - f_r) + e_r*(γ - g_r)

    S = M / (2 * det(a_r))

    Eigenvalues found via trace and determinant of S.

    Note: the paper's eq. (7) explicit formula has errors in the strain matrix
    entries. This function uses the correctly derived form.
    """
    det_r = metric_det(e_r, f_r, g_r)

    # Entries of M = [[g_r, -f_r], [-f_r, e_r]] @ [[eps-e_r, phi-f_r], [phi-f_r, gam-g_r]]
    de = eps - e_r
    df = phi - f_r
    dg = gam - g_r

    M11 = g_r * de - f_r * df
    M12 = g_r * df - f_r * dg
    M21 = -f_r * de + e_r * df
    M22 = -f_r * df + e_r * dg

    # Eigenvalues of S = M / (2*det_r) via trace and determinant
    tr_M = (M11 + M22) / det_r          # trace of M/det_r
    det_M = (M11 * M22 - M12 * M21) / det_r**2  # det of M/det_r

    half_tr = 0.5 * tr_M
    disc = half_tr**2 - det_M
    disc = np.maximum(disc, 0.0)
    sqrt_disc = np.sqrt(disc)

    # Eigenvalues of strain S = (1/2)(M/det_r):
    lam1 = 0.5 * (half_tr + sqrt_disc)
    lam2 = 0.5 * (half_tr - sqrt_disc)

    # Select eigenvalue with largest absolute value
    lam = np.where(np.abs(lam1) >= np.abs(lam2), lam1, lam2)
    return lam


# =============================================================================
# Interpolation schemes
# =============================================================================

def linear_interpolation(lam_endpoint: np.ndarray, t: float) -> np.ndarray:
    """
    Linear interpolation: Λ(t) = (1 - t) * Λ_endpoint.

    For the forward direction (eq. 11), Λ_endpoint = Λ[ε_of].
    At t=0, Λ(0) = Λ[ε_of] (maximum strain to final).
    At t=1, Λ(1) = 0 (we've arrived at the final surface).
    """
    return (1.0 - t) * lam_endpoint


def logarithmic_interpolation(lam_endpoint: np.ndarray, t: float, k: float = 5.0) -> np.ndarray:
    """
    Logarithmic (pathological) interpolation that deliberately breaks point
    symmetry about t = 0.5.

    Λ(t) = Λ_endpoint * log(1 + k(1-t)) / log(1 + k)

    This compresses deformation toward the beginning of the trajectory.
    """
    return lam_endpoint * np.log(1.0 + k * (1.0 - t)) / np.log(1.0 + k)


# =============================================================================
# Algebraic constraint evaluation (eqs. 11, 12)
# =============================================================================

def algebraic_constraint_forward(
    e: np.ndarray, f: np.ndarray, g: np.ndarray,
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    t: float,
    lam_of: np.ndarray,
    interp_fn: Callable = linear_interpolation,
) -> np.ndarray:
    """
    Residual of the forward algebraic constraint (eq. 11):

        interp(Λ[ε_of], t) - Λ_direct(a_o; ε, φ, γ) = 0

    Parameters
    ----------
    e, f, g : ndarray
        Initial surface metric (a_o).
    eps, phi, gam : ndarray
        Current metric components.
    t : float
        Current time in [0, 1].
    lam_of : ndarray
        Precomputed Λ[ε_of] field (strain from initial to final).
    interp_fn : callable
        Interpolation function mapping (lam_endpoint, t) -> target.

    Returns
    -------
    residual : ndarray
        Should be ~0 when the constraint is satisfied.
    """
    target = interp_fn(lam_of, t)
    actual = green_strain_eigenvalue_direct(e, f, g, eps, phi, gam)
    return target - actual


def algebraic_constraint_reverse(
    E: np.ndarray, F: np.ndarray, G: np.ndarray,
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    t: float,
    lam_fo: np.ndarray,
    interp_fn: Callable = linear_interpolation,
) -> np.ndarray:
    """
    Residual of the reverse algebraic constraint (eq. 12):

        interp(Λ[ε_fo], 1-t) - Λ_direct(a_f; ε, φ, γ) = 0

    Parameters
    ----------
    E, F, G : ndarray
        Final surface metric (a_f).
    eps, phi, gam : ndarray
        Current metric components.
    t : float
        Current time in [0, 1].
    lam_fo : ndarray
        Precomputed Λ[ε_fo] field (strain from final to initial).
    interp_fn : callable
        Interpolation function.

    Returns
    -------
    residual : ndarray
    """
    target = interp_fn(lam_fo, 1.0 - t)
    actual = green_strain_eigenvalue_direct(E, F, G, eps, phi, gam)
    return target - actual


# =============================================================================
# Convenience: build a grid and compute all endpoint quantities
# =============================================================================

def build_endpoint_data(K: int, margin: float = 1e-6):
    """
    Set up the K×K grid on [-1,1]^2 and compute all endpoint geometry.

    Parameters
    ----------
    K : int
        Number of grid points in each direction.
    margin : float
        Small inset from the boundary to avoid singularities at corners
        of the elliptic grid transformation (where the Jacobian is singular).

    Returns
    -------
    data : dict with keys:
        'x', 'y'       : 2D coordinate arrays (K, K)
        'dx', 'dy'     : grid spacings
        'e', 'f', 'g'  : initial metric (square)
        'E', 'F', 'G'  : final metric (circle)
        'lam_of'        : Λ[ε_of] field
        'lam_fo'        : Λ[ε_fo] field
    """
    lin = np.linspace(-1.0 + margin, 1.0 - margin, K)
    x, y = np.meshgrid(lin, lin, indexing='ij')
    dx = lin[1] - lin[0]
    dy = dx  # uniform grid

    # Endpoint metrics
    e, f, g = metric_square(x, y)
    E, F, G = metric_circle(x, y)

    # Strain eigenvalues at endpoints
    lam_of = green_strain_eigenvalue(e, f, g, E, F, G)
    lam_fo = green_strain_eigenvalue(E, F, G, e, f, g)

    return {
        'x': x, 'y': y,
        'dx': dx, 'dy': dy,
        'K': K,
        'e': e, 'f': f, 'g': g,
        'E': E, 'F': F, 'G': G,
        'lam_of': lam_of,
        'lam_fo': lam_fo,
    }
