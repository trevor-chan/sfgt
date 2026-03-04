"""
visualization.py — Plotting and surface reconstruction for the SFG solver.

Provides:
  - reconstruct_surface()    : Flat-surface embedding from first fundamental form
  - plot_metric_fields()     : Heatmaps of (ε, φ, γ, det) at a snapshot
  - plot_morphing_mesh()     : Deformed grid at multiple time snapshots
  - plot_convergence()       : Residual norms and Newton iterations vs. time
  - plot_trajectory_summary(): Combined multi-panel figure
  - plot_det_evolution()     : Determinant field evolution across time
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from typing import Optional, List, Tuple, Dict

from geometry import metric_det, square_to_circle


# =============================================================================
# Surface reconstruction from first fundamental form
# =============================================================================

def _fd_gradient(f, h, axis):
    """
    4th-order central finite differences in interior, 2nd-order at boundaries.
    Much more accurate than np.gradient for computing Christoffel symbols,
    especially near the domain boundary where the elliptic map metric
    has strong gradients.
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


def _compute_christoffel_symbols(eps, phi, gam, du, dv):
    """
    Compute all 6 Christoffel symbols Γⁱⱼₖ from the metric and its derivatives.

    Uses 4th-order central differences in the interior and 2nd-order
    one-sided stencils at boundaries for accuracy near domain edges where
    the elliptic map metric has strong gradients.

    Uses the standard formula:
        Γⁱⱼₖ = ½ gⁱˡ (∂gₗⱼ/∂xᵏ + ∂gₗₖ/∂xʲ - ∂gⱼₖ/∂xˡ)

    Returns
    -------
    dict with keys 'G111', 'G112', 'G122', 'G211', 'G212', 'G222',
    each an (K,K) array.  Notation: G_ijk = Γⁱⱼₖ = Γ^i_{jk}.
    """
    eps_u = _fd_gradient(eps, du, axis=0)
    eps_v = _fd_gradient(eps, dv, axis=1)
    phi_u = _fd_gradient(phi, du, axis=0)
    phi_v = _fd_gradient(phi, dv, axis=1)
    gam_u = _fd_gradient(gam, du, axis=0)
    gam_v = _fd_gradient(gam, dv, axis=1)

    det = eps * gam - phi**2
    det = np.maximum(det, 1e-30)
    inv2det = 0.5 / det

    return {
        'G111': inv2det * (gam * eps_u - 2 * phi * phi_u + phi * eps_v),
        'G211': inv2det * (2 * eps * phi_u - eps * eps_v - phi * eps_u),
        'G112': inv2det * (gam * eps_v - phi * gam_u),
        'G212': inv2det * (eps * gam_u - phi * eps_v),
        'G122': inv2det * (2 * gam * phi_v - gam * gam_u - phi * gam_v),
        'G222': inv2det * (eps * gam_v - 2 * phi * phi_v + phi * gam_u),
    }


def reconstruct_surface(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    du: float, dv: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a flat surface embedding r(u,v) = (x,y) from its first
    fundamental form a = [[ε, φ], [φ, γ]].

    For a flat surface (K_Gauss = 0), the embedding Jacobian J = [e₁|e₂]
    satisfies JᵀJ = a and the parallel-transport equation:

        ∂J/∂u = J · M_u,    ∂J/∂v = J · M_v

    where M_u, M_v are the Christoffel-symbol matrices:

        (M_β)_{γα} = Γ^γ_{αβ}

    Algorithm:
      1. Compute Christoffel symbols from the metric using np.gradient
         (proper one-sided stencils at boundaries)
      2. Initialize J at (0,0) via Cholesky (fixes the global rotation gauge)
      3. Propagate J along the left edge (v-direction) via trapezoidal integration
      4. Propagate J across each row (u-direction) via trapezoidal integration
      5. Integrate r(u,v) from J using trapezoidal quadrature

    Parameters
    ----------
    eps, phi, gam : ndarray (K, K)
        First fundamental form components.
    du, dv : float
        Grid spacing.

    Returns
    -------
    X, Y : ndarray (K, K)
        Reconstructed surface coordinates.
    """
    K = eps.shape[0]

    # Christoffel symbols (using np.gradient for correct boundary derivatives)
    G = _compute_christoffel_symbols(eps, phi, gam, du, dv)

    det = eps * gam - phi**2
    det = np.maximum(det, 1e-30)
    sqrt_eps = np.sqrt(np.maximum(eps, 1e-30))
    sqrt_det = np.sqrt(det)

    # Jacobian field J[i,j] is a 2×2 matrix:
    #   J[:,:,0,0] = ∂x/∂u,  J[:,:,0,1] = ∂x/∂v
    #   J[:,:,1,0] = ∂y/∂u,  J[:,:,1,1] = ∂y/∂v
    J = np.zeros((K, K, 2, 2))

    # Initialize at (0,0) via upper Cholesky (gauge choice: frame aligned with x-axis)
    # The metric is g = JᵀJ, so J should satisfy JᵀJ = a.
    # Upper Cholesky: a = UᵀU with U = [[√ε, φ/√ε], [0, √det/√ε]]
    # gives JᵀJ = UᵀU = a.  ✓
    J[0, 0, 0, 0] = sqrt_eps[0, 0]                       # ∂x/∂u
    J[0, 0, 0, 1] = phi[0, 0] / sqrt_eps[0, 0]           # ∂x/∂v
    J[0, 0, 1, 0] = 0.0                                   # ∂y/∂u
    J[0, 0, 1, 1] = sqrt_det[0, 0] / sqrt_eps[0, 0]      # ∂y/∂v

    # Propagation matrices at each grid point
    #   ∂(J col α)/∂x^β = Σ_γ Γ^γ_{αβ} (J col γ)
    #   In matrix form: ∂J/∂x^β = J · M_β  where (M_β)_{γα} = Γ^γ_{αβ}
    #
    # M_u: (M_u)_{γα} = Γ^γ_{α,1}
    #   [[Γ¹₁₁, Γ¹₂₁], [Γ²₁₁, Γ²₂₁]] = [[G111, G112], [G211, G212]]
    M_u = np.zeros((K, K, 2, 2))
    M_u[:, :, 0, 0] = G['G111']
    M_u[:, :, 0, 1] = G['G112']
    M_u[:, :, 1, 0] = G['G211']
    M_u[:, :, 1, 1] = G['G212']

    # M_v: (M_v)_{γα} = Γ^γ_{α,2}
    #   [[Γ¹₁₂, Γ¹₂₂], [Γ²₁₂, Γ²₂₂]] = [[G112, G122], [G212, G222]]
    M_v = np.zeros((K, K, 2, 2))
    M_v[:, :, 0, 0] = G['G112']
    M_v[:, :, 0, 1] = G['G122']
    M_v[:, :, 1, 0] = G['G212']
    M_v[:, :, 1, 1] = G['G222']

    # Propagate J along left edge (i=0, varying j in v-direction)
    for j in range(1, K):
        rhs_prev = J[0, j-1] @ M_v[0, j-1]
        J_pred = J[0, j-1] + dv * rhs_prev                    # Euler predictor
        rhs_curr = J_pred @ M_v[0, j]
        J[0, j] = J[0, j-1] + 0.5 * dv * (rhs_prev + rhs_curr)  # trapezoidal

    # Propagate J across each row (varying i in u-direction)
    for j in range(K):
        for i in range(1, K):
            rhs_prev = J[i-1, j] @ M_u[i-1, j]
            J_pred = J[i-1, j] + du * rhs_prev
            rhs_curr = J_pred @ M_u[i, j]
            J[i, j] = J[i-1, j] + 0.5 * du * (rhs_prev + rhs_curr)

    # =====================================================================
    # Reconstruct positions via Poisson solve
    # =====================================================================
    # The parallel-transported J has approximately correct metric (JᵀJ ≈ a)
    # but accumulates path-dependent rotation drift from numerical curvature
    # at corners.  Instead of path-integrating r(u,v) from J (which bakes
    # in the drift), we solve:
    #
    #   ΔX = ∂J₀₀/∂u + ∂J₀₁/∂v   (divergence of row-0 of J)
    #   ΔY = ∂J₁₀/∂u + ∂J₁₁/∂v   (divergence of row-1 of J)
    #
    # with Dirichlet BCs from path-integrated boundary values.
    # This projects J onto the closest curl-free gradient field, spreading
    # errors globally rather than accumulating along integration paths.

    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    # --- Step A: Path-integrate boundary positions (for Dirichlet BCs) ---
    X_bnd = np.full((K, K), np.nan)
    Y_bnd = np.full((K, K), np.nan)

    # Start at (0,0) = origin
    X_bnd[0, 0] = 0.0
    Y_bnd[0, 0] = 0.0

    # Left edge: (0,0) → (0,K-1) in v
    for j in range(1, K):
        X_bnd[0, j] = X_bnd[0, j-1] + 0.5 * dv * (J[0, j-1, 0, 1] + J[0, j, 0, 1])
        Y_bnd[0, j] = Y_bnd[0, j-1] + 0.5 * dv * (J[0, j-1, 1, 1] + J[0, j, 1, 1])

    # Bottom edge: (0,0) → (K-1,0) in u
    for i in range(1, K):
        X_bnd[i, 0] = X_bnd[i-1, 0] + 0.5 * du * (J[i-1, 0, 0, 0] + J[i, 0, 0, 0])
        Y_bnd[i, 0] = Y_bnd[i-1, 0] + 0.5 * du * (J[i-1, 0, 1, 0] + J[i, 0, 1, 0])

    # Top edge: (0,K-1) → (K-1,K-1) in u
    for i in range(1, K):
        X_bnd[i, -1] = X_bnd[i-1, -1] + 0.5 * du * (J[i-1, -1, 0, 0] + J[i, -1, 0, 0])
        Y_bnd[i, -1] = Y_bnd[i-1, -1] + 0.5 * du * (J[i-1, -1, 1, 0] + J[i, -1, 1, 0])

    # Right edge: (K-1,0) → (K-1,K-1) in v
    for j in range(1, K):
        X_bnd[-1, j] = X_bnd[-1, j-1] + 0.5 * dv * (J[-1, j-1, 0, 1] + J[-1, j, 0, 1])
        Y_bnd[-1, j] = Y_bnd[-1, j-1] + 0.5 * dv * (J[-1, j-1, 1, 1] + J[-1, j, 1, 1])

    # Average the two estimates at corners (K-1, K-1)
    # Top-right corner gets values from both top edge and right edge
    # Average them for consistency
    X_tr_top = X_bnd[-1, -1]   # from top edge (already set)
    Y_tr_top = Y_bnd[-1, -1]
    # Recompute from right edge endpoint:
    X_tr_right = X_bnd[-1, -1]  # already the right-edge value
    Y_tr_right = Y_bnd[-1, -1]
    # (These might differ due to curvature; we already have the last write)

    # --- Step B: Build discrete Laplacian (5-point stencil, full grid) ---
    N = K * K
    # Laplacian = D_uu + D_vv with standard 2nd-order central differences
    # Interior: (f[i+1,j] - 2f[i,j] + f[i-1,j])/du² + (f[i,j+1] - 2f[i,j] + f[i,j-1])/dv²

    rows, cols, vals = [], [], []
    rhs_X = np.zeros(N)
    rhs_Y = np.zeros(N)

    def idx(i, j):
        return i * K + j

    # Divergence of J rows (RHS of Poisson equation)
    div_X = np.gradient(J[:, :, 0, 0], du, axis=0) + np.gradient(J[:, :, 0, 1], dv, axis=1)
    div_Y = np.gradient(J[:, :, 1, 0], du, axis=0) + np.gradient(J[:, :, 1, 1], dv, axis=1)

    for i in range(K):
        for j in range(K):
            n = idx(i, j)
            is_bnd = (i == 0 or i == K-1 or j == 0 or j == K-1)

            if is_bnd:
                # Dirichlet BC: X[n] = X_bnd, Y[n] = Y_bnd
                rows.append(n); cols.append(n); vals.append(1.0)
                rhs_X[n] = X_bnd[i, j]
                rhs_Y[n] = Y_bnd[i, j]
            else:
                # 5-point Laplacian
                cu = 1.0 / (du * du)
                cv = 1.0 / (dv * dv)
                rows.append(n); cols.append(n);          vals.append(-2*cu - 2*cv)
                rows.append(n); cols.append(idx(i-1,j)); vals.append(cu)
                rows.append(n); cols.append(idx(i+1,j)); vals.append(cu)
                rows.append(n); cols.append(idx(i,j-1)); vals.append(cv)
                rows.append(n); cols.append(idx(i,j+1)); vals.append(cv)
                rhs_X[n] = div_X[i, j]
                rhs_Y[n] = div_Y[i, j]

    L_mat = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Solve
    X = spla.spsolve(L_mat, rhs_X).reshape(K, K)
    Y = spla.spsolve(L_mat, rhs_Y).reshape(K, K)

    X -= X.mean()
    Y -= Y.mean()

    return X, Y


def reconstruct_endpoint_surfaces(
    data: Dict, K: int, margin: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the known endpoint surfaces for comparison.

    Returns
    -------
    dict with keys 'square' and 'circle', each (X, Y) arrays.
    """
    L = 1.0 - 2 * margin
    du = L / (K - 1)
    u = np.linspace(margin, 1 - margin, K)
    v = np.linspace(margin, 1 - margin, K)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Square: identity metric → positions are just (u, v)
    X_sq = U - U.mean()
    Y_sq = V - V.mean()

    # Circle: elliptic map (vectorized)
    X_circ, Y_circ = square_to_circle(U, V)
    X_circ -= X_circ.mean()
    Y_circ -= Y_circ.mean()

    return {'square': (X_sq, Y_sq), 'circle': (X_circ, Y_circ)}


# =============================================================================
# Grid mesh drawing helper
# =============================================================================

def _draw_mesh(ax, X, Y, color='k', linewidth=0.5, alpha=0.7):
    """Draw a 2D mesh from coordinate arrays X, Y on axes ax."""
    K = X.shape[0]

    # u-lines (constant v)
    for j in range(K):
        ax.plot(X[:, j], Y[:, j], color=color, linewidth=linewidth, alpha=alpha)
    # v-lines (constant u)
    for i in range(K):
        ax.plot(X[i, :], Y[i, :], color=color, linewidth=linewidth, alpha=alpha)


def _draw_colored_mesh(ax, X, Y, field, cmap='viridis', linewidth=0.5):
    """Draw mesh with line color determined by a scalar field."""
    K = X.shape[0]
    norm = mcolors.Normalize(vmin=np.nanmin(field), vmax=np.nanmax(field))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # u-lines
    for j in range(K):
        points = np.column_stack([X[:, j], Y[:, j]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vals = 0.5 * (field[:-1, j] + field[1:, j])
        lc = LineCollection(segments, colors=sm.to_rgba(vals), linewidth=linewidth)
        ax.add_collection(lc)

    # v-lines
    for i in range(K):
        points = np.column_stack([X[i, :], Y[i, :]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vals = 0.5 * (field[i, :-1] + field[i, 1:])
        lc = LineCollection(segments, colors=sm.to_rgba(vals), linewidth=linewidth)
        ax.add_collection(lc)

    return sm


# =============================================================================
# Plot: metric fields at a single snapshot
# =============================================================================

def plot_metric_fields(
    eps: np.ndarray, phi: np.ndarray, gam: np.ndarray,
    t: float,
    figsize: Tuple[float, float] = (14, 3.5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot heatmaps of ε, φ, γ, and det(a) at a single time snapshot.
    """
    det = metric_det(eps, phi, gam)

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fields = [eps, phi, gam, det]
    titles = [r'$\varepsilon$ (a₁₁)', r'$\varphi$ (a₁₂)',
              r'$\gamma$ (a₂₂)', r'det(a)']
    cmaps = ['RdBu_r', 'PiYG', 'RdBu_r', 'YlOrRd']

    for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
        vabs = max(abs(np.nanmin(field)), abs(np.nanmax(field)))
        if cmap in ('RdBu_r', 'PiYG'):
            im = ax.imshow(field.T, origin='lower', cmap=cmap,
                           vmin=-vabs, vmax=vabs)
        else:
            im = ax.imshow(field.T, origin='lower', cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f'Metric fields at t = {t:.3f}', fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Plot: morphing mesh at multiple times
# =============================================================================

def plot_morphing_mesh(
    snapshots: list,
    data: Dict,
    margin: float = 0.05,
    times: Optional[List[float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    color_by_det: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot reconstructed surface meshes at selected time snapshots.

    Parameters
    ----------
    snapshots : list of Snapshot objects from TrajectoryResult
    data : endpoint data dict
    margin : float
    times : list of float, times to plot (picks nearest snapshot)
    color_by_det : if True, color lines by det(a)
    """
    if times is None:
        n = len(snapshots)
        if n <= 6:
            indices = list(range(n))
        else:
            indices = [int(i) for i in np.linspace(0, n-1, 6)]
        selected = [snapshots[i] for i in indices]
    else:
        snap_times = np.array([s.t for s in snapshots])
        selected = []
        for t in times:
            idx = np.argmin(np.abs(snap_times - t))
            selected.append(snapshots[idx])

    n_panels = len(selected)
    if figsize is None:
        figsize = (3.5 * n_panels, 3.5)

    K = selected[0].eps.shape[0]
    L = 1.0 - 2 * margin
    du = L / (K - 1)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Compute axis limits from all panels
    all_x, all_y = [], []
    surfaces = []
    for snap in selected:
        X, Y = reconstruct_surface(snap.eps, snap.phi, snap.gam, du, du)
        surfaces.append((X, Y))
        all_x.extend([X.min(), X.max()])
        all_y.extend([Y.min(), Y.max()])

    pad = 0.05
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    for ax, snap, (X, Y) in zip(axes, selected, surfaces):
        det = metric_det(snap.eps, snap.phi, snap.gam)

        if color_by_det:
            sm = _draw_colored_mesh(ax, X, Y, det, cmap='viridis', linewidth=0.6)
        else:
            _draw_mesh(ax, X, Y, color='steelblue', linewidth=0.6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f't = {snap.t:.3f}', fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle('Surface morphing trajectory', fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Plot: endpoint comparison
# =============================================================================

def plot_endpoint_comparison(
    result,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare reconstructed endpoints (t=0, t=1) with known square and circle.
    """
    data = result.endpoint_data
    K = result.config.K
    margin = result.config.margin
    L = 1.0 - 2 * margin
    du = L / (K - 1)

    endpoints = reconstruct_endpoint_surfaces(data, K, margin)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # t=0: compare with square
    snap_0 = result.snapshots[0]
    X0, Y0 = reconstruct_surface(snap_0.eps, snap_0.phi, snap_0.gam, du, du)
    X_sq, Y_sq = endpoints['square']

    ax = axes[0]
    _draw_mesh(ax, X_sq, Y_sq, color='gray', linewidth=0.4, alpha=0.5)
    _draw_mesh(ax, X0, Y0, color='steelblue', linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_title(f't = 0 (blue) vs. square (gray)', fontsize=10)
    ax.tick_params(labelsize=7)

    # t=1: compare with circle
    snap_1 = result.snapshots[-1]
    X1, Y1 = reconstruct_surface(snap_1.eps, snap_1.phi, snap_1.gam, du, du)
    X_circ, Y_circ = endpoints['circle']

    ax = axes[1]
    _draw_mesh(ax, X_circ, Y_circ, color='gray', linewidth=0.4, alpha=0.5)
    _draw_mesh(ax, X1, Y1, color='orangered', linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_title(f't = {snap_1.t:.3f} (red) vs. circle (gray)', fontsize=10)
    ax.tick_params(labelsize=7)

    fig.suptitle('Endpoint comparison: reconstructed vs. known', fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Plot: convergence diagnostics
# =============================================================================

def plot_convergence(
    result,
    figsize: Tuple[float, float] = (12, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot residual norms and Newton iterations across the trajectory.
    """
    times = result.times
    gauss_norms = [s.residual_norms.get('gauss', np.nan) for s in result.snapshots]
    alg_fwd = [s.residual_norms.get('alg_fwd', np.nan) for s in result.snapshots]
    alg_rev = [s.residual_norms.get('alg_rev', np.nan) for s in result.snapshots]
    total = [s.residual_norms.get('total', np.nan) for s in result.snapshots]
    iters = [s.newton_iters for s in result.snapshots]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: residual norms
    ax1.semilogy(times, total, 'k-o', markersize=3, label='Total')
    ax1.semilogy(times, gauss_norms, 's-', markersize=2, alpha=0.7, label='Gauss')
    ax1.semilogy(times, alg_fwd, '^-', markersize=2, alpha=0.7, label='Alg fwd')
    ax1.semilogy(times, alg_rev, 'v-', markersize=2, alpha=0.7, label='Alg rev')
    ax1.set_xlabel('t')
    ax1.set_ylabel('Residual norm (L∞)')
    ax1.set_title('Residual norms along trajectory')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: Newton iterations per step
    ax2.bar(times, iters, width=0.8 * np.min(np.diff(times)) if len(times) > 1 else 0.01,
            color='steelblue', alpha=0.7)
    ax2.set_xlabel('t')
    ax2.set_ylabel('Newton iterations')
    ax2.set_title('Newton iterations per time step')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Plot: determinant evolution
# =============================================================================

def plot_det_evolution(
    result,
    times: Optional[List[float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the metric determinant field at selected time snapshots.
    """
    if times is None:
        n = len(result.snapshots)
        if n <= 6:
            indices = list(range(n))
        else:
            indices = [int(i) for i in np.linspace(0, n-1, 6)]
        selected = [result.snapshots[i] for i in indices]
    else:
        snap_times = result.times
        selected = []
        for t in times:
            idx = np.argmin(np.abs(snap_times - t))
            selected.append(result.snapshots[idx])

    n_panels = len(selected)
    if figsize is None:
        figsize = (3 * n_panels, 3)

    # Global colorbar range
    all_dets = [metric_det(s.eps, s.phi, s.gam) for s in selected]
    vmin = min(d.min() for d in all_dets)
    vmax = max(d.max() for d in all_dets)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, snap, det in zip(axes, selected, all_dets):
        im = ax.imshow(det.T, origin='lower', cmap='YlOrRd',
                       vmin=vmin, vmax=vmax)
        ax.set_title(f't = {snap.t:.3f}\ndet ∈ [{det.min():.3f}, {det.max():.3f}]',
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04, label='det(a)')
    fig.suptitle('Metric determinant evolution', fontsize=12, y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Plot: full trajectory summary dashboard
# =============================================================================

def plot_trajectory_summary(
    result,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Combined summary figure: morphing mesh + convergence + determinant range.
    """
    snapshots = result.snapshots
    data = result.endpoint_data
    K = result.config.K
    margin = result.config.margin
    L = 1.0 - 2 * margin
    du = L / (K - 1)

    # Select up to 5 snapshots for mesh panels
    n = len(snapshots)
    if n <= 5:
        mesh_indices = list(range(n))
    else:
        mesh_indices = [int(i) for i in np.linspace(0, n-1, 5)]
    mesh_snaps = [snapshots[i] for i in mesh_indices]
    n_mesh = len(mesh_snaps)

    fig = plt.figure(figsize=(16, 10))

    # Top row: morphing meshes
    mesh_axes = []
    for i in range(n_mesh):
        ax = fig.add_subplot(2, n_mesh, i + 1)
        mesh_axes.append(ax)

    surfaces = []
    all_x, all_y = [], []
    for snap in mesh_snaps:
        X, Y = reconstruct_surface(snap.eps, snap.phi, snap.gam, du, du)
        surfaces.append((X, Y))
        all_x.extend([X.min(), X.max()])
        all_y.extend([Y.min(), Y.max()])

    pad = 0.05
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    for ax, snap, (X, Y) in zip(mesh_axes, mesh_snaps, surfaces):
        det = metric_det(snap.eps, snap.phi, snap.gam)
        _draw_colored_mesh(ax, X, Y, det, cmap='viridis', linewidth=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f't = {snap.t:.3f}', fontsize=9)
        ax.tick_params(labelsize=6)

    # Bottom left: residual norms
    ax_res = fig.add_subplot(2, 3, 4)
    times = result.times
    total = [s.residual_norms.get('total', np.nan) for s in snapshots]
    gauss = [s.residual_norms.get('gauss', np.nan) for s in snapshots]
    ax_res.semilogy(times, total, 'k-o', markersize=3, label='Total')
    ax_res.semilogy(times, gauss, 's-', markersize=2, alpha=0.7, label='Gauss')
    ax_res.set_xlabel('t', fontsize=9)
    ax_res.set_ylabel('|R|∞', fontsize=9)
    ax_res.set_title('Residual norms', fontsize=10)
    ax_res.legend(fontsize=7)
    ax_res.grid(True, alpha=0.3)
    ax_res.tick_params(labelsize=7)

    # Bottom center: Newton iterations
    ax_newton = fig.add_subplot(2, 3, 5)
    iters = [s.newton_iters for s in snapshots]
    bar_width = 0.8 * np.min(np.diff(times)) if len(times) > 1 else 0.01
    ax_newton.bar(times, iters, width=bar_width, color='steelblue', alpha=0.7)
    ax_newton.set_xlabel('t', fontsize=9)
    ax_newton.set_ylabel('Iterations', fontsize=9)
    ax_newton.set_title('Newton iterations', fontsize=10)
    ax_newton.grid(True, alpha=0.3, axis='y')
    ax_newton.tick_params(labelsize=7)

    # Bottom right: det(a) range
    ax_det = fig.add_subplot(2, 3, 6)
    det_min = [np.min(metric_det(s.eps, s.phi, s.gam)) for s in snapshots]
    det_max = [np.max(metric_det(s.eps, s.phi, s.gam)) for s in snapshots]
    ax_det.fill_between(times, det_min, det_max, alpha=0.3, color='forestgreen')
    ax_det.plot(times, det_min, 'g-', linewidth=1, label='min det(a)')
    ax_det.plot(times, det_max, 'g--', linewidth=1, label='max det(a)')
    ax_det.axhline(0, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax_det.set_xlabel('t', fontsize=9)
    ax_det.set_ylabel('det(a)', fontsize=9)
    ax_det.set_title('Metric determinant range', fontsize=10)
    ax_det.legend(fontsize=7)
    ax_det.grid(True, alpha=0.3)
    ax_det.tick_params(labelsize=7)

    fig.suptitle(
        f'SFG Trajectory Summary — K={K}, dt={result.config.dt}, '
        f'{n} steps, {"CONVERGED" if result.converged else "FAILED"}',
        fontsize=13, y=1.01,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.92)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# =============================================================================
# Generate all plots for a trajectory result
# =============================================================================

def generate_all_plots(
    result,
    output_dir: str = '/mnt/user-data/outputs',
    prefix: str = 'sfg',
) -> List[str]:
    """
    Generate and save all standard plots for a trajectory result.

    Returns list of saved file paths.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    # 1. Summary dashboard
    path = os.path.join(output_dir, f'{prefix}_summary.png')
    fig = plot_trajectory_summary(result, save_path=path)
    plt.close(fig)
    saved.append(path)

    # 2. Morphing mesh
    path = os.path.join(output_dir, f'{prefix}_morphing.png')
    fig = plot_morphing_mesh(
        result.snapshots, result.endpoint_data,
        margin=result.config.margin, save_path=path,
    )
    plt.close(fig)
    saved.append(path)

    # 3. Convergence
    path = os.path.join(output_dir, f'{prefix}_convergence.png')
    fig = plot_convergence(result, save_path=path)
    plt.close(fig)
    saved.append(path)

    # 4. Det evolution
    path = os.path.join(output_dir, f'{prefix}_det_evolution.png')
    fig = plot_det_evolution(result, save_path=path)
    plt.close(fig)
    saved.append(path)

    # 5. Endpoint comparison (only if trajectory reached near t=1)
    if result.snapshots[-1].t > 0.9:
        path = os.path.join(output_dir, f'{prefix}_endpoints.png')
        fig = plot_endpoint_comparison(result, save_path=path)
        plt.close(fig)
        saved.append(path)

    # 6. Metric fields at a few snapshots
    n = len(result.snapshots)
    snap_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1] if n > 4 else range(n)
    for idx in snap_indices:
        snap = result.snapshots[idx]
        path = os.path.join(output_dir, f'{prefix}_metric_t{snap.t:.3f}.png')
        fig = plot_metric_fields(snap.eps, snap.phi, snap.gam, snap.t, save_path=path)
        plt.close(fig)
        saved.append(path)

    return saved
