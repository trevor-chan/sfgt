"""
visualization.py — Surface reconstruction and visualization for neural metric flows.

Adapted from the finite_differencing implementation.
Reconstructs flat surfaces from first fundamental form (E, F, G) using
parallel transport of the Jacobian frame.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import torch
from typing import Optional, List, Tuple

import os
import sys


# =============================================================================
# Basic geometry helpers
# =============================================================================

def metric_det(E, F, G):
    """Determinant of the metric tensor: det(a) = EG - F²."""
    return E * G - F**2


def square_to_circle(x, y):
    """
    Elliptic grid mapping from square [-1,1]² to unit disc.
    
        u = x * sqrt(1 - y²/2)
        v = y * sqrt(1 - x²/2)
    """
    u = x * np.sqrt(1.0 - 0.5 * y**2)
    v = y * np.sqrt(1.0 - 0.5 * x**2)
    return u, v


# =============================================================================
# Surface reconstruction from first fundamental form
# =============================================================================

def _fd_gradient(f, h, axis):
    """
    4th-order central finite differences in interior, 2nd-order at boundaries.
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


def _compute_christoffel_symbols(E, F, G, du, dv):
    """
    Compute all 6 Christoffel symbols Γⁱⱼₖ from the metric and its derivatives.
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


def reconstruct_surface(E, F, G, du, dv):
    """
    Reconstruct a flat surface embedding r(u,v) = (x,y) from its first
    fundamental form a = [[E, F], [F, G]].

    For a flat surface (K_Gauss = 0), the embedding Jacobian J = [e₁|e₂]
    satisfies JᵀJ = a and the parallel-transport equation.

    Parameters
    ----------
    E, F, G : ndarray (K, K)
        First fundamental form components.
    du, dv : float
        Grid spacing.

    Returns
    -------
    X, Y : ndarray (K, K)
        Reconstructed surface coordinates.
    """
    K = E.shape[0]

    # Christoffel symbols
    Gamma = _compute_christoffel_symbols(E, F, G, du, dv)

    det = E * G - F**2
    det = np.maximum(det, 1e-30)
    sqrt_E = np.sqrt(np.maximum(E, 1e-30))
    sqrt_det = np.sqrt(det)

    # Jacobian field J[i,j] is a 2×2 matrix
    J = np.zeros((K, K, 2, 2))

    # Initialize at (0,0) via Cholesky
    J[0, 0, 0, 0] = sqrt_E[0, 0]
    J[0, 0, 0, 1] = F[0, 0] / sqrt_E[0, 0]
    J[0, 0, 1, 0] = 0.0
    J[0, 0, 1, 1] = sqrt_det[0, 0] / sqrt_E[0, 0]

    # Propagation matrices
    M_u = np.zeros((K, K, 2, 2))
    M_u[:, :, 0, 0] = Gamma['G111']
    M_u[:, :, 0, 1] = Gamma['G112']
    M_u[:, :, 1, 0] = Gamma['G211']
    M_u[:, :, 1, 1] = Gamma['G212']

    M_v = np.zeros((K, K, 2, 2))
    M_v[:, :, 0, 0] = Gamma['G112']
    M_v[:, :, 0, 1] = Gamma['G122']
    M_v[:, :, 1, 0] = Gamma['G212']
    M_v[:, :, 1, 1] = Gamma['G222']

    # Propagate J along left edge (v-direction)
    for j in range(1, K):
        rhs_prev = J[0, j-1] @ M_v[0, j-1]
        J_pred = J[0, j-1] + dv * rhs_prev
        rhs_curr = J_pred @ M_v[0, j]
        J[0, j] = J[0, j-1] + 0.5 * dv * (rhs_prev + rhs_curr)

    # Propagate J across each row (u-direction)
    for j in range(K):
        for i in range(1, K):
            rhs_prev = J[i-1, j] @ M_u[i-1, j]
            J_pred = J[i-1, j] + du * rhs_prev
            rhs_curr = J_pred @ M_u[i, j]
            J[i, j] = J[i-1, j] + 0.5 * du * (rhs_prev + rhs_curr)

    # Reconstruct positions via Poisson solve
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    # Path-integrate boundary positions
    X_bnd = np.full((K, K), np.nan)
    Y_bnd = np.full((K, K), np.nan)

    X_bnd[0, 0] = 0.0
    Y_bnd[0, 0] = 0.0

    # Left edge
    for j in range(1, K):
        X_bnd[0, j] = X_bnd[0, j-1] + 0.5 * dv * (J[0, j-1, 0, 1] + J[0, j, 0, 1])
        Y_bnd[0, j] = Y_bnd[0, j-1] + 0.5 * dv * (J[0, j-1, 1, 1] + J[0, j, 1, 1])

    # Bottom edge
    for i in range(1, K):
        X_bnd[i, 0] = X_bnd[i-1, 0] + 0.5 * du * (J[i-1, 0, 0, 0] + J[i, 0, 0, 0])
        Y_bnd[i, 0] = Y_bnd[i-1, 0] + 0.5 * du * (J[i-1, 0, 1, 0] + J[i, 0, 1, 0])

    # Top edge
    for i in range(1, K):
        X_bnd[i, -1] = X_bnd[i-1, -1] + 0.5 * du * (J[i-1, -1, 0, 0] + J[i, -1, 0, 0])
        Y_bnd[i, -1] = Y_bnd[i-1, -1] + 0.5 * du * (J[i-1, -1, 1, 0] + J[i, -1, 1, 0])

    # Right edge
    for j in range(1, K):
        X_bnd[-1, j] = X_bnd[-1, j-1] + 0.5 * dv * (J[-1, j-1, 0, 1] + J[-1, j, 0, 1])
        Y_bnd[-1, j] = Y_bnd[-1, j-1] + 0.5 * dv * (J[-1, j-1, 1, 1] + J[-1, j, 1, 1])

    # Build discrete Laplacian
    N = K * K
    rows, cols, vals = [], [], []
    rhs_X = np.zeros(N)
    rhs_Y = np.zeros(N)

    def idx(i, j):
        return i * K + j

    div_X = np.gradient(J[:, :, 0, 0], du, axis=0) + np.gradient(J[:, :, 0, 1], dv, axis=1)
    div_Y = np.gradient(J[:, :, 1, 0], du, axis=0) + np.gradient(J[:, :, 1, 1], dv, axis=1)

    for i in range(K):
        for j in range(K):
            n = idx(i, j)
            is_bnd = (i == 0 or i == K-1 or j == 0 or j == K-1)

            if is_bnd:
                rows.append(n); cols.append(n); vals.append(1.0)
                rhs_X[n] = X_bnd[i, j]
                rhs_Y[n] = Y_bnd[i, j]
            else:
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

    X = spla.spsolve(L_mat, rhs_X).reshape(K, K)
    Y = spla.spsolve(L_mat, rhs_Y).reshape(K, K)

    X -= X.mean()
    Y -= Y.mean()

    return X, Y


# =============================================================================
# Mesh drawing helpers
# =============================================================================

def _draw_mesh(ax, X, Y, color='k', linewidth=0.5, alpha=0.7):
    """Draw a 2D mesh from coordinate arrays X, Y."""
    K = X.shape[0]
    for j in range(K):
        ax.plot(X[:, j], Y[:, j], color=color, linewidth=linewidth, alpha=alpha)
    for i in range(K):
        ax.plot(X[i, :], Y[i, :], color=color, linewidth=linewidth, alpha=alpha)


def _draw_colored_mesh(ax, X, Y, field, cmap='viridis', linewidth=0.5):
    """Draw mesh with line color determined by a scalar field."""
    K = X.shape[0]
    norm = mcolors.Normalize(vmin=np.nanmin(field), vmax=np.nanmax(field))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for j in range(K):
        points = np.column_stack([X[:, j], Y[:, j]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vals = 0.5 * (field[:-1, j] + field[1:, j])
        lc = LineCollection(segments, colors=sm.to_rgba(vals), linewidth=linewidth)
        ax.add_collection(lc)

    for i in range(K):
        points = np.column_stack([X[i, :], Y[i, :]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vals = 0.5 * (field[i, :-1] + field[i, 1:])
        lc = LineCollection(segments, colors=sm.to_rgba(vals), linewidth=linewidth)
        ax.add_collection(lc)

    return sm


# =============================================================================
# Model evaluation helpers
# =============================================================================

def evaluate_model_at_time(model, t_val, K_grid=25, margin=0.15, device='cpu'):
    """
    Evaluate the neural metric model at a specific time t.
    
    Returns numpy arrays for E, F, G and the uv grid.
    """
    lo = -1.0 + margin
    hi = 1.0 - margin
    uv_1d = torch.linspace(lo, hi, K_grid, device=device)
    U, V = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    uv_flat = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)
    
    N = uv_flat.shape[0]
    t_col = torch.full((N, 1), t_val, device=device)
    tuv = torch.cat([t_col, uv_flat], dim=1)
    
    model.eval()
    with torch.no_grad():
        E, F, G = model(tuv)
    
    E = E.reshape(K_grid, K_grid).cpu().numpy()
    F = F.reshape(K_grid, K_grid).cpu().numpy()
    G = G.reshape(K_grid, K_grid).cpu().numpy()
    
    U_np = U.cpu().numpy()
    V_np = V.cpu().numpy()
    
    return E, F, G, U_np, V_np


# =============================================================================
# Main visualization functions
# =============================================================================

def plot_morphing_trajectory(
    model,
    times: List[float] = None,
    K_grid: int = 25,
    margin: float = 0.15,
    color_by_det: bool = True,
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Visualize the surface morphing trajectory at selected timepoints.
    
    Parameters
    ----------
    model : FundamentalFormNet
        Trained neural metric model.
    times : list of float
        Timepoints to visualize (default: [0, 0.25, 0.5, 0.75, 1.0])
    K_grid : int
        Grid resolution for evaluation.
    margin : float
        Domain margin.
    color_by_det : bool
        If True, color mesh lines by metric determinant.
    save_path : str, optional
        Path to save the figure.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    n_panels = len(times)
    if figsize is None:
        figsize = (3.5 * n_panels, 4)
    
    L = (1.0 - margin) - (-1.0 + margin)
    du = L / (K_grid - 1)
    
    # Evaluate model and reconstruct surfaces at each time
    surfaces = []
    metrics = []
    for t_val in times:
        E, F, G, U, V = evaluate_model_at_time(model, t_val, K_grid, margin, device)
        X, Y = reconstruct_surface(E, F, G, du, du)
        surfaces.append((X, Y))
        metrics.append((E, F, G))
    
    # Compute global axis limits
    all_x = [X.min() for X, Y in surfaces] + [X.max() for X, Y in surfaces]
    all_y = [Y.min() for X, Y in surfaces] + [Y.max() for X, Y in surfaces]
    pad = 0.05
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)
    
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]
    
    for ax, t_val, (X, Y), (E, F, G) in zip(axes, times, surfaces, metrics):
        det = metric_det(E, F, G)
        
        if color_by_det:
            sm = _draw_colored_mesh(ax, X, Y, det, cmap='viridis', linewidth=0.6)
        else:
            _draw_mesh(ax, X, Y, color='steelblue', linewidth=0.6)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f't = {t_val:.2f}', fontsize=11)
        ax.tick_params(labelsize=8)
    
    fig.suptitle('Neural Metric Flow: Surface Morphing', fontsize=13, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_endpoint_comparison(
    model,
    K_grid: int = 25,
    margin: float = 0.15,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Compare reconstructed endpoints with known square and circle.
    """
    L = (1.0 - margin) - (-1.0 + margin)
    du = L / (K_grid - 1)
    lo = -1.0 + margin
    hi = 1.0 - margin
    
    # Ground truth surfaces
    u_1d = np.linspace(lo, hi, K_grid)
    U, V = np.meshgrid(u_1d, u_1d, indexing='ij')
    
    X_sq = U - U.mean()
    Y_sq = V - V.mean()
    
    X_circ, Y_circ = square_to_circle(U, V)
    X_circ -= X_circ.mean()
    Y_circ -= Y_circ.mean()
    
    # Reconstructed surfaces from model
    E0, F0, G0, _, _ = evaluate_model_at_time(model, 0.0, K_grid, margin, device)
    X0, Y0 = reconstruct_surface(E0, F0, G0, du, du)
    
    E1, F1, G1, _, _ = evaluate_model_at_time(model, 1.0, K_grid, margin, device)
    X1, Y1 = reconstruct_surface(E1, F1, G1, du, du)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # t=0: compare with square
    ax = axes[0]
    _draw_mesh(ax, X_sq, Y_sq, color='gray', linewidth=0.4, alpha=0.5)
    _draw_mesh(ax, X0, Y0, color='steelblue', linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_title('t = 0: Model (blue) vs Square (gray)', fontsize=10)
    ax.tick_params(labelsize=8)
    
    # t=1: compare with circle
    ax = axes[1]
    _draw_mesh(ax, X_circ, Y_circ, color='gray', linewidth=0.4, alpha=0.5)
    _draw_mesh(ax, X1, Y1, color='orangered', linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_title('t = 1: Model (red) vs Circle (gray)', fontsize=10)
    ax.tick_params(labelsize=8)
    
    fig.suptitle('Endpoint Comparison: Neural Metric vs Ground Truth', fontsize=12, y=1.02)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_metric_evolution(
    model,
    times: List[float] = None,
    K_grid: int = 25,
    margin: float = 0.15,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot the metric components E, F, G at multiple timepoints.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    n_t = len(times)
    fig, axes = plt.subplots(4, n_t, figsize=(3 * n_t, 10))
    
    for col, t_val in enumerate(times):
        E, F, G, U, V = evaluate_model_at_time(model, t_val, K_grid, margin, device)
        det = metric_det(E, F, G)
        
        for row, (field, name, cmap) in enumerate([
            (E, 'E', 'viridis'),
            (F, 'F', 'RdBu_r'),
            (G, 'G', 'viridis'),
            (det, 'det(a)', 'YlOrRd'),
        ]):
            ax = axes[row, col]
            if name == 'F':
                vabs = max(abs(field.min()), abs(field.max()))
                im = ax.imshow(field.T, origin='lower', cmap=cmap, vmin=-vabs, vmax=vabs)
            else:
                im = ax.imshow(field.T, origin='lower', cmap=cmap)
            
            plt.colorbar(im, ax=ax, fraction=0.046)
            if row == 0:
                ax.set_title(f't = {t_val:.2f}', fontsize=10)
            if col == 0:
                ax.set_ylabel(name, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.suptitle('Metric Components Evolution', fontsize=13, y=1.01)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Main script: visualize a trained model
# =============================================================================

if __name__ == '__main__':
    from model import FundamentalFormNet
    from train_flat_trajectory import metric_square, metric_circle
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load trained model
    model_path = 'results_flat_trajectory/model.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run train_flat_trajectory.py first.")
        sys.exit(1)
    
    model = FundamentalFormNet(
        mode='2d',
        hidden_dim=128,
        n_layers=4,
        activation='silu',
        endpoint_a_o=metric_square,
        endpoint_a_f=metric_circle,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    output_dir = 'results_flat_trajectory'
    
    # Generate visualizations
    print("\nGenerating surface morphing visualization...")
    fig = plot_morphing_trajectory(
        model,
        times=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        K_grid=30,
        margin=0.05,
        color_by_det=True,
        save_path=os.path.join(output_dir, 'surface_morphing.png'),
        device=device,
    )
    plt.close(fig)
    
    print("Generating endpoint comparison...")
    fig = plot_endpoint_comparison(
        model,
        K_grid=30,
        margin=0.05,
        save_path=os.path.join(output_dir, 'endpoint_comparison.png'),
        device=device,
    )
    plt.close(fig)
    
    print("Generating metric evolution...")
    fig = plot_metric_evolution(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=30,
        margin=0.05,
        save_path=os.path.join(output_dir, 'metric_evolution.png'),
        device=device,
    )
    plt.close(fig)
    
    print("\nDone! Visualizations saved to", output_dir)
