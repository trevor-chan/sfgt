"""
visualization_3d.py — 3D surface reconstruction and visualization.

Reconstructs surfaces from both fundamental forms (E, F, G, L, M, N)
using Gauss-Weingarten frame integration.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import torch
from typing import Optional, List, Tuple
import os


# =============================================================================
# Christoffel symbols from first fundamental form
# =============================================================================

def _fd_gradient(f, h, axis):
    """4th-order central differences in interior, 2nd-order at boundaries."""
    if axis == 1:
        return _fd_gradient(f.T, h, 0).T

    g = np.zeros_like(f)
    K = f.shape[0]

    if K > 4:
        g[2:-2] = (-f[4:] + 8*f[3:-1] - 8*f[1:-3] + f[:-4]) / (12*h)
    if K > 2:
        g[1] = (f[2] - f[0]) / (2*h)
        g[-2] = (f[-1] - f[-3]) / (2*h)
    if K > 2:
        g[0] = (-3*f[0] + 4*f[1] - f[2]) / (2*h)
        g[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2*h)
    elif K == 2:
        g[0] = (f[1] - f[0]) / h
        g[1] = (f[1] - f[0]) / h

    return g


def compute_christoffel_symbols(E, F, G, du, dv):
    """Compute Christoffel symbols from first fundamental form."""
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
# 3D Surface reconstruction using Gauss-Weingarten equations
# =============================================================================

def reconstruct_surface_3d(E, F, G, L, M, N, du, dv):
    """
    Reconstruct a 3D surface embedding r(u,v) = (x, y, z) from both
    fundamental forms using Gauss-Weingarten frame integration.
    
    The Gauss equations evolve the tangent frame:
        ∂r_u/∂u = Γ¹₁₁ r_u + Γ²₁₁ r_v + L·n
        ∂r_u/∂v = Γ¹₁₂ r_u + Γ²₁₂ r_v + M·n
        ∂r_v/∂v = Γ¹₂₂ r_u + Γ²₂₂ r_v + N·n
    
    Parameters
    ----------
    E, F, G : ndarray (K, K)
        First fundamental form coefficients.
    L, M, N : ndarray (K, K)
        Second fundamental form coefficients.
    du, dv : float
        Grid spacing.
    
    Returns
    -------
    X, Y, Z : ndarray (K, K)
        Surface coordinates in 3D.
    """
    K = E.shape[0]
    
    # Christoffel symbols
    Gamma = compute_christoffel_symbols(E, F, G, du, dv)
    
    # Initialize frame at center of grid
    # Position r, tangent vectors r_u, r_v (each 3D vectors at each grid point)
    r = np.zeros((K, K, 3))      # position
    r_u = np.zeros((K, K, 3))    # ∂r/∂u
    r_v = np.zeros((K, K, 3))    # ∂r/∂v
    
    # Initialize at (0, 0) using Cholesky-like decomposition
    # r_u · r_u = E, r_u · r_v = F, r_v · r_v = G
    sqrt_E = np.sqrt(np.maximum(E[0, 0], 1e-10))
    det = E[0, 0] * G[0, 0] - F[0, 0]**2
    sqrt_det_over_E = np.sqrt(np.maximum(det, 1e-10)) / sqrt_E
    
    # Initial frame in xy-plane (z=0)
    # As we integrate, curvature terms (L, M, N) will lift it into 3D
    r[0, 0] = np.array([0.0, 0.0, 0.0])
    r_u[0, 0] = np.array([sqrt_E, 0.0, 0.0])
    r_v[0, 0] = np.array([F[0, 0] / sqrt_E, sqrt_det_over_E, 0.0])
    
    def compute_normal(ru, rv):
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
    X -= X.mean()
    Y -= Y.mean()
    Z -= Z.mean()
    
    return X, Y, Z


# =============================================================================
# Ground truth hemisphere for comparison
# =============================================================================

def hemisphere_ground_truth(K, margin=0.3):
    """Generate ground truth hemisphere surface."""
    lo = -1.0 + margin
    hi = 1.0 - margin
    u = np.linspace(lo, hi, K)
    v = np.linspace(lo, hi, K)
    U, V = np.meshgrid(u, v, indexing='ij')
    
    # z = sqrt(1 - u² - v²)
    r2 = U**2 + V**2
    Z = np.sqrt(np.maximum(1.0 - r2, 1e-6))
    
    X = U - U.mean()
    Y = V - V.mean()
    Z = Z - Z.mean()
    
    return X, Y, Z


# =============================================================================
# 3D mesh drawing
# =============================================================================

def draw_mesh_3d(ax, X, Y, Z, color='steelblue', linewidth=0.5, alpha=0.7):
    """Draw a 3D wireframe mesh."""
    K = X.shape[0]
    
    # u-lines (constant v)
    for j in range(K):
        ax.plot(X[:, j], Y[:, j], Z[:, j], color=color, linewidth=linewidth, alpha=alpha)
    
    # v-lines (constant u)
    for i in range(K):
        ax.plot(X[i, :], Y[i, :], Z[i, :], color=color, linewidth=linewidth, alpha=alpha)


def draw_colored_mesh_3d(ax, X, Y, Z, field, cmap='viridis', linewidth=0.5):
    """Draw 3D mesh with lines colored by a scalar field."""
    K = X.shape[0]
    norm = mcolors.Normalize(vmin=np.nanmin(field), vmax=np.nanmax(field))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # u-lines
    for j in range(K):
        for i in range(K - 1):
            color = sm.to_rgba(0.5 * (field[i, j] + field[i+1, j]))
            ax.plot([X[i, j], X[i+1, j]], 
                   [Y[i, j], Y[i+1, j]], 
                   [Z[i, j], Z[i+1, j]], 
                   color=color, linewidth=linewidth)
    
    # v-lines
    for i in range(K):
        for j in range(K - 1):
            color = sm.to_rgba(0.5 * (field[i, j] + field[i, j+1]))
            ax.plot([X[i, j], X[i, j+1]], 
                   [Y[i, j], Y[i, j+1]], 
                   [Z[i, j], Z[i, j+1]], 
                   color=color, linewidth=linewidth)
    
    return sm


# =============================================================================
# Model evaluation
# =============================================================================

def evaluate_model_3d_at_time(model, t_val, K_grid=20, margin=0.3, device='cpu'):
    """Evaluate a 3D neural metric model at a specific time t."""
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
        E, F, G, L, M, N_coef = model(tuv)
    
    E = E.reshape(K_grid, K_grid).cpu().numpy()
    F = F.reshape(K_grid, K_grid).cpu().numpy()
    G = G.reshape(K_grid, K_grid).cpu().numpy()
    L = L.reshape(K_grid, K_grid).cpu().numpy()
    M = M.reshape(K_grid, K_grid).cpu().numpy()
    N_coef = N_coef.reshape(K_grid, K_grid).cpu().numpy()
    
    return E, F, G, L, M, N_coef


# =============================================================================
# Visualization functions
# =============================================================================

def plot_3d_morphing(
    model,
    times: List[float] = None,
    K_grid: int = 20,
    margin: float = 0.3,
    color_by_curvature: bool = True,
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Visualize 3D surface morphing at multiple timepoints.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    n_panels = len(times)
    if figsize is None:
        figsize = (4 * n_panels, 4)
    
    L_domain = (1.0 - margin) - (-1.0 + margin)
    du = L_domain / (K_grid - 1)
    
    # Evaluate and reconstruct at each time
    surfaces = []
    curvatures = []
    
    for t_val in times:
        E, F, G, L, M, N_coef = evaluate_model_3d_at_time(
            model, t_val, K_grid, margin, device
        )
        X, Y, Z = reconstruct_surface_3d(E, F, G, L, M, N_coef, du, du)
        surfaces.append((X, Y, Z))
        
        # Gaussian curvature for coloring
        det_a = E * G - F**2
        K_gauss = (L * N_coef - M**2) / (det_a + 1e-12)
        curvatures.append(K_gauss)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Compute global axis limits
    all_coords = []
    for X, Y, Z in surfaces:
        all_coords.extend([X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()])
    coord_range = max(all_coords) - min(all_coords)
    
    for idx, (t_val, (X, Y, Z), K_gauss) in enumerate(zip(times, surfaces, curvatures)):
        ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')
        
        if color_by_curvature:
            draw_colored_mesh_3d(ax, X, Y, Z, K_gauss, cmap='coolwarm', linewidth=0.5)
        else:
            draw_mesh_3d(ax, X, Y, Z, color='steelblue', linewidth=0.5)
        
        ax.set_title(f't = {t_val:.2f}', fontsize=11)
        
        # Set equal aspect ratio
        max_range = coord_range / 2
        mid_x, mid_y, mid_z = X.mean(), Y.mean(), Z.mean()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    fig.suptitle('3D Surface Morphing (Sheet → Hemisphere)', fontsize=13, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_3d_endpoint_comparison(
    model,
    K_grid: int = 20,
    margin: float = 0.3,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Compare reconstructed t=1 surface with ground truth hemisphere.
    """
    L_domain = (1.0 - margin) - (-1.0 + margin)
    du = L_domain / (K_grid - 1)
    
    # Ground truth
    X_gt, Y_gt, Z_gt = hemisphere_ground_truth(K_grid, margin)
    
    # Reconstructed from model at t=1
    E, F, G, L, M, N_coef = evaluate_model_3d_at_time(model, 1.0, K_grid, margin, device)
    X_rec, Y_rec, Z_rec = reconstruct_surface_3d(E, F, G, L, M, N_coef, du, du)
    
    # Also get t=0 (flat sheet)
    E0, F0, G0, L0, M0, N0 = evaluate_model_3d_at_time(model, 0.0, K_grid, margin, device)
    X0, Y0, Z0 = reconstruct_surface_3d(E0, F0, G0, L0, M0, N0, du, du)
    
    fig = plt.figure(figsize=figsize)
    
    # t=0: flat sheet
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    draw_mesh_3d(ax1, X0, Y0, Z0, color='steelblue', linewidth=0.6)
    ax1.set_title('t = 0 (Flat Sheet)', fontsize=10)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # t=1: reconstructed
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    draw_mesh_3d(ax2, X_rec, Y_rec, Z_rec, color='orangered', linewidth=0.6)
    ax2.set_title('t = 1 (Reconstructed)', fontsize=10)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    # Ground truth hemisphere
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    draw_mesh_3d(ax3, X_gt, Y_gt, Z_gt, color='green', linewidth=0.6)
    ax3.set_title('Ground Truth Hemisphere', fontsize=10)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    
    # Set consistent axis limits
    all_X = np.concatenate([X0.ravel(), X_rec.ravel(), X_gt.ravel()])
    all_Y = np.concatenate([Y0.ravel(), Y_rec.ravel(), Y_gt.ravel()])
    all_Z = np.concatenate([Z0.ravel(), Z_rec.ravel(), Z_gt.ravel()])
    max_range = max(all_X.max() - all_X.min(), 
                    all_Y.max() - all_Y.min(),
                    all_Z.max() - all_Z.min()) / 2
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
    
    fig.suptitle('Endpoint Comparison: Flat Sheet → Hemisphere', fontsize=12, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_curvature_colored_surface(
    model,
    t_val: float = 1.0,
    K_grid: int = 25,
    margin: float = 0.3,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot a single surface colored by Gaussian and mean curvature.
    """
    L_domain = (1.0 - margin) - (-1.0 + margin)
    du = L_domain / (K_grid - 1)
    
    E, F, G, L, M, N_coef = evaluate_model_3d_at_time(model, t_val, K_grid, margin, device)
    X, Y, Z = reconstruct_surface_3d(E, F, G, L, M, N_coef, du, du)
    
    det_a = E * G - F**2
    K_gauss = (L * N_coef - M**2) / (det_a + 1e-12)
    H_mean = (E * N_coef + G * L - 2 * F * M) / (2 * det_a + 1e-12)
    
    fig = plt.figure(figsize=figsize)
    
    # Gaussian curvature
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sm1 = draw_colored_mesh_3d(ax1, X, Y, Z, K_gauss, cmap='coolwarm', linewidth=0.6)
    ax1.set_title(f'Gaussian Curvature K\nt = {t_val:.2f}', fontsize=10)
    fig.colorbar(sm1, ax=ax1, shrink=0.5, label='K')
    
    # Mean curvature
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sm2 = draw_colored_mesh_3d(ax2, X, Y, Z, H_mean, cmap='coolwarm', linewidth=0.6)
    ax2.set_title(f'Mean Curvature H\nt = {t_val:.2f}', fontsize=10)
    fig.colorbar(sm2, ax=ax2, shrink=0.5, label='H')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    
    from model import FundamentalFormNet
    from train_hemisphere import metric_flat_sheet, metric_hemisphere
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model_path = 'results_hemisphere/model.pt'
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Run train_hemisphere.py first.")
        sys.exit(1)
    
    model = FundamentalFormNet(
        mode='3d',
        hidden_dim=128,
        n_layers=5,
        activation='silu',
        endpoint_a_o=metric_flat_sheet,
        endpoint_a_f=metric_hemisphere,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    output_dir = 'results_hemisphere'
    
    print("\nGenerating 3D morphing visualization...")
    fig = plot_3d_morphing(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=20,
        margin=0.5,
        color_by_curvature=True,
        save_path=os.path.join(output_dir, 'surface_morphing_3d.png'),
        device=device,
    )
    plt.close(fig)
    
    print("Generating endpoint comparison...")
    fig = plot_3d_endpoint_comparison(
        model,
        K_grid=20,
        margin=0.5,
        save_path=os.path.join(output_dir, 'endpoint_comparison_3d.png'),
        device=device,
    )
    plt.close(fig)
    
    print("Generating curvature-colored surface...")
    fig = plot_curvature_colored_surface(
        model,
        t_val=1.0,
        K_grid=25,
        margin=0.5,
        save_path=os.path.join(output_dir, 'curvature_surface_3d.png'),
        device=device,
    )
    plt.close(fig)
    
    print(f"\nDone! 3D visualizations saved to {output_dir}/")
