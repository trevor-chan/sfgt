"""
visualization.py — Unified visualization for neural metric flows.

Provides plotting functions for both 2D (flat) and 3D surface trajectories.
Uses reconstruction.py for all surface embedding recovery.

Plot types:
- Training curves
- Surface morphing trajectories (2D and 3D)
- Endpoint comparisons
- Metric component evolution (E, F, G, L, M, N)
- Curvature field evolution (K, H)
- Curvature-colored surfaces
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import torch
from typing import Optional, List, Tuple, Dict

from .reconstruction import reconstruct_surface, enforce_periodic_closure


# =============================================================================
# Model evaluation
# =============================================================================

def evaluate_model_at_time(
    model,
    t_val: float,
    K_grid: int = 25,
    margin: float = 0.15,
    device: str = 'cpu',
) -> Dict[str, np.ndarray]:
    """
    Evaluate neural metric model at a specific time.

    Parameters
    ----------
    model : FundamentalFormNet
        Trained model.
    t_val : float
        Time in [0, 1].
    K_grid : int
        Grid resolution.
    margin : float
        Domain margin.
    device : str
        Torch device.

    Returns
    -------
    result : dict
        Contains E, F, G, L, M, N, K, H, det, U, V arrays.
    """
    lo = -1.0 + margin
    hi = 1.0 - margin
    uv_1d = torch.linspace(lo, hi, K_grid, device=device)
    U, V = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
    uv_flat = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)

    N_pts = uv_flat.shape[0]
    t_col = torch.full((N_pts, 1), t_val, device=device)
    tuv = torch.cat([t_col, uv_flat], dim=1)

    model.eval()
    E, F, G, L, M, N_coef = model(tuv)

    # Reshape to grid
    E = E.reshape(K_grid, K_grid).detach().cpu().numpy()
    F = F.reshape(K_grid, K_grid).detach().cpu().numpy()
    G = G.reshape(K_grid, K_grid).detach().cpu().numpy()
    L = L.reshape(K_grid, K_grid).detach().cpu().numpy()
    M = M.reshape(K_grid, K_grid).detach().cpu().numpy()
    N_coef = N_coef.reshape(K_grid, K_grid).detach().cpu().numpy()

    # Curvatures
    det = E * G - F**2
    K_gauss = (L * N_coef - M**2) / (det + 1e-12)
    H_mean = (E * N_coef + G * L - 2 * F * M) / (2 * det + 1e-12)

    return {
        'E': E, 'F': F, 'G': G,
        'L': L, 'M': M, 'N': N_coef,
        'K': K_gauss, 'H': H_mean, 'det': det,
        'U': U.cpu().numpy(), 'V': V.cpu().numpy(),
        't': t_val,
    }


# =============================================================================
# 2D mesh drawing
# =============================================================================

def draw_mesh_2d(ax, X, Y, color='steelblue', linewidth=0.5, alpha=0.7):
    """Draw a 2D mesh from coordinate arrays."""
    K_u, K_v = X.shape
    for j in range(K_v):
        ax.plot(X[:, j], Y[:, j], color=color, linewidth=linewidth, alpha=alpha)
    for i in range(K_u):
        ax.plot(X[i, :], Y[i, :], color=color, linewidth=linewidth, alpha=alpha)


def draw_colored_mesh_2d(ax, X, Y, field, cmap='viridis', linewidth=0.5):
    """Draw 2D mesh with lines colored by a scalar field."""
    norm = mcolors.Normalize(vmin=np.nanmin(field), vmax=np.nanmax(field))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    K_u, K_v = X.shape

    for j in range(K_v):
        points = np.column_stack([X[:, j], Y[:, j]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vals = 0.5 * (field[:-1, j] + field[1:, j])
        lc = LineCollection(segments, colors=sm.to_rgba(vals), linewidth=linewidth)
        ax.add_collection(lc)

    for i in range(K_u):
        points = np.column_stack([X[i, :], Y[i, :]]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vals = 0.5 * (field[i, :-1] + field[i, 1:])
        lc = LineCollection(segments, colors=sm.to_rgba(vals), linewidth=linewidth)
        ax.add_collection(lc)

    return sm


# =============================================================================
# 3D mesh drawing
# =============================================================================

def draw_mesh_3d(ax, X, Y, Z, color='steelblue', linewidth=0.5, alpha=0.7):
    """Draw a 3D wireframe mesh."""
    K_u, K_v = X.shape
    for j in range(K_v):
        ax.plot(X[:, j], Y[:, j], Z[:, j], color=color, linewidth=linewidth, alpha=alpha)
    for i in range(K_u):
        ax.plot(X[i, :], Y[i, :], Z[i, :], color=color, linewidth=linewidth, alpha=alpha)


def draw_colored_mesh_3d(ax, X, Y, Z, field, cmap='coolwarm', linewidth=0.5):
    """Draw 3D mesh with lines colored by a scalar field."""
    norm = mcolors.Normalize(vmin=np.nanmin(field), vmax=np.nanmax(field))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    K_u, K_v = X.shape

    for j in range(K_v):
        for i in range(K_u - 1):
            color = sm.to_rgba(0.5 * (field[i, j] + field[i+1, j]))
            ax.plot([X[i, j], X[i+1, j]],
                    [Y[i, j], Y[i+1, j]],
                    [Z[i, j], Z[i+1, j]],
                    color=color, linewidth=linewidth)

    for i in range(K_u):
        for j in range(K_v - 1):
            color = sm.to_rgba(0.5 * (field[i, j] + field[i, j+1]))
            ax.plot([X[i, j], X[i, j+1]],
                    [Y[i, j], Y[i, j+1]],
                    [Z[i, j], Z[i, j+1]],
                    color=color, linewidth=linewidth)

    return sm


# =============================================================================
# Ground truth surfaces
# =============================================================================

def square_to_circle(x, y):
    """Elliptic mapping from square [-1,1]² to unit disc."""
    u = x * np.sqrt(1.0 - 0.5 * y**2)
    v = y * np.sqrt(1.0 - 0.5 * x**2)
    return u, v


def hemisphere_ground_truth(K, margin=0.3):
    """Generate ground truth hemisphere surface."""
    lo = -1.0 + margin
    hi = 1.0 - margin
    u = np.linspace(lo, hi, K)
    v = np.linspace(lo, hi, K)
    U, V = np.meshgrid(u, v, indexing='ij')

    r2 = U**2 + V**2
    Z = np.sqrt(np.maximum(1.0 - r2, 1e-6))

    X = U - U.mean()
    Y = V - V.mean()
    Z = Z - Z.mean()

    return X, Y, Z


# =============================================================================
# Training curves
# =============================================================================

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (15, 4),
):
    """
    Plot training loss and curvature curves.

    Shows mean curvature on left y-axis (blue) and std on right y-axis (red).

    Parameters
    ----------
    history : dict
        Training history from MetricFlowTrainer.
    save_path : str, optional
        Path to save figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Loss
    axes[0].semilogy(history['loss_total'], label='Total', color='k')
    for key in ['compatibility', 'flatness', 'strain_rate', 'elastic']:
        if key in history and len(history[key]) > 0 and max(history[key]) > 0:
            axes[0].semilogy(history[key], label=key.capitalize(), alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()

    # Gaussian curvature
    if 'K_mean' in history:
        ax1_left = axes[1]
        ax1_left.plot(history['K_mean'], label='|K| mean', color='C0')
        ax1_left.set_xlabel('Step')
        ax1_left.set_ylabel('|K|', color='C0')
        ax1_left.tick_params(axis='y', labelcolor='C0')
        ax1_left.set_title('Gaussian Curvature')

        # Secondary axis for K std
        if 'K_std' in history and len(history['K_std']) > 0:
            ax1_right = ax1_left.twinx()
            ax1_right.plot(history['K_std'], label='|K| std', color='red', alpha=0.8, linestyle='--')
            ax1_right.set_ylabel('Std(K)', color='red')
            ax1_right.tick_params(axis='y', labelcolor='red')

            # Combined legend
            lines1, labels1 = ax1_left.get_legend_handles_labels()
            lines2, labels2 = ax1_right.get_legend_handles_labels()
            ax1_left.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax1_left.legend()

    # Mean curvature
    if 'H_mean' in history:
        ax2_left = axes[2]
        ax2_left.plot(history['H_mean'], label='|H| mean', color='C1')
        ax2_left.set_xlabel('Step')
        ax2_left.set_ylabel('|H|', color='C1')
        ax2_left.tick_params(axis='y', labelcolor='C1')
        ax2_left.set_title('Mean Curvature')

        # Secondary axis for H std
        if 'H_std' in history and len(history['H_std']) > 0:
            ax2_right = ax2_left.twinx()
            ax2_right.plot(history['H_std'], label='|H| std', color='red', alpha=0.8, linestyle='--')
            ax2_right.set_ylabel('Std(H)', color='red')
            ax2_right.tick_params(axis='y', labelcolor='red')

            # Combined legend
            lines1, labels1 = ax2_left.get_legend_handles_labels()
            lines2, labels2 = ax2_right.get_legend_handles_labels()
            ax2_left.legend(lines1 + lines2, labels1 + labels2, loc='best')
        else:
            ax2_left.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# 2D surface trajectory (flat surfaces)
# =============================================================================

def plot_surface_trajectory_2d(
    model,
    times: List[float] = None,
    K_grid: int = 30,
    margin: float = 0.15,
    color_by_det: bool = True,
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot 2D surface morphing trajectory (for flat surfaces).

    Parameters
    ----------
    model : FundamentalFormNet
        Trained model.
    times : list of float
        Timepoints to visualize.
    K_grid : int
        Grid resolution.
    margin : float
        Domain margin.
    color_by_det : bool
        Color mesh by metric determinant.
    save_path : str, optional
        Path to save figure.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_panels = len(times)
    if figsize is None:
        figsize = (3.5 * n_panels, 4)

    lo, hi = -1.0 + margin, 1.0 - margin
    du = (hi - lo) / (K_grid - 1)

    # Evaluate and reconstruct at each time
    surfaces = []
    metrics = []

    for t_val in times:
        result = evaluate_model_at_time(model, t_val, K_grid, margin, device)
        X, Y, Z = reconstruct_surface(
            result['E'], result['F'], result['G'],
            result['L'], result['M'], result['N'],
            du, du
        )
        surfaces.append((X, Y, Z))
        metrics.append(result)

    # Global axis limits
    all_x = [X.min() for X, Y, Z in surfaces] + [X.max() for X, Y, Z in surfaces]
    all_y = [Y.min() for X, Y, Z in surfaces] + [Y.max() for X, Y, Z in surfaces]
    pad = 0.05
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    for ax, t_val, (X, Y, Z), result in zip(axes, times, surfaces, metrics):
        if color_by_det:
            sm = draw_colored_mesh_2d(ax, X, Y, result['det'], cmap='viridis', linewidth=0.6)
        else:
            draw_mesh_2d(ax, X, Y, color='steelblue', linewidth=0.6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f't = {t_val:.2f}', fontsize=11)

    fig.suptitle('Surface Trajectory (2D)', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# 3D surface trajectory
# =============================================================================

def plot_surface_trajectory_3d(
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
    Plot 3D surface morphing trajectory.

    Parameters
    ----------
    model : FundamentalFormNet
        Trained model.
    times : list of float
        Timepoints to visualize.
    K_grid : int
        Grid resolution.
    margin : float
        Domain margin.
    color_by_curvature : bool
        Color mesh by Gaussian curvature.
    save_path : str, optional
        Path to save figure.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_panels = len(times)
    if figsize is None:
        figsize = (4 * n_panels, 4)

    lo, hi = -1.0 + margin, 1.0 - margin
    du = (hi - lo) / (K_grid - 1)

    # Evaluate and reconstruct
    surfaces = []
    results = []

    for t_val in times:
        result = evaluate_model_at_time(model, t_val, K_grid, margin, device)
        X, Y, Z = reconstruct_surface(
            result['E'], result['F'], result['G'],
            result['L'], result['M'], result['N'],
            du, du
        )
        surfaces.append((X, Y, Z))
        results.append(result)

    # Global axis limits
    all_coords = []
    for X, Y, Z in surfaces:
        all_coords.extend([X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()])
    coord_range = max(all_coords) - min(all_coords)

    fig = plt.figure(figsize=figsize)

    for idx, (t_val, (X, Y, Z), result) in enumerate(zip(times, surfaces, results)):
        ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')

        if color_by_curvature:
            draw_colored_mesh_3d(ax, X, Y, Z, result['K'], cmap='coolwarm', linewidth=0.5)
        else:
            draw_mesh_3d(ax, X, Y, Z, color='steelblue', linewidth=0.5)

        ax.set_title(f't = {t_val:.2f}', fontsize=11)

        max_range = coord_range / 2
        mid_x, mid_y, mid_z = X.mean(), Y.mean(), Z.mean()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig.suptitle('Surface Trajectory (3D)', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Endpoint comparison
# =============================================================================

def plot_endpoint_comparison_2d(
    model,
    K_grid: int = 30,
    margin: float = 0.15,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Compare reconstructed 2D endpoints with ground truth square and circle.
    """
    lo, hi = -1.0 + margin, 1.0 - margin
    du = (hi - lo) / (K_grid - 1)

    # Ground truth
    u_1d = np.linspace(lo, hi, K_grid)
    U, V = np.meshgrid(u_1d, u_1d, indexing='ij')
    X_sq = U - U.mean()
    Y_sq = V - V.mean()
    X_circ, Y_circ = square_to_circle(U, V)
    X_circ -= X_circ.mean()
    Y_circ -= Y_circ.mean()

    # Reconstructed
    result0 = evaluate_model_at_time(model, 0.0, K_grid, margin, device)
    X0, Y0, _ = reconstruct_surface(
        result0['E'], result0['F'], result0['G'],
        result0['L'], result0['M'], result0['N'], du, du
    )

    result1 = evaluate_model_at_time(model, 1.0, K_grid, margin, device)
    X1, Y1, _ = reconstruct_surface(
        result1['E'], result1['F'], result1['G'],
        result1['L'], result1['M'], result1['N'], du, du
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # t=0
    draw_mesh_2d(axes[0], X_sq, Y_sq, color='gray', linewidth=0.4, alpha=0.5)
    draw_mesh_2d(axes[0], X0, Y0, color='steelblue', linewidth=0.6)
    axes[0].set_aspect('equal')
    axes[0].set_title('t=0: Model (blue) vs Square (gray)')

    # t=1
    draw_mesh_2d(axes[1], X_circ, Y_circ, color='gray', linewidth=0.4, alpha=0.5)
    draw_mesh_2d(axes[1], X1, Y1, color='orangered', linewidth=0.6)
    axes[1].set_aspect('equal')
    axes[1].set_title('t=1: Model (red) vs Circle (gray)')

    fig.suptitle('Endpoint Comparison (2D)', fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_endpoint_comparison_3d(
    model,
    K_grid: int = 20,
    margin: float = 0.3,
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Compare reconstructed 3D endpoints: flat sheet → hemisphere.
    """
    lo, hi = -1.0 + margin, 1.0 - margin
    du = (hi - lo) / (K_grid - 1)

    # Ground truth hemisphere
    X_gt, Y_gt, Z_gt = hemisphere_ground_truth(K_grid, margin)

    # Reconstructed at t=0 (flat)
    result0 = evaluate_model_at_time(model, 0.0, K_grid, margin, device)
    X0, Y0, Z0 = reconstruct_surface(
        result0['E'], result0['F'], result0['G'],
        result0['L'], result0['M'], result0['N'], du, du
    )

    # Reconstructed at t=1 (hemisphere)
    result1 = evaluate_model_at_time(model, 1.0, K_grid, margin, device)
    X1, Y1, Z1 = reconstruct_surface(
        result1['E'], result1['F'], result1['G'],
        result1['L'], result1['M'], result1['N'], du, du
    )

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    draw_mesh_3d(ax1, X0, Y0, Z0, color='steelblue', linewidth=0.6)
    ax1.set_title('t=0 (Flat Sheet)')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    draw_mesh_3d(ax2, X1, Y1, Z1, color='orangered', linewidth=0.6)
    ax2.set_title('t=1 (Reconstructed)')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    draw_mesh_3d(ax3, X_gt, Y_gt, Z_gt, color='green', linewidth=0.6)
    ax3.set_title('Ground Truth Hemisphere')

    # Consistent limits
    all_X = np.concatenate([X0.ravel(), X1.ravel(), X_gt.ravel()])
    all_Y = np.concatenate([Y0.ravel(), Y1.ravel(), Y_gt.ravel()])
    all_Z = np.concatenate([Z0.ravel(), Z1.ravel(), Z_gt.ravel()])
    max_range = max(np.ptp(all_X), np.ptp(all_Y), np.ptp(all_Z)) / 2

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig.suptitle('Endpoint Comparison (3D)', fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Metric component evolution
# =============================================================================

def plot_metric_evolution(
    model,
    times: List[float] = None,
    K_grid: int = 25,
    margin: float = 0.15,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot first fundamental form (E, F, G) and determinant at multiple times.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_t = len(times)
    fig, axes = plt.subplots(4, n_t, figsize=(3 * n_t, 10))

    for col, t_val in enumerate(times):
        result = evaluate_model_at_time(model, t_val, K_grid, margin, device)

        for row, (field, name, cmap) in enumerate([
            (result['E'], 'E', 'viridis'),
            (result['F'], 'F', 'RdBu_r'),
            (result['G'], 'G', 'viridis'),
            (result['det'], 'det(a)', 'YlOrRd'),
        ]):
            ax = axes[row, col]
            if name == 'F':
                vabs = max(abs(field.min()), abs(field.max()), 0.01)
                im = ax.imshow(field.T, origin='lower', cmap=cmap, vmin=-vabs, vmax=vabs)
            else:
                im = ax.imshow(field.T, origin='lower', cmap=cmap)

            plt.colorbar(im, ax=ax, fraction=0.046)
            if row == 0:
                ax.set_title(f't = {t_val:.2f}')
            if col == 0:
                ax.set_ylabel(name)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('First Fundamental Form Evolution', fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_second_form_evolution(
    model,
    times: List[float] = None,
    K_grid: int = 25,
    margin: float = 0.3,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot second fundamental form (L, M, N) at multiple times.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_t = len(times)
    fig, axes = plt.subplots(3, n_t, figsize=(3 * n_t, 8))

    for col, t_val in enumerate(times):
        result = evaluate_model_at_time(model, t_val, K_grid, margin, device)

        for row, (field, name) in enumerate([
            (result['L'], 'L'),
            (result['M'], 'M'),
            (result['N'], 'N'),
        ]):
            ax = axes[row, col]
            vabs = max(abs(field.min()), abs(field.max()), 0.01)
            im = ax.imshow(field.T, origin='lower', cmap='RdBu_r', vmin=-vabs, vmax=vabs)

            plt.colorbar(im, ax=ax, fraction=0.046)
            if row == 0:
                ax.set_title(f't = {t_val:.2f}')
            if col == 0:
                ax.set_ylabel(name)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Second Fundamental Form Evolution', fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Curvature evolution
# =============================================================================

def plot_curvature_evolution(
    model,
    times: List[float] = None,
    K_grid: int = 25,
    margin: float = 0.3,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot Gaussian (K) and mean (H) curvature at multiple times.
    """
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    n_t = len(times)
    fig, axes = plt.subplots(2, n_t, figsize=(4 * n_t, 7))

    for col, t_val in enumerate(times):
        result = evaluate_model_at_time(model, t_val, K_grid, margin, device)

        # Gaussian curvature
        K = result['K']
        vmax = max(abs(K.min()), abs(K.max()), 0.1)
        im = axes[0, col].imshow(K.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(f"K, t={t_val:.2f}\n[{K.min():.2f}, {K.max():.2f}]")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)

        # Mean curvature
        H = result['H']
        vmax = max(abs(H.min()), abs(H.max()), 0.1)
        im = axes[1, col].imshow(H.T, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1, col].set_title(f"H, t={t_val:.2f}\n[{H.min():.2f}, {H.max():.2f}]")
        plt.colorbar(im, ax=axes[1, col], fraction=0.046)

    axes[0, 0].set_ylabel('Gaussian K')
    axes[1, 0].set_ylabel('Mean H')
    fig.suptitle('Curvature Evolution', fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_curvature_summary(
    model,
    n_t: int = 11,
    K_grid: int = 20,
    margin: float = 0.3,
    target_K: float = None,
    target_H: float = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot curvature statistics vs time.

    Shows mean curvature on left y-axis (blue) and std on right y-axis (red).
    """
    times = np.linspace(0, 1, n_t)
    results = [evaluate_model_at_time(model, t, K_grid, margin, device) for t in times]

    K_mean = [np.mean(r['K']) for r in results]
    K_std = [np.std(r['K']) for r in results]
    K_min = [np.min(r['K']) for r in results]
    K_max = [np.max(r['K']) for r in results]
    H_mean = [np.mean(r['H']) for r in results]
    H_std = [np.std(r['H']) for r in results]
    H_min = [np.min(r['H']) for r in results]
    H_max = [np.max(r['H']) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gaussian curvature K
    ax1_left = axes[0]
    ax1_left.plot(times, K_mean, 'o-', label='K mean', color='C0')
    ax1_left.fill_between(times, K_min, K_max, alpha=0.2, color='C0')
    ax1_left.axhline(0, color='gray', linestyle='--', alpha=0.5)
    if target_K is not None:
        ax1_left.axhline(target_K, color='green', linestyle='--', alpha=0.5, label=f'Target K={target_K}')
    ax1_left.set_xlabel('t')
    ax1_left.set_ylabel('Gaussian Curvature K', color='C0')
    ax1_left.tick_params(axis='y', labelcolor='C0')
    ax1_left.set_title('K vs Time')

    # Secondary axis for K std
    ax1_right = ax1_left.twinx()
    ax1_right.plot(times, K_std, 's--', label='K std', color='red', alpha=0.8)
    ax1_right.set_ylabel('Std(K)', color='red')
    ax1_right.tick_params(axis='y', labelcolor='red')

    # Combined legend
    lines1, labels1 = ax1_left.get_legend_handles_labels()
    lines2, labels2 = ax1_right.get_legend_handles_labels()
    ax1_left.legend(lines1 + lines2, labels1 + labels2, loc='best')

    # Mean curvature H
    ax2_left = axes[1]
    ax2_left.plot(times, H_mean, 'o-', label='H mean', color='C1')
    ax2_left.fill_between(times, H_min, H_max, alpha=0.2, color='C1')
    ax2_left.axhline(0, color='gray', linestyle='--', alpha=0.5)
    if target_H is not None:
        ax2_left.axhline(target_H, color='green', linestyle='--', alpha=0.5, label=f'Target H={target_H}')
    ax2_left.set_xlabel('t')
    ax2_left.set_ylabel('Mean Curvature H', color='C1')
    ax2_left.tick_params(axis='y', labelcolor='C1')
    ax2_left.set_title('H vs Time')

    # Secondary axis for H std
    ax2_right = ax2_left.twinx()
    ax2_right.plot(times, H_std, 's--', label='H std', color='red', alpha=0.8)
    ax2_right.set_ylabel('Std(H)', color='red')
    ax2_right.tick_params(axis='y', labelcolor='red')

    # Combined legend
    lines1, labels1 = ax2_left.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2_left.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Curvature-colored 3D surface
# =============================================================================

def plot_curvature_surface_3d(
    model,
    t_val: float = 1.0,
    K_grid: int = 25,
    margin: float = 0.3,
    figsize: Tuple[float, float] = (10, 8),
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """
    Plot single 3D surface colored by Gaussian and mean curvature.
    """
    lo, hi = -1.0 + margin, 1.0 - margin
    du = (hi - lo) / (K_grid - 1)

    result = evaluate_model_at_time(model, t_val, K_grid, margin, device)
    X, Y, Z = reconstruct_surface(
        result['E'], result['F'], result['G'],
        result['L'], result['M'], result['N'], du, du
    )

    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    sm1 = draw_colored_mesh_3d(ax1, X, Y, Z, result['K'], cmap='coolwarm', linewidth=0.6)
    ax1.set_title(f'Gaussian Curvature K\nt = {t_val:.2f}')
    fig.colorbar(sm1, ax=ax1, shrink=0.5, label='K')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    sm2 = draw_colored_mesh_3d(ax2, X, Y, Z, result['H'], cmap='coolwarm', linewidth=0.6)
    ax2.set_title(f'Mean Curvature H\nt = {t_val:.2f}')
    fig.colorbar(sm2, ax=ax2, shrink=0.5, label='H')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


# =============================================================================
# Spherical coordinate visualization
# =============================================================================

def evaluate_model_spherical(
    model,
    t_val: float,
    K_grid: Optional[int] = None,
    K_theta: int = 30,
    K_phi: int = 60,
    theta_range: Tuple[float, float] = (0.1, 3.04),
    phi_range: Tuple[float, float] = (0.0, 2 * np.pi),
    device: str = 'cpu',
) -> Dict[str, np.ndarray]:
    """
    Evaluate neural metric model in spherical coordinates.

    Parameters
    ----------
    model : FundamentalFormNet
        Trained model with spherical topology.
    t_val : float
        Time in [0, 1].
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
    result : dict
        Contains E, F, G, L, M, N, K, H, Theta, Phi arrays.
    """
    import math

    if K_grid is not None:
        K_theta = K_grid
        K_phi = K_grid

    theta_min, theta_max = theta_range
    phi_min, phi_max = phi_range

    theta_1d = torch.linspace(theta_min, theta_max, K_theta, device=device)
    phi_1d = torch.linspace(phi_min, phi_max, K_phi, device=device)
    Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_phi_flat = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)

    N_pts = theta_phi_flat.shape[0]
    t_col = torch.full((N_pts, 1), t_val, device=device)
    t_theta_phi = torch.cat([t_col, theta_phi_flat], dim=1)

    model.eval()
    E, F, G, L, M, N_coef = model(t_theta_phi)

    E = E.reshape(K_theta, K_phi).detach().cpu().numpy()
    F = F.reshape(K_theta, K_phi).detach().cpu().numpy()
    G = G.reshape(K_theta, K_phi).detach().cpu().numpy()
    L = L.reshape(K_theta, K_phi).detach().cpu().numpy()
    M = M.reshape(K_theta, K_phi).detach().cpu().numpy()
    N_coef = N_coef.reshape(K_theta, K_phi).detach().cpu().numpy()

    det = E * G - F**2
    K_gauss = (L * N_coef - M**2) / (det + 1e-12)
    H_mean = (E * N_coef + G * L - 2 * F * M) / (2 * det + 1e-12)

    return {
        'E': E, 'F': F, 'G': G,
        'L': L, 'M': M, 'N': N_coef,
        'K': K_gauss, 'H': H_mean, 'det': det,
        'Theta': Theta.cpu().numpy(),
        'Phi': Phi.cpu().numpy(),
        't': t_val,
    }


def _evaluate_model_rectangular(
    model,
    t_val: float,
    u_range: Tuple[float, float],
    v_range: Tuple[float, float],
    K_u: int,
    K_v: int,
    device: str,
    labels: Tuple[str, str],
) -> Dict[str, np.ndarray]:
    u_min, u_max = u_range
    v_min, v_max = v_range

    u_1d = torch.linspace(u_min, u_max, K_u, device=device)
    v_1d = torch.linspace(v_min, v_max, K_v, device=device)
    U, V = torch.meshgrid(u_1d, v_1d, indexing='ij')
    uv_flat = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)

    N_pts = uv_flat.shape[0]
    t_col = torch.full((N_pts, 1), t_val, device=device)
    tuv = torch.cat([t_col, uv_flat], dim=1)

    model.eval()
    E, F, G, L, M, N_coef = model(tuv)

    E = E.reshape(K_u, K_v).detach().cpu().numpy()
    F = F.reshape(K_u, K_v).detach().cpu().numpy()
    G = G.reshape(K_u, K_v).detach().cpu().numpy()
    L = L.reshape(K_u, K_v).detach().cpu().numpy()
    M = M.reshape(K_u, K_v).detach().cpu().numpy()
    N_coef = N_coef.reshape(K_u, K_v).detach().cpu().numpy()

    det = E * G - F**2
    K_gauss = (L * N_coef - M**2) / (det + 1e-12)
    H_mean = (E * N_coef + G * L - 2 * F * M) / (2 * det + 1e-12)

    return {
        'E': E, 'F': F, 'G': G,
        'L': L, 'M': M, 'N': N_coef,
        'K': K_gauss, 'H': H_mean, 'det': det,
        labels[0]: U.cpu().numpy(),
        labels[1]: V.cpu().numpy(),
        't': t_val,
    }


def evaluate_model_cylindrical(
    model,
    t_val: float,
    K_z: int = 40,
    K_phi: int = 60,
    z_range: Tuple[float, float] = (-1.0, 1.0),
    phi_range: Tuple[float, float] = (0.0, 2 * np.pi),
    device: str = 'cpu',
) -> Dict[str, np.ndarray]:
    """Evaluate a cylindrical-topology model on a (z, phi) grid."""
    return _evaluate_model_rectangular(
        model=model,
        t_val=t_val,
        u_range=z_range,
        v_range=phi_range,
        K_u=K_z,
        K_v=K_phi,
        device=device,
        labels=('Zeta', 'Phi'),
    )


def evaluate_model_toroidal(
    model,
    t_val: float,
    K_u: int = 48,
    K_v: int = 48,
    u_range: Tuple[float, float] = (0.0, 2 * np.pi),
    v_range: Tuple[float, float] = (0.0, 2 * np.pi),
    device: str = 'cpu',
) -> Dict[str, np.ndarray]:
    """Evaluate a toroidal-topology model on a periodic (u, v) grid."""
    return _evaluate_model_rectangular(
        model=model,
        t_val=t_val,
        u_range=u_range,
        v_range=v_range,
        K_u=K_u,
        K_v=K_v,
        device=device,
        labels=('U', 'V'),
    )


def plot_surface_trajectory_spherical(
    model,
    times: List[float] = None,
    K_grid: Optional[int] = None,
    K_theta: int = 30,
    K_phi: int = 60,
    theta_range: Tuple[float, float] = (0.1, 3.04),
    phi_range: Tuple[float, float] = (0.0, 2 * np.pi),
    color_by_curvature: bool = True,
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
    surface_fn: Optional[callable] = None,
):
    """
    Plot 3D surface morphing trajectory for spherical topology.

    Parameters
    ----------
    model : FundamentalFormNet
        Trained model with spherical topology.
    times : list of float
        Timepoints to visualize.
    K_theta, K_phi : int
        Grid resolution.
    theta_range : tuple
        (θ_min, θ_max) range.
    color_by_curvature : bool
        Color mesh by Gaussian curvature.
    surface_fn : callable, optional
        Function (theta, phi, t) -> (x, y, z) for analytic surface.
        If None, uses numerical reconstruction.
    """
    from .reconstruction import reconstruct_surface_spherical
    import math

    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    if K_grid is not None:
        K_theta = K_grid
        K_phi = K_grid

    n_panels = len(times)
    if figsize is None:
        figsize = (4 * n_panels, 4)

    theta_min, theta_max = theta_range
    d_theta = (theta_max - theta_min) / (K_theta - 1)
    phi_min, phi_max = phi_range
    d_phi = (phi_max - phi_min) / (K_phi - 1) if K_phi > 1 else 0.0

    surfaces = []
    results = []

    for t_val in times:
        result = evaluate_model_spherical(
            model,
            t_val,
            K_theta=K_theta,
            K_phi=K_phi,
            theta_range=theta_range,
            phi_range=phi_range,
            device=device,
        )
        results.append(result)

        if surface_fn is not None and (t_val == 0.0 or t_val == 1.0):
            # Use analytic formula for endpoints
            X, Y, Z = surface_fn(result['Theta'], result['Phi'], t_val)
        else:
            # Numerical reconstruction
            X, Y, Z = reconstruct_surface_spherical(
                result['E'], result['F'], result['G'],
                result['L'], result['M'], result['N'],
                d_theta, d_phi
            )
        surfaces.append((X, Y, Z))

    # Global axis limits
    all_coords = []
    for X, Y, Z in surfaces:
        all_coords.extend([X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()])
    coord_range = max(all_coords) - min(all_coords)

    fig = plt.figure(figsize=figsize)

    for idx, (t_val, (X, Y, Z), result) in enumerate(zip(times, surfaces, results)):
        ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')

        if color_by_curvature:
            draw_colored_mesh_3d(ax, X, Y, Z, result['K'], cmap='coolwarm', linewidth=0.5)
        else:
            draw_mesh_3d(ax, X, Y, Z, color='steelblue', linewidth=0.5)

        ax.set_title(f't = {t_val:.2f}', fontsize=11)

        max_range = coord_range / 2
        mid_x, mid_y, mid_z = X.mean(), Y.mean(), Z.mean()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig.suptitle('Surface Trajectory (Spherical)', fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def _plot_surface_trajectory_rectangular(
    model,
    evaluator,
    times: List[float],
    du: float,
    dv: float,
    figsize: Optional[Tuple[float, float]],
    save_path: Optional[str],
    color_by_curvature: bool,
    title: str,
    periodic_u: bool = False,
    periodic_v: bool = False,
):
    n_panels = len(times)
    if figsize is None:
        figsize = (4 * n_panels, 4)

    surfaces = []
    results = []

    for t_val in times:
        result = evaluator(t_val)
        X, Y, Z = reconstruct_surface(
            result['E'], result['F'], result['G'],
            result['L'], result['M'], result['N'],
            du, dv,
        )
        X, Y, Z = enforce_periodic_closure(X, Y, Z, periodic_u=periodic_u, periodic_v=periodic_v)
        surfaces.append((X, Y, Z))
        results.append(result)

    all_coords = []
    for X, Y, Z in surfaces:
        all_coords.extend([X.min(), X.max(), Y.min(), Y.max(), Z.min(), Z.max()])
    coord_range = max(all_coords) - min(all_coords)

    fig = plt.figure(figsize=figsize)

    for idx, (t_val, (X, Y, Z), result) in enumerate(zip(times, surfaces, results)):
        ax = fig.add_subplot(1, n_panels, idx + 1, projection='3d')
        if color_by_curvature:
            draw_colored_mesh_3d(ax, X, Y, Z, result['K'], cmap='coolwarm', linewidth=0.5)
        else:
            draw_mesh_3d(ax, X, Y, Z, color='steelblue', linewidth=0.5)

        ax.set_title(f't = {t_val:.2f}', fontsize=11)
        max_range = coord_range / 2
        mid_x, mid_y, mid_z = X.mean(), Y.mean(), Z.mean()
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_surface_trajectory_cylindrical(
    model,
    times: List[float] = None,
    K_z: int = 40,
    K_phi: int = 60,
    z_range: Tuple[float, float] = (-1.0, 1.0),
    phi_range: Tuple[float, float] = (0.0, 2 * np.pi),
    color_by_curvature: bool = True,
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """Plot a 3D surface trajectory for cylindrical topology."""
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    du = (z_range[1] - z_range[0]) / (K_z - 1)
    dv = (phi_range[1] - phi_range[0]) / (K_phi - 1)

    return _plot_surface_trajectory_rectangular(
        model=model,
        evaluator=lambda t: evaluate_model_cylindrical(
            model, t, K_z=K_z, K_phi=K_phi, z_range=z_range, phi_range=phi_range, device=device
        ),
        times=times,
        du=du,
        dv=dv,
        figsize=figsize,
        save_path=save_path,
        color_by_curvature=color_by_curvature,
        title='Surface Trajectory (Cylindrical)',
        periodic_v=True,
    )


def plot_surface_trajectory_toroidal(
    model,
    times: List[float] = None,
    K_u: int = 48,
    K_v: int = 48,
    u_range: Tuple[float, float] = (0.0, 2 * np.pi),
    v_range: Tuple[float, float] = (0.0, 2 * np.pi),
    color_by_curvature: bool = True,
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
    device: str = 'cpu',
):
    """Plot a 3D surface trajectory for toroidal topology."""
    if times is None:
        times = [0.0, 0.25, 0.5, 0.75, 1.0]

    du = (u_range[1] - u_range[0]) / (K_u - 1)
    dv = (v_range[1] - v_range[0]) / (K_v - 1)

    return _plot_surface_trajectory_rectangular(
        model=model,
        evaluator=lambda t: evaluate_model_toroidal(
            model, t, K_u=K_u, K_v=K_v, u_range=u_range, v_range=v_range, device=device
        ),
        times=times,
        du=du,
        dv=dv,
        figsize=figsize,
        save_path=save_path,
        color_by_curvature=color_by_curvature,
        title='Surface Trajectory (Toroidal)',
        periodic_u=True,
        periodic_v=True,
    )
