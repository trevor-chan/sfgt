"""
generate_simple_test.py — Visualize linear interpolation of mesh vertices
from a square grid to an elliptic-map circle grid.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

from geometry import square_to_circle


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


def generate_morphing_plot(K=25, n_panels=6, save_path='plots/simple_morphing.png'):
    """
    Generate a plot_morphing_mesh-style figure showing linear interpolation
    of grid vertices from a unit square to an elliptic-map circle.
    """
    # Square grid in [-1, 1]^2
    u = np.linspace(-1, 1, K)
    v = np.linspace(-1, 1, K)
    X_sq, Y_sq = np.meshgrid(u, v, indexing='ij')

    # Circle grid via elliptic map
    X_circ, Y_circ = square_to_circle(X_sq, Y_sq)

    # Center both
    X_sq -= X_sq.mean()
    Y_sq -= Y_sq.mean()
    X_circ -= X_circ.mean()
    Y_circ -= Y_circ.mean()

    # Time parameters
    times = np.linspace(0, 1, n_panels)

    # Precompute surfaces and axis limits
    surfaces = []
    all_x, all_y = [], []
    for t in times:
        X = (1 - t) * X_sq + t * X_circ
        Y = (1 - t) * Y_sq + t * Y_circ
        surfaces.append((X, Y))
        all_x.extend([X.min(), X.max()])
        all_y.extend([Y.min(), Y.max()])

    pad = 0.05
    xlim = (min(all_x) - pad, max(all_x) + pad)
    ylim = (min(all_y) - pad, max(all_y) + pad)

    # Plot
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    for ax, t, (X, Y) in zip(axes, times, surfaces):
        # Color by local area distortion: det of the Jacobian approximation
        # Use cell area as a proxy
        dx_du = np.gradient(X, axis=0)
        dy_du = np.gradient(Y, axis=0)
        dx_dv = np.gradient(X, axis=1)
        dy_dv = np.gradient(Y, axis=1)
        det = dx_du * dy_dv - dx_dv * dy_du

        _draw_colored_mesh(ax, X, Y, det, cmap='viridis', linewidth=0.6)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.2f}', fontsize=10)
        ax.tick_params(labelsize=7)

    fig.suptitle('Square → Circle morphing (linear vertex interpolation)', fontsize=12, y=1.02)
    fig.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')

    return fig


if __name__ == '__main__':
    fig = generate_morphing_plot()
    plt.close(fig)
