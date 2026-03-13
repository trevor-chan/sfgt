#!/usr/bin/env python3
"""
test_superquadric_reconstruction.py — Verify superquadric metric and reconstruction.

This test verifies that:
1. The superquadric metric computation is correct
2. The surface can be reconstructed from the metric via integration
3. The reconstructed surface matches the analytic superquadric

This helps isolate issues between metric computation, reconstruction, and training.
"""

import os
import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from superquadric import compute_fundamental_forms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_metric_flows.tests.test_sphere_to_cube import (
    superquadric_surface,
    superquadric_metric_spherical,
    sphere_metric_spherical,
)
from neural_metric_flows.reconstruction import reconstruct_surface_spherical


def test_autograd_superquadric_survives_outer_no_grad():
    """Analytic superquadric derivatives should work inside evaluation helpers."""
    theta = torch.linspace(0.2, 1.4, 5)
    phi = torch.linspace(0.2, 1.4, 5)

    with torch.no_grad():
        g, b = compute_fundamental_forms(
            math.pi / 2 - theta,
            phi,
            e1=2.0 / 6.0,
            e2=2.0 / 6.0,
        )

    det = g[:, 0, 0] * g[:, 1, 1] - g[:, 0, 1] ** 2
    assert torch.isfinite(g).all()
    assert torch.isfinite(b).all()
    assert torch.all(det > 0)


def test_metric_properties(n: float = 6.0, K_grid: int = 20):
    """Test that the superquadric metric has expected properties."""
    print(f"\n{'='*60}")
    print(f"Testing superquadric metric properties (n={n})")
    print('='*60)

    # Create grid avoiding poles
    theta_range = (0.2, math.pi - 0.2)
    phi_range = (0.0, 2 * math.pi)

    theta_1d = torch.linspace(theta_range[0], theta_range[1], K_grid)
    phi_1d = torch.linspace(phi_range[0], phi_range[1], K_grid)
    Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_phi = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)

    # Compute metric
    E, F, G, L, M, N = superquadric_metric_spherical(theta_phi, n=n)

    # Check metric is positive definite (det = EG - F^2 > 0)
    det = E * G - F**2
    print(f"\nFirst fundamental form:")
    print(f"  E: min={E.min():.4f}, max={E.max():.4f}, mean={E.mean():.4f}")
    print(f"  F: min={F.min():.4f}, max={F.max():.4f}, mean={F.mean():.4f}")
    print(f"  G: min={G.min():.4f}, max={G.max():.4f}, mean={G.mean():.4f}")
    print(f"  det(I): min={det.min():.4f}, max={det.max():.4f}")
    print(f"  Positive definite: {(det > 0).all().item()}")

    # Compute curvatures
    K = (L * N - M**2) / (det + 1e-12)
    H = (E * N + G * L - 2 * F * M) / (2 * det + 1e-12)

    print(f"\nSecond fundamental form:")
    print(f"  L: min={L.min():.4f}, max={L.max():.4f}")
    print(f"  M: min={M.min():.4f}, max={M.max():.4f}")
    print(f"  N: min={N.min():.4f}, max={N.max():.4f}")

    print(f"\nCurvatures:")
    print(f"  K (Gaussian): min={K.min():.4f}, max={K.max():.4f}, mean={K.mean():.4f}")
    print(f"  H (Mean): min={H.min():.4f}, max={H.max():.4f}, mean={H.mean():.4f}")

    # For sphere (n=2), K should be 1 everywhere
    if n == 2.0:
        print(f"\n  [Sphere check: K should be ~1, H should be ~1]")

    return det.min() > 0


def test_surface_constraint(n: float = 6.0, K_grid: int = 20):
    """Test that superquadric surface satisfies |x|^n + |y|^n + |z|^n = 1."""
    print(f"\n{'='*60}")
    print(f"Testing superquadric surface constraint (n={n})")
    print('='*60)

    theta_range = (0.2, math.pi - 0.2)
    phi_range = (0.0, 2 * math.pi)

    theta_1d = torch.linspace(theta_range[0], theta_range[1], K_grid)
    phi_1d = torch.linspace(phi_range[0], phi_range[1], K_grid)
    Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_phi = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)

    xyz = superquadric_surface(theta_phi, n=n)
    constraint = torch.abs(xyz[:, 0])**n + torch.abs(xyz[:, 1])**n + torch.abs(xyz[:, 2])**n

    print(f"\n|x|^{n} + |y|^{n} + |z|^{n} = 1 constraint:")
    print(f"  min={constraint.min():.6f}, max={constraint.max():.6f}")
    print(f"  Satisfied: {torch.allclose(constraint, torch.ones_like(constraint), atol=1e-4)}")

    return torch.allclose(constraint, torch.ones_like(constraint), atol=1e-4)


def test_reconstruction(n: float = 6.0, K_grid: int = 25):
    """Test surface reconstruction from metric vs analytic surface."""
    print(f"\n{'='*60}")
    print(f"Testing surface reconstruction (n={n})")
    print('='*60)

    # IMPORTANT: Superquadric has metric singularities at cube edges where
    # sin/cos of θ or φ = 0. Safe regions avoid: θ = 0, π/2, π; φ = 0, π/2, π, 3π/2, 2π
    # Use one octant for clean reconstruction:
    theta_range = (0.2, 1.4)  # Between pole and equator
    phi_range = (0.2, 1.4)    # First quadrant only

    theta_1d = torch.linspace(theta_range[0], theta_range[1], K_grid)
    phi_1d = torch.linspace(phi_range[0], phi_range[1], K_grid)
    Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_phi_flat = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)
    
    print(theta_phi_flat.shape)

    # Compute metric on grid
    # First version -------------------
    # E, F, G, L, M, N = superquadric_metric_spherical(theta_phi_flat, n=n)
    # Second version ------------------
    # Convert coordinates: superquadric.py uses u=latitude, we have θ=co-latitude
    # u = π/2 - θ, v = φ
    u = math.pi / 2 - theta_phi_flat[:, 0]
    v = theta_phi_flat[:, 1]
    # For |x|^n + |y|^n + |z|^n = 1, exponents are e1 = e2 = 2/n
    exponent = 2.0 / n
    g_uv, b_uv = compute_fundamental_forms(u, v, e1=exponent, e2=exponent)
    g_uv = g_uv.detach()
    b_uv = b_uv.detach()
    # Coordinate transformation: du/dθ = -1, dv/dφ = 1
    # g_θφ = J^T g_uv J where J = [[-1, 0], [0, 1]]
    # This gives: E_θφ = E_uv, F_θφ = -F_uv, G_θφ = G_uv
    # Same for second fundamental form: L_θφ = L_uv, M_θφ = -M_uv, N_θφ = N_uv
    E = g_uv[:, 0, 0]
    F = -g_uv[:, 0, 1]  # Sign flip due to coordinate transformation
    G = g_uv[:, 1, 1]
    L = b_uv[:, 0, 0]
    M = -b_uv[:, 0, 1]  # Sign flip due to coordinate transformation
    N = b_uv[:, 1, 1]
    # --------------------------------

    # Reshape to grid
    E_grid = E.reshape(K_grid, K_grid).numpy()
    F_grid = F.reshape(K_grid, K_grid).numpy()
    G_grid = G.reshape(K_grid, K_grid).numpy()
    L_grid = L.reshape(K_grid, K_grid).numpy()
    M_grid = M.reshape(K_grid, K_grid).numpy()
    N_grid = N.reshape(K_grid, K_grid).numpy()

    d_theta = (theta_range[1] - theta_range[0]) / (K_grid - 1)
    d_phi = (phi_range[1] - phi_range[0]) / (K_grid - 1)

    # Reconstruct surface from metric
    print("\nReconstructing surface from metric...")
    X_recon, Y_recon, Z_recon = reconstruct_surface_spherical(
        E_grid, F_grid, G_grid,
        L_grid, M_grid, N_grid,
        d_theta, d_phi,
    )

    # Ground truth analytic surface
    xyz_true = superquadric_surface(theta_phi_flat, n=n)
    X_true = xyz_true[:, 0].reshape(K_grid, K_grid).numpy()
    Y_true = xyz_true[:, 1].reshape(K_grid, K_grid).numpy()
    Z_true = xyz_true[:, 2].reshape(K_grid, K_grid).numpy()

    # Center the reconstructed surface (integration gives arbitrary translation)
    X_recon -= X_recon.mean()
    Y_recon -= Y_recon.mean()
    Z_recon -= Z_recon.mean()

    X_true_c = X_true - X_true.mean()
    Y_true_c = Y_true - Y_true.mean()
    Z_true_c = Z_true - Z_true.mean()

    # Scale to match (integration can have scale issues)
    scale_recon = np.sqrt(X_recon**2 + Y_recon**2 + Z_recon**2).mean()
    scale_true = np.sqrt(X_true_c**2 + Y_true_c**2 + Z_true_c**2).mean()
    if scale_recon > 1e-6:
        scale_factor = scale_true / scale_recon
        X_recon *= scale_factor
        Y_recon *= scale_factor
        Z_recon *= scale_factor

    # Compute error
    error = np.sqrt((X_recon - X_true_c)**2 + (Y_recon - Y_true_c)**2 + (Z_recon - Z_true_c)**2)
    print(f"\nReconstruction error (after centering and scaling):")
    print(f"  Mean: {error.mean():.4f}")
    print(f"  Max: {error.max():.4f}")
    print(f"  RMS: {np.sqrt((error**2).mean()):.4f}")

    return X_true, Y_true, Z_true, X_recon, Y_recon, Z_recon


def visualize_comparison(X_true, Y_true, Z_true, X_recon, Y_recon, Z_recon,
                         n: float, save_path: str = None):
    """Visualize analytic vs reconstructed surface."""
    fig = plt.figure(figsize=(15, 5))

    # Ground truth
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X_true, Y_true, Z_true, alpha=0.8, cmap='viridis', edgecolor='none')
    ax1.set_title(f'Analytic Superquadric (n={n})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Reconstructed
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X_recon, Y_recon, Z_recon, alpha=0.8, cmap='plasma', edgecolor='none')
    ax2.set_title('Reconstructed from Metric')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Error map
    ax3 = fig.add_subplot(133, projection='3d')
    error = np.sqrt((X_recon - (X_true - X_true.mean()))**2 +
                    (Y_recon - (Y_true - Y_true.mean()))**2 +
                    (Z_recon - (Z_true - Z_true.mean()))**2)
    surf = ax3.plot_surface(X_true, Y_true, Z_true, facecolors=plt.cm.Reds(error / error.max()),
                            alpha=0.8, edgecolor='none')
    ax3.set_title('Error Map (on analytic surface)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    plt.show()


def test_sphere_baseline(K_grid: int = 25):
    """Test reconstruction for unit sphere (n=2) as a baseline."""
    print(f"\n{'='*60}")
    print("Testing sphere (n=2) reconstruction as baseline")
    print('='*60)

    theta_range = (0.3, math.pi - 0.3)
    phi_range = (0.1, 2 * math.pi - 0.1)

    theta_1d = torch.linspace(theta_range[0], theta_range[1], K_grid)
    phi_1d = torch.linspace(phi_range[0], phi_range[1], K_grid)
    Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_phi_flat = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)

    # Use the known sphere metric
    E, F, G, L, M, N = sphere_metric_spherical(theta_phi_flat)

    print(f"\nSphere metric check:")
    print(f"  E (should be 1): {E.mean():.4f}")
    print(f"  F (should be 0): {F.mean():.4f}")
    print(f"  G (should be sin^2(theta)): sample={G[0]:.4f}, expected={torch.sin(theta_phi_flat[0, 0])**2:.4f}")

    # Reshape to grid
    E_grid = E.reshape(K_grid, K_grid).numpy()
    F_grid = F.reshape(K_grid, K_grid).numpy()
    G_grid = G.reshape(K_grid, K_grid).numpy()
    L_grid = L.reshape(K_grid, K_grid).numpy()
    M_grid = M.reshape(K_grid, K_grid).numpy()
    N_grid = N.reshape(K_grid, K_grid).numpy()

    d_theta = (theta_range[1] - theta_range[0]) / (K_grid - 1)
    d_phi = (phi_range[1] - phi_range[0]) / (K_grid - 1)

    # Reconstruct
    print("\nReconstructing sphere from metric...")
    X_recon, Y_recon, Z_recon = reconstruct_surface_spherical(
        E_grid, F_grid, G_grid,
        L_grid, M_grid, N_grid,
        d_theta, d_phi,
    )

    # Ground truth sphere
    X_true = (torch.sin(Theta) * torch.cos(Phi)).numpy()
    Y_true = (torch.sin(Theta) * torch.sin(Phi)).numpy()
    Z_true = torch.cos(Theta).numpy()

    # Center
    X_recon -= X_recon.mean()
    Y_recon -= Y_recon.mean()
    Z_recon -= Z_recon.mean()

    X_true_c = X_true - X_true.mean()
    Y_true_c = Y_true - Y_true.mean()
    Z_true_c = Z_true - Z_true.mean()

    # Scale
    scale_recon = np.sqrt(X_recon**2 + Y_recon**2 + Z_recon**2).mean()
    scale_true = np.sqrt(X_true_c**2 + Y_true_c**2 + Z_true_c**2).mean()
    if scale_recon > 1e-6:
        scale_factor = scale_true / scale_recon
        X_recon *= scale_factor
        Y_recon *= scale_factor
        Z_recon *= scale_factor

    error = np.sqrt((X_recon - X_true_c)**2 + (Y_recon - Y_true_c)**2 + (Z_recon - Z_true_c)**2)
    print(f"\nSphere reconstruction error:")
    print(f"  Mean: {error.mean():.4f}")
    print(f"  Max: {error.max():.4f}")

    return error.mean() < 0.1


def main():
    n = 10.0
    
    os.makedirs('results_sphere_to_cube', exist_ok=True)

    # Test sphere first (baseline)
    sphere_ok = test_sphere_baseline(K_grid=100)
    print(f"\nSphere baseline: {'PASS' if sphere_ok else 'FAIL'}")

    # # Test superquadric metric properties
    # for n in [2.0, 4.0, 6.0]:
    #     metric_ok = test_metric_properties(n=n, K_grid=20)
    #     print(f"\nMetric positive definite (n={n}): {'PASS' if metric_ok else 'FAIL'}")

    # # Test surface constraint
    # for n in [2.0, 4.0, 6.0]:
    #     constraint_ok = test_surface_constraint(n=n, K_grid=20)
    #     print(f"Surface constraint (n={n}): {'PASS' if constraint_ok else 'FAIL'}")

    # Test reconstruction for superquadric
    print("\n" + "="*60)
    print(f"Full reconstruction test for n={n} superquadric")
    print("="*60)
    X_true, Y_true, Z_true, X_recon, Y_recon, Z_recon = test_reconstruction(n=n, K_grid=100)

    # Visualize
    visualize_comparison(
        X_true, Y_true, Z_true,
        X_recon, Y_recon, Z_recon,
        n=n,
        save_path='results_sphere_to_cube/reconstruction_test.png'
    )


if __name__ == '__main__':
    main()
