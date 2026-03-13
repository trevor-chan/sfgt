#!/usr/bin/env python3
"""
test_sphere_to_cube.py — Test: Sphere to superquadric (cube-like) deformation.

Demonstrates learning a surface trajectory from sphere to superquadric.

NOTE: The superquadric parameterization has metric singularities at "cube edges"
where sin/cos of θ or φ = 0. To avoid numerical issues, this test uses a single
octant (θ ∈ [0.2, 1.4], φ ∈ [0.2, 1.4]) which covers one face of the cube-like
surface without hitting singularities.

Endpoints:
- t=0: Unit sphere patch (K=1, H=1)
- t=1: Superquadric patch |x|^n + |y|^n + |z|^n = 1 with n=6
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_metric_flows import (
    FundamentalFormNet,
    CombinedLoss,
    MetricFlowTrainer,
    sphere_metric_spherical,
    sample_collocation_spherical,
    plot_training_curves,
    plot_surface_trajectory_spherical,
)

# Import the autograd-based metric computation
from superquadric import compute_fundamental_forms_spherical


# =============================================================================
# Superquadric utilities (specific to this test)
# =============================================================================

def signed_power(x: torch.Tensor, p: float) -> torch.Tensor:
    """Compute sign(x) * |x|^p, handling x=0 gracefully."""
    return torch.sign(x) * torch.abs(x).clamp(min=1e-10) ** p


def superquadric_surface(theta_phi: torch.Tensor, n: float = 6.0) -> torch.Tensor:
    """
    Compute superquadric surface embedding from spherical coordinates.

    Parameterization: |x|^n + |y|^n + |z|^n = 1

    Using spherical-like parameterization:
        x = s(sin θ, 2/n) * s(cos φ, 2/n)
        y = s(sin θ, 2/n) * s(sin φ, 2/n)
        z = s(cos θ, 2/n)

    where s(w, m) = sign(w) * |w|^m

    Parameters
    ----------
    theta_phi : Tensor (N, 2)
        Columns are (θ, φ) where θ ∈ [0, π], φ ∈ [0, 2π].
    n : float
        Superquadric exponent. n=2 is sphere, large n approaches cube.

    Returns
    -------
    xyz : Tensor (N, 3)
        Cartesian coordinates (x, y, z).
    """
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]

    m = 2.0 / n

    sin_theta_m = signed_power(torch.sin(theta), m)
    cos_theta_m = signed_power(torch.cos(theta), m)
    sin_phi_m = signed_power(torch.sin(phi), m)
    cos_phi_m = signed_power(torch.cos(phi), m)

    x = sin_theta_m * cos_phi_m
    y = sin_theta_m * sin_phi_m
    z = cos_theta_m

    return torch.stack([x, y, z], dim=1)


def superquadric_metric_spherical(
    theta_phi: torch.Tensor,
    n: float = 6.0,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, ...]:
    """
    Compute fundamental forms for superquadric |x|^n + |y|^n + |z|^n = 1.

    Uses finite differences to compute derivatives (works without autograd).

    Parameters
    ----------
    theta_phi : Tensor (N, 2)
        Columns are (θ, φ) where θ ∈ [0, π], φ ∈ [0, 2π].
    n : float
        Superquadric exponent.
    eps : float
        Finite difference step size.

    Returns
    -------
    (E, F, G, L, M, N) : tuple of Tensor (N,)
    """
    theta = theta_phi[:, 0:1]
    phi = theta_phi[:, 1:2]

    # Central differences for first derivatives
    theta_p = torch.cat([theta + eps, phi], dim=1)
    theta_m = torch.cat([theta - eps, phi], dim=1)
    phi_p = torch.cat([theta, phi + eps], dim=1)
    phi_m = torch.cat([theta, phi - eps], dim=1)

    xyz = superquadric_surface(theta_phi, n=n)
    xyz_theta_p = superquadric_surface(theta_p, n=n)
    xyz_theta_m = superquadric_surface(theta_m, n=n)
    xyz_phi_p = superquadric_surface(phi_p, n=n)
    xyz_phi_m = superquadric_surface(phi_m, n=n)

    # First derivatives: r_θ, r_φ
    r_theta = (xyz_theta_p - xyz_theta_m) / (2 * eps)
    r_phi = (xyz_phi_p - xyz_phi_m) / (2 * eps)

    # First fundamental form
    E = (r_theta * r_theta).sum(dim=1)
    F = (r_theta * r_phi).sum(dim=1)
    G = (r_phi * r_phi).sum(dim=1)

    # Normal vector
    cross = torch.cross(r_theta, r_phi, dim=1)
    cross_norm = torch.norm(cross, dim=1, keepdim=True).clamp(min=1e-10)
    normal = cross / cross_norm

    # Second derivatives for second fundamental form (mixed partials)
    theta_p_phi_p = torch.cat([theta + eps, phi + eps], dim=1)
    theta_p_phi_m = torch.cat([theta + eps, phi - eps], dim=1)
    theta_m_phi_p = torch.cat([theta - eps, phi + eps], dim=1)
    theta_m_phi_m = torch.cat([theta - eps, phi - eps], dim=1)

    xyz_tp_pp = superquadric_surface(theta_p_phi_p, n=n)
    xyz_tp_pm = superquadric_surface(theta_p_phi_m, n=n)
    xyz_tm_pp = superquadric_surface(theta_m_phi_p, n=n)
    xyz_tm_pm = superquadric_surface(theta_m_phi_m, n=n)

    # r_θθ, r_φφ, r_θφ
    r_theta_theta = (xyz_theta_p - 2*xyz + xyz_theta_m) / (eps**2)
    r_phi_phi = (xyz_phi_p - 2*xyz + xyz_phi_m) / (eps**2)
    r_theta_phi = (xyz_tp_pp - xyz_tp_pm - xyz_tm_pp + xyz_tm_pm) / (4 * eps**2)

    # Second fundamental form: L = n · r_θθ, M = n · r_θφ, N = n · r_φφ
    L = (normal * r_theta_theta).sum(dim=1)
    M_coef = (normal * r_theta_phi).sum(dim=1)
    N_coef = (normal * r_phi_phi).sum(dim=1)

    return (E, F, G, L, M_coef, N_coef)


def superquadric_metric_wrapper(theta_phi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Wrapper for superquadric metric with n=6 using autograd."""
    # detach=False so gradients flow for endpoint matching during training
    return compute_fundamental_forms_spherical(theta_phi[:, 0], theta_phi[:, 1], n=6.0, detach=False)


# =============================================================================
# Custom trainer for spherical topology
# =============================================================================

class SphericalMetricFlowTrainer:
    """
    Trainer for metric flows on spherical topology.

    Similar to MetricFlowTrainer but uses spherical collocation sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: CombinedLoss,
        endpoint_initial,
        endpoint_final,
        device: torch.device,
        theta_range: Tuple[float, float] = (0.1, 3.04),
        phi_range: Tuple[float, float] = (0.0, 2 * math.pi),
        n_collocation: int = 4096,
        scheduler=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.endpoint_initial = endpoint_initial
        self.endpoint_final = endpoint_final
        self.device = device
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.n_collocation = n_collocation
        self.scheduler = scheduler

        self.history = {
            'loss_total': [],
            'compatibility': [],
            'strain_rate': [],
        }

    def train_step(self) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Sample collocation points in spherical coordinates
        t_theta_phi = sample_collocation_spherical(
            n_points=self.n_collocation,
            theta_range=self.theta_range,
            phi_range=self.phi_range,
            device=self.device,
        )

        # Forward pass
        E, F, G, L, M, N = self.model(t_theta_phi)

        # Compute loss (CombinedLoss expects E, F, G, L, M, N, tuv)
        total_loss, loss_dict = self.loss_fn(E, F, G, L, M, N, t_theta_phi)

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # Return loss values for logging
        result = {'total': total_loss.item()}
        result.update(loss_dict)
        return result

    def train(self, n_steps: int, log_every: int = 100) -> dict:
        """Full training loop."""
        for step in range(n_steps):
            losses = self.train_step()

            self.history['loss_total'].append(losses['total'])
            self.history['compatibility'].append(losses.get('compatibility', 0))
            self.history['strain_rate'].append(losses.get('strain_rate', 0))

            if (step + 1) % log_every == 0:
                print(f"Step {step + 1:5d} | Loss: {losses['total']:.6f} | "
                      f"Compat: {losses.get('compatibility', 0):.6f} | "
                      f"Strain: {losses.get('strain_rate', 0):.6f}")

        return self.history

    def evaluate(self, n_t: int = 11, K_grid: int = 20) -> list:
        """Evaluate curvatures at multiple time points."""
        results = []
        theta_range = self.theta_range
        phi_range = self.phi_range

        theta_1d = torch.linspace(theta_range[0], theta_range[1], K_grid, device=self.device)
        phi_1d = torch.linspace(phi_range[0], phi_range[1], K_grid, device=self.device)
        Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
        theta_phi_flat = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)

        t_vals = torch.linspace(0, 1, n_t)

        self.model.eval()
        for t_val in t_vals:
            t_col = torch.full((theta_phi_flat.shape[0], 1), t_val.item(), device=self.device)
            t_theta_phi = torch.cat([t_col, theta_phi_flat], dim=1)

            E, F, G, L, M, N = self.model(t_theta_phi)

            # Compute curvatures
            det = E * G - F**2
            K = (L * N - M**2) / (det + 1e-12)
            H = (E * N + G * L - 2 * F * M) / (2 * det + 1e-12)

            results.append({
                't': t_val.item(),
                'K': K.detach().cpu().numpy(),
                'H': H.detach().cpu().numpy(),
            })

        return results

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, path)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Model
    'hidden_dim': 128,
    'n_layers': 5,
    'activation': 'silu',
    'topology': 'open',  # Open patch (one octant, not closed manifold)
    'n_frequencies': 4,

    # Superquadric
    'superquadric_n': 6.0,  # n=2 is sphere, large n approaches cube

    # Training
    'n_steps': 1000,
    'n_collocation': 2 << 14,
    'lr': 1e-3,

    # Domain: one octant to avoid metric singularities at cube edges
    # Singularities occur at θ ∈ {0, π/2, π} and φ ∈ {0, π/2, π, 3π/2, 2π}
    'theta_range': (0.2, 1.4),
    'phi_range': (0.2, 1.4),

    # Loss weights
    'w_compatibility': 1.0,
    'w_strain_rate': 1e-8,  # Very small - metrics differ greatly between sphere and superquadric

    # Output
    'output_dir': 'results_sphere_to_cube',
    'log_every': 200,
}


# =============================================================================
# Main
# =============================================================================

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Create model with spherical topology
    model = FundamentalFormNet(
        hidden_dim=CONFIG['hidden_dim'],
        n_layers=CONFIG['n_layers'],
        activation=CONFIG['activation'],
        endpoint_a_0=sphere_metric_spherical,
        endpoint_a_1=superquadric_metric_wrapper,
        topology=CONFIG['topology'],
        n_frequencies=CONFIG['n_frequencies'],
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['n_steps'])

    # Create loss function
    loss_fn = CombinedLoss(
        config={
            'compatibility': CONFIG['w_compatibility'],
            'strain_rate': CONFIG['w_strain_rate'],
        },
        reference_metric=sphere_metric_spherical,
    )

    # Create trainer
    trainer = SphericalMetricFlowTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        endpoint_initial=sphere_metric_spherical,
        endpoint_final=superquadric_metric_wrapper,
        device=device,
        theta_range=CONFIG['theta_range'],
        phi_range=CONFIG['phi_range'],
        n_collocation=CONFIG['n_collocation'],
        scheduler=scheduler,
    )

    # Train
    print("=" * 60)
    print("Training: Sphere -> Superquadric (n=6, cube-like)")
    print("=" * 60)
    history = trainer.train(
        n_steps=CONFIG['n_steps'],
        log_every=CONFIG['log_every'],
    )

    # Evaluate
    print("\nEvaluating curvatures...")
    results = trainer.evaluate(n_t=11, K_grid=20)
    for r in results:
        print(f"  t={r['t']:.2f}: K=[{r['K'].min():.3f}, {r['K'].max():.3f}]  "
              f"H=[{r['H'].min():.3f}, {r['H'].max():.3f}]")

    # Save model
    trainer.save(os.path.join(CONFIG['output_dir'], 'model.pt'))

    # Generate plots
    print("\nGenerating plots...")

    plot_training_curves(
        history,
        save_path=os.path.join(CONFIG['output_dir'], 'training_curves.png'),
    )

    # Plot surface trajectory using spherical visualization
    plot_surface_trajectory_spherical(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=30,
        theta_range=CONFIG['theta_range'],
        phi_range=CONFIG['phi_range'],
        color_by_curvature=True,
        save_path=os.path.join(CONFIG['output_dir'], 'surface_trajectory_spherical.png'),
        device=str(device),
    )

    # Also plot ground truth comparison
    print("\nGenerating ground truth comparison...")
    plot_endpoint_comparison(
        model,
        superquadric_surface,
        n=CONFIG['superquadric_n'],
        theta_range=CONFIG['theta_range'],
        phi_range=CONFIG['phi_range'],
        K_grid=30,
        save_path=os.path.join(CONFIG['output_dir'], 'endpoint_comparison.png'),
        device=str(device),
    )

    print(f"\nResults saved to {CONFIG['output_dir']}/")
    print("Done!")


def plot_endpoint_comparison(
    model,
    target_surface_fn,
    n: float,
    theta_range: Tuple[float, float],
    phi_range: Tuple[float, float],
    K_grid: int = 30,
    save_path: str = None,
    device: str = 'cpu',
):
    """
    Plot comparison between learned t=1 surface and ground truth superquadric.

    Uses integration for learned metric and analytic formula for ground truth.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from neural_metric_flows.reconstruction import reconstruct_from_model_spherical

    fig = plt.figure(figsize=(14, 6))

    # Ground truth surface
    ax1 = fig.add_subplot(121, projection='3d')
    theta_1d = torch.linspace(theta_range[0], theta_range[1], K_grid)
    phi_1d = torch.linspace(phi_range[0], phi_range[1], K_grid)
    Theta, Phi = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
    theta_phi = torch.stack([Theta.reshape(-1), Phi.reshape(-1)], dim=1)

    xyz_true = target_surface_fn(theta_phi, n=n).numpy()
    X_true = xyz_true[:, 0].reshape(K_grid, K_grid)
    Y_true = xyz_true[:, 1].reshape(K_grid, K_grid)
    Z_true = xyz_true[:, 2].reshape(K_grid, K_grid)

    ax1.plot_surface(X_true, Y_true, Z_true, alpha=0.8, cmap='coolwarm', edgecolor='none')
    ax1.set_title(f'Ground Truth: Superquadric n={n}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_box_aspect([1, 1, 1])

    # Learned surface at t=1
    ax2 = fig.add_subplot(122, projection='3d')
    X_learned, Y_learned, Z_learned = reconstruct_from_model_spherical(
        model,
        t_val=1.0,
        K_grid=K_grid,
        theta_range=theta_range,
        phi_range=phi_range,
        device=device,
    )

    ax2.plot_surface(X_learned, Y_learned, Z_learned, alpha=0.8, cmap='coolwarm', edgecolor='none')
    ax2.set_title('Learned (t=1.0)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


if __name__ == '__main__':
    main()
