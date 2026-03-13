#!/usr/bin/env python3
"""
test_flat_to_fold.py — Test: flat square to a tightly folded sheet.

Endpoint surfaces are developable extrusions, so the trajectory is encouraged to
stay intrinsically flat while bending into 3D.
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_metric_flows import (
    CombinedLoss,
    FundamentalFormNet,
    MetricFlowTrainer,
    compute_fundamental_forms_from_embedding,
    identity_metric,
    plot_curvature_evolution,
    plot_metric_evolution,
    plot_surface_trajectory_3d,
    plot_training_curves,
    reconstruct_from_model,
)


CONFIG = {
    'hidden_dim': 128,
    'n_layers': 4,
    'activation': 'silu',
    'n_steps': 1000,
    'n_collocation': 2048,
    'lr': 1e-3,
    'domain_margin': 0.05,
    'fold_angle': 1.0 * math.pi,
    'hinge_half_width': 0.1,
    'w_compatibility': 1.0,
    'w_flatness': 50.0,
    'w_strain_rate': 0.02,
    'output_dir': 'results_flat_to_fold',
    'log_every': 250,
}


def folded_sheet_surface(uv: torch.Tensor) -> torch.Tensor:
    """Smooth single-hinge fold as a developable extrusion."""
    u = uv[:, 0]
    v = uv[:, 1]

    alpha = CONFIG['fold_angle']
    hinge_half = CONFIG['hinge_half_width']
    radius = 2.0 * hinge_half / alpha

    theta = ((u + hinge_half) / (2.0 * hinge_half)).clamp(0.0, 1.0) * alpha

    arc_x = radius * torch.sin(theta)
    arc_z = radius * (1.0 - torch.cos(theta))

    left_x = u + hinge_half
    left_z = torch.zeros_like(u)

    end_x = radius * math.sin(alpha)
    end_z = radius * (1.0 - math.cos(alpha))
    right_x = end_x + (u - hinge_half) * math.cos(alpha)
    right_z = end_z + (u - hinge_half) * math.sin(alpha)

    x = torch.where(u < -hinge_half, left_x, torch.where(u > hinge_half, right_x, arc_x))
    z = torch.where(u < -hinge_half, left_z, torch.where(u > hinge_half, right_z, arc_z))

    return torch.stack([x, v, z], dim=1)


def folded_sheet_metric(uv: torch.Tensor):
    return compute_fundamental_forms_from_embedding(uv, folded_sheet_surface, detach=False)


def plot_endpoint_comparison(model, margin: float, save_path: str, device: str = 'cpu'):
    K_grid = 40
    lo = -1.0 + margin
    hi = 1.0 - margin

    u_1d = torch.linspace(lo, hi, K_grid)
    v_1d = torch.linspace(lo, hi, K_grid)
    U, V = torch.meshgrid(u_1d, v_1d, indexing='ij')
    uv = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)

    xyz_true = folded_sheet_surface(uv).detach().numpy()
    X_true = xyz_true[:, 0].reshape(K_grid, K_grid)
    Y_true = xyz_true[:, 1].reshape(K_grid, K_grid)
    Z_true = xyz_true[:, 2].reshape(K_grid, K_grid)

    X0, Y0, Z0, _ = reconstruct_from_model(model, 0.0, K_grid=K_grid, margin=margin, device=device)
    X1, Y1, Z1, _ = reconstruct_from_model(model, 1.0, K_grid=K_grid, margin=margin, device=device)

    fig = plt.figure(figsize=(14, 4.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.plot_surface(X0, Y0, Z0, cmap='Blues', edgecolor='none', alpha=0.9)
    ax1.set_title('t = 0.0')
    ax2.plot_surface(X1, Y1, Z1, cmap='Oranges', edgecolor='none', alpha=0.9)
    ax2.set_title('t = 1.0 (reconstructed)')
    ax3.plot_surface(X_true, Y_true, Z_true, cmap='Greens', edgecolor='none', alpha=0.9)
    ax3.set_title('Target folded sheet')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1.2, 1.0, 0.8])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    model = FundamentalFormNet(
        hidden_dim=CONFIG['hidden_dim'],
        n_layers=CONFIG['n_layers'],
        activation=CONFIG['activation'],
        endpoint_a_0=identity_metric,
        endpoint_a_1=folded_sheet_metric,
        topology='open',
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['n_steps'])

    loss_fn = CombinedLoss(
        config={
            'compatibility': CONFIG['w_compatibility'],
            'flatness': CONFIG['w_flatness'],
            'strain_rate': CONFIG['w_strain_rate'],
        },
        reference_metric=identity_metric,
    )

    margin = CONFIG['domain_margin']
    trainer = MetricFlowTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        endpoint_initial=identity_metric,
        endpoint_final=folded_sheet_metric,
        device=device,
        domain_bounds=(-1.0 + margin, 1.0 - margin),
        n_collocation=CONFIG['n_collocation'],
        scheduler=scheduler,
    )

    print("=" * 60)
    print("Training: Flat Sheet -> Folded Sheet")
    print("=" * 60)
    history = trainer.train(CONFIG['n_steps'], log_every=CONFIG['log_every'])

    print("\nEvaluating on grid...")
    results = trainer.evaluate(n_t=11, K_grid=25)
    for r in results:
        print(f"  t={r['t']:.2f}: |K|_max={np.abs(r['K']).max():.4e}  det=[{r['det'].min():.4f}, {r['det'].max():.4f}]")

    trainer.save(os.path.join(CONFIG['output_dir'], 'model.pt'))

    print("\nGenerating plots...")
    plot_training_curves(history, save_path=os.path.join(CONFIG['output_dir'], 'training_curves.png'))
    plot_surface_trajectory_3d(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=35,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'surface_trajectory_3d.png'),
        device=str(device),
    )
    plot_metric_evolution(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=25,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'metric_evolution.png'),
        device=str(device),
    )
    plot_curvature_evolution(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=25,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'curvature_evolution.png'),
        device=str(device),
    )
    plot_endpoint_comparison(
        model,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'endpoint_comparison.png'),
        device=str(device),
    )

    print(f"\nResults saved to {CONFIG['output_dir']}/")
    print("Done!")


if __name__ == '__main__':
    main()
