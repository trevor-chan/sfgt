#!/usr/bin/env python3
"""
test_cylinder_to_hyperboloid.py — Test: cylinder to one-sheet hyperboloid.
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
    evaluate_model_cylindrical,
    plot_surface_trajectory_cylindrical,
    plot_training_curves,
    enforce_periodic_closure,
    reconstruct_surface,
    sample_collocation_cylindrical,
)


CONFIG = {
    'hidden_dim': 128,
    'n_layers': 5,
    'activation': 'silu',
    'n_steps': 1000,
    'n_collocation': 4096,
    'lr': 1e-3,
    'z_range': (-1.0, 1.0),
    'phi_range': (0.0, 2 * math.pi),
    'radius': 0.7,
    'hyperboloid_slope': 0.9,
    'w_compatibility': 1.0,
    'w_strain_rate': 0.02,
    'output_dir': 'results_cylinder_to_hyperboloid',
    'log_every': 250,
}


def cylinder_surface(z_phi: torch.Tensor) -> torch.Tensor:
    z = z_phi[:, 0]
    phi = z_phi[:, 1]
    r = CONFIG['radius']
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)


def hyperboloid_surface(z_phi: torch.Tensor) -> torch.Tensor:
    z = z_phi[:, 0]
    phi = z_phi[:, 1]
    r = torch.sqrt(CONFIG['radius'] ** 2 + (CONFIG['hyperboloid_slope'] * z) ** 2)
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=1)


def cylinder_metric(z_phi: torch.Tensor):
    return compute_fundamental_forms_from_embedding(z_phi, cylinder_surface, detach=False)


def hyperboloid_metric(z_phi: torch.Tensor):
    return compute_fundamental_forms_from_embedding(z_phi, hyperboloid_surface, detach=False)


def reconstruct_from_result(result, z_range, phi_range):
    du = (z_range[1] - z_range[0]) / (result['E'].shape[0] - 1)
    dv = (phi_range[1] - phi_range[0]) / (result['E'].shape[1] - 1)
    X, Y, Z = reconstruct_surface(
        result['E'], result['F'], result['G'],
        result['L'], result['M'], result['N'],
        du, dv,
    )
    return enforce_periodic_closure(X, Y, Z, periodic_v=True)


def plot_endpoint_comparison(model, save_path: str, device: str = 'cpu'):
    K_z = 40
    K_phi = 60
    result0 = evaluate_model_cylindrical(
        model,
        0.0,
        K_z=K_z,
        K_phi=K_phi,
        z_range=CONFIG['z_range'],
        phi_range=CONFIG['phi_range'],
        device=device,
    )
    result1 = evaluate_model_cylindrical(
        model,
        1.0,
        K_z=K_z,
        K_phi=K_phi,
        z_range=CONFIG['z_range'],
        phi_range=CONFIG['phi_range'],
        device=device,
    )

    X0, Y0, Z0 = reconstruct_from_result(result0, CONFIG['z_range'], CONFIG['phi_range'])
    X1, Y1, Z1 = reconstruct_from_result(result1, CONFIG['z_range'], CONFIG['phi_range'])

    z_1d = torch.linspace(CONFIG['z_range'][0], CONFIG['z_range'][1], K_z)
    phi_1d = torch.linspace(CONFIG['phi_range'][0], CONFIG['phi_range'][1], K_phi)
    Zeta, Phi = torch.meshgrid(z_1d, phi_1d, indexing='ij')
    coords = torch.stack([Zeta.reshape(-1), Phi.reshape(-1)], dim=1)
    xyz_true = hyperboloid_surface(coords).detach().numpy()
    X_true = xyz_true[:, 0].reshape(K_z, K_phi)
    Y_true = xyz_true[:, 1].reshape(K_z, K_phi)
    Z_true = xyz_true[:, 2].reshape(K_z, K_phi)

    fig = plt.figure(figsize=(14, 4.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.plot_surface(X0, Y0, Z0, cmap='Blues', edgecolor='none', alpha=0.9)
    ax1.set_title('t = 0.0')
    ax2.plot_surface(X1, Y1, Z1, cmap='Oranges', edgecolor='none', alpha=0.9)
    ax2.set_title('t = 1.0 (reconstructed)')
    ax3.plot_surface(X_true, Y_true, Z_true, cmap='Greens', edgecolor='none', alpha=0.9)
    ax3.set_title('Target hyperboloid')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1.0, 1.0, 1.6])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def seam_error(result):
    return {
        'metric': float(np.max(np.abs(result['E'][:, 0] - result['E'][:, -1]))),
        'curvature': float(np.max(np.abs(result['K'][:, 0] - result['K'][:, -1]))),
    }


def plot_curvature_summary_cylindrical(model, save_path: str, device: str = 'cpu'):
    times = np.linspace(0.0, 1.0, 9)
    K_mean = []
    H_mean = []
    K_min = []
    K_max = []
    H_min = []
    H_max = []

    for t_val in times:
        result = evaluate_model_cylindrical(
            model,
            t_val,
            K_z=32,
            K_phi=48,
            z_range=CONFIG['z_range'],
            phi_range=CONFIG['phi_range'],
            device=device,
        )
        K_mean.append(np.mean(result['K']))
        H_mean.append(np.mean(result['H']))
        K_min.append(np.min(result['K']))
        K_max.append(np.max(result['K']))
        H_min.append(np.min(result['H']))
        H_max.append(np.max(result['H']))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, K_mean, 'o-', color='C0')
    axes[0].fill_between(times, K_min, K_max, color='C0', alpha=0.2)
    axes[0].axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Gaussian Curvature')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('K')

    axes[1].plot(times, H_mean, 'o-', color='C1')
    axes[1].fill_between(times, H_min, H_max, color='C1', alpha=0.2)
    axes[1].axhline(0.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('Mean Curvature')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('H')

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
        endpoint_a_0=cylinder_metric,
        endpoint_a_1=hyperboloid_metric,
        topology='cylindrical',
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['n_steps'])
    loss_fn = CombinedLoss(
        config={
            'compatibility': CONFIG['w_compatibility'],
            'strain_rate': CONFIG['w_strain_rate'],
        },
        reference_metric=cylinder_metric,
    )
    trainer = MetricFlowTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        endpoint_initial=cylinder_metric,
        endpoint_final=hyperboloid_metric,
        device=device,
        n_collocation=CONFIG['n_collocation'],
        scheduler=scheduler,
        collocation_sampler=lambda n_points, device: sample_collocation_cylindrical(
            n_points,
            z_range=CONFIG['z_range'],
            phi_range=CONFIG['phi_range'],
            device=device,
        ),
    )

    print("=" * 60)
    print("Training: Cylinder -> Hyperboloid")
    print("=" * 60)
    history = trainer.train(CONFIG['n_steps'], log_every=CONFIG['log_every'])

    print("\nEvaluating on grid...")
    for t_val in np.linspace(0.0, 1.0, 6):
        result = evaluate_model_cylindrical(
            model,
            t_val,
            K_z=36,
            K_phi=48,
            z_range=CONFIG['z_range'],
            phi_range=CONFIG['phi_range'],
            device=str(device),
        )
        print(f"  t={t_val:.2f}: K=[{result['K'].min():.3f}, {result['K'].max():.3f}]  seam={seam_error(result)['metric']:.3e}")

    trainer.save(os.path.join(CONFIG['output_dir'], 'model.pt'))

    print("\nGenerating plots...")
    plot_training_curves(history, save_path=os.path.join(CONFIG['output_dir'], 'training_curves.png'))
    plot_surface_trajectory_cylindrical(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_z=40,
        K_phi=60,
        z_range=CONFIG['z_range'],
        phi_range=CONFIG['phi_range'],
        save_path=os.path.join(CONFIG['output_dir'], 'surface_trajectory_3d.png'),
        device=str(device),
    )
    plot_curvature_summary_cylindrical(
        model,
        save_path=os.path.join(CONFIG['output_dir'], 'curvature_summary.png'),
        device=str(device),
    )
    plot_endpoint_comparison(
        model,
        save_path=os.path.join(CONFIG['output_dir'], 'endpoint_comparison.png'),
        device=str(device),
    )

    print(f"\nResults saved to {CONFIG['output_dir']}/")
    print("Done!")


if __name__ == '__main__':
    main()
