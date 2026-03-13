#!/usr/bin/env python3
"""
test_torus_to_horseshoe.py — Test: torus to a baseball-seam-like horseshoe torus.
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
    evaluate_model_toroidal,
    enforce_periodic_closure,
    plot_surface_trajectory_toroidal,
    plot_training_curves,
    reconstruct_surface,
    sample_collocation_toroidal,
)


CONFIG = {
    'hidden_dim': 160,
    'n_layers': 6,
    'activation': 'silu',
    'n_steps': 1000,
    'n_collocation': 4096,
    'lr': 1e-3,
    'u_range': (0.0, 2 * math.pi),
    'v_range': (0.0, 2 * math.pi),
    'major_radius': 1.35,
    'minor_radius': 0.35,
    'seam_a': 1.5,
    'seam_b': 0.95,
    'seam_c': 0.35,
    'w_compatibility': 1.0,
    'w_strain_rate': 0.02,
    'output_dir': 'results_torus_to_horseshoe',
    'log_every': 250,
}


def torus_surface(uv: torch.Tensor) -> torch.Tensor:
    u = uv[:, 0]
    v = uv[:, 1]
    R = CONFIG['major_radius']
    r = CONFIG['minor_radius']
    return torch.stack(
        [
            (R + r * torch.cos(v)) * torch.cos(u),
            (R + r * torch.cos(v)) * torch.sin(u),
            r * torch.sin(v),
        ],
        dim=1,
    )


def horseshoe_centerline(u: torch.Tensor):
    a = CONFIG['seam_a']
    b = CONFIG['seam_b']
    c = CONFIG['seam_c']
    return torch.stack(
        [
            a * torch.cos(u),
            b * torch.sin(u),
            c * torch.sin(2.0 * u),
        ],
        dim=1,
    )


def horseshoe_centerline_derivatives(u: torch.Tensor):
    a = CONFIG['seam_a']
    b = CONFIG['seam_b']
    c = CONFIG['seam_c']
    first = torch.stack(
        [
            -a * torch.sin(u),
            b * torch.cos(u),
            2.0 * c * torch.cos(2.0 * u),
        ],
        dim=1,
    )
    second = torch.stack(
        [
            -a * torch.cos(u),
            -b * torch.sin(u),
            -4.0 * c * torch.sin(2.0 * u),
        ],
        dim=1,
    )
    return first, second


def horseshoe_surface(uv: torch.Tensor) -> torch.Tensor:
    u = uv[:, 0]
    v = uv[:, 1]
    r = CONFIG['minor_radius']

    center = horseshoe_centerline(u)
    d1, d2 = horseshoe_centerline_derivatives(u)
    tangent = d1 / torch.norm(d1, dim=1, keepdim=True).clamp(min=1e-10)

    normal = d2 - (d2 * tangent).sum(dim=1, keepdim=True) * tangent
    fallback = torch.tensor([0.0, 0.0, 1.0], device=uv.device, dtype=uv.dtype).expand_as(normal)
    use_fallback = torch.norm(normal, dim=1, keepdim=True) < 1e-6
    normal = torch.where(use_fallback, fallback, normal)
    normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(min=1e-10)

    binormal = torch.cross(tangent, normal, dim=1)
    binormal = binormal / torch.norm(binormal, dim=1, keepdim=True).clamp(min=1e-10)

    tube_offset = torch.cos(v).unsqueeze(1) * normal + torch.sin(v).unsqueeze(1) * binormal
    return center + r * tube_offset


def torus_metric(uv: torch.Tensor):
    return compute_fundamental_forms_from_embedding(uv, torus_surface, detach=False)


def horseshoe_metric(uv: torch.Tensor):
    return compute_fundamental_forms_from_embedding(uv, horseshoe_surface, detach=False)


def reconstruct_from_result(result):
    du = (CONFIG['u_range'][1] - CONFIG['u_range'][0]) / (result['E'].shape[0] - 1)
    dv = (CONFIG['v_range'][1] - CONFIG['v_range'][0]) / (result['E'].shape[1] - 1)
    X, Y, Z = reconstruct_surface(
        result['E'], result['F'], result['G'],
        result['L'], result['M'], result['N'],
        du, dv,
    )
    return enforce_periodic_closure(X, Y, Z, periodic_u=True, periodic_v=True)


def plot_endpoint_comparison(model, save_path: str, device: str = 'cpu'):
    K_u = 44
    K_v = 44
    result0 = evaluate_model_toroidal(
        model,
        0.0,
        K_u=K_u,
        K_v=K_v,
        u_range=CONFIG['u_range'],
        v_range=CONFIG['v_range'],
        device=device,
    )
    result1 = evaluate_model_toroidal(
        model,
        1.0,
        K_u=K_u,
        K_v=K_v,
        u_range=CONFIG['u_range'],
        v_range=CONFIG['v_range'],
        device=device,
    )

    X0, Y0, Z0 = reconstruct_from_result(result0)
    X1, Y1, Z1 = reconstruct_from_result(result1)

    u_1d = torch.linspace(CONFIG['u_range'][0], CONFIG['u_range'][1], K_u)
    v_1d = torch.linspace(CONFIG['v_range'][0], CONFIG['v_range'][1], K_v)
    U, V = torch.meshgrid(u_1d, v_1d, indexing='ij')
    coords = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)
    xyz_true = horseshoe_surface(coords).detach().numpy()
    X_true = xyz_true[:, 0].reshape(K_u, K_v)
    Y_true = xyz_true[:, 1].reshape(K_u, K_v)
    Z_true = xyz_true[:, 2].reshape(K_u, K_v)

    fig = plt.figure(figsize=(14, 4.5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.plot_surface(X0, Y0, Z0, cmap='Blues', edgecolor='none', alpha=0.9)
    ax1.set_title('t = 0.0')
    ax2.plot_surface(X1, Y1, Z1, cmap='Oranges', edgecolor='none', alpha=0.9)
    ax2.set_title('t = 1.0 (reconstructed)')
    ax3.plot_surface(X_true, Y_true, Z_true, cmap='Greens', edgecolor='none', alpha=0.9)
    ax3.set_title('Target horseshoe torus')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect([1.4, 1.0, 0.9])

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def seam_error(result):
    return {
        'u_metric': float(np.max(np.abs(result['E'][0, :] - result['E'][-1, :]))),
        'v_metric': float(np.max(np.abs(result['E'][:, 0] - result['E'][:, -1]))),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    model = FundamentalFormNet(
        hidden_dim=CONFIG['hidden_dim'],
        n_layers=CONFIG['n_layers'],
        activation=CONFIG['activation'],
        endpoint_a_0=torus_metric,
        endpoint_a_1=horseshoe_metric,
        topology='toroidal',
    )

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['n_steps'])
    loss_fn = CombinedLoss(
        config={
            'compatibility': CONFIG['w_compatibility'],
            'strain_rate': CONFIG['w_strain_rate'],
        },
        reference_metric=torus_metric,
    )

    trainer = MetricFlowTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        endpoint_initial=torus_metric,
        endpoint_final=horseshoe_metric,
        device=device,
        n_collocation=CONFIG['n_collocation'],
        scheduler=scheduler,
        collocation_sampler=lambda n_points, device: sample_collocation_toroidal(
            n_points,
            u_range=CONFIG['u_range'],
            v_range=CONFIG['v_range'],
            device=device,
        ),
    )

    print("=" * 60)
    print("Training: Torus -> Horseshoe Torus")
    print("=" * 60)
    history = trainer.train(CONFIG['n_steps'], log_every=CONFIG['log_every'])

    print("\nEvaluating on grid...")
    for t_val in np.linspace(0.0, 1.0, 6):
        result = evaluate_model_toroidal(
            model,
            t_val,
            K_u=36,
            K_v=36,
            u_range=CONFIG['u_range'],
            v_range=CONFIG['v_range'],
            device=str(device),
        )
        seam = seam_error(result)
        print(
            f"  t={t_val:.2f}: K=[{result['K'].min():.3f}, {result['K'].max():.3f}]  "
            f"seam_u={seam['u_metric']:.3e} seam_v={seam['v_metric']:.3e}"
        )

    trainer.save(os.path.join(CONFIG['output_dir'], 'model.pt'))

    print("\nGenerating plots...")
    plot_training_curves(history, save_path=os.path.join(CONFIG['output_dir'], 'training_curves.png'))
    plot_surface_trajectory_toroidal(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_u=40,
        K_v=40,
        u_range=CONFIG['u_range'],
        v_range=CONFIG['v_range'],
        save_path=os.path.join(CONFIG['output_dir'], 'surface_trajectory_3d.png'),
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
