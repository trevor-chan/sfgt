#!/usr/bin/env python3
"""
test_hemisphere.py — Test: 3D surface trajectory from flat sheet to hemisphere.

Demonstrates learning a curved surface trajectory using the unified
neural metric flow framework.

Endpoints:
- t=0: Flat sheet (identity metric, L=M=N=0)
- t=1: Unit hemisphere (curved, K=1, H=1)

Constraint: Gauss + Codazzi compatibility equations.
"""

import os
import sys
import torch
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_metric_flows import (
    FundamentalFormNet,
    CombinedLoss,
    MetricFlowTrainer,
    identity_metric,
    hemisphere_metric,
    plot_training_curves,
    plot_surface_trajectory_3d,
    plot_endpoint_comparison_3d,
    plot_curvature_evolution,
    plot_curvature_summary,
    plot_curvature_surface_3d,
    plot_second_form_evolution,
)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Model
    'hidden_dim': 128,
    'n_layers': 5,
    'activation': 'silu',

    # Training
    'n_steps': 1000,
    'n_collocation': 2<<14,
    'lr': 1e-3,
    'domain_margin': 0.5,  # Larger margin to stay inside unit disk

    # Loss weights
    'w_compatibility': 1.0,  # Gauss + Codazzi
    'w_strain_rate': 0.01,   # Temporal smoothness

    # Output
    'output_dir': 'results_hemisphere',
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

    # Create model
    model = FundamentalFormNet(
        hidden_dim=CONFIG['hidden_dim'],
        n_layers=CONFIG['n_layers'],
        activation=CONFIG['activation'],
        endpoint_a_0=identity_metric,
        endpoint_a_1=hemisphere_metric,
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['n_steps'])

    # Create loss function (3D: compatibility only, no flatness)
    loss_fn = CombinedLoss(
        config={
            'compatibility': CONFIG['w_compatibility'],
            'strain_rate': CONFIG['w_strain_rate'],
        },
        reference_metric=identity_metric,
    )

    # Create trainer
    margin = CONFIG['domain_margin']
    trainer = MetricFlowTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        endpoint_initial=identity_metric,
        endpoint_final=hemisphere_metric,
        device=device,
        domain_bounds=(-1.0 + margin, 1.0 - margin),
        n_collocation=CONFIG['n_collocation'],
        scheduler=scheduler,
    )

    # Train
    print("=" * 60)
    print("Training: 3D Trajectory (Flat Sheet -> Hemisphere)")
    print("=" * 60)
    history = trainer.train(
        n_steps=CONFIG['n_steps'],
        log_every=CONFIG['log_every'],
    )

    # Evaluate
    print("\nEvaluating on grid...")
    results = trainer.evaluate(n_t=11, K_grid=20)
    for r in results:
        print(f"  t={r['t']:.2f}: K=[{r['K'].min():.3f}, {r['K'].max():.3f}]  "
              f"H=[{r['H'].min():.3f}, {r['H'].max():.3f}]")

    # Save model
    trainer.save(os.path.join(CONFIG['output_dir'], 'model.pt'))

    # Generate plots using visualization module
    print("\nGenerating plots...")

    plot_training_curves(
        history,
        save_path=os.path.join(CONFIG['output_dir'], 'training_curves.png'),
    )

    plot_surface_trajectory_3d(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=20,
        margin=margin,
        color_by_curvature=True,
        save_path=os.path.join(CONFIG['output_dir'], 'surface_trajectory_3d.png'),
        device=str(device),
    )

    plot_endpoint_comparison_3d(
        model,
        K_grid=20,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'endpoint_comparison_3d.png'),
        device=str(device),
    )

    plot_curvature_evolution(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=20,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'curvature_evolution.png'),
        device=str(device),
    )

    plot_curvature_summary(
        model,
        n_t=11,
        K_grid=20,
        margin=margin,
        target_K=1.0,
        target_H=1.0,
        save_path=os.path.join(CONFIG['output_dir'], 'curvature_summary.png'),
        device=str(device),
    )

    plot_curvature_surface_3d(
        model,
        t_val=1.0,
        K_grid=25,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'curvature_surface_3d.png'),
        device=str(device),
    )

    plot_second_form_evolution(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=20,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'second_form_evolution.png'),
        device=str(device),
    )

    print(f"\nResults saved to {CONFIG['output_dir']}/")
    print("Done!")


if __name__ == '__main__':
    main()
