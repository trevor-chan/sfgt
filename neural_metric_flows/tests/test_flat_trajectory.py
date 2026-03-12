#!/usr/bin/env python3
"""
test_flat_trajectory.py — Test: flat metric trajectory from square to circle.

Demonstrates learning a flat (K=0) surface trajectory using the unified
neural metric flow framework.

Endpoints:
- t=0: Identity metric (square)
- t=1: Elliptic disc metric (circle)

Constraint: Gaussian curvature K = 0 throughout (flatness).
"""

import os
import sys
import numpy as np
import torch
import torch.optim as optim

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from neural_metric_flows import (
    FundamentalFormNet,
    CombinedLoss,
    MetricFlowTrainer,
    identity_metric,
    elliptic_disc_metric,
    plot_training_curves,
    plot_surface_trajectory_2d,
    plot_endpoint_comparison_2d,
    plot_curvature_evolution,
    plot_metric_evolution,
)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # Model
    'hidden_dim': 128,
    'n_layers': 4,
    'activation': 'silu',

    # Training
    'n_steps': 5000,
    'n_collocation': 2048,
    'lr': 1e-3,
    'domain_margin': 0.15,

    # Loss weights
    'w_compatibility': 1.0,  # Gauss + Codazzi
    'w_flatness': 100.0,     # Strong K=0 constraint for flat surface
    'w_strain_rate': 0.01,   # Temporal smoothness

    # Output
    'output_dir': 'results_flat_trajectory',
    'log_every': 500,
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
        endpoint_a_1=elliptic_disc_metric,
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['n_steps'])

    # Create loss function
    loss_fn = CombinedLoss(
        config={
            'compatibility': CONFIG['w_compatibility'],
            'flatness': CONFIG['w_flatness'],
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
        endpoint_final=elliptic_disc_metric,
        device=device,
        domain_bounds=(-1.0 + margin, 1.0 - margin),
        n_collocation=CONFIG['n_collocation'],
        scheduler=scheduler,
    )

    # Train
    print("=" * 60)
    print("Training: Flat Trajectory (Square -> Circle)")
    print("=" * 60)
    history = trainer.train(
        n_steps=CONFIG['n_steps'],
        log_every=CONFIG['log_every'],
    )

    # Evaluate
    print("\nEvaluating on grid...")
    results = trainer.evaluate(n_t=11, K_grid=25)
    for r in results:
        print(f"  t={r['t']:.2f}: |K|_max={np.abs(r['K']).max():.4e}  "
              f"det=[{r['det'].min():.4f}, {r['det'].max():.4f}]")

    # Save model
    trainer.save(os.path.join(CONFIG['output_dir'], 'model.pt'))

    # Generate plots using visualization module
    print("\nGenerating plots...")

    plot_training_curves(
        history,
        save_path=os.path.join(CONFIG['output_dir'], 'training_curves.png'),
    )

    plot_surface_trajectory_2d(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=30,
        margin=margin,
        color_by_det=True,
        save_path=os.path.join(CONFIG['output_dir'], 'surface_trajectory.png'),
        device=str(device),
    )

    plot_endpoint_comparison_2d(
        model,
        K_grid=30,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'endpoint_comparison.png'),
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

    plot_metric_evolution(
        model,
        times=[0.0, 0.25, 0.5, 0.75, 1.0],
        K_grid=25,
        margin=margin,
        save_path=os.path.join(CONFIG['output_dir'], 'metric_evolution.png'),
        device=str(device),
    )

    print(f"\nResults saved to {CONFIG['output_dir']}/")
    print("Done!")


if __name__ == '__main__':
    main()
