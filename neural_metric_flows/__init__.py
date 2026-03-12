"""
Neural Metric Flows — Learning surface trajectories via differentiable fundamental forms.
"""

from .model import (
    FundamentalFormNet,
    SirenLayer,
    enforce_metric_positivity,
    identity_metric,
    elliptic_disc_metric,
    hemisphere_metric,
)

from .training import (
    CombinedLoss,
    MetricFlowTrainer,
    sample_collocation,
)

from .reconstruction import (
    reconstruct_surface,
    reconstruct_flat_surface,
    reconstruct_from_model,
    compute_christoffel_symbols,
)

from .geometry import (
    christoffel_symbols,
    gaussian_curvature_brioschi,
    gauss_residual_3d,
    codazzi_residuals,
    gaussian_curvature,
    mean_curvature,
    green_strain_invariants,
)

from .losses import (
    flatness_loss,
    gauss_equation_loss,
    codazzi_loss,
    compatibility_loss_3d,
    elastic_energy_loss,
    strain_rate_loss,
    target_gaussian_curvature_loss,
    target_mean_curvature_loss,
    conformality_loss,
)

from .visualization import (
    evaluate_model_at_time,
    plot_training_curves,
    plot_surface_trajectory_2d,
    plot_surface_trajectory_3d,
    plot_endpoint_comparison_2d,
    plot_endpoint_comparison_3d,
    plot_metric_evolution,
    plot_second_form_evolution,
    plot_curvature_evolution,
    plot_curvature_summary,
    plot_curvature_surface_3d,
    draw_mesh_2d,
    draw_mesh_3d,
)

__all__ = [
    # Model
    'FundamentalFormNet',
    'SirenLayer',
    'enforce_metric_positivity',
    'identity_metric',
    'elliptic_disc_metric',
    'hemisphere_metric',
    # Training
    'CombinedLoss',
    'MetricFlowTrainer',
    'sample_collocation',
    # Reconstruction
    'reconstruct_surface',
    'reconstruct_flat_surface',
    'reconstruct_from_model',
    'compute_christoffel_symbols',
    # Geometry
    'christoffel_symbols',
    'gaussian_curvature_brioschi',
    'gauss_residual_3d',
    'codazzi_residuals',
    'gaussian_curvature',
    'mean_curvature',
    'green_strain_invariants',
    # Losses
    'flatness_loss',
    'gauss_equation_loss',
    'codazzi_loss',
    'compatibility_loss_3d',
    'elastic_energy_loss',
    'strain_rate_loss',
    'target_gaussian_curvature_loss',
    'target_mean_curvature_loss',
    'conformality_loss',
    # Visualization
    'evaluate_model_at_time',
    'plot_training_curves',
    'plot_surface_trajectory_2d',
    'plot_surface_trajectory_3d',
    'plot_endpoint_comparison_2d',
    'plot_endpoint_comparison_3d',
    'plot_metric_evolution',
    'plot_second_form_evolution',
    'plot_curvature_evolution',
    'plot_curvature_summary',
    'plot_curvature_surface_3d',
    'draw_mesh_2d',
    'draw_mesh_3d',
]
