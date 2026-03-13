"""
Neural Metric Flows — Learning surface trajectories via differentiable fundamental forms.
"""

from .model import (
    FundamentalFormNet,
    SirenLayer,
    PeriodicEncoding,
    TopologyEncoding,
    enforce_metric_positivity,
    identity_metric,
    elliptic_disc_metric,
    hemisphere_metric,
    sphere_metric_spherical,
)

from .training import (
    CombinedLoss,
    MetricFlowTrainer,
    sample_collocation,
    sample_collocation_spherical,
    sample_collocation_cylindrical,
    sample_collocation_toroidal,
)

from .reconstruction import (
    reconstruct_surface,
    reconstruct_flat_surface,
    reconstruct_from_model,
    compute_christoffel_symbols,
    reconstruct_surface_spherical,
    reconstruct_from_model_spherical,
    enforce_periodic_closure,
)

from .geometry import (
    christoffel_symbols,
    gaussian_curvature_brioschi,
    gauss_residual_3d,
    codazzi_residuals,
    gaussian_curvature,
    mean_curvature,
    green_strain_invariants,
    compute_fundamental_forms_from_embedding,
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
    evaluate_model_spherical,
    evaluate_model_cylindrical,
    evaluate_model_toroidal,
    plot_training_curves,
    plot_surface_trajectory_2d,
    plot_surface_trajectory_3d,
    plot_surface_trajectory_spherical,
    plot_surface_trajectory_cylindrical,
    plot_surface_trajectory_toroidal,
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
    'PeriodicEncoding',
    'TopologyEncoding',
    'enforce_metric_positivity',
    'identity_metric',
    'elliptic_disc_metric',
    'hemisphere_metric',
    'sphere_metric_spherical',
    # Training
    'CombinedLoss',
    'MetricFlowTrainer',
    'sample_collocation',
    'sample_collocation_spherical',
    'sample_collocation_cylindrical',
    'sample_collocation_toroidal',
    # Reconstruction
    'reconstruct_surface',
    'reconstruct_flat_surface',
    'reconstruct_from_model',
    'compute_christoffel_symbols',
    'reconstruct_surface_spherical',
    'reconstruct_from_model_spherical',
    'enforce_periodic_closure',
    # Geometry
    'christoffel_symbols',
    'gaussian_curvature_brioschi',
    'gauss_residual_3d',
    'codazzi_residuals',
    'gaussian_curvature',
    'mean_curvature',
    'green_strain_invariants',
    'compute_fundamental_forms_from_embedding',
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
    'evaluate_model_spherical',
    'evaluate_model_cylindrical',
    'evaluate_model_toroidal',
    'plot_training_curves',
    'plot_surface_trajectory_2d',
    'plot_surface_trajectory_3d',
    'plot_surface_trajectory_spherical',
    'plot_surface_trajectory_cylindrical',
    'plot_surface_trajectory_toroidal',
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
