"""
losses.py — Loss functions for neural fundamental form parameterization.

Categories:
1. Compatibility losses (Gauss, Codazzi-Mainardi)
2. Geometric regularization (curvature, conformality)
3. Mechanical regularization (elastic energy, strain rate)
4. Boundary condition losses
"""

import torch
from geometry import (
    gaussian_curvature_brioschi,
    gauss_residual_3d,
    codazzi_residuals,
    gaussian_curvature,
    mean_curvature,
    green_strain_invariants,
    partial_t,
)


# =============================================================================
# 1. Compatibility losses
# =============================================================================

def flatness_loss(E, F, G, uv):
    """2D: penalize K_Brioschi != 0."""
    K = gaussian_curvature_brioschi(E, F, G, uv)
    return torch.mean(K ** 2)


def gauss_equation_loss(E, F, G, L, M, N, uv):
    """3D: K_extrinsic must equal K_intrinsic."""
    R = gauss_residual_3d(E, F, G, L, M, N, uv)
    return torch.mean(R ** 2)


def codazzi_loss(E, F, G, L, M, N, uv):
    """3D: Codazzi-Mainardi compatibility."""
    R1, R2 = codazzi_residuals(E, F, G, L, M, N, uv)
    return torch.mean(R1 ** 2 + R2 ** 2)


def compatibility_loss_3d(E, F, G, L, M, N, uv):
    """Combined Gauss + Codazzi for 3D."""
    return gauss_equation_loss(E, F, G, L, M, N, uv) + codazzi_loss(E, F, G, L, M, N, uv)


# =============================================================================
# 2. Geometric regularization
# =============================================================================

def target_gaussian_curvature_loss(K, K_target):
    """Match prescribed Gaussian curvature."""
    return torch.mean((K - K_target) ** 2)


def target_mean_curvature_loss(H, H_target):
    """Match prescribed mean curvature (H=0 for minimal surfaces)."""
    return torch.mean((H - H_target) ** 2)


def metric_matching_loss(E, F, G, Et, Ft, Gt):
    """Match a target first fundamental form."""
    return torch.mean((E - Et) ** 2 + 2 * (F - Ft) ** 2 + (G - Gt) ** 2)


def conformality_loss(E, F, G):
    """Encourage conformal parameterization: E = G, F = 0."""
    return torch.mean((E - G) ** 2 + 4 * F ** 2)


def det_regularity_loss(E, F, G, target_det=1.0):
    """Encourage uniform area element."""
    det_a = E * G - F * F
    return torch.mean((det_a - target_det) ** 2)


# =============================================================================
# 3. Mechanical regularization
# =============================================================================

def elastic_energy_loss(E, F, G, E0, F0, G0, alpha=1.0, beta=1.0):
    """
    St. Venant-Kirchhoff elastic energy from Green strain.

    alpha: bulk modulus, beta: shear modulus.
    """
    tr_S, det_S = green_strain_invariants(E, F, G, E0, F0, G0)
    tr_S2 = tr_S ** 2 - 2 * det_S
    return torch.mean(alpha * tr_S ** 2 + beta * tr_S2)


def strain_rate_loss(E, F, G, tuv):
    """
    Penalize rate of metric change ||da/dt||².
    Encourages temporally smooth trajectories.
    """
    E_t = partial_t(E, tuv)
    F_t = partial_t(F, tuv)
    G_t = partial_t(G, tuv)
    return torch.mean(E_t ** 2 + 2 * F_t ** 2 + G_t ** 2)


# =============================================================================
# 4. Boundary condition losses (soft)
# =============================================================================

def endpoint_metric_loss(model, uv_pts, a_target_fn, t_val):
    """
    Soft endpoint BC: match fundamental forms at t = t_val.
    """
    N = uv_pts.shape[0]
    t_col = torch.full((N, 1), t_val, device=uv_pts.device)
    tuv = torch.cat([t_col, uv_pts], dim=1)
    tuv.requires_grad_(True)

    outputs = model(tuv)
    targets = a_target_fn(uv_pts)

    loss = sum(torch.mean((p - tgt) ** 2) for p, tgt in zip(outputs, targets))
    return loss


def spatial_boundary_loss(E, F, G, E_bc, F_bc, G_bc):
    """Match metric on spatial boundary (Dirichlet)."""
    return torch.mean((E - E_bc) ** 2 + (F - F_bc) ** 2 + (G - G_bc) ** 2)
