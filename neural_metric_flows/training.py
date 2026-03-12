"""
training.py — Unified training infrastructure for neural metric flows.

Provides:
- CombinedLoss: Assembles multiple loss terms from a config dict
- MetricFlowTrainer: Training loop for neural fundamental form optimization
"""

import time
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .losses import (
    compatibility_loss_3d,
    flatness_loss,
    elastic_energy_loss,
    strain_rate_loss,
    target_gaussian_curvature_loss,
    target_mean_curvature_loss,
    conformality_loss,
)
from .geometry import gaussian_curvature, mean_curvature


# =============================================================================
# Collocation sampling
# =============================================================================

def sample_collocation(
    n_points: int,
    domain_bounds: Tuple[float, float] = (-0.85, 0.85),
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample (t, u, v) uniformly for PINN collocation.

    Parameters
    ----------
    n_points : int
        Number of collocation points.
    domain_bounds : tuple
        (lo, hi) bounds for u, v coordinates.
    device : torch.device
        Target device.

    Returns
    -------
    tuv : Tensor (n_points, 3)
        Collocation points with requires_grad=True.
    """
    lo, hi = domain_bounds
    t = torch.rand(n_points, 1, device=device)
    uv = lo + (hi - lo) * torch.rand(n_points, 2, device=device)
    tuv = torch.cat([t, uv], dim=1)
    tuv.requires_grad_(True)
    return tuv


# =============================================================================
# Combined loss function
# =============================================================================

class CombinedLoss:
    """
    Assembles multiple loss terms from a configuration dict.

    Supported loss terms:
    - 'compatibility': Gauss + Codazzi equations (3D surface validity)
    - 'flatness': K_Brioschi = 0 (for 2D/flat surfaces)
    - 'strain_rate': Temporal smoothness ||da/dt||²
    - 'elastic': St. Venant-Kirchhoff elastic energy
    - 'target_K': Match target Gaussian curvature
    - 'target_H': Match target mean curvature
    - 'conformality': Encourage E = G, F = 0

    Custom losses can be added via custom_losses parameter.
    """

    def __init__(
        self,
        config: Dict[str, float],
        reference_metric: Optional[Callable] = None,
        target_K: Optional[Callable] = None,
        target_H: Optional[Callable] = None,
        custom_losses: Optional[List[Tuple[float, Callable]]] = None,
    ):
        """
        Parameters
        ----------
        config : dict
            Maps loss names to weights, e.g., {'compatibility': 1.0, 'flatness': 100.0}
        reference_metric : callable, optional
            (uv) -> (E0, F0, G0, L0, M0, N0) for elastic energy computation.
        target_K : callable, optional
            (t, u, v) -> K_target for target Gaussian curvature loss.
        target_H : callable, optional
            (t, u, v) -> H_target for target mean curvature loss.
        custom_losses : list of (weight, callable), optional
            Each callable has signature: (E, F, G, L, M, N, tuv) -> scalar tensor.
        """
        self.config = config
        self.reference_metric = reference_metric
        self.target_K = target_K
        self.target_H = target_H
        self.custom_losses = custom_losses or []

    def __call__(
        self,
        E: torch.Tensor,
        F: torch.Tensor,
        G: torch.Tensor,
        L: torch.Tensor,
        M: torch.Tensor,
        N: torch.Tensor,
        tuv: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Parameters
        ----------
        E, F, G : Tensor (N,)
            First fundamental form components.
        L, M, N : Tensor (N,)
            Second fundamental form components.
        tuv : Tensor (N, 3)
            Collocation points (t, u, v) with requires_grad=True.

        Returns
        -------
        total_loss : Tensor
            Scalar loss value.
        loss_components : dict
            Individual loss values for logging.
        """
        device = E.device
        total = torch.tensor(0.0, device=device)
        components = {}

        # Compatibility loss (Gauss + Codazzi)
        w = self.config.get('compatibility', 0.0)
        if w > 0:
            loss = compatibility_loss_3d(E, F, G, L, M, N, tuv, u_idx=1, v_idx=2)
            total = total + w * loss
            components['compatibility'] = loss.item()

        # Flatness loss (K = 0 constraint for 2D case)
        w = self.config.get('flatness', 0.0)
        if w > 0:
            loss = flatness_loss(E, F, G, tuv, u_idx=1, v_idx=2)
            total = total + w * loss
            components['flatness'] = loss.item()

        # Strain rate loss (temporal smoothness)
        w = self.config.get('strain_rate', 0.0)
        if w > 0:
            loss = strain_rate_loss(E, F, G, tuv)
            total = total + w * loss
            components['strain_rate'] = loss.item()

        # Elastic energy loss
        w = self.config.get('elastic', 0.0)
        if w > 0 and self.reference_metric is not None:
            uv = tuv[:, 1:3]
            ref = self.reference_metric(uv)
            E0, F0, G0 = ref[0], ref[1], ref[2]
            loss = elastic_energy_loss(E, F, G, E0, F0, G0)
            total = total + w * loss
            components['elastic'] = loss.item()

        # Target Gaussian curvature
        w = self.config.get('target_K', 0.0)
        if w > 0 and self.target_K is not None:
            K = gaussian_curvature(E, F, G, L, M, N)
            K_target = self.target_K(tuv[:, 0], tuv[:, 1], tuv[:, 2])
            loss = target_gaussian_curvature_loss(K, K_target)
            total = total + w * loss
            components['target_K'] = loss.item()

        # Target mean curvature
        w = self.config.get('target_H', 0.0)
        if w > 0 and self.target_H is not None:
            H = mean_curvature(E, F, G, L, M, N)
            H_target = self.target_H(tuv[:, 0], tuv[:, 1], tuv[:, 2])
            loss = target_mean_curvature_loss(H, H_target)
            total = total + w * loss
            components['target_H'] = loss.item()

        # Conformality loss
        w = self.config.get('conformality', 0.0)
        if w > 0:
            loss = conformality_loss(E, F, G)
            total = total + w * loss
            components['conformality'] = loss.item()

        # Custom losses
        for i, (weight, loss_fn) in enumerate(self.custom_losses):
            loss = loss_fn(E, F, G, L, M, N, tuv)
            total = total + weight * loss
            components[f'custom_{i}'] = loss.item()

        components['total'] = total.item()
        return total, components


# =============================================================================
# Trainer class
# =============================================================================

@dataclass
class MetricFlowTrainer:
    """
    Trainer for neural metric flow optimization.

    Follows PINN paradigm: samples collocation points each step,
    evaluates PDE residuals as losses, optimizes via gradient descent.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: CombinedLoss
    endpoint_initial: Callable  # (uv) -> (E, F, G, L, M, N)
    endpoint_final: Callable    # (uv) -> (E, F, G, L, M, N)
    device: torch.device = None
    domain_bounds: Tuple[float, float] = (-0.85, 0.85)
    n_collocation: int = 2048
    grad_clip: float = 1.0
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    # Training state
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        'loss_total': [], 'K_mean': [], 'H_mean': []
    })

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def train_step(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single training step.

        Returns
        -------
        loss : Tensor
            Total loss value.
        metrics : dict
            Loss components and curvature statistics.
        """
        self.optimizer.zero_grad()

        # Sample collocation points
        tuv = sample_collocation(
            self.n_collocation,
            domain_bounds=self.domain_bounds,
            device=self.device,
        )

        # Forward pass
        E, F, G, L, M, N = self.model(tuv)

        # Compute loss
        loss, loss_components = self.loss_fn(E, F, G, L, M, N, tuv)

        # Compute curvature statistics for monitoring
        with torch.no_grad():
            det_a = E * G - F * F
            K = (L * N - M * M) / (det_a + 1e-12)
            H = (E * N + G * L - 2 * F * M) / (2 * det_a + 1e-12)
            metrics = {
                **loss_components,
                'K_max': torch.max(torch.abs(K)).item(),
                'K_mean': torch.mean(torch.abs(K)).item(),
                'H_max': torch.max(torch.abs(H)).item(),
                'H_mean': torch.mean(torch.abs(H)).item(),
            }

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Optimizer step
        self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        return loss, metrics

    def train(
        self,
        n_steps: int,
        log_every: int = 500,
        callback: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """
        Run the training loop.

        Parameters
        ----------
        n_steps : int
            Number of training steps.
        log_every : int
            Logging frequency.
        callback : callable, optional
            Called every log_every steps with (step, metrics).

        Returns
        -------
        history : dict
            Training history with loss and curvature values.
        """
        self.model.train()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {n_params:,} parameters")
        print(f"Device: {self.device}")
        print(f"Training: {n_steps} steps, {self.n_collocation} collocation points")
        print(f"Domain: [{self.domain_bounds[0]:.2f}, {self.domain_bounds[1]:.2f}]²")
        print()

        t_start = time.time()

        for step in range(n_steps):
            loss, metrics = self.train_step()

            # Record history
            self.history['loss_total'].append(metrics['total'])
            self.history['K_mean'].append(metrics['K_mean'])
            self.history['H_mean'].append(metrics['H_mean'])

            # Record component losses
            for key, value in metrics.items():
                if key not in self.history:
                    self.history[key] = []
                if key not in ['total', 'K_mean', 'H_mean', 'K_max', 'H_max']:
                    self.history[key].append(value)

            # Logging
            if (step + 1) % log_every == 0 or step == 0:
                elapsed = time.time() - t_start
                loss_str = f"loss={metrics['total']:.4e}"
                curv_str = f"|K|={metrics['K_mean']:.4f} |H|={metrics['H_mean']:.4f}"
                print(f"Step {step+1:5d}/{n_steps}  {loss_str}  {curv_str}  ({elapsed:.1f}s)")

                if callback is not None:
                    callback(step + 1, metrics)

        print(f"\nTraining complete in {time.time() - t_start:.1f}s")
        return self.history

    def evaluate(
        self,
        n_t: int = 11,
        K_grid: int = 25,
    ) -> List[Dict]:
        """
        Evaluate model on a regular grid at multiple timepoints.

        Parameters
        ----------
        n_t : int
            Number of time points.
        K_grid : int
            Spatial grid resolution.

        Returns
        -------
        results : list of dict
            Each dict contains metric components, curvatures, etc. at one time.
        """
        lo, hi = self.domain_bounds
        uv_1d = torch.linspace(lo, hi, K_grid, device=self.device)
        U, V = torch.meshgrid(uv_1d, uv_1d, indexing='ij')
        uv_flat = torch.stack([U.reshape(-1), V.reshape(-1)], dim=1)

        t_vals = torch.linspace(0, 1, n_t, device=self.device)
        results = []

        self.model.eval()

        for t_val in t_vals:
            N_pts = uv_flat.shape[0]
            t_col = torch.full((N_pts, 1), t_val.item(), device=self.device)
            tuv = torch.cat([t_col, uv_flat], dim=1)

            with torch.no_grad():
                E, F, G, L, M, N = self.model(tuv)

            det_a = E * G - F * F
            K = (L * N - M * M) / (det_a + 1e-12)
            H = (E * N + G * L - 2 * F * M) / (2 * det_a + 1e-12)

            results.append({
                't': t_val.item(),
                'E': E.reshape(K_grid, K_grid).cpu().numpy(),
                'F': F.reshape(K_grid, K_grid).cpu().numpy(),
                'G': G.reshape(K_grid, K_grid).cpu().numpy(),
                'L': L.reshape(K_grid, K_grid).cpu().numpy(),
                'M': M.reshape(K_grid, K_grid).cpu().numpy(),
                'N': N.reshape(K_grid, K_grid).cpu().numpy(),
                'K': K.reshape(K_grid, K_grid).cpu().numpy(),
                'H': H.reshape(K_grid, K_grid).cpu().numpy(),
                'det': det_a.reshape(K_grid, K_grid).cpu().numpy(),
            })

        self.model.train()
        return results

    def save(self, path: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
