"""
model.py — Neural network architectures for fundamental form parameterization.

FundamentalFormNet: Maps (t, u, v) -> (E, F, G, L, M, N)

Always outputs both first and second fundamental forms. For flat (2D) surfaces,
use endpoint functions that return L=M=N=0 and add a flatness constraint.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Tuple


class SirenLayer(nn.Module):
    """Sinusoidal activation layer (SIREN)."""

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.is_first = is_first
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            n = self.linear.in_features
            if self.is_first:
                self.linear.weight.uniform_(-1 / n, 1 / n)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / n) / self.omega_0,
                    np.sqrt(6 / n) / self.omega_0,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


def enforce_metric_positivity(
    E_raw: torch.Tensor,
    F_raw: torch.Tensor,
    G_raw: torch.Tensor,
    margin: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enforce E > 0, G > 0, EG - F² > 0.

    E, G through shifted softplus calibrated so that f(1) = 1.
    This preserves the identity metric at boundary conditions.
    F constrained via tanh * sqrt(EG) * (1 - margin).
    """
    # Shifted softplus: softplus(x - offset) where offset ≈ 0.459
    # makes softplus(1 - 0.459) = softplus(0.541) ≈ 1.0
    # This ensures the identity metric (E=1, G=1) passes through unchanged.
    offset = 0.4586751453870819  # log(e - 1)
    E = torch.nn.functional.softplus(E_raw - offset)
    G = torch.nn.functional.softplus(G_raw - offset)
    max_F = torch.sqrt(E * G) * (1.0 - margin)
    F = torch.tanh(F_raw) * max_F
    return E, F, G


class FundamentalFormNet(nn.Module):
    """
    Maps (t, u, v) -> fundamental forms (E, F, G, L, M, N) of a surface.

    Always outputs 6 components (first and second fundamental forms).
    For flat surfaces, endpoint functions should return L=M=N=0.

    Endpoint conditioning (hard BCs):
        a(t, u, v) = (1-t)*a_0(u,v) + t*a_1(u,v) + t*(1-t)*NN(t,u,v)

    The t*(1-t) factor ensures the correction vanishes at endpoints.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        activation: str = 'silu',
        omega_0: float = 30.0,
        endpoint_a_0: Optional[Callable] = None,
        endpoint_a_1: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        hidden_dim : int
            Hidden layer dimension.
        n_layers : int
            Number of hidden layers.
        activation : str
            Activation function: 'silu', 'softplus', or 'siren'.
        omega_0 : float
            SIREN frequency parameter.
        endpoint_a_0 : callable, optional
            (uv) -> (E, F, G, L, M, N) at t=0.
        endpoint_a_1 : callable, optional
            (uv) -> (E, F, G, L, M, N) at t=1.
        """
        super().__init__()
        self.endpoint_a_0 = endpoint_a_0
        self.endpoint_a_1 = endpoint_a_1

        in_dim = 3  # (t, u, v)
        n_out = 6   # (E, F, G, L, M, N)
        layers = []

        if activation == 'siren':
            layers.append(SirenLayer(in_dim, hidden_dim, omega_0=omega_0, is_first=True))
            for _ in range(n_layers - 1):
                layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
            layers.append(nn.Linear(hidden_dim, n_out))
        else:
            act_fn = nn.SiLU() if activation == 'silu' else nn.Softplus()
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_dim, n_out))

        self.net = nn.Sequential(*layers)

        # Small init on final layer so correction starts near zero
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(
        self, tuv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        tuv : Tensor (N, 3)
            Columns are (t, u, v).

        Returns
        -------
        E, F, G, L, M, N : Tensor (N,)
            Fundamental form components.
        """
        t = tuv[:, 0:1]   # (N, 1)
        uv = tuv[:, 1:3]  # (N, 2)

        raw = self.net(tuv)  # (N, 6)

        if self.endpoint_a_0 is not None and self.endpoint_a_1 is not None:
            # Hard BC mode: baseline is convex combination of endpoints
            # Skip positivity enforcement to preserve exact boundary values
            a_0 = self.endpoint_a_0(uv)
            a_1 = self.endpoint_a_1(uv)

            t_s = t.squeeze(-1)  # (N,)
            blend = t_s * (1 - t_s)

            # baseline = (1-t)*a_0 + t*a_1,  correction = t*(1-t)*NN
            E = (1 - t_s) * a_0[0] + t_s * a_1[0] + blend * raw[:, 0]
            F = (1 - t_s) * a_0[1] + t_s * a_1[1] + blend * raw[:, 1]
            G = (1 - t_s) * a_0[2] + t_s * a_1[2] + blend * raw[:, 2]
            L = (1 - t_s) * a_0[3] + t_s * a_1[3] + blend * raw[:, 3]
            M = (1 - t_s) * a_0[4] + t_s * a_1[4] + blend * raw[:, 4]
            N = (1 - t_s) * a_0[5] + t_s * a_1[5] + blend * raw[:, 5]
        else:
            # No hard BCs: apply positivity enforcement for first fundamental form
            E_raw, F_raw, G_raw = raw[:, 0], raw[:, 1], raw[:, 2]
            E, F, G = enforce_metric_positivity(E_raw, F_raw, G_raw)
            L, M, N = raw[:, 3], raw[:, 4], raw[:, 5]

        return E, F, G, L, M, N


# =============================================================================
# Standard endpoint functions
# =============================================================================

def identity_metric(uv: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Identity metric for a flat sheet: E=1, F=0, G=1, L=M=N=0.

    Parameters
    ----------
    uv : Tensor (N, 2)

    Returns
    -------
    (E, F, G, L, M, N) : tuple of Tensor (N,)
    """
    N = uv.shape[0]
    device, dtype = uv.device, uv.dtype
    ones = torch.ones(N, device=device, dtype=dtype)
    zeros = torch.zeros(N, device=device, dtype=dtype)
    return (ones, zeros, ones, zeros, zeros, zeros)


def elliptic_disc_metric(uv: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Metric of the elliptic (square-to-disc) map, flat (L=M=N=0).

    The map: (x,y) -> (x√(1-y²/2), y√(1-x²/2))

    Parameters
    ----------
    uv : Tensor (N, 2)

    Returns
    -------
    (E, F, G, L, M, N) : tuple of Tensor (N,)
    """
    x, y = uv[:, 0], uv[:, 1]
    x2, y2 = x * x, y * y

    su = torch.sqrt(1.0 - 0.5 * y2)
    sv = torch.sqrt(1.0 - 0.5 * x2)

    J11 = su
    J12 = x * (-y) / (2 * su + 1e-10)
    J21 = y * (-x) / (2 * sv + 1e-10)
    J22 = sv

    E = J11 * J11 + J21 * J21
    F = J11 * J12 + J21 * J22
    G = J12 * J12 + J22 * J22

    zeros = torch.zeros_like(E)
    return (E, F, G, zeros, zeros, zeros)


def hemisphere_metric(uv: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Metric of the unit hemisphere as Monge patch: z = sqrt(1 - u² - v²).

    First fundamental form from Jacobian of the parametrization.
    Second fundamental form gives K=1, H=1 (unit sphere).

    Parameters
    ----------
    uv : Tensor (N, 2)

    Returns
    -------
    (E, F, G, L, M, N) : tuple of Tensor (N,)
    """
    u, v = uv[:, 0], uv[:, 1]
    u2, v2 = u * u, v * v

    # z² = 1 - u² - v², need to stay inside the unit disk
    z2 = 1.0 - u2 - v2
    z2 = torch.clamp(z2, min=1e-6)

    # First fundamental form
    E = (1.0 - v2) / z2
    F = (u * v) / z2
    G = (1.0 - u2) / z2

    # Second fundamental form (for unit hemisphere, shape operator ~ identity)
    # This gives K = 1, H = 1 everywhere
    L = E.clone()
    M = F.clone()
    N_coef = G.clone()

    return (E, F, G, L, M, N_coef)
