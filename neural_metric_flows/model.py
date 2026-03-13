"""
model.py — Neural network architectures for fundamental form parameterization.

FundamentalFormNet: Maps (t, u, v) -> (E, F, G, L, M, N)

Always outputs both first and second fundamental forms. For flat (2D) surfaces,
use endpoint functions that return L=M=N=0 and add a flatness constraint.

Supports different topologies via positional encodings:
- 'open': Standard coordinates (u, v) ∈ ℝ²
- 'cylindrical': One periodic axis (e.g., φ ∈ [0, 2π])
- 'spherical': Spherical coordinates (θ, φ) with φ periodic
- 'toroidal': Both axes periodic
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Tuple


# =============================================================================
# Positional Encodings
# =============================================================================

class PeriodicEncoding(nn.Module):
    """
    Fourier feature encoding for periodic coordinates.

    Maps x -> [cos(x), sin(x), cos(2x), sin(2x), ..., cos(kx), sin(kx)]

    This naturally enforces periodicity since cos/sin are 2π-periodic.
    """

    def __init__(self, n_frequencies: int = 4, include_identity: bool = False):
        """
        Parameters
        ----------
        n_frequencies : int
            Number of frequency components (k = 1, 2, ..., n_frequencies).
        include_identity : bool
            If True, also include the raw coordinate.
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        self.include_identity = include_identity
        # Output dimension: 2 * n_frequencies (cos + sin for each freq)
        # Plus 1 if include_identity
        self.out_dim = 2 * n_frequencies + (1 if include_identity else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (...,)
            Input coordinates.

        Returns
        -------
        encoded : Tensor (..., out_dim)
            Fourier features.
        """
        features = []
        if self.include_identity:
            features.append(x.unsqueeze(-1))

        for k in range(1, self.n_frequencies + 1):
            features.append(torch.cos(k * x).unsqueeze(-1))
            features.append(torch.sin(k * x).unsqueeze(-1))

        return torch.cat(features, dim=-1)


class TopologyEncoding(nn.Module):
    """
    Positional encoding that respects manifold topology.

    Topologies:
    - 'open': No periodicity, use raw coordinates with optional Fourier features
    - 'cylindrical': Second coordinate (v/φ) is periodic
    - 'spherical': Second coordinate (φ) is periodic, first (θ) includes cos(θ)
    - 'toroidal': Both coordinates are periodic
    """

    def __init__(
        self,
        topology: str = 'open',
        n_frequencies: int = 4,
        include_raw: bool = True,
    ):
        """
        Parameters
        ----------
        topology : str
            One of 'open', 'cylindrical', 'spherical', 'toroidal'.
        n_frequencies : int
            Number of Fourier frequencies for periodic axes.
        include_raw : bool
            Include raw coordinates for non-periodic axes.
        """
        super().__init__()
        self.topology = topology
        self.n_frequencies = n_frequencies
        self.include_raw = include_raw

        # Build encodings based on topology
        if topology == 'open':
            # Both coordinates raw, optionally with Fourier features
            self.out_dim = 3  # t, u, v
        elif topology == 'cylindrical':
            # u raw, v periodic
            self.periodic_v = PeriodicEncoding(n_frequencies, include_identity=False)
            self.out_dim = 2 + self.periodic_v.out_dim  # t, u, periodic(v)
        elif topology == 'spherical':
            # θ raw + cos(θ) for pole behavior, φ periodic
            self.periodic_phi = PeriodicEncoding(n_frequencies, include_identity=False)
            self.out_dim = 3 + self.periodic_phi.out_dim  # t, θ, cos(θ), periodic(φ)
        elif topology == 'toroidal':
            # Both periodic
            self.periodic_u = PeriodicEncoding(n_frequencies, include_identity=False)
            self.periodic_v = PeriodicEncoding(n_frequencies, include_identity=False)
            self.out_dim = 1 + self.periodic_u.out_dim + self.periodic_v.out_dim
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def forward(self, tuv: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tuv : Tensor (N, 3)
            Coordinates (t, u, v) or (t, θ, φ).

        Returns
        -------
        encoded : Tensor (N, out_dim)
            Topology-aware positional encoding.
        """
        t = tuv[:, 0:1]
        u = tuv[:, 1]
        v = tuv[:, 2]

        if self.topology == 'open':
            return tuv
        elif self.topology == 'cylindrical':
            v_enc = self.periodic_v(v)
            return torch.cat([t, u.unsqueeze(-1), v_enc], dim=-1)
        elif self.topology == 'spherical':
            # θ is u, φ is v
            theta = u
            phi = v
            phi_enc = self.periodic_phi(phi)
            # Include cos(θ) to help with pole behavior
            return torch.cat([t, theta.unsqueeze(-1), torch.cos(theta).unsqueeze(-1), phi_enc], dim=-1)
        elif self.topology == 'toroidal':
            u_enc = self.periodic_u(u)
            v_enc = self.periodic_v(v)
            return torch.cat([t, u_enc, v_enc], dim=-1)


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

    Supports different topologies via positional encoding:
    - 'open': Standard (u, v) coordinates
    - 'cylindrical': v is periodic (useful for surfaces of revolution)
    - 'spherical': (θ, φ) where φ is periodic
    - 'toroidal': Both coordinates periodic
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        activation: str = 'silu',
        omega_0: float = 30.0,
        endpoint_a_0: Optional[Callable] = None,
        endpoint_a_1: Optional[Callable] = None,
        topology: str = 'open',
        n_frequencies: int = 4,
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
        topology : str
            Coordinate topology: 'open', 'cylindrical', 'spherical', 'toroidal'.
        n_frequencies : int
            Number of Fourier frequencies for periodic encodings.
        """
        super().__init__()
        self.endpoint_a_0 = endpoint_a_0
        self.endpoint_a_1 = endpoint_a_1
        self.topology = topology

        # Positional encoding
        self.pos_encoding = TopologyEncoding(
            topology=topology,
            n_frequencies=n_frequencies,
        )
        in_dim = self.pos_encoding.out_dim
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
            Columns are (t, u, v) or (t, θ, φ) for spherical topology.

        Returns
        -------
        E, F, G, L, M, N : Tensor (N,)
            Fundamental form components.
        """
        t = tuv[:, 0:1]   # (N, 1)
        uv = tuv[:, 1:3]  # (N, 2)

        # Apply topology-aware positional encoding
        encoded = self.pos_encoding(tuv)
        raw = self.net(encoded)  # (N, 6)

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


# =============================================================================
# Spherical coordinate metrics
# =============================================================================

def sphere_metric_spherical(theta_phi: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Unit sphere metric in spherical coordinates (θ, φ).

    Parameterization: r(θ,φ) = (sinθ cosφ, sinθ sinφ, cosθ)
    where θ ∈ [0, π] (polar), φ ∈ [0, 2π] (azimuthal).

    First fundamental form:
        E = 1, F = 0, G = sin²(θ)

    Second fundamental form (shape operator = -I for unit sphere):
        L = 1, M = 0, N = sin²(θ)

    This gives K = 1, H = 1 everywhere.

    Parameters
    ----------
    theta_phi : Tensor (N, 2)
        Columns are (θ, φ).

    Returns
    -------
    (E, F, G, L, M, N) : tuple of Tensor (N,)
    """
    theta = theta_phi[:, 0]

    E = torch.ones_like(theta)
    F = torch.zeros_like(theta)
    G = torch.sin(theta) ** 2

    # Second fundamental form
    # For unit sphere with outward normal, L = 1, N = sin²θ
    L = torch.ones_like(theta)
    M = torch.zeros_like(theta)
    N_coef = torch.sin(theta) ** 2

    return (E, F, G, L, M, N_coef)
