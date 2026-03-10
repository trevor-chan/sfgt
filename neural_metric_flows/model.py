"""
model.py — Neural network architectures for fundamental form parameterization.

FundamentalFormNet: Maps (t, u, v) -> metric components.
Supports 2D (E, F, G only) and 3D (E, F, G, L, M, N) modes.
"""

import torch
import torch.nn as nn
import numpy as np


class SirenLayer(nn.Module):
    """Sinusoidal activation layer (SIREN)."""

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
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

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


def enforce_metric_positivity(E_raw, F_raw, G_raw, margin=0.01):
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
    Maps (t, u, v) -> fundamental forms of a surface.

    mode='2d': outputs (E, F, G)
    mode='3d': outputs (E, F, G, L, M, N)

    Endpoint conditioning (hard BCs):
        a(t, u, v) = (1-t)*a_o(u,v) + t*a_f(u,v) + t*(1-t)*NN(t,u,v)
    """

    def __init__(
        self,
        mode='2d',
        hidden_dim=128,
        n_layers=4,
        activation='silu',
        omega_0=30.0,
        endpoint_a_o=None,
        endpoint_a_f=None,
    ):
        """
        Parameters
        ----------
        mode : '2d' or '3d'
        hidden_dim : int
        n_layers : int
        activation : 'silu', 'softplus', or 'siren'
        omega_0 : float — SIREN frequency
        endpoint_a_o : callable (uv) -> tuple of tensors at t=0
        endpoint_a_f : callable (uv) -> tuple of tensors at t=1
        """
        super().__init__()
        self.mode = mode
        self.n_out_raw = 3 if mode == '2d' else 6
        self.endpoint_a_o = endpoint_a_o
        self.endpoint_a_f = endpoint_a_f

        in_dim = 3  # (t, u, v)
        layers = []

        if activation == 'siren':
            layers.append(SirenLayer(in_dim, hidden_dim, omega_0=omega_0, is_first=True))
            for _ in range(n_layers - 1):
                layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
            layers.append(nn.Linear(hidden_dim, self.n_out_raw))
        else:
            act_fn = nn.SiLU() if activation == 'silu' else nn.Softplus()
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_dim, self.n_out_raw))

        self.net = nn.Sequential(*layers)

        # Small init on final layer so correction starts near zero
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.mul_(0.01)

    def forward(self, tuv):
        """
        Parameters
        ----------
        tuv : Tensor (N, 3) — columns are (t, u, v)

        Returns
        -------
        2d mode: (E, F, G) each Tensor (N,)
        3d mode: (E, F, G, L, M, N) each Tensor (N,)
        """
        t = tuv[:, 0:1]   # (N, 1)
        uv = tuv[:, 1:3]  # (N, 2)

        raw = self.net(tuv)  # (N, n_out_raw)

        if self.endpoint_a_o is not None and self.endpoint_a_f is not None:
            # Hard BC mode: baseline is already a valid metric (convex combo of valid metrics)
            # Skip positivity enforcement to preserve exact boundary values
            a_o = self.endpoint_a_o(uv)
            a_f = self.endpoint_a_f(uv)

            t_s = t.squeeze(-1)  # (N,)
            blend = t_s * (1 - t_s)

            # baseline = (1-t)*a_o + t*a_f,  correction = t*(1-t)*NN
            baseline = tuple(
                (1 - t_s) * ao_i + t_s * af_i
                for ao_i, af_i in zip(a_o, a_f)
            )

            if self.mode == '2d':
                E = baseline[0] + blend * raw[:, 0]
                F = baseline[1] + blend * raw[:, 1]
                G = baseline[2] + blend * raw[:, 2]
            else:
                E = baseline[0] + blend * raw[:, 0]
                F = baseline[1] + blend * raw[:, 1]
                G = baseline[2] + blend * raw[:, 2]
                L_out = baseline[3] + blend * raw[:, 3]
                M_out = baseline[4] + blend * raw[:, 4]
                N_out = baseline[5] + blend * raw[:, 5]
        else:
            # No hard BCs: apply positivity enforcement
            if self.mode == '2d':
                E_raw, F_raw, G_raw = raw[:, 0], raw[:, 1], raw[:, 2]
            else:
                E_raw, F_raw, G_raw = raw[:, 0], raw[:, 1], raw[:, 2]
                L_out, M_out, N_out = raw[:, 3], raw[:, 4], raw[:, 5]
            E, F, G = enforce_metric_positivity(E_raw, F_raw, G_raw)

        if self.mode == '2d':
            return E, F, G
        else:
            return E, F, G, L_out, M_out, N_out
