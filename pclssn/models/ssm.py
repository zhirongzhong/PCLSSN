"""Parallel-Scan State-Space Model (SSM) for Sequence Processing.

This module implements a discretized state-space model with parallel
scan computation, inspired by recent advances in structured state-space
sequence models. The parallel scan formulation enables efficient training
on GPUs while maintaining the sequential inductive bias of SSMs.

The SSM dynamics follow:

    h_t = exp(log_A + dt) * h_{t-1} + B_t * x_t
    y_t = C_t * h_t + D * x_t

where A, B, C, D are (learned) system matrices and dt is a learned
time-constant modulator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pclssn.lipschitz import LipschitzConstrainedLinear


class ParallelScanSSM(nn.Module):
    """State-space model with parallel scan for efficient sequence processing.

    Uses Lipschitz-constrained linear projections for the input-to-state
    mapping, ensuring the SSM dynamics preserve the overall network's
    Lipschitz property.

    Parameters
    ----------
    d_model : int
        Dimensionality of the input and output features.
    d_state : int
        Dimensionality of the internal state. Default: ``16``.
    lipschitz_bound : float
        Lipschitz constant for the constrained linear projections.
        Default: ``1.0``.
    dt_min : float
        Minimum allowed time delta to prevent degenerate dynamics.
        Default: ``1e-4``.
    dt_max : float
        Maximum allowed time delta to prevent instability.
        Default: ``0.05``.

    Notes
    -----
    The parallel scan uses cumulative-sum operations to compute all hidden
    states simultaneously, avoiding the sequential bottleneck of RNNs.
    Numerical stability is maintained through log-space accumulation
    and clamping of the cumulative exponent.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        lipschitz_bound: float = 1.0,
        dt_min: float = 1e-4,
        dt_max: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_min = dt_min
        self.dt_max = dt_max

        # ---- Learnable Lipschitz constant (can be updated externally) ----
        self.lipschitz_bound = lipschitz_bound

        # ---- System dynamics parameters ----
        # State transition log-matrix A: (d_model, d_state)
        # Initialized so eigenvalues are in (0, 1), encouraging stable dynamics
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32) + 1.0
            ).repeat(d_model, 1)
        )

        # Direct feedthrough (skip connection) parameter D: (d_model,)
        self.D = nn.Parameter(torch.ones(d_model))

        # ---- Lipschitz-constrained projections ----
        # Input projection: x → (dt_raw, B, C)
        proj_out = d_model // 16 + d_state * 2
        self.input_proj = LipschitzConstrainedLinear(
            d_model,
            proj_out,
            bias=False,
            lipschitz_bound=lipschitz_bound,
            norm_kind="one-inf",
        )

        # Time-delta projection: bottleneck → dt
        self.delta_proj = LipschitzConstrainedLinear(
            d_model // 16,
            d_model,
            bias=True,
            lipschitz_bound=lipschitz_bound,
            norm_kind="one",
        )

        # Initialize delta bias to favor small initial steps
        nn.init.constant_(self.delta_proj.bias, -4.0)

    def sync_lipschitz(self, value: float | torch.Tensor) -> None:
        """Synchronize the Lipschitz bound to all inner constrained layers.

        Parameters
        ----------
        value : float or torch.Tensor
            New Lipschitz bound value.
        """
        if isinstance(value, torch.Tensor):
            target = value
        else:
            target = torch.tensor(float(value))

        self.lipschitz_bound = target
        self.input_proj.lipschitz_bound.data.copy_(target)
        self.delta_proj.lipschitz_bound.data.copy_(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with parallel scan.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape ``(batch, seq_len, d_model)``.

        Returns
        -------
        torch.Tensor
            Output sequence of shape ``(batch, seq_len, d_model)``.
        """
        B_sz, L_sz, D_sz = x.shape

        # ---- 1. Discretization parameters ----
        # A: (d_model, d_state) — continuous-time state matrix
        A = -torch.exp(self.A_log.float())

        # Project input to dt, B, C
        projected = self.input_proj(x)
        dt_raw, B_state, C_state = torch.split(
            projected,
            [self.d_model // 16, self.d_state, self.d_state],
            dim=-1,
        )

        # Time delta with softplus gating and clamping
        dt = F.softplus(self.delta_proj(dt_raw))
        dt = torch.clamp(dt, min=self.dt_min, max=self.dt_max)  # (B, L, D)

        # ---- 2. Discretize A and B ----
        # log(dA) = dt * A — accumulate in log-space for stability
        log_dA = dt.unsqueeze(-1) * A.view(1, 1, D_sz, self.d_state)
        # dB * x = dt * B * x
        dBx = (dt.unsqueeze(-1) * B_state.unsqueeze(2)) * x.unsqueeze(-1)

        # Exponential of discretized A
        dA = torch.exp(log_dA)

        # ---- 3. Parallel scan via cumulative sums ----
        # Clamp cumulative log for numerical safety
        log_dA_cumsum = torch.clamp(
            torch.cumsum(log_dA, dim=1), max=20.0
        )
        exp_dA_cumsum = torch.exp(log_dA_cumsum)

        # Parallel scan: h_t = exp(cumsum_1:t log_dA) * cumsum_1:t (dBx / exp_cumsum)
        h = exp_dA_cumsum * torch.cumsum(
            dBx / (exp_dA_cumsum + 1e-12), dim=1
        )

        # NaN guard
        h = torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- 4. Output projection ----
        y = torch.einsum("bldn,bln->bld", h, C_state)
        return y + x * self.D
