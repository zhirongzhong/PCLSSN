"""Lipschitz-Constrained Linear Layers and Normalization.

This module provides linear layer variants that architecturally enforce
Lipschitz bounds and, optionally, monotonicity constraints on their input
features. These layers serve as the fundamental building blocks for
constructing deep networks with provable stability guarantees.

Layer Summary
-------------
- ``LipschitzConstrainedLinear`` : Linear layer with bounded operator norm.
- ``MonotonicLinear`` : Linear layer with both Lipschitz bound and per-feature
  monotonicity constraints (non-decreasing / non-increasing / unconstrained).
- ``RootMeanSquareNorm`` : RMS normalization layer for internal activation
  stabilization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Union

from .norm import apply_weight_norm

# ---------------------------------------------------------------------------
# Helper: Monotonicity constraint buffer
# ---------------------------------------------------------------------------


def _validate_and_register_monotonicity(
    module: nn.Module,
    constraints: Optional[Iterable[float]],
    in_features: int,
    out_features: int,
) -> None:
    """Validate monotonicity constraints and register them as a buffer.

    Parameters
    ----------
    module : nn.Module
        The module to register the buffer on.
    constraints : iterable of float or None
        Per-input constraint: +1 (non-decreasing), -1 (non-increasing), 0 (free).
        ``None`` defaults to all +1.
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.

    Raises
    ------
    ValueError
        If the constraint shape does not match expected dimensions.
    """
    if constraints is None:
        constraint_tensor = torch.ones(in_features, 1)
    else:
        constraint_tensor = torch.tensor(list(constraints), dtype=torch.float32)

    ndim = constraint_tensor.ndim
    shape = constraint_tensor.shape

    valid = (
        (ndim == 1 and shape[0] == in_features)
        or (ndim == 2 and shape == (in_features, 1))
        or (ndim == 2 and shape == (in_features, out_features))
    )
    if not valid:
        raise ValueError(
            f"Monotonicity constraints must have shape ({in_features},), "
            f"({in_features}, 1), or ({in_features}, {out_features}). "
            f"Got {shape}."
        )

    # Store as (in_features, out_features) for consistent matmul
    if ndim == 1:
        constraint_tensor = constraint_tensor.unsqueeze(-1)
    if constraint_tensor.shape[1] == 1 and out_features > 1:
        constraint_tensor = constraint_tensor.expand(-1, out_features)

    module.register_buffer("_monotonic_signs", constraint_tensor)


# ---------------------------------------------------------------------------
# Lipschitz-Constrained Linear Layer
# ---------------------------------------------------------------------------


class LipschitzConstrainedLinear(nn.Linear):
    """Linear transformation with an architecturally enforced Lipschitz bound.

    The weight matrix is constrained so that its operator norm does not exceed
    ``lipschitz_bound``. This is achieved via differentiable weight
    normalization registered as a PyTorch parametrization.

    Parameters
    ----------
    in_features : int
        Dimensionality of each input sample.
    out_features : int
        Dimensionality of each output sample.
    bias : bool
        If ``True``, the layer learns an additive bias. Default: ``True``.
    lipschitz_bound : float
        Maximum allowed Lipschitz constant for this layer. Default: ``1.0``.
    norm_kind : str
        Type of matrix norm to constrain. One of ``"one"``, ``"inf"``,
        ``"one-inf"``, ``"two"``. See :func:`~.norm.normalize_weights`
        for details. Default: ``"one"``.

    Notes
    -----
    The Lipschitz constraint is enforced **before** each forward pass via
    a parametrization, so gradients flow through the normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lipschitz_bound: float = 1.0,
        norm_kind: str = "one",
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer(
            "lipschitz_bound", torch.tensor(float(lipschitz_bound))
        )
        # Enforce the constraint via differentiable parametrization
        apply_weight_norm(
            self,
            kind=norm_kind,
            max_norm=float(lipschitz_bound),
            always_clamp=False,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"lipschitz_bound={self.lipschitz_bound.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Monotonic Linear Layer
# ---------------------------------------------------------------------------


class MonotonicLinear(LipschitzConstrainedLinear):
    """Linear layer with Lipschitz bound **and** per-feature monotonicity.

    In addition to the weight norm constraint inherited from
    :class:`LipschitzConstrainedLinear`, this layer adds a residual path
    that enforces the output to be monotonic (non-decreasing or
    non-increasing) with respect to specified input features.

    The monotonicity is enforced via the residual formulation:

        output = (W @ x + b + c * x @ M) / 2

    where ``M`` is the monotonicity sign matrix and ``c`` is the Lipschitz
    bound. This construction guarantees that the entire operation remains
    within the Lipschitz bound while satisfying the monotonicity constraints.

    Parameters
    ----------
    in_features : int
        Dimensionality of each input sample.
    out_features : int
        Dimensionality of each output sample.
    bias : bool
        If ``True``, the layer learns an additive bias. Default: ``True``.
    lipschitz_bound : float
        Maximum allowed Lipschitz constant. Default: ``1.0``.
    monotonic_signs : iterable of float or None
        Per-input-feature monotonicity direction:
        - ``+1`` : output is non-decreasing in this input.
        - ``-1`` : output is non-increasing in this input.
        - ``0``  : no constraint (free).
        If ``None``, defaults to all ``+1`` (non-decreasing).
    norm_kind : str
        Norm type for the Lipschitz constraint. Default: ``"one"``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lipschitz_bound: float = 1.0,
        monotonic_signs: Optional[Iterable[float]] = None,
        norm_kind: str = "one",
    ) -> None:
        # Initialize Lipschitz constraint first
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            lipschitz_bound=lipschitz_bound,
            norm_kind=norm_kind,
        )
        # Register monotonicity signs
        _validate_and_register_monotonicity(
            self, monotonic_signs, in_features, out_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with monotonic residual.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(..., in_features)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(..., out_features)``.
        """
        # Main linear path (weight is constrained by parametrization)
        main_out = F.linear(x, self.weight, self.bias)
        # Monotonic residual: x @ signs scales each output by input direction
        monotonic_residual = self.lipschitz_bound * (x @ self._monotonic_signs)
        return (main_out + monotonic_residual) / 2.0


# ---------------------------------------------------------------------------
# Root Mean Square Normalization
# ---------------------------------------------------------------------------


class RootMeanSquareNorm(nn.Module):
    """Root Mean Square (RMS) normalization layer.

    Normalizes inputs by their root-mean-square statistic, then applies
    learnable affine parameters (scale and shift). This is a simplified
    variant of LayerNorm that is cheaper to compute and does not subtract
    the mean.

    RMSNorm is commonly used in Lipschitz networks as an internal
    normalization mechanism that does not break the Lipschitz property.

    Parameters
    ----------
    normalized_shape : int or iterable of int
        Shape of the normalization dimension(s). Typically the feature
        dimension (e.g., ``hidden_size``).
    affine : bool
        If ``True`` (default), learn an affine scale and shift after
        normalization.

    Notes
    -----
    Unlike BatchNorm, RMSNorm computes statistics per-sample rather than
    per-batch, making it suitable for small batch sizes and recurrent
    architectures. The squared-weight scaling (``weight^2``) is used
    instead of linear scaling to keep the operation within the Lipschitz
    constraint framework.
    """

    def __init__(
        self,
        normalized_shape: Union[Iterable[int], int],
        affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)

        # Scale and shift parameters
        init_weight = torch.ones(normalized_shape) / torch.tensor(
            normalized_shape, dtype=torch.float32
        ).sqrt()
        self.weight = nn.Parameter(init_weight, requires_grad=affine)
        self.bias = nn.Parameter(
            torch.zeros(normalized_shape), requires_grad=affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor. Normalization is applied along the last dimension(s)
            matching ``normalized_shape``.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as input.
        """
        # Compute RMS along the last dimension(s)
        rms = torch.sqrt(
            torch.mean(x.pow(2), dim=-1, keepdim=True)
        ).clamp(min=1.0)
        # Affine: scale by weight^2, shift by bias
        return (x / rms) * self.weight.pow(2) + self.bias

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}"
