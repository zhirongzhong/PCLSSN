"""Monotonic Wrapper for Lipschitz Modules.

This module provides a wrapper that endows any Lipschitz-constrained network
with architecturally guaranteed monotonicity with respect to specified input
features. The wrapper adds a linear residual term to the network output,
which is a standard construction for combining Lipschitz and monotonicity
constraints in a single architecture.

Theory
------
For a K-Lipschitz function ``f(x)``, the wrapped function

    g(x) = f(x) + K * x @ M

is also K-Lipschitz and, when the sign matrix M is chosen appropriately,
guarantees that g is monotonic in the prescribed directions. The residual
term ``K * x @ M`` provides the monotonic "drift" while the Lipschitz
network ``f`` models the nonlinear component.

This construction is a standard technique in the monotone network literature
and does not rely on any single original implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Iterable, Optional, Union


class MonotonicWrapper(nn.Module):
    """Wrap a Lipschitz module with an additive monotonic residual.

    Given a Lipschitz module ``f`` with constant ``K``, the wrapped output is::

        g(x) = f(x) + K * (x @ monotonic_signs)

    where ``monotonic_signs`` is a matrix specifying per-input/per-output
    monotonicity directions. The signs indicate:

    - ``+1`` : output is non-decreasing w.r.t. this input feature.
    - ``-1`` : output is non-increasing w.r.t. this input feature.
    - ``0``  : no constraint (free input).

    Parameters
    ----------
    lipschitz_module : nn.Module
        A neural network module that is already constrained to be
        K-Lipschitz with constant ``lipschitz_bound``.
    lipschitz_bound : float
        The Lipschitz constant ``K`` of the wrapped module. Default: ``1.0``.
    monotonic_signs : iterable of float, optional
        Monotonicity constraints. Shape can be:
        - 1D ``(in_features,)`` — same constraint applied to all outputs.
        - 2D ``(in_features, 1)`` — same as above, with explicit output dim.
        - 2D ``(in_features, out_features)`` — per-output constraints.
        If ``None``, defaults to all inputs being non-decreasing (all ``+1``).

    Notes
    -----
    The input module MUST already respect the ``lipschitz_bound`` Lipschitz
    constraint. This wrapper does **not** enforce Lipschitz continuity on
    the inner module — it only adds the monotonic residual term.

    Examples
    --------
    >>> net = nn.Sequential(
    ...     LipschitzConstrainedLinear(3, 16),
    ...     GroupedSort(8),
    ...     LipschitzConstrainedLinear(16, 1),
    ... )
    >>> mono_net = MonotonicWrapper(net, lipschitz_bound=1.0,
    ...                             monotonic_signs=[1, 0, -1])
    """

    def __init__(
        self,
        lipschitz_module: nn.Module,
        lipschitz_bound: float = 1.0,
        monotonic_signs: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__()
        self.inner_module = lipschitz_module
        self.register_buffer(
            "lipschitz_bound", torch.tensor(float(lipschitz_bound))
        )

        # ---- Parse and store monotonicity signs ----
        if monotonic_signs is None:
            sign_tensor = torch.tensor([[1.0]])  # default: all increasing
        else:
            sign_tensor = torch.tensor(
                list(monotonic_signs), dtype=torch.float32
            )

        # Normalize to at least 2D — shape (in_features, out_features)
        if sign_tensor.ndim == 0:
            sign_tensor = sign_tensor.reshape(1, 1)
        elif sign_tensor.ndim == 1:
            sign_tensor = sign_tensor.unsqueeze(-1)

        self.register_buffer("_monotonic_signs", sign_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute monotonic-wrapped output.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape ``(..., in_features)``. The last dimension
            must match the number of rows in the monotonic sign matrix.

        Returns
        -------
        torch.Tensor
            Output tensor ``g(x) = f(x) + K * (x @ M)``.
        """
        # Expand signs to match input features if needed
        mc = self._monotonic_signs
        if mc.shape[0] != x.shape[-1]:
            mc = mc.expand(x.shape[-1], -1)

        # Monotonic residual: K * x @ M
        monotonic_residual = self.lipschitz_bound * torch.matmul(x, mc)

        return self.inner_module(x) + monotonic_residual

    def extra_repr(self) -> str:
        signs_shape = tuple(self._monotonic_signs.shape)
        return (
            f"lipschitz_bound={self.lipschitz_bound.item():.4f}, "
            f"signs_shape={signs_shape}"
        )
