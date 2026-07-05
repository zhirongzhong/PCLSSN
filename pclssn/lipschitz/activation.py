"""Grouped Sorting Activation for Lipschitz Networks.

This module implements the grouped sorting activation, which is a
gradient-norm-preserving nonlinearity used in Lipschitz-constrained
networks. Unlike standard activations (ReLU, tanh) that can shrink
gradients, group sorting is an orthogonal projection that preserves
the norm of the input while introducing nonlinearity through ordering.

Mathematical Background
----------------------
Given an input vector ``x`` of dimension ``d``, GroupSort splits ``x``
into ``n_groups`` equally-sized segments, sorts each segment in ascending
(or descending) order, and concatenates the results. This operation:

1. **Is 1-Lipschitz**: The sorting operation is non-expansive.
2. **Preserves gradient norm**: The Jacobian of the sorting operation
   is a permutation matrix, which has operator norm exactly 1.
3. **Is nonlinear**: Despite being norm-preserving, reordering elements
   is a valid nonlinearity that enables universal approximation when
   combined with Lipschitz linear layers.

Reference
---------
The grouped sorting activation for Lipschitz networks was introduced in:
    Anil, Lucas, et al. "Sorting Out Lipschitz Function Approximation"
    (ICML 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def grouped_sort_1d(
    x: torch.Tensor,
    n_groups: int,
    axis: int = -1,
    descending: bool = False,
) -> torch.Tensor:
    """Apply grouped sorting along one axis of a tensor.

    Splits the feature dimension into ``n_groups`` equal segments, sorts
    each segment independently, and reassembles the result.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape. The dimension specified by ``axis``
        must be evenly divisible by ``n_groups``.
    n_groups : int
        Number of groups to split the feature dimension into. Must evenly
        divide ``x.shape[axis]``.
    axis : int
        Axis along which to apply the grouping and sorting (default: ``-1``,
        the last dimension).
    descending : bool
        If ``True``, sort each group in descending order instead of ascending.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as ``x``, with each group sorted.

    Raises
    ------
    ValueError
        If the feature dimension size is not a multiple of ``n_groups``.

    Examples
    --------
    >>> x = torch.tensor([[3.0, 1.0, 4.0, 2.0]])
    >>> grouped_sort_1d(x, n_groups=2)
    tensor([[1., 3., 2., 4.]])  # First group [3,1]→[1,3], second [4,2]→[2,4]
    """
    # Short-circuit on empty inputs
    if x.shape[0] == 0:
        return x

    dim_size = x.shape[axis]
    if dim_size % n_groups != 0:
        raise ValueError(
            f"Feature dimension size ({dim_size}) must be divisible by "
            f"n_groups ({n_groups})."
        )

    group_size = dim_size // n_groups

    # ---- Reshape into groups ----
    # Strategy: move target axis to last position, reshape to expose groups,
    # sort, then restore original shape.
    if axis == -1 or axis == x.ndim - 1:
        # Fast path — axis is already the last dimension
        reshaped = x.reshape(*x.shape[:-1], n_groups, group_size)
    else:
        # Move target axis to the end
        perm = list(range(x.ndim))
        perm.pop(axis)
        perm.append(axis)
        x_permuted = x.permute(*perm)
        reshaped = x_permuted.reshape(*x_permuted.shape[:-1], n_groups, group_size)

    # Sort within each group (along the last dimension of the reshaped tensor)
    sorted_reshaped, _ = torch.sort(reshaped, dim=-1, descending=descending)

    if axis == -1 or axis == x.ndim - 1:
        # Restore original shape
        return sorted_reshaped.reshape_as(x)
    else:
        # Restore original axis ordering
        flat = sorted_reshaped.reshape_as(x_permuted)
        inv_perm = [0] * x.ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        return flat.permute(*inv_perm)


# ---------------------------------------------------------------------------
# Module API
# ---------------------------------------------------------------------------


class GroupedSort(nn.Module):
    """Grouped sorting activation as a reusable ``nn.Module``.

    Applies :func:`grouped_sort_1d` as a layer within a network. This is
    the standard nonlinearity used between Lipschitz-constrained linear
    layers to build provably 1-Lipschitz deep networks.

    Parameters
    ----------
    n_groups : int
        Number of groups to partition the feature dimension into.
        Must evenly divide the input feature size.
    axis : int
        Axis along which to sort (default: ``-1``).
    descending : bool
        If ``True``, sort each group in descending order.
    """

    def __init__(
        self,
        n_groups: int,
        axis: int = -1,
        descending: bool = False,
    ) -> None:
        super().__init__()
        if n_groups < 1:
            raise ValueError(f"n_groups must be >= 1, got {n_groups}.")
        self.n_groups = n_groups
        self.axis = axis
        self.descending = descending

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply grouped sorting to the input tensor."""
        return grouped_sort_1d(
            x, self.n_groups, self.axis, descending=self.descending
        )

    def extra_repr(self) -> str:
        return (
            f"n_groups={self.n_groups}, axis={self.axis}, "
            f"descending={self.descending}"
        )
