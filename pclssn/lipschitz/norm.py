"""Weight Normalization for Lipschitz-Constrained Layers.

This module provides utilities for constraining the matrix norm of linear
layer weights, which is the core mechanism for enforcing Lipschitz continuity
in feedforward networks. By bounding the operator norm of each weight matrix,
we guarantee that the entire network has a bounded Lipschitz constant.

Supported norm types (applied column-wise or row-wise to the weight matrix):
    - "one":       L1 column norm  (||W||_1)
    - "inf":       L1 row norm     (||W||_inf)
    - "one-inf":   Element-wise abs clamping
    - "two":       L2 row norm     (||W||_2,inf)

Mathematical Background
----------------------
A function f is K-Lipschitz if ||f(x) - f(y)|| <= K * ||x - y|| for all x,y.
For linear layers f(x) = Wx + b, the Lipschitz constant equals the operator
norm of W. By normalizing ||W|| <= c after each gradient step, we bound the
layer's Lipschitz constant by c. The chain of such layers preserves the
product bound.

References
----------
The normalization techniques implemented here are based on well-established
convex optimization projections onto norm balls. The specific approach of
using parametrizations for differentiable normalization follows the general
pattern described in:
    Miyato et al., "Spectral Normalization for Generative Adversarial Networks"
    (ICLR 2018), generalized to L1/L2/Linf norms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

VALID_NORM_KINDS: tuple[str, ...] = (
    "one",       # ||W||_1  — max column-wise L1 norm
    "inf",       # ||W||_inf — max row-wise L1 norm
    "one-inf",   # element-wise L1,Linf — clamp each entry
    "two",       # ||W||_2,inf — max row-wise L2 norm
)

# ---------------------------------------------------------------------------
# Core normalization logic
# ---------------------------------------------------------------------------


def normalize_weights(
    weight: torch.Tensor,
    kind: str = "one",
    always_clamp: bool = False,
    max_norm: float | None = None,
    per_column: bool = True,
) -> torch.Tensor:
    """Project a weight matrix onto the specified norm ball.

    This is the shared computational core used by both the differentiable
    parametrization path (:func:`apply_weight_norm`) and the projection path
    (:func:`project_weight_norm`).

    Parameters
    ----------
    weight : torch.Tensor
        The weight matrix to normalize, shape ``(out_features, in_features)``.
    kind : str
        Norm type. Must be one of ``"one"``, ``"inf"``, ``"one-inf"``, ``"two"``.
    always_clamp : bool
        If ``True``, always scale weights to exactly ``max_norm``.
        If ``False``, only scale when the norm exceeds ``max_norm``.
    max_norm : float or None
        Target maximum norm. Defaults to ``1.0`` when ``None``.
    per_column : bool
        If ``True`` (default), constrain each column/row independently
        (vector-wise constraint, which is less restrictive).
        If ``False``, constrain the overall matrix norm.

    Returns
    -------
    torch.Tensor
        The normalized weight tensor, same shape as input.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the valid options.
    """
    if kind not in VALID_NORM_KINDS:
        raise ValueError(
            f"Unknown norm kind '{kind}'. Expected one of {VALID_NORM_KINDS}."
        )

    _max_norm = 1.0 if max_norm is None else float(max_norm)

    # ---- compute per-vector norms ----
    if kind == "one":
        # Column-wise L1: sum over rows for each column
        per_vector_norms = weight.abs().sum(dim=0)
    elif kind == "inf":
        # Row-wise L1: sum over columns for each row
        per_vector_norms = weight.abs().sum(dim=1, keepdim=True)
    elif kind == "one-inf":
        # Element-wise: treat each entry as its own "vector"
        per_vector_norms = weight.abs()
    elif kind == "two":
        # Row-wise L2
        per_vector_norms = torch.linalg.norm(weight, ord=2, dim=1, keepdim=True)
    else:
        # Defensive — should be unreachable after the check above
        raise ValueError(f"Unexpected kind: {kind}")

    # ---- optionally collapse to scalar norm ----
    if not per_column:
        per_vector_norms = per_vector_norms.max()

    # ---- compute scaling factors ----
    if always_clamp:
        # Always divide by max_norm (forces exact constraint satisfaction)
        scale_factors = per_vector_norms / _max_norm
    else:
        # Only scale entries whose norm exceeds max_norm
        scale_factors = torch.max(
            torch.ones_like(per_vector_norms),
            per_vector_norms / _max_norm,
        )

    # Prevent division by zero with a small epsilon floor
    safe_scale = torch.clamp(scale_factors, min=1e-10)

    return weight / safe_scale


# ---------------------------------------------------------------------------
# Differentiable parametrization (gradients flow through normalization)
# ---------------------------------------------------------------------------


def apply_weight_norm(
    module: nn.Linear,
    kind: str = "one",
    always_clamp: bool = False,
    max_norm: float | None = None,
    param_name: str = "weight",
    per_column: bool = True,
) -> nn.Linear:
    """Apply differentiable weight normalization to a linear layer.

    Registers a parametrization on the weight parameter so that every forward
    pass uses the normalized weight matrix. Gradients propagate through the
    normalization, allowing the optimizer to account for the constraint.

    This is the recommended method during **training** because gradients are
    computed with respect to the unconstrained parameters while the forward
    computation sees the constrained ones.

    Parameters
    ----------
    module : nn.Linear
        The linear layer whose weights should be constrained.
    kind : str
        Norm type for the constraint (see :func:`normalize_weights`).
    always_clamp : bool
        If ``True``, always normalize to ``max_norm`` exactly.
    max_norm : float or None
        Target maximum norm bound. Defaults to ``1.0``.
    param_name : str
        Name of the parameter to constrain (usually ``"weight"``).
    per_column : bool
        If ``True``, enforce the constraint per-column/per-row rather
        than on the full matrix.

    Returns
    -------
    nn.Linear
        The same module, with the parametrization registered in-place.
    """
    if kind not in VALID_NORM_KINDS:
        raise ValueError(
            f"Unknown norm kind '{kind}'. Expected one of {VALID_NORM_KINDS}."
        )

    # ---- Inner parametrization module ----
    class _WeightNormalizer(nn.Module):
        """Parametrization that normalizes weights on every forward pass."""

        def forward(self, w: torch.Tensor) -> torch.Tensor:
            return normalize_weights(
                weight=w,
                kind=kind,
                always_clamp=always_clamp,
                max_norm=max_norm,
                per_column=per_column,
            )

    register_parametrization(module, param_name, _WeightNormalizer())
    return module


# ---------------------------------------------------------------------------
# Projection-based normalization (no gradient through normalization)
# ---------------------------------------------------------------------------


def project_weight_norm(
    module: nn.Linear,
    kind: str = "one",
    always_clamp: bool = True,
    max_norm: float | None = None,
    param_name: str = "weight",
    per_column: bool = True,
) -> nn.Linear:
    """Apply weight normalization via hard projection (no gradient flow).

    Registers a forward pre-hook that projects the weight matrix onto the
    constraint set **in-place** before each forward pass. Unlike
    :func:`apply_weight_norm`, gradients do **not** flow through the
    normalization step — the optimizer sees the unprojected gradient.

    This method can be useful during **inference** or when the
    parametrization-based approach causes training instability.

    Parameters
    ----------
    module : nn.Linear
        The linear layer whose weights should be constrained.
    kind : str
        Norm type for the constraint (see :func:`normalize_weights`).
    always_clamp : bool
        If ``True``, always normalize to ``max_norm`` exactly.
    max_norm : float or None
        Target maximum norm bound. Defaults to ``1.0``.
    param_name : str
        Name of the parameter to constrain (usually ``"weight"``).
    per_column : bool
        If ``True``, enforce the constraint per-column/per-row.

    Returns
    -------
    nn.Linear
        The same module, with the pre-forward hook registered in-place.
    """
    if kind not in VALID_NORM_KINDS:
        raise ValueError(
            f"Unknown norm kind '{kind}'. Expected one of {VALID_NORM_KINDS}."
        )

    @torch.no_grad()
    def _project_weights(mod: nn.Linear, _input: object) -> None:
        """In-place weight projection executed before every forward pass."""
        raw_weight = getattr(mod, param_name).detach()
        projected = normalize_weights(
            weight=raw_weight,
            kind=kind,
            always_clamp=always_clamp,
            max_norm=max_norm,
            per_column=per_column,
        )
        getattr(mod, param_name).data.copy_(projected)

    module.register_forward_pre_hook(_project_weights)
    return module
