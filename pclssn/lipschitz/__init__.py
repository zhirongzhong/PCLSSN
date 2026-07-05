"""Lipschitz-Constrained Neural Network Layers.

This subpackage provides building blocks for constructing neural networks
with guaranteed Lipschitz continuity and monotonicity properties. These
constraints are architecturally enforced rather than learned, ensuring
provable stability and consistency guarantees.

Modules:
    norm:       Weight normalization utilities for Lipschitz constraints
    layers:     Lipschitz-constrained and monotonic linear layers
    monotonic:  Wrapper for combining Lipschitz and monotonicity constraints
    activation: Grouped sorting activation for gradient-preserving nonlinearity
"""

from .norm import apply_weight_norm, project_weight_norm, normalize_weights
from .layers import (
    LipschitzConstrainedLinear,
    MonotonicLinear,
    RootMeanSquareNorm,
)
from .monotonic import MonotonicWrapper
from .activation import GroupedSort, grouped_sort_1d

__all__ = [
    # Weight normalization
    "apply_weight_norm",
    "project_weight_norm",
    "normalize_weights",
    # Layers
    "LipschitzConstrainedLinear",
    "MonotonicLinear",
    "RootMeanSquareNorm",
    # Monotonic wrapper
    "MonotonicWrapper",
    # Activations
    "GroupedSort",
    "grouped_sort_1d",
]
