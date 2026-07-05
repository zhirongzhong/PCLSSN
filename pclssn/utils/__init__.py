"""Training Utilities and Evaluation Metrics.

This subpackage provides loss functions, evaluation metrics, and training
helpers for prognostic (RUL prediction) tasks.

Modules:
    metrics:   Loss functions (masked MSE), point-wise and trajectory-level metrics
    training:  Training and evaluation loop helpers, seed configuration
"""

from .metrics import (
    masked_mse_loss,
    compute_pointwise_metrics,
    compute_trajectory_metrics,
)
from .training import (
    seed_everything,
    train_epoch,
    evaluate_epoch,
)

__all__ = [
    # Metrics
    "masked_mse_loss",
    "compute_pointwise_metrics",
    "compute_trajectory_metrics",
    # Training
    "seed_everything",
    "train_epoch",
    "evaluate_epoch",
]
