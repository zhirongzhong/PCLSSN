"""Loss Functions and Evaluation Metrics for RUL Prediction.

This module provides metrics designed for Remaining Useful Life (RUL)
prediction tasks, where the target is a monotonically decreasing
sequence and some positions may be masked (padded).

Metrics Summary
---------------
- ``masked_mse_loss`` : Mean squared error with sentinel-based masking.
- ``compute_pointwise_metrics`` : Point-wise RMSE, MAE, and Flu penalty.
- ``compute_trajectory_metrics`` : Per-trajectory RMSE, MAE, Monotonicity,
  and Flu score aggregation.
"""

from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss with automatic masking of padded positions.

    Masked positions are identified by ``target <= 0`` (since true RUL
    labels are always positive). Padding sentinels (e.g., ``-1``) are
    therefore excluded from the loss computation.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted RUL values, shape ``(...,)``.
    target : torch.Tensor
        Ground-truth RUL values, same shape as ``pred``.

    Returns
    -------
    torch.Tensor
        Scalar loss value (mean over valid positions).
    """
    mask = (target > 0).float()
    squared_error = (pred - target) ** 2
    masked_se = squared_error * mask
    n_valid = mask.sum() + 1e-8
    return masked_se.sum() / n_valid


# ---------------------------------------------------------------------------
# Point-wise Metrics
# ---------------------------------------------------------------------------


def compute_pointwise_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    flu_scale: float = 100.0,
) -> tuple[float, float, float]:
    """Compute point-wise error metrics between flattened predictions and targets.

    Parameters
    ----------
    pred : torch.Tensor
        1D tensor of predictions (already masked/filtered).
    target : torch.Tensor
        1D tensor of ground-truth values (same length as ``pred``).
    flu_scale : float
        Temperature parameter for the Flu (Fluctuation penalty) computation.
        Default: ``100.0``.

    Returns
    -------
    rmse : float
        Root Mean Squared Error.
    mae : float
        Mean Absolute Error.
    flu : float
        Accumulated Fluctuation penalty (sum over all samples).
    """
    delta = pred - target

    rmse = torch.sqrt(torch.mean(delta ** 2)).item()
    mae = torch.mean(torch.abs(delta)).item()

    # Flu: asymmetric exponential penalty
    # Penalizes over-estimation and under-estimation differently
    flu = torch.where(
        delta < 0,
        torch.exp(-delta / flu_scale) - 1.0,
        torch.exp(delta / flu_scale) - 1.0,
    ).sum().item()

    return rmse, mae, flu


# ---------------------------------------------------------------------------
# Trajectory-Level Aggregated Metrics
# ---------------------------------------------------------------------------


def compute_trajectory_metrics(
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
    flu_scale: float = 100.0,
) -> dict[str, float]:
    """Compute aggregated metrics across multiple degradation trajectories.

    For each trajectory, computes monotonicity (Mon) — a measure of how
    well the predicted RUL sequence respects the physical constraint of
    being non-increasing.

    Parameters
    ----------
    targets : list of np.ndarray
        Ground-truth RUL sequences, one array per test unit.
    predictions : list of np.ndarray
        Predicted RUL sequences, one array per test unit.
    flu_scale : float
        Temperature for Flu computation. Default: ``100.0``.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"RMSE"``, ``"MAE"``, ``"Mon"``, ``"Flu"``.

    Notes
    -----
    The trajectory-level Mon score measures the mean absolute deviation
    from perfect monotonic decrease, weighted by temporal distance:

        Mon_j = | Σ_{i<k} Δt * sign(ŷ_k - ŷ_i) | / Σ_{i<k} Δt

    A score of 1.0 indicates perfect monotonicity; lower values indicate
    violations of the physical degradation consistency constraint.
    """
    # Normalize to list-of-arrays
    if isinstance(targets, np.ndarray):
        targets = [targets]
    if isinstance(predictions, np.ndarray):
        predictions = [predictions]

    # ---- Point-wise metrics (pooled across all trajectories) ----
    flat_targets = torch.tensor(
        np.concatenate([np.atleast_1d(x) for x in targets])
    )
    flat_preds = torch.tensor(
        np.concatenate([np.atleast_1d(x) for x in predictions])
    )
    rmse, mae, flu = compute_pointwise_metrics(
        flat_preds, flat_targets, flu_scale=flu_scale
    )

    # ---- Per-trajectory monotonicity ----
    mon_scores: list[float] = []
    for pred_seq in predictions:
        pred_seq = np.atleast_1d(np.asarray(pred_seq))
        n_points = len(pred_seq)

        if n_points < 2:
            mon_scores.append(0.0)
            continue

        numerator = 0.0
        denominator = 0.0
        for i in range(n_points):
            for k in range(i + 1, n_points):
                delta_t = k - i
                sign = np.sign(pred_seq[k] - pred_seq[i])
                numerator += delta_t * sign
                denominator += delta_t

        mon_j = abs(numerator / (denominator + 1e-12))
        mon_scores.append(mon_j)

    avg_mon = float(np.mean(mon_scores)) if mon_scores else 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Mon": avg_mon,
        "Flu": flu,
    }
