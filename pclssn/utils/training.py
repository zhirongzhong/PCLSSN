"""Training and Evaluation Loop Helpers.

Provides reusable training and evaluation routines for PCLNN models,
including random seed configuration, the training step, and the
evaluation (validation/test) step.
"""

from __future__ import annotations

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import masked_mse_loss, compute_pointwise_metrics


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducible experiments.

    Configures Python's ``random``, NumPy, PyTorch, and CUDA random number
    generators. Also enables deterministic CUDA behavior at the cost of
    some performance.

    Parameters
    ----------
    seed : int
        Seed value for all random number generators. Default: ``42``.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_lipschitz: float = 0.01,
    max_grad_norm: float = 1.0,
) -> float:
    """Run a single training epoch.

    Parameters
    ----------
    model : nn.Module
        The PCLNN model in training mode.
    dataloader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer instance.
    device : torch.device
        Target device (``"cuda"`` or ``"cpu"``).
    lambda_lipschitz : float
        Regularization weight for the Lipschitz constant penalty.
        Default: ``0.01``.
    max_grad_norm : float
        Maximum gradient norm for clipping. Default: ``1.0``.

    Returns
    -------
    float
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).squeeze(-1)

        optimizer.zero_grad()

        predictions = model(features)

        # Primary loss: masked MSE over valid (non-padding) positions
        mse = masked_mse_loss(predictions, labels)

        # Regularization: penalize large Lipschitz constants
        lip_reg = lambda_lipschitz * (model.learnable_lipschitz ** 2)

        loss = mse + lip_reg
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * features.size(0)

    return total_loss / len(dataloader.dataset)


# ---------------------------------------------------------------------------
# Evaluation Loop
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Run a single evaluation epoch.

    Parameters
    ----------
    model : nn.Module
        The PCLNN model in evaluation mode.
    dataloader : DataLoader
        Validation or test data loader.
    device : torch.device
        Target device.

    Returns
    -------
    avg_loss : float
        Average masked MSE loss over the dataset.
    rmse : float
        Root Mean Squared Error computed over all valid predictions.
    """
    model.sync_lipschitz()
    model.eval()

    total_loss = 0.0
    all_predictions: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).squeeze(-1)

        predictions = model(features)

        total_loss += (
            masked_mse_loss(predictions, labels).item() * features.size(0)
        )

        # Collect valid (masked) predictions per sample
        for i in range(features.size(0)):
            mask = labels[i] > 0
            all_predictions.append(predictions[i][mask])
            all_targets.append(labels[i][mask])

    avg_loss = total_loss / len(dataloader.dataset)

    # Compute scalar RMSE across all valid predictions
    if all_predictions:
        rmse, _, _ = compute_pointwise_metrics(
            torch.cat(all_predictions), torch.cat(all_targets)
        )
    else:
        rmse = float("nan")

    return avg_loss, rmse
