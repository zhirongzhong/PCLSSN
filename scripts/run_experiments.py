#!/usr/bin/env python3
"""Experiment Runner for PCLSSN on the Prognostics Dataset.

This script orchestrates multi-round training and evaluation of the PCLNN
model on the benchmark dataset.

Usage
-----
.. code-block:: bash

    python scripts/run_experiments.py

Configuration is managed via the ``ExperimentConfig`` dataclass at the
top of the ``main()`` function. Modify the values there to adjust
hyperparameters, dataset paths, and output directories.

Output
------
- ``checkpoints/`` — Saved model checkpoints (one per round)
- ``results/``    — Aggregated results in .npy format
- ``mat_results/`` — Per-round predictions saved as .mat files
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader
from tqdm import tqdm

# PCLSSN imports
from pclssn.models import PCLNN
from pclssn.data import Dataset, pad_collate_fn
from pclssn.utils.training import seed_everything, train_epoch, evaluate_epoch
from pclssn.utils.metrics import compute_trajectory_metrics


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class ExperimentConfig:
    """Configuration container for a full experiment run.

    Attributes
    ----------
    dataset_name : str
        Dataset identifier
    data_root : str
        Path to the directory containing .mat data files.
    subset : str
        Subset identifier within the dataset
    num_rounds : int
        Number of independent training runs with different seeds.
    num_epochs : int
        Maximum number of training epochs per round.
    batch_size : int
        Training batch size.
    learning_rate : float
        Initial learning rate for the AdamW optimizer.
    weight_decay : float
        Weight decay (L2 regularization) coefficient.
    rul_scale : float
        Normalization factor for RUL labels (raw RUL / scale).
    lambda_lipschitz : float
        Regularization weight for the Lipschitz constant penalty.
    base_seed : int
        Base random seed (incremented per round for diversity).
    scheduler_step : int
        Step size (in epochs) for the learning rate scheduler.
    scheduler_gamma : float
        Multiplicative decay factor for the scheduler.
    checkpoint_dir : str
        Directory for saving model checkpoints.
    results_dir : str
        Directory for saving aggregated results.
    mat_output_dir : str
        Directory for saving per-round .mat prediction files.
    model_kwargs : dict
        Keyword arguments forwarded to the model constructor.
    """

    dataset_name: str = "IGBT"
    data_root: str = "./dataset/IGBT"
    subset: str = "IGBT"
    num_rounds: int = 5
    num_epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    rul_scale: float = 1.0
    lambda_lipschitz: float = 1.0
    base_seed: int = 42
    scheduler_step: int = 50
    scheduler_gamma: float = 0.99
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    mat_output_dir: str = "mat_results"
    model_kwargs: dict = field(default_factory=lambda: {
        "input_dim": 7,
        "conv_features": 128,
        "motor_units": 32,
        "num_layers": 5,
        "initial_lipschitz": 1.0,
    })


# ===========================================================================
# Experiment Orchestration
# ===========================================================================


def setup_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create training, validation, and test data loaders.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, and test loaders.
    """
    train_set = Dataset(
        root=cfg.data_root,
        subset=cfg.subset,
        split="train",
        rul_scale=cfg.rul_scale,
    )
    val_set = Dataset(
        root=cfg.data_root,
        subset=cfg.subset,
        split="val",
        rul_scale=cfg.rul_scale,
    )
    test_set = Dataset(
        root=cfg.data_root,
        subset=cfg.subset,
        split="test",
        rul_scale=cfg.rul_scale,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=pad_collate_fn,
    )

    return train_loader, val_loader, test_loader


def train_one_round(
    cfg: ExperimentConfig,
    round_idx: int,
    model: PCLNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> str:
    """Train the model for one experimental round.

    Parameters
    ----------
    cfg : ExperimentConfig
        Experiment configuration.
    round_idx : int
        Current round index (0-based).
    model : PCLNN
        The model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    device : torch.device
        Target compute device.

    Returns
    -------
    str
        Path to the saved checkpoint for the best validation RMSE.
    """
    ckpt_path = os.path.join(
        cfg.checkpoint_dir,
        f"{cfg.dataset_name}_PCLNN_round{round_idx}.pth",
    )

    # Skip training if checkpoint already exists
    if os.path.exists(ckpt_path):
        print(f"  Checkpoint {ckpt_path} already exists — skipping training.")
        return ckpt_path

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.scheduler_step,
        gamma=cfg.scheduler_gamma,
    )

    best_val_rmse = float("inf")
    pbar = tqdm(range(cfg.num_epochs), desc=f"Round {round_idx}")

    for _ in pbar:
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            lambda_lipschitz=cfg.lambda_lipschitz,
        )
        val_loss, val_rmse = evaluate_epoch(model, val_loader, device)

        scheduler.step()

        # Denormalize RMSE for reporting
        val_rmse_scaled = val_rmse * cfg.rul_scale

        if val_rmse_scaled < best_val_rmse:
            best_val_rmse = val_rmse_scaled
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_rmse": best_val_rmse,
                    "round": round_idx,
                    "lipschitz": model.learnable_lipschitz.item(),
                },
                ckpt_path,
            )

        pbar.set_postfix({
            "Train": f"{train_loss:.4f}",
            "Val": f"{val_loss:.4f}",
            "RMSE": f"{val_rmse_scaled:.4f}",
            "Best": f"{best_val_rmse:.4f}",
        })

    return ckpt_path


def evaluate_round(
    model: PCLNN,
    test_loader: DataLoader,
    device: torch.device,
    rul_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, float]]:
    """Evaluate a trained model on the test set.

    Parameters
    ----------
    model : PCLNN
        Trained model.
    test_loader : DataLoader
        Test data loader.
    device : torch.device
        Target compute device.
    rul_scale : float
        RUL scaling factor for denormalization.

    Returns
    -------
    all_preds : list of np.ndarray
        Denormalized per-sample predictions.
    all_targets : list of np.ndarray
        Denormalized per-sample ground-truth values.
    avg_metrics : dict
        Aggregated evaluation metrics.
    """
    model.sync_lipschitz()
    model.eval()

    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device).squeeze(-1)

            predictions = model(features)

            for i in range(features.size(0)):
                mask = labels[i] > 0
                all_preds.append(
                    predictions[i][mask].cpu().numpy() * rul_scale
                )
                all_targets.append(
                    labels[i][mask].cpu().numpy() * rul_scale
                )

    # Compute per-trajectory metrics and average
    per_sample_metrics = [
        compute_trajectory_metrics(t, p)
        for p, t in zip(all_preds, all_targets)
    ]
    avg_metrics = {
        key: float(np.mean([m[key] for m in per_sample_metrics]))
        for key in per_sample_metrics[0].keys()
    }

    return all_preds, all_targets, avg_metrics


# ===========================================================================
# Main Entry Point
# ===========================================================================


def main() -> None:
    """Run the full multi-round experiment pipeline."""
    cfg = ExperimentConfig()

    # Determine compute device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.mat_output_dir, exist_ok=True)

    # Setup data
    train_loader, val_loader, test_loader = setup_dataloaders(cfg)

    # Accumulate results across rounds
    all_round_results: dict[int, dict[str, float]] = {}

    for round_idx in range(cfg.num_rounds):
        seed_everything(cfg.base_seed + round_idx)
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1}/{cfg.num_rounds}")
        print(f"{'='*60}")

        # Initialize fresh model
        model = PCLNN(**cfg.model_kwargs).to(device)

        # Train
        ckpt_path = train_one_round(
            cfg, round_idx, model, train_loader, val_loader, device,
        )

        # Load best checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"  Loaded checkpoint (best RMSE: {checkpoint['best_rmse']:.4f})")

        # Evaluate on test set
        all_preds, all_targets, avg_metrics = evaluate_round(
            model, test_loader, device, cfg.rul_scale,
        )

        all_round_results[round_idx] = avg_metrics
        print(
            f"  Test Results — "
            f"RMSE: {avg_metrics['RMSE']:.4f}, "
            f"MAE: {avg_metrics['MAE']:.4f}, "
            f"Mon: {avg_metrics['Mon']:.4f}, "
            f"Flu: {avg_metrics['Flu']:.2f}"
        )

        # Save per-round predictions as .mat
        mat_path = os.path.join(
            cfg.mat_output_dir,
            f"{cfg.dataset_name}_PCLNN_round{round_idx}_preds.mat",
        )
        savemat(
            mat_path,
            {
                "predictions": np.array(all_preds, dtype=object),
                "targets": np.array(all_targets, dtype=object),
                "metrics": avg_metrics,
                "info": (
                    f"Method: PCLNN, Round: {round_idx}, "
                    f"Data: {cfg.dataset_name}"
                ),
            },
        )
        print(f"  Saved predictions to: {mat_path}")

    # Save aggregated results
    results_path = os.path.join(
        cfg.results_dir,
        f"{cfg.dataset_name}_PCLNN_results.npy",
    )
    np.save(results_path, all_round_results)
    print(f"\nExperiment finished. Results saved to {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary Across All Rounds")
    print(f"{'='*60}")
    for metric_name in all_round_results[0].keys():
        values = [r[metric_name] for r in all_round_results.values()]
        print(
            f"  {metric_name}: "
            f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        )


if __name__ == "__main__":
    # Suppress deprecation warnings from external dependencies
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
