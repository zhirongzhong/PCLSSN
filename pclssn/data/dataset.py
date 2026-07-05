"""PyTorch Dataset for the Prognostics Dataset.

Each sample is a 3D tensor of shape ``(num_cycles, seq_len
, num_features)`` representing the multi-cycle degradation
trajectory of a single tool.

Data Format
-----------
The .mat files are expected to contain:
    X : (num_samples, 1) object array of cell arrays.
        Each cell contains a ``(cycles, seq_len, features)`` numeric array.
    Y : (num_samples, 1) object array.
        Each cell contains a ``(cycles,)`` or ``(cycles, 1)`` array of
        RUL (Remaining Useful Life) labels.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class Dataset(Dataset):
    """Dataset loader for PHM2010-format .mat files.

    Loads multi-cycle time-series data where each sample is a cell array
    containing a ``(cycles, seq_len, features)`` tensor and corresponding
    ``(cycles,)`` RUL labels.

    Parameters
    ----------
    root : str
        Path to the directory containing the .mat data files.
    subset : str
        Subset identifier (e.g., ``"c1"``, ``"c4"``) used to construct
        the filename pattern ``{split}_{subset}.mat``.
    split : str
        Which data split to load. Must be one of:
        - ``"train"`` — training set
        - ``"val"``   — validation set
        - ``"test"``  — test set
    rul_scale : float
        Scaling factor applied to RUL labels. Raw labels are divided by
        this value. Default: ``1.0`` (no scaling).

    Raises
    ------
    FileNotFoundError
        If the constructed file path does not exist.

    Notes
    -----
    RUL labels are stored as 2D tensors of shape ``(cycles, 1)`` for
    compatibility with masked loss functions during training.
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root: str,
        subset: str = "c1",
        split: str = "train",
        rul_scale: float = 1.0,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. Expected one of {self.VALID_SPLITS}."
            )

        file_name = f"{split}_{subset}.mat"
        full_path = os.path.join(root, file_name)

        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Dataset file not found: {full_path}"
            )

        # Load MATLAB v5/v7 file
        mat_data = loadmat(full_path)

        # Extract feature cell array X and label cell array Y
        X_raw = mat_data["X"]  # (N_samples, 1) object array
        Y_raw = mat_data["Y"]  # (N_samples, 1) object array

        self.features: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

        for i in range(X_raw.shape[0]):
            # ---- Parse features ----
            seq = torch.from_numpy(X_raw[i, 0]).float()
            num_cycles = seq.shape[0]

            # ---- Parse labels ----
            y_np = Y_raw[i, 0]
            # Handle uint16 arrays that cannot be directly converted to float
            if y_np.dtype == np.uint16:
                y_np = y_np.astype(np.int32)

            rul = torch.from_numpy(y_np).float() / rul_scale

            # Ensure labels are 2D: (cycles, 1)
            if rul.ndim == 1:
                rul = rul.unsqueeze(-1)

            # ---- Validate cycle-count alignment ----
            if seq.shape[0] != rul.shape[0]:
                # Attempt transpose fix for mis-shaped labels
                if rul.shape[1] == num_cycles:
                    rul = rul.t()
                else:
                    print(
                        f"Warning: Sample {i} has mismatched cycle counts "
                        f"(features: {seq.shape[0]}, labels: {rul.shape[0]})."
                    )

            self.features.append(seq)
            self.labels.append(rul)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A ``(features, labels)`` pair where features has shape
            ``(cycles, seq_len, features)`` and labels has shape
            ``(cycles, 1)``.
        """
        return self.features[idx], self.labels[idx]
