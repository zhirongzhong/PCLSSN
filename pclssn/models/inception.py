"""Multi-Scale 1D Inception Block for Time-Series Feature Extraction.

This module implements a 1D variant of the Inception architecture adapted
for multivariate time-series processing. By applying convolutional filters
at multiple receptive field sizes (1, 3, 5, and max-pool) in parallel, the
block captures both fine-grained local patterns and broader temporal context
within a single layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """Multi-branch 1D convolutional block with Inception-style parallelism.

    Four parallel branches process the input at different temporal scales:
    1. 1×1 convolution (point-wise)
    2. 1×1 → 3×1 convolution (local context)
    3. 1×1 → 5×1 convolution (broader context)
    4. MaxPool → 1×1 convolution (downsampled context)

    All branches produce ``out_channels // 4`` channels, which are
    concatenated to form the final ``out_channels``-dimensional output.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels. Must be divisible by 4.

    Notes
    -----
    A BatchNorm + ReLU pair is applied after concatenation to stabilize
    training and introduce nonlinearity.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by 4."
            )
        branch_channels = out_channels // 4

        # Branch 1: point-wise (1x1 conv)
        self.branch_1x1 = nn.Conv1d(in_channels, branch_channels, kernel_size=1)

        # Branch 2: 1x1 → 3x1 conv (bottleneck + local context)
        self.branch_3x1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=3, padding=1),
        )

        # Branch 3: 1x1 → 5x1 conv (bottleneck + broader context)
        self.branch_5x1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.Conv1d(branch_channels, branch_channels, kernel_size=5, padding=2),
        )

        # Branch 4: max-pool → 1x1 conv (global context)
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
        )

        # Post-concatenation normalization
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Inception block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, in_channels, seq_len)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, out_channels, seq_len)``.
        """
        out = torch.cat(
            [
                self.branch_1x1(x),
                self.branch_3x1(x),
                self.branch_5x1(x),
                self.branch_pool(x),
            ],
            dim=1,
        )
        return F.relu(self.norm(out))
