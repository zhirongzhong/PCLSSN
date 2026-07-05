"""PCLNN Architecture for the PHM2010 Battery Degradation Dataset.

This module defines the Physics-Consistent Liquid Neural Network (PCLNN)
tailored for the Prognostics dataset.

Architecture Overview
---------------------
1. **Intra-Cycle Encoder** (CNN + SSM):
   Processes sensor readings within each cycle to extract a compact
   cycle-level feature representation.

2. **Monotonic Bottleneck** (MonotonicWrapper + GroupedSort):
   Maps cycle features to a monotonically evolving latent state.

3. **Inter-Cycle Dynamics** (CfC Liquid Layers):
   Models the long-term degradation trajectory across cycles.

4. **Monotonic Readout Head**:
   Produces the final RUL prediction with enforced monotonic decrease.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch.utils.checkpoint import checkpoint

from pclssn.lipschitz import (
    LipschitzConstrainedLinear,
    MonotonicWrapper,
    GroupedSort,
)
from .inception import InceptionBlock
from .ssm import ParallelScanSSM


class PCLNN(nn.Module):
    """Physics-Consistent Liquid Neural Network for the PHM2010 dataset.

    Handles 3D inputs with an explicit cycle dimension, processing intra-cycle
    sensor sequences and inter-cycle degradation patterns separately.

    Parameters
    ----------
    input_dim : int
        Number of sensor channels per time step. Default: ``7``.
    conv_features : int
        Hidden channels in the convolutional encoder. Default: ``128``.
    motor_units : int
        Dimensionality of the CfC liquid hidden state. Default: ``32``.
    num_layers : int
        Number of stacked CfC liquid layers. Default: ``5``.
    initial_lipschitz : float
        Initial value for the learnable Lipschitz constant. Default: ``1.0``.
    """

    def __init__(
        self,
        input_dim: int = 7,
        conv_features: int = 128,
        motor_units: int = 32,
        num_layers: int = 5,
        initial_lipschitz: float = 1.0,
    ) -> None:
        super().__init__()

        # ---- Learnable Lipschitz constant ----
        self.learnable_lipschitz = nn.Parameter(
            torch.tensor(float(initial_lipschitz))
        )
        self.motor_units = motor_units
        self.conv_features = conv_features

        # ============================================================
        # Stage 1: Intra-Cycle Encoder
        #   Processes (Seq_len × Features) within each cycle
        # ============================================================
        self.cnn_init = nn.Conv1d(input_dim, conv_features, kernel_size=1)
        self.cnn_norm = nn.LayerNorm(conv_features)

        self.inception_blocks = nn.ModuleList([
            InceptionBlock(conv_features, conv_features) for _ in range(2)
        ])

        # Models dependencies among sampling points within a cycle
        self.intra_ssm = ParallelScanSSM(
            d_model=conv_features,
            lipschitz_bound=self.learnable_lipschitz,
        )

        # Compress the temporal (Seq_len) dimension to a single vector per cycle
        self.cycle_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # ============================================================
        # Stage 2: Monotonic Bottleneck (SSM features → Liquid state)
        #   Forces physical features to map to monotonically evolving
        #   degradation states
        # ============================================================
        self.intermediate_proj = nn.Sequential(
            LipschitzConstrainedLinear(
                conv_features, motor_units,
                lipschitz_bound=self.learnable_lipschitz, norm_kind="one",
            ),
            GroupedSort(n_groups=2),
            LipschitzConstrainedLinear(
                motor_units, motor_units,
                lipschitz_bound=self.learnable_lipschitz, norm_kind="one",
            ),
            GroupedSort(n_groups=2),
        )
        # Negative signs encode the expectation that RUL decreases as
        # degradation features increase (e.g., internal resistance rises,
        # capacity falls)
        self.intermediate_monotonic = MonotonicWrapper(
            self.intermediate_proj,
            lipschitz_bound=self.learnable_lipschitz,
            monotonic_signs=[-1] * conv_features,
        )

        # ============================================================
        # Stage 3: Inter-Cycle Liquid Dynamics (CfC)
        #   Models the long-range degradation trajectory across cycles
        # ============================================================
        self.liquid_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == num_layers - 1:
                wiring = AutoNCP(motor_units * 2, motor_units)
                layer = CfC(motor_units, wiring, batch_first=True)
            else:
                layer = CfC(motor_units, motor_units, batch_first=True)
            self.liquid_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(motor_units))

        # ============================================================
        # Stage 4: Monotonic Readout Head
        #   Projects hidden states to scalar RUL with enforced monotonicity
        # ============================================================
        self.readout = nn.Sequential(
            LipschitzConstrainedLinear(
                motor_units, 16,
                lipschitz_bound=self.learnable_lipschitz, norm_kind="one",
            ),
            GroupedSort(n_groups=2),
            LipschitzConstrainedLinear(
                16, 16,
                lipschitz_bound=self.learnable_lipschitz, norm_kind="one",
            ),
            GroupedSort(n_groups=2),
            LipschitzConstrainedLinear(
                16, 1,
                lipschitz_bound=self.learnable_lipschitz, norm_kind="one",
            ),
        )
        self.head = MonotonicWrapper(
            self.readout,
            lipschitz_bound=self.learnable_lipschitz,
            monotonic_signs=[-1],
        )

    def sync_lipschitz(self, value: float | torch.Tensor | None = None) -> None:
        """Synchronize the Lipschitz constant across all constrained sub-modules.

        Must be called after loading a checkpoint or before inference.

        Parameters
        ----------
        value : float, torch.Tensor, or None
            Target Lipschitz value. Uses current learned value if ``None``.
        """
        if isinstance(value, torch.Tensor):
            target = value
        elif value is not None:
            target = torch.tensor(float(value), device=self.learnable_lipschitz.device)
        else:
            target = self.learnable_lipschitz.detach()

        # Synchronize intra-cycle SSM (updates inner constrained layers)
        self.intra_ssm.sync_lipschitz(target)

        # Synchronize intermediate monotonic projection
        for layer in self.intermediate_proj:
            if hasattr(layer, 'lipschitz_bound'):
                layer.lipschitz_bound.data.copy_(target)
        self.intermediate_monotonic.lipschitz_bound.data.copy_(target)

        # Synchronize readout head
        for layer in self.readout:
            if hasattr(layer, 'lipschitz_bound'):
                layer.lipschitz_bound.data.copy_(target)
        self.head.lipschitz_bound.data.copy_(target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for PHM2010 RUL prediction.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, cycles, seq_len, features)``.

        Returns
        -------
        torch.Tensor
            Predicted RUL values of shape ``(batch, cycles)``.
        """
        B_sz, C_sz, L_sz, F_dim = x.shape

        # ---- 1. Intra-Cycle Encoding ----
        # Merge batch and cycle dims for parallel processing
        x_flat = x.view(B_sz * C_sz, L_sz, F_dim)
        x_intra = x_flat.transpose(1, 2)  # → (B*C, F_dim, L)

        x_intra = self.cnn_init(x_intra)
        for block in self.inception_blocks:
            x_intra = block(x_intra) + x_intra  # Residual

        x_intra = x_intra.transpose(1, 2)  # → (B*C, L, conv_features)
        x_intra = self.cnn_norm(x_intra)

        # SSM for intra-cycle dependencies
        x_ssm = self.intra_ssm(x_intra)

        # Pool the time dimension to get a single vector per cycle
        z_cycle = self.cycle_pool(x_ssm.transpose(1, 2)).squeeze(-1)  # (B*C, C_feat)

        # ---- 2. Monotonic Bottleneck ----
        current_out = self.intermediate_monotonic(z_cycle)
        current_out = current_out.view(B_sz, C_sz, self.motor_units)

        # ---- 3. Inter-Cycle Liquid Dynamics ----
        for i in range(len(self.liquid_layers)):
            if self.training:
                def _cfc_forward(inp: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
                    out, _ = self.liquid_layers[idx.item()](inp)
                    return out

                layer_out = checkpoint(
                    _cfc_forward, current_out, torch.tensor(i),
                    use_reentrant=False,
                )
            else:
                layer_out, _ = self.liquid_layers[i](current_out)

            current_out = self.layer_norms[i](layer_out + current_out)

        # ---- 4. Monotonic Readout ----
        z_final = current_out.reshape(-1, self.motor_units)
        y = self.head(z_final)

        return y.view(B_sz, C_sz)
