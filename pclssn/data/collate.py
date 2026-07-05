"""Custom Collate Functions for Variable-Length Sequence Batching.

The PHM2010 dataset contains samples with varying numbers of cycles and
varying sequence lengths. Standard PyTorch batching cannot handle this
variability directly, so these collate functions pad samples to uniform
dimensions within each batch.
"""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate and pad a batch of variable-length sequence samples.

    Pads both features and labels to the maximum cycle count in the batch
    using zero-padding for features and ``-1`` for labels (since RUL is
    always non-negative, ``-1`` serves as a mask sentinel).

    Parameters
    ----------
    batch : list of (features, labels) tuples
        Each ``features`` tensor has shape ``(cycles_i, seq_len, features)``.
        Each ``labels`` tensor has shape ``(cycles_i, 1)``.

    Returns
    -------
    features_padded : torch.Tensor
        Shape ``(batch_size, max_cycles, seq_len, features)``.
    labels_padded : torch.Tensor
        Shape ``(batch_size, max_cycles, 1)`` with ``-1`` padding.
    """
    features_list, labels_list = zip(*batch)

    # Pad features to max cycles in batch
    features_padded = pad_sequence(
        features_list, batch_first=True, padding_value=0.0
    )

    # Pad labels; -1 is used as a mask sentinel (RUL is always >= 0)
    labels_padded = pad_sequence(
        labels_list, batch_first=True, padding_value=-1.0
    )

    return features_padded, labels_padded
