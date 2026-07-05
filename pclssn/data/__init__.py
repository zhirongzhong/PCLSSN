"""Data Loading and Processing Utilities.

This subpackage provides dataset classes and data-loading utilities
for the PHM2010 and CMAPSS prognostic datasets.

Modules:
    dataset:  PyTorch Dataset classes for loading .mat-format data files
    collate:  Custom collate functions for variable-length sequence batching
"""

from .dataset import Dataset
from .collate import pad_collate_fn

__all__ = [
    "Dataset",
    "pad_collate_fn",
]
