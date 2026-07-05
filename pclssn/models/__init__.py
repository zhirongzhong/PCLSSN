"""PCLSSN Model Architectures.

This subpackage contains the complete model architectures for the
Physics-Consistent Liquid State-Space Network (PCLSSN), organized
into reusable building blocks and task-specific network definitions.

Modules:
    inception:     Multi-scale 1D Inception blocks for local feature extraction
    ssm:           Parallel-scan State-Space Model (SSM) for sequence modeling
    pclnn_cmapss:  PCLNN architecture for the CMAPSS turbofan dataset
    pclnn_phm2010: PCLNN architecture for the PHM2010 battery dataset
"""

from .inception import InceptionBlock
from .ssm import ParallelScanSSM
from .pclnn import PCLNN

__all__ = [
    "InceptionBlock",
    "ParallelScanSSM",
    "PCLNN",
]
