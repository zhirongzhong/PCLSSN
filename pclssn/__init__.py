"""PCLSSN: Physics-Consistent Liquid State-Space Network.

PCLSSN is a deep learning framework for Remaining Useful Life (RUL)
prediction with architecturally enforced physical consistency constraints.
The network combines:

- **Lipschitz-constrained layers** for provable stability
- **Monotonicity constraints** for degradation consistency
- **Parallel-scan state-space models** for efficient sequence processing
- **Liquid time-constant networks** (CfC) for temporal dynamics modeling

Key Features
------------
* Guaranteed monotonic degradation predictions
* Learnable Lipschitz constants
* Built-in evaluation metrics for prognostics
* Support for CMAPSS (turbofan) and PHM2010 (battery/milling) datasets

Reference
---------
Zhong, Z., Yue, Y., Zhang, Z., Zhai, Z., Ma, M., & Liu, J. (2026).
"Physics-Consistent Liquid State-Space Network for Remaining Useful Life
Prediction with Architecturally Enforced Degradation Consistency."
*Information Fusion*, 104597.
"""

from . import lipschitz
from . import models
from . import data
from . import utils
from .models import PCLNN

__version__ = "1.0.0"

__all__ = [
    "lipschitz",
    "models",
    "data",
    "utils",
    "PCLNN",
]
