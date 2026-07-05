# PCLSSN: Physics-Consistent Liquid State-Space Network

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official implementation of Physics-Consistent Liquid State-Space Network for Remaining Useful Life Prediction with Architecturally Enforced Degradation Consistency, published in *Information Fusion* (2026).

> **Zhirong Zhong, Yinuo Yue, Zhongyi Zhang, Zhi Zhai, Meng Ma, Jinxin Liu**
>
> *Information Fusion*, 2026. [DOI: 10.1016/j.inffus.2026.104597](https://doi.org/10.1016/j.inffus.2026.104597)

## Overview

PCLSSN is a deep learning method for Remaining Useful Life (RUL) prediction that architecturally enforces physical consistency constraints in neural networks.

## Installation

### Prerequisites

- Python 3.9 or later
- PyTorch 2.0 or later (install the version matching your CUDA setup from [pytorch.org](https://pytorch.org/get-started/locally/))

### Install from source

```bash
git clone https://github.com/username/pclssn.git
cd pclssn
pip install -e .
```

### Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the full PHM2010 experiment pipeline
python scripts/run_experiments.py
```

Configuration is managed via the `ExperimentConfig` dataclass in `scripts/run_experiments.py`. Modify parameters there to adjust hyperparameters, dataset paths, or the number of experimental rounds.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{ZHONG2026104597,
    title = {Physics-Consistent Liquid State-Space Network for Remaining Useful Life Prediction with Architecturally Enforced Degradation Consistency},
    journal = {Information Fusion},
    pages = {104597},
    year = {2026},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2026.104597},
    author = {Zhirong Zhong and Yinuo Yue and Zhongyi Zhang and Zhi Zhai and Meng Ma and Jinxin Liu},
}
```

## License

This project is released under the [MIT License](LICENSE).
