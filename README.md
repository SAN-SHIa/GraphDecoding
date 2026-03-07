# Graph Neural Networks can Recover the Hidden Features Solely from the Graph Structure (ICML 2023)

[![arXiv](https://img.shields.io/badge/arXiv-2301.10956-b31b1b.svg)](https://arxiv.org/abs/2301.10956)

This repository is an improved and modularized implementation inspired by the paper:

**Graph Neural Networks can Recover the Hidden Features Solely from the Graph Structure**  
Ryoma Sato, ICML 2023  
Paper: https://arxiv.org/abs/2301.10956

---

## 🚀 SimpleScale

**SimpleScale** is the core model in this repo (`GraphSAGE_SimpleScale`).

<img src="./imgs/simple_scale.png" alt="SimpleScale" />

- **Input**: trivial/random node features + structural descriptors (e.g., PageRank, in-degree, clustering coefficient).
- **Architecture**: GraphSAGE-based message passing with density-aware scaling.
- **Objective**: reconstruct hidden geometry/manifold from pure graph structure.

In this codebase:
- `src/main.py` runs the main 2D manifold decoding experiment.
- `tools/semi_adult.py` runs the Adult dataset decoding experiment.
- `tools/feature_analysis.py` analyzes and compares structural features.

---

## 💿 Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails due to CUDA/GPU environment issues, install a suitable build from the official website:
https://pytorch.org/

---

## ⚙️ Quick Start

> Recommended from repository root. Use `PYTHONPATH=src` so scripts can import `utils` correctly.

### 1) Main experiment (moon and other synthetic datasets)

```bash
PYTHONPATH=src python src/main.py
```

Use config file:

```bash
PYTHONPATH=src python src/main.py --config configs/default.yaml
```

Override key args from CLI:

```bash
PYTHONPATH=src python src/main.py --dataset spiral --n 3000 --m 300
```

### 2) Adult experiment

```bash
PYTHONPATH=src python tools/semi_adult.py
```

### 3) Feature analysis tool

Quick mode:

```bash
PYTHONPATH=src python tools/feature_analysis.py --dataset moon --n 1000 --m 100
```

Full search mode (slower):

```bash
PYTHONPATH=src python tools/feature_analysis.py --dataset moon --n 1000 --m 100 --full_search
```

---

## 🗃️ Dataset Notes

- `src/adult.data` is used by `tools/semi_adult.py`.
- If you want to refresh/download the Adult dataset manually:

```bash
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O src/adult.data
```

---

## 🧪 Configuration

Main experiment reads `configs/default.yaml`:

```yaml
task:
  dataset: moon
  n: 5000
  m: 500
  train_ratio: 0.7

graph:
  knn_k_divisor: 10.0
  eball_percentile: 5.0
  eball_scaling_factor: 2.7

training:
  epochs: 100
  lr: 0.002

runtime:
  seed: 0

visualization:
  keep_percentile: 97.0
  fallback_keep_percentile: 95.0
  min_keep_ratio: 0.5
```

---

## 📂 Repository Structure

```text
new-version/
├── configs/
│   └── default.yaml
├── src/
│   ├── main.py
│   ├── adult.data
│   └── utils/
│       ├── model.py
│       ├── logging.py
│       ├── datasets.py
│       ├── visualization.py
│       └── __init__.py
├── tools/
│   ├── semi_adult.py
│   └── feature_analysis.py
├── imgs/
├── logs/
├── visualize/
├── requirements.txt
└── README.md
```

---

## 📈 Outputs

- Logs: `logs/*.log`
- Reconstruction figures: `visualize/*.png`
- Feature-correlation plots: `visualize/feature_correlation_*.png`

---

## 🖋️ Citation

```bibtex
@inproceedings{sato2023graph,
  author    = {Ryoma Sato},
  title     = {Graph Neural Networks can Recover the Hidden Features Solely from the Graph Structure},
  booktitle = {International Conference on Machine Learning, {ICML}},
  year      = {2023},
}
```

---

## License

This project follows the license in `LICENSE`.
