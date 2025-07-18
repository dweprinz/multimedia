# HIVE: Hyperbolic Interactive Visualization Explorer

<p align="center">
  <a href="https://openreview.net/pdf?id=D9LlujFg7d" target="_blank"><img src="https://img.shields.io/badge/View%20Paper-OpenReview-blue" alt="View Paper"></a>
  <a href="#demo"><img src="https://img.shields.io/badge/Demo-Dashboard-green" alt="Demo"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Cite%20Us-arXiv%20preprint-orange" alt="Cite"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License"></a>
  <a href="https://github.com/thijmen/multimedia/issues"><img src="https://img.shields.io/badge/Issues-Report%20Issue-red" alt="Issues"></a>
</p>

---

## Overview

**HIVE** is an interactive dashboard for visualizing and exploring hierarchical and hyperbolic representations of data. The dashboard is the main contribution of this repository and is designed to be extensible: while we currently provide the HyCoCLIP model and the GRIT and ImageNet datasets as examples, **any model and dataset combination can be added by forking this repository**.

---

## [View Paper](https://openreview.net/pdf?id=D9LlujFg7d)

If you use this dashboard or its ideas in your research, please consider citing our paper!

---

## Features
- Interactive visualization of hierarchical and hyperbolic embeddings
- Support for multiple datasets (GRIT, ImageNet, and more via extension)
- Compare different projection methods (e.g., HoroPCA, CO-SNE)
- Dual-view and single-view modes
- Tree, neighbor, and interpolation exploration modes
- Extensible to new models and datasets

---

## Setup Instructions

### Prerequisites
First, install uv if you haven't already:
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Running the Dashboard

```bash
# 1. Create a virtual environment
uv venv

# 2. Activate the virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# 3. Install dependencies
uv sync

# 4. Run the dashboard
uv run src/main.py
```

The dashboard will be available at `http://localhost:8081`

---

## Usage
- Select a dataset (e.g., GRIT or ImageNet) from the dropdown.
- Choose a projection method (HoroPCA, CO-SNE, etc.).
- Explore the data using the interactive plot:
  - **Compare:** Select up to 5 points to compare.
  - **Interpolate:** Select 2 points to interpolate between.
  - **Tree:** Select a point to view its lineage.
  - **Neighbors:** Select a point to view its neighbors.
- Use the dual-view mode to compare projections side-by-side.

---

## Adding New Models or Datasets
To add your own model or dataset:
1. Fork this repository.
2. Follow the structure of the provided examples (see `hierchical_datasets/`).
3. Add your data and update the dashboard configuration as needed.
4. Submit a pull request if you think your extension would benefit the community!

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, new features, or dataset/model integrations.

---

## Citation
If you use HIVE or its dashboard in your research, please cite:

```
@inproceedings{nijdamhive,
  title={HIVE: A Hyperbolic Interactive Visualization Explorer for Representation Learning},
  author={Nijdam, Thijmen and Prinzhorn, Derck WE and de Heus, Jurgen and Brouwer, Thomas},
  booktitle={2nd Beyond Euclidean Workshop: Hyperbolic and Hyperspherical Learning for Computer Vision}
}
```

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
