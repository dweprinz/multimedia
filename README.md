# HIVE: Hyperbolic Interactive Visualization Explorer

<p align="center">
  <a href="https://openreview.net/pdf?id=D9LlujFg7d" target="_blank"><img src="https://img.shields.io/badge/View%20Paper-OpenReview-blue" alt="View Paper"></a>
  <a href="./HIVE_demo.mp4"><img src="https://img.shields.io/badge/Demo-Video-green" alt="Demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License"></a>
  <a href="https://github.com/thijmennijdam/HIVE/issues"><img src="https://img.shields.io/badge/Issues-Report%20Issue-red" alt="Issues"></a>
</p>

<video src="./HIVE_demo.mp4" controls width="600" style="display:block;margin:2em auto;max-width:100%;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
  Your browser does not support the video tag. <a href="./HIVE_demo.mp4">Watch the demo video here.</a>
</video>

## Overview

**HIVE** is an interactive dashboard for visualizing and exploring hierarchical and hyperbolic representations of data. The dashboard is the main contribution of this repository and is designed to be extensible: while we currently provide the HyCoCLIP model and the GRIT and ImageNet datasets as examples, **any model and dataset combination can be added by following the modular pipeline below**.

---

## Features
- Interactive visualization of hierarchical and hyperbolic embeddings
- Support for multiple datasets (GRIT, ImageNet, and more via extension)
- Compare different projection methods (HoroPCA, CO-SNE)
- Dual-view and single-view modes
- Tree, neighbor, and interpolation exploration modes
- Extensible to new models and datasets

---

## How the data was processed, and how to add your own

To visualize your data in HIVE, follow this modular pipeline:

1. **Tree Structure Creation** (`create_tree_structure.py`)
   - **Purpose:** Organizes your raw dataset (e.g., ImageNet, GRIT) into a hierarchical tree structure with metadata.
   - **Input:** Raw data (images, text, etc.) and any required metadata (e.g., synsets.csv for ImageNet).
   - **Output:**
     - Tree folders (e.g., `tree1`, `tree2`, ...)
     - `meta_data_trees.json` (describes the tree structure and metadata)
   - **Example:**
     ```bash
     python hierchical_datasets/create_tree_structure.py --dataset ImageNet --synsets_csv <PATH_TO_SYNSETS_CSV> --base_path <PATH_TO_RAW_IMAGES> --output_dir hierchical_datasets
     ```

2. **Embedding Generation** (`generate_embeddings.py`)
   - **Purpose:** Loads a model (e.g., HyCoCLIP) and generates embeddings for each node (image/text) in the tree structure.
   - **Input:** The tree structure and metadata created in step 1, plus your model checkpoint/config.
   - **Output:**
     - `embeddings.pkl` (or similar) containing all generated embeddings
   - **Example:**
     ```bash
     python hierchical_datasets/generate_embeddings.py --dataset ImageNet --checkpoint_path <MODEL_CHECKPOINT> --train_config <CONFIG_PATH> --output_dir hierchical_datasets
     ```

3. **Projection Creation** (`create_projections.py`)
   - **Purpose:** Applies dimensionality reduction (HoroPCA, CO-SNE) to the embeddings for visualization.
   - **Input:** Embeddings file (`embeddings.pkl`) and tree metadata (`meta_data_trees.json`).
   - **Output:** 2D projections for visualization in the dashboard (e.g., `horopca_embeddings.pkl`, `cosne_embeddings.pkl`).
   - **Example:**
     ```bash
     python projection_methods/create_projections.py --dataset-path hierchical_datasets/ImageNet --methods horopca cosne
     ```

4. **Visualization** (Dashboard)
   - **Purpose:** Explore and analyze your data interactively.
   - **Input:** Projected embeddings and metadata from previous steps.
   - **How:**
     ```bash
     uv run src/main.py
     ```
   - The dashboard will be available at `http://localhost:8081`


To add your own dataset or model for visualization in HIVE:

1. **Prepare your dataset:**
   - Organize your raw data and metadata as required (see the format used for ImageNet/GRIT).
   - Use `create_tree_structure.py` to build the tree structure and metadata.
2. **Generate embeddings:**
   - Use `generate_embeddings.py` with your model checkpoint/config to create embeddings for each node in the tree.
3. **Create projections:**
   - Use `create_projections.py` to generate 2D projections for the dashboard.
4. **Visualize:**
   - Start the dashboard and select your dataset and projection method.

**You can use any model that outputs embeddings for your data.** Just provide the correct checkpoint/config and ensure your data is organized in the expected tree format.

---

## Arguments for create_tree_structure.py
- `--dataset` (str, required): Name of the dataset (e.g., `ImageNet`).
- `--synsets_csv` (str, required for ImageNet): Path to synsets.csv.
- `--base_path` (str, required for ImageNet): Path to raw images.
- `--output_dir` (str, optional): Output directory for processed data (default: `hierchical_datasets/`).

## Arguments for generate_embeddings.py
- `--dataset` (str, required): Name of the dataset (e.g., `ImageNet`).
- `--checkpoint_path` (str, required): Path to the model checkpoint to use for embedding generation.
- `--train_config` (str, required for HyCoCLIP): Path to the model config file (for HyCoCLIP).
- `--output_dir` (str, optional): Output directory for processed data (default: `hierchical_datasets/`).

## Arguments for create_projections.py
- `--dataset-path` (str, required): Path to the processed dataset folder (e.g., `hierchical_datasets/ImageNet`).
- `--methods` (list, required): Projection methods to use (`horopca`, `cosne`).
- `--n-project` (int, optional): Number of samples to project (0 = all).
- `--children-per-tree` (int, optional): Number of child images per tree (for balanced sampling).
- `--seed` (int, optional): Random seed for reproducibility.
- `--plot` (flag): Generate and save plots of the projections.

**HoroPCA-specific:**
- `--horopca-components` (int, default=2): Number of output dimensions.
- `--horopca-lr` (float, default=0.05): Learning rate.
- `--horopca-steps` (int, default=500): Maximum optimization steps.

**CO-SNE-specific:**
- `--cosne-reduce-method` (str): Pre-reduction method (`none`, `horopca`).
- `--cosne-reduce-dim` (int): Pre-reduction dimension.
- `--cosne-lr` (float): Main learning rate.
- `--cosne-lr-h` (float): Hyperbolic learning rate.
- `--cosne-perplexity` (float): Perplexity.
- `--cosne-exaggeration` (float): Early exaggeration.
- `--cosne-gamma` (float): Student-t gamma.

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for bug fixes, new features, or dataset/model integrations.

---

## Citation
If you use this dashboard or its visualizations in your work, please consider citing our paper!

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
