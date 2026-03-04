# StyleAutoEncoder

[![DOI](https://zenodo.org/badge/795693126.svg)](https://doi.org/10.5281/zenodo.18859733)

Deep learning framework and analysis code for the paper ["Deep Learning Framework for Quantifying Self-Organization in Myxococcus xanthus"](https://doi.org/10.1101/2025.11.07.686839).

Author: Jiangguo Zhang (see project provenance in the repository)

This repository contains training and analysis code used to train style-based autoencoders and evaluate learned representations on Myxococcus xanthus image data. It includes data loaders, training loops, evaluation metrics, visualization tools, and Jupyter notebooks used to generate figures for the paper.

**Pre-trained Assets & Data**

Pre-trained models, pre-computed features, and datasets are hosted on BioStudies: [S-BIAD2328](https://doi.org/10.6019/S-BIAD2328).

* Pre-computed Features: Download Roy_training_features.zip and WT_features.zip from the /encoded_features directory for immediate analysis.
* Model Checkpoints: The trained model network-snapshot-003024-patched.pkl is available in the /models directory.
* Full Datasets: Raw and processed image data are located in the /dataset folder.
* Documentation: Detailed dataset structure and metadata are described in the paper’s Supplementary Materials.

**Installation**

- Environment: create the conda environment from the included YAML: `styleautoencoder.yaml`.
	- Example: `conda env create -f styleautoencoder.yaml` and `conda activate styleautoencoder`.
- The codebase targets Python 3.8+ and uses PyTorch; GPU-enabled training requires CUDA-compatible drivers.

**Quick start — Training**

Run the main training script with appropriate arguments. Minimal example:

```
python train_Myxo.py \
	--outdir {OUT_DIR} \
	--data {TRAINING_DATA_DIR} \
	--validation-data {VALIDATION_DATA_DIR} \
	--gpus {GPUS} \
	--batch {BATCH_SIZE} \
	--z-dim {Z_DIM} \
	--workers {NUM_WORKERS}
```

The training script supports additional flags such as `--label-dict` (for stratified sampling), `--mbstd-group`, `--tick`, `--snap`, `--glr`, `--dlr`, `--brightness-norm`, `--resize-by`, `--noise-mode`, and `--calculate_training_metrics`. See [train_Myxo.py](train_Myxo.py) for the full list of flags and usage examples.

An example command used for pretraining in the project:

```
python train_Myxo.py --outdir DAE_project/models/run_e13_random \
	--data DAE_project/dataset/Larry/pre-training.zip \
	--validation-data DAE_project/dataset/Roy_training/images.zip \
	--validation-label-dict DAE_project/dataset/Roy_training/classification_model/merged_test_dict.pkl \
	--gpus 4 --mbstd-group 4 --batch 80 --z-dim 13 --workers 16 --tick 2 --snap 50 \
	--glr 0.001 --dlr 0.001 --brightness-norm 0 --resize-by 1.15 --noise-mode random --calculate_training_metrics
```

**Data format**

- Input images: compressed archives or folder paths accepted by the data loaders in `dataloaders/`.
- Optional `--label-dict`: provide a pickled label dictionary for stratified sampling. If omitted, samples are drawn uniformly.
- Images are resized using the `--resize-by` parameter (project experiments use 2 µm/pixel when appropriate).

**Repository layout (high level)**

- `train_Myxo.py` — main training entry point.
- `dataloaders/` — dataset classes and loaders.
- `training/` — training loops, networks, losses.
- `metrics/` — evaluation metrics (FID, IS, KID, PPL, precision/recall).
- `viz/`, `gui_utils/` — visualization and interactive widgets.
- `preprocessing/` — helper scripts and figure notebooks.
- `styleautoencoder.yaml` — conda environment spec for reproducibility.

**Notebooks & Figures**

Reproducible notebooks used to generate figures for the paper are in `publication_figures/` and in the `preprocessing/` folder.

**Reproducibility & tips**

- Use the provided `styleautoencoder.yaml` to recreate the environment.
- For multi-GPU training set `--gpus` appropriately and ensure CUDA drivers match the environment.
- Save checkpoints (`--snap`) regularly; evaluation utilities are available under `metrics/` and `Turing_test/`.

**Citation**

If you use this code or our pre-trained models in your research, please cite our preprint:
```
@article {Zhang2025.11.07.686839,
	author = {Zhang, Jiangguo and Caro, Eduardo A. and Chen, Peiying and Khan, Trosporsha Tasnim and Murphy, Patrick A. and Shimkets, Lawrence J. and Patel, Ankit B. and Welch, Roy D. and Igoshin, Oleg A.},
	title = {A Deep Learning Framework for Quantifying Dynamic Self-Organization in Myxococcus xanthus},
	elocation-id = {2025.11.07.686839},
	year = {2025},
	doi = {10.1101/2025.11.07.686839},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.07.686839},
	eprint = {https://www.biorxiv.org/content/early/2025/11/09/2025.11.07.686839.full.pdf},
	journal = {bioRxiv}
}
```

**Contact**

If you have questions about the repository or reproducing results, contact: igoshin@rice.edu
