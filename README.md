# E. coli Tracking and Analysis Pipeline

This repository contains code for segmentation, tracking, and analysis of E. coli cells in time-lapse microscopy images. The pipeline combines Omnipose for segmentation and btrack for cell tracking, providing a comprehensive solution for analyzing bacterial growth dynamics and morphological characteristics.

## Features

- High-accuracy cell segmentation using Omnipose with domain-specific training
- Robust cell tracking with Bayesian tracking (btrack) and napari visualization
- Comprehensive morphological feature extraction and elongated cell analysis
- Growth analysis with Painter-Marr balanced growth validation
- Model comparison framework (Omnipose, Cellpose, DeLTA2, Random Forest Watershed)
- Media condition comparison (LB vs M9)
- Visualization tools for results interpretation
- Quality control and validation metrics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/LucasGeno/ECT.git
cd ECT
```

2. Create the conda environment:
```bash
conda env create -f environment.yml
conda activate ecoli-tracking
```

3. For Omnipose-specific setup, follow the detailed instructions in:
   - `notebooks/01_env_setup/install_omnipose.md`
   - `notebooks/01_env_setup/omni_env_setup_mac.ipynb` (for macOS users)

## Usage

The pipeline is organized into sequential notebook-based workflows:

### 1. **Environment Setup** (`01_env_setup/`)
   - `01_environment_setup.ipynb` - Basic environment validation
   - `omni_env_setup_mac.ipynb` - Omnipose installation for macOS
   - `tracking_env_setup.ipynb` - Tracking environment setup
   - `install_omnipose.md` - Detailed Omnipose installation guide

### 2. **Data Preprocessing** (`02_data_preprocessing/`)
   - `data_loader.ipynb` - Load and validate microscopy time-lapse data
   - `data_preperation.ipynb` - Prepare images for segmentation
   - `data_preprocess.ipynb` - Advanced preprocessing workflows

### 3. **Model Comparison** (`03_model_comparison/`)
   - `model_comparison_final.ipynb` - Comprehensive evaluation of segmentation models
     - Omnipose (300e, 30e variants)
     - Cellpose Custom
     - DeLTA2
     - Random Forest Watershed

### 4. **Elongated Morphology Analysis** (`04_elongated_morphology/`)
   - `elongated_morphology_validation.ipynb` - Analysis of cell filamentation and morphological features

### 5. **Tracking Analysis** (`05_tracking_analysis/`)
   - `tracking_cycle_analysis.ipynb` - Cell cycle analysis with Painter-Marr validation

## Project Structure

```
ECT/
├── data/                           # Data directory
│   ├── examples/                   # Example datasets
│   │   ├── LB_medium/             # LB medium time-lapse data
│   │   ├── LB_sample/             # LB sample data
│   │   ├── M9_sample/             # M9 sample data
│   │   └── public/                # Public datasets
│   ├── models/                     # Trained models
│   │   ├── omnipose_1ch_2cl_e300   # Primary Omnipose model
│   │   ├── omni_80_24e            # Alternative model
│   │   └── omnipose_trainset/     # Training set models
│   ├── processed/                  # Processed data and results
│   └── test_data/                  # Test datasets for validation
│       ├── LB_data/               # LB test data with ground truth
│       └── M9_data/               # M9 test data with ground truth
│
├── src/                           # Source code modules
│   ├── analysis/                  # Analysis utilities
│   ├── data/                      # Data loading and preprocessing
│   ├── segmentation/              # Segmentation code
│   ├── tracking/                  # Tracking code
│   └── utils/                     # Shared utilities
│
├── notebooks/                     # Analysis notebooks
│   ├── 01_env_setup/             # Environment setup
│   ├── 02_data_preprocessing/    # Data preprocessing
│   ├── 03_model_comparison/      # Model evaluation
│   ├── 04_elongated_morphology/  # Morphological analysis
│   └── 05_tracking_analysis/     # Tracking and growth analysis
│
├── configs/                       # Configuration files
│   ├── experiment_config.yaml    # General experiment settings
│   ├── omnipose_config.yaml      # Omnipose parameters
│   └── tracking_config.yaml      # Tracking parameters
│
├── docs/                          # Generated documentation
│   ├── elongated_morphology_validation.html
│   ├── model_comparison_final.html
│   └── tracking_cycle_analysis.html
│
├── results/                       # Analysis outputs
│   ├── figures/                   # Generated figures
│   ├── models/                    # Analysis results
│   └── tables/                    # Summary tables
│
├── environment.yml               # Conda environment specification
├── setup.py                     # Package setup
└── README.md                    # This file
```

## Key Analysis Workflows

### Model Performance Evaluation
The `model_comparison_final.ipynb` notebook provides comprehensive evaluation of different segmentation approaches:
- Quantitative metrics (IoU, precision, recall)
- Morphological accuracy assessment
- Computational performance comparison

### Growth Analysis and Validation
The `tracking_cycle_analysis.ipynb` notebook implements:
- Painter-Marr balanced growth validation
- Cell cycle parsing and analysis
- Growth rate calculations
- Media condition comparisons

### Elongated Cell Morphology
The `elongated_morphology_validation.ipynb` notebook focuses on:
- Detection and analysis of cell filamentation
- Morphological feature extraction
- Statistical validation of morphological changes

## Configuration

The pipeline uses YAML configuration files:

- `experiment_config.yaml` - General settings, data paths, analysis parameters
- `omnipose_config.yaml` - Omnipose segmentation parameters
- `tracking_config.yaml` - Cell tracking and analysis parameters

## Data Requirements

The pipeline expects the following data structure:
- Raw microscopy images in TIFF format
- Ground truth masks for validation
- Metadata files for experimental conditions
- Pre-computed segmentation results for comparison

## Example Datasets

The repository includes example datasets for:
- **LB medium**: Rich medium with rapid growth
- **M9 medium**: Minimal medium with slower growth
- **Gold standard**: Manually annotated ground truth
- **Test cases**: Cell division, missing masks, under-segmentation


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Omnipose team for the segmentation model
- btrack developers for the tracking algorithm
- Systems Biology Lab Amsterdam for support and resources
- Contributors to the cell tracking and analysis community
