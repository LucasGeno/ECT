# E. coli Tracking (ECT) Project

A comprehensive framework for bacterial cell tracking, segmentation, and analysis using deep learning and classical computer vision methods.

## Overview

This project provides tools and analysis pipelines for:
- **Cell Segmentation**: Multiple deep learning models (Omnipose, Cellpose, DeLTA2) and a classical method
- **Cell Tracking**: Lineage tracking and cycle analysis
- **Morphological Analysis**: Cell shape and size measurements
- **Growth Dynamics**: Population and single-cell growth analysis

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/LucasGeno/ECT.git
cd ECT

# Create conda environment
conda create -n ect_env python=3.9
conda activate ect_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Data

```bash
# Create data directory structure and download sample data
python setup_data.py --download-sample --setup-models --verbose
```

### 3. Run Analysis

```bash
# Start Jupyter Lab
jupyter lab

# Open notebooks in order:
# 1. notebooks/01_env_setup/01_environment_setup.ipynb
# 2. notebooks/02_data_preprocessing/data_loader.ipynb
# 3. notebooks/03_model_comparison/model_comparison_final.ipynb
# 4. notebooks/04_elongated_morphology/elongated_morphology_validation.ipynb
# 5. notebooks/05_tracking_analysis/tracking_cycle_analysis.ipynb
```

## Project Structure

```
ECT/
├── data/                          # Data directory (created by setup_data.py)
│   ├── examples/                   # Sample datasets
│   ├── models/                    # Pre-trained models
│   ├── processed/                 # Analysis results
│   └── test_data/                 # Test datasets
├── docs/                          # Documentation and thesis
│   ├── LR_Thesis.pdf             # Main thesis document
│   └── *.html                     # Analysis reports
├── notebooks/                     # Jupyter notebooks
│   ├── 01_env_setup/             # Environment setup
│   ├── 02_data_preprocessing/    # Data loading and preprocessing
│   ├── 03_model_comparison/      # Model evaluation
│   ├── 04_elongated_morphology/  # Elongated cell analysis
│   └── 05_tracking_analysis/     # Tracking and cycle analysis
├── src/                          # Source code modules
│   ├── data/                     # Data loading utilities
│   ├── analysis/                 # Analysis functions
│   ├── segmentation/             # Segmentation models
│   ├── tracking/                 # Tracking algorithms
│   └── utils/                    # Utility functions
├── setup_data.py                # Data setup script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Data Requirements

### Input Data Format

**Time-lapse Images:**
- Format: TIFF stack (.tif)
- Dimensions: (time, height, width) or (time, height, width, channels)
- Naming: `original_images.tif`

**Segmentation Masks:**
- Format: TIFF stack (.tif)
- Values: Integer labels (0 = background, >0 = cell ID)
- Naming: `masks.tif`

**Tracking Data:**
- Format: Parquet (.parquet) or CSV (.csv)
- Required columns: `track_id`, `t`, `area`, `parent`
- Optional columns: `x`, `y`, `width`, `length`

### Sample Data

The project includes sample datasets for testing:
- **LB Medium**: Rich growth conditions
- **M9 Medium**: Minimal growth conditions
- **Elongated Morphology**: LB Medium with some filamentous cells

## Analysis Workflow

### 1. Data Preprocessing
- Load and inspect microscopy data
- Validate data quality

### 2. Model Comparison
- Evaluate segmentation models
- Compare performance metrics
- Generate publication figures

### 3. Morphological Analysis
- Analyze cell shapes and sizes
- Validate on elongated cells
- Compute width distributions

### 4. Tracking Analysis
- Parse cell lineages
- Compute generation times
- Validate balanced growth

## Key Features

### Segmentation Models
- **Omnipose**: State-of-the-art bacterial segmentation (morphologically-independent)
- **Cellpose**: General-purpose cell segmentation (generalist)
- **DeLTA2**: Deep learning tracking (binary masks)
- **Random Forest Watershed**: Classical method

### Analysis Capabilities
- **Cell Counting**: Accurate cell enumeration
- **Morphometrics**: Size and shape measurements
- **Lineage Tracking**: Parent-child relationships
- **Growth Analysis**: Population and single-cell dynamics
- **Cycle Analysis**: Generation time computation

### Validation Methods
- **Cross-scale Validation**: Painter-Marr ratio analysis
- **Bootstrap Confidence Intervals**: Statistical uncertainty
- **Bias Correction**: Track vs. cycle time comparison

## Usage Examples

### Basic Data Loading

```python
from src.data.loader import DataLoader

# Initialize loader
loader = DataLoader()

# Load sample data
images = loader.load_image_stack('original_images.tif', 'test_data/LB_data')
masks = loader.load_masks('masks.tif', 'test_data/LB_data')
tracks = loader.load_tracking_data('tracks_LB_enhanced.parquet', 'test_data/LB_data/napari')
```

### Quick Data Check

```python
from src.data.loader import quick_data_check

# Check available data
quick_data_check()
```

### Model Evaluation

```python
# Run model comparison (see notebooks/03_model_comparison/)
# This will generate performance metrics and figures
```

## Configuration

The project uses a configuration file (`config.json`) with default settings:

```json
{
  "data": {
    "pixel_size_um": 0.065,
    "frame_interval_min": 0.5
  },
  "analysis": {
    "min_cycle_frames": 4,
    "boundary_margin_frames": 3,
    "bootstrap_samples": 5000
  }
}
```

## Dependencies

### Core Requirements
- Python 3.9+
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- scikit-image, scikit-learn

### Deep Learning
- PyTorch
- Cellpose
- Omnipose

### Data Formats
- tifffile
- h5py
- parquet

### Visualization
- Napari (recommended)
- Fiji
- Jupyter Lab

## Troubleshooting

### Common Issues

1. **Data directory not found**
   ```bash
   python setup_data.py
   ```

2. **Missing dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **CUDA/GPU issues**
   - Install appropriate CUDA version
   - Check PyTorch installation

4. **Memory issues**
   - Use smaller datasets for testing
   - Process data in chunks

### Getting Help

- Check the notebooks for examples
- Review the documentation in `docs/`
- Examine the thesis for detailed methodology



## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- Omnipose and Cellpose communities
- DeLTA2 developers
- Napari project
- Scientific Python ecosystem
- O2 lab