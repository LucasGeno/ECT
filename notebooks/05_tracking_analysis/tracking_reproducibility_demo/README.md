# Napari + btrack + Arboretum Reproducibility Demo

This demo shows how to reproduce the tracking analysis from the ECT project using Napari with btrack and Arboretum plugins.

## Overview

This demo provides:
- **Environment setup** for Napari with tracking plugins
- **Pre-configured btrack settings** for LB and M9 datasets
- **Interactive visualization** of cell lineages
- **Reproducible tracking workflow** matching the thesis analysis

## Quick Start

### 1. Environment Setup
```bash
# Create the tracking environment
conda env create -f env_napari.yml
conda activate napari_tracking_env
```

### 2. Launch Napari
```bash
# Start Napari (may take a moment on first launch)
napari
```

### 3. Load Data
- **LB Dataset**: `data/timelapse_data/LB_data/` (original_images.tif, masks.tif)
- **M9 Dataset**: `data/timelapse_data/M9_data/` (bf_frames.tif, stacked_masks.tif)

### 4. Run Tracking
- Open the **btrack** plugin
- Load segmentation masks
- Use pre-configured settings:
  - **LB**: `btrack_configs/LB_ROI1_config.json`
  - **M9**: `btrack_configs/M9_ROI1_config.json`
- Run tracking algorithm

### 5. Visualize Lineages
- Open the **Arboretum** plugin
- Select the tracks layer
- Double-click any cell to view its lineage tree
- Navigate through the family tree interactively

### 6. Save Results (Optional)
```python
# Save the complete Napari session
viewer.layers.save("LB_lineage.layers")
```

## Configuration Details

The provided configurations are optimized for:
- **LB Medium**: Rich growth conditions, higher cell density
- **M9 Medium**: Minimal growth conditions, lower cell density

Both configurations include:
- Motion model parameters for bacterial tracking
- Hypothesis model settings for cell division detection
- Optimized search radii and thresholds

## Troubleshooting

- **Plugin not visible**: Restart Napari after environment activation
- **Tracking errors**: Check mask format (integer labels, 0=background)
- **Performance issues**: Reduce dataset size for testing

## Related Analysis

This demo reproduces the tracking analysis from:
- `notebooks/05_tracking_analysis/tracking_cycle_analysis.ipynb`
- See `docs/tracking_cycle_analysis.html` for complete results