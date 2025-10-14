"""
Utility functions for the E. coli tracking and analysis pipeline.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage import measure


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Path to check/create
        
    Returns:
        Path: Path to the directory
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def save_metadata(metadata: Dict[str, Any], filepath: Union[str, Path]) -> Path:
    """
    Save metadata to a JSON file.
    
    Args:
        metadata: Dictionary of metadata
        filepath: Path to save the metadata
        
    Returns:
        Path: Path to the saved metadata file
    """
    filepath = Path(filepath)
    
    # Create directory if it doesn't exist
    filepath.parent.mkdir(exist_ok=True, parents=True)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filepath


def load_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.
    
    Args:
        filepath: Path to the metadata file
        
    Returns:
        dict: Dictionary of metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def validate_paths(paths: Dict[str, Union[str, Path]]) -> Dict[str, Path]:
    """
    Validate and convert paths to Path objects.
    
    Args:
        paths: Dictionary of paths
        
    Returns:
        dict: Dictionary of validated Path objects
    """
    validated_paths = {}
    
    for key, path in paths.items():
        path = Path(path)
        if not path.exists():
            print(f"Warning: Path {path} does not exist")
        validated_paths[key] = path
    
    return validated_paths


def plot_mask_overlay(image: np.ndarray, 
                     mask: np.ndarray, 
                     alpha: float = 0.3,
                     figsize: Tuple[int, int] = (10, 8)) -> Figure:
    """
    Plot an image with a mask overlay.
    
    Args:
        image: Input image
        mask: Mask to overlay
        alpha: Transparency of the overlay
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the image
    ax.imshow(image, cmap='gray')
    
    # Overlay the mask
    if mask.dtype == bool:
        # Binary mask
        masked = np.ma.masked_where(~mask, mask)
        ax.imshow(masked, alpha=alpha, cmap='jet')
    else:
        # Label mask
        from skimage.color import label2rgb
        overlay = label2rgb(mask, bg_label=0)
        ax.imshow(overlay, alpha=alpha)
    
    ax.set_title("Image with Mask Overlay")
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig


def extract_region_props(mask: np.ndarray, 
                       properties: Optional[List[str]] = None) -> Dict[int, Dict[str, Any]]:
    """
    Extract region properties from a labeled mask.
    
    Args:
        mask: Labeled mask
        properties: List of properties to extract
        
    Returns:
        dict: Dictionary of region properties (label -> properties)
    """
    if properties is None:
        properties = ['area', 'perimeter', 'centroid', 'major_axis_length', 
                     'minor_axis_length', 'eccentricity', 'solidity']
    
    # Extract region properties
    regions = measure.regionprops_table(mask, properties=properties)
    
    # Convert to dictionary format (label -> properties)
    result = {}
    
    # Get labels
    labels = np.unique(mask)
    labels = labels[labels > 0]  # Exclude background
    
    for i, label in enumerate(labels):
        props = {}
        for prop in properties:
            if isinstance(regions[prop], np.ndarray) and regions[prop].ndim > 1:
                props[prop] = regions[prop][i].tolist()
            else:
                props[prop] = regions[prop][i]
        
        result[int(label)] = props
    
    return result 