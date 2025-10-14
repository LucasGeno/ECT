"""
Utility functions for working with the ECT data directory structure.
"""

import os
import glob
from pathlib import Path
import shutil
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

# Define the base data directory
DEFAULT_DATA_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"))

def get_data_dir() -> Path:
    """Return the base data directory path."""
    return DEFAULT_DATA_DIR

def list_raw_experiments(media_type: str = "LB_medium") -> List[str]:
    """
    List all experiment directories for a given media type in the raw directory.
    
    Args:
        media_type: The media type (e.g., "LB_medium", "M9_medium").
        
    Returns:
        A list of experiment directory names.
    """
    raw_dir = get_data_dir() / "raw" / media_type
    if not raw_dir.exists():
        return []
    
    return [d.name for d in raw_dir.iterdir() if d.is_dir()]

def get_raw_experiment_path(experiment_name: str, media_type: str = "LB_medium") -> Path:
    """
    Get the path to a raw experiment directory.
    
    Args:
        experiment_name: The name of the experiment.
        media_type: The media type (e.g., "LB_medium", "M9_medium").
        
    Returns:
        Path to the experiment directory.
    """
    return get_data_dir() / "raw" / media_type / experiment_name

def list_training_images() -> List[str]:
    """
    List all training images in the processed/train directory.
    
    Returns:
        A list of training image filenames (without the path).
    """
    train_dir = get_data_dir() / "processed" / "train"
    if not train_dir.exists():
        return []
    
    # Get all images that don't have "_masks" in the name
    image_files = [f.name for f in train_dir.iterdir() 
                  if f.is_file() and not "_masks" in f.name and f.suffix in ['.tif', '.png', '.jpg', '.jpeg']]
    
    return image_files

def list_training_masks() -> List[str]:
    """
    List all training mask files in the processed/train directory.
    
    Returns:
        A list of training mask filenames (without the path).
    """
    train_dir = get_data_dir() / "processed" / "train"
    if not train_dir.exists():
        return []
    
    # Get all mask files
    mask_files = [f.name for f in train_dir.iterdir() 
                 if f.is_file() and "_masks" in f.name and f.suffix in ['.tif', '.png', '.jpg', '.jpeg']]
    
    return mask_files

def get_training_path() -> Path:
    """
    Get the path to the training directory.
    
    Returns:
        Path to the training directory.
    """
    return get_data_dir() / "processed" / "train"

def get_test_path() -> Path:
    """
    Get the path to the test directory.
    
    Returns:
        Path to the test directory.
    """
    return get_data_dir() / "processed" / "test"

def get_models_path(model_type: str = "omnipose_combined") -> Path:
    """
    Get the path to a models directory.
    
    Args:
        model_type: The type of model (e.g., "omnipose_LB", "omnipose_M9", "omnipose_combined").
        
    Returns:
        Path to the models directory.
    """
    return get_data_dir() / "models" / model_type

def get_results_path(experiment_name: str, media_type: str = "LB_medium") -> Path:
    """
    Get the path to a results directory for a specific experiment.
    
    Args:
        experiment_name: The name of the experiment.
        media_type: The media type (e.g., "LB_medium", "M9_medium").
        
    Returns:
        Path to the results directory.
    """
    return get_data_dir() / "results" / media_type / experiment_name

def get_example_dataset_path(dataset_name: str) -> Path:
    """
    Get the path to an example dataset directory.
    
    Args:
        dataset_name: The name of the example dataset (e.g., "cell_division", "under_segmentation").
        
    Returns:
        Path to the example dataset directory.
    """
    return get_data_dir() / "examples" / dataset_name

def prepare_training_data(source_images: List[Path], 
                         source_masks: Optional[List[Path]] = None,
                         copy_to_train: bool = True) -> Path:
    """
    Prepare training data by copying images and masks to the processed/train directory.
    
    Args:
        source_images: List of paths to source images.
        source_masks: List of paths to source masks (must be in same order as source_images).
        copy_to_train: If True, copy files to the training directory; if False, just return the path.
        
    Returns:
        Path to the training directory.
    """
    train_dir = get_training_path()
    train_dir.mkdir(parents=True, exist_ok=True)
    
    if copy_to_train:
        # Copy images to training directory
        for img_path in source_images:
            dest_path = train_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            print(f"Copied {img_path.name} to {dest_path}")
        
        # Copy masks to training directory if provided
        if source_masks:
            for mask_path in source_masks:
                # Ensure mask name has _masks suffix
                base_name = mask_path.stem
                if not base_name.endswith('_masks'):
                    mask_name = f"{base_name}_masks{mask_path.suffix}"
                else:
                    mask_name = mask_path.name
                
                dest_path = train_dir / mask_name
                shutil.copy2(mask_path, dest_path)
                print(f"Copied {mask_path.name} to {dest_path}")
    
    return train_dir

def save_model(model_path: Path, model_type: str = "omnipose_combined") -> Path:
    """
    Save a model to the appropriate models directory.
    
    Args:
        model_path: Path to the model file or directory to save.
        model_type: The type of model (e.g., "omnipose_LB", "omnipose_M9", "omnipose_combined").
        
    Returns:
        Path to the saved model.
    """
    models_dir = get_models_path(model_type)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    if model_path.is_file():
        dest_path = models_dir / model_path.name
        shutil.copy2(model_path, dest_path)
        print(f"Saved model file {model_path.name} to {dest_path}")
    else:
        # If it's a directory, copy the whole directory
        dest_dir = models_dir / model_path.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(model_path, dest_dir)
        print(f"Saved model directory {model_path.name} to {dest_dir}")
    
    return models_dir

def save_results(masks_path: Path, 
                experiment_name: str, 
                media_type: str = "LB_medium",
                result_type: str = "masks") -> Path:
    """
    Save segmentation results to the appropriate results directory.
    
    Args:
        masks_path: Path to the segmentation masks or tracking results.
        experiment_name: The name of the experiment.
        media_type: The media type (e.g., "LB_medium", "M9_medium").
        result_type: The type of result ("masks" or "tracks").
        
    Returns:
        Path to the saved results.
    """
    results_dir = get_results_path(experiment_name, media_type) / result_type
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if masks_path.is_file():
        dest_path = results_dir / masks_path.name
        shutil.copy2(masks_path, dest_path)
        print(f"Saved result file {masks_path.name} to {dest_path}")
    else:
        # If it's a directory, copy the whole directory
        dest_dir = results_dir / masks_path.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        shutil.copytree(masks_path, dest_dir)
        print(f"Saved result directory {masks_path.name} to {dest_dir}")
    
    return results_dir

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
    --------
    Path
        Path to the project root directory
    """
    # This assumes the script is in src/data/data_utils.py
    return Path(__file__).parent.parent.parent

def get_data_path() -> Path:
    """
    Get the path to the data directory.
    
    Returns:
    --------
    Path
        Path to the data directory
    """
    return get_project_root() / "data"

def get_train_path() -> Path:
    """
    Get the path to the training data directory.
    
    Returns:
    --------
    Path
        Path to the training data directory
    """
    return get_data_path() / "train"

def get_test_path() -> Path:
    """
    Get the path to the test data directory.
    
    Returns:
    --------
    Path
        Path to the test data directory
    """
    return get_data_path() / "test"

def get_models_path() -> Path:
    """
    Get the path to the models directory.
    
    Returns:
    --------
    Path
        Path to the models directory
    """
    return get_project_root() / "models"

def get_results_path() -> Path:
    """
    Get the path to the results directory.
    
    Returns:
    --------
    Path
        Path to the results directory
    """
    return get_project_root() / "results"

def get_config_path() -> Path:
    """
    Get the path to the configuration directory.
    
    Returns:
    --------
    Path
        Path to the configuration directory
    """
    return get_project_root() / "config"

def ensure_dir(path: Path) -> Path:
    """
    Ensure that a directory exists.
    
    Parameters:
    -----------
    path : Path
        Path to the directory
        
    Returns:
    --------
    Path
        The same path that was passed in
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_all_image_files(directory: Path, pattern: str = "*.tif") -> list:
    """
    Get all image files in a directory that match a pattern.
    
    Parameters:
    -----------
    directory : Path
        Directory to search in
    pattern : str
        Glob pattern to match files
        
    Returns:
    --------
    list
        List of paths to image files
    """
    return sorted(list(directory.glob(pattern)))

def get_time_series_data(directory: Path) -> dict:
    """
    Find and organize time series data in a directory.
    
    Parameters:
    -----------
    directory : Path
        Directory to search in
        
    Returns:
    --------
    dict
        Dictionary mapping series name to lists of image and mask files
    """
    # Try to find time series data with different patterns
    time_series = {}
    
    # First pattern: files named with _t{number}
    t_series_files = list(directory.glob("*_t*.*"))
    
    if t_series_files:
        # Group by series name (everything before _t)
        for file_path in t_series_files:
            file_name = file_path.name
            series_name = file_name.split("_t")[0]
            
            if series_name not in time_series:
                time_series[series_name] = {"images": [], "masks": []}
            
            # Check if it's an image or mask
            if "mask" in file_name.lower() or "seg" in file_name.lower():
                time_series[series_name]["masks"].append(file_path)
            else:
                time_series[series_name]["images"].append(file_path)
        
        # Sort files by time index
        for series_name, files in time_series.items():
            files["images"] = sorted(files["images"], key=lambda x: int(x.stem.split("_t")[-1].split("_")[0]))
            files["masks"] = sorted(files["masks"], key=lambda x: int(x.stem.split("_t")[-1].split("_")[0]))
    
    # If no time series found, check for directories containing frames
    if not time_series:
        for subdir in directory.iterdir():
            if subdir.is_dir():
                # Check if directory contains frame-like files
                frame_files = list(subdir.glob("frame*.*"))
                if frame_files:
                    series_name = subdir.name
                    time_series[series_name] = {
                        "images": sorted([f for f in frame_files if "mask" not in f.name.lower()]),
                        "masks": sorted([f for f in frame_files if "mask" in f.name.lower()])
                    } 