"""
Data loading utilities for microscopy time-lapse data.
"""
import os
import json
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
import tifffile
from skimage.io import imread
from skimage import io, util


class DataLoader:
    """Handles loading and preprocessing of microscopy data."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        
    def load_image_stack(self, filename: str) -> np.ndarray:
        """
        Load a stack of images from a single file.
        
        Args:
            filename: Name of the image file
            
        Returns:
            numpy.ndarray: Image stack with shape (frames, height, width)
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        return tifffile.imread(filepath)
    
    def load_masks(self, filename: str) -> np.ndarray:
        """
        Load segmentation masks.
        
        Args:
            filename: Name of the mask file
            
        Returns:
            numpy.ndarray: Mask stack with shape (frames, height, width)
        """
        return self.load_image_stack(filename)
    
    def load_image_sequence(self, directory: Union[str, Path], pattern: str = "*.tif") -> np.ndarray:
        """
        Load a sequence of images from a directory.
        
        Args:
            directory: Directory containing the images
            pattern: Glob pattern for image files
            
        Returns:
            numpy.ndarray: Image stack with shape (frames, height, width)
        """
        dir_path = self.data_dir / directory
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # Get sorted list of files
        image_files = sorted(list(dir_path.glob(pattern)))
        if not image_files:
            raise FileNotFoundError(f"No image files found matching pattern {pattern} in {dir_path}")
        
        # Load first image to get dimensions
        first_image = io.imread(image_files[0])
        h, w = first_image.shape
        
        # Initialize stack
        stack = np.zeros((len(image_files), h, w), dtype=first_image.dtype)
        stack[0] = first_image
        
        # Load remaining images
        for i, file_path in enumerate(image_files[1:], 1):
            stack[i] = io.imread(file_path)
        
        return stack
    
    def load_tracks(self, filename: str) -> pd.DataFrame:
        """
        Load tracking data.
        
        Args:
            filename: Name of the track file
            
        Returns:
            pandas.DataFrame: DataFrame containing track information
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine file extension and load accordingly
        extension = filepath.suffix.lower()
        if extension == '.csv':
            return pd.read_csv(filepath)
        elif extension == '.pkl':
            return pd.read_pickle(filepath)
        elif extension == '.h5':
            return pd.read_hdf(filepath)
        else:
            raise ValueError(f"Unsupported track file format: {extension}")
    
    def load_metadata(self, filename: str = "metadata.json") -> Dict[str, Any]:
        """
        Load metadata from a JSON file.
        
        Args:
            filename: Name of the metadata file
            
        Returns:
            dict: Dictionary containing metadata
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Metadata file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def list_available_data(self) -> Dict[str, List[Path]]:
        """
        List available data files in the data directory.
        
        Returns:
            dict: Dictionary of file types and paths
        """
        data_files = {
            'images': list(self.data_dir.glob('*.tif')) + list(self.data_dir.glob('*.tiff')),
            'tracks': list(self.data_dir.glob('*.csv')) + list(self.data_dir.glob('*.pkl')) + list(self.data_dir.glob('*.h5')),
            'metadata': list(self.data_dir.glob('*.json')),
            'other': list(self.data_dir.glob('*.*'))
        }
        
        # Remove files already categorized from 'other'
        all_categorized = []
        for key, files in data_files.items():
            if key != 'other':
                all_categorized.extend(files)
        
        data_files['other'] = [f for f in data_files['other'] if f not in all_categorized]
        
        return data_files
    
    def validate_image_sequence(self, stack: np.ndarray) -> bool:
        """
        Validate image sequence integrity.
        
        Args:
            stack: Image stack to validate
            
        Returns:
            bool: True if sequence is valid
        """
        if len(stack.shape) != 3:
            return False
        if stack.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            return False
            
        # Check for consistency in dimensions
        height, width = stack[0].shape
        for i in range(1, stack.shape[0]):
            if stack[i].shape != (height, width):
                return False
        
        return True
    
    def split_tif_stack(self, 
                      filename: str, 
                      output_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Split a multi-page TIFF into separate images.
        
        Args:
            filename: Name of the TIFF file
            output_dir: Directory to save split frames
            
        Returns:
            Path: Path to the output directory
        """
        # Load the stack
        stack = self.load_image_stack(filename)
        
        # Set output directory
        if output_dir is None:
            output_dir = self.data_dir / f"{Path(filename).stem}_frames"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save individual frames
        for i in range(stack.shape[0]):
            output_path = output_dir / f"frame_{i:04d}.tif"
            tifffile.imwrite(output_path, stack[i])
        
        return output_dir
    
    def concatenate_tif_stacks(self, 
                             filenames: List[str], 
                             output_filename: str) -> Path:
        """
        Concatenate multiple TIFF stacks into a single stack.
        
        Args:
            filenames: List of TIFF filenames
            output_filename: Name of the output TIFF file
            
        Returns:
            Path: Path to the output file
        """
        # Load the first stack to get dimensions
        first_stack = self.load_image_stack(filenames[0])
        total_frames = first_stack.shape[0]
        
        # Count total frames and check dimensions
        for filename in filenames[1:]:
            stack = self.load_image_stack(filename)
            if stack.shape[1:] != first_stack.shape[1:]:
                raise ValueError(f"Stack dimensions don't match: {stack.shape} vs {first_stack.shape}")
            total_frames += stack.shape[0]
        
        # Create output stack
        output_stack = np.zeros((total_frames, *first_stack.shape[1:]), dtype=first_stack.dtype)
        
        # Fill output stack
        frame_idx = 0
        for filename in filenames:
            stack = self.load_image_stack(filename)
            output_stack[frame_idx:frame_idx+stack.shape[0]] = stack
            frame_idx += stack.shape[0]
        
        # Save output stack
        output_path = self.data_dir / output_filename
        tifffile.imwrite(output_path, output_stack)
        
        return output_path 