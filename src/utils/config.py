"""
Configuration utilities for loading and managing YAML configurations.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

import yaml


class ConfigManager:
    """Configuration manager for loading and managing YAML configurations."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Path to the configuration directory
        """
        # Use default config directory if not specified
        if config_dir is None:
            # Try to find the config directory relative to the current file
            current_dir = Path(__file__).parent.parent.parent
            config_dir = current_dir / 'configs'
            
            # If not found, use the current working directory
            if not config_dir.exists():
                config_dir = Path(os.getcwd()) / 'configs'
        
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {self.config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the configuration file (with or without .yaml extension)
            
        Returns:
            dict: Dictionary containing the configuration
        """
        # Add extension if not provided
        if not config_name.endswith(('.yaml', '.yml')):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> Path:
        """
        Save a configuration to a file.
        
        Args:
            config: Configuration dictionary
            config_name: Name of the configuration file (with or without .yaml extension)
            
        Returns:
            Path: Path to the saved configuration file
        """
        # Add extension if not provided
        if not config_name.endswith(('.yaml', '.yml')):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        
        # Create directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config_path
    
    def update_config(self, config_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update an existing configuration.
        
        Args:
            config_name: Name of the configuration file
            updates: Dictionary of updates to apply
            
        Returns:
            dict: Updated configuration
        """
        # Load existing config
        config = self.load_config(config_name)
        
        # Apply updates (recursive merging)
        config = self._merge_configs(config, updates)
        
        # Save updated config
        self.save_config(config, config_name)
        
        return config
    
    def _merge_configs(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration
            updates: Updates to apply
            
        Returns:
            dict: Merged configuration
        """
        result = base.copy()
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result 