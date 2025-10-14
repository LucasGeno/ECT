#!/usr/bin/env python3
"""
ECT Data Structure Checker

This script checks the current data structure and provides guidance
on what needs to be set up for the ECT project.
"""

import os
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple

def check_data_structure(project_root: Path) -> Dict[str, any]:
    """Check the current data structure and return status."""
    
    data_dir = project_root / 'data'
    status = {
        'data_dir_exists': data_dir.exists(),
        'data_dir_path': str(data_dir),
        'expected_dirs': {},
        'available_files': {},
        'missing_critical': [],
        'recommendations': []
    }
    
    # Expected directory structure
    expected_dirs = {
        'examples': 'Sample datasets for testing',
        'examples/LB_sample': 'LB medium sample data',
        'examples/M9_sample': 'M9 medium sample data',
        'models': 'Pre-trained models',
        'processed': 'Analysis results',
        'test_data': 'Test datasets',
        'test_data/LB_data': 'LB test data',
        'test_data/M9_data': 'M9 test data',
        'elongated_morphology': 'Elongated cell data'
    }
    
    # Check each expected directory
    for dir_name, description in expected_dirs.items():
        dir_path = data_dir / dir_name
        status['expected_dirs'][dir_name] = {
            'exists': dir_path.exists(),
            'path': str(dir_path),
            'description': description
        }
    
    # Check for critical files
    critical_files = [
        'test_data/LB_data/original_images.tif',
        'test_data/LB_data/masks.tif',
        'test_data/LB_data/napari/tracks_LB_enhanced.parquet',
        'test_data/M9_data/POS1_ROI1/combined_masks.tif',
        'test_data/M9_data/napari/tracks_M9_enhanced.parquet'
    ]
    
    for file_path in critical_files:
        full_path = data_dir / file_path
        if not full_path.exists():
            status['missing_critical'].append(file_path)
    
    # Scan for available files
    if data_dir.exists():
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.tif', '.tiff', '.parquet', '.csv', '.json')):
                    rel_path = Path(root).relative_to(data_dir)
                    if str(rel_path) not in status['available_files']:
                        status['available_files'][str(rel_path)] = []
                    status['available_files'][str(rel_path)].append(file)
    
    # Generate recommendations
    if not status['data_dir_exists']:
        status['recommendations'].append("Run 'python setup_data.py' to create the data structure")
    else:
        missing_dirs = [name for name, info in status['expected_dirs'].items() if not info['exists']]
        if missing_dirs:
            status['recommendations'].append(f"Create missing directories: {', '.join(missing_dirs)}")
        
        if status['missing_critical']:
            status['recommendations'].append(f"Add critical files: {', '.join(status['missing_critical'][:3])}...")
    
    return status

def print_status_report(status: Dict[str, any]) -> None:
    """Print a formatted status report."""
    
    print("ECT Data Structure Check")
    print("=" * 50)
    print(f"Project root: {Path.cwd()}")
    print(f"Data directory: {status['data_dir_path']}")
    print(f"Data directory exists: {'✅' if status['data_dir_exists'] else '❌'}")
    print()
    
    if status['data_dir_exists']:
        print("Directory Structure:")
        print("-" * 30)
        for dir_name, info in status['expected_dirs'].items():
            status_icon = "✅" if info['exists'] else "❌"
            print(f"{status_icon} {dir_name}: {info['description']}")
        print()
        
        print("Available Data Files:")
        print("-" * 30)
        if status['available_files']:
            for subdir, files in status['available_files'].items():
                print(f"📁 {subdir}: {len(files)} files")
                for file in files[:3]:  # Show first 3 files
                    print(f"   - {file}")
                if len(files) > 3:
                    print(f"   ... and {len(files) - 3} more")
        else:
            print("No data files found")
        print()
        
        if status['missing_critical']:
            print("Missing Critical Files:")
            print("-" * 30)
            for file_path in status['missing_critical']:
                print(f"❌ {file_path}")
            print()
    
    if status['recommendations']:
        print("Recommendations:")
        print("-" * 30)
        for i, rec in enumerate(status['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
    
    # Overall status
    if status['data_dir_exists'] and not status['missing_critical']:
        print("🎉 Data structure looks good!")
    elif status['data_dir_exists']:
        print("⚠️  Data structure exists but some files are missing")
    else:
        print("❌ Data structure needs to be created")

def main():
    """Main function."""
    project_root = Path.cwd()
    status = check_data_structure(project_root)
    print_status_report(status)
    
    # Return appropriate exit code
    if not status['data_dir_exists'] or status['missing_critical']:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
