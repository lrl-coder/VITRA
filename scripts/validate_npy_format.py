"""
NPY Format Validation Tool for VITRA Dataset.

This script validates that generated .npy files conform to the expected VITRA metadata format
as documented in data/data.md.

Usage:
    python scripts/validate_npy_format.py --input path/to/file.npy
    python scripts/validate_npy_format.py --input path/to/directory
"""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys


# Expected format specification based on data/data.md
EXPECTED_FORMAT = {
    # Top-level keys with their expected types and shapes
    'video_clip_id_segment': {'type': np.ndarray, 'dtype': np.int64, 'shape_check': lambda s, T: s == (T,)},
    'extrinsics': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (T, 4, 4)},
    'intrinsics': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (3, 3)},
    'video_decode_frame': {'type': np.ndarray, 'dtype': np.int64, 'shape_check': lambda s, T: s == (T,)},
    'video_name': {'type': str},
    'avg_speed': {'type': (np.floating, float), 'scalar': True},
    'total_rotvec_degree': {'type': (np.floating, float), 'scalar': True},
    'total_transl_dist': {'type': (np.floating, float), 'scalar': True},
    'anno_type': {'type': str},
    'text': {'type': dict},
    'text_rephrase': {'type': dict},
    'left': {'type': dict},
    'right': {'type': dict},
}

# Expected hand data format
HAND_FORMAT = {
    'beta': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (10,)},
    'global_orient_camspace': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (T, 3, 3)},
    'global_orient_worldspace': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (T, 3, 3)},
    'hand_pose': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (T, 15, 3, 3)},
    'transl_camspace': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (T, 3)},
    'transl_worldspace': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (T, 3)},
    'kept_frames': {'type': np.ndarray, 'dtype': np.int64, 'shape_check': lambda s, T: s == (T,)},
    'joints_camspace': {'type': np.ndarray, 'dtype': np.float32, 'shape_check': lambda s, T: s == (T, 21, 3)},
    'joints_worldspace': {'type': np.ndarray, 'dtype': np.float64, 'shape_check': lambda s, T: s == (T, 21, 3)},
    'wrist': {'type': np.ndarray, 'dtype': np.float32, 'shape_check': lambda s, T: s == (T, 1, 3)},
    'max_translation_movement': {'type': (np.floating, float, type(None)), 'scalar': True, 'optional': True},
    'max_wrist_rotation_movement': {'type': (np.floating, float, type(None)), 'scalar': True, 'optional': True},
    'max_finger_joint_angle_movement': {'type': (np.floating, float, type(None)), 'scalar': True, 'optional': True},
}


class ValidationResult:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        
    def add_error(self, msg: str):
        self.errors.append(msg)
        
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        
    def add_info(self, msg: str):
        self.info.append(msg)
        
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def print_report(self):
        print(f"\n{'='*60}")
        print(f"Validation Report: {self.file_path}")
        print(f"{'='*60}")
        
        if self.info:
            print("\n📊 Info:")
            for msg in self.info:
                print(f"  ℹ️  {msg}")
        
        if self.warnings:
            print("\n⚠️  Warnings:")
            for msg in self.warnings:
                print(f"  ⚠️  {msg}")
                
        if self.errors:
            print("\n❌ Errors:")
            for msg in self.errors:
                print(f"  ❌ {msg}")
        
        if self.is_valid():
            print(f"\n✅ VALID: File conforms to VITRA format specification")
        else:
            print(f"\n❌ INVALID: {len(self.errors)} error(s) found")
        
        print(f"{'='*60}\n")


def get_num_frames(data: Dict) -> int:
    """Determine the number of frames from the data."""
    # Try to get T from video_decode_frame or extrinsics
    if 'video_decode_frame' in data:
        vdf = data['video_decode_frame']
        if isinstance(vdf, np.ndarray):
            return len(vdf)
        elif isinstance(vdf, list):
            return len(vdf)
    if 'extrinsics' in data:
        ext = data['extrinsics']
        if isinstance(ext, np.ndarray) and len(ext.shape) >= 1:
            return ext.shape[0]
    return 25  # Default expected value


def validate_field(data: Dict, key: str, spec: Dict, T: int, result: ValidationResult, prefix: str = ""):
    """Validate a single field against its specification."""
    full_key = f"{prefix}{key}" if prefix else key
    
    if key not in data:
        if spec.get('optional', False):
            result.add_info(f"Optional field '{full_key}' not present")
        else:
            result.add_error(f"Missing required field: '{full_key}'")
        return
    
    value = data[key]
    
    # Type check
    expected_type = spec['type']
    if isinstance(expected_type, tuple):
        type_match = isinstance(value, expected_type)
    else:
        type_match = isinstance(value, expected_type)
    
    if not type_match:
        result.add_error(f"Field '{full_key}': Expected type {expected_type}, got {type(value)}")
        return
    
    # For numpy arrays, check dtype and shape
    if isinstance(value, np.ndarray):
        if 'dtype' in spec:
            expected_dtype = spec['dtype']
            if value.dtype != expected_dtype:
                result.add_warning(f"Field '{full_key}': Expected dtype {expected_dtype}, got {value.dtype}")
        
        if 'shape_check' in spec:
            if not spec['shape_check'](value.shape, T):
                result.add_error(f"Field '{full_key}': Shape {value.shape} does not match expected format (T={T})")
            else:
                result.add_info(f"Field '{full_key}': Shape {value.shape} ✓")


def validate_hand_data(hand_data: Dict, hand_side: str, T: int, result: ValidationResult):
    """Validate hand-specific data fields."""
    prefix = f"{hand_side}."
    
    for key, spec in HAND_FORMAT.items():
        validate_field(hand_data, key, spec, T, result, prefix)


def validate_text_field(text_data: Dict, field_name: str, result: ValidationResult):
    """Validate text annotation fields."""
    if not isinstance(text_data, dict):
        result.add_error(f"Field '{field_name}': Expected dict, got {type(text_data)}")
        return
    
    for side in ['left', 'right']:
        if side not in text_data:
            result.add_warning(f"Field '{field_name}': Missing key '{side}'")
        else:
            if not isinstance(text_data[side], list):
                result.add_error(f"Field '{field_name}.{side}': Expected list, got {type(text_data[side])}")


def validate_npy_file(file_path: Path) -> ValidationResult:
    """Validate a single .npy file."""
    result = ValidationResult(str(file_path))
    
    try:
        data = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        result.add_error(f"Failed to load file: {e}")
        return result
    
    if not isinstance(data, dict):
        result.add_error(f"Root data should be a dict, got {type(data)}")
        return result
    
    # Determine number of frames
    T = get_num_frames(data)
    result.add_info(f"Detected {T} frames")
    
    # Validate top-level fields
    for key, spec in EXPECTED_FORMAT.items():
        if key in ['left', 'right']:
            if key in data:
                if isinstance(data[key], dict):
                    validate_hand_data(data[key], key, T, result)
                else:
                    result.add_error(f"Field '{key}': Expected dict, got {type(data[key])}")
        elif key in ['text', 'text_rephrase']:
            if key in data:
                validate_text_field(data[key], key, result)
            else:
                result.add_error(f"Missing required field: '{key}'")
        else:
            validate_field(data, key, spec, T, result)
    
    # Check for unexpected keys
    expected_keys = set(EXPECTED_FORMAT.keys())
    actual_keys = set(data.keys())
    extra_keys = actual_keys - expected_keys
    if extra_keys:
        result.add_warning(f"Unexpected top-level keys: {extra_keys}")
    
    return result


def print_structure(data: Dict, indent: int = 0):
    """Print the structure of a loaded npy file."""
    prefix = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{prefix}- {key}: Shape: {value.shape}, Dtype: {value.dtype}")
        elif isinstance(value, dict):
            print(f"{prefix}- {key}:")
            print_structure(value, indent + 1)
        elif isinstance(value, list):
            if len(value) > 0:
                print(f"{prefix}- {key}: List of length {len(value)}, First item type: {type(value[0])}")
            else:
                print(f"{prefix}- {key}: Empty list")
        elif isinstance(value, str):
            print(f"{prefix}- {key}: Type: {type(value)}")
            print(f"{prefix}  Value: {value[:50]}..." if len(value) > 50 else f"{prefix}  Value: {value}")
        else:
            if isinstance(value, (np.floating, float)):
                print(f"{prefix}- {key}: Shape: (), Dtype: {type(value).__name__}")
                print(f"{prefix}  Value: {value}")
            else:
                print(f"{prefix}- {key}: Type: {type(value)}")


def main():
    parser = argparse.ArgumentParser(description="Validate VITRA NPY file format")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to .npy file or directory containing .npy files")
    parser.add_argument("--structure", "-s", action="store_true",
                        help="Print the structure of the file(s)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed info messages")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)
    
    # Collect files to validate
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("**/*.npy"))
    
    if not files:
        print(f"No .npy files found in {input_path}")
        sys.exit(1)
    
    print(f"Found {len(files)} .npy file(s) to validate")
    
    valid_count = 0
    invalid_count = 0
    
    for file_path in files:
        if args.structure:
            print(f"\n{'='*60}")
            print(f"Structure: {file_path}")
            print(f"{'='*60}")
            try:
                data = np.load(file_path, allow_pickle=True).item()
                print("Dictionary Structure:")
                print_structure(data)
            except Exception as e:
                print(f"Error loading file: {e}")
            continue
        
        result = validate_npy_file(file_path)
        
        if not args.verbose:
            # Filter out info messages for non-verbose mode
            result.info = []
        
        result.print_report()
        
        if result.is_valid():
            valid_count += 1
        else:
            invalid_count += 1
    
    if not args.structure:
        print(f"\n{'='*60}")
        print(f"Summary: {valid_count} valid, {invalid_count} invalid out of {len(files)} files")
        print(f"{'='*60}")
        
        sys.exit(0 if invalid_count == 0 else 1)


if __name__ == "__main__":
    main()
