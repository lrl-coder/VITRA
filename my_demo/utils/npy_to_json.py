import numpy as np
import json
import argparse
import os
import sys

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def recursive_numpy_conversion(obj):
    """Recursively convert numpy types to native python types for objects specifically."""
    if isinstance(obj, dict):
        return {k: recursive_numpy_conversion(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_numpy_conversion(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32,
                          np.float64)):
        return float(obj)
    else:
        return obj

def convert_npy_to_json(npy_file_path):
    if not os.path.exists(npy_file_path):
        print(f"Error: File not found at {npy_file_path}")
        return

    print(f"Loading {npy_file_path}...")
    try:
        data = np.load(npy_file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    # Handle 0-d array wrapping an object (common in saved dicts)
    if data.ndim == 0 and data.dtype == 'O':
        data = data.item()

    # Create output filename
    base_name = os.path.splitext(npy_file_path)[0]
    json_file_path = base_name + ".json"

    print("Converting to JSON compatible format...")
    # We do a pre-conversion pass to handle nested numpy numbers inside dicts/lists 
    # that JSONEncoder might sometimes miss if they are keys or deep inside.
    # But usually JSONEncoder is enough for values. 
    # However, to be safe and ensure full compatibility especially if there are other types:
    serializable_data = recursive_numpy_conversion(data)

    print(f"Saving to {json_file_path}...")
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)
        print("Done!")
    except Exception as e:
        print(f"Error saving .json file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert an .npy file to .json")
    parser.add_argument("file", help="Path to the .npy file")
    
    args = parser.parse_args()
    convert_npy_to_json(args.file)
