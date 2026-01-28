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

def convert_npy_to_json(npy_file_path, output_dir=None):
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

    # Determine output path
    file_name = os.path.basename(npy_file_path)
    base_name_no_ext = os.path.splitext(file_name)[0]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        json_file_path = os.path.join(output_dir, base_name_no_ext + ".json")
    else:
        # Default: side-by-side with original
        dir_name = os.path.dirname(npy_file_path)
        json_file_path = os.path.join(dir_name, base_name_no_ext + ".json")

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
    parser = argparse.ArgumentParser(description="Convert .npy file(s) to .json")
    parser.add_argument("path", help="Path to the .npy file or directory containing .npy files")
    parser.add_argument("--match", "-m", help="String to match in filenames (only for directory mode)", default=None)
    parser.add_argument("--output", "-o", help="Directory to save the output .json files", default=None)
    
    args = parser.parse_args()
    
    input_path = os.path.abspath(args.path)
    
    if os.path.isfile(input_path):
        convert_npy_to_json(input_path, args.output)
    elif os.path.isdir(input_path):
        print(f"Scanning directory: {input_path}")
        if args.match:
            print(f"Filter pattern: '{args.match}'")
            
        files = sorted([f for f in os.listdir(input_path) if f.endswith('.npy')])
        if not files:
            print("No .npy files found in directory.")
            sys.exit(0)

        count = 0
        for f in files:
            if args.match and args.match not in f:
                continue
                
            full_path = os.path.join(input_path, f)
            print("-" * 30)
            convert_npy_to_json(full_path, args.output)
            count += 1
            
        print("-" * 30)
        print(f"Processed {count} files.")
    else:
        print(f"Error: Path does not exist: {input_path}")
