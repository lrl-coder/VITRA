import numpy as np
import os

file_path = r'd:\project\PYProject\VITRA\data\examples\annotations\Ego4D_03cc49c3-a7d1-445b-9a2a-545c4fae6843_ep_example.npy'

def print_structure(data, indent=0):
    indent_str = "  " * indent
    if isinstance(data, dict):
        for key, value in data.items():
            info = ""
            is_dict = isinstance(value, dict)
            
            if hasattr(value, 'shape'):
                info = f"Shape: {value.shape}, Dtype: {value.dtype}"
            elif isinstance(value, list):
                info = f"List of length {len(value)}"
                if len(value) > 0:
                    info += f", First item type: {type(value[0])}"
            elif not is_dict:
                info = f"Type: {type(value)}"
            
            print(f"{indent_str}- {key}: {info}")
            
            if is_dict:
                print_structure(value, indent + 1)
            elif isinstance(value, (str, int, float, bool, np.integer, np.floating)):
                 print(f"{indent_str}  Value: {value}")

def inspect_npy(path):
    print(f"Loading {path}...")
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"Type: {type(data)}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")

    # Check if it's a 0-d array wrapping an object (like a dict)
    if data.ndim == 0 and data.dtype == 'O':
        print("\nDetected 0-d object array. Extracting item...")
        content = data.item()
        print(f"Content Type: {type(content)}")
        
        if isinstance(content, dict):
            print("\nDictionary Structure:")
            print_structure(content)
        else:
            print(f"Content: {content}")
            
    # Check if it's a structured array or regular array
    elif data.ndim > 0:
        print("\nArray contents sample:")
        if data.size > 10:
             print(data.flat[:10])
        else:
             print(data)

if __name__ == "__main__":
    if os.path.exists(file_path):
        inspect_npy(file_path)
    else:
        print(f"File not found: {file_path}")
