import json
import os
from collections import defaultdict

data_dir = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/ForgeryNet/OpenDataLab___ForgeryNet/raw/validation/Validation"

def get_split_type(file_path):
    # Extract the second directory number from path
    # Format: val_perturb_release/6/...
    parts = file_path.split('/')
    if len(parts) > 2:
        try:
            dir_num = int(parts[1])  # Changed from parts[2] to parts[1]
            if dir_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]:
                return "train"
            elif dir_num in [13, 14, 15, 19]:
                return "test"
        except ValueError:
            pass
    return None

def process_input_file(input_file):
    train_data = []
    test_data = []
    dir_counts = defaultdict(int)
    total_count = 0

    with open(input_file, 'r') as f:
        for line in f:
            total_count += 1
            parts = line.strip().split()
            #print("parts:", parts)
            if len(parts) >= 4:
                file_path = parts[0]
                binary_label = int(parts[1])
                triple_label = int(parts[2])
                cls_label = int(parts[3])
                
                # Count images in each directory
                dir_num = file_path.split('/')[1]  # Changed from [2] to [1]
                dir_counts[dir_num] += 1
                
                # Create data entry
                entry = {
                    "file_path": os.path.join(data_dir, file_path),
                    "binary_cls_label": binary_label,
                    "triple_cls_label": triple_label,
                    "16cls_label": cls_label
                }
                if not os.path.exists(entry["file_path"]):
                    print("entry:", entry)
                    exit()
                
                # Determine split and add to appropriate list
                split_type = get_split_type(file_path)
                if split_type == "train":
                    train_data.append(entry)
                elif split_type == "test":
                    test_data.append(entry)

    return train_data, test_data, dir_counts, total_count

def generate_json(train_data, test_data, output_file):
    output_data = {
        "train": train_data,
        "test": test_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

def main():
    input_file = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/ForgeryNet/OpenDataLab___ForgeryNet/raw/validation/list/Validation/image_list.txt"
    output_file = "/root/autodl-tmp/benchmark_deepfakes/ssl_vits_df/Py_data/ForgeryNet.json"
    
    # Process input file
    train_data, test_data, dir_counts, total_count = process_input_file(input_file)
    
    # Generate output JSON
    generate_json(train_data, test_data, output_file)
    
    # Print statistics
    print(f"\nTotal images in input file: {total_count}")
    print(f"Total images in output JSON: {len(train_data) + len(test_data)}")
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Verify counts match
    if total_count != len(train_data) + len(test_data):
        print("\nERROR: Total count mismatch!")
        print(f"Input file count: {total_count}")
        print(f"Output JSON count: {len(train_data) + len(test_data)}")
    else:
        print("\nCount verification passed!")
    
    # Print directory statistics
    print("\nDirectory statistics:")
    for dir_num in sorted(dir_counts.keys(), key=int):
        print(f"Directory {dir_num}: {dir_counts[dir_num]} images")

if __name__ == "__main__":
    main()
