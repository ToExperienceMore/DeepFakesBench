import os
from collections import defaultdict

def get_second_dir(file_path):
    """Extract the second directory number from the file path."""
    parts = file_path.split('/')
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None

def split_dataset(input_file, train_output, test_output):
    """Split the dataset into train and test sets based on the second directory number."""
    train_data = []
    test_data = []
    dir_stats = defaultdict(int)  # 用于统计每个目录的图片数量
    
    # Read input file
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split the line into file path and labels
            parts = line.split()
            if len(parts) < 4:
                continue
                
            file_path = parts[0]
            binary_label = parts[1]  # We only need the binary classification label
            
            # Get the second directory number
            dir_num = get_second_dir(file_path)
            if dir_num is None:
                continue
                
            # Update directory statistics
            dir_stats[dir_num] += 1
                
            # Split based on directory number
            if dir_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18]:
                train_data.append(f"{file_path} {binary_label}")
            elif dir_num in [13, 14, 15, 19]:
                test_data.append(f"{file_path} {binary_label}")
    
    # Write train data
    with open(train_output, 'w') as f:
        for line in train_data:
            f.write(line + '\n')
    
    # Write test data
    with open(test_output, 'w') as f:
        for line in test_data:
            f.write(line + '\n')
            
    return len(train_data), len(test_data), dir_stats

def main():
    # Define dataset name and file paths
    dataset_name = "ForgeryNet"
    input_file = "/root/autodl-tmp/benchmark_deepfakes/DeepfakeBench/ForgeryNet/OpenDataLab___ForgeryNet/raw/validation/list/Validation/image_list.txt"
    train_output = f"{dataset_name}_train_list.txt"  # Output file for training set
    test_output = f"{dataset_name}_test_list.txt"    # Output file for test set
    
    # Count total lines in input file
    with open(input_file, 'r') as f:
        total_input_lines = sum(1 for _ in f)
    
    # Split the dataset
    train_count, test_count, dir_stats = split_dataset(input_file, train_output, test_output)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total images in input file: {total_input_lines}")
    print(f"Images in train set: {train_count}")
    print(f"Images in test set: {test_count}")
    print(f"Total after split: {train_count + test_count}")
    
    # Validate the split
    if total_input_lines != train_count + test_count:
        print("\nERROR: Total number of images after split does not match input file!")
        print(f"Difference: {total_input_lines - (train_count + test_count)} images")
    else:
        print("\nValidation passed: Total images match!")
    
    # Print directory statistics
    print("\n=== Directory Statistics ===")
    for dir_num in sorted(dir_stats.keys()):
        print(f"Directory {dir_num}: {dir_stats[dir_num]} images")

if __name__ == "__main__":
    main()
