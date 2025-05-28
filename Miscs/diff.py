import os
import torch
import argparse
from pathlib import Path
import re

def normalize_filename(filename):
    """Normalize filename by removing common prefixes and suffixes."""
    # Remove _output.pt suffix
    name = filename.replace('_output.pt', '')
    
    # Remove common prefixes
    prefixes_to_remove = [
        'image_0__forward_module.',
        'feature_extractor.base_model.model.',
        'feature_extractor.'
    ]
    for prefix in prefixes_to_remove:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    return name

def find_matching_files(files1, files2):
    """Find matching files between two sets based on normalized names."""
    # Create normalized name to original filename mapping
    norm_to_orig1 = {normalize_filename(f): f for f in files1}
    norm_to_orig2 = {normalize_filename(f): f for f in files2}
    
    # Find common normalized names
    common_norm_names = set(norm_to_orig1.keys()).intersection(set(norm_to_orig2.keys()))
    
    # Map back to original filenames
    matches = [(norm_to_orig1[norm], norm_to_orig2[norm]) for norm in common_norm_names]
    
    return matches

def compare_tensors(tensor1, tensor2, name1, name2):
    """Compare two tensors and print their differences."""
    # Move tensors to CPU for comparison
    tensor1 = tensor1.cpu()
    tensor2 = tensor2.cpu()
    
    diff = torch.abs(tensor1 - tensor2)
    print(f"\nLayer comparison:")
    print(f"Name 1: {name1}")
    print(f"Name 2: {name2}")
    print(f"Shape 1: {tensor1.shape}")
    print(f"Shape 2: {tensor2.shape}")
    print(f"Max difference: {diff.max().item():.6f}")
    print(f"Mean difference: {diff.mean().item():.6f}")
    print(f"Min value 1: {tensor1.min().item():.6f}")
    print(f"Max value 1: {tensor1.max().item():.6f}")
    print(f"Min value 2: {tensor2.min().item():.6f}")
    print(f"Max value 2: {tensor2.max().item():.6f}")

def main():
    parser = argparse.ArgumentParser(description='Compare model outputs from two different directories.')
    parser.add_argument('--dir1', type=str, required=True,
                        help='First directory containing model outputs')
    parser.add_argument('--dir2', type=str, required=True,
                        help='Second directory containing model outputs')
    parser.add_argument('--threshold', type=float, default=1e-6,
                        help='Threshold for significant differences (default: 1e-6)')
    args = parser.parse_args()

    # Convert to Path objects
    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)

    # Get all .pt files from both directories
    files1 = set(f.name for f in dir1.glob('*_output.pt'))
    files2 = set(f.name for f in dir2.glob('*_output.pt'))
    
    # Find matching files
    matching_pairs = find_matching_files(files1, files2)
    
    if not matching_pairs:
        print("No matching output files found between the two directories!")
        return

    print(f"Found {len(matching_pairs)} matching output files to compare")
    
    # Compare each matching pair
    significant_diffs = []
    for filename1, filename2 in sorted(matching_pairs):
        try:
            # Load tensors and move to CPU
            tensor1 = torch.load(dir1 / filename1, map_location='cpu')
            tensor2 = torch.load(dir2 / filename2, map_location='cpu')
            
            # Compare tensors
            diff = torch.abs(tensor1 - tensor2)
            mean_diff = diff.mean().item()
            
            if mean_diff > args.threshold:
                significant_diffs.append((filename1, filename2, mean_diff))
                compare_tensors(tensor1, tensor2, filename1, filename2)
                
        except Exception as e:
            print(f"Error comparing {filename1} with {filename2}: {str(e)}")

    # Print summary
    if significant_diffs:
        print("\nSummary of significant differences:")
        for filename1, filename2, mean_diff in sorted(significant_diffs, key=lambda x: x[2], reverse=True):
            print(f"{filename1} <-> {filename2}: {mean_diff:.6f}")
    else:
        print("\nNo significant differences found!")

if __name__ == '__main__':
    main()