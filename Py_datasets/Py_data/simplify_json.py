import json
import os

def simplify_json(input_file, output_file):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create a new simplified structure
    simplified_data = {}
    
    # Process each dataset (train, test, val)
    for dataset in ['train', 'test', 'val']:
        if dataset in data['UADFV']['UADFV_Real']:
            simplified_data[dataset] = {}
            
            # Process each video
            for video_id, video_data in data['UADFV']['UADFV_Real'][dataset].items():
                frames = video_data['frames']
                if len(frames) >= 2:
                    # Keep only first and last frame
                    simplified_data[dataset][video_id] = {
                        'label': video_data['label'],
                        'frames': [frames[0], frames[-1]]
                    }
    
    # Write the simplified JSON to output file
    with open(output_file, 'w') as f:
        json.dump(simplified_data, f, indent=2)

if __name__ == '__main__':
    input_file = 'UADFV.json'
    output_file = 'UADFV_simplified.json'
    simplify_json(input_file, output_file)
