"""
Helper script to create labels.json from Oxford Flowers 102 raw data.
This is needed because the raw dataset comes with .mat files and a text file for labels,
but our generator expects a simple JSON mapping.
"""

import json
import scipy.io
import os
import sys

def create_labels():
    print("Creating labels.json...")
    
    # Check if files exist
    if not os.path.exists('oxford_flowers_102/imagelabels.mat'):
        print("Error: oxford_flowers_102/imagelabels.mat not found")
        sys.exit(1)
        
    if not os.path.exists('oxford_flowers_102/Oxford-102_Flower_dataset_labels.txt'):
         # Try creating it if missing (sometimes wget fails or URL changes)
         # But usually we expect the notebook to have downloaded it
        print("Error: oxford_flowers_102/Oxford-102_Flower_dataset_labels.txt not found")
        sys.exit(1)

    try:
        # Load image labels (1-102 indices for each image)
        mat_data = scipy.io.loadmat('oxford_flowers_102/imagelabels.mat')
        image_labels = mat_data['labels'][0]  # Array of category IDs (1-102)

        # Load category names
        with open('oxford_flowers_102/Oxford-102_Flower_dataset_labels.txt', 'r') as f:
            category_names = [line.strip() for line in f.readlines()]

        # Create category ID to name mapping
        # Dataset IDs are 1-based, list is 0-based
        cat_to_name = {str(i+1): name for i, name in enumerate(category_names)}
        
        # Save validation of categories
        print(f"Loaded {len(cat_to_name)} categories")

        # Get image files
        image_dir = 'oxford_flowers_102/jpg'
        if not os.path.exists(image_dir):
             print(f"Error: {image_dir} not found")
             sys.exit(1)
             
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        print(f"Found {len(image_files)} images")

        # Create labels.json mapping: filename -> flower_name
        labels_dict = {}
        for idx, image_file in enumerate(image_files):
            # The mat file has labels in order of image indices
            # image_00001.jpg corresponds to index 0 in labels array (if sorted correctly)
            # Note: Oxford dataset numbering aligns 1:1 with sorted filenames usually
            if idx < len(image_labels):
                category_id = str(image_labels[idx])
                flower_name = cat_to_name.get(category_id, f"unknown_{category_id}")
                labels_dict[image_file] = flower_name
            else:
                print(f"Warning: No label for {image_file}")

        # Save labels.json
        output_file = 'oxford_flowers_102/labels.json'
        with open(output_file, 'w') as f:
            json.dump(labels_dict, f, indent=2)

        print(f"Success! Created {output_file} with {len(labels_dict)} entries.")
        
    except Exception as e:
        print(f"Error creating labels: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_labels()
