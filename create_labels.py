"""
Helper script to create labels.json from Oxford Flowers 102 dataset.
Extracts flower names from the dataset's imagelabels.mat and cat_to_name.json files.
"""

import json
import scipy.io
import os


def create_labels_json(
    imagelabels_mat_path: str,
    cat_to_name_json_path: str,
    output_json_path: str,
    image_dir: str
):
    """
    Create labels.json mapping image filenames to flower names.
    
    Args:
        imagelabels_mat_path: Path to imagelabels.mat file
        cat_to_name_json_path: Path to cat_to_name.json file
        output_json_path: Path to save output labels.json
        image_dir: Directory containing jpg images
    """
    # Load image labels (1-indexed category IDs)
    mat_data = scipy.io.loadmat(imagelabels_mat_path)
    image_labels = mat_data['labels'][0]  # Array of category IDs
    
    # Load category names
    with open(cat_to_name_json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    # Get list of image files
    image_files = sorted([
        f for f in os.listdir(image_dir) 
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    # Create mapping
    labels_dict = {}
    for idx, image_file in enumerate(image_files):
        if idx < len(image_labels):
            category_id = str(image_labels[idx])  # Convert to string for JSON lookup
            flower_name = cat_to_name.get(category_id, f"unknown_{category_id}")
            labels_dict[image_file] = flower_name
    
    # Save to JSON
    with open(output_json_path, 'w') as f:
        json.dump(labels_dict, f, indent=2)
    
    print(f"Created labels.json with {len(labels_dict)} entries")
    print(f"Saved to: {output_json_path}")
    
    # Show sample
    print("\nSample entries:")
    for i, (img, label) in enumerate(list(labels_dict.items())[:5]):
        print(f"  {img}: {label}")


if __name__ == "__main__":
    # Update these paths according to your setup
    IMAGELABELS_MAT = "oxford_flowers_102/imagelabels.mat"
    CAT_TO_NAME_JSON = "oxford_flowers_102/cat_to_name.json"
    OUTPUT_JSON = "oxford_flowers_102/labels.json"
    IMAGE_DIR = "oxford_flowers_102/jpg"
    
    create_labels_json(
        imagelabels_mat_path=IMAGELABELS_MAT,
        cat_to_name_json_path=CAT_TO_NAME_JSON,
        output_json_path=OUTPUT_JSON,
        image_dir=IMAGE_DIR
    )
