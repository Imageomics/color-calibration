import json
import numpy as np
import cv2
from pathlib import Path
from pycocotools import mask as mask_utils

def visualize_color_card_extraction(image, segmentation, detailed_outputs_path):
    """
    Debug function to visualize color card extraction steps.
    
    Args:
        image: Original image
        segmentation: RLE segmentation data
        detailed_outputs_path: Base path for saving detailed outputs images
    """
    # Convert detailed_outputs_path to Path object
    detailed_outputs_dir = Path(detailed_outputs_path).parent / 'detailed-outputs'
    detailed_outputs_dir.mkdir(exist_ok=True)
    base_name = Path(detailed_outputs_path).stem
    
    # 1. Decode and save the binary mask
    binary_mask = mask_utils.decode(segmentation)
    cv2.imwrite(
        str(detailed_outputs_dir / f"{base_name}_mask.png"),
        binary_mask.astype(np.uint8) * 255
    )
    
    # 2. Save the masked region on original image
    masked_detailed_outputs = image.copy()
    masked_detailed_outputs[~binary_mask] = 0
    cv2.imwrite(
        str(detailed_outputs_dir / f"{base_name}_masked.png"),
        masked_detailed_outputs
    )
    
    # 3. Get the cropped region
    y_indices, x_indices = np.where(binary_mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return
        
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Draw rectangle on original image
    detailed_outputs_bbox = image.copy()
    cv2.rectangle(detailed_outputs_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.imwrite(
        str(detailed_outputs_dir / f"{base_name}_bbox.png"),
        detailed_outputs_bbox
    )
    
    # Save the cropped and masked result
    cropped = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = binary_mask[y_min:y_max+1, x_min:x_max+1]
    masked_crop = cropped.copy()
    masked_crop[~cropped_mask] = 0
    cv2.imwrite(
        str(detailed_outputs_dir / f"{base_name}_cropped_masked.png"),
        masked_crop
    )
    
    return {
        'mask': binary_mask,
        'bbox': (x_min, y_min, x_max, y_max),
        'cropped_masked': masked_crop
    }
    
def load_coco_annotations(dataset_output_dir):
    """
    Load COCO format annotations from the standard output directory structure.
    
    Args:
        dataset_output_dir (str or Path): Base directory containing segmentation output
        
    Returns:
        dict: Loaded COCO annotations
    """
    coco_path = Path(dataset_output_dir) / "metadata" / "coco_annotations.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO annotations not found at {coco_path}")
        
    with open(coco_path, 'r') as f:
        return json.load(f)

def get_color_card_annotation(coco_data, image_id, color_card_category_id=8):
    """
    Find the color card annotation for a specific image.
    
    Args:
        coco_data (dict): Loaded COCO annotations
        image_id (int): ID of the image to find the color card in
        color_card_category_id (int): Category ID for color card annotations
        
    Returns:
        dict: Color card annotation if found, None otherwise
    """
    for ann in coco_data['annotations']:
        if (ann['image_id'] == image_id and 
            ann['category_id'] == color_card_category_id):
            return ann
    return None

def extract_color_card(image, segmentation):
    """
    Extract the color card region from an image using RLE segmentation.
    
    Args:
        image (np.ndarray): Input image.
        segmentation (dict): RLE segmentation data.
        
    Returns:
        np.ndarray: Cropped color card image.
        np.ndarray: Binary mask of the color card.
    """
    # Decode RLE mask
    binary_mask = mask_utils.decode(segmentation)
    
    # Find bounding box of the mask
    y_indices, x_indices = np.where(binary_mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        raise ValueError("Empty mask - no color card found")
        
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    
    # Crop the image and mask
    cropped_image = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = binary_mask[y_min:y_max+1, x_min:x_max+1].astype(bool)  # Convert to bool here
    
    # Apply mask to image: set pixels outside the mask to 0
    masked_image = cropped_image.copy()
    masked_image[~cropped_mask] = 0
    
    return masked_image, cropped_mask
