import cv2
from pathlib import Path
from .coco_utils import (
    load_coco_annotations,
    get_color_card_annotation,
    extract_color_card
)
from .match_histograms import match_histograms

class DatasetPaths:
    """Helper class to manage dataset paths and structure"""
    def __init__(self, dataset_base_dir, dataset_output_dir):
        self.base_dir = Path(dataset_base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Dataset base directory not found: {self.base_dir}")
        self.output_dir = Path(dataset_output_dir)
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Dataset output directory not found: {self.output_dir}")
        
        self.resized_dir = self.output_dir / "resized"
        
        self.coco_data = load_coco_annotations(self.output_dir)
    
    def get_image_path(self, file_name):
        """
        Get full path to an image, checking resized directory first.
        
        Args:
            file_name (str): Relative path to the image file
            
        Returns:
            Path: Full path to the image (resized version if available, otherwise original)
        """
        # Check resized directory first
        if self.resized_dir.exists():
            resized_path = self.resized_dir / file_name
            if resized_path.exists():
                return resized_path
                
            # If file name includes subdirectories, try just the base name
            resized_base_path = self.resized_dir / Path(file_name).name
            if resized_base_path.exists():
                return resized_base_path
        
        # Fall back to original image if resized not found
        original_path = self.base_dir / file_name
        if original_path.exists():
            print(f"Using original image {original_path}; no resized version found")
            return original_path
            
        raise FileNotFoundError(f"Image not found in either resized or original locations: {file_name}")
    
    def find_image_by_name(self, image_name):
        """
        Find image info by file name or path.
        
        Args:
            image_name (str): Name or relative path of the image
            
        Returns:
            dict: Image information from COCO annotations
            
        Raises:
            ValueError: If image not found in annotations
        """
        # Try exact match first
        for img in self.coco_data['images']:
            if img['file_name'] == image_name:
                return img
        
        # Try matching just the file name part
        image_base_name = Path(image_name).name
        for img in self.coco_data['images']:
            if Path(img['file_name']).name == image_base_name:
                return img
        
        raise ValueError(f"Image not found in annotations: {image_name}")


def calibrate_dataset_images(
    dataset_base_dir,
    dataset_output_dir,
    reference_image_name,
    target_image_name,
    color_card_category_id=8,
    output_path=None,
    detailed_outputs=True
):
    # Initialize dataset paths
    dataset = DatasetPaths(dataset_base_dir, dataset_output_dir)
    
    # Get image info from COCO annotations
    ref_info = dataset.find_image_by_name(reference_image_name)
    tgt_info = dataset.find_image_by_name(target_image_name)
    
    # Get color card annotations
    ref_ann = get_color_card_annotation(dataset.coco_data, ref_info['id'], color_card_category_id)
    tgt_ann = get_color_card_annotation(dataset.coco_data, tgt_info['id'], color_card_category_id)
    
    if ref_ann is None:
        raise ValueError(f"No color card found in reference image: {reference_image_name}")
    if tgt_ann is None:
        raise ValueError(f"No color card found in target image: {target_image_name}")
    
    # Resolve image paths (using resized if available)
    ref_path = dataset.get_image_path(ref_info['file_name'])
    tgt_path = dataset.get_image_path(tgt_info['file_name'])
    
    # Load images
    ref_image = cv2.imread(str(ref_path))
    tgt_image = cv2.imread(str(tgt_path))
    
    if ref_image is None:
        raise ValueError(f"Could not read reference image: {ref_path}")
    if tgt_image is None:
        raise ValueError(f"Could not read target image: {tgt_path}")
            
    # Extract the color cards from the images (and get the segmentation mask for the target)
    ref_card, _ = extract_color_card(ref_image, ref_ann['segmentation'])
    tgt_card, tgt_mask = extract_color_card(tgt_image, tgt_ann['segmentation'])
    
    # Perform histogram matching (new version returns a tuple)
    calibrated_full_image, calibrated_tgt_card = match_histograms(ref_card, tgt_card, tgt_image)
    
    # Save the calibrated full image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), calibrated_full_image)
    
    # Save detailed outputs if requested
    if detailed_outputs:
        from .detailed_outputs_utils import save_all_detailed_outputs, save_histogram_detailed_outputs
        detailed_outputs_dir = output_path.parent / "detailed-outputs"
        detailed_outputs_dir.mkdir(parents=True, exist_ok=True)
        base_filename = Path(output_path).name
        
        # Save the detailed output images
        save_all_detailed_outputs(
            ref_image, ref_ann['segmentation'], ref_card,
            tgt_image, tgt_ann['segmentation'], tgt_card, tgt_mask,
            calibrated_full_image, calibrated_tgt_card,
            output_path.parent, base_filename
        )
        
        # Save interactive Plotly histograms
        hist_detailed_outputs_path = output_path.parent / f"hist_{base_filename}.html"
        save_histogram_detailed_outputs(ref_card, tgt_card, calibrated_tgt_card, hist_detailed_outputs_path)
        print(f"Saved detailed outputs histogram figure: {hist_detailed_outputs_path}")
    
    return calibrated_full_image
