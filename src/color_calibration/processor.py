import sys
import datetime
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2

from .coco_integration import DatasetPaths
from .coco_utils import get_color_card_annotation, extract_color_card
from .match_histograms import match_histograms
from .detailed_outputs_utils import save_histogram_detailed_outputs

def get_dataset_files(dataset_dir: Path):
    """
    Recursively get all valid image files in the dataset directory by trying to open them with Pillow.
    """
    files = set()
    for file_path in dataset_dir.rglob('*'):
        if file_path.is_file():
            try:
                with Image.open(file_path) as img:
                    img.verify()
                files.add(str(file_path.relative_to(dataset_dir)))
            except Exception:
                # Skip file if PIL doesn't verify as an image
                continue
    return files

def validate_dataset(dataset_dir: Path, coco_data: dict):
    """
    Validate that files in dataset match COCO annotations.
    Returns a tuple: (files_in_both, only_in_dataset, only_in_coco)
    """
    dataset_files = get_dataset_files(dataset_dir)
    coco_files = {img['file_name'] for img in coco_data['images']}
    files_in_both = dataset_files.intersection(coco_files)
    only_in_dataset = dataset_files - coco_files
    only_in_coco = coco_files - dataset_files
    return files_in_both, only_in_dataset, only_in_coco

def write_log_file(args, ref_info):
    """Write a JSON log file at the root of the save directory."""
    log_path = args.save_dir / "calibration_log.json"
    now = datetime.datetime.now().isoformat()
    if args.reference_image:
        ref_spec = "User-specified"
        ref_value = args.reference_image
    else:
        ref_spec = "Default (first image with color card)"
        ref_value = ref_info['file_name']
    log_data = {
        "timestamp": now,
        "command_parameters": {
            "dataset-dir": str(args.dataset_dir),
            "seg-output-dir": str(args.seg_output_dir),
            "reference-image": ref_value,
            "save-dir": str(args.save_dir),
            "color-card-category": args.color_card_category,
            "detailed-outputs": args.detailed_outputs
        },
        "reference_image_used": {
            "file": ref_value,
            "specification": ref_spec
        }
    }
    with open(log_path, "w") as log_file:
        json.dump(log_data, log_file, indent=2)
    print(f"Log file written to {log_path}")

def process_dataset(args):
    """
    Process the dataset using wingseg output (COCO annotations) for segmentation.
    This function mirrors the directory structure of the input dataset when saving outputs.
    """
    dataset = DatasetPaths(args.dataset_dir, args.seg_output_dir)
    files_in_both, only_in_dataset, only_in_coco = validate_dataset(args.dataset_dir, dataset.coco_data)
    if only_in_dataset:
        print("Warning: The following files are in the dataset but not in COCO annotations:", file=sys.stderr)
        for f in sorted(only_in_dataset):
            print(f"  {f}", file=sys.stderr)
    if only_in_coco:
        print("Warning: The following files are in COCO annotations but not in the dataset:", file=sys.stderr)
        for f in sorted(only_in_coco):
            print(f"  {f}", file=sys.stderr)

    # Find images with color cards
    images_with_cards = []
    for img in dataset.coco_data['images']:
        if img['file_name'] in files_in_both:
            has_card = any(
                ann['category_id'] == args.color_card_category
                for ann in dataset.coco_data['annotations']
                if ann['image_id'] == img['id']
            )
            if has_card:
                images_with_cards.append(img)
    if not images_with_cards:
        print("Error: No images with color cards found!", file=sys.stderr)
        sys.exit(1)

    # Select reference image
    if args.reference_image:
        ref_info = dataset.find_image_by_name(args.reference_image)
        if ref_info not in images_with_cards:
            print(f"Error: Reference image {args.reference_image} not found or doesn't have a color card!", file=sys.stderr)
            sys.exit(1)
    else:
        ref_info = images_with_cards[0]
        print(f"Using {ref_info['file_name']} as reference image")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    write_log_file(args, ref_info)

    # Process reference image (preserving directory structure)
    ref_rel_path = Path(ref_info['file_name'])
    ref_path = dataset.get_image_path(ref_info['file_name'])
    ref_image = cv2.imread(str(ref_path))
    if ref_image is None:
        print(f"Error: Could not read reference image: {ref_path}", file=sys.stderr)
        sys.exit(1)
    ref_ann = get_color_card_annotation(dataset.coco_data, ref_info['id'], args.color_card_category)
    if ref_ann is None:
        print(f"Error: No color card found in reference image: {ref_info['file_name']}", file=sys.stderr)
        sys.exit(1)
    ref_card, _ = extract_color_card(ref_image, ref_ann['segmentation'])
    
    # Save the reference full image to the mirrored output structure
    ref_output_path = args.save_dir / ref_rel_path
    ref_output_png = Path(ref_output_path).with_suffix(".png")
    ref_output_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ref_output_png), ref_image)
    print(f"Copied reference full image to {ref_output_path}")
    
    if args.detailed_outputs:
        detailed_images_dir = args.save_dir / "detailed-outputs" / ref_rel_path.parent
        detailed_images_dir.mkdir(parents=True, exist_ok=True)
        ref_detailed_card_path = detailed_images_dir / f"ref_color_card_{ref_rel_path.name}.png"
        ref_detailed_full_path = detailed_images_dir / f"ref_full_image_{ref_rel_path.name}.png"
        cv2.imwrite(str(ref_detailed_card_path), ref_card)
        cv2.imwrite(str(ref_detailed_full_path), ref_image)
    
    # Process target images
    processed_count = 0
    error_count = 0
    for img_info in tqdm(images_with_cards, total=len(images_with_cards), desc="Processing images"):
        if img_info['id'] == ref_info['id']:
            continue
        try:
            target_rel_path = Path(img_info['file_name'])

            output_path = args.save_dir / target_rel_path
            png_output_path = Path(output_path).with_suffix(".png")
            png_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            tgt_path = dataset.get_image_path(img_info['file_name'])
            tgt_image = cv2.imread(str(tgt_path))
            if tgt_image is None:
                raise ValueError(f"Could not read target image: {tgt_path}")
            
            tgt_ann = get_color_card_annotation(dataset.coco_data, img_info['id'], args.color_card_category)
            if tgt_ann is None:
                raise ValueError(f"No color card found in target image: {img_info['file_name']}")
            tgt_card, tgt_mask = extract_color_card(tgt_image, tgt_ann['segmentation'])
            
            calibrated_full_image, calibrated_tgt_card = match_histograms(ref_card, tgt_card, tgt_image)
            
            cv2.imwrite(str(png_output_path), calibrated_full_image)

            if args.detailed_outputs:
                detailed_images_dir = args.save_dir / "detailed-outputs" / target_rel_path.parent
                detailed_images_dir.mkdir(parents=True, exist_ok=True)

                target_filename = Path(target_rel_path).stem

                orig_tgt_color_card_path = detailed_images_dir / f"orig_tgt_color_card_{target_filename}.png"
                orig_tgt_full_image_path = detailed_images_dir / f"orig_tgt_full_image_{target_filename}.png"
                trans_tgt_color_card_path = detailed_images_dir / f"trans_tgt_color_card_{target_filename}.png"
                trans_tgt_full_image_path = detailed_images_dir / f"trans_tgt_full_image_{target_filename}.png"

                cv2.imwrite(str(orig_tgt_color_card_path), tgt_card)
                cv2.imwrite(str(orig_tgt_full_image_path), tgt_image)
                cv2.imwrite(str(trans_tgt_color_card_path), calibrated_tgt_card)
                cv2.imwrite(str(trans_tgt_full_image_path), calibrated_full_image)
                
                # Save histogram outputs mirroring the input structure.
                detailed_histograms_dir = args.save_dir / "detailed-outputs" / "histograms" / target_rel_path.parent
                detailed_histograms_dir.mkdir(parents=True, exist_ok=True)
                hist_detailed_path = detailed_histograms_dir / f"hist_{target_filename}.html"
                save_histogram_detailed_outputs(ref_card, tgt_card, calibrated_tgt_card, hist_detailed_path)
                print(f"Saved detailed outputs for {target_filename}")
            
            processed_count += 1
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            error_count += 1
    
    print("\nProcessing complete:")
    print(f"  Reference image: {ref_info['file_name']}")
    print(f"  Successfully processed: {processed_count} images")
    if error_count:
        print(f"  Errors encountered: {error_count} images")
