import argparse
from pathlib import Path
from .processor import process_dataset

def create_parser():
    parser = argparse.ArgumentParser(
        description="Calibrate images using color cards from wingseg COCO annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path, 
        required=True,
        help="Base directory containing the original image dataset"
    )
    parser.add_argument(
        "--seg-output-dir",
        type=Path,
        required=True,
        help="Root directory with segmentation output (must contain a metadata/coco_annotations.json file)"
    )
    parser.add_argument(
        "--reference-image", 
        type=str,
        help="Name or relative path of the reference image (default: first image with color card)"
    )
    parser.add_argument(
        "--save-dir", 
        type=Path, 
        required=True,
        help="Directory where calibrated images will be saved (output tree will mirror the input dataset)"
    )
    parser.add_argument(
        "--color-card-category", 
        type=int, 
        default=8,
        help="Category ID for color card annotations in COCO format"
    )
    parser.add_argument(
        "--detailed-outputs", 
        action="store_true",
        help="Enable detailed outputs mode and save extra visuals (e.g. histograms, intermediate images)"
    )
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    process_dataset(args)

if __name__ == "__main__":
    main()
