# color-calibration
This package provides color calibration for images with segmented color calibration cards to make colors internally consistent across the dataset despite variations in lighting conditions.

It relies on:
- The segmented color card from a reference image to specify a reference RGB histogram.

And produces:
- Matched histograms for each segmented color card in the target images, applying each calibration to its respective image.

### Installation
To install in development mode, create and activate a virtual environment, and install this package using:
```bash
pip install -e .[dev]
```

To use in your environment, install directly from GitHub with HTTPS:
```bash
pip install git+https://github.com/Imageomics/color-calibration
```
or with SSH:
```bash
pip install git+ssh://git@github.com/Imageomics/color-calibration.git
```

### Usage
```console
usage: colorcal [-h] --dataset-dir DATASET_DIR --seg-output-dir SEG_OUTPUT_DIR [--reference-image REFERENCE_IMAGE] --save-dir SAVE_DIR
                [--color-card-category COLOR_CARD_CATEGORY] [--detailed-outputs]

Calibrate images using color cards from wingseg COCO annotations

options:
  -h, --help            show this help message and exit
  --dataset-dir DATASET_DIR
                        Base directory containing the original image dataset (default: None)
  --seg-output-dir SEG_OUTPUT_DIR
                        Root directory with segmentation output (must contain a metadata/coco_annotations.json file) (default: None)
  --reference-image REFERENCE_IMAGE
                        Name or relative path of the reference image (default: first image with color card) (default: None)
  --save-dir SAVE_DIR   Directory where calibrated images will be saved (output tree will mirror the input dataset) (default: None)
  --color-card-category COLOR_CARD_CATEGORY
                        Category ID for color card annotations in COCO format (default: 8)
  --detailed-outputs    Enable detailed outputs mode and save extra visuals (e.g. histograms, intermediate images) (default: False)

```

For example, to calibrate a dataset of images containing a color card:
```console
colorcal --dataset-dir <path-to-base-dataset> \
	--seg-output-dir <path-to-segmented-dataset> \
	--save-dir <path-to-calibrated-output> \
	--color-card-category 8 \
	--detailed-outputs
```

## Inputs
> [!IMPORTANT]  
> The package currently requires the `--seg-output-dir` to follow the output formatting of the [`wingseg`](https://github.com/Imageomics/wing-segmentation) package.
> 
> Specifically, it expects the `--seg-output-dir` to contain a `metadata/coco_annotations.json` file with the COCO annotations for the color card segmentation.

If the images from the `--dataset-dir` base directory were resized during segmentation, the color calibration will be applied to the resized images in `--seg-output-dir`. 

If the images were not resized, the color calibration will be applied to the original images in `--dataset-dir`.


## Outputs
- `--save-dir` will contain the full dataset of calibrated images in the same directory structure as the `--dataset-dir` input dataset. The image used as reference will be copied as-is from the input dataset to keep the full dataset together.
- `calibration_log.json` tracks the inputs used for the calibration process.
- If `--detailed-outputs` is enabled, the `--save-dir` will also contain a `detailed-outputs/` directory. This directory will contain:
    - A directory structure matching the `--dataset-dir` input dataset with the following image types:
        - `orig_tgt_color_card_<img-name>.png`
        - `orig_tgt_full_image_<img-name>.png`
        - `ref_color_card.png`
        - `ref_full_image_<img-name>.png`
        - `trans_tgt_color_card_<img-name>.png`
        - `trans_tgt_full_image_<img-name>.png`
    - `histograms/` with a directory structure matching the `--dataset-dir` input dataset. The histograms saved as `hist_<img-name>.html` will show the RGB histograms for the reference color card and the target color card before and after transformation for each image. These may be used to interactively inspect the effect of the color calibration on the images.
