# color-calibration
Provides python tools for calibrating image colors containing color cards.

This repository is under development and may change at any time.


### Installation
To install in development mode, create and activate a virtual environment, and install this package using:
```bash
pip install -e .[dev]
```

To test as part of another project, install directly from GitHub:
```bash
pip install git+https://github.com/Imageomics/color-calibration
```

### Usage
This package requires cropped color cards from reference and target images. The color cards must be in the same orientation (portrait or landscape).

To calibrate an image containing a color card:
```python
from color_calibration import match_histograms
import cv2

ref_card = cv2.imread('path/to/reference/color/card')
tgt_card = cv2.imread('path/to/target/color/card')
tgt_image = cv2.imread('path/to/target/image')

calibrated_image = match_histograms(ref_card, tgt_card, tgt_image)
```
