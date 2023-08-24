from setuptools import setup, find_packages

setup(
    author="David Carlyn",
    description="Tools for calibrating color in images",
    name="color_calibration",
    version='0.0.1',
    url="https://github.com/Imageomics/color-calibration",
    packages=find_packages(where="color_calibration"),
    package_dir={"": "color_calibration"},
    install_requires=[
        'numpy',
        'opencv-python'
    ]
)