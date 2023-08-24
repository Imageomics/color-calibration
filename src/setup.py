from setuptools import setup, find_packages

setup(
    author="David Carlyn",
    description="Tools for calibrating color in images",
    name="color-calibration",
    version='0.0.1',
    packages=find_packages(include=['color-calibration', 'color-calibration.*']),
    install_requires=[
        'numpy',
        'opencv-python'
    ]
)