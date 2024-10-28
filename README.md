# Satellite Image Processing with U-Net

This repository contains an implementation of semantic segmentation using the U-Net architecture. The model is trained on a dataset of satellite images, performing pixel-level classification to identify different classes within the images.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)

## Dataset

The dataset used in this project is a semantic segmentation dataset of satellite images. The images and their corresponding masks are stored in the following directory structure:

data_root_folder/<br />
├── Semantic segmentation dataset/<br />
├── images/<br />
├── masks/<br />

- **images/**: Contains the input images in `.jpg` format.
- **masks/**: Contains the segmentation masks in `.png` format.


## Requirements

Ensure you have the following libraries installed:

- `opencv-python`
- `numpy`
- `Pillow`
- `matplotlib`
- `scikit-learn`
- `tensorflow`
- `keras`
- `segmentation_models`
- `patchify`
