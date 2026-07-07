# PoPArt: Pose Estimation & Style Classification in Artworks

**Work in Progress** — This project is under active development. Code, model checkpoints, and results are experimental and subject to change.

Deep learning pipelines for detecting human poses and classifying artistic style in a dataset of paintings, using a custom COCO-style annotated dataset ("PoPArt").

## Table of Contents

- [Overview](#overview)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Outputs](#outputs)
- [Current Limitations](#current-limitations)
- [Notes](#notes)

## Overview

This project explores two related computer vision tasks on a dataset of artwork images (paintings featuring people, e.g. from WikiArt):

1. **Human pose estimation** — detecting body keypoints (17-point COCO skeleton) and bounding boxes of people depicted in paintings, using a fine-tuned Keypoint R-CNN.
2. **Art style classification** — classifying the artistic style (e.g. Impressionism, Cubism, etc.) of each artwork using a ResNet50 image classifier.

Both tasks share the same underlying image dataset and COCO-style JSON annotations (`person_keypoints_train.json`, `person_keypoints_val.json`, `person_keypoints_test.json`), which include keypoints, bounding boxes, and style metadata per image.

## Notebooks

| Notebook | Purpose |
|---|---|
| `Pose_estimation.ipynb` | Data exploration of the pose annotations, custom PyTorch `Dataset` classes, and training of a Keypoint R-CNN (`keypointrcnn_resnet50_fpn`) model on the PoPArt dataset. |
| `poseEstimationTesting.ipynb` | Loads the trained Keypoint R-CNN checkpoint (`popart_keypointrcnn_256.pth`) and evaluates it on the test set — visualizing predicted vs. ground-truth keypoints/skeletons and computing quantitative metrics (keypoint pixel error, bounding box IoU, detection confidence). |
| `style_classification.ipynb` | Builds a `ResNet50`-based image classifier (transfer learning) to predict the artwork's style (`wikiart_style` metadata field), with train/val/test evaluation, confusion matrix, and classification report. |

## Dataset

The notebooks expect the following directory structure (not included in this repository):

```
images/
├── train/
├── val/
└── test/

annotations/
├── person_keypoints_train.json
├── person_keypoints_val.json
└── person_keypoints_test.json
```

Each annotation file follows a COCO-style keypoints format, extended with a `metadata.wikiart_style` field per image used for the style classification task.

## Requirements

- Python 3.x
- `torch`, `torchvision`
- `opencv-python`
- `numpy`, `pandas`
- `matplotlib`
- `scikit-learn`
- `Pillow`
- CUDA-capable GPU recommended for training (CPU fallback supported)

Install with:

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib scikit-learn Pillow
```

## Usage

Clone the repository, place the dataset in the expected structure above, and run the notebooks in this order:

```bash
git clone <your-repo-url>
cd <your-repo-name>
jupyter notebook
```

1. **`Pose_estimation.ipynb`** — explore the data and train the Keypoint R-CNN model (saves a checkpoint, e.g. `popart_keypointrcnn_256.pth`).
2. **`poseEstimationTesting.ipynb`** — load the saved checkpoint and evaluate pose predictions on the test set.
3. **`style_classification.ipynb`** — train and evaluate the ResNet50 style classifier independently.

> Some notebook cells contain exploratory/scratch code (commented-out blocks, alternate dataset class versions) left in from iterative development — expect some cells to require reordering or cleanup before a clean end-to-end run.

## Outputs

**Pose estimation:**
- Visualizations of annotated keypoints/skeletons on cropped person images
- Training loss components from Keypoint R-CNN forward passes
- Predicted vs. ground-truth keypoint overlays on test images
- Histograms of keypoint pixel error, bounding box IoU, and detection confidence scores

**Style classification:**
- Per-epoch training loss/accuracy and validation accuracy
- Test set accuracy and full classification report (precision/recall/F1 per style)
- Raw and row-normalized confusion matrices across style classes
- Sample grid of test images with true vs. predicted style labels

## Current Limitations

- Absolute file paths from the original development environment appear in some cells and will need to be updated to relative paths for portability.
- Several cells are experimental/commented-out alternatives kept for reference rather than a clean linear pipeline.
- No formal train/validation split tuning or hyperparameter search has been done yet for either model.
- Model checkpoints and dataset files are not versioned in this repository.

## Notes

- The pose estimation dataset resizes images to a fixed size (e.g. 256×256) for training via `PoPArtKeypointRCNN256Dataset`.
- The style classifier uses ImageNet normalization statistics and standard augmentation (random resized crop, horizontal flip, rotation) during training.
- Label encoding for style classes is done once via `LabelEncoder` on the training set and reused consistently across validation and test sets.