# CSPR-Net

**Self-supervised Curved Surface Projection Rectification Network for Geometric Distortion Correction in Non-planar Projections**

## Overview

When a projector casts an image onto a curved surface (e.g. a cylinder), the image appears distorted from the camera's viewpoint. CSPR-Net solves the inverse problem — computing a pre-warped projector input so the camera sees an undistorted result — via a self-supervised cycle-consistent neural network. The method is validated through both ray-tracing simulation and real-world experiments.

A ray-tracing pipeline provides ground-truth pixel mappings for simulation-based training and quantitative evaluation, while the experimental setup demonstrates the approach works on real hardware with captured photographs.

## Project Structure

```
CSPR-Net/
├── generate_ununiform_mesh.py          # Test pattern generation
├── qurdric_transfer.py                 # Ray-tracing simulation & mapping
├── deep_learning_simulation_for_gradient.py  # Neural training on simulated data
├── deep_learning_exp_for_gradient.py   # Neural training on real captured data
├── sim_data/                           # Simulated input images (projector patterns)
├── real_data/                          # Real captured images from experiments
├── data/                               # Ground-truth warping outputs (from ray-tracing)
├── neural_results_simulation/          # Trained models & outputs (simulation)
└── neural_results_exp/                 # Trained models & outputs (experimental)
```

## Scripts

### `generate_ununiform_mesh.py`

Generates colorful test patterns at 1920x1080 used as projector inputs. Produces three grid patterns (varying density) and one solid red image for mask extraction.

- Output: `sim_data/{1,2,3}_pro.png`, `real_data/{1,2,3}_pro.png`, `real_data/4_proj.png`

### `qurdric_transfer.py`

Ray-tracing-based simulation of the full projector → surface → camera pipeline.

- Defines a quadric surface (cylinder by default), camera intrinsics (from physical focal length + sensor size), and projector intrinsics (from throw ratio + offset).
- Traces rays from every pixel through the quadric surface to compute bidirectional mapping tables (`cam_to_proj_map`, `proj_to_cam_map`).
- Uses these maps to generate distorted views, pre-warped images, and verified reconstructions via `cv2.remap`.
- Includes an optional PyVista 3D visualization of the setup (surface mesh, device frustums, ray bundles).

Run: `python qurdric_transfer.py`

### `deep_learning_simulation_for_gradient.py`

Neural compensation trained on **simulated** (ray-tracing) data.

- Two `CoordinateNet` MLPs learn the mapping functions: `C→P` (camera to projector coords) and `P→C` (projector to camera coords).
- Training losses: photometric (L1 + gradient), cycle consistency, smoothness regularization, and mask consistency.
- Uses a `GradientLoss` module with Sobel-filtered edge maps to preserve texture detail.
- Evaluates SSIM, PSNR, and RMSE against ray-tracing ground truth.

Run: `python deep_learning_simulation_for_gradient.py train` or `inference`

### `deep_learning_exp_for_gradient.py`

Neural compensation trained on **real experimental** data (photos of actual projections).

- Same architecture as the simulation version, with added positional encoding support, chunked high-resolution inference (to avoid OOM), loss history CSV export, and displacement field visualization.
- Uses red-channel thresholding for mask extraction from real photos.
- Supports loading pretrained weights for fine-tuning.

Run: `python deep_learning_exp_for_gradient.py train` or `inference`

## Requirements

```
numpy
opencv-python
matplotlib
pyvista
torch
scikit-image
Pillow
```

Install with:

```bash
pip install numpy opencv-python matplotlib pyvista torch scikit-image Pillow
```

> **Note**: PyVista is only needed for the 3D visualization in `qurdric_transfer.py`. If you skip that step, you can omit it.
