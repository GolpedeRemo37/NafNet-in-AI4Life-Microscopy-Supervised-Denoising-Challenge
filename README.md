# NAFNet-GAN: An Adversarial Framework for Supervised Fluorescence Microscopy Denoising

A high-performance deep learning framework for biological image restoration based on **NAFNet (Nonlinear Activation-Free Network)** combined with **adversarial (GAN)** and **perceptual losses**. The system is designed to achieve **real-time inference**, **low-latency deployment**, and **high-fidelity recovery of fine biological structures** in microscopy-like data.

This repository implements a hybrid restoration model targeting tasks such as **denoising** and **deblurring** in 2D biological images. The core idea is to leverage the computational efficiency of NAFNet as a generator while enhancing perceptual realism and high-frequency detail preservation through adversarial training and multi-term loss optimization.

---

## Description

The generator architecture is based on NAFNet, an activation-free CNN that replaces traditional nonlinearities with a lightweight multiplicative gating mechanism (SimpleGate). This design significantly reduces computational overhead while maintaining a large effective receptive field and strong restoration capacity.

To compensate for the tendency of purely pixel-wise optimization to oversmooth biological textures, the generator is trained jointly with a **Deep Convolutional Discriminator**, enabling adversarial supervision. The training objective is further enriched by a **MasterLoss** formulation that combines pixel-level accuracy, edge preservation, structural similarity, and perceptual consistency.

The implementation is modular, research-oriented, and optimized for experimentation, while remaining practical for real-world inference pipelines.

---

## Key Features

- **NAFNet Generator**
  - Activation-free design using SimpleGate (channel-wise multiplicative gating)
  - LayerNorm2d normalization
  - Encoder–decoder structure with skip connections
  - Extremely low inference latency and memory footprint

- **Adversarial Training**
  - Deep convolutional discriminator
  - GAN-based supervision to preserve fine textures
  - Stabilized training with BCE-with-logits formulation

- **Hybrid Loss Landscape**
  - Charbonnier / L1-style pixel reconstruction loss
  - Laplacian / frequency-aware components
  - Perceptual loss using VGG16 feature space
  - SSIM-based structural consistency
  - Adversarial GAN loss

- **Dual Data Pipelines**
  - `dataloader.py`: paired datasets (Noisy, Ground Truth)

- **Inference Optimization**
  - Automatic padding to multiples of 32 (NAFNet requirement)
  - Output cropped back to original resolution
  - Handles arbitrary image sizes transparently

---

## Requirements

### Software
- Python 3.8 or higher

### Python Packages
```
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
scikit-image
tqdm
matplotlib
torchmetrics
```

## Hardware

Developed and tested on:
- CPU: AMD Ryzen 7 5800X
- RAM: 32 GB
- GPU: NVIDIA RTX 5060 Ti (16 GB)

CUDA-enabled GPU is **strongly recommended** for training. Inference can run on CPU but with reduced throughput.

## Directory Structure
```
.
├── dataloader.py # Dataset class for paired (Noisy, GT) images
├── dataloader_selfsup.py # Dataset class for self-supervised training with synthetic noise
├── losses.py # MasterLoss and auxiliary loss definitions (VGG, SSIM, GAN)
├── models.py # NAFNet generator and Deep Discriminator architectures
├── Main_Notebook.ipynb # Main training notebook (training loop, logging, visualization)
├── Inference.ipynb # Inference notebook with automatic padding and cropping
└── training_visuals/ # Saved checkpoints and visual validation outputs
```
## Storage

At least **10 GB of free disk space** is recommended to store:
- Model checkpoints
- Intermediate validation outputs
- Training visualizations

## Usage

### Training

Training is performed using `Main_Notebook.ipynb`.

1. Open the notebook and install all required dependencies.
2. Select the appropriate dataset loader:
   - Paired training (Noisy ↔ GT):
     ```python
     from dataloader import DenoisingDataset2D
     ```
   - Self-supervised training (clean images only):
     ```python
     from dataloader_selfsup import DenoisingDataset2D
     ```
3. Configure dataset paths:
   ```python
   INPUT_DIR = "path/to/input_images"
   TARGET_DIR = "path/to/gt_images"  # Only required for paired training
4. Run all cells to initialize:
  - NAFNet generator
  - Deep discriminator
  - Composite loss functions
5. Training checkpoints and visual comparisons are automatically saved to `./training_visuals`


### Inference
Inference is handled in `Inference.ipynb`.

1. Specify the path to a trained .pth checkpoint.
2. Define input and output directories:
  ```
  INPUT_DIR = "path/to/noisy_images"
  OUTPUT_DIR = "path/to/restored_images"
  ```
3. Run all notebook cells.

## Loss Function
Defined in `losses.py` as a composite objective. The training objective combines:

  - Pixel reconstruction loss (L1 / Charbonnier-style)
  - Frequency-aware loss components
  - Perceptual loss using VGG16 feature maps
  - Structural similarity loss (SSIM)
  - Adversarial GAN loss via a deep discriminator

This combination balances numerical accuracy, edge sharpness, perceptual realism, and structural consistency.

## Acknowledgments
The generator architecture is inspired by the NAFNet / NAFSSR family of models introduced by Chu et al.
Perceptual and structural metrics are implemented using torchmetrics.

**NAFNet original paper citation:**

```bibtex
@InProceedings{chu2022nafssr,
    author    = {Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
    title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1239-1248}
}

