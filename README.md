# Multi-attention Mechanism for Enhanced Pseudo-3D Prostate Zonal Segmentation

This repository contains the **official code implementation** of our paper:

**Multi-attention Mechanism for Enhanced Pseudo-3D Prostate Zonal Segmentation**  
📄 *Published in Journal of Digital Imaging, 2025*  
🔗 [Read the paper](https://link.springer.com/article/10.1007/s10278-025-01401-0)

---

## 📌 Overview

We propose a novel deep learning architecture that incorporates a multi-attention mechanism to enhance prostate zonal segmentation in pseudo-3D MR images. Our method improves segmentation accuracy across the transition and peripheral zones by effectively capturing inter-slice context and channel-spatial dependencies.

---

## 🧠 Key Features

- Pseudo-3D multi-slice input strategy for contextual learning
- Hybrid channel-spatial attention modules for enhanced feature representation
- State of art performance on peripheral zone segmentation
- Superior performance on multi-center prostate MRI datasets
- Modular and extensible PyTorch-based implementation

---

## 🧾 Dataset Preparation

This implementation is designed to work with batched prostate MRI datasets stored as 2D volumes. Before training or inference, images should be preprocessed as follows:

### ✅ Preprocessing Steps

1. **Center Cropping**:  
   All images are center-cropped to focus on the prostate region.

2. **Resizing**:  
   Cropped images are reshaped to a fixed spatial resolution of **128 × 128** pixels.

3. **Normalization**:  
   Image intensities are normalized to the range **[0, 255]** for consistency across different centers and scanners.

4. **Batch Size Support**:  
   The current implementation supports batch sizes of **2**, **4**, **16**, and **20**. Choose based on available GPU memory.

5. **Multi-class Segmentation**:
   Expects ground truth to have 3 classes (0=Background, 1=Transitional Zone, 2=Peripheral Zone).

> ⚠️ Make sure to adjust your DataLoader and augmentation settings to match these preprocessing steps.

---

## 📚 Datasets Used

The models in this repository have been **trained and evaluated** on the following datasets:

- **[ProstateX](https://www.cancerimagingarchive.net/collection/prostatex/)**:  
  A publicly available prostate MRI dataset containing T2-weighted images with zonal and lesion annotations.

- **Proprietary Clinical Datasets**:  
  Additional internal datasets from collaborating institutions were used to validate model generalizability across different acquisition protocols and scanner vendors.

> 🛡️ Due to data sharing restrictions, the proprietary datasets cannot be released. However, the model generalizes well to other prostate MRI datasets following the same preprocessing pipeline.

---

## 📦 Requirements

This project is implemented in **Python 3.9+** and built on **PyTorch**. Below are the main dependencies required to run the code:

### 🐍 Python Dependencies

- `torch >= 1.10`
- `torchvision >= 0.11`
- `numpy`
- `opencv-python`
- `scikit-image`
- `matplotlib`
- `tqdm`
- `pydicom`
- `SimpleITK`
- `scipy`
- `albumentations`
- `nibabel` (if working with NIfTI files)
- `pandas`
- `pyyaml`

---

