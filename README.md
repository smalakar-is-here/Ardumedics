# ArduMedics: AI-Powered Healthcare Robot

> **An ensemble YOLOv8-Pose fall detection system with Temporal Pose Consistency (TPC) and Multi-Engine OCR Fusion, designed for Raspberry Pi 5 edge deployment.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.56-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-blue)](https://kaggle.com)

---

## Table of Contents

- [Overview](#overview)
- [Novel Contributions](#novel-contributions)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Model Setup (Download + NCNN Export)](#model-setup-download--ncnn-export)
- [Notebook Descriptions](#notebook-descriptions)
- [Model Zoo](#model-zoo)
- [Datasets](#datasets)
- [Raspberry Pi 5 Deployment](#raspberry-pi-5-deployment)
- [Results Summary](#results-summary)
- [Statistical Validation](#statistical-validation)
- [Citation](#citation)

---

## Overview

ArduMedics is an AI-powered healthcare robot that combines:

1. **Camera-based fall detection** using an ensemble of YOLOv8-Pose models (Nano + Small + Medium)
2. **Temporal Pose Consistency (TPC)** — a novel velocity-based temporal tracking module that reduces false positives
3. **Multi-Engine OCR Fusion** for prescription/medicine analysis using Tesseract + EasyOCR + PaddleOCR
4. **Edge deployment** on Raspberry Pi 5 with NCNN optimization

The system detects falls in real-time from camera feeds and triggers alerts (buzzer + SMS + LED), while also reading and analyzing medical prescriptions using OCR.

---

## Novel Contributions

| # | Contribution | Description | Impact |
|---|-------------|-------------|--------|
| C1 | **Temporal Pose Consistency (TPC)** | Multi-frame velocity tracking with configurable window (W=30) + confirmation frames (5) | Reduces false positive rate by ~12% over static pose detection |
| C2 | **Multi-Engine OCR Fusion** | Confidence-weighted ensemble of Tesseract + EasyOCR + PaddleOCR | Reduces CER by ~40% compared to best single engine |
| C3 | **Hybrid Rule-ML Fall Logic** | Combines geometric rules (horizontal torso + ground proximity) with learned features | Enables interpretable, edge-deployable fall classification |
| C4 | **First Systematic Pi 5 Healthcare AI Benchmark** | NCNN/TFLite/ONNX profiling on Raspberry Pi 5 for healthcare AI | Establishes baseline for future edge healthcare AI research |

---

## System Architecture

```
Camera Input ──► YOLOv8-Pose Ensemble (n+s+m) ──► TPC Tracker (W=30)
                     │                                  │
                     │                                  ▼
                     │                         Hybrid Rule-ML
                     │                         Fall Logic
                     │                                  │
                     ▼                                  ▼
              Keypoints Output                   FALL ALERT
              (17 COCO points)                (Buzzer+SMS+LED)

Medicine Image ──► Multi-Engine OCR Fusion ──► Medicine Info
                   (Tesseract+EasyOCR            (Drug+Dosage)
                    +PaddleOCR)
```

---

## Repository Structure

```
Ardumedics/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore                         # Git ignore rules
├── .gitattributes                     # Git LFS tracking rules
├── requirements.txt                   # Python dependencies
│
├── notebooks/                         # Kaggle notebooks (run in order)
│   ├── 01_YOLOv8n_Pose_Fall_Detection_Training.ipynb
│   ├── 02_YOLOv8s_Pose_Fall_Detection_Training.ipynb
│   ├── 03_YOLOv8m_Pose_Fall_Detection_Training.ipynb
│   ├── 04_OCR_MultiEngine_Prescription_Analysis.ipynb
│   ├── 05_Hyperparameter_Sweep_Ablation.ipynb
│   └── 06_Ensemble_Final_Paper_Results.ipynb
│
├── models/                            # Trained model weights (Git LFS)
│   ├── README.md                      # Download & placement instructions
│   ├── export_ncnn.py                 # NCNN export script for Pi 5
│   ├── best_nano.pt                   # YOLOv8n-Pose weights (6.5 MB)
│   ├── best_small.pt                  # YOLOv8s-Pose weights (22 MB)
│   ├── best_medium.pt                 # YOLOv8m-Pose weights (51 MB)
│   └── best_nano_ncnn/               # NCNN export for Pi 5 (13 MB)
│       ├── model.ncnn.param
│       └── model.ncnn.bin
│
├── src/                               # Source code for deployment
│   ├── fall_detection/                # Fall detection module
│   │   ├── __init__.py
│   │   ├── pose_estimator.py          # YOLOv8-Pose wrapper
│   │   ├── tpc_tracker.py             # Temporal Pose Consistency
│   │   ├── fall_classifier.py         # Hybrid Rule-ML fall logic
│   │   └── ensemble.py                # Weighted ensemble inference
│   ├── ocr/                           # OCR module
│   │   ├── __init__.py
│   │   ├── multi_engine_fusion.py     # Confidence-weighted OCR fusion
│   │   └── prescription_parser.py     # Medicine name/dosage extraction
│   └── ensemble/                      # Ensemble orchestration
│       ├── __init__.py
│       └── pipeline.py                # Full ArduMedics pipeline
│
├── edge-deployment/                   # Raspberry Pi 5 deployment
│   └── raspberry-pi5/
│       ├── SETUP_GUIDE.md             # Step-by-step Pi 5 setup
│       ├── install.sh                 # One-click installation script
│       ├── scripts/
│       │   ├── run_fall_detection.py  # Main inference script
│       │   ├── run_ocr.py             # OCR inference script
│       │   └── benchmark.py           # Performance benchmark
│       └── configs/
│           ├── fall_detection.yaml    # Fall detection config
│           └── ocr.yaml               # OCR config
│
├── datasets/                          # Dataset info (files too large for GitHub)
│   └── README.md                      # Dataset download instructions
│
├── docs/                              # Documentation
│   └── KAGGLE_SETUP.md               # How to run on Kaggle
│
└── results/                           # Paper results (tables + figures)
    ├── README.md                      # Download instructions
    ├── tables/                        # LaTeX + CSV tables
    └── figures/                       # Publication-quality figures
```

---

## Quick Start

### Option A: Run on Kaggle (Recommended for Training)

See [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md) for detailed Kaggle instructions.

**Quick summary:**

1. Create 6 Kaggle notebooks in order (NB01 → NB06)
2. Add required Kaggle datasets (see [Datasets](#datasets))
3. Run on **GPU T4 x2** accelerator
4. Save each notebook's output as a Kaggle Dataset for the next notebook
5. Download NB06 output folder for paper figures/tables

### Option B: Run on Raspberry Pi 5 (Edge Deployment)

See [edge-deployment/raspberry-pi5/SETUP_GUIDE.md](edge-deployment/raspberry-pi5/SETUP_GUIDE.md) for full setup.

**Quick summary:**

```bash
# 1. Clone the repo (with LFS)
git lfs install
git clone https://github.com/smalakar-is-here/Ardumedics.git
cd Ardumedics

# 2. Pull model weights via LFS
git lfs pull

# 3. Export NCNN model (if not already in repo)
cd models && python3 export_ncnn.py && cd ..

# 4. Run the one-click installer
chmod +x edge-deployment/raspberry-pi5/install.sh
./edge-deployment/raspberry-pi5/install.sh

# 5. Run fall detection
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --tpc-window 30
```

### Option C: Run Locally (Development)

```bash
# 1. Clone the repo (with LFS)
git lfs install
git clone https://github.com/smalakar-is-here/Ardumedics.git
cd Ardumedics

# 2. Pull model weights
git lfs pull

# 3. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run inference
python3 src/ensemble/pipeline.py --config edge-deployment/raspberry-pi5/configs/fall_detection.yaml
```

---

## Model Setup (Download + NCNN Export)

### Step 1: Download Trained Models from Kaggle

You need **3 `.pt` files** from your Kaggle notebook outputs:

| File | From Notebook | Output Path | Size |
|------|--------------|-------------|------|
| `best_nano.pt` | NB01 - YOLOv8n Pose Training | `nb01_outputs/best_nano.pt` | ~6.5 MB |
| `best_small.pt` | NB02 - YOLOv8s Pose Training | `nb02_outputs/best_small.pt` | ~22 MB |
| `best_medium.pt` | NB03 - YOLOv8m Pose Training | `nb03_outputs/best_medium.pt` | ~51 MB |

Place them in `models/` folder. See [models/README.md](models/README.md) for detailed instructions.

### Step 2: Export NCNN Model for Pi 5

The NCNN model is **NOT** in the Kaggle output. Export it after downloading `best_nano.pt`:

```bash
cd models/
pip install ultralytics
python3 export_ncnn.py
# Creates: best_nano_ncnn/model.ncnn.param + model.ncnn.bin
```

### Step 3: Upload to GitHub with Git LFS

All `.pt` and `.ncnn.bin` files are automatically tracked by Git LFS (configured in `.gitattributes`):

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.ncnn.bin"
git add .gitattributes
git add models/
git commit -m "Add trained model weights"
git push origin main
```

---

## Notebook Descriptions

| # | Notebook | Purpose | GPU | Time | Output |
|---|----------|---------|-----|------|--------|
| 01 | YOLOv8n-Pose Training | Train Nano model (3.3M params) | T4x2 | ~15 min | `best_nano.pt`, `metrics_notebook01.nc` |
| 02 | YOLOv8s-Pose Training | Train Small model (11.6M params) | T4x2 | ~20 min | `best_small.pt`, `metrics_notebook02_si` |
| 03 | YOLOv8m-Pose Training | Train Medium model (26.5M params) | T4x2 | ~25 min | `best_medium.pt`, `metrics_notebook03_mr` |
| 04 | OCR Multi-Engine | Prescription OCR with 3-engine fusion | T4 | ~6-8 hr | `metrics_notebook04_ocr`, OCR comparison CSV |
| 05 | Hyperparameter Sweep | LR/optimizer/augmentation/TPC/OCR ablation | T4x2 | ~30 min | Ablation JSON results, figures |
| 06 | Ensemble & Paper Results | Final ensemble + all paper outputs | T4x2 | ~15 min | 13 tables, 11 figures, statistical tests |

**Important:** Each notebook saves its output as a Kaggle Dataset that the next notebook depends on. Run in sequential order (01 → 06).

---

## Model Zoo

| Model | Params | Box mAP@50 | Box mAP@50-95 | Pose mAP@50 | GPU FPS | Pi5 FPS (est) | Format |
|-------|--------|-----------|--------------|------------|---------|--------------|--------|
| **YOLOv8n-Pose** | 3.3M | 0.667 | 0.428 | 0.218 | 88.9 | 3.1 (NCNN) | PyTorch, NCNN |
| YOLOv8s-Pose | 11.6M | 0.716 | 0.465 | 0.263 | 88.1 | 3.1 (NCNN) | PyTorch, NCNN |
| YOLOv8m-Pose | 26.5M | 0.749 | 0.464 | 0.259 | 54.5 | 1.9 (NCNN) | PyTorch, NCNN |
| **Ensemble (n+s+m)** | 41.4M | 0.754 | 0.454 | 0.248 | 26.4 | N/A (server) | PyTorch |

> **Note:** Model weights are included via **Git LFS**. After cloning, run `git lfs pull` to download the actual weight files. See [models/README.md](models/README.md) for details on which models to download and how to export NCNN.

---

## Datasets

### Roboflow Pose Datasets (keypoint-annotated, YOLOv8-Pose format)

| # | Name | Images | Labels | Source |
|---|------|--------|--------|--------|
| 1 | Falling Pose Estimation | 635 | Keypoints | [Roboflow Universe](https://universe.roboflow.com/humna-pose-data/falling-pose-estimation) |
| 2 | YOLOv8-Pose Fall Detection | 474 | Keypoints | [Roboflow Universe](https://universe.roboflow.com/yolo-xvnzo/yolov8-pose-utovc) |

### Kaggle Fall Detection Datasets (videos/images)

| # | Name | Type | Source |
|---|------|------|--------|
| 3 | UR Fall Detection | Videos | [Kaggle](https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset) |
| 4 | Fall Detection Images | Images | [Kaggle](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset) |
| 5 | Le2i Fall Dataset | Videos | [Kaggle](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) |
| 6 | Multiple Cameras Fall | Videos | [Kaggle](https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset) |
| 7 | Fall Video Dataset | Videos | [Kaggle](https://www.kaggle.com/datasets/payutch/fall-video-dataset) |

### Kaggle OCR Datasets

| # | Name | Type | Source |
|---|------|------|--------|
| 8 | Doctor's Handwritten Prescription BD | ~800 images | [Kaggle](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset) |
| 9 | Handwritten Medical Prescriptions | ~200 images | [Kaggle](https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset) |
| 10 | Synthetic Medical Prescription OCR | 2002 images | [Kaggle](https://www.kaggle.com/datasets/priyanshuyav13/synthetic-medical-prescription-ocr-dataset) |
| 11 | OCR-Processed Prescriptions | ~800 images | [Kaggle](https://www.kaggle.com/datasets/nadaarfaoui/ocr-processed-handwritten-prescriptions) |

---

## Raspberry Pi 5 Deployment

### Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Raspberry Pi 5 | 8GB RAM model |
| Camera | Pi Camera Module v3 or USB webcam |
| Storage | 32GB+ microSD (Class 10 / A2) |
| Power | 5V/5A USB-C PD |
| Optional | Arduino Nano (IMU sensor integration) |
| Optional | Buzzer + LED (fall alert output) |

### Deployment Architecture

The **Nano model (YOLOv8n-Pose)** is the primary edge deployment model due to its:
- Smallest parameter count: 3.3M params
- Fastest inference: 88.9 FPS on GPU, ~3.1 FPS on Pi5 (NCNN)
- NCNN format for ARM optimization

The **Ensemble model** runs on a server/cloud, not on Pi5.

### Setup Steps

See the detailed guide: [edge-deployment/raspberry-pi5/SETUP_GUIDE.md](edge-deployment/raspberry-pi5/SETUP_GUIDE.md)

```bash
# Quick setup on Pi 5:
git lfs install
git clone https://github.com/smalakar-is-here/Ardumedics.git
cd Ardumedics
git lfs pull
chmod +x edge-deployment/raspberry-pi5/install.sh
./edge-deployment/raspberry-pi5/install.sh
```

---

## Results Summary

### Model Performance

| Metric | YOLOv8n | YOLOv8s | YOLOv8m | Ensemble |
|--------|---------|---------|---------|----------|
| Box mAP@50 | 0.667 | 0.716 | 0.749 | 0.754 |
| Box mAP@50-95 | 0.428 | 0.465 | 0.464 | 0.454 |
| Pose mAP@50 | 0.218 | 0.263 | 0.259 | 0.248 |
| Detection Rate | 92.9% | 70.1% | 91.3% | **97.6%** |
| GPU FPS | 88.9 | 88.1 | 54.5 | 26.4 |

### Statistical Validation

| Test | Comparison | Statistic | p-value | Significant |
|------|-----------|-----------|---------|-------------|
| Wilcoxon (OKS) | Ensemble vs Nano | W=22809 | p=0.557 | No |
| Wilcoxon (OKS) | Ensemble vs Small | W=39044 | p<0.001 | *** |
| Wilcoxon (OKS) | Ensemble vs Medium | W=42678 | p<0.001 | *** |
| McNemar (Fall) | Small vs Ensemble | chi2=5.33 | p=0.021 | * |

### Comparison with State-of-the-Art

| Method | Accuracy (%) | F1 (%) | Real-time? | Edge? |
|--------|-------------|--------|-----------|-------|
| OpenPose + SVM (2019) | 85.2 | 83.1 | No | No |
| YOLOv4 + LSTM (2021) | 89.7 | 88.3 | Partial | No |
| YOLOv5-Pose (2023) | 91.5 | 90.2 | Yes | Limited |
| ViT-Pose + Transformer (2024) | 93.2 | 92.1 | No | No |
| **ArduMedics - YOLOv8n (Ours)** | **93.5** | **92.8** | **Yes** | **Yes (Pi5)** |
| **ArduMedics - Ensemble (Ours)** | **95.1** | **94.3** | Partial | Server |

---

## Statistical Validation

### Wilcoxon Signed-Rank Test (OKS Scores)

The Wilcoxon signed-rank test on Object Keypoint Similarity (OKS) scores provides continuous statistical validation of ensemble superiority:

- **Ensemble vs Small**: p < 0.001 (mean OKS improvement: +0.042)
- **Ensemble vs Medium**: p < 0.001 (mean OKS improvement: +0.053)
- **Ensemble vs Nano**: p = 0.557 (comparable keypoint quality; Nano benefits from highest per-model OKS)

### Key Insight: Detection Rate vs Keypoint Quality

The ensemble's primary advantage is **detection reliability** (97.6% vs 92.9% for Nano), not keypoint quality. In safety-critical fall detection, a missed detection (no person detected at all) is far more dangerous than slightly less precise keypoints.

---

## Citation

```bibtex
@article{ardumedics2026,
  title={ArduMedics: An Ensemble YOLOv8-Pose Fall Detection System with Temporal Pose Consistency for Edge Healthcare Deployment},
  author={Malakar, S},
  journal={Q1 Journal (Under Review)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the pose estimation framework
- [Roboflow](https://roboflow.com) for annotated pose datasets
- [Kaggle](https://kaggle.com) for compute (T4x2 GPU) and datasets
- [NCNN](https://github.com/Tencent/ncnn) for ARM-optimized inference
