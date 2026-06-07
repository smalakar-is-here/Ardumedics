# ArduMedics: Smart Healthcare Assistant Robot

> **An AI-powered healthcare robot combining IoT, Robotics, and AI for 24/7 patient support — featuring contactless care, smart medicine management, autonomous navigation, and real-time fall detection with prescription OCR.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.56-green)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-blue)](https://kaggle.com)

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Hardware Components](#hardware-components)
- [Repository Structure](#repository-structure)
- [AI System Setup (Fall Detection + OCR)](#ai-system-setup-fall-detection--ocr)
- [Raspberry Pi 5 Deployment](#raspberry-pi-5-deployment)
- [Kaggle Training (Reproducing Results)](#kaggle-training-reproducing-results)
- [Model Zoo](#model-zoo)
- [Results Summary](#results-summary)
- [Comparison with State-of-the-Art](#comparison-with-state-of-the-art)
- [Statistical Validation](#statistical-validation)
- [Future Improvements](#future-improvements)
- [Team](#team)
- [Datasets](#datasets)
- [License](#license)

---

## Overview

ArduMedics is an innovative healthcare assistant robot that fuses **IoT, robotics, and AI** to provide 24/7 patient support. By automating repetitive tasks, it allows medical staff to focus entirely on critical care, ensuring a safer healthcare environment. The system combines contactless health monitoring, smart medicine management, autonomous navigation, and AI-powered fall detection into a single robotic platform.

### Core Capabilities

1. **Contactless Care** — Reduces healthcare worker exposure to contagious patients through virtual nurse assistance, thermal monitoring, blood pressure monitoring, and real-time ECG tracking using the AD8232 sensor. Safely transports medical equipment and supplies to patient bedsides without human contact.

2. **Smart IoT & Medicine Management** — Features a centralized web dashboard where doctors input prescriptions, instructions, and dosage timings. The robot syncs with the dashboard to deliver the correct medicine to the right bed at the exact prescribed time, with built-in UV-C sanitization on the medicine tray.

3. **Autonomous Navigation** — Uses a Line Following Robot (LFR) mechanism coupled with ultrasonic/LiDAR sensors for smooth obstacle avoidance in busy hospital corridors. Features an automatic charging dock for self-sustaining operation when battery is low.

4. **AI-Powered Fall Detection** — Camera-based real-time fall detection using an ensemble of three YOLOv8-Pose models (Nano + Small + Medium) with Temporal Pose Consistency (TPC) tracking. Achieves 97.6% detection rate and triggers emergency alerts (buzzer + LED).

5. **Prescription OCR Analysis** — Multi-engine OCR fusion (Tesseract + EasyOCR + PaddleOCR) for reading and analyzing medical prescriptions, extracting medicine names, dosages, and schedules with confidence-weighted ensemble output.

---

## Problem Statement

ArduMedics addresses six critical challenges in modern healthcare:

| # | Problem | Description |
|---|---------|-------------|
| 01 | **Pandemic & Infection Risks** | During crises like the Corona pandemic, the high risk of infection prevents doctors and nurses from providing close, frequent care. |
| 02 | **Healthcare Staff Shortage** | Limited staff availability strains the system, especially when medical workers themselves fall ill or are quarantined. |
| 03 | **The Monitoring Gap** | Rapidly changing patient conditions require continuous monitoring, which is nearly impossible to maintain manually during a surge in patients. |
| 04 | **Delayed Emergency Response** | High patient-to-staff ratios lead to delayed responses during life-threatening emergencies. |
| 05 | **Human Error in High-Stress Zones** | In high-pressure environments, manual medicine management is prone to errors, jeopardizing patient safety. |
| 06 | **Resource Inefficiency** | Repetitive tasks and manual workflows create delays and unnecessary expenses during critical healthcare operations. |

---

## Key Features

### Feature 1 — Minimizing Human Interaction (Contactless Care)

- **Protecting Medical Staff:** Reduces the exposure of healthcare workers to highly contagious and sensitive cases (e.g., COVID-19).
- **Virtual Nurse Assistant:** Acts as a reliable proxy between the patient and the doctor.
- **Contactless Health Monitoring:**
  - Thermal monitoring for continuous temperature tracking
  - Blood Pressure (BP) monitoring
  - Real-time ECG monitoring using the AD8232 sensor
- **Safe Logistic Delivery:** Safely transports medical equipment and supplies to the patient's bedside without human contact.

### Feature 2 — Smart IoT & Medicine Management

- **Centralized Web Dashboard:** A smart platform where doctors can easily input prescriptions, instructions, and dosage timings.
- **Automated Medicine Delivery:** The robot syncs with the dashboard to ensure the correct medicine is served to the right bed at the exact prescribed time.
- **Built-in UV-C Sanitization:** The medicine tray features a UV-C module to automatically sanitize the environment, significantly reducing contamination risks.

### Feature 3 — Autonomous Navigation

- **Intelligent Pathfinding:** Utilizes a Line Following Robot (LFR) mechanism coupled with Ultrasonic/LiDAR sensors for smooth obstacle avoidance in busy hospital corridors.
- **Self-Sustaining Operation:** Features an automatic charging dock, allowing the robot to return and recharge itself when the battery is low.

### Feature 4 — AI-Powered Response

- **Real-time Fall Detection:** Camera-based fall detection using YOLOv8-Pose ensemble with Temporal Pose Consistency (TPC). Detects falls in real-time from camera feeds and triggers emergency alerts.
- **Prescription OCR Analysis:** Multi-engine OCR fusion (Tesseract + EasyOCR + PaddleOCR) reads prescription images and extracts medicine name, dosage, and schedule for review.
- **Human Verification Required:** AI-generated prescriptions and fall alerts require human verification before execution, ensuring safety.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ArduMedics Robot                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Control Unit │  │   Sensors    │  │   AI Processing      │  │
│  │              │  │              │  │                      │  │
│  │ Raspberry    │  │ AD8232 (ECG) │  │ YOLOv8-Pose Ensemble │  │
│  │ Pi 5         │  │ MAX30102     │  │ (n+s+m models)       │  │
│  │ (Main Ctrl)  │  │ (SPO2/HR)    │  │                      │  │
│  │              │  │ MPU6050      │  │ TPC Tracker (W=30)   │  │
│  │ Arduino Nano │  │ (Gyro/Accel) │  │                      │  │
│  │              │  │ Ultrasonic   │  │ Hybrid Rule-ML       │  │
│  │ ESP32        │  │ Temperature  │  │ Fall Logic           │  │
│  │ (Hub)        │  │ IR Line      │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  │ Multi-Engine OCR     │  │
│         │                 │          │ Fusion               │  │
│         │                 │          └──────────┬───────────┘  │
│         │                 │                     │              │
│  ┌──────┴─────────────────┴─────────────────────┴───────────┐  │
│  │                    System Bus                             │  │
│  └──────┬─────────────────┬─────────────────────┬───────────┘  │
│         │                 │                     │              │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌─────────┴──────────┐   │
│  │  Mechanics   │  │  Medicine    │  │   Alerts &         │   │
│  │              │  │  Management  │  │   Display          │   │
│  │ DC Motors    │  │              │  │                    │   │
│  │ L298N Driver │  │ Web Dashboard│  │ Buzzer (GPIO 17)  │   │
│  │ Servo Motors │  │ UV-C Sanitize│  │ LED (GPIO 27)     │   │
│  │ Wheels       │  │ Auto Delivery│  │ SMS Notification   │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Power: Rechargeable Battery + Charging Dock              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### AI Fall Detection Pipeline (Detail)

```
Camera Input ──► YOLOv8-Pose Ensemble (n+s+m) ──► TPC Tracker (W=30)
                     │                                  │
                     ▼                                  ▼
              Keypoints Output              Hybrid Rule-ML Fall Logic
              (17 COCO points)             (torso angle + ground proximity
                                            + velocity)
                                                    │
                                                    ▼
                                            FALL ALERT
                                      (Buzzer + LED + SMS)

Medicine Image ──► Multi-Engine OCR Fusion ──► Prescription Parser
                   (Tesseract + EasyOCR           (Drug name +
                    + PaddleOCR)                   Dosage + Frequency)
```

---

## Hardware Components

### Control Unit

| Component | Role |
|-----------|------|
| **Raspberry Pi 5** | Main controller and gateway — runs AI inference (fall detection + OCR), web dashboard, and coordinates all subsystems |
| **Arduino Nano** | Handles real-time sensor readings (ECG, SPO2, temperature) and motor control |
| **ESP32** | Functional hub — manages WiFi connectivity and cloud communication |

### Sensors

| Sensor | Function |
|--------|----------|
| **AD8232** | ECG (electrocardiogram) monitoring — real-time heart activity tracking |
| **MAX30102** | SPO2 and heart rate monitoring — blood oxygen saturation and pulse |
| **MPU6050** | Gyroscope and accelerometer — IMU-based motion and orientation sensing |
| **Ultrasonic Sensor** | Distance measurement — obstacle detection for autonomous navigation |
| **Temperature Sensor** | Continuous body temperature monitoring |
| **IR Line Sensors** | Line following — path tracking for hospital corridor navigation |

### Mechanics

| Component | Function |
|-----------|----------|
| **DC Motors + Wheels** | Locomotion — drives the robot along hospital corridors |
| **L298N Motor Driver** | Motor control — regulates speed and direction of DC motors |
| **Servo Motors** | Precise positioning — medicine tray dispensing and camera angle control |

### Power

| Component | Function |
|-----------|----------|
| **Rechargeable Battery** | Portable power supply for mobile operation |
| **Charging Dock** | Automatic self-charging station — robot returns when battery is low |

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
├── models/                            # Trained model weights (Git LFS)
│   ├── best_nano.pt                   # YOLOv8n-Pose weights (~6.5 MB)
│   ├── best_small.pt                  # YOLOv8s-Pose weights (~22 MB)
│   ├── best_medium.pt                 # YOLOv8m-Pose weights (~51 MB)
│   ├── export_ncnn.py                 # NCNN export script for Pi 5
│   ├── README.md                      # Model download & placement instructions
│   └── best_nano_ncnn/               # NCNN export for Pi 5 (~13 MB)
│       ├── model.ncnn.param           # NCNN network definition
│       └── model.ncnn.bin             # NCNN weights
│
├── notebooks/                         # Kaggle notebooks (run in order)
│   ├── 01_YOLOv8n_Pose_Fall_Detection_Training.ipynb
│   ├── 02_YOLOv8s_Pose_Fall_Detection_Training.ipynb
│   ├── 03_YOLOv8m_Pose_Fall_Detection_Training.ipynb
│   ├── 04_OCR_MultiEngine_Prescription_Analysis.ipynb
│   ├── 05_Hyperparameter_Sweep_Ablation.ipynb
│   └── 06_Ensemble_Final_Paper_Results.ipynb
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
    ├── README.md                      # Download instructions for results
    ├── tables/                        # LaTeX + CSV tables
    └── figures/                       # Publication-quality figures
```

---

## AI System Setup (Fall Detection + OCR)

This section covers setting up the AI components (fall detection and prescription OCR) on Raspberry Pi 5 or your computer.

### Prerequisites

- **Git LFS** installed (`git lfs install`)
- **Python 3.10+**
- For training: **Kaggle account** with GPU T4x2 access
- For Pi 5: **Raspberry Pi 5 (8GB)** + camera + active cooler

### Step 1: Clone the Repository

```bash
git lfs install
git clone https://github.com/smalakar-is-here/Ardumedics.git
cd Ardumedics
git lfs pull    # Downloads .pt and .ncnn.bin model files via LFS
```

### Step 2: Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# or: venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### Step 3: Run Fall Detection

#### On your computer (PyTorch — uses .pt model):

```bash
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano.pt \
    --camera 0 \
    --pytorch \
    --tpc-window 30 \
    --confidence 0.25 \
    --display
```

#### On Raspberry Pi 5 (NCNN — uses NCNN model, faster on ARM):

```bash
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --tpc-window 30 \
    --confidence 0.25 \
    --display
```

#### Available Command-Line Arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | required | Path to model directory (NCNN) or .pt file (PyTorch) |
| `--camera` | 0 | Camera device index |
| `--tpc-window` | 30 | TPC sliding window size |
| `--tpc-confirm` | 5 | Consecutive fall frames needed for TPC confirmation |
| `--confidence` | 0.25 | Detection confidence threshold |
| `--imgsz` | 640 | Input image size (use 320 for faster Pi5 inference) |
| `--skip-frames` | 1 | Process every Nth frame (use 2-3 for speed) |
| `--display` | off | Show annotated video window |
| `--headless` | off | Run without display (for background service) |
| `--alert-gpio` | None | GPIO pin for fall alert buzzer/LED |
| `--log-file` | None | Log file path for fall events |
| `--pytorch` | off | Force PyTorch format instead of NCNN |

### Step 4: Export NCNN Model (for Pi 5)

The NCNN model is already included in this repo (`models/best_nano_ncnn/`). If you need to re-export:

```bash
cd models/
pip install ultralytics
python3 export_ncnn.py
# Creates: best_nano_ncnn/model.ncnn.param + model.ncnn.bin
cd ..
```

### Step 5: Run OCR Prescription Analysis

```bash
# Install OCR engines first
pip install easyocr paddleocr paddlepaddle pytesseract
sudo apt install tesseract-ocr    # Linux
# or: Download from https://github.com/UB-Mannheim/tesseract/wiki  # Windows

python3 edge-deployment/raspberry-pi5/scripts/run_ocr.py \
    --image /path/to/prescription.jpg \
    --config edge-deployment/raspberry-pi5/configs/ocr.yaml
```

---

## Raspberry Pi 5 Deployment

See the full guide: [edge-deployment/raspberry-pi5/SETUP_GUIDE.md](edge-deployment/raspberry-pi5/SETUP_GUIDE.md)

### Hardware Setup

| Component | Specification | Purpose |
|-----------|--------------|---------|
| Raspberry Pi 5 | 8GB RAM model | Main controller — runs AI + web dashboard |
| Arduino Nano | ATmega328P | Real-time sensor reading + motor control |
| ESP32 | WiFi + BLE | Cloud connectivity and communication |
| Camera | Pi Camera Module v3 or USB webcam | Fall detection input |
| Active Cooler | Fan + heatsink | **Mandatory** for AI inference (55-70°C) |
| Buzzer | 3.3V active buzzer on GPIO 17 | Fall alert sound |
| LED | Red LED + 220Ω resistor on GPIO 27 | Visual fall indicator |
| Power | 5V/5A USB-C PD + Rechargeable Battery | Stationary + mobile power |

### Quick Setup on Pi 5

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install -y git-lfs python3-pip python3-venv \
    libopencv-dev python3-opencv libncnn-dev ncnn v4l-utils libcamera-apps

# 2. Clone repo with LFS
git lfs install
git clone https://github.com/smalakar-is-here/Ardumedics.git
cd Ardumedics
git lfs pull

# 3. One-click install
chmod +x edge-deployment/raspberry-pi5/install.sh
./edge-deployment/raspberry-pi5/install.sh

# 4. Run fall detection (NCNN — recommended for Pi 5)
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --tpc-window 30

# 5. Headless mode (background service with GPIO alert)
python3 edge-deployment/raspberry-pi5/scripts/run_fall_detection.py \
    --model models/best_nano_ncnn/ \
    --camera 0 \
    --tpc-window 30 \
    --headless \
    --alert-gpio 17 \
    --log-file /tmp/fall_detection.log
```

### Pi 5 AI Performance

| Metric | Nano (NCNN) | Small (NCNN) | Medium (NCNN) |
|--------|------------|-------------|---------------|
| Inference FPS | ~3.1 | ~3.1 | ~1.9 |
| Inference latency | ~320 ms | ~320 ms | ~530 ms |
| RAM usage | ~400-600 MB | ~800 MB | ~1.5 GB |
| Temperature (with cooler) | 55-70°C | 55-70°C | 65-75°C |

---

## Kaggle Training (Reproducing Results)

See [docs/KAGGLE_SETUP.md](docs/KAGGLE_SETUP.md) for detailed Kaggle setup instructions.

### Notebook Execution Order

| # | Notebook | Purpose | GPU | Time | Output |
|---|----------|---------|-----|------|--------|
| 01 | YOLOv8n-Pose Training | Train Nano model (3.3M params) | T4x2 | ~15 min | `best_nano.pt`, training metrics |
| 02 | YOLOv8s-Pose Training | Train Small model (11.6M params) | T4x2 | ~20 min | `best_small.pt`, training metrics |
| 03 | YOLOv8m-Pose Training | Train Medium model (26.5M params) | T4x2 | ~25 min | `best_medium.pt`, training metrics |
| 04 | OCR Multi-Engine | Prescription OCR with 3-engine fusion | T4 | ~6-8 hr | OCR comparison results, figures |
| 05 | Hyperparameter Sweep | LR/optimizer/augmentation/TPC/OCR ablation | T4x2 | ~30 min | Ablation JSON results, figures |
| 06 | Ensemble & Paper Results | Final ensemble + all paper outputs | T4x2 | ~15 min | 13 tables, 11 figures, statistical tests |

**Important:** Each notebook saves its output as a Kaggle Dataset that the next notebook depends on. Run in sequential order (01 → 06). Use **GPU T4 x2** accelerator for all notebooks.

### Model Download from Kaggle Outputs

After training, download the trained models from Kaggle notebook outputs:

| File | From Notebook | Location in Output |
|------|--------------|-------------------|
| `best_nano.pt` | NB01 | `nb01_outputs/best_nano.pt` |
| `best_small.pt` | NB02 | `nb02_outputs/best_small.pt` |
| `best_medium.pt` | NB03 | `nb03_outputs/best_medium.pt` |

See [models/README.md](models/README.md) for detailed download and placement instructions.

---

## Model Zoo

| Model | Params | Box mAP@50 | Box mAP@50-95 | Pose mAP@50 | Detection Rate | GPU FPS | Pi5 FPS (NCNN) |
|-------|--------|-----------|--------------|------------|---------------|---------|----------------|
| **YOLOv8n-Pose** | 3.3M | 0.667 | 0.428 | 0.218 | 92.9% | 88.9 | ~3.1 |
| YOLOv8s-Pose | 11.6M | 0.716 | 0.465 | 0.263 | 70.1% | 88.1 | ~3.1 |
| YOLOv8m-Pose | 26.5M | 0.749 | 0.464 | 0.259 | 91.3% | 54.5 | ~1.9 |
| **Ensemble (n+s+m)** | 41.4M | 0.754 | 0.454 | 0.248 | **97.6%** | 26.4 | N/A (server) |

> **Note:** Model weights are tracked via **Git LFS**. Run `git lfs pull` after cloning to download the actual files. See [models/README.md](models/README.md) for details.

### Key Observations

- **Detection Rate** is the most critical metric for safety-critical fall detection. The ensemble's 97.6% detection rate means it misses only 2.4% of persons, compared to Nano's 7.1% and Small's 29.9%.
- **Small model** has surprisingly low detection rate (70.1%) — it fails to detect persons in many challenging frames (occlusion, motion blur, low light), despite having higher mAP than Nano when it does detect.
- **Pi 5 deployment** uses Nano exclusively due to its smallest memory footprint and fastest ARM inference.

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

### Ablation Study Results (from NB05)

| Component | Ablation | Impact |
|-----------|----------|--------|
| TPC Window (W) | W=10 vs W=30 vs W=50 | W=30 optimal: balances responsiveness and false positive suppression |
| TPC Confirm Frames | 3 vs 5 vs 8 | 5 frames optimal: fewer = more false positives, more = delayed detection |
| OCR Engine | Single vs Fusion | Fusion reduces CER by ~40% vs best single engine |
| Learning Rate | 0.001 vs 0.01 vs 0.1 | 0.01 with cosine scheduler optimal |
| Optimizer | SGD vs Adam vs AdamW | AdamW with weight decay 0.05 gives best convergence |

---

## Comparison with State-of-the-Art

| Method | Accuracy (%) | F1 (%) | Real-time? | Edge Deployable? |
|--------|-------------|--------|-----------|-----------------|
| OpenPose + SVM (2019) | 85.2 | 83.1 | No | No |
| YOLOv4 + LSTM (2021) | 89.7 | 88.3 | Partial | No |
| YOLOv5-Pose (2023) | 91.5 | 90.2 | Yes | Limited |
| ViT-Pose + Transformer (2024) | 93.2 | 92.1 | No | No |
| **ArduMedics - YOLOv8n (Ours)** | **93.5** | **92.8** | **Yes** | **Yes (Pi 5, 3.1 FPS)** |
| **ArduMedics - Ensemble (Ours)** | **95.1** | **94.3** | Partial (26.4 FPS GPU) | Server only |

### Why ArduMedics Outperforms

- **vs OpenPose + SVM:** YOLOv8-Pose is end-to-end trainable and significantly faster. OpenPose requires separate detection + keypoint estimation stages, making it too slow for edge deployment.

- **vs YOLOv4 + LSTM:** Our TPC module serves a similar temporal consistency role as LSTM but is far more lightweight and interpretable. TPC uses simple velocity tracking + geometric rules instead of learned hidden states, making it suitable for Pi 5 deployment.

- **vs YOLOv5-Pose:** YOLOv8-Pose has a more efficient architecture with decoupled detection heads and anchor-free design. Our ensemble approach (combining 3 model sizes) provides higher detection reliability than any single YOLOv5-Pose model.

- **vs ViT-Pose + Transformer:** While ViT-Pose achieves competitive accuracy, it requires massive compute (GPU only, no edge deployment). Our system is specifically designed for edge deployment with NCNN optimization on Pi 5.

---

## Statistical Validation

### Wilcoxon Signed-Rank Test (OKS Scores)

The Wilcoxon signed-rank test on Object Keypoint Similarity (OKS) scores provides continuous statistical validation of ensemble superiority.

| Comparison | W Statistic | p-value | Significant? |
|-----------|------------|---------|-------------|
| Ensemble vs Nano | W=22809 | p=0.557 | No — comparable keypoint quality |
| Ensemble vs Small | W=39044 | p<0.001 | *** — Ensemble significantly better |
| Ensemble vs Medium | W=42678 | p<0.001 | *** — Ensemble significantly better |

### McNemar's Test (Fall Classification)

| Comparison | chi² | p-value | Significant? |
|-----------|------|---------|-------------|
| Small vs Ensemble | 5.33 | p=0.021 | * — Ensemble detects significantly more falls |

### Key Insight: Detection Rate vs Keypoint Quality

The Wilcoxon test shows Ensemble ≈ Nano (p=0.557) for keypoint quality, meaning the ensemble does not produce significantly more accurate keypoints than Nano alone. However, the **detection rate** tells a different story: 97.6% (Ensemble) vs 92.9% (Nano). In safety-critical fall detection, a missed detection (no person detected at all) is far more dangerous than slightly less precise keypoints. The ensemble's value is in **detection reliability**, not keypoint precision.

---

## Future Improvements

| # | Improvement | Description |
|---|------------|-------------|
| 01 | **Multi-patient support** | Manage multiple patient profiles efficiently from a single platform |
| 02 | **Mobile app integration** | On-the-go access for patients and doctors via smartphones |
| 03 | **Cloud-based data storage** | Secure and instant access to medical records from anywhere |
| 04 | **Advanced AI diagnostics** | AI-powered tools for faster and highly accurate diagnoses |
| 05 | **Integration with hospital systems** | Unified workflow through direct connection to hospital systems |

---

## Team

| Name | ID |
|------|-----|
| Swagotam Malakar | 0112231068 |
| Nishat Rasul | 0112230981 |
| Mahmudul Hasan | 011222023 |
| Sharif Ahmed | 011222307 |


---

## Datasets

### Pose Datasets (Roboflow — keypoint-annotated in YOLOv8-Pose format)

| # | Name | Images | Labels | Source |
|---|------|--------|--------|--------|
| 1 | Falling Pose Estimation | 635 | Keypoints | [Roboflow Universe](https://universe.roboflow.com/humna-pose-data/falling-pose-estimation) |
| 2 | YOLOv8-Pose Fall Detection | 474 | Keypoints | [Roboflow Universe](https://universe.roboflow.com/yolo-xvnzo/yolov8-pose-utovc) |

### Fall Detection Datasets (Kaggle — videos/images)

| # | Name | Type | Source |
|---|------|------|--------|
| 3 | UR Fall Detection | Videos | [Kaggle](https://www.kaggle.com/datasets/shahliza27/ur-fall-detection-dataset) |
| 4 | Fall Detection Images | Images | [Kaggle](https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset) |
| 5 | Le2i Fall Dataset | Videos | [Kaggle](https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia) |
| 6 | Multiple Cameras Fall | Videos | [Kaggle](https://www.kaggle.com/datasets/soumicksarker/multiple-cameras-fall-dataset) |
| 7 | Fall Video Dataset | Videos | [Kaggle](https://www.kaggle.com/datasets/payutch/fall-video-dataset) |

### OCR Datasets (Kaggle — prescription images)

| # | Name | Type | Source |
|---|------|------|--------|
| 8 | Doctor's Handwritten Prescription BD | ~800 images | [Kaggle](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset) |
| 9 | Handwritten Medical Prescriptions | ~200 images | [Kaggle](https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset) |
| 10 | Synthetic Medical Prescription OCR | 2002 images | [Kaggle](https://www.kaggle.com/datasets/priyanshuyav13/synthetic-medical-prescription-ocr-dataset) |
| 11 | OCR-Processed Prescriptions | ~800 images | [Kaggle](https://www.kaggle.com/datasets/nadaarfaoui/ocr-processed-handwritten-prescriptions) |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
