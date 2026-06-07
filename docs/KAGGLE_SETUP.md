# Kaggle Setup Guide — ArduMedics

This guide explains how to run all 6 ArduMedics notebooks on Kaggle with GPU.

---

## Prerequisites

1. A [Kaggle](https://kaggle.com) account
2. Phone verification enabled (required for GPU access)
3. Internet enabled in notebook settings

---

## Step 1: Create the Pose Dataset (One-Time)

The two Roboflow pose datasets must be uploaded as a **single Kaggle Dataset**:

1. Download from Roboflow Universe:
   - [Falling Pose Estimation](https://universe.roboflow.com/humna-pose-data/falling-pose-estimation) (635 images)
   - [YOLOv8-Pose Fall Detection](https://universe.roboflow.com/yolo-xvnzo/yolov8-pose-utovc) (474 images)

2. Create a ZIP file containing both subfolders:
   ```
   ardumedics-roboflow-pose-datasets.zip
   ├── falling_pose_estimation/
   │   ├── data.yaml
   │   ├── images/
   │   └── labels/
   └── yolov8_pose_fall/
       ├── data.yaml
       ├── images/
       └── labels/
   ```

3. Upload to Kaggle: **New Dataset** → Upload the ZIP → Name: `ardumedics-roboflow-pose-datasets`

> **Important:** Upload as a SINGLE ZIP. Do NOT upload extracted folders (exceeds 1000-file limit).

---

## Step 2: Run Notebooks in Order

Each notebook saves outputs as a Kaggle Dataset for the next notebook.

### Notebook 01: YOLOv8n-Pose Training

1. Create new Kaggle Notebook
2. Upload `01_YOLOv8n_Pose_Fall_Detection_Training.ipynb`
3. Settings → Accelerator: **GPU T4 x2**
4. Settings → Internet: **On**
5. Add datasets:
   - `ardumedics-roboflow-pose-datasets` (from Step 1)
   - `shahliza27/ur-fall-detection-dataset`
   - `uttejkumarkandagatla/fall-detection-dataset`
   - `tuyenldvn/falldataset-imvia`
   - `soumicksarker/multiple-cameras-fall-dataset`
   - `payutch/fall-video-dataset`
6. Run all cells
7. Save output as Kaggle Dataset: `ardumedics-nb01-nano-outputs`

### Notebook 02: YOLOv8s-Pose Training

Same process, but also add `ardumedics-nb01-nano-outputs` as input.
Save output as: `ardumedics-nb02-small-outputs`

### Notebook 03: YOLOv8m-Pose Training

Add previous outputs as inputs.
Save output as: `ardumedics-nb03-medium-outputs`

### Notebook 04: OCR Multi-Engine

Add OCR datasets:
- `mamun1113/doctors-handwritten-prescription-bd-dataset`
- `mehaksingal/illegible-medical-prescription-images-dataset`
- `priyanshuyav13/synthetic-medical-prescription-ocr-dataset`
- `nadaarfaoui/ocr-processed-handwritten-prescriptions`

Save output as: `ardumedics-nb04-ocr-outputs`

### Notebook 05: Hyperparameter Sweep

Add all previous outputs (NB01-NB04).
Save output as: `ardumedics-nb05-ablation-outputs`

### Notebook 06: Ensemble & Paper Results

Add ALL previous outputs + ALL datasets.
Download `/kaggle/working/final_results/` folder — this contains all paper tables and figures.

---

## Step 3: Save Outputs as Kaggle Datasets

After each notebook finishes:

1. Go to **Output** tab
2. Click **Save & Run All (Commit)**
3. After commit, click **Save as Dataset**
4. Name appropriately (e.g., `ardumedics-nb01-nano-outputs`)

This makes the outputs available as input for the next notebook.

---

## Runtime Estimates

| Notebook | GPU | Estimated Time |
|----------|-----|---------------|
| NB01 | T4x2 | ~15 minutes |
| NB02 | T4x2 | ~20 minutes |
| NB03 | T4x2 | ~25 minutes |
| NB04 | T4 | ~6-8 hours |
| NB05 | T4x2 | ~30 minutes |
| NB06 | T4x2 | ~15 minutes |

**Total: ~8-9 hours** (NB04 is the longest due to OCR processing)

---

## Troubleshooting

### "Disk quota exceeded"
Kaggle has a 20GB disk limit on `/kaggle/working/`. If you hit this:
- Clear previous runs: `rm -rf /kaggle/working/runs/`
- Some notebooks skip large video datasets when disk is low

### "GPU not available"
- Enable phone verification on Kaggle
- Select T4 x2 in notebook settings

### Model not found in next notebook
- Ensure you saved the output as a Kaggle Dataset
- Ensure the dataset is added as input to the next notebook
