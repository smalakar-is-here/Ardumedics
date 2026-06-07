# Datasets

Due to GitHub file size limits, dataset files are not included in this repository. Download them from the original sources.

## Roboflow Pose Datasets

These are keypoint-annotated datasets in YOLOv8-Pose format. **Upload as a single ZIP** to Kaggle (auto-extracts).

| # | Name | Images | Download |
|---|------|--------|----------|
| 1 | Falling Pose Estimation | 635 | [Roboflow Universe](https://universe.roboflow.com/humna-pose-data/falling-pose-estimation) |
| 2 | YOLOv8-Pose Fall Detection | 474 | [Roboflow Universe](https://universe.roboflow.com/yolo-xvnzo/yolov8-pose-utovc) |

**Important:**
- Dataset 3 (nafzzan/falling-pose-estimation-0xme8) was **removed** — it is a pixel-identical duplicate of Dataset 1
- Upload both datasets as a **single ZIP** file named `ardumedics-roboflow-pose-datasets.zip` containing both subfolders
- Do NOT upload extracted folders directly (exceeds Kaggle's 1000-file limit)
- Do NOT upload two separate ZIPs (data.yaml conflict)

## Kaggle Datasets

Add these as Kaggle Dataset inputs when running the notebooks:

### Fall Detection Datasets
| # | Name | Kaggle Slug |
|---|------|-------------|
| 3 | UR Fall Detection | `shahliza27/ur-fall-detection-dataset` |
| 4 | Fall Detection Images | `uttejkumarkandagatla/fall-detection-dataset` |
| 5 | Le2i Fall Dataset | `tuyenldvn/falldataset-imvia` |
| 6 | Multiple Cameras Fall | `soumicksarker/multiple-cameras-fall-dataset` |
| 7 | Fall Video Dataset | `payutch/fall-video-dataset` |

### OCR Datasets
| # | Name | Kaggle Slug |
|---|------|-------------|
| 8 | Doctor's Handwritten Prescription BD | `mamun1113/doctors-handwritten-prescription-bd-dataset` |
| 9 | Handwritten Medical Prescriptions | `mehaksingal/illegible-medical-prescription-images-dataset` |
| 10 | Synthetic Medical Prescription OCR | `priyanshuyav13/synthetic-medical-prescription-ocr-dataset` |
| 11 | OCR-Processed Prescriptions | `nadaarfaoui/ocr-processed-handwritten-prescriptions` |
