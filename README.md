# KITTI Car Detection & Distance Estimation (Task 2)

YOLOv8-based car detection pipeline evaluated on a selected subset of the KITTI dataset, with monocular distance estimation using camera intrinsics.

---

## Overview

This project detects cars in KITTI road images using a pretrained YOLOv8 model, matches detections to ground-truth annotations via IoU, and estimates the distance to each detected car using a ground-plane projection method. Detection quality is reported as per-image precision and recall, and distance estimates are compared against KITTI's ground-truth 3D distances.

> **Note:** The dataset (`KITTI_Selected/`) and model weights (`yolov8n.pt`) are not included in this repository. See [Setup](#setup) below.

---

## Project Structure

```
KITTI_Selection/
├── KITTI_Selected/              # ⚠️ Not included — see Setup
│   ├── images/                  # 20 KITTI road scene images
│   ├── labels/                  # Ground-truth labels (class, bbox, gt_distance)
│   └── calib/                   # Camera calibration files (3×3 intrinsic matrix K)
├── results/                     # Output visualizations and CSV (generated at runtime)
│
├── TASK2 working.py                         # Core pipeline: detection + evaluation + distance plot
├── TASK2 precision recall each image.py     # Per-image precision/recall bar chart + CSV export
├── distance estimation.py                   # Distance estimation with projection-point visualization
├── visualizationscript working.py           # Bounding box overlay visualizations
├── YOLO and GT boxes each image.py          # Side-by-side YOLO vs GT box comparison
├── avg precision vs recall.py               # Overall average precision vs recall plot
├── pr curve.py                              # Precision-recall curve
├── web cam detection.py                     # Live webcam car detection
├── web cam with depth estimation.py         # Live webcam detection + distance estimation
└── README.md
```

> Each script also has an `explanation` variant (excluded from this repo) with detailed line-by-line comments — these are identical in logic but serve as annotated study versions.

---

## Method

### 1. Object Detection
- **Model:** YOLOv8 nano (`yolov8n.pt`) via [Ultralytics](https://github.com/ultralytics/ultralytics)
- Only `car` class detections are retained

### 2. Detection Evaluation (IoU Matching)
Each YOLO detection is matched to a ground-truth bounding box:
- **IoU threshold:** 0.5
- Greedy matching (highest IoU first, each GT matched at most once)
- Computes **TP, FP, FN** → **Precision** and **Recall** per image

### 3. Monocular Distance Estimation
Uses the bottom-center pixel of each matched bounding box and the camera intrinsic matrix **K**:

```
ray = K⁻¹ · [u, v, 1]ᵀ
scale = -camera_height / ray[1]
distance = |ray[2] · scale|
```

- **Camera height assumption:** 1.65 m (KITTI standard)
- Distance is the forward (Z-axis) depth where the ray intersects the ground plane

---

## Dataset

20 frames selected from the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/).

**Label format** (per line in `.txt`):
```
class  xmin  ymin  xmax  ymax  gt_distance
```

**Calibration files** contain the 3×3 camera intrinsic matrix K extracted from KITTI's standard calibration format.

---

## Setup

### 1. Install dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib pandas tqdm
```

Python 3.8+ recommended.

### 2. Download model weights
The YOLOv8 nano weights download automatically on first run via Ultralytics. Or manually:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Prepare the dataset
Place your `KITTI_Selected/` folder in the project root with this structure:
```
KITTI_Selected/
├── images/     # .jpg or .png files
├── labels/     # .txt files (one per image)
└── calib/      # .txt calibration files (one per image)
```

---

## Usage

### Core pipeline (detection + distance scatter plot)
```bash
python "TASK2 working.py"
```

### Per-image precision & recall bar chart + CSV
```bash
python "TASK2 precision recall each image.py"
```
Saves `results/summary.csv` with per-image TP, FP, FN, Precision, Recall, and Mean Distance Error.

### Distance estimation with projection visualization
```bash
python "distance estimation.py"
```

### Bounding box overlay visualization
```bash
python "visualizationscript working.py"
```

### YOLO vs Ground-Truth box comparison
```bash
python "YOLO and GT boxes each image.py"
```

### Live webcam detection
```bash
python "web cam detection.py"
```

### Live webcam detection + distance estimation
```bash
python "web cam with depth estimation.py"
```

---

## Outputs

| Output | Description |
|--------|-------------|
| Distance scatter plot | Estimated vs ground-truth distance for all matched cars |
| Precision/Recall bar chart | Per-image precision and recall |
| `results/summary.csv` | Tabular per-image metrics |
| Bounding box visualizations | YOLO detections overlaid on images |

---

## Limitations

- Distance estimation assumes a flat ground plane — performance degrades on slopes or distant objects
- YOLOv8 nano is a lightweight model; larger variants (`yolov8s/m/l`) would improve detection recall
- Monocular depth estimation cannot recover scale without the ground-plane assumption

---

## References

- [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
