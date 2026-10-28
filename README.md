# KITTI Car Detection & Monocular Distance Estimation

YOLOv8-based car detection pipeline evaluated on a selected subset of the KITTI dataset, with monocular distance estimation using camera intrinsics and the ground-plane assumption.

![Detection output sample](results/detection_sample.png)

Developed as coursework for **Computer Vision**, M.Sc. Mechatronics at Ravensburg-Weingarten University of Applied Sciences (RWU).

---

## Overview

This project detects cars in KITTI road images using a pretrained YOLOv8 model, matches detections to ground-truth annotations via IoU, and estimates the distance to each detected car using a ground-plane projection method. Detection quality is reported as per-image precision and recall, and distance estimates are compared against KITTI's ground-truth 3D distances.

> **Note:** The dataset (`KITTI_Selected/`) and model weights (`yolov8n.pt`) are not included in this repository. See [Setup](#setup) below.

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
ray = K⁻¹ · [u, v, 1]ᵀ
scale = -camera_height / ray[1]
distance = |ray[2] · scale|


- **Camera height assumption:** 1.65 m (KITTI standard)
- Distance is the forward (Z-axis) depth where the ray intersects the ground plane

---

## Repository Structure
```
KITTI-car-detection/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│ ├── main.py # Entry point: detection + evaluation + distance plot
│ ├── per_image_precision_recall.py # Per-image P/R bar chart + CSV export
│ ├── distance_estimation.py # Distance estimation with projection visualization
│ ├── visualize_detections.py # Bounding box overlay visualizations
│ ├── yolo_vs_gt_comparison.py # Side-by-side YOLO vs GT box comparison
│ ├── pr_curve.py # Precision-recall curve
│ ├── webcam_detection.py # Live webcam car detection
│ └── webcam_depth_estimation.py # Live webcam detection + distance estimation
│
├── KITTI_Selected/
│ └── README.md # Dataset download instructions (not included)
│
└── results/ # Output visualizations and CSV (generated at runtime)
└── detection_sample.png
```

## Dataset

20 frames selected from the [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/).

### Label format

Each line in a label `.txt` file:


### Calibration files

Contain the 3×3 camera intrinsic matrix **K** extracted from KITTI's standard calibration format.

### License

The KITTI dataset is distributed under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)**. Please respect the original authors' licensing terms when using the data.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.8+ recommended.

### 2. Download model weights

The YOLOv8 nano weights download automatically on first run via Ultralytics. Or manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### 3. Prepare the dataset

See [`KITTI_Selected/README.md`](KITTI_Selected/README.md) for download instructions and expected folder structure.

---

## Usage

### Main pipeline — detection + evaluation + distance plot

```bash
python src/main.py
```

### Per-image precision and recall + CSV export

```bash
python src/per_image_precision_recall.py
```

Saves `results/summary.csv` with per-image TP, FP, FN, Precision, Recall, and Mean Distance Error.

### Distance estimation with projection visualization

```bash
python src/distance_estimation.py
```

### Bounding box overlay visualization

```bash
python src/visualize_detections.py
```

### YOLO vs Ground-Truth box comparison

```bash
python src/yolo_vs_gt_comparison.py
```

### Precision-recall curve

```bash
python src/pr_curve.py
```

### Live webcam detection

```bash
python src/webcam_detection.py
```

### Live webcam detection + distance estimation

```bash
python src/webcam_depth_estimation.py
```

---

## Outputs

| Output | Description |
|--------|-------------|
| Distance scatter plot | Estimated vs ground-truth distance for all matched cars |
| Precision/Recall bar chart | Per-image precision and recall |
| Precision-recall curve | Overall PR curve across all images |
| `results/summary.csv` | Tabular per-image metrics |
| Bounding box visualizations | YOLO detections overlaid on images |

---

## Limitations

- **Flat ground-plane assumption** — Distance estimation degrades on slopes or for distant objects where the ground is not level
- **Lightweight model** — YOLOv8 nano prioritizes speed over accuracy; larger variants (`yolov8s/m/l`) would improve detection recall
- **Monocular depth scale ambiguity** — Cannot recover true scale without the ground-plane assumption or stereo pairs
- **Fixed camera height** — The 1.65 m KITTI standard does not generalize to arbitrary mounting positions

---

## References

1. [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/)
2. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

## Author

**Aditya Athavale** — M.Sc. Mechatronics, Ravensburg-Weingarten University of Applied Sciences (RWU)

[GitHub](https://github.com/adityaathavale2) · [LinkedIn](https://www.linkedin.com/in/adityaathavale/)