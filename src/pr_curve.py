# ==========================================================
# IMPORTS
# ==========================================================
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd

# ==========================================================
# PATH SETUP
# ==========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "KITTI_Selected")
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
CALIB_DIR = os.path.join(BASE_DIR, "calib")
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CAMERA_HEIGHT = 1.65
IOU_THRESHOLD = 0.5

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def load_calibration(calib_path):
    with open(calib_path, "r") as f:
        text = f.read().replace(":", " ")
    vals = [float(x) for x in text.split() if is_float(x)]
    if len(vals) < 9:
        raise ValueError(f"Invalid calibration file: {calib_path}")
    return np.array(vals[:9]).reshape(3, 3)

def load_gt_labels(label_path):
    gts = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            gts.append({
                "class": parts[0],
                "bbox": list(map(float, parts[1:5])),
                "distance": float(parts[5])
            })
    return gts

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def estimate_distance(bbox, K):
    xmin, ymin, xmax, ymax = bbox
    u = (xmin + xmax) / 2
    v = ymax
    pixel = np.array([u, v, 1.0])
    ray = np.linalg.inv(K) @ pixel
    scale = -CAMERA_HEIGHT / ray[1]
    point_3d = ray * scale
    return abs(point_3d[2])

# ==========================================================
# MAIN
# ==========================================================
model = YOLO("yolov8n.pt")

all_gt_distances = []
all_pred_distances = []
image_summary = []

image_files = sorted(os.listdir(IMG_DIR))

for img_name in tqdm(image_files, desc="Processing images"):
    base = os.path.splitext(img_name)[0]

    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    gt_labels = load_gt_labels(os.path.join(LABEL_DIR, base + ".txt"))
    K = load_calibration(os.path.join(CALIB_DIR, base + ".txt"))

    results = model(img, verbose=False)[0]

    detections = []
    for box, cls in zip(results.boxes.xyxy.cpu().numpy(),
                        results.boxes.cls.cpu().numpy()):
        if model.names[int(cls)].lower() == "car":
            detections.append(box.tolist())

    TP = 0
    matched_gt = set()
    pred_d = []
    gt_d = []

    for det in detections:
        best_iou = 0
        best_gt = None
        for i, gt in enumerate(gt_labels):
            if gt["class"].lower() != "car" or i in matched_gt:
                continue
            iou = compute_iou(det, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= IOU_THRESHOLD and best_gt is not None:
            TP += 1
            matched_gt.add(best_gt)
            est = estimate_distance(det, K)
            gtval = gt_labels[best_gt]["distance"]
            pred_d.append(est)
            gt_d.append(gtval)
            all_pred_distances.append(est)
            all_gt_distances.append(gtval)

    FP = len(detections) - TP
    FN = len([g for g in gt_labels if g["class"].lower() == "car"]) - TP

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)

    mean_dist_error = np.nan
    if gt_d:
        mean_dist_error = np.mean(np.abs(np.array(pred_d) - np.array(gt_d)))

    image_summary.append({
        "Image": img_name,
        "Precision": precision,
        "Recall": recall,
        "MeanDistanceError": mean_dist_error
    })

    vis = img.copy()
    for gt in gt_labels:
        if gt["class"].lower() == "car":
            x1, y1, x2, y2 = map(int, gt["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for det in detections:
        x1, y1, x2, y2 = map(int, det)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(os.path.join(RESULT_DIR, f"{base}_vis.jpg"), vis)

# ==========================================================
# SAVE CSV SUMMARY
# ==========================================================
df_summary = pd.DataFrame(image_summary)
df_summary.to_csv(os.path.join(RESULT_DIR, "summary.csv"), index=False)

# ==========================================================
# PRECISION–RECALL CURVE (PER IMAGE)
# ==========================================================
df_sorted = df_summary.sort_values("Recall")

plt.figure(figsize=(8, 6))
plt.plot(
    df_sorted["Recall"],
    df_sorted["Precision"],
    marker="o",
    linestyle="-"
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve (Per Image)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.show()

# ==========================================================
# DISTANCE SCATTER PLOT
# ==========================================================
if all_gt_distances:
    plt.figure(figsize=(8, 6))
    plt.scatter(all_gt_distances, all_pred_distances)
    max_d = max(all_gt_distances)
    plt.plot([0, max_d], [0, max_d], "r--")
    plt.xlabel("Ground Truth Distance (m)")
    plt.ylabel("Estimated Distance (m)")
    plt.title("Distance Estimation vs Ground Truth")
    plt.grid()
    plt.show()
else:
    print("⚠️ No matched cars found.")
