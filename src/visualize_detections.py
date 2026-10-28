import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tqdm import tqdm

# ================= PATH SETUP =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "KITTI_Selected")
IMG_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
CALIB_DIR = os.path.join(BASE_DIR, "calib")
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CAMERA_HEIGHT = 1.65
IOU_THRESHOLD = 0.5

# ================= HELPERS =================
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
        raise ValueError(f"Not enough calibration values in {calib_path}")
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
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
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

# ================= MAIN =================
model = YOLO("yolov8n.pt")

all_gt_distances = []
all_pred_distances = []
precision_list = []
recall_list = []

for img_name in tqdm(sorted(os.listdir(IMG_DIR))):
    base = os.path.splitext(img_name)[0]
    img_path = os.path.join(IMG_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, base + ".txt")
    calib_path = os.path.join(CALIB_DIR, base + ".txt")

    img = cv2.imread(img_path)
    gt_labels = load_gt_labels(label_path)
    K = load_calibration(calib_path)

    # YOLO detection
    results = model(img, verbose=False)[0]
    detections = []
    for box, cls in zip(results.boxes.xyxy.cpu().numpy(),
                        results.boxes.cls.cpu().numpy()):
        if model.names[int(cls)].lower() == "car":
            detections.append(box.tolist())

    # Matching & IoU calculation
    TP = 0
    matched_gt = set()
    img_vis = img.copy()

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
            est_d = estimate_distance(det, K)
            gt_d = gt_labels[best_gt]["distance"]
            all_pred_distances.append(est_d)
            all_gt_distances.append(gt_d)
            # Draw YOLO red, GT green, add IoU text
            x1, y1, x2, y2 = map(int, det)
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red
            gx1, gy1, gx2, gy2 = map(int, gt_labels[best_gt]["bbox"])
            cv2.rectangle(img_vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)  # green
            cv2.putText(img_vis, f"{best_iou:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    FP = len(detections) - TP
    FN = len([g for g in gt_labels if g["class"].lower()=="car"]) - TP
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    precision_list.append(precision)
    recall_list.append(recall)

    print(f"{img_name}: TP={TP}, FP={FP}, FN={FN}")

    # Save visualization
    cv2.imwrite(os.path.join(RESULT_DIR, img_name), img_vis)

# ================= RESULTS =================
print("\n===== FINAL RESULTS =====")
print(f"Mean Precision: {np.mean(precision_list):.3f}")
print(f"Mean Recall:    {np.mean(recall_list):.3f}")

if len(all_gt_distances) > 0:
    plt.figure(figsize=(8, 6))
    plt.scatter(all_gt_distances, all_pred_distances, alpha=0.7)
    max_d = max(all_gt_distances)
    plt.plot([0, max_d], [0, max_d], "r--")
    plt.xlabel("Ground Truth Distance (m)")
    plt.ylabel("Estimated Distance (m)")
    plt.title("Distance Estimation vs Ground Truth (All Cars)")
    plt.grid()
    plt.show()
else:
    print("⚠️ No matched cars found. Distance plot skipped.")
