# ============================================================
# IMPORTS
# ============================================================

import os                                  # Provides functions for file and directory handling
import cv2                                 # OpenCV library for image loading and processing
import numpy as np                          # Numerical operations (arrays, matrices, linear algebra)
import matplotlib.pyplot as plt             # Plotting library for graphs and visualizations
from ultralytics import YOLO                # YOLOv8 object detection model
from tqdm import tqdm                       # Progress bar for loops

# ============================================================
# PATHS & CONSTANTS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  
# Absolute path of the directory containing this script

BASE_DIR  = os.path.join(SCRIPT_DIR, "KITTI_Selected")   
# Root directory of the selected KITTI dataset

IMG_DIR   = os.path.join(BASE_DIR, "images")             
# Directory containing KITTI images

LABEL_DIR = os.path.join(BASE_DIR, "labels")             
# Directory containing ground-truth label files

CALIB_DIR = os.path.join(BASE_DIR, "calib")              
# Directory containing camera calibration files

CAMERA_HEIGHT = 1.65      # Height of the camera above ground in meters (KITTI standard)
IOU_THRESHOLD = 0.5       # Minimum IoU required to count a detection as a true positive

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_calibration(calib_path):
    """
    Load KITTI calibration file and extract camera intrinsic matrix K (3x3)
    """
    with open(calib_path, "r") as f:        # Open calibration file
        data = f.read().replace(":", " ")   # Replace ':' to simplify parsing

    values = [float(v) for v in data.split() if _is_float(v)]  
    # Extract all numeric values from the file

    if len(values) < 9:                     # Ensure enough values for a 3x3 matrix
        raise ValueError("Invalid calibration file")

    K = np.array(values[:9]).reshape(3, 3)  # Construct the intrinsic matrix K
    return K                                # Return camera intrinsic matrix


def _is_float(x):
    try:
        float(x)                            # Attempt to convert string to float
        return True                         # Conversion successful
    except ValueError:
        return False                        # Conversion failed


def load_gt_labels(label_path):
    """
    Load ground-truth labels
    Expected format per line:
    class xmin ymin xmax ymax gt_distance
    """
    gts = []                                # List to store ground-truth objects
    with open(label_path, "r") as f:        # Open label file
        for line in f:                      # Read each object annotation
            parts = line.strip().split()    # Split line into elements
            gts.append({
                "class": parts[0],          # Object class (e.g., car)
                "bbox": list(map(float, parts[1:5])),  # Bounding box coordinates
                "distance": float(parts[5]) # Ground-truth distance (meters)
            })
    return gts                              # Return list of ground-truth objects


def compute_iou(boxA, boxB):
    """
    Compute Intersection-over-Union (IoU) between two bounding boxes
    """
    xA = max(boxA[0], boxB[0])              # Left boundary of intersection
    yA = max(boxA[1], boxB[1])              # Top boundary of intersection
    xB = min(boxA[2], boxB[2])              # Right boundary of intersection
    yB = min(boxA[3], boxB[3])              # Bottom boundary of intersection

    inter = max(0, xB - xA) * max(0, yB - yA)  
    # Area of intersection region

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])  
    # Area of first bounding box

    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])  
    # Area of second bounding box

    return inter / (areaA + areaB - inter + 1e-6)  
    # IoU formula with small epsilon to avoid division by zero


def estimate_distance_midpoint_lower_edge(bbox, K):
    """
    Estimate object distance by projecting the midpoint of the
    lower bounding-box edge onto the ground plane (KITTI-consistent)
    """
    xmin, ymin, xmax, ymax = bbox           # Unpack bounding box coordinates

    u = (xmin + xmax) / 2.0                 # Horizontal midpoint of lower edge
    v = ymax                                # Vertical coordinate of lower edge

    pixel = np.array([u, v, 1.0])           # Convert pixel to homogeneous coordinates
    ray = np.linalg.inv(K) @ pixel          # Back-project pixel to 3D camera ray

    scale = -CAMERA_HEIGHT / ray[1]         # Scale ray to intersect ground plane
    point_3d = ray * scale                  # 3D ground intersection point

    return abs(point_3d[2])                 # Forward distance along Z-axis


def draw_projection_point(img, bbox, color):
    """
    Draw the projection point (midpoint of lower edge) on the image
    """
    xmin, ymin, xmax, ymax = map(int, bbox) # Convert bounding box to integers
    u = int((xmin + xmax) / 2)              # X-coordinate of projection point
    v = int(ymax)                           # Y-coordinate of projection point
    cv2.circle(img, (u, v), 5, color, -1)   # Draw filled circle on image

# ============================================================
# MAIN
# ============================================================

model = YOLO("yolov8n.pt")                  # Load pretrained YOLOv8 nano model

all_gt_distances = []                       # Store all ground-truth distances
all_pred_distances = []                     # Store all estimated distances
precision_list = []                         # Store per-image precision
recall_list = []                            # Store per-image recall

image_files = sorted(os.listdir(IMG_DIR))   # Get sorted list of image filenames

for img_name in tqdm(image_files):           # Loop over images with progress bar
    base = os.path.splitext(img_name)[0]     # Filename without extension

    img_path   = os.path.join(IMG_DIR, img_name)  
    label_path = os.path.join(LABEL_DIR, base + ".txt")  
    calib_path = os.path.join(CALIB_DIR, base + ".txt")  

    img = cv2.imread(img_path)               # Load image
    gt_labels = load_gt_labels(label_path)   # Load ground-truth annotations
    K = load_calibration(calib_path)         # Load camera intrinsic matrix

    # ---------------------------
    # YOLO Detection
    # ---------------------------
    results = model(img, verbose=False)[0]   # Run YOLO inference on image

    detections = []                          # Store detected car bounding boxes
    for box, cls in zip(results.boxes.xyxy.cpu().numpy(),
                        results.boxes.cls.cpu().numpy()):
        if model.names[int(cls)].lower() == "car":  
            detections.append(box.tolist())  # Keep only car detections

    # ---------------------------
    # Matching
    # ---------------------------
    TP = 0                                   # True Positives counter
    matched_gt = set()                       # Track matched GT indices

    for det in detections:                   # Loop over detections
        best_iou = 0
        best_gt_idx = None

        for i, gt in enumerate(gt_labels):   # Loop over ground-truth objects
            if gt["class"].lower() != "car" or i in matched_gt:
                continue

            iou = compute_iou(det, gt["bbox"])  
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= IOU_THRESHOLD and best_gt_idx is not None:
            TP += 1                          # Count true positive
            matched_gt.add(best_gt_idx)      # Mark GT as matched

            est_d = estimate_distance_midpoint_lower_edge(det, K)  
            gt_d = gt_labels[best_gt_idx]["distance"]

            all_pred_distances.append(est_d) # Store estimated distance
            all_gt_distances.append(gt_d)    # Store GT distance

    FP = len(detections) - TP                # False Positives
    FN = len([g for g in gt_labels if g["class"].lower() == "car"]) - TP  
    # False Negatives

    precision = TP / (TP + FP + 1e-6)        # Precision calculation
    recall = TP / (TP + FN + 1e-6)           # Recall calculation

    precision_list.append(precision)         # Save precision
    recall_list.append(recall)               # Save recall

    print(f"{img_name}: TP={TP}, FP={FP}, FN={FN}")  
    # Print per-image detection statistics

# ============================================================
# RESULTS
# ============================================================

print("\n================ FINAL RESULTS ================")
print(f"Mean Precision: {np.mean(precision_list):.3f}")  
# Average precision over all images
print(f"Mean Recall:    {np.mean(recall_list):.3f}")     
# Average recall over all images

# ============================================================
# DISTANCE COMPARISON PLOT
# ============================================================

if len(all_gt_distances) > 0:                # Check if valid matches exist
    plt.figure(figsize=(8, 6))               # Create figure
    plt.scatter(all_gt_distances, all_pred_distances, alpha=0.7)  
    # Scatter plot of GT vs estimated distances
    max_d = max(all_gt_distances)
    plt.plot([0, max_d], [0, max_d], "r--")  # Ideal y=x reference line
    plt.xlabel("Ground Truth Distance (m)")  # X-axis label
    plt.ylabel("Estimated Distance (m)")     # Y-axis label
    plt.title("KITTI Distance Estimation (Lower-Edge Midpoint)")  
    plt.grid()                               # Enable grid
    plt.show()                               # Display plot
else:
    print("No matched cars found — plot skipped.")  
    # Handle case with no valid detections
