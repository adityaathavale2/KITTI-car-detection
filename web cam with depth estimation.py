import cv2
import numpy as np
from ultralytics import YOLO
import time

# -----------------------------
# PARAMETERS
# -----------------------------
CAMERA_HEIGHT = 1.65  # meters
CONF_THRESHOLD = 0.3   # minimum confidence to show detection
GROUND_CLASSES = ["car", "person", "chair", "bicycle", "motorcycle"]  # objects assumed on ground

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("yolov8n.pt")  # or your custom-trained weights
class_names = model.names

# -----------------------------
# INITIALIZE WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # output video FPS

# -----------------------------
# VIDEO WRITER
# -----------------------------
out_path = "webcam_distance_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

# -----------------------------
# CAMERA INTRINSICS (approx.)
# -----------------------------
fx = 800
fy = 800
cx = frame_width / 2
cy = frame_height / 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]])

print("Press 'q' to quit the webcam feed.")

# -----------------------------
# REAL-TIME LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    results = model(frame)[0]
    inference_time = time.time() - start_time
    fps_display = 1 / inference_time

    for box, cls_id, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                 results.boxes.cls.cpu().numpy(),
                                 results.boxes.conf.cpu().numpy()):
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        label = class_names[int(cls_id)]

        # Only estimate distance for ground objects
        if label.lower() in GROUND_CLASSES:
            u = (x1 + x2) / 2
            v = y2  # bottom-center
            pixel = np.array([u, v, 1.0])
            ray = np.linalg.inv(K) @ pixel
            scale = -CAMERA_HEIGHT / ray[1]
            distance = np.linalg.norm(ray * scale)
            distance_label = f"{distance:.2f} m"
        else:
            distance_label = ""

        # Draw bounding box + labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if distance_label:
            cv2.putText(frame, f"Dist: {distance_label}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Show frame
    cv2.imshow("YOLO Webcam + Distance Demo", frame)

    # Write frame to video
    out.write(frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# RELEASE RESOURCES
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Demo video saved at: {out_path}")
