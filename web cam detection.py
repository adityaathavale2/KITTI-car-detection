import cv2
from ultralytics import YOLO
import time

# -----------------------------
# 1. Load YOLO model
# -----------------------------
# COCO pre-trained (80 classes)
model = YOLO("yolov8n.pt")  # or your custom-trained weights: "your_model.pt"

# -----------------------------
# 2. Initialize webcam
# -----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# -----------------------------
# 3. Prepare video writer
# -----------------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # recording FPS
out_path = "webcam_detection_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' or 'XVID'
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

print("Press 'q' to quit the webcam feed.")
class_names = model.names

# -----------------------------
# 4. Real-time detection loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    results = model(frame)[0]
    inference_time = time.time() - start_time
    fps_display = 1 / inference_time

    # Draw detections
    for box, cls_id, conf in zip(results.boxes.xyxy.cpu().numpy(),
                                results.boxes.cls.cpu().numpy(),
                                results.boxes.conf.cpu().numpy()):
        if conf < 0.3:  # filter low-confidence detections
            continue
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls_id)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show live feed
    cv2.imshow("YOLO Webcam Detection", frame)

    # Write frame to video
    out.write(frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 5. Release resources
# -----------------------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\n✅ Video saved at: {out_path}")