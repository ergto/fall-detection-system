# yolo_inference.py
# Requires: pip install ultralytics opencv-python
from ultralytics import YOLO
import cv2
import time

# Load a pre-trained YOLO11 nano model.
# If you fine-tuned on the fall dataset, replace with your own weight path.
model = YOLO("yolo11n.pt")  # nano version: fastest, smallest


def detect_yolo(frame):
    """
    Run YOLO inference on a single BGR frame.
    Returns: annotated frame, list of detections, inference time in ms.
    """
    start = time.time()
    results = model(frame, verbose=False)  # verbose=False suppresses console spam
    inference_ms = (time.time() - start) * 1000

    # results[0].boxes contains bounding boxes, confidences, and class IDs
    annotated = results[0].plot()  # draws boxes on the frame
    detections = []

    for box in results[0].boxes:
        detections.append({
            "class_id": int(box.cls[0]),
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        })

    return annotated, detections, inference_ms

# Example usage with a video file
# cap = cv2.VideoCapture("fall_video.avi")
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret: break
#     annotated, dets, ms = detect_yolo(frame)
#     cv2.imshow("YOLO", annotated)
#     if cv2.waitKey(1) & 0xFF == ord('q'): break