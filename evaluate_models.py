import cv2
import pandas as pd
import os
from yolo_inference import detect_yolo
from fasterrcnn_inference import detect_fasterrcnn

VIDEO_DIR = "./videos"
LABELS_FILE = "labels.csv"
df_labels = pd.read_csv(LABELS_FILE)

# 'yolo' or 'fasterrcnn'
MODEL_TO_TEST = 'fasterrcnn'

metrics = {
    "total_inference_time_ms": 0,
    "total_frames": 0,
    "false_positives": 0,
    "missed_detections": 0
}

for index, row in df_labels.iterrows():
    video_path = os.path.join(VIDEO_DIR, row['filename'])
    true_label = row['label']

    cap = cv2.VideoCapture(video_path)
    fall_detected_in_video = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if MODEL_TO_TEST == 'yolo':
            annotated, detections, inference_ms = detect_yolo(frame)
        else:
            annotated, detections, inference_ms = detect_fasterrcnn(frame)

        metrics["total_inference_time_ms"] += inference_ms
        metrics["total_frames"] += 1

        if len(detections) > 0:
            fall_detected_in_video = True

    cap.release()

    if true_label == "not_fall" and fall_detected_in_video:
        metrics["false_positives"] += 1
    elif true_label == "fall" and not fall_detected_in_video:
        metrics["missed_detections"] += 1

avg_inference_ms = metrics["total_inference_time_ms"] / metrics["total_frames"] if metrics["total_frames"] > 0 else 0
avg_fps = 1000 / avg_inference_ms if avg_inference_ms > 0 else 0

print(f"=== Evaluation Results ({MODEL_TO_TEST.upper()}) ===")
print(f"Average Inference Time: {avg_inference_ms:.2f} ms")
print(f"Average FPS: {avg_fps:.2f}")
print(f"False Positives: {metrics['false_positives']}")
print(f"Missed Detections: {metrics['missed_detections']}")