# fasterrcnn_inference.py
# Requires: pip install torch torchvision opencv-python
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import time

# Load a pre-trained Faster R-CNN with ResNet-50 backbone.
# COCO pre-trained weights are used by default. For fall detection,
# you would need to fine-tune on the fall dataset with 2 classes (fall, not_fall).
# Here we show inference with COCO weights for structural comparison.
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # set to evaluation mode (disables dropout, batchnorm uses running stats)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def detect_fasterrcnn(frame):
    """
    Run Faster R-CNN inference on a single BGR frame.
    Returns: annotated frame, list of detections, inference time in ms.
    """
    # Convert BGR (OpenCV) to RGB (PyTorch) and to tensor
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(rgb).unsqueeze(0).to(device)  # add batch dimension

    start = time.time()
    with torch.no_grad():  # disable gradient computation for inference
        outputs = model(tensor)

    inference_ms = (time.time() - start) * 1000

    # outputs[0] contains 'boxes', 'labels', 'scores' for the single image
    boxes = outputs[0]["boxes"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()

    # Filter by confidence threshold (e.g., 0.5)
    threshold = 0.5
    detections = []
    annotated = frame.copy()

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label}: {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        detections.append({
            "label": int(label),
            "confidence": float(score),
            "bbox": [x1, y1, x2, y2]
        })

    return annotated, detections, inference_ms