
def evaluate_detection(detections, confidence_threshold):
    """
    Evaluate model detections and return the strongest
    qualifying fall detection.
    """

    fall_confidence = 0.0

    for detection in detections:
        class_name = detection.get("class_name", "").lower()
        confidence = float(detection.get("confidence", 0.0))

        if class_name == "fall":
            fall_confidence = max(fall_confidence, confidence)

    fall_detected = fall_confidence >= confidence_threshold

    return fall_detected, fall_confidence
