
import streamlit as st
from detection_logic import evaluate_detection

# Page configuration
st.set_page_config(
    page_title="Fall Detection System",
    page_icon="🚨",
    layout="wide"
)

# Header
st.title("Fall Detection Alert System")
st.write("Real-time fall detection for elderly care and public safety.")

# Sidebar
st.sidebar.header("Detection Settings")

input_source = st.sidebar.radio(
    "Input Source",
    ["Upload Video", "Live Camera"]
)

selected_model = st.sidebar.selectbox(
    "Detection Model",
    ["YOLO11n", "Faster R-CNN"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Main dashboard

# Main dashboard
st.subheader("Detection Preview")

if input_source == "Upload Video":
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.success("Video uploaded successfully!")
        st.write(f"Filename: {uploaded_video.name}")
        st.write(f"File size: {uploaded_video.size / (1024 * 1024):.2f} MB")

        st.info("Video is ready for model processing.")
    else:
        st.info("Please upload a video to start detection.")

elif input_source == "Live Camera":
    st.info("Live camera mode selected.")

    start_camera = st.button("Start Camera")

    if start_camera:
        st.warning("Camera integration will be connected to the inference module.")



# Initialize session metrics
if "frame_number" not in st.session_state:
    st.session_state.frame_number = 0

if "detection_confidence" not in st.session_state:
    st.session_state.detection_confidence = 0.0

if "fall_events" not in st.session_state:
    st.session_state.fall_events = 0

# Metrics panel
col1, col2, col3 = st.columns(3)

col1.metric(
    "Frame Number",
    st.session_state.frame_number
)

col2.metric(
    "Detection Confidence",
    f"{st.session_state.detection_confidence:.1%}"
)

col3.metric(
    "Fall Events",
    st.session_state.fall_events
)


# Fall detection alert logic (simulation mode)
st.subheader("Fall Detection Test")

test_confidence = st.slider(
    "Simulated Fall Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05
)

if st.button("Simulate Detection"):
    st.session_state.frame_number += 1
    st.session_state.detection_confidence = test_confidence

    
    # Simulate the standardized model output
    detections = [
        {
            "class_name": "fall",
            "confidence": test_confidence
        }
    ]

    fall_detected, fall_confidence = evaluate_detection(
        detections,
        confidence_threshold
    )

    st.session_state.detection_confidence = fall_confidence
    st.session_state.fall_alert = fall_detected

    if fall_detected:
        st.session_state.fall_events += 1


    st.rerun()

# Display warning when a fall is detected
if st.session_state.get("fall_alert", False):
    st.error("🚨 FALL DETECTED! Immediate attention required.")
else:
    st.success("No fall detected.")

st.divider()

st.caption(
    f"Input: {input_source} | "
    f"Model: {selected_model} | "
    f"Threshold: {confidence_threshold:.0%}"
)
