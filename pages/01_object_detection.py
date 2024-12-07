import streamlit as st
from models.yolo_detector import YOLODetector
from config.settings import Config
from utils.camera import run_webcam, run_image_upload
import torch


def show():
    st.markdown("""
    <style>
        .title-text { text-align: center; margin-bottom: 2rem; }
        .subtitle-text { text-align: center; color: #666666; margin-bottom: 2rem; }
        .info-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .sidebar-content { padding: 1rem; background-color: #f8f9fa; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title-text'>🔍 Real-time Object Detection</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Detect objects in real-time using webcam or uploaded images</p>",
                unsafe_allow_html=True)

    # Add info box
    with st.expander("ℹ️ About this application", expanded=False):
        st.markdown("""
        This application uses YOLOv8 to:
        - Detect 80+ different types of objects
        - Process real-time webcam feed
        - Analyze uploaded images
        - Provide confidence scores for detections
        
        **Supported image formats:** JPG, JPEG, PNG
        """)

    # Initialize detector with loading message
    with st.spinner('🔧 Initializing YOLOv8 model...'):
        detector = YOLODetector()
        st.success(f"✅ Model loaded successfully on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Sidebar controls with better styling
    st.sidebar.markdown("<div class='sidebar-content'>",
                        unsafe_allow_html=True)
    source = st.sidebar.selectbox("📥 Select Input Source", [
                                  "Webcam", "Upload Image"])

    confidence_threshold = st.sidebar.slider(
        "🎯 Confidence Threshold",
        0.0, 1.0,
        Config.CONFIDENCE_THRESHOLD,
        0.05
    )
    detector.model.conf = confidence_threshold
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # Main content
    if source == "Webcam":
        st.markdown("### 📹 Webcam Feed")
        st.markdown("Using your computer's webcam for real-time detection")
        run_webcam(detector)
    else:
        st.markdown("### 🖼️ Image Upload")
        st.markdown("Upload an image to perform object detection")
        run_image_upload(detector)


if __name__ == "__main__":
    show()
