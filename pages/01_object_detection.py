import streamlit as st
from models.yolo_detector import YOLODetector
from config.settings import Config
from utils.camera import run_webcam, run_image_upload
import torch
import pandas as pd


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

    # Initialize session state for webcam control
    if 'webcam_on' not in st.session_state:
        st.session_state.webcam_on = False

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

    # Create columns for the main content
    col1, col2 = st.columns([2, 1])

    with col1:
        if source == "Webcam":
            st.markdown("### 📹 Webcam Feed")
            
            # Add webcam control button
            if st.button("Toggle Webcam" if st.session_state.webcam_on else "Start Webcam"):
                st.session_state.webcam_on = not st.session_state.webcam_on
            
            if st.session_state.webcam_on:
                # Create placeholders for detection results
                with col2:
                    st.markdown("### 📊 Live Detection Results")
                    current_detections_placeholder = st.empty()
                    
                    st.markdown("### 📈 Accumulated Detections")
                    total_detections_placeholder = st.empty()
                    
                    if st.button("Clear Detections"):
                        detector.clear_detections()
                
                # Run webcam with detection updates
                def detection_callback(detector):
                    # Update current detections
                    current_detections = detector.get_current_detections()
                    if current_detections:
                        df_current = pd.DataFrame({
                            'Object': list(current_detections.keys()),
                            'Count': list(current_detections.values())
                        })
                        current_detections_placeholder.table(df_current)
                    
                    # Update total detections
                    total_detections = detector.get_all_detections()
                    if total_detections:
                        df_total = pd.DataFrame({
                            'Object': list(total_detections.keys()),
                            'Count': list(total_detections.values())
                        })
                        total_detections_placeholder.table(df_total)

                run_webcam(detector, callback=detection_callback)
            else:
                st.info("Click 'Start Webcam' to begin detection")
        else:
            st.markdown("### 🖼️ Image Upload")
            st.markdown("Upload an image to perform object detection")
            run_image_upload(detector)

    if not st.session_state.webcam_on:
        with col2:
            st.markdown("### 📊 Detection Results")
            st.info("Start webcam or upload an image to see detections")

if __name__ == "__main__":
    show()
