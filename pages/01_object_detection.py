import streamlit as st
from models.yolo_detector import YOLODetector
from config.settings import Config
from utils.camera import run_webcam, run_image_upload

def show():
    st.title("Real-time Object Detection")
    
    detector = YOLODetector()
    
    source = st.sidebar.selectbox("Select Input Source", ["Webcam", "Upload Image"])
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 
        Config.CONFIDENCE_THRESHOLD, 
        0.05
    )
    detector.model.conf = confidence_threshold

    if source == "Webcam":
        st.header("Webcam Live Feed")
        run_webcam(detector)
    else:
        st.header("Image Upload") 
        run_image_upload(detector)

if __name__ == "__main__":
    show()
