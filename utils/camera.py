import cv2
import streamlit as st
import numpy as np
from config.settings import Config

def run_webcam(detector, callback=None):
    """Run webcam detection with optional callback for detection updates"""
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = detector.process_frame(frame)
        FRAME_WINDOW.image(processed_frame)

        # Call callback function if provided
        if callback:
            callback(detector)

    cap.release()

def run_image_upload(detector):
    uploaded_file = st.file_uploader("Choose an image...", type=Config.SUPPORTED_IMAGE_TYPES)

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            processed_image = detector.process_image(image)
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            st.image(rgb_image, channels="RGB", use_container_width=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
