import cv2
import streamlit as st
import numpy as np
from config.settings import Config

def run_webcam(detector):
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    try:
        cap = None
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                break
            if cap is not None:
                cap.release()

        if cap is None or not cap.isOpened():
            st.error("No working webcam found.")
            return

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = detector.process_frame(frame)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

    finally:
        if 'cap' in locals() and cap is not None:
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
