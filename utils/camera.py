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


def run_image_upload(detector, callback=None):
    uploaded_file = st.file_uploader(
        "Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Process the image
        processed_image = detector.process_image(image)

        # Call the callback function if provided
        if callback:
            callback(detector)

        # Display the processed image
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                 caption='Processed Image',
                 use_container_width=True)
