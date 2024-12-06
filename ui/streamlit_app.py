# ui/streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from models.yolo_detector import YOLODetector
from config.settings import Config


def main():
    st.set_page_config(page_title="Real-time Object Detection", layout="wide")
    st.title("Real-time Object Detection with YOLOv8")

    # Initialize detector
    detector = YOLODetector()

    # Add source selection
    source = st.sidebar.selectbox("Select Input Source", [
                                  "Webcam", "Upload Image"])

    # Add confidence threshold slider
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


def run_webcam(detector):
    """Handle webcam input"""
    # Create placeholder for webcam feed
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")

    try:
        # Try multiple camera indices
        cap = None
        for i in range(3):  # Try first 3 camera indices
            cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                break
            if cap is not None:
                cap.release()

        if cap is None or not cap.isOpened():
            st.error(
                "No working webcam found. Please check your camera connection.")
            return

        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break

            try:
                # Process frame
                processed_frame = detector.process_frame(frame)

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Display frame
                frame_placeholder.image(
                    rgb_frame, channels="RGB", use_container_width=True)

            except Exception as e:
                st.error(f"Error processing frame: {str(e)}")
                break

    except Exception as e:
        st.error(f"Error accessing webcam: {str(e)}")
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()


def run_image_upload(detector):
    """Handle image upload"""
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=Config.SUPPORTED_IMAGE_TYPES
    )

    if uploaded_file is not None:
        try:
            # Convert uploaded file to opencv image
            file_bytes = np.asarray(
                bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Process image
            processed_image = detector.process_image(image)

            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            # Display image
            st.image(rgb_image, channels="RGB", use_container_width=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()
