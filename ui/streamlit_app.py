# ui/streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2

from models.pretrained_models import model_loader
from detection.video_processor import video_processor
from analytics.anomaly_detection import anomaly_detector
from config.settings import Config


def draw_detections(image, boxes, scores, classes):
    """
    Draw detection results on image

    Args:
        image (PIL.Image): Input image
        boxes, scores, classes (numpy arrays): Detection results

    Returns:
        PIL.Image: Image with detections drawn
    """
    draw = ImageDraw.Draw(image)

    for box, score, cls in zip(boxes, scores, classes):
        ymin, xmin, ymax, xmax = box
        (left, top, right, bottom) = (
            xmin * image.width,
            ymin * image.height,
            xmax * image.width,
            ymax * image.height
        )

        draw.rectangle(
            [left, top, right, bottom],
            outline="red",
            width=3
        )

        label = f"{model_loader.category_index.get(cls, 'Unknown')} ({
            score:.2f})"
        draw.text((left, top), label, fill="red")

    return image


def main():
    st.title("Advanced Object Detection System")

    # Sidebar configuration
    st.sidebar.header("Detection Configuration")
    input_type = st.sidebar.radio(
        "Choose Input Type",
        ["Image", "Video"]
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0, 1.0,
        Config.CONFIDENCE_THRESHOLD,
        0.01
    )

    # Load labels if not already loaded
    if not model_loader.category_index:
        model_loader.load_labels()

    if input_type == "Image":
        handle_image_detection(confidence_threshold)
    else:
        handle_video_detection(confidence_threshold)


def handle_image_detection(confidence_threshold):
    """
    Handle image detection workflow
    """
    uploaded_image = st.file_uploader(
        "Upload an image...",
        type=Config.SUPPORTED_IMAGE_TYPES
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to numpy for detection
        image_np = np.array(image)

        # Detect objects
        boxes, scores, classes = model_loader.detect_objects(
            image_np,
            confidence_threshold
        )

        # Update anomaly detector
        anomaly_detector.update_history((boxes, scores, classes))

        # Draw detections
        detected_image = draw_detections(
            image.copy
            # ui/streamlit_app.py (continued)
        )

        # Detect anomalies
        anomaly_score = anomaly_detector.calculate_anomaly_score()
        unusual_patterns = anomaly_detector.detect_unusual_patterns()

        # Display anomaly information
        if anomaly_score > 0.5:
            st.warning(f"Potential Anomalies Detected! Anomaly Score: {
                       anomaly_score:.2f}")
            if unusual_patterns:
                st.write("Unusual Patterns:")
                for pattern in unusual_patterns:
                    st.info(f"New Objects Appeared: {pattern['objects']}")

        # Display detected image
        st.image(detected_image, caption="Detected Objects",
                 use_column_width=True)


def handle_video_detection(confidence_threshold):
    """
    Handle video detection workflow
    """
    uploaded_video = st.file_uploader(
        "Upload a video...",
        type=Config.SUPPORTED_VIDEO_TYPES
    )

    if uploaded_video is not None:
        # Save uploaded video
        video_path = f"temp_video_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        # Display original video
        st.video(uploaded_video)

        # Create placeholders for video processing
        status_placeholder = st.empty()
        video_placeholder = st.empty()
        anomaly_placeholder = st.empty()

        # Process video with detection callback
        def process_frame(frame, boxes, scores, classes):
            """
            Callback for processing each video frame
            """
            for box, score, cls in zip(boxes, scores, classes):
                if score >= confidence_threshold:
                    ymin, xmin, ymax, xmax = box
                    (left, top, right, bottom) = (
                        int(xmin * frame.shape[1]),
                        int(ymin * frame.shape[0]),
                        int(xmax * frame.shape[1]),
                        int(ymax * frame.shape[0])
                    )
                    cv2.rectangle(
                        frame,
                        (left, top),
                        (right, bottom),
                        (0, 255, 0),
                        2
                    )

                    label = f"{model_loader.category_index.get(cls, 'Unknown')}: {
                        score:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            return frame

        # Track video processing
        total_frames = 0
        processed_frames = 0
        anomaly_frames = 0

        for frame, (boxes, scores, classes) in video_processor.process_video(
            video_path,
            callback=process_frame
        ):
            total_frames += 1

            # Update anomaly detection
            anomaly_detector.update_history((boxes, scores, classes))

            # Check for anomalies
            anomaly_score = anomaly_detector.calculate_anomaly_score()
            if anomaly_score > 0.5:
                anomaly_frames += 1

            # Update placeholders
            status_placeholder.text(
                f"Processing: {processed_frames}/{total_frames} frames"
            )
            video_placeholder.image(frame, channels="BGR")

            processed_frames += 1

            # Optional: Add a small delay for visualization
            if st.button("Stop Processing"):
                break

        # Final anomaly report
        anomaly_percentage = (anomaly_frames / total_frames) * \
            100 if total_frames > 0 else 0
        anomaly_placeholder.metric(
            "Anomaly Analysis",
            f"{anomaly_percentage:.2f}% of frames with potential anomalies"
        )


def setup_logging():
    """
    Configure logging for the application
    """
    import logging
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=Config.LOG_FORMAT
    )


if __name__ == "__main__":
    setup_logging()
    main()
