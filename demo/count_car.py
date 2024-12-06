import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import tempfile
import numpy as np
from pathlib import Path
import os

# Enable CUDA optimization
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Set page config
st.set_page_config(page_title="Car Counter", layout="wide")
st.title("Car Counter using YOLOv8")

# Load YOLOv8 model


@st.cache_resource
def load_model():
    model_name = "yolov8x.pt"
    model_path = f"models/{model_name}"

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Download model if it doesn't exist
    if not os.path.exists(model_path):
        st.info(f"Downloading {model_name}...")
        torch.hub.download_url_to_file(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{model_name}", model_path)

    # Load model
    model = YOLO(model_path)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model


# Initialize model with loading message
with st.spinner('Loading model...'):
    model = load_model()
    st.success(f"Model loaded on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# File uploader
video_file = st.file_uploader(
    "Upload a video file", type=['mp4', 'avi', 'mov'])

if video_file is not None:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    # Read video
    cap = cv2.VideoCapture(tfile.name)

    # Video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup video writer with GPU acceleration if available
    output_path = "output_video.mp4"
    if torch.cuda.is_available():
        fourcc = cv2.VideoWriter_fourcc(*'H264')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Progress bar and placeholders
    progress_bar = st.progress(0)
    col1, col2 = st.columns(2)
    frame_placeholder = col1.empty()
    metrics_placeholder = col2.empty()

    # Add new placeholder for speed table
    speed_table_placeholder = st.empty()

    # Initialize tracking
    model.tracker = "bytetrack.yaml"
    unique_cars = set()
    car_count = 0
    frame_number = 0

    # Initialize speed tracking with better structure
    car_positions = {}  # {track_id: [(x, y, timestamp), ...]}
    car_speeds = {}     # {track_id: current_speed}
    car_max_speeds = {}  # {track_id: max_speed}
    pixels_per_meter = 30  # Adjust based on your video

    # Function to update speed table
    def update_speed_table():
        table_data = {
            "Car ID": [],
            "Current Speed (km/h)": [],
            "Max Speed (km/h)": []
        }

        for car_id in sorted(car_speeds.keys()):
            table_data["Car ID"].append(car_id)
            table_data["Current Speed (km/h)"].append(
                f"{car_speeds.get(car_id, 0):.1f}")
            table_data["Max Speed (km/h)"].append(
                f"{car_max_speeds.get(car_id, 0):.1f}")

        speed_table_placeholder.table(table_data)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_number / fps  # Calculate current timestamp

            # Update progress
            progress = frame_number / frame_count
            progress_bar.progress(progress)

            # Process frame
            with torch.no_grad():  # Disable gradient calculation
                results = model.track(frame, persist=True)

            # Process detections
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    if box.cls[0] == 2:  # Car class
                        track_id = box.id[0] if box.id is not None else None
                        if track_id is not None:
                            track_id = int(track_id)
                            unique_cars.add(track_id)
                            x1, y1, x2, y2 = box.xyxy[0].astype(int)

                            # Calculate center point
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2

                            # Store position and timestamp
                            if track_id not in car_positions:
                                car_positions[track_id] = []
                            car_positions[track_id].append(
                                (center_x, center_y, current_time))

                            # Keep only last 5 positions for memory efficiency
                            car_positions[track_id] = car_positions[track_id][-5:]

                            # Calculate speed if we have at least 2 positions
                            if len(car_positions[track_id]) >= 2:
                                pos1 = car_positions[track_id][-2]
                                pos2 = car_positions[track_id][-1]

                                # Calculate distance and speed
                                distance = np.sqrt(
                                    (pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                                distance_meters = distance / pixels_per_meter
                                time_diff = pos2[2] - pos1[2]

                                if time_diff > 0:
                                    current_speed = (
                                        distance_meters / time_diff) * 3.6
                                    car_speeds[track_id] = current_speed

                                    # Update max speed
                                    if track_id not in car_max_speeds:
                                        car_max_speeds[track_id] = current_speed
                                    else:
                                        car_max_speeds[track_id] = max(
                                            car_max_speeds[track_id], current_speed)

                            # Draw bounding box with current speed
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)
                            speed_text = f"Car {track_id} ({car_speeds.get(track_id, 0):.1f} km/h)"
                            cv2.putText(frame, speed_text, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update speed table every frame
            update_speed_table()

            # Add counter overlay in top right corner
            rect_width = 240
            rect_height = 80
            padding = 10
            rect_x = width - rect_width - padding
            rect_y = padding

            cv2.rectangle(frame, (rect_x, rect_y), (rect_x +
                          rect_width, rect_y + rect_height), (0, 0, 0), -1)
            cv2.putText(frame, f'Current Cars: {len(result.boxes)}', (rect_x + 10, rect_y + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Total Unique Cars: {len(unique_cars)}', (rect_x + 10, rect_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write and display frame
            out.write(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Create speed table data
            table_data = {"Car ID": [], "Max Speed (km/h)": []}
            for car_id, speed in car_speeds.items():
                table_data["Car ID"].append(car_id)
                table_data["Max Speed (km/h)"].append(f"{speed:.1f}")

            # Display speed table
            metrics_placeholder.markdown("### Car Speed Statistics")
            metrics_placeholder.table(table_data)

            # Display original metrics
            metrics_placeholder.markdown(f"""
            ### Metrics
            - FPS: {fps}
            - Current Frame: {frame_number}/{frame_count}
            - Cars in Frame: {len(result.boxes)}
            - Total Unique Cars: {len(unique_cars)}
            """)

            frame_number += 1

            # Memory optimization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

    finally:
        # Cleanup
        cap.release()
        out.release()

        # Add download button
        with open(output_path, 'rb') as f:
            st.download_button(
                label="Download processed video",
                data=f,
                file_name="cars_tracked.mp4",
                mime="video/mp4"
            )

        # Display final speed statistics
        st.markdown("### Final Speed Statistics")
        final_stats = {
            "Car ID": [],
            "Max Speed (km/h)": []
        }
        for car_id, max_speed in sorted(car_max_speeds.items()):
            final_stats["Car ID"].append(car_id)
            final_stats["Max Speed (km/h)"].append(f"{max_speed:.1f}")

        st.table(final_stats)

        # Cleanup files
        if Path(output_path).exists():
            Path(output_path).unlink()
        Path(tfile.name).unlink()

        st.success(f"Video processing complete! Tracked {len(unique_cars)} unique cars.")
