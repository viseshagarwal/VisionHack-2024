import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import tempfile
import numpy as np
from pathlib import Path
import os
from config.settings import Config


class CarDetector:
    def __init__(self):
        # Enable CUDA optimization
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Download model if it doesn't exist
        if not os.path.exists(Config.MODEL_PATH):
            print(f"Downloading YOLOv8x model...")
            torch.hub.download_url_to_file(
                Config.MODEL_URL,
                Config.MODEL_PATH
            )

        self.model = YOLO(Config.MODEL_PATH)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Tracking parameters
        self.pixels_per_meter = 30
        self.model.tracker = "bytetrack.yaml"
        self._reset_tracking_vars()

    def _load_model(self):
        if not os.path.exists(Config.MODEL_PATH):
            print(f"Downloading YOLOv8x model...")
            os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
            torch.hub.download_url_to_file(
                Config.MODEL_URL,
                Config.MODEL_PATH
            )

        return YOLO(Config.MODEL_PATH)

    def _reset_tracking_vars(self):
        self.unique_cars = set()
        self.car_positions = {}
        self.car_speeds = {}
        self.car_max_speeds = {}
        self.frame_number = 0

    def _calculate_speed(self, track_id, center_x, center_y, current_time):
        if track_id not in self.car_positions:
            self.car_positions[track_id] = []

        self.car_positions[track_id].append((center_x, center_y, current_time))
        # Keep last 5 positions
        self.car_positions[track_id] = self.car_positions[track_id][-5:]

        if len(self.car_positions[track_id]) >= 2:
            pos1 = self.car_positions[track_id][-2]
            pos2 = self.car_positions[track_id][-1]

            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            distance_meters = distance / self.pixels_per_meter
            time_diff = pos2[2] - pos1[2]

            if time_diff > 0:
                current_speed = (distance_meters / time_diff) * 3.6
                self.car_speeds[track_id] = current_speed

                if track_id not in self.car_max_speeds:
                    self.car_max_speeds[track_id] = current_speed
                else:
                    self.car_max_speeds[track_id] = max(
                        self.car_max_speeds[track_id], current_speed)

    def process_video(self, video_file):
        # Create temporary file for video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = tfile.name
        tfile.write(video_file.read())
        tfile.close()  # Close the file immediately after writing

        cap = cv2.VideoCapture(temp_filename)

        # Video properties
        self.video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(
            *'H264') if torch.cuda.is_available() else cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc,
                              self.video_fps, (width, height))

        # Create Streamlit UI elements with better organization
        st.markdown("### 📊 Processing Status")
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Create two columns for video and metrics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎬 Video Feed")
            frame_placeholder = st.empty()
        
        with col2:
            st.markdown("#### 📈 Live Metrics")
            metrics_placeholder = st.empty()
            
        st.markdown("#### 🚗 Vehicle Speed Tracking")
        speed_table_placeholder = st.empty()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = self.frame_number / self.video_fps
                progress_bar.progress(self.frame_number / frame_count)

                # Process frame
                with torch.no_grad():
                    results = self.model.track(frame, persist=True)

                processed_frame = self._process_detections(
                    frame, results[0], width, height)

                # Update UI
                self._update_ui(frame_placeholder, metrics_placeholder, speed_table_placeholder,
                                processed_frame, results[0], self.video_fps, frame_count)

                out.write(processed_frame)
                self.frame_number += 1

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            cap.release()
            out.release()
            try:
                # Wait a bit before trying to delete the file
                import time
                time.sleep(0.1)
                os.unlink(temp_filename)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")
                # Continue execution even if temp file deletion fails

            # Add download button
            with open(output_path, 'rb') as f:
                st.download_button(
                    label="Download processed video",
                    data=f,
                    file_name="cars_tracked.mp4",
                    mime="video/mp4"
                )

            self._display_final_stats()

    def _process_detections(self, frame, result, width, height):
        for box in result.boxes:
            if box.cls[0] == 2:  # Car class
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is not None:
                    self.unique_cars.add(track_id)
                    # Convert tensor to numpy array before using astype
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    current_time = self.frame_number / self.video_fps
                    self._calculate_speed(
                        track_id, center_x, center_y, current_time)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    speed_text = f"Car {
                        track_id} ({self.car_speeds.get(track_id, 0):.1f} km/h)"
                    cv2.putText(frame, speed_text, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add counter overlay
        rect_width = 240
        rect_height = 80
        padding = 10
        rect_x = width - rect_width - padding
        rect_y = padding

        cv2.rectangle(frame, (rect_x, rect_y),
                      (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
        cv2.putText(frame, f'Current Cars: {len(result.boxes)}',
                    (rect_x + 10, rect_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Unique Cars: {len(self.unique_cars)}',
                    (rect_x + 10, rect_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def _update_ui(self, frame_placeholder, metrics_placeholder, speed_table_placeholder,
                   frame, result, fps, frame_count):
        # Update frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update metrics with better formatting
        metrics_placeholder.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;'>
            <h4>Real-time Statistics</h4>
            <ul>
                <li>FPS: {fps}</li>
                <li>Frame: {self.frame_number}/{frame_count}</li>
                <li>Cars in Frame: {len(result.boxes)}</li>
                <li>Total Unique Cars: {len(self.unique_cars)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Update speed table
        table_data = {
            "Car ID": [],
            "Current Speed (km/h)": [],
            "Max Speed (km/h)": []
        }

        for car_id in sorted(self.car_speeds.keys()):
            table_data["Car ID"].append(car_id)
            table_data["Current Speed (km/h)"].append(
                f"{self.car_speeds[car_id]:.1f}")
            # Fix the double colon format specifier
            table_data["Max Speed (km/h)"].append(
                f"{self.car_max_speeds.get(car_id, 0):.1f}")

        speed_table_placeholder.table(table_data)

    def _display_final_stats(self):
        st.markdown("### 📊 Final Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Cars Tracked", len(self.unique_cars))
        with col2:
            if self.car_max_speeds:
                max_speed = max(self.car_max_speeds.values())
                st.metric("Highest Speed Recorded", f"{max_speed:.1f} km/h")
        
        st.markdown("#### 🚗 Vehicle Speed Summary")
        final_stats = {
            "Car ID": [],
            "Max Speed (km/h)": []
        }

        for car_id, max_speed in sorted(self.car_max_speeds.items()):
            final_stats["Car ID"].append(car_id)
            final_stats["Max Speed (km/h)"].append(f"{max_speed:.1f}")

        st.table(final_stats)
        st.success(f"Video processing complete! Tracked {
                   len(self.unique_cars)} unique cars.")
