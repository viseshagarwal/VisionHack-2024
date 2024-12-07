# pages/2_Car_Counter.py
import streamlit as st
from models.count_car import CarDetector
import torch


def show():
    # st.set_page_config(page_title="Car Counter", layout="wide")
    st.title("Car Counter and Tracker")

    # Initialize model with loading message
    with st.spinner('Loading model...'):
        detector = CarDetector()
        st.success(f"Model loaded on: {torch.cuda.get_device_name(
            0) if torch.cuda.is_available() else 'CPU'}")

    # Add unique key to file uploader
    video_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov'],
        key="car_counter_video_upload"
    )

    if video_file:
        detector.process_video(video_file)


if __name__ == "__main__":
    show()
