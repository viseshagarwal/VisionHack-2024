import streamlit as st
from models.count_car import CarDetector
import torch

def show():
    st.markdown("""
    <style>
        .title-text { text-align: center; margin-bottom: 2rem; }
        .subtitle-text { text-align: center; color: #666666; margin-bottom: 2rem; }
        .info-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title-text'>Car Counter and Speed Tracker</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Upload a video to detect, count, and track cars with speed estimation</p>", unsafe_allow_html=True)

    # Add info box
    with st.expander("ℹ️ About this application", expanded=False):
        st.markdown("""
        This application uses YOLOv8 with ByteTrack to:
        - Detect and count cars in video footage
        - Track individual vehicles across frames
        - Estimate vehicle speeds
        - Generate detailed statistics
        
        **Supported video formats:** MP4, AVI, MOV
        """)

    # Initialize model with loading message in a container
    with st.container():
        with st.spinner('🔧 Initializing model...'):
            detector = CarDetector()
            st.success(f"✅ Model loaded successfully on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # File uploader with clear instructions
    st.markdown("### 📤 Upload Video")
    st.markdown("Select a video file to begin processing:")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        video_file = st.file_uploader(
            "",  # Remove label as we have a header above
            type=['mp4', 'avi', 'mov'],
            key="car_counter_video_upload"
        )

    if video_file:
        st.markdown("### 🎥 Processing Video")
        st.markdown("The video is being processed. Please wait...")
        detector.process_video(video_file)

if __name__ == "__main__":
    show()
