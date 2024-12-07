import streamlit as st
from config.settings import Config

def main():
    st.set_page_config(page_title="Computer Vision Dashboard", layout="wide")
    st.title("Computer Vision Applications")

    st.markdown("""
    ### Available Applications:
    1. **Object Detection**: Real-time object detection using YOLOv8
    2. **Car Counter**: Count and track vehicles in videos
    
    Select an application from the sidebar to begin.
    """)

if __name__ == "__main__":
    Config.initialize()
    main()
