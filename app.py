import streamlit as st
import os
import sys
from config.settings import Config
from pages import object_detection, car_counter

def initialize_project():
    """Initialize project for cloud deployment"""
    try:
        # Ensure the project root is in Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)

        # Initialize configuration
        Config.initialize()

    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

def main():
    # Set up the main page configuration
    st.set_page_config(
        page_title="Computer Vision Demo",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add a navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Object Detection", "Car Counter"]
    )

    # Display the selected page
    if page == "Home":
        st.title("Computer Vision Demo")
        st.write("""
        Welcome to our Computer Vision Demo application! 
        
        This application showcases different computer vision capabilities:
        - Real-time Object Detection
        - Car Counting and Speed Detection
        
        Select a demo from the sidebar to get started.
        """)
        
    elif page == "Object Detection":
        from pages.object_detection import show
        show()
        
    elif page == "Car Counter":
        from pages.car_counter import show
        show()

if __name__ == "__main__":
    initialize_project()
    main()
