import streamlit as st
from config.settings import Config


def main():
    st.set_page_config(page_title="Computer Vision Dashboard", layout="wide")

    st.markdown("""
    <style>
        .main-title { text-align: center; color: #2e7d32; margin-bottom: 2rem; }
        .app-card { background-color: #f0f2f6; padding: 1.5rem; border-radius: 0.8rem; margin-bottom: 1rem; }
        .app-title { color: #1e88e5; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-title'>🖼️ VisionHack - 2024 :  Computer Vision Applications</h1>",
                unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align: center; font-size: 1.2em; color: #666; margin-bottom: 2rem;'>
        Explore our suite of computer vision applications powered by YOLOv8
    </p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='app-card'>
            <h3 class='app-title'>🔍 Object Detection</h3>
            <p>Real-time object detection using YOLOv8</p>
            <ul>
                <li>Detect multiple object classes</li>
                <li>Real-time webcam processing</li>
                <li>Support for image uploads</li>
                <li>Adjustable confidence thresholds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='app-card'>
            <h3 class='app-title'>🚗 Car Counter</h3>
            <p>Advanced vehicle tracking and analysis</p>
            <ul>
                <li>Count vehicles in videos</li>
                <li>Track individual vehicles</li>
                <li>Estimate vehicle speeds</li>
                <li>Generate detailed statistics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 2rem; padding: 1rem; background-color: #e3f2fd; border-radius: 0.5rem;'>
        <h4>Getting Started</h4>
        <p>Select an application from the sidebar to begin your computer vision journey!</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    Config.initialize()
    main()
