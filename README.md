# VisionHack-2024: Computer Vision Applications

A comprehensive computer vision application suite built with YOLOv8 and Streamlit, featuring real-time object detection and vehicle tracking capabilities.

---

## 🌟 Features

- **Real-time Object Detection**

  - Webcam integration for live detection
  - Support for image upload and analysis
  - Adjustable confidence thresholds
  - Detection statistics and counting

- **Vehicle Tracking System**
  - Car counting in video footage
  - Speed estimation
  - Vehicle tracking across frames
  - Detailed statistics and reporting

---

## 🛠️ Technology Stack

- Python 3.8+
- YOLOv8
- Streamlit
- OpenCV
- PyTorch

---

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for optimal performance)
- Webcam (for real-time detection)

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/viseshagarwal/VisionHack-2024.git
   cd VisionHack-2024
   ```

2. Create a new virtual environment:

   ```bash
   # For Linux and Mac
   python -m venv venv
   source venv/bin/activate

   # For Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download Cuda Toolkit :

   ```bash
   https://developer.nvidia.com/cuda-downloads
   ```

5. Run the Streamlit app:
   ```bash
   streamlit run Home.py
   ```

---

## 🚀 Running the Application

1. Start the Streamlit app by running the following command:
   ```bash
   streamlit run Home.py
   ```
2. Open the browser and navigate to the URL displayed in the terminal.
3. Select the desired application mode from the sidebar.
4. Upload an image or start the webcam to begin object detection.
5. For vehicle tracking, upload a video file and adjust the settings as needed.

---

## 📱 Usage Guide

### Object Detection Module

1. Select **"Object Detection"** from the sidebar.
2. Choose input source:
   - **Webcam**: For real-time detection
   - **Upload Image**: For image analysis
3. Adjust the confidence threshold using the slider.
4. View detection results and statistics in real-time.

### Car Counter Module

1. Select **"Car Counter"** from the sidebar.
2. Upload a video file for processing.
3. Monitor:
   - Real-time car counting
   - Speed estimation
   - Vehicle tracking
   - Statistical analysis

---

## 📁 Project Structure

```
VisionHack-2024/
├── Home.py                # Main application entry point
├── app.py                 # Application initialization
├── config/
│   └── settings.py        # Configuration settings
├── models/
│   ├── yolo_detector.py   # YOLOv8 detection implementation
│   └── count_car.py       # Car counting implementation
├── pages/
│   ├── object_detection.py
│   └── car_counter.py
├── ui/
│   └── streamlit_app.py   # Streamlit UI components
├── utils/
│   └── camera.py          # Camera handling utilities
└── requirements.txt       # Project dependencies
```

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/NewFeature
   ```
3. Commit changes:
   ```bash
   git commit -m 'Add NewFeature'
   ```
4. Push to branch:
   ```bash
   git push origin feature/NewFeature
   ```
5. Submit a Pull Request.

---

## 🐛 Bug Reports

Please report bugs by creating issues on GitHub with:

- A detailed description of the issue.
- Steps to reproduce.
- Expected vs actual behavior.
- System information.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **Streamlit** team for the amazing framework
- **ByteTrack** for object tracking capabilities

---

## 📬 Contact

For questions or feedback, please reach out through:

- GitHub Issues
- Email:
  - [Visesh Agarwal](mailto:viseshagarwal@outlook.com)
  - [Shreya Goel](mailto:shreyagoel9560@gmail.com)

---

## ⭐ Support

If you find this project helpful, please consider giving it a ⭐ on GitHub!

Made with ❤️ by **Team Visionary Minds**

---

## Contributors

- [Visesh Agarwal](https://www.linkedin.com/in/viseshagarwal/)
- [Shreya Goel](https://www.linkedin.com/in/shreya-goel-94694422a/)
