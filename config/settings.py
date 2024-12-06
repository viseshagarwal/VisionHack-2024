# config/settings.py
import os
import platform
import torch


class Config:
    # Path handling for Windows
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "weights", "yolov8n.pt")
    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

    # MIDAS_MODEL_PATH = "MiDaS_small"  # or "DPT_Large" for better accuracy
    MIDAS_MODEL_PATH = "DPT_Large"  # or "DPT_Large" for better accuracy
    CAMERA_FOCAL_LENGTH = 100  # adjust based on your webcam
    CALIB_FILE_PATH = os.path.join(
        BASE_DIR, "config", "camera_calibration.pkl")
    # Windows-specific settings
    IS_WINDOWS = platform.system() == "Windows"

    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    SPEAK_COOLDOWN = 3
    SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

    # Windows camera index (usually 0 for built-in webcam)
    WEBCAM_INDEX = 0

    # Text-to-speech settings for Windows
    TTS_RATE = 150
    TTS_VOLUME = 1.0

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 640  # Input image size
    CONF_THRESHOLD = 0.25  # Confidence threshold
    IOU_THRESHOLD = 0.45  # NMS IoU threshold

    @classmethod
    def initialize(cls):
        """Initialize configuration for Windows environment"""
        try:
            # Create model directory using Windows-safe path
            model_dir = os.path.dirname(cls.MODEL_PATH)
            os.makedirs(model_dir, exist_ok=True)

            # Download model if not exists
            if not os.path.exists(cls.MODEL_PATH):
                import wget
                print(f"Downloading YOLOv8 model to {cls.MODEL_PATH}...")
                wget.download(cls.MODEL_URL, cls.MODEL_PATH)
                print("\nModel downloaded successfully!")
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise
