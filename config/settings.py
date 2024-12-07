# config/settings.py
import os
import torch


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models/model")
    MODEL_PATH = os.path.join(MODEL_DIR, "yolov8x.pt")
    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"

    CONFIDENCE_THRESHOLD = 0.5
    SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def initialize(cls):
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
