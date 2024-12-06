# # # config/settings.py
# # import os
# # import warnings

# # # Suppress warnings
# # warnings.filterwarnings("ignore")


# # class Config:
# #     # Project Paths
# #     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# #     MODELS_DIR = os.path.join(BASE_DIR, 'models')
# #     LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# #     # Model Configuration
# #     PRETRAINED_MODEL_PATH = os.path.join(MODELS_DIR, 'faster_rcnn_resnet50')
# #     LABEL_MAP_PATH = os.path.join(MODELS_DIR, 'mscoco_label_map.pbtxt')

# #     # Detection Settings
# #     CONFIDENCE_THRESHOLD = 0.5
# #     IOU_THRESHOLD = 0.5

# #     # Logging Configuration
# #     LOG_LEVEL = 'INFO'
# #     LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# #     # Performance Tracking
# #     MAX_TRACKING_HISTORY = 100

# #     # Supported Input Types
# #     SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
# #     SUPPORTED_VIDEO_TYPES = ['.mp4', '.avi', '.mov']

# #     @classmethod
# #     def create_directories(cls):
# #         """Create necessary project directories."""
# #         os.makedirs(cls.MODELS_DIR, exist_ok=True)
# #         os.makedirs(cls.LOGS_DIR, exist_ok=True)


# # # Initialize directories on import
# # Config.create_directories()

# # config/settings.py
# class Config:
#     # Model settings
#     CONFIDENCE_THRESHOLD = 0.5
#     IOU_THRESHOLD = 0.45
    
#     # Voice settings
#     SPEECH_RATE = 150
#     SPEAK_COOLDOWN = 3
    
#     # Logging
#     LOG_LEVEL = "INFO"
#     LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
#     LOGS_DIR = "logs"
    
#     # Interface
#     SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
#     MAX_DETECTION_HISTORY = 100

# config/settings.py
# config/settings.py
import os

class Config:
    # Use environment variable for port if available
    PORT = int(os.environ.get('PORT', 8501))
    
    # Model path handling
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "weights", "yolov8n.pt")
    
    # Download URL for model
    MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    CONFIDENCE_THRESHOLD = 0.5
    SPEAK_COOLDOWN = 3
    SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

    @classmethod
    def initialize(cls):
        """Initialize configuration"""
        try:
            # Create necessary directories
            os.makedirs(os.path.dirname(cls.MODEL_PATH), exist_ok=True)
            
            # Download model if not exists
            if not os.path.exists(cls.MODEL_PATH):
                import wget
                print(f"Downloading YOLOv8 model to {cls.MODEL_PATH}...")
                wget.download(cls.MODEL_URL, cls.MODEL_PATH)
                print("\nModel downloaded successfully!")
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise