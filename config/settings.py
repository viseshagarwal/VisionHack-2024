# config/settings.py
import os

class Config:
    # Project Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Model Configuration
    PRETRAINED_MODEL_PATH = os.path.join(MODELS_DIR, 'faster_rcnn_resnet50')
    LABEL_MAP_PATH = os.path.join(MODELS_DIR, 'mscoco_label_map.pbtxt')
    
    # Detection Settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Logging Configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Performance Tracking
    MAX_TRACKING_HISTORY = 100
    
    # Supported Input Types
    SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png']
    SUPPORTED_VIDEO_TYPES = ['.mp4', '.avi', '.mov']

    @classmethod
    def create_directories(cls):
        """Create necessary project directories."""
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.LOGS_DIR, exist_ok=True)

# Initialize directories on import
Config.create_directories()
