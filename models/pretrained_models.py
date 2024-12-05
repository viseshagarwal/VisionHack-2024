# # models/pretrained_models.py
# import os
# import tensorflow as tf
# import numpy as np
# from config.settings import Config
# import logging
# import requests
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")


# class ModelLoader:
#     MODELS = {
#         'faster_rcnn_resnet50': {
#             'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
#             'local_path': os.path.join(Config.MODELS_DIR, 'faster_rcnn_resnet50')
#         }
#     }

#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.model = None
#         self.category_index = {}

#     def _download_model(self, model_key='faster_rcnn_resnet50'):
#         """
#         Download pre-trained model if not exists
#         """
#         model_info = self.MODELS[model_key]
#         os.makedirs(model_info['local_path'], exist_ok=True)

#         # Check if model is already downloaded
#         saved_model_path = os.path.join(
#             model_info['local_path'], 'faster_rcnn_resnet50_coco_2018_01_28/saved_model/')
#         if os.path.exists(saved_model_path):
#             return saved_model_path

#         self.logger.info(f"Downloading model: {model_key}")
#         try:
#             import urllib.request
#             import tarfile

#             # Download the model
#             tar_path = os.path.join(model_info['local_path'], 'model.tar.gz')
#             urllib.request.urlretrieve(model_info['url'], tar_path)

#             # Extract the model
#             with tarfile.open(tar_path, 'r:gz') as tar:
#                 tar.extractall(path=model_info['local_path'])

#             # Remove the tar file
#             os.remove(tar_path)

#             return saved_model_path
#         except Exception as e:
#             self.logger.error(f"Model download failed: {e}")
#             raise

#     def load_model(self, model_key='faster_rcnn_resnet50'):
#         """
#         Load a pre-trained TensorFlow object detection model
#         """
#         try:
#             # Ensure model is downloaded
#             model_path = self._download_model(model_key)

#             # Verify model path
#             saved_model_files = ['saved_model.pb', 'saved_model.pbtxt']
#             if not any(os.path.exists(os.path.join(model_path, f)) for f in saved_model_files):
#                 raise FileNotFoundError(
#                     f"No saved model found in {model_path}")

#             self.model = tf.saved_model.load(model_path)
#             self.logger.info(f"Model loaded successfully from {model_path}")
#             return self.model
#         except Exception as e:
#             self.logger.error(f"Error loading model: {e}")
#             raise

#     def load_labels(self):
#         """
#         Load COCO labels
#         """
#         try:
#             # Use TensorFlow's built-in COCO labels
#             from object_detection.utils import label_map_util

#             label_map_path = tf.keras.utils.get_file(
#                 'mscoco_label_map.pbtxt',
#                 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
#             )

#             label_map = label_map_util.load_labelmap(label_map_path)
#             categories = label_map_util.convert_label_map_to_categories(
#                 label_map, max_num_classes=90, use_display_name=True
#             )

#             self.category_index = {
#                 item['id']: item['name'] for item in categories
#             }

#             self.logger.info(f"Loaded {len(self.category_index)} labels")
#             return self.category_index
#         except Exception as e:
#             self.logger.error(f"Error loading labels: {e}")
#             raise

#     def detect_objects(self, image_np, confidence_threshold=0.5):
#         """
#         Perform object detection on an image
#         """
#         if self.model is None:
#             self.load_model()

#         input_tensor = tf.convert_to_tensor(image_np)
#         input_tensor = input_tensor[tf.newaxis, ...]

#         detections = self.model(input_tensor)

#         num_detections = int(detections.pop("num_detections"))
#         detections = {key: value[0, :num_detections].numpy()
#                       for key, value in detections.items()}

#         detection_boxes = detections["detection_boxes"]
#         detection_scores = detections["detection_scores"]
#         detection_classes = detections["detection_classes"].astype(np.int64)

#         # Filter detections by confidence
#         mask = detection_scores >= confidence_threshold
#         return (
#             detection_boxes[mask],
#             detection_scores[mask],
#             detection_classes[mask]
#         )


# # Singleton instance for easy import
# model_loader = ModelLoader()

# models/pretrained_models.py
import os
import logging
import warnings
import urllib.request
import tarfile

import tensorflow as tf
import numpy as np

from config.settings import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress warnings
warnings.filterwarnings("ignore")


class ModelLoader:
    """
    A class to manage loading and using pre-trained object detection models.

    Supports downloading, loading, and using TensorFlow object detection models.
    """

    # Predefined model configurations
    MODELS = {
        'faster_rcnn_resnet50': {
            'url': 'http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz',
            'local_path': os.path.join(Config.MODELS_DIR, 'faster_rcnn_resnet50')
        }
    }

    # Comprehensive COCO labels for fallback
    COCO_LABELS = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
        6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
        11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
        16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
        21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
        26: 'backpack', 27: 'umbrella', 28: 'handbag', 29: 'tie', 30: 'suitcase',
        31: 'frisbee', 32: 'skis', 33: 'snowboard', 34: 'sports ball', 35: 'kite',
        36: 'baseball bat', 37: 'baseball glove', 38: 'skateboard', 39: 'surfboard',
        40: 'tennis racket', 41: 'bottle', 42: 'wine glass', 43: 'cup', 44: 'fork',
        45: 'knife', 46: 'spoon', 47: 'bowl', 48: 'banana', 49: 'apple',
        50: 'sandwich', 51: 'orange', 52: 'broccoli', 53: 'carrot', 54: 'hot dog'
    }

    def __init__(self):
        """
        Initialize the ModelLoader with logging and default states.
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.category_index = {}

    def _download_model(self, model_key='faster_rcnn_resnet50'):
        """
        Download pre-trained model if it doesn't exist.

        Args:
            model_key (str): Key of the model to download from MODELS dict.

        Returns:
            str: Path to the downloaded and extracted model
        """
        model_info = self.MODELS[model_key]
        os.makedirs(model_info['local_path'], exist_ok=True)

        # Check if model is already downloaded
        saved_model_path = os.path.join(
            model_info['local_path'],
            'faster_rcnn_resnet50_coco_2018_01_28/saved_model/'
        )
        if os.path.exists(saved_model_path):
            return saved_model_path

        self.logger.info(f"Downloading model: {model_key}")
        try:
            # Download the model
            tar_path = os.path.join(model_info['local_path'], 'model.tar.gz')
            urllib.request.urlretrieve(model_info['url'], tar_path)

            # Extract the model
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=model_info['local_path'])

            # Remove the tar file
            os.remove(tar_path)

            return saved_model_path
        except Exception as e:
            self.logger.error(f"Model download failed: {e}")
            raise

    def load_model(self, model_key='faster_rcnn_resnet50'):
        """
        Load a pre-trained TensorFlow object detection model.

        Args:
            model_key (str): Key of the model to load from MODELS dict.

        Returns:
            tf.saved_model: Loaded TensorFlow model
        """
        try:
            # Ensure model is downloaded
            model_path = self._download_model(model_key)

            # Verify model path
            saved_model_files = ['saved_model.pb', 'saved_model.pbtxt']
            if not any(os.path.exists(os.path.join(model_path, f)) for f in saved_model_files):
                raise FileNotFoundError(
                    f"No saved model found in {model_path}")

            self.model = tf.saved_model.load(model_path)
            self.logger.info(f"Model loaded successfully from {model_path}")
            return self.model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def load_labels(self):
        """
        Load object detection labels with a robust fallback mechanism.

        Returns:
            dict: Mapping of class IDs to label names
        """
        try:
            # First, try importing label_map_util if available
            try:
                from object_detection.utils import label_map_util
                label_map_path = tf.keras.utils.get_file(
                    'mscoco_label_map.pbtxt',
                    'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt'
                )
                label_map = label_map_util.load_labelmap(label_map_path)
                categories = label_map_util.convert_label_map_to_categories(
                    label_map, max_num_classes=90, use_display_name=True
                )
                self.category_index = {
                    item['id']: item['name'] for item in categories
                }
            except ImportError:
                # Fallback to predefined COCO labels
                self.category_index = self.COCO_LABELS

            self.logger.info(f"Loaded {len(self.category_index)} labels")
            return self.category_index
        except Exception as e:
            self.logger.error(f"Error loading labels: {e}")
            return self.COCO_LABELS

    def detect_objects(self, image_np, confidence_threshold=None):
        """
        Perform object detection on an image.

        Args:
            image_np (np.ndarray): Input image as a numpy array
            confidence_threshold (float, optional): Minimum confidence for detection

        Returns:
            tuple: Filtered detection boxes, scores, and classes
        """
        # Use default confidence threshold if not provided
        threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD

        # Ensure model is loaded
        if self.model is None:
            self.load_model()

        # Prepare input tensor
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Perform detection
        detections = self.model(input_tensor)

        # Process detections
        num_detections = int(detections.pop("num_detections"))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}

        detection_boxes = detections["detection_boxes"]
        detection_scores = detections["detection_scores"]
        detection_classes = detections["detection_classes"].astype(np.int64)

        # Filter detections by confidence
        mask = detection_scores >= threshold
        return (
            detection_boxes[mask],
            detection_scores[mask],
            detection_classes[mask]
        )


# Create a singleton instance for easy import
model_loader = ModelLoader()
