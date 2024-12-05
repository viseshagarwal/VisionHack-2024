# models/pretrained_models.py
import tensorflow as tf
import numpy as np
from config.settings import Config
import logging

class ModelLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.category_index = {}

    def load_model(self, model_path=None):
        """
        Load a pre-trained TensorFlow object detection model
        
        Args:
            model_path (str, optional): Path to the saved model. 
                                        Defaults to config path if not provided.
        
        Returns:
            tf.saved_model: Loaded TensorFlow model
        """
        try:
            path = model_path or Config.PRETRAINED_MODEL_PATH
            self.model = tf.saved_model.load(path)
            self.logger.info(f"Model loaded successfully from {path}")
            return self.model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def load_labels(self, labels_path=None):
        """
        Load label mapping for detected objects
        
        Args:
            labels_path (str, optional): Path to label map. 
                                         Defaults to config path if not provided.
        
        Returns:
            dict: Mapping of class IDs to names
        """
        try:
            path = labels_path or Config.LABEL_MAP_PATH
            self.category_index = {}
            
            with open(path, "r") as f:
                label_data = f.readlines()
            
            current_id = None
            current_name = None
            
            for line in label_data:
                if "id:" in line:
                    current_id = int(line.split(":")[1].strip())
                if "display_name:" in line:
                    current_name = line.split(":")[1].strip().replace("\"", "")
                    if current_id and current_name:
                        self.category_index[current_id] = current_name
            
            self.logger.info(f"Loaded {len(self.category_index)} labels")
            return self.category_index
        except Exception as e:
            self.logger.error(f"Error loading labels: {e}")
            raise

    def detect_objects(self, image_np, confidence_threshold=None):
        """
        Perform object detection on an image
        
        Args:
            image_np (np.array): Input image as numpy array
            confidence_threshold (float, optional): Minimum confidence for detection
        
        Returns:
            tuple: Detection boxes, scores, and classes
        """
        if self.model is None:
            self.load_model()
        
        threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = self.model(input_tensor)
        
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

# Singleton instance for easy import
model_loader = ModelLoader()
