# analytics/anomaly_detection.py
import numpy as np
from config.settings import Config
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class AnomalyDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.detection_history = []

    def update_history(self, detection_results):
        """
        Update detection history for anomaly analysis
        
        Args:
            detection_results (tuple): Boxes, scores, classes
        """
        boxes, scores, classes = detection_results
        
        # Keep only recent detections
        self.detection_history.append({
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'timestamp': len(self.detection_history)
        })
        
        # Limit history size
        if len(self.detection_history) > Config.MAX_TRACKING_HISTORY:
            self.detection_history.pop(0)

    def calculate_anomaly_score(self):
        """
        Calculate overall anomaly score based on detection history
        
        Returns:
            float: Anomaly score
        """
        if len(self.detection_history) < 2:
            return 0.0
        
        try:
            # Calculate variations in detection
            score_variations = [
                np.std(hist['scores']) for hist in self.detection_history
            ]
            
            # Calculate frequency of unexpected objects
            unexpected_objects = sum(
                len(np.where(hist['scores'] < 0.3)[0]) 
                for hist in self.detection_history
            )
            
            # Composite anomaly score
            anomaly_score = (
                np.mean(score_variations) + 
                unexpected_objects * 0.1
            )
            
            self.logger.info(f"Calculated Anomaly Score: {anomaly_score}")
            return anomaly_score
        
        except Exception as e:
            self.logger.error(f"Anomaly score calculation error: {e}")
            return 0.0

    def detect_unusual_patterns(self):
        """
        Identify unusual object detection patterns
        
        Returns:
            list: Detected anomalies
        """
        anomalies = []
        
        for i in range(1, len(self.detection_history)):
            prev = self.detection_history[i-1]
            curr = self.detection_history[i]
            
            # Check for sudden appearance of new objects
            new_objects = set(curr['classes']) - set(prev['classes'])
            
            if new_objects:
                anomalies.append({
                    'type': 'new_object_appearance',
                    'objects': list(new_objects),
                    'timestamp': curr['timestamp']
                })
        
        return anomalies

# Singleton instance
anomaly_detector = AnomalyDetector()
