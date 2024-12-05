# # detection/video_processor.py
# import cv2
# import numpy as np
# from models.pretrained_models import model_loader
# from config.settings import Config
# import logging
# import warnings

# # Suppress warnings
# warnings.filterwarnings("ignore")

# class VideoProcessor:
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
#         self.tracker = cv2.MultiTracker_create()

#     def process_video(self, video_path, callback=None):
#         """
#         Process video and perform object detection on each frame
        
#         Args:
#             video_path (str): Path to video file
#             callback (function, optional): Function to process each frame
        
#         Yields:
#             tuple: Processed frame, detection results
#         """
#         try:
#             cap = cv2.VideoCapture(video_path)
            
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Convert to RGB for model input
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
#                 # Detect objects
#                 boxes, scores, classes = model_loader.detect_objects(frame_rgb)
                
#                 # Process frame (optional callback)
#                 if callback:
#                     frame = callback(frame, boxes, scores, classes)
                
#                 yield frame, (boxes, scores, classes)
            
#             cap.release()
        
#         except Exception as e:
#             self.logger.error(f"Video processing error: {e}")
#             raise

#     def track_objects(self, video_path):
#         """
#         Advanced object tracking across video frames
        
#         Args:
#             video_path (str): Path to video file
        
#         Returns:
#             list: Object tracking trajectories
#         """
#         trajectories = []
        
#         for frame, (boxes, scores, classes) in self.process_video(video_path):
#             # Track object movement and trajectories
#             for box in boxes:
#                 trajectory = self._calculate_trajectory(box)
#                 trajectories.append(trajectory)
        
#         return trajectories

#     def _calculate_trajectory(self, box):
#         """
#         Calculate object movement trajectory
        
#         Args:
#             box (np.array): Bounding box coordinates
        
#         Returns:
#             dict: Trajectory information
#         """
#         # Basic trajectory calculation
#         ymin, xmin, ymax, xmax = box
#         center_x = (xmin + xmax) / 2
#         center_y = (ymin + ymax) / 2
        
#         return {
#             'center': (center_x, center_y),
#             'size': (xmax - xmin, ymax - ymin)
#         }

# # Singleton instance
# video_processor = VideoProcessor()

# detection/video_processor.py
import cv2
import numpy as np
from models.pretrained_models import model_loader
from config.settings import Config
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class VideoProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Updated tracker initialization
        self.trackers = []
        
    def create_tracker(self):
        # Create individual trackers using CSRT tracker (more accurate) or KCF tracker (faster)
        return cv2.TrackerCSRT_create()
        # Alternative: return cv2.TrackerKCF_create()

    def process_video(self, video_path, callback=None):
        """
        Process video and perform object detection on each frame
        
        Args:
            video_path (str): Path to video file
            callback (function, optional): Function to process each frame
        
        Yields:
            tuple: Processed frame, detection results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for model input
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect objects
                boxes, scores, classes = model_loader.detect_objects(frame_rgb)
                
                # Update trackers
                if len(self.trackers) == 0 and len(boxes) > 0:
                    # Initialize trackers for new objects
                    for box in boxes:
                        tracker = self.create_tracker()
                        bbox = (int(box[1]), int(box[0]), 
                               int(box[3] - box[1]), int(box[2] - box[0]))  # Convert to (x,y,w,h)
                        tracker.init(frame, bbox)
                        self.trackers.append(tracker)
                
                # Process frame (optional callback)
                if callback:
                    frame = callback(frame, boxes, scores, classes)
                
                yield frame, (boxes, scores, classes)
            
            cap.release()
        
        except Exception as e:
            self.logger.error(f"Video processing error: {e}")
            raise

    def track_objects(self, video_path):
        """
        Advanced object tracking across video frames
        
        Args:
            video_path (str): Path to video file
        
        Returns:
            list: Object tracking trajectories
        """
        trajectories = []
        
        for frame, (boxes, scores, classes) in self.process_video(video_path):
            # Update tracking info
            tracked_boxes = []
            tracked_trackers = []
            
            for tracker in self.trackers:
                success, box = tracker.update(frame)
                if success:
                    tracked_boxes.append(box)
                    tracked_trackers.append(tracker)
            
            # Update tracker list to remove lost trackers
            self.trackers = tracked_trackers
            
            # Calculate trajectories
            for box in tracked_boxes:
                trajectory = self._calculate_trajectory(box)
                trajectories.append(trajectory)
        
        return trajectories

    def _calculate_trajectory(self, box):
        """
        Calculate object movement trajectory
        
        Args:
            box (tuple): Bounding box coordinates (x, y, w, h)
        
        Returns:
            dict: Trajectory information
        """
        x, y, w, h = box
        center_x = x + w/2
        center_y = y + h/2
        
        return {
            'center': (center_x, center_y),
            'size': (w, h)
        }

# Singleton instance
video_processor = VideoProcessor()