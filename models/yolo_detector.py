# # models/yolo_detector.py
# import torch
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import pyttsx3
# import time
# import os
# from config.settings import Config

# # Enable cuDNN auto-tuner for better performance
# torch.backends.cudnn.benchmark = True
# torch.cuda.set_device(0)


# class YOLODetector:
#     def __init__(self):
#         try:
#             # Check if model file exists
#             if not os.path.exists(Config.MODEL_PATH):
#                 raise FileNotFoundError(
#                     f"Model file not found at {Config.MODEL_PATH}")

#             # Check GPU availability
#             self.device = torch.device(
#                 'cuda:0' if torch.cuda.is_available() else 'cpu')
#             print(
#                 f"\n ------------------------------------ \nUsing device: {self.device}")

#             # Load model and move to GPU
#             self.model = YOLO(Config.MODEL_PATH)
#             self.model.to("cuda")

#             # Initialize text-to-speech
#             # try:
#             #     self.engine = pyttsx3.init()
#             #     self.engine.setProperty('rate', 150)
#             #     self.voice_enabled = True
#             # except Exception as e:
#             #     print(f"Warning: Text-to-speech initialization failed: {e}")
#             #     self.voice_enabled = False

#             # # Track last spoken time for each class
#             # self.last_spoken = {}
#             # self.speak_cooldown = Config.SPEAK_COOLDOWN

#         except Exception as e:
#             raise Exception(f"Failed to initialize YOLODetector: {str(e)}")

#     def process_frame(self, frame):
#         """Process a single frame"""
#         try:
#             # Move input to GPU if available
#             if torch.cuda.is_available():
#                 results = self.model(frame, device=self.device)
#             else:
#                 results = self.model(frame)
#             return self.draw_boxes(frame, results)
#         except Exception as e:
#             print(f"Warning: Failed to process frame: {e}")
#             return frame

#     def process_image(self, image):
#         """Process an uploaded image"""
#         try:
#             # Move input to GPU if available
#             if torch.cuda.is_available():
#                 results = self.model(image, device=self.device)
#             else:
#                 results = self.model(image)
#             return self.draw_boxes(image, results)
#         except Exception as e:
#             print(f"Warning: Failed to process image: {e}")
#             return image

#     # def speak_detection(self, class_name, confidence):
#     #     """Speak detected object with cooldown"""
#     #     if not self.voice_enabled:
#     #         return

#     #     current_time = time.time()
#     #     if (class_name not in self.last_spoken or
#     #             current_time - self.last_spoken[class_name] > self.speak_cooldown):
#     #         try:
#     #             text = f"Detected {class_name} with {
#     #                 confidence:.0%} confidence"
#     #             self.engine.say(text)
#     #             self.engine.runAndWait()
#     #             self.last_spoken[class_name] = current_time
#     #         except Exception as e:
#     #             print(f"Warning: Failed to speak detection: {e}")

#     def draw_boxes(self, frame, results):
#         """Draw detection boxes and labels on frame"""
#         annotated_frame = frame.copy()

#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 # Get box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#                 # Get class details
#                 conf = float(box.conf)
#                 cls = int(box.cls)
#                 class_name = result.names[cls]

#                 # Draw box
#                 cv2.rectangle(annotated_frame, (x1, y1),
#                               (x2, y2), (0, 255, 0), 2)

#                 # Draw label with confidence
#                 label = f'{class_name} {conf:.2f}'
#                 t_size = cv2.getTextSize(
#                     label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                 cv2.rectangle(annotated_frame, (x1, y1), c2, (0, 255, 0), -1)
#                 cv2.putText(annotated_frame, label, (x1, y1-2),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

#                 # # Speak detection if confidence is high
#                 # if conf > Config.CONFIDENCE_THRESHOLD:
#                 #     self.speak_detection(class_name, conf)

#         return annotated_frame

#     # def process_frame(self, frame):
#     #     """Process a single frame"""
#     #     try:
#     #         results = self.model(frame)
#     #         return self.draw_boxes(frame, results)
#     #     except Exception as e:
#     #         print(f"Warning: Failed to process frame: {e}")
#     #         return frame

#     # def process_image(self, image):
#     #     """Process an uploaded image"""
#     #     try:
#     #         results = self.model(image)
#     #         return self.draw_boxes(image, results)
#     #     except Exception as e:
#     #         print(f"Warning: Failed to process image: {e}")
#     #         return image

# models/yolo_detector.py
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import pyttsx3
import time
import os
from config.settings import Config
from collections import defaultdict

# Enable cuDNN auto-tuner for better performance
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)


class YOLODetector:
    def __init__(self):
        try:
            # Use larger YOLOv8x model
            self.model_name = "yolov8x.pt"
            self.model_path = os.path.join(
                os.path.dirname(Config.MODEL_PATH), self.model_name)

            # Download larger model if not exists
            if not os.path.exists(self.model_path):
                print(f"Downloading {self.model_name}...")
                torch.hub.download_url_to_file(f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{self.model_name}",
                                               self.model_path
                                               )

            # Initialize model
            self.model = YOLO(self.model_path)
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            # Initialize detection tracking
            self.detected_objects = defaultdict(int)
            self.detection_threshold = 0.5
            self.min_detections = 3  # Minimum detections to consider object present

            print(f"\nUsing YOLOv8x on: {self.device}")

        except Exception as e:
            raise Exception(f"Failed to initialize YOLODetector: {str(e)}")

    def update_detections(self, results):
        """Update detected objects list"""
        for result in results:
            for box in result.boxes:
                if float(box.conf) > self.detection_threshold:
                    cls = int(box.cls)
                    class_name = result.names[cls]
                    self.detected_objects[class_name] += 1

    def get_detected_items(self):
        """Get list of consistently detected items"""
        return [item for item, count in self.detected_objects.items()
                if count >= self.min_detections]

    def clear_detections(self):
        """Clear detection history"""
        self.detected_objects.clear()

    def process_frame(self, frame):
        """Process a single frame"""
        try:
            results = self.model(frame, device=self.device)
            self.update_detections(results)
            return self.draw_boxes(frame, results)
        except Exception as e:
            print(f"Warning: Failed to process frame: {e}")
            return frame

    def draw_boxes(self, frame, results):
        """Draw detection boxes and labels on frame"""
        annotated_frame = frame.copy()

        # Add detected items list
        detected_items = self.get_detected_items()
        if detected_items:
            y_pos = 30
            cv2.putText(annotated_frame, "Detected Items:", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i, item in enumerate(detected_items, 1):
                y_pos += 30
                cv2.putText(annotated_frame, f"{i}. {item}", (30, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw detection boxes
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                class_name = result.names[cls]

                if conf > self.detection_threshold:
                    cv2.rectangle(annotated_frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return annotated_frame

    def process_image(self, image):
        """Process an uploaded image"""
        try:
            results = self.model(image, device=self.device)
            self.update_detections(results)
            return self.draw_boxes(image, results)
        except Exception as e:
            print(f"Warning: Failed to process image: {e}")
            return image
