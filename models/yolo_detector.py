import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
from config.settings import Config
from collections import defaultdict

torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)

class YOLODetector:
    def __init__(self):
        try:
            # Download model if it doesn't exist
            if not os.path.exists(Config.MODEL_PATH):
                print(f"Downloading YOLOv8x model...")
                torch.hub.download_url_to_file(
                    Config.MODEL_URL,
                    Config.MODEL_PATH
                )

            self.model = YOLO(Config.MODEL_PATH)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            self.detected_objects = defaultdict(int)
            self.detection_threshold = 0.5
            self.min_detections = 3

            print(f"\nUsing YOLOv8x on: {self.device}")

        except Exception as e:
            raise Exception(f"Failed to initialize YOLODetector: {str(e)}")

    def update_detections(self, results):
        for result in results:
            for box in result.boxes:
                if float(box.conf) > self.detection_threshold:
                    cls = int(box.cls)
                    class_name = result.names[cls]
                    self.detected_objects[class_name] += 1

    def get_detected_items(self):
        return [item for item, count in self.detected_objects.items()
                if count >= self.min_detections]

    def clear_detections(self):
        self.detected_objects.clear()

    def process_frame(self, frame):
        try:
            results = self.model(frame, device=self.device)
            self.update_detections(results)
            return self.draw_boxes(frame, results)
        except Exception as e:
            print(f"Warning: Failed to process frame: {e}")
            return frame

    def draw_boxes(self, frame, results):
        annotated_frame = frame.copy()

        detected_items = self.get_detected_items()
        if detected_items:
            y_pos = 30
            cv2.putText(annotated_frame, "Detected Items:", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for i, item in enumerate(detected_items, 1):
                y_pos += 30
                cv2.putText(annotated_frame, f"{i}. {item}", (30, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
        try:
            results = self.model(image, device=self.device)
            self.update_detections(results)
            return self.draw_boxes(image, results)
        except Exception as e:
            print(f"Warning: Failed to process image: {e}")
            return image
