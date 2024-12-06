# # ui/streamlit_app.py
# import streamlit as st
# import numpy as np
# from PIL import Image, ImageDraw
# import cv2

# from models.pretrained_models import model_loader
# from object_detection.video_processor import video_processor
# from analytics.anomaly_detection import anomaly_detector
# from config.settings import Config
# import warnings


# # Suppress warnings
# warnings.filterwarnings("ignore")


# # def draw_detections(image, boxes, scores, classes):
# #     """
# #     Draw detection results on image

# #     Args:
# #         image (PIL.Image): Input image
# #         boxes, scores, classes (numpy arrays): Detection results

# #     Returns:
# #         PIL.Image: Image with detections drawn
# #     """
# #     draw = ImageDraw.Draw(image)

# #     for box, score, cls in zip(boxes, scores, classes):
# #         ymin, xmin, ymax, xmax = box
# #         (left, top, right, bottom) = (
# #             xmin * image.width,
# #             ymin * image.height,
# #             xmax * image.width,
# #             ymax * image.height
# #         )

# #         draw.rectangle(
# #             [left, top, right, bottom],
# #             outline="red",
# #             width=3
# #         )

# #         label = f"{model_loader.category_index.get(cls, 'Unknown')} ({score:.2f})"
# #         draw.text((left, top), label, fill="red")

# #     return image

# def draw_detections(image, boxes, scores, classes):
#     """
#     Draw detection results on image with enhanced visibility

#     Args:
#         image (PIL.Image): Input image
#         boxes, scores, classes (numpy arrays): Detection results

#     Returns:
#         PIL.Image: Image with detections drawn
#     """
#     draw = ImageDraw.Draw(image)
    
#     # Try to import a font for better text rendering
#     try:
#         from PIL import ImageFont
#         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
#     except:
#         font = None

#     # Color map for different objects
#     colors = {
#         'person': "red",
#         'car': "blue",
#         'truck': "green",
#         'bicycle': "yellow",
#         'motorcycle': "purple"
#         # Add more colors for other objects as needed
#     }

#     for box, score, cls in zip(boxes, scores, classes):
#         ymin, xmin, ymax, xmax = box
#         (left, top, right, bottom) = (
#             xmin * image.width,
#             ymin * image.height,
#             xmax * image.width,
#             ymax * image.height
#         )

#         # Get object name and color
#         object_name = model_loader.category_index.get(cls, 'Unknown')
#         color = colors.get(object_name.lower(), "red")

#         # Draw bounding box
#         draw.rectangle(
#             [left, top, right, bottom],
#             outline=color,
#             width=3
#         )

#         # Draw filled rectangle for text background
#         label = f"{object_name} ({score:.1%})"
#         if font:
#             text_bbox = draw.textbbox((left, top-20), label, font=font)
#         else:
#             text_bbox = draw.textbbox((left, top-20), label)
            
#         draw.rectangle(
#             [text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2],
#             fill=color
#         )

#         # Draw text
#         draw.text(
#             (left, top-20),
#             label,
#             fill="white",
#             font=font
#         )

#     return image

# def main():
#     st.title("Advanced Object Detection System")

#     # Sidebar configuration
#     st.sidebar.header("Detection Configuration")
#     input_type = st.sidebar.radio(
#         "Choose Input Type",
#         ["Image", "Video"]
#     )

#     confidence_threshold = st.sidebar.slider(
#         "Confidence Threshold",
#         0.0, 1.0,
#         Config.CONFIDENCE_THRESHOLD,
#         0.01
#     )

#     # Load labels if not already loaded
#     if not model_loader.category_index:
#         model_loader.load_labels()

#     if input_type == "Image":
#         handle_image_detection(confidence_threshold)
#     else:
#         handle_video_detection(confidence_threshold)

# # def handle_image_detection(confidence_threshold):
# #     """
# #     Handle image detection workflow
# #     """
# #     uploaded_image = st.file_uploader(
# #         "Upload an image...",
# #         type=Config.SUPPORTED_IMAGE_TYPES
# #     )

# #     if uploaded_image is not None:
# #         image = Image.open(uploaded_image).convert("RGB")
# #         st.image(image, caption="Uploaded Image", use_container_width=True)

# #         # Convert to numpy for detection
# #         image_np = np.array(image)

# #         # Detect objects
# #         boxes, scores, classes = model_loader.detect_objects(
# #             image_np,
# #             confidence_threshold
# #         )

# #         # Update anomaly detector
# #         anomaly_detector.update_history((boxes, scores, classes))

# #         # Draw detections - Fixed version
# #         detected_image = draw_detections(
# #             image=image.copy(),  # Call copy() method
# #             boxes=boxes,
# #             scores=scores,
# #             classes=classes
# #         )

# #         # Detect anomalies
# #         anomaly_score = anomaly_detector.calculate_anomaly_score()
# #         unusual_patterns = anomaly_detector.detect_unusual_patterns()

# #         # Display anomaly information
# #         if anomaly_score > 0.5:
# #             st.warning(f"Potential Anomalies Detected! Anomaly Score: {anomaly_score:.2f}")
# #             if unusual_patterns:
# #                 st.write("Unusual Patterns:")
# #                 for pattern in unusual_patterns:
# #                     st.info(f"New Objects Appeared: {pattern['objects']}")

# #         # Display detected image
# #         st.image(detected_image, caption="Detected Objects",
# #                  use_container_width=True)
        
# def handle_image_detection(confidence_threshold):
#     """
#     Handle image detection workflow with detailed object information
#     """
#     uploaded_image = st.file_uploader(
#         "Upload an image...",
#         type=Config.SUPPORTED_IMAGE_TYPES
#     )

#     if uploaded_image is not None:
#         image = Image.open(uploaded_image).convert("RGB")
#         st.image(image, caption="Uploaded Image", use_container_width=True)

#         # Show "Detecting objects..." message while processing
#         with st.spinner('Detecting objects...'):
#             # Convert to numpy for detection
#             image_np = np.array(image)

#             # Detect objects
#             boxes, scores, classes = model_loader.detect_objects(
#                 image_np,
#                 confidence_threshold
#             )

#             # Create a summary of detected objects
#             detected_objects = {}
#             for cls, score in zip(classes, scores):
#                 object_name = model_loader.category_index.get(cls, 'Unknown')
#                 if object_name in detected_objects:
#                     detected_objects[object_name]['count'] += 1
#                     detected_objects[object_name]['scores'].append(score)
#                 else:
#                     detected_objects[object_name] = {
#                         'count': 1,
#                         'scores': [score]
#                     }

#             # Display detection summary
#             st.subheader("Detected Objects:")
#             if len(detected_objects) > 0:
#                 for obj_name, data in detected_objects.items():
#                     avg_confidence = sum(data['scores']) / len(data['scores'])
#                     st.write(f"✓ {obj_name}")
#                     st.text(f"   Count: {data['count']}")
#                     st.text(f"   Average Confidence: {avg_confidence:.2%}")
#             else:
#                 st.write("No objects detected in the image.")

#             # Update anomaly detector
#             anomaly_detector.update_history((boxes, scores, classes))

#             # Draw detections
#             detected_image = draw_detections(
#                 image=image.copy(),
#                 boxes=boxes,
#                 scores=scores,
#                 classes=classes
#             )

#             # Detect anomalies
#             anomaly_score = anomaly_detector.calculate_anomaly_score()
#             unusual_patterns = anomaly_detector.detect_unusual_patterns()

#             # Display anomaly information
#             if anomaly_score > 0.5:
#                 st.warning(f"⚠️ Potential Anomalies Detected! Anomaly Score: {anomaly_score:.2f}")
#                 if unusual_patterns:
#                     st.write("Unusual Patterns:")
#                     for pattern in unusual_patterns:
#                         st.info(f"🔍 New Objects Appeared: {pattern['objects']}")

#             # Display detected image
#             st.image(detected_image, caption="Detected Objects", use_container_width=True)

#             # Display additional statistics
#             st.subheader("Detection Statistics:")
#             total_objects = sum(data['count'] for data in detected_objects.values())
#             st.text(f"Total Objects Detected: {total_objects}")
            
#             if total_objects > 0:
#                 avg_confidence_all = sum(
#                     score for data in detected_objects.values() 
#                     for score in data['scores']
#                 ) / total_objects
#                 st.text(f"Overall Average Confidence: {avg_confidence_all:.2%}")

# def handle_video_detection(confidence_threshold):
#     """
#     Handle video detection workflow
#     """
#     uploaded_video = st.file_uploader(
#         "Upload a video...",
#         type=Config.SUPPORTED_VIDEO_TYPES
#     )

#     if uploaded_video is not None:
#         # Save uploaded video
#         video_path = f"temp_video_{uploaded_video.name}"
#         with open(video_path, "wb") as f:
#             f.write(uploaded_video.read())

#         # Display original video
#         st.video(uploaded_video)

#         # Create placeholders for video processing
#         status_placeholder = st.empty()
#         video_placeholder = st.empty()
#         anomaly_placeholder = st.empty()

#         # Process video with detection callback
#         def process_frame(frame, boxes, scores, classes):
#             """
#             Callback for processing each video frame
#             """
#             for box, score, cls in zip(boxes, scores, classes):
#                 if score >= confidence_threshold:
#                     ymin, xmin, ymax, xmax = box
#                     (left, top, right, bottom) = (
#                         int(xmin * frame.shape[1]),
#                         int(ymin * frame.shape[0]),
#                         int(xmax * frame.shape[1]),
#                         int(ymax * frame.shape[0])
#                     )
#                     cv2.rectangle(
#                         frame,
#                         (left, top),
#                         (right, bottom),
#                         (0, 255, 0),
#                         2
#                     )

#                     label = f"{model_loader.category_index.get(cls, 'Unknown')}: {score:.2f}"
#                     cv2.putText(
#                         frame,
#                         label,
#                         (left, top - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (0, 255, 0),
#                         2
#                     )

#             return frame

#         # Track video processing
#         total_frames = 0
#         processed_frames = 0
#         anomaly_frames = 0

#         for frame, (boxes, scores, classes) in video_processor.process_video(
#             video_path,
#             callback=process_frame
#         ):
#             total_frames += 1

#             # Update anomaly detection
#             anomaly_detector.update_history((boxes, scores, classes))

#             # Check for anomalies
#             anomaly_score = anomaly_detector.calculate_anomaly_score()
#             if anomaly_score > 0.5:
#                 anomaly_frames += 1

#             # Update placeholders
#             status_placeholder.text(f"Processing: {processed_frames}/{total_frames} frames")
#             video_placeholder.image(frame, channels="BGR")

#             processed_frames += 1

#             # Optional: Add a small delay for visualization
#             if st.button("Stop Processing"):
#                 break

#         # Final anomaly report
#         anomaly_percentage = (anomaly_frames / total_frames) * \
#             100 if total_frames > 0 else 0
#         anomaly_placeholder.metric("Anomaly Analysis",f"{anomaly_percentage:.2f}% of frames with potential anomalies")


# def setup_logging():
#     """
#     Configure logging for the application
#     """
#     import logging
#     logging.basicConfig(
#         level=Config.LOG_LEVEL,
#         format=Config.LOG_FORMAT
#     )


# if __name__ == "__main__":
#     setup_logging()
#     main()

# ui/streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from models.yolo_detector import YOLODetector
from config.settings import Config

def main():
    st.set_page_config(page_title="Real-time Object Detection", layout="wide")
    st.title("Real-time Object Detection with YOLOv8")
    
    # Initialize detector
    detector = YOLODetector()
    
    # Add source selection
    source = st.sidebar.selectbox("Select Input Source", ["Webcam", "Upload Image"])
    
    # Add confidence threshold slider
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        0.0, 1.0, 
        Config.CONFIDENCE_THRESHOLD, 
        0.05
    )
    detector.model.conf = confidence_threshold
    
    if source == "Webcam":
        st.header("Webcam Live Feed")
        run_webcam(detector)
    else:
        st.header("Image Upload")
        run_image_upload(detector)

def run_webcam(detector):
    """Handle webcam input"""
    # Create placeholder for webcam feed
    frame_placeholder = st.empty()
    stop_button = st.button("Stop")
    
    try:
        cap = cv2.VideoCapture(0)
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
            
            # Process frame
            processed_frame = detector.process_frame(frame)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
    except Exception as e:
        st.error(f"Error accessing webcam: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

def run_image_upload(detector):
    """Handle image upload"""
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=Config.SUPPORTED_IMAGE_TYPES
    )
    
    if uploaded_file is not None:
        try:
            # Convert uploaded file to opencv image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process image
            processed_image = detector.process_image(image)
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Display image
            st.image(rgb_image, channels="RGB", use_column_width=True)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()