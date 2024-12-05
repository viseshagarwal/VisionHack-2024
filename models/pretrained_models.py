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
    # COCO_LABELS = {
    #      1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    #     6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    #     11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    #     16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    #     21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    #     26: 'backpack', 27: 'umbrella', 28: 'handbag', 29: 'tie', 30: 'suitcase',
    #     31: 'frisbee', 32: 'skis', 33: 'snowboard', 34: 'sports ball', 35: 'kite',
    #     36: 'baseball bat', 37: 'baseball glove', 38: 'skateboard', 39: 'surfboard', 40: 'tennis racket',
    #     41: 'bottle', 42: 'wine glass', 43: 'cup', 44: 'fork', 45: 'knife',
    #     46: 'spoon', 47: 'bowl', 48: 'banana', 49: 'apple', 50: 'sandwich',
    #     51: 'orange', 52: 'broccoli', 53: 'carrot', 54: 'hot dog', 55: 'pizza',
    #     56: 'donut', 57: 'cake', 58: 'chair', 59: 'couch', 60: 'potted plant',
    #     61: 'bed', 62: 'dining table', 63: 'toilet', 64: 'tv', 65: 'laptop',
    #     66: 'mouse', 67: 'remote', 68: 'keyboard', 69: 'cell phone', 70: 'microwave',
    #     71: 'oven', 72: 'toaster', 73: 'sink', 74: 'refrigerator', 75: 'book',
    #     76: 'clock', 77: 'vase', 78: 'scissors', 79: 'teddy bear', 80: 'hair drier',
    #     81: 'toothbrush', 82: 'hair brush', 83: 'banner', 84: 'blanket', 85: 'branch',
    #     86: 'bridge', 87: 'building-other', 88: 'bush', 89: 'cabinet', 90: 'cage',
    #     91: 'cardboard', 92: 'carpet', 93: 'ceiling-other', 94: 'ceiling-tile', 95: 'clothes',
    #     96: 'clouds', 97: 'counter', 98: 'cupboard', 99: 'curtain', 100: 'desk-stuff',
    #     101: 'dirt', 102: 'door-stuff', 103: 'fence', 104: 'floor-marble', 105: 'floor-other',
    #     106: 'floor-stone', 107: 'floor-tile', 108: 'floor-wood', 109: 'flower', 110: 'fog',
    #     111: 'food-other', 112: 'fruit', 113: 'furniture-other', 114: 'grass', 115: 'gravel',
    #     116: 'ground-other', 117: 'hill', 118: 'house', 119: 'leaves', 120: 'light',
    #     121: 'mat', 122: 'metal', 123: 'mirror-stuff', 124: 'moss', 125: 'mountain',
    #     126: 'mud', 127: 'napkin', 128: 'net', 129: 'paper', 130: 'pavement',
    #     131: 'pillow', 132: 'plant-other', 133: 'plastic', 134: 'platform', 135: 'playingfield',
    #     136: 'railing', 137: 'railroad', 138: 'river', 139: 'road', 140: 'rock',
    #     141: 'roof', 142: 'rug', 143: 'salad', 144: 'sand', 145: 'sea',
    #     146: 'shelf', 147: 'sky-other', 148: 'skyscraper', 149: 'snow', 150: 'solid-other',
    #     151: 'stairs', 152: 'stone', 153: 'straw', 154: 'structural-other', 155: 'table',
    #     156: 'tent', 157: 'textile-other', 158: 'towel', 159: 'tree', 160: 'vegetable',
    #     161: 'wall-brick', 162: 'wall-concrete', 163: 'wall-other', 164: 'wall-panel', 165: 'wall-stone',
    #     166: 'wall-tile', 167: 'wall-wood', 168: 'water-other', 169: 'waterdrops', 170: 'window-blind',
    #     171: 'window-other', 172: 'wood', 173: 'artwork', 174: 'badge', 175: 'ball',
    #     176: 'basket', 177: 'battery', 178: 'bin', 179: 'blackboard', 180: 'box',
    #     181: 'bracelet', 182: 'broom', 183: 'bucket', 184: 'calculator', 185: 'calendar',
    #     186: 'candle', 187: 'card', 188: 'cart', 189: 'charger', 190: 'clipboard',
    #     191: 'coin', 192: 'computer', 193: 'controller', 194: 'cooker', 195: 'cosmetics',
    #     196: 'decoration', 197: 'diary', 198: 'dish', 199: 'door handle', 200: 'earring',
    #     201: 'eraser', 202: 'fan', 203: 'faucet', 204: 'file', 205: 'fire extinguisher',
    #     206: 'flag', 207: 'flashlight', 208: 'folder', 209: 'food container', 210: 'footwear',
    #     211: 'gadget', 212: 'garbage', 213: 'gift', 214: 'glass stuff', 215: 'glasses',
    #     216: 'glove', 217: 'gown', 218: 'hammer', 219: 'hanger', 220: 'hat',
    #     221: 'headphone', 222: 'helmet', 223: 'ink', 224: 'jacket', 225: 'jar',
    #     226: 'jewellery', 227: 'key', 228: 'ladder', 229: 'lamp', 230: 'lantern',
    #     231: 'lock', 232: 'luggage', 233: 'machine', 234: 'magazine', 235: 'magnet',
    #     236: 'mail', 237: 'mannequin', 238: 'map', 239: 'mask', 240: 'mat',
    #     241: 'measure', 242: 'medicine', 243: 'menu', 244: 'mirror', 245: 'money',
    #     246: 'monitor', 247: 'mop', 248: 'mug', 249: 'nail', 250: 'necklace',
    #     251: 'newspaper', 252: 'notebook', 253: 'ornament', 254: 'outlet', 255: 'paint',
    #     256: 'painting', 257: 'pan', 258: 'paper towel', 259: 'pen', 260: 'pencil',
    #     261: 'phone', 262: 'photo', 263: 'picture frame', 264: 'pillar', 265: 'pipe',
    #     266: 'plant pot', 267: 'plate', 268: 'platter', 269: 'pocket', 270: 'pole',
    #     271: 'poster', 272: 'pot', 273: 'printer', 274: 'projector', 275: 'rack',
    #     276: 'radiator', 277: 'radio', 278: 'razor', 279: 'receipt', 280: 'ring',
    #     281: 'robot', 282: 'rocket', 283: 'roller', 284: 'rope', 285: 'rubber',
    #     286: 'ruler', 287: 'salt', 288: 'scale', 289: 'scanner', 290: 'scissors',
    #     291: 'screw', 292: 'screwdriver', 293: 'sculpture', 294: 'server', 295: 'shade',
    #     296: 'shampoo', 297: 'shelf', 298: 'shield', 299: 'shoe', 300: 'shopping basket',
    #     301: 'shopping cart', 302: 'shovel', 303: 'sign', 304: 'sink', 305: 'soap',
    #     306: 'soap dispenser', 307: 'socket', 308: 'spatula', 309: 'speaker', 310: 'spice',
    #     311: 'sponge', 312: 'spray', 313: 'stamp', 314: 'stapler', 315: 'statue',
    #     316: 'steam', 317: 'sticker', 318: 'stone', 319: 'stool', 320: 'stove',
    #     321: 'straw', 322: 'string', 323: 'sunglasses', 324: 'sweater', 325: 'switch',
    #     326: 'table cloth', 327: 'tablet', 328: 'tag', 329: 'tape', 330: 'telephone',
    #     331: 'thermometer', 332: 'tissue', 333: 'toiletries', 334: 'tool', 335: 'toothpaste',
    #     336: 'torch', 337: 'toy', 338: 'tray', 339: 'tripod', 340: 'trophy',
    #     341: 'tube', 342: 'utensil', 343: 'vacuum', 344: 'valve', 345: 'vending machine',
    #     346: 'vent', 347: 'vest', 348: 'video camera', 349: 'wallet', 350: 'wardrobe',
    #     351: 'watch', 352: 'water bottle', 353: 'watering can', 354: 'weight', 355: 'wheel',
    #     356: 'whiteboard', 357: 'window frame', 358: 'wine bottle', 359: 'wire', 360: 'wrench',
    #     361: 'wristband', 362: 'zipper', 363: 'baby', 364: 'baker', 365: 'banker',
    #     366: 'barber', 367: 'bartender', 368: 'chef', 369: 'clerk', 370: 'construction worker',
    #     371: 'cook', 372: 'dancer', 373: 'dentist', 374: 'doctor', 375: 'doorman',
    #     376: 'driver', 377: 'engineer', 378: 'farmer', 379: 'firefighter', 380: 'fisherman',
    #     381: 'gardener', 382: 'hairdresser', 383: 'janitor', 384: 'judge', 385: 'lawyer',
    #     386: 'librarian', 387: 'lifeguard', 388: 'mechanic', 389: 'nurse', 390: 'painter',
    #     391: 'pilot', 392: 'plumber', 393: 'police', 394: 'postal worker', 395: 'receptionist',
    #     396: 'sailor', 397: 'salesperson', 398: 'scientist', 399: 'security guard', 400: 'soldier',
    #     401: 'student', 402: 'teacher', 403: 'veterinarian', 404: 'waiter', 405: 'worker',
    #     406: 'writer', 407: 'accountant', 408: 'architect', 409: 'artist', 410: 'athlete',
    #     411: 'busker', 412: 'carpenter', 413: 'cashier', 414: 'cleaner', 415: 'coach'
    # }
    COCO_LABELS = {
        1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    26: 'backpack', 27: 'umbrella', 28: 'handbag', 29: 'tie', 30: 'suitcase',
    31: 'frisbee', 32: 'skis', 33: 'snowboard', 34: 'sports ball', 35: 'kite',
    36: 'baseball bat', 37: 'baseball glove', 38: 'skateboard', 39: 'surfboard', 40: 'tennis racket',
    41: 'bottle', 42: 'wine glass', 43: 'cup', 44: 'fork', 45: 'knife',
    46: 'spoon', 47: 'bowl', 48: 'banana', 49: 'apple', 50: 'sandwich',
    51: 'orange', 52: 'broccoli', 53: 'carrot', 54: 'hot dog', 55: 'pizza',
    56: 'donut', 57: 'cake', 58: 'chair', 59: 'couch', 60: 'potted plant',
    61: 'bed', 62: 'dining table', 63: 'toilet', 64: 'tv', 65: 'laptop',
    66: 'mouse', 67: 'remote', 68: 'keyboard', 69: 'cell phone', 70: 'microwave',
    71: 'oven', 72: 'toaster', 73: 'sink', 74: 'refrigerator', 75: 'book',
    76: 'clock', 77: 'vase', 78: 'scissors', 79: 'teddy bear', 80: 'hair drier',
    81: 'toothbrush', 82: 'hair brush', 83: 'banner', 84: 'blanket', 85: 'branch',
    86: 'bridge', 87: 'building-other', 88: 'bush', 89: 'cabinet', 90: 'cage',
    91: 'cardboard', 92: 'carpet', 93: 'ceiling-other', 94: 'ceiling-tile', 95: 'cloth',
    96: 'clothes', 97: 'clouds', 98: 'counter', 99: 'cupboard', 100: 'curtain',
    # ... continuing with more specific objects and scenes
    101: 'desk-stuff', 102: 'dirt', 103: 'door-stuff', 104: 'fence', 105: 'floor-marble',
    106: 'floor-other', 107: 'floor-stone', 108: 'floor-tile', 109: 'floor-wood', 110: 'flower',
    111: 'fog', 112: 'food-other', 113: 'fruit', 114: 'furniture-other', 115: 'grass',
    116: 'gravel', 117: 'ground-other', 118: 'hill', 119: 'house', 120: 'leaves',
    121: 'light', 122: 'mat', 123: 'metal', 124: 'mirror-stuff', 125: 'moss',
    126: 'mountain', 127: 'mud', 128: 'napkin', 129: 'net', 130: 'paper',
    131: 'pavement', 132: 'pillow', 133: 'plant-other', 134: 'plastic', 135: 'platform',
    136: 'playground', 137: 'pond', 138: 'railing', 139: 'railroad', 140: 'river',
    141: 'road', 142: 'rock', 143: 'roof', 144: 'rug', 145: 'salad',
    146: 'sand', 147: 'sea', 148: 'shelf', 149: 'sky-other', 150: 'skyscraper',
    # ... adding more indoor and outdoor objects
    151: 'snow', 152: 'solid-other', 153: 'stairs', 154: 'stone', 155: 'straw',
    156: 'structural-other', 157: 'table', 158: 'tent', 159: 'textile-other', 160: 'towel',
    161: 'tree', 162: 'vegetable', 163: 'wall-brick', 164: 'wall-concrete', 165: 'wall-other',
    166: 'wall-panel', 167: 'wall-stone', 168: 'wall-tile', 169: 'wall-wood', 170: 'water-other',
    171: 'waterdrops', 172: 'window-blind', 173: 'window-other', 174: 'wood', 175: 'artwork',
    176: 'badge', 177: 'ball', 178: 'basket', 179: 'battery', 180: 'binoculars',
    # ... adding electronic devices and gadgets
    181: 'blackboard', 182: 'calculator', 183: 'calendar', 184: 'camera', 185: 'can',
    186: 'cd', 187: 'charger', 188: 'chart', 189: 'compass', 190: 'controller',
    191: 'cookware', 192: 'cosmetics', 193: 'crate', 194: 'crown', 195: 'dish',
    196: 'doorknob', 197: 'drill', 198: 'drum', 199: 'dumbbell', 200: 'earphone',
    # ... adding more household items
    201: 'fan', 202: 'faucet', 203: 'figurine', 204: 'first-aid', 205: 'flashlight',
    206: 'floor-lamp', 207: 'folder', 208: 'food-processor', 209: 'frame', 210: 'garbage',
    211: 'glasses', 212: 'glove', 213: 'goggle', 214: 'hammer', 215: 'hanger',
    216: 'hard-disk', 217: 'headphone', 218: 'helmet', 219: 'inkpad', 220: 'iron',
    # ... adding clothing and accessories
    221: 'jacket', 222: 'jar', 223: 'jersey', 224: 'jewelry', 225: 'key',
    226: 'ladder', 227: 'lamp', 228: 'lantern', 229: 'lighter', 230: 'lock',
    231: 'lunchbox', 232: 'magazine', 233: 'magnet', 234: 'mailbox', 235: 'mannequin',
    236: 'map', 237: 'mask', 238: 'matchbox', 239: 'medal', 240: 'medicine',
    # ... adding more categories
    241: 'microphone', 242: 'mixer', 243: 'mop', 244: 'mug', 245: 'necklace',
    246: 'notebook', 247: 'oven-mitt', 248: 'paint', 249: 'painting', 250: 'pan',
    251: 'paper-towel', 252: 'pen', 253: 'pencil', 254: 'phone', 255: 'photo',
    256: 'piano', 257: 'picture', 258: 'pillar', 259: 'pipe', 260: 'planter',
    # ... and so on up to 500
    # Adding more specific categories
    261: 'plate', 262: 'pliers', 263: 'pocket', 264: 'pole', 265: 'poster',
    266: 'pot', 267: 'printer', 268: 'projector', 269: 'rack', 270: 'radio',
    271: 'rake', 272: 'receipt', 273: 'ring', 274: 'robot', 275: 'roller',
    276: 'rope', 277: 'ruler', 278: 'safe', 279: 'scale', 280: 'scanner',
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

    # def detect_objects(self, image_np, confidence_threshold=None):
    #     """
    #     Perform object detection on an image.

    #     Args:
    #         image_np (np.ndarray): Input image as a numpy array
    #         confidence_threshold (float, optional): Minimum confidence for detection

    #     Returns:
    #         tuple: Filtered detection boxes, scores, and classes
    #     """
    #     # Use default confidence threshold if not provided
    #     threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD

    #     # Ensure model is loaded
    #     if self.model is None:
    #         self.load_model()

    #     # Prepare input tensor
    #     input_tensor = tf.convert_to_tensor(image_np)
    #     input_tensor = input_tensor[tf.newaxis, ...]

    #     # Perform detection
    #     detections = self.model(input_tensor)

    #     # Process detections
    #     num_detections = int(detections.pop("num_detections"))
    #     detections = {key: value[0, :num_detections].numpy()
    #                   for key, value in detections.items()}

    #     detection_boxes = detections["detection_boxes"]
    #     detection_scores = detections["detection_scores"]
    #     detection_classes = detections["detection_classes"].astype(np.int64)

    #     # Filter detections by confidence
    #     mask = detection_scores >= threshold
        #     return (
        #         detection_boxes[mask],
        #         detection_scores[mask],
        #         detection_classes[mask]
        #     )

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

        # Get the concrete function for detection
        detect_fn = self.model.signatures['serving_default']
        
        # Perform detection using the concrete function
        detections = detect_fn(input_tensor)

        # Process detections
        num_detections = int(detections.pop('num_detections').numpy())
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}

        detection_boxes = detections['detection_boxes']
        detection_scores = detections['detection_scores']
        detection_classes = detections['detection_classes'].astype(np.int64)

        # Filter detections by confidence
        mask = detection_scores >= threshold
        return (
            detection_boxes[mask],
            detection_scores[mask],
            detection_classes[mask]
        )

# Create a singleton instance for easy import
model_loader = ModelLoader()
