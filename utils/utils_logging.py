# utils/data_logging.py
import os
import json
import logging
from datetime import datetime
from config.settings import Config
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")


class DataLogger:
    def __init__(self, log_dir=None):
        self.logger = logging.getLogger(__name__)
        self.log_dir = log_dir or Config.LOGS_DIR
        os.makedirs(self.log_dir, exist_ok=True)

    def log_detection_results(self, image_path, results):
        """
        Log object detection results to a JSON file

        Args:
            image_path (str): Path to the processed image
            results (dict): Detection results
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"detection_log_{timestamp}.json"
            log_path = os.path.join(self.log_dir, log_filename)

            log_entry = {
                'timestamp': timestamp,
                'image_path': image_path,
                'detection_results': {
                    'total_objects': len(results.get('classes', [])),
                    'objects': [
                        {
                            'class': model_loader.category_index.get(cls, 'Unknown'),
                            'confidence': float(score)
                        }
                        for cls, score in zip(results.get('classes', []), results.get('scores', []))
                    ]
                }
            }

            with open(log_path, 'w') as f:
                json.dump(log_entry, f, indent=4)

            self.logger.info(f"Logged detection results to {log_path}")
        except Exception as e:
            self.logger.error(f"Error logging detection results: {e}")

    def get_recent_logs(self, limit=10):
        """
        Retrieve recent detection logs

        Args:
            limit (int): Number of recent logs to retrieve

        Returns:
            list: Recent log entries
        """
        try:
            log_files = sorted(
                [f for f in os.listdir(self.log_dir)
                 if f.startswith('detection_log_')],
                reverse=True
            )[:limit]

            recent_logs = []
            for log_file in log_files:
                with open(os.path.join(self.log_dir, log_file), 'r') as f:
                    recent_logs.append(json.load(f))

            return recent_logs
        except Exception as e:
            self.logger.error(f"Error retrieving logs: {e}")
            return []


# Singleton instance
data_logger = DataLogger()
