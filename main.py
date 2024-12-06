# # # main.py
# # from config.settings import Config
# # import logging
# # import sys
# # import os
# # import warnings

# # # Suppress warnings
# # warnings.filterwarnings("ignore")
# # # Ensure the project root is in Python path
# # project_root = os.path.dirname(os.path.abspath(__file__))
# # sys.path.insert(0, project_root)


# # def initialize_project():
# #     """
# #     Initialize project components and perform startup checks
# #     """
# #     # Configure logging
# #     logging.basicConfig(
# #         level=Config.LOG_LEVEL,
# #         format=Config.LOG_FORMAT,
# #         filename=os.path.join(Config.LOGS_DIR, 'app.log'),
# #         filemode='w'
# #     )
# #     logger = logging.getLogger(__name__)

# #     try:
# #         # Perform initial checks
# #         logger.info("Initializing Object Detection Project")

# #         # Check model availability
# #         from models.pretrained_models import model_loader
# #         model_loader.load_model()
# #         model_loader.load_labels()

# #         logger.info("Project initialization complete")

# #     except Exception as e:
# #         logger.error(f"Initialization failed: {e}")
# #         raise


# # def run_streamlit_app():
# #     """
# #     Launch the Streamlit application
# #     """
# #     from ui.streamlit_app import main
# #     main()


# # def main():
# #     """
# #     Main entry point for the application
# #     """
# #     try:
# #         # Initialize project components
# #         initialize_project()

# #         # Run the Streamlit app
# #         run_streamlit_app()

# #     except Exception as e:
# #         print(f"Error running application: {e}")
# #         sys.exit(1)


# # if __name__ == "__main__":
# #     main()

# # main.py
# import os
# import sys
# from ui.streamlit_app import main
# import logging
# import sentry_sdk

# # Initialize Sentry for error tracking (optional)
# sentry_sdk.init(
#     dsn="your-sentry-dsn",
#     traces_sample_rate=1.0,
# )

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler()
#     ]
# )

# def initialize_project():
#     """Initialize project and verify dependencies"""
#     try:
#         # Ensure the project root is in Python path
#         project_root = os.path.dirname(os.path.abspath(__file__))
#         sys.path.insert(0, project_root)
        
#         # You can add other initialization steps here
        
#     except Exception as e:
#         print(f"Initialization failed: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     initialize_project()
#     main()
# main.py
import os
import sys
from ui.streamlit_app import main
from config.settings import Config

def initialize_project():
    """Initialize project for cloud deployment"""
    try:
        # Ensure the project root is in Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, project_root)
        
        # Initialize configuration
        Config.initialize()
        
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    initialize_project()
    main()