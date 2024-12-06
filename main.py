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
