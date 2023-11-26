import sys
import os

# Get the path to the project's root directory (urban_studio)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Check if the base folder path is not already in sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
    