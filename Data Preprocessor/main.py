"""
Main Execution Script for IMDb Data Preprocessing.

This script is designed to be run in a Google Colab environment.
It orchestrates the entire preprocessing workflow by:
1. Setting up the necessary project paths.
2. Importing and running the refactored PreprocessingPipeline.
"""

import os
import sys
from google.colab import drive

# --- 1. Mount Google Drive (if needed) ---
# This step is often done once per Colab session.
# If your drive is already mounted, you can comment this block out.
# print("Mounting Google Drive...")
# drive.mount('/content/drive')

# --- 2. Set Up Project Paths ---
# This is the root directory for your project inside Google Drive
# IMPORTANT: This path must match the folder structure you created.
PROJECT_ROOT = '/content/drive/My Drive/CSE 6363 Project/Data Preprocessor'

# Add the project root to Python's path to allow for module imports
# This is what allows us to use `from preprocess...` and `from resources...`
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Define key directories
# The raw data is in a parallel folder to the 'Data Preprocessor' folder
RAW_DATA_PATH = '/content/drive/My Drive/CSE 6363 Project/aclImdb'
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "processed_data")

# Create the output directory if it doesn't exist
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

print("-" * 50)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Raw Data Path: {RAW_DATA_PATH}")
print(f"Processed Data Path: {PROCESSED_DATA_PATH}")
print("-" * 50)


# --- 3. Run the Preprocessing Pipeline ---
# Now we can import our custom pipeline module
try:
    # Use absolute imports now that the project root is in sys.path
    from preprocess.pipeline import PreprocessingPipeline

    # Check if the raw data path exists before running
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"The specified raw data path does not exist: {RAW_DATA_PATH}\n"
            "Please ensure the 'aclImdb' folder is correctly placed."
        )

    # Instantiate the pipeline with our defined paths
    pipeline = PreprocessingPipeline(
        input_path=RAW_DATA_PATH,
        output_path=PROCESSED_DATA_PATH
    )

    # Execute the pipeline
    pipeline.run()

except ImportError as e:
    print("\n--- ERROR ---")
    print("Could not import the PreprocessingPipeline.")
    print(f"Error details: {e}")
    print("\nPlease ensure the following structure exists in your Drive:")
    print(f"{PROJECT_ROOT}/")
    print("├── preprocess/")
    print("│   ├── __init__.py")
    print("│   ├── data_loader.py")
    print("│   ├── partitioner.py")
    print("│   ├── pipeline.py")
    print("│   └── text_cleaner.py")
    print("├── resources/")
    print("│   ├── __init__.py")
    print("│   └── contractions.py")
    print("└── main.py")
    print("-" * 50)

except Exception as e:
    print(f"\nAn unexpected error occurred during pipeline execution: {e}")

