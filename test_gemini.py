import os
import cv2
import json
import logging
import sys
from pathlib import Path

# Add root to sys.path to resolve src imports cleanly
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)

# Import your actual config loader and the extractor
from src.config import load_config
from src.vlm_cloud import CloudVlmExtractor

def main():
    # Load the real configuration from your config.yaml
    config = load_config()

    # The vlm_cloud.py file will automatically check if GOOGLE_API_KEY
    # is available via your .env file or environment variables based
    # on the 'api_key_env' setting in config.yaml.
    
    # Update this path to one of the actual folders you just generated!
    # e.g., "explore/output/patient_1/phi_safe_payload.jpg"
    payload_path = Path("/Users/mohit/Documents/GitHub/timesheet-ocr/explore/output/C.Ferguson Timesheets - 010726-011326/phi_safe_payload.jpg")
    
    if not payload_path.exists():
        print(f"ERROR: Cannot find {payload_path}")
        print("Please update the payload_path variable to point to a real image.")
        return

    print(f"Loading payload: {payload_path}")
    img = cv2.imread(str(payload_path))
    
    if img is None:
        print(f"ERROR: Failed to load image at {payload_path}")
        return

    # Initialize the extractor with your actual loaded config
    extractor = CloudVlmExtractor(config)
    
    print(f"Sending payload to Gemini API ({config.cloud_vlm.model})...")
    result = extractor.extract_table_crop(img)
    
    print("\n--- GEMINI EXTRACTION RESULT ---")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()