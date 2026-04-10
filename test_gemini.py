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
    config = load_config()
    
    # Target directory for the specific patient/file you just processed
    # CHANGE THIS to the folder you want to test (e.g., "explore/output/K_Drewry")
    target_dir = Path("explore/output/N_Rivera")
    
    if not target_dir.exists():
        print(f"ERROR: Cannot find directory {target_dir}")
        return

    # Find all payload files in that directory
    payload_files = sorted(target_dir.glob("phi_safe_payload_page_*.jpg"))
    
    if not payload_files:
        print(f"ERROR: No phi_safe_payload_page_*.jpg files found in {target_dir}")
        return

    print(f"Found {len(payload_files)} payload(s) to process.")
    
    # Initialize the extractor once
    extractor = CloudVlmExtractor(config)
    
    all_results = {}

    # Loop through every page payload found
    for payload_path in payload_files:
        print(f"\n--- Processing {payload_path.name} ---")
        
        img = cv2.imread(str(payload_path))
        if img is None:
            print(f"ERROR: Failed to load image at {payload_path}")
            continue

        print(f"Sending to Gemini API ({config.cloud_vlm.model})...")
        result = extractor.extract_table_crop(img)
        
        # Store results keyed by the filename
        all_results[payload_path.name] = result
    
    print("\n========================================")
    print("      FINAL MULTI-PAGE EXTRACTION         ")
    print("========================================")
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()