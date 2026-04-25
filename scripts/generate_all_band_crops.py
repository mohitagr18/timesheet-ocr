import os
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
import logging

from src.band_crop_extractor import BandCropExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    input_dir = Path("input")
    output_dir = Path("output/band_crop_debug_all")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    extractor = BandCropExtractor(config)
    
    # Get all pdf files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf") and not f.startswith(".")]
    logger.info(f"Found {len(pdf_files)} PDF files in input/")
    
    for filename in sorted(pdf_files):
        file_path = input_dir / filename
        safe_name = filename.replace(" ", "_").replace(".pdf", "")
        
        logger.info(f"Processing: {filename}")
        try:
            pages = convert_from_path(file_path, dpi=300)
            
            for i, page in enumerate(pages):
                image = np.array(page)
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    
                # Exact logic used in pipeline
                payload_img, is_sig = extractor.build_phi_safe_payload(image)
                
                if payload_img is not None:
                    out_path = output_dir / f"{safe_name}_page_{i+1}.png"
                    # cv2 requires BGR
                    cv2.imwrite(str(out_path), cv2.cvtColor(payload_img, cv2.COLOR_RGB2BGR))
                    logger.info(f"  -> Saved {out_path.name}")
                else:
                    logger.warning(f"  -> Page {i+1} returned None (likely blank or failed table detection completely)")
                    
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            
    logger.info("Done generating all crops!")

if __name__ == "__main__":
    main()
