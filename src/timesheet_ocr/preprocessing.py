"""Image preprocessing — grayscale, deskew, denoise, binarize for OCR-ready images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as a BGR numpy array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        # Fallback: try PIL for formats cv2 can't handle
        pil_img = Image.open(path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    logger.info(f"Loaded image: {path.name} ({img.shape[1]}x{img.shape[0]})")
    return img


def preprocess_image(img: np.ndarray, config: AppConfig) -> np.ndarray:
    """Apply full preprocessing pipeline to prepare image for OCR.

    Steps:
    1. Convert to grayscale
    2. Deskew (straighten rotated scans)
    3. Denoise (remove scan noise)
    4. Binarize (adaptive thresholding for clean text)

    Returns the preprocessed image (grayscale, uint8).
    """
    prep = config.preprocessing
    result = img.copy()

    # 1. Grayscale
    if len(result.shape) == 3:
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    else:
        gray = result

    # 2. Deskew
    if prep.deskew:
        gray = _deskew(gray)

    # 3. Denoise
    if prep.denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Binarize (adaptive threshold)
    if prep.binarize:
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            prep.adaptive_block_size,
            prep.adaptive_c,
        )

    logger.info(f"Preprocessed image: {gray.shape[1]}x{gray.shape[0]}")
    return gray


def _deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Deskew a grayscale image by detecting dominant line angle.

    Uses Hough Line Transform to find the median angle of detected lines,
    then rotates the image to correct the skew.
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        logger.debug("No lines detected for deskew, skipping")
        return image

    # Calculate angles
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (within max_angle of horizontal)
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        logger.debug("No near-horizontal lines found for deskew")
        return image

    median_angle = float(np.median(angles))

    if abs(median_angle) < 0.5:
        logger.debug(f"Skew angle {median_angle:.2f}° is negligible, skipping")
        return image

    logger.info(f"Deskewing by {median_angle:.2f}°")

    # Rotate
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def pdf_to_images(pdf_path: str | Path, dpi: int = 300) -> list[np.ndarray]:
    """Convert a PDF file to a list of images (one per page).

    Requires poppler to be installed (e.g., `brew install poppler` on macOS).
    """
    from pdf2image import convert_from_path

    pdf_path = Path(pdf_path)
    logger.info(f"Converting PDF to images: {pdf_path.name} at {dpi} DPI")

    pil_images = convert_from_path(str(pdf_path), dpi=dpi)

    images = []
    for i, pil_img in enumerate(pil_images):
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        images.append(img)
        logger.info(f"  Page {i + 1}: {img.shape[1]}x{img.shape[0]}")

    return images
