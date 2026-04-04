"""PP-OCRv5 wrapper — primary OCR engine for timesheet extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class OcrBox:
    """A single detected text box from PP-OCRv5."""

    text: str
    confidence: float
    bbox: list[list[float]]  # 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    @property
    def x_center(self) -> float:
        """Center x-coordinate of the bounding box."""
        return sum(p[0] for p in self.bbox) / 4

    @property
    def y_center(self) -> float:
        """Center y-coordinate of the bounding box."""
        return sum(p[1] for p in self.bbox) / 4

    @property
    def x_min(self) -> float:
        return min(p[0] for p in self.bbox)

    @property
    def y_min(self) -> float:
        return min(p[1] for p in self.bbox)

    @property
    def x_max(self) -> float:
        return max(p[0] for p in self.bbox)

    @property
    def y_max(self) -> float:
        return max(p[1] for p in self.bbox)


@dataclass
class OcrResult:
    """Result of running PP-OCRv5 on an image or crop."""

    boxes: list[OcrBox] = field(default_factory=list)
    raw_output: Any = None

    @property
    def full_text(self) -> str:
        """Concatenated text from all boxes, sorted top-to-bottom, left-to-right."""
        sorted_boxes = sorted(self.boxes, key=lambda b: (b.y_center, b.x_center))
        return " ".join(b.text for b in sorted_boxes)


class OcrEngine:
    """Wrapper around PaddleOCR (PP-OCRv5) for text extraction.

    Initializes the OCR model once and provides methods for running
    inference on full images or cropped regions.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._ocr = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of PaddleOCR (heavy import)."""
        if self._initialized:
            return

        from paddleocr import PaddleOCR

        cfg = self.config.ppocr
        logger.info("Initializing PP-OCRv5 engine (mobile models)...")
        self._ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_textline_orientation=cfg.use_textline_orientation,
            device=cfg.device,
            text_det_thresh=cfg.text_det_thresh,
            text_recognition_batch_size=cfg.text_rec_batch_size,
        )
        self._initialized = True
        logger.info("PP-OCRv5 engine ready")

    def run(self, image: np.ndarray) -> OcrResult:
        """Run OCR on a full image. Returns structured OcrResult."""
        self._ensure_initialized()

        # PaddleOCR v3.4.0 requires 3-channel images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        orig_h, orig_w = image.shape[:2]
        result = self._ocr.ocr(image)

        boxes = []
        if result:
            for page in result:
                rec_texts = page.get("rec_texts", [])
                rec_scores = page.get("rec_scores", [])
                rec_polys = page.get("rec_polys", [])

                # Scale coordinates back to original image space if PaddleOCR
                # internally resized the image. The output_img shape tells us
                # the space the coordinates are in.
                output_shape = page.get("doc_preprocessor_res", {}).get(
                    "output_img", None
                )
                if output_shape is not None:
                    out_h, out_w = output_shape.shape[:2]
                    sx = orig_w / out_w if out_w != orig_w else 1.0
                    sy = orig_h / out_h if out_h != orig_h else 1.0
                else:
                    sx, sy = 1.0, 1.0

                for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                    bbox_points = [[float(p[0]) * sx, float(p[1]) * sy] for p in poly]
                    boxes.append(
                        OcrBox(text=text, confidence=float(score), bbox=bbox_points)
                    )

        logger.info(f"OCR detected {len(boxes)} text boxes")
        return OcrResult(boxes=boxes, raw_output=result)

    def run_on_crop(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> OcrResult:
        """Run OCR on a cropped region of the image.

        Coordinates are absolute pixel positions in the original image.
        Returns boxes with coordinates adjusted back to the original image space.
        """
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return OcrResult()

        result = self.run(crop)

        # Adjust bbox coordinates to original image space
        for box in result.boxes:
            for point in box.bbox:
                point[0] += x1
                point[1] += y1

        return result

    def extract_text_from_zone(
        self, image: np.ndarray, x_start: int, y_start: int, x_end: int, y_end: int
    ) -> tuple[str, float]:
        """Extract text and average confidence from a rectangular zone.

        Returns (text, min_confidence) for the zone.
        """
        result = self.run_on_crop(image, x_start, y_start, x_end, y_end)

        if not result.boxes:
            return ("", 0.0)

        text = result.full_text
        min_conf = min(b.confidence for b in result.boxes) if result.boxes else 0.0

        return (text, min_conf)
