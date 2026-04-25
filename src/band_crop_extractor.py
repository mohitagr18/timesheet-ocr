"""Band-crop extractor — PHI-safe two-band payload builder for Gemini.

Surgically extracts two narrow image bands per page:
  Band 1 — DATE (Month/Day/Year) row (shifted UP to capture handwriting)
  Band 2 — Footer block: TIME IN + TIME OUT + NUMBER OF HOURS

Only billing fields are transmitted to Gemini — zero clinical PHI.
"""

from __future__ import annotations

import difflib
import logging
import time as time_module
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .vlm_cloud import CloudVlmExtractor

if TYPE_CHECKING:
    from .config import AppConfig, BandCropConfig

logger = logging.getLogger(__name__)

# Default constants - will be overridden by config if available
DEFAULT_PAD = 50
DEFAULT_BREATHING_ROOM = 35
DEFAULT_DATE_Y_START_FRAC = 0.005
DEFAULT_DATE_Y_END_FRAC = 0.210
DEFAULT_DATE_ROI_FRAC = 0.25
DEFAULT_FOOTER_BOTTOM_MARGIN_FRAC = 0.02
DEFAULT_FOOTER_HEIGHT_FRAC = 0.12
DEFAULT_STITCH_GAP = 20  # pixels of white space between date and footer crops

MIN_OCR_SCORE = 0.20
FUZZY_THRESHOLD = 0.45

DATE_KEYWORDS = [
    "date (month/day/year)",
    "date (month",
    "date month/day",
    "month/day/year",
    "month/day",
    "anchOayYear",
    "date",
]


def _get_config_values(config: "AppConfig | None") -> dict:
    """Get band_crop config values with sensible defaults."""
    if config and hasattr(config, "band_crop"):
        bc: BandCropConfig = config.band_crop
        return {
            "pad": DEFAULT_PAD,
            "breathing": bc.date_breathing_room,
            "date_y_start_frac": bc.date_band_top_margin_frac,
            "date_y_end_frac": bc.date_band_bottom_margin_frac,
            "date_roi_frac": DEFAULT_DATE_ROI_FRAC,
            "footer_bottom_frac": DEFAULT_FOOTER_BOTTOM_MARGIN_FRAC,
            "footer_height_frac": DEFAULT_FOOTER_HEIGHT_FRAC,
            "date_retry_expansion": bc.date_retry_expansion_frac,
            "enable_retry": bc.enable_date_retry,
            "stitch_gap": bc.stitch_gap if hasattr(bc, "stitch_gap") else DEFAULT_STITCH_GAP,
        }
    return {
        "pad": DEFAULT_PAD,
        "breathing": DEFAULT_BREATHING_ROOM,
        "date_y_start_frac": DEFAULT_DATE_Y_START_FRAC,
        "date_y_end_frac": DEFAULT_DATE_Y_END_FRAC,
        "date_roi_frac": DEFAULT_DATE_ROI_FRAC,
        "footer_bottom_frac": DEFAULT_FOOTER_BOTTOM_MARGIN_FRAC,
        "footer_height_frac": DEFAULT_FOOTER_HEIGHT_FRAC,
        "date_retry_expansion": 0.05,
        "enable_retry": True,
        "stitch_gap": DEFAULT_STITCH_GAP,
    }


def _add_padding(image: np.ndarray, pad: int = DEFAULT_PAD) -> np.ndarray:
    """Add white padding around the image to avoid edge clipping."""
    return cv2.copyMakeBorder(
        image, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


def _detect_table_bbox(padded_image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect table/grid region using PP-DocLayoutV3.

    Returns bbox in padded-image coordinates, or None if no table found.
    """
    try:
        from paddlex import create_model

        model = create_model("PP-DocLayoutV3")
        img = padded_image.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for result in model.predict(img, batch_size=1):
            for box in result.get("boxes", []):
                if box.get("label", "").lower() in ("table", "grid"):
                    coord = box.get("coordinate")
                    if coord and len(coord) == 4:
                        bbox = tuple(int(v) for v in coord)
                        logger.info("DocLayoutV3 table bbox (padded): %s", bbox)
                        return bbox
    except Exception as exc:
        logger.warning("DocLayoutV3 error: %s", exc)

    logger.warning("No table detected on this page — falling back to full image width.")
    return None


def _run_ocr(crop: np.ndarray, label: str) -> list[dict]:
    """Run PaddleOCR on a crop and return list of box dicts."""
    from paddleocr import PaddleOCR

    img = crop.copy()
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_textline_orientation=False,
        device="cpu",
        text_det_thresh=0.20,
    )
    raw = ocr.predict(img)
    boxes: list[dict] = []
    if not raw:
        return boxes
    for page in raw:
        if not page:
            continue
        for text, score, poly in zip(
            page.get("rec_texts", []),
            page.get("rec_scores", []),
            page.get("rec_polys", []),
        ):
            if score < MIN_OCR_SCORE:
                continue
            pts = [[float(p[0]), float(p[1])] for p in poly]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append({
                "text": text.strip(), "score": float(score),
                "x_min": min(xs), "x_max": max(xs),
                "y_min": min(ys), "y_max": max(ys),
            })
    logger.info("OCR [%s]: %d boxes.", label, len(boxes))
    return boxes


def _fuzzy(keyword: str, text: str) -> float:
    """Fuzzy-match a keyword against text (case-insensitive, spaces ignored)."""
    kw = keyword.lower().replace(" ", "")
    tx = text.lower().replace(" ", "")
    if kw in tx:
        return 1.0
    return difflib.SequenceMatcher(None, kw, tx).ratio()


def _find_date_band(
    ocr_boxes: list[dict], roi_h: int, padded_h: int,
    date_y_start_frac: float = DEFAULT_DATE_Y_START_FRAC,
    date_y_end_frac: float = DEFAULT_DATE_Y_END_FRAC,
) -> dict:
    """Find the DATE row Y-coordinates via fuzzy matching or hardcoded fallback."""
    best_score: float = 0.0
    best_box: dict | None = None

    for box in ocr_boxes:
        for kw in DATE_KEYWORDS:
            s = _fuzzy(kw, box["text"])
            if s > best_score:
                best_score, best_box = s, box

    if best_score >= FUZZY_THRESHOLD and best_box is not None:
        logger.info(
            "  DATE fuzzy match: '%-40s' score=%.2f",
            best_box["text"], best_score,
        )
        box_h = best_box["y_max"] - best_box["y_min"]
        expand_up = box_h * 1.5
        expand_down = box_h * 2.0
        return {
            "y_min": best_box["y_min"] - expand_up,
            "y_max": best_box["y_max"] + expand_down,
            "matched_text": best_box["text"],
            "score": best_score,
            "coords_in": "roi_local",
        }

    y_min = padded_h * date_y_start_frac
    y_max = padded_h * date_y_end_frac
    logger.warning(
        "  DATE fuzzy failed (best=%.2f). Hardcoded fallback: y=[%.0f:%.0f]",
        best_score, y_min, y_max,
    )
    return {
        "y_min": y_min, "y_max": y_max,
        "matched_text": "[hardcoded fallback]",
        "score": 0.0,
        "coords_in": "padded",
    }


def _get_footer_coords(padded_h: int, footer_bottom_frac: float = DEFAULT_FOOTER_BOTTOM_MARGIN_FRAC, footer_height_frac: float = DEFAULT_FOOTER_HEIGHT_FRAC) -> tuple[int, int]:
    """Compute footer block Y-coordinates (anchored to image bottom)."""
    footer_y1 = int(padded_h * (1.0 - footer_bottom_frac))
    footer_y0 = int(padded_h * (1.0 - footer_bottom_frac - footer_height_frac))
    logger.info(
        "Footer block (padded): y=[%d:%d] height=%d",
        footer_y0, footer_y1, footer_y1 - footer_y0,
    )
    return footer_y0, footer_y1


def _slice_bands(
    original: np.ndarray,
    date_band: dict,
    date_roi_pad_y0: int,
    footer_pad_y0: int,
    footer_pad_y1: int,
    table_bbox_padded: tuple[int, int, int, int],
    pad: int = DEFAULT_PAD,
    breathing: int = DEFAULT_BREATHING_ROOM,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Slice date and footer bands from the original image, clamped to table X."""
    img_h, img_w = original.shape[:2]
    tx0, _, tx1, _ = table_bbox_padded
    ox0 = max(0, tx0 - pad)
    ox1 = min(img_w, tx1 - pad)

    if date_band.get("coords_in") == "padded":
        date_y0 = max(0, int(date_band["y_min"]) - breathing)
        date_y1 = min(img_h, int(date_band["y_max"]) + breathing)
    else:
        date_y0 = max(0, int(date_band["y_min"] + date_roi_pad_y0) - breathing)
        date_y1 = min(img_h, int(date_band["y_max"] + date_roi_pad_y0) + breathing)

    date_crop = original[date_y0:date_y1, ox0:ox1]

    foot_y0 = max(0, footer_pad_y0 - pad - breathing)
    foot_y1 = min(img_h, footer_pad_y1 - pad + breathing)
    foot_crop = original[foot_y0:foot_y1, ox0:ox1]

    logger.debug(
        "Band slicing: date_y=[%d:%d], foot_y=[%d:%d], ox=[%d:%d]",
        date_y0, date_y1, foot_y0, foot_y1, ox0, ox1,
    )

    return (
        date_crop if date_crop.size > 0 else None,
        foot_crop if foot_crop.size > 0 else None,
    )


def _stitch(bands: list[np.ndarray], gap_pixels: int = DEFAULT_STITCH_GAP) -> np.ndarray | None:
    """Vertically concatenate bands with a white gap between them.

    Args:
        bands: List of image bands to concatenate
        gap_pixels: Number of white pixels to add between bands (default 20)
    """
    if not bands:
        return None
    w = max(b.shape[1] for b in bands)

    # Create white gap band
    gap = np.full((gap_pixels, w, 3), 255, dtype=np.uint8)

    out = []
    for i, b in enumerate(bands):
        if b.shape[1] < w:
            b = cv2.copyMakeBorder(
                b, 0, 0, 0, w - b.shape[1],
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
        out.append(b)
        # Add gap between bands (but not after the last one)
        if i < len(bands) - 1:
            out.append(gap)

    return cv2.vconcat(out)


class BandCropExtractor:
    """Builds PHI-safe band-crop payloads and sends them to Gemini.

    Public interface:
        extract_page(image) -> {"shifts": [...], "rn_lpn_name": ""}
        build_phi_safe_payload(image) -> (stitched_image | None, is_signature_page)
        build_date_band_retry(image) -> (expanded_date_crop | None, is_signature_page)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cloud_vlm = CloudVlmExtractor(config)
        self._cfg = _get_config_values(config)

    def _pad(self) -> int:
        return self._cfg["pad"]

    def _breathing(self) -> int:
        return self._cfg["breathing"]

    def _date_roi_frac(self) -> float:
        return self._cfg["date_roi_frac"]

    def _date_retry_expansion(self) -> float:
        return self._cfg["date_retry_expansion"]

    def _enable_retry(self) -> bool:
        return self._cfg["enable_retry"]

    def _get_date_y_start_frac(self) -> float:
        return self._cfg["date_y_start_frac"]

    def _get_date_y_end_frac(self) -> float:
        return self._cfg["date_y_end_frac"]

    def _get_footer_bottom_frac(self) -> float:
        return self._cfg["footer_bottom_frac"]

    def _get_footer_height_frac(self) -> float:
        return self._cfg["footer_height_frac"]

    def _get_stitch_gap(self) -> int:
        return self._cfg.get("stitch_gap", DEFAULT_STITCH_GAP)

    def _get_date_coords_with_retry(self, padded_h: int, retry: bool = False) -> dict:
        """Get date band coordinates with optional retry expansion."""
        if retry:
            retry_exp = self._date_retry_expansion()
            y_min = padded_h * (self._get_date_y_start_frac() - retry_exp)
            y_max = padded_h * (self._get_date_y_end_frac() + retry_exp)
            y_min = max(0, y_min)
            y_max = min(padded_h, y_max)
            logger.info(
                "DATE band (retry): y=[%.0f:%.0f] (expanded by %.1f%%)",
                y_min, y_max, retry_exp * 100,
            )
            return {
                "y_min": y_min, "y_max": y_max,
                "matched_text": "[retry expanded]",
                "score": 0.0,
                "coords_in": "padded",
            }
        else:
            return {
                "y_min": padded_h * self._get_date_y_start_frac(),
                "y_max": padded_h * self._get_date_y_end_frac(),
                "matched_text": "[default]",
                "score": 0.0,
                "coords_in": "padded",
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_phi_safe_payload(
        self, image: np.ndarray,
    ) -> tuple[np.ndarray | None, bool]:
        """Build the stitched PHI-safe band payload for one page.

        Returns:
            (stitched_payload_image, is_signature_page)
            - (None, True)  → signature / summary page, skip
            - (stitched, False) → ready to send to Gemini
        """
        pad = self._pad()
        padded = _add_padding(image, pad)
        ph, pw = padded.shape[:2]

        table_bbox = _detect_table_bbox(padded)

        if table_bbox is None:
            sig_threshold = getattr(
                self.config, "debug", None
            )
            sig_threshold = (
                sig_threshold.signature_ocr_threshold
                if sig_threshold is not None
                else 100
            )

            ocr_boxes = _run_ocr(padded, "page-classify")
            if len(ocr_boxes) < sig_threshold:
                logger.info(
                    "No table detected AND only %d OCR boxes (< %d) — "
                    "treating as signature page.",
                    len(ocr_boxes),
                    sig_threshold,
                )
                return None, True

            logger.warning(
                "No table bbox from DocLayoutV3 but %d OCR boxes — "
                "proceeding with full-width band extraction.",
                len(ocr_boxes),
            )
            table_bbox = (pad, pad, pw - pad, ph - pad)

        date_roi_h = int(ph * self._date_roi_frac())
        
        if 'ocr_boxes' not in locals():
            ocr_boxes = _run_ocr(padded, "full-page")
            
        date_ocr = [box for box in ocr_boxes if box["y_min"] < date_roi_h]

        date_band = _find_date_band(
            date_ocr, date_roi_h, ph,
            date_y_start_frac=self._get_date_y_start_frac(),
            date_y_end_frac=self._get_date_y_end_frac(),
        )

        footer_y0, footer_y1 = _get_footer_coords(
            ph,
            footer_bottom_frac=self._get_footer_bottom_frac(),
            footer_height_frac=self._get_footer_height_frac(),
        )

        date_crop, foot_crop = _slice_bands(
            image, date_band, 0, footer_y0, footer_y1, table_bbox,
            pad, self._breathing(),
        )

        bands = [b for b in [date_crop, foot_crop] if b is not None]
        if not bands:
            logger.error("No bands extracted for this page.")
            return None, False

        stitched = _stitch(bands, gap_pixels=self._get_stitch_gap())
        if stitched is None or stitched.size == 0:
            logger.error("Stitched band payload is empty.")
            return None, False

        return stitched, False

    def build_date_band_retry(self, image: np.ndarray) -> tuple[np.ndarray | None, bool]:
        """Build expanded date band only for retry when initial extraction has no dates.

        Returns:
            (date_crop_only, is_signature_page)
            - (None, True) → signature page, skip
            - (crop, False) → expanded date band only (not stitched with footer)
        """
        pad = self._pad()
        padded = _add_padding(image, pad)
        ph, pw = padded.shape[:2]

        table_bbox = _detect_table_bbox(padded)

        if table_bbox is None:
            sig_threshold = getattr(self.config, "debug", None)
            sig_threshold = (
                sig_threshold.signature_ocr_threshold
                if sig_threshold is not None
                else 100
            )

            ocr_boxes = _run_ocr(padded, "page-classify")
            if len(ocr_boxes) < sig_threshold:
                return None, True

            table_bbox = (pad, pad, pw - pad, ph - pad)

        date_band = self._get_date_coords_with_retry(ph, retry=True)

        img_h, img_w = image.shape[:2]
        tx0, _, tx1, _ = table_bbox
        ox0 = max(0, tx0 - pad)
        ox1 = min(img_w, tx1 - pad)

        date_y0 = max(0, int(date_band["y_min"]) - self._breathing())
        date_y1 = min(img_h, int(date_band["y_max"]) + self._breathing())

        date_crop = image[date_y0:date_y1, ox0:ox1]

        if date_crop is None or date_crop.size == 0:
            logger.warning("Retry date band extraction returned empty crop.")
            return None, False

        logger.info(
            "Date band retry: y=[%d:%d], size=%dx%d",
            date_y0, date_y1, date_crop.shape[1], date_crop.shape[0],
        )

        return date_crop, False

    def extract_page(self, image: np.ndarray) -> dict:
        """Build the PHI-safe payload and send to Gemini.

        Returns standard shifts dict:
            {"shifts": [...], "rn_lpn_name": ""}
        Signature pages return {"shifts": [], "rn_lpn_name": ""}.
        """
        stitched, is_sig_page = self.build_phi_safe_payload(image)

        if is_sig_page or stitched is None:
            return {"shifts": [], "rn_lpn_name": ""}

        # Send stitched payload to Gemini via CloudVlmExtractor
        return self.cloud_vlm.extract_table_crop(stitched)
