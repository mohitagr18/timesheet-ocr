"""Band-crop extractor — PHI-safe two-band payload builder for Gemini.

Surgically extracts two narrow image bands per page:
  Band 1 — DATE (Month/Day/Year) row (shifted UP to capture handwriting)
  Band 2 — Footer block: TIME IN + TIME OUT + NUMBER OF HOURS

Only billing fields are transmitted to Gemini — zero clinical PHI.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .vlm_cloud import CloudVlmExtractor

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — tune these if bands are misaligned
# ---------------------------------------------------------------------------
PAD = 50
BREATHING_ROOM = 35  # extra px above/below each band
DATE_Y_SHIFT_MULTIPLIER = -1.5  # shift crop UP by 1.5x label height to get handwriting

# DATE row: hardcoded fractional position in PADDED IMAGE HEIGHT
DATE_Y_START_FRAC = 0.005
DATE_Y_END_FRAC = 0.210

# DATE OCR search: top fraction of padded image height
DATE_ROI_FRAC = 0.25

# Footer block: fractions of PADDED IMAGE HEIGHT, anchored to image bottom
FOOTER_BOTTOM_MARGIN_FRAC = 0.02
FOOTER_HEIGHT_FRAC = 0.12

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


def _add_padding(image: np.ndarray, pad: int = PAD) -> np.ndarray:
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
        # Expand instead of shift to capture the dates without exposing PHI
        box_h = best_box["y_max"] - best_box["y_min"]
        expand_up = box_h * 0.5    # slight expansion up for tall handwriting
        expand_down = box_h * 0.5  # slight expansion down for low handwriting
        return {
            "y_min": best_box["y_min"] - expand_up,
            "y_max": best_box["y_max"] + expand_down,
            "matched_text": best_box["text"],
            "score": best_score,
            "coords_in": "roi_local",
        }

    # Hardcoded fallback
    y_min = padded_h * DATE_Y_START_FRAC
    y_max = padded_h * DATE_Y_END_FRAC
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


def _get_footer_coords(padded_h: int) -> tuple[int, int]:
    """Compute footer block Y-coordinates (anchored to image bottom)."""
    footer_y1 = int(padded_h * (1.0 - FOOTER_BOTTOM_MARGIN_FRAC))
    footer_y0 = int(padded_h * (1.0 - FOOTER_BOTTOM_MARGIN_FRAC - FOOTER_HEIGHT_FRAC))
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
    pad: int = PAD,
    breathing: int = BREATHING_ROOM,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Slice date and footer bands from the original image, clamped to table X."""
    img_h, img_w = original.shape[:2]
    tx0, _, tx1, _ = table_bbox_padded
    ox0 = max(0, tx0 - pad)
    ox1 = min(img_w, tx1 - pad)

    if date_band.get("coords_in") == "padded":
        date_y0 = max(0, int(date_band["y_min"] - pad) - breathing)
        date_y1 = min(img_h, int(date_band["y_max"] - pad) + breathing)
    else:
        date_y0 = max(0, int(date_band["y_min"] + date_roi_pad_y0 - pad) - breathing)
        date_y1 = min(img_h, int(date_band["y_max"] + date_roi_pad_y0 - pad) + breathing)

    date_crop = original[date_y0:date_y1, ox0:ox1]

    foot_y0 = max(0, footer_pad_y0 - pad - breathing)
    foot_y1 = min(img_h, footer_pad_y1 - pad + breathing)
    foot_crop = original[foot_y0:foot_y1, ox0:ox1]

    return (
        date_crop if date_crop.size > 0 else None,
        foot_crop if foot_crop.size > 0 else None,
    )


def _stitch(bands: list[np.ndarray]) -> np.ndarray | None:
    """Vertically concatenate bands, padding narrower bands to widest width."""
    if not bands:
        return None
    w = max(b.shape[1] for b in bands)
    out = []
    for b in bands:
        if b.shape[1] < w:
            b = cv2.copyMakeBorder(
                b, 0, 0, 0, w - b.shape[1],
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
        out.append(b)
    return cv2.vconcat(out)


class BandCropExtractor:
    """Builds PHI-safe band-crop payloads and sends them to Gemini.

    Public interface:
        extract_page(image) -> {"shifts": [...], "rn_lpn_name": ""}
        build_phi_safe_payload(image) -> (stitched_image | None, is_signature_page)
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cloud_vlm = CloudVlmExtractor(config)

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

        Privacy guarantee: only the top date-row band and bottom footer
        band are ever extracted.  The middle of the page (where patient
        names, clinical notes, and other PHI live) is never included,
        regardless of whether table detection succeeds or fails.
        """
        padded = _add_padding(image, PAD)
        ph, pw = padded.shape[:2]

        # Detect table bbox — used for X-clipping optimisation.
        table_bbox = _detect_table_bbox(padded)

        if table_bbox is None:
            # PP-DocLayoutV3 didn't find a table.  This could mean:
            #   (a) It's a genuine signature/summary page → skip, OR
            #   (b) The model just missed the grid → still extract bands.
            #
            # Distinguish via OCR box count (same heuristic other
            # approaches use).  Run a quick OCR on the full padded image
            # and check the number of detected text boxes.
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

            # Enough text on the page → it's a grid page whose table
            # PP-DocLayoutV3 missed.  Use full padded-image width for
            # X-clipping.  PHI safety is preserved because we still
            # only slice the top (date) and bottom (footer) Y-bands.
            logger.warning(
                "No table bbox from DocLayoutV3 but %d OCR boxes — "
                "proceeding with full-width band extraction.",
                len(ocr_boxes),
            )
            table_bbox = (PAD, PAD, pw - PAD, ph - PAD)

        # OCR on date ROI for DATE row localisation
        date_roi_h = int(ph * DATE_ROI_FRAC)
        date_roi_crop = padded[0:date_roi_h, 0:pw].copy()
        date_ocr = _run_ocr(date_roi_crop, "date-roi")

        # Find DATE band coordinates
        date_band = _find_date_band(date_ocr, date_roi_h, ph)

        # Footer block coordinates
        footer_y0, footer_y1 = _get_footer_coords(ph)

        # Slice bands from original image
        date_crop, foot_crop = _slice_bands(
            image, date_band, 0, footer_y0, footer_y1, table_bbox, PAD, BREATHING_ROOM,
        )

        bands = [b for b in [date_crop, foot_crop] if b is not None]
        if not bands:
            logger.error("No bands extracted for this page.")
            return None, False

        stitched = _stitch(bands)
        if stitched is None or stitched.size == 0:
            logger.error("Stitched band payload is empty.")
            return None, False

        return stitched, False

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
