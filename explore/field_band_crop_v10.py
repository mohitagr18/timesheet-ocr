#!/usr/bin/env python
"""
field_band_crop_v7.py  —  PHI-Safe Timesheet Row Extractor
==========================================================

Two-band approach:
  Band 1 (top)    — DATE (Month/Day/Year) row
  Band 2 (bottom) — entire footer block as ONE crop:
                    TIME IN + TIME OUT + NUMBER OF HOURS

Gemini receives the stitched image and deciphers all values.

DATE detection strategy:
  1. Try OCR fuzzy match on top ROI
  2. Fall back to hardcoded fractional position (DATE is always at a
     known position in this form — calibrate DATE_Y_START_FRAC / DATE_Y_END_FRAC)

Config to tune
--------------
  DATE_Y_START_FRAC         top edge of DATE row as fraction of padded image height
  DATE_Y_END_FRAC           bottom edge of DATE row as fraction of padded image height
  FOOTER_BOTTOM_MARGIN_FRAC fraction of image height to skip at very bottom (white margin)
  FOOTER_HEIGHT_FRAC        fraction of image height the footer block spans
  BREATHING_ROOM            extra px added above/below each band

Output files
------------
  debug_01_padded.jpg     padded input
  debug_02_date_roi.jpg   strip used for DATE OCR
  debug_03_bands.jpg      both bands overlaid on padded image
  debug_04_footer_crop.jpg raw footer crop for tuning
  phi_safe_payload.jpg    final stitched output (DATE + footer)

Usage
-----
    python explore/field_band_crop_v7.py <path/to/timesheet.pdf> [--out-dir ./explore/output]
"""

from __future__ import annotations

import difflib
import logging
import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — tune these if bands are misaligned
# ---------------------------------------------------------------------------
PAD             = 50
BREATHING_ROOM  = 35     # extra px above/below each band — increase for taller bands
DATE_Y_SHIFT_MULTIPLIER = -1.5
# DATE row: hardcoded fractional position in PADDED IMAGE HEIGHT
# Calibrated from debug images: DATE row ~px 580-650 in 3319px padded image
DATE_Y_START_FRAC = 0.005   # top edge of DATE row  (580/3319 ≈ 0.175)
DATE_Y_END_FRAC   = 0.210   # bottom edge of DATE row (696/3319 ≈ 0.210)

# DATE OCR search: top fraction of padded image height
DATE_ROI_FRAC     = 0.25

# Footer block: fractions of PADDED IMAGE HEIGHT, anchored to image bottom
FOOTER_BOTTOM_MARGIN_FRAC = 0.02   # skip this much at very bottom (white margin)
FOOTER_HEIGHT_FRAC        = 0.12   # height of the 3-row footer block

MIN_OCR_SCORE   = 0.20
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


# ===========================================================================
# I/O
# ===========================================================================

def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path
        pages = convert_from_path(str(path), dpi=300)
        return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def add_padding(image: np.ndarray, pad: int = PAD) -> np.ndarray:
    return cv2.copyMakeBorder(
        image, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255),
    )


# ===========================================================================
# Table bbox — used only for X coordinates
# ===========================================================================

def detect_table_bbox(padded_image: np.ndarray) -> tuple[int, int, int, int]:
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    h, w = padded_image.shape[:2]
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
            break
    except Exception as exc:
        logger.warning("DocLayoutV3 error: %s — using full image.", exc)
    logger.warning("No table detected — using full padded image.")
    return (0, 0, w, h)


# ===========================================================================
# OCR + DATE matching
# ===========================================================================

def run_ocr(crop: np.ndarray, label: str) -> list[dict]:
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
    for b in boxes:
        logger.info("  [%s]  x=%.0f  score=%.2f  text='%s'",
                    label, b["x_min"], b["score"], b["text"])
    return boxes


def _fuzzy(keyword: str, text: str) -> float:
    kw = keyword.lower().replace(" ", "")
    tx = text.lower().replace(" ", "")
    if kw in tx:
        return 1.0
    return difflib.SequenceMatcher(None, kw, tx).ratio()


def find_date_band(ocr_boxes: list[dict], roi_h: int, padded_h: int) -> dict:
    """
    Try OCR fuzzy match first.
    If it fails, use hardcoded fractional position — DATE is always at a
    known vertical position in this form layout.
    Y coords returned are in PADDED IMAGE SPACE.
    """
    best_score, best_box = 0.0, None
    for box in ocr_boxes:
        for kw in DATE_KEYWORDS:
            s = _fuzzy(kw, box["text"])
            if s > best_score:
                best_score, best_box = s, box

    if best_score >= FUZZY_THRESHOLD and best_box is not None:
        logger.info("  DATE fuzzy match: '%-40s'  score=%.2f", best_box["text"], best_score)
        # OCR y coords are ROI-local; shift to padded-image space
        # FIX: Shift UP to capture the handwritten values above the label
        box_h = best_box["y_max"] - best_box["y_min"]
        shift_y = box_h * DATE_Y_SHIFT_MULTIPLIER

        return {
            "y_min": best_box["y_min"] + shift_y, # ROI-local; caller adds roi_pad_y0
            "y_max": best_box["y_max"] + shift_y,
            "matched_text": best_box["text"],
            "score": best_score,
            "coords_in": "roi_local",
        }

    # Hardcoded fallback — coords already in padded-image space
    y_min = padded_h * DATE_Y_START_FRAC
    y_max = padded_h * DATE_Y_END_FRAC
    logger.warning(
        "  DATE fuzzy failed (best=%.2f). "
        "Hardcoded fallback: padded y=[%.0f:%.0f] "
        "(DATE_Y_START_FRAC=%.3f  DATE_Y_END_FRAC=%.3f)",
        best_score, y_min, y_max, DATE_Y_START_FRAC, DATE_Y_END_FRAC,
    )
    return {
        "y_min": y_min, "y_max": y_max,
        "matched_text": "[hardcoded fallback]",
        "score": 0.0,
        "coords_in": "padded",
    }


# ===========================================================================
# Footer block — single crop anchored to image bottom
# ===========================================================================

def get_footer_coords(padded_h: int) -> tuple[int, int]:
    """Returns (footer_y0, footer_y1) in PADDED image space."""
    footer_y1 = int(padded_h * (1.0 - FOOTER_BOTTOM_MARGIN_FRAC))
    footer_y0 = int(padded_h * (1.0 - FOOTER_BOTTOM_MARGIN_FRAC - FOOTER_HEIGHT_FRAC))
    logger.info(
        "Footer block (padded): y=[%d:%d]  height=%d  "
        "(FOOTER_BOTTOM_MARGIN_FRAC=%.2f  FOOTER_HEIGHT_FRAC=%.2f)",
        footer_y0, footer_y1, footer_y1 - footer_y0,
        FOOTER_BOTTOM_MARGIN_FRAC, FOOTER_HEIGHT_FRAC,
    )
    return footer_y0, footer_y1


# ===========================================================================
# Slice both bands from ORIGINAL (unpadded) image
# ===========================================================================

def slice_bands(
    original: np.ndarray,
    date_band: dict,
    date_roi_pad_y0: int,
    footer_pad_y0: int,
    footer_pad_y1: int,
    table_bbox_padded: tuple[int, int, int, int],
    pad: int       = PAD,
    breathing: int = BREATHING_ROOM,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    img_h, img_w = original.shape[:2]
    tx0, _, tx1, _ = table_bbox_padded
    ox0 = max(0,     tx0 - pad)
    ox1 = min(img_w, tx1 - pad)

    # DATE: convert to original-image Y depending on coord space
    if date_band.get("coords_in") == "padded":
        # Already in padded space — just subtract PAD
        date_y0 = max(0,     int(date_band["y_min"] - pad) - breathing)
        date_y1 = min(img_h, int(date_band["y_max"] - pad) + breathing)
    else:
        # ROI-local → padded → original
        date_y0 = max(0,     int(date_band["y_min"] + date_roi_pad_y0 - pad) - breathing)
        date_y1 = min(img_h, int(date_band["y_max"] + date_roi_pad_y0 - pad) + breathing)

    date_crop = original[date_y0:date_y1, ox0:ox1]
    logger.info("  [DATE   ]  orig y=[%d:%d]  shape=%s", date_y0, date_y1, date_crop.shape)

    # Footer: padded → original
    foot_y0 = max(0,     footer_pad_y0 - pad - breathing)
    foot_y1 = min(img_h, footer_pad_y1 - pad + breathing)
    foot_crop = original[foot_y0:foot_y1, ox0:ox1]
    logger.info("  [FOOTER ]  orig y=[%d:%d]  shape=%s", foot_y0, foot_y1, foot_crop.shape)

    return (
        date_crop if date_crop.size > 0 else None,
        foot_crop if foot_crop.size > 0 else None,
    )


# ===========================================================================
# Stitch
# ===========================================================================

def stitch(bands: list[np.ndarray]) -> np.ndarray | None:
    if not bands:
        return None
    w = max(b.shape[1] for b in bands)
    out = []
    for b in bands:
        if b.shape[1] < w:
            b = cv2.copyMakeBorder(b, 0, 0, 0, w - b.shape[1],
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        out.append(b)
    return cv2.vconcat(out)


# ===========================================================================
# Debug overlay
# ===========================================================================

def save_debug_overlay(
    padded: np.ndarray,
    table_bbox: tuple[int, int, int, int],
    date_band: dict,
    date_roi_pad_y0: int,
    footer_pad_y0: int,
    footer_pad_y1: int,
    out_path: Path,
) -> None:
    vis = padded.copy()
    pw  = vis.shape[1]
    tx0, ty0, tx1, ty1 = table_bbox
    cv2.rectangle(vis, (tx0, ty0), (tx1, ty1), (160, 160, 160), 2)
    cv2.putText(vis, "DocLayoutV3 table bbox",
                (tx0 + 4, ty0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 130, 130), 2)

    # DATE band — convert to padded space for drawing
    if date_band.get("coords_in") == "padded":
        py0 = int(date_band["y_min"])
        py1 = int(date_band["y_max"])
    else:
        py0 = int(date_band["y_min"]) + date_roi_pad_y0
        py1 = int(date_band["y_max"]) + date_roi_pad_y0
    cv2.rectangle(vis, (0, py0), (pw - 1, py1), (220, 20, 60), 3)
    cv2.putText(vis, f"DATE: {date_band.get('matched_text','')[:40]}",
                (4, max(py0 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 20, 60), 2)

    # Footer block
    cv2.rectangle(vis, (0, footer_pad_y0), (pw - 1, footer_pad_y1), (34, 139, 34), 3)
    cv2.putText(vis, "FOOTER (TIME IN + TIME OUT + NUMBER OF HOURS)",
                (4, max(footer_pad_y0 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 139, 34), 2)

    cv2.imwrite(str(out_path), vis)
    logger.info("Debug overlay -> %s", out_path)


# ===========================================================================
# Main
# ===========================================================================

def run(image_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("field_band_crop_v7  |  %s", image_path)
    logger.info("=" * 64)

    # 1. Load + pad
    logger.info("Step 1 — Load and pad ...")
    original = load_image(Path(image_path))
    padded   = add_padding(original, PAD)
    cv2.imwrite(str(out / "debug_01_padded.jpg"), padded)
    ph, pw = padded.shape[:2]
    logger.info("Original: %s   Padded: h=%d w=%d", original.shape, ph, pw)
    logger.info("DATE hardcoded position: padded y=[%.0f:%.0f]",
                ph * DATE_Y_START_FRAC, ph * DATE_Y_END_FRAC)

    # 2. Table bbox (X coords only — Y bottom unreliable)
    logger.info("Step 2 — Detect table bbox (X coords only) ...")
    table_bbox = detect_table_bbox(padded)
    tx0, ty0, tx1, ty1 = table_bbox
    logger.info("Table bbox: x=[%d:%d]  y=[%d:%d]", tx0, tx1, ty0, ty1)

    # 3. DATE ROI — OCR on full width to avoid label clipping
    logger.info("Step 3 — DATE ROI + OCR ...")
    date_roi_h  = int(ph * DATE_ROI_FRAC)
    date_roi_crop = padded[0:date_roi_h, 0:pw].copy()   # full width, not clipped to tx0
    cv2.imwrite(str(out / "debug_02_date_roi.jpg"), date_roi_crop)
    logger.info("DATE ROI  padded y=[0:%d]  shape=%s", date_roi_h, date_roi_crop.shape)

    date_ocr  = run_ocr(date_roi_crop, "date-roi")
    date_band = find_date_band(date_ocr, date_roi_h, ph)
    logger.info("  DATE result: coords_in=%s  y=[%.1f:%.1f]  text='%s'",
                date_band["coords_in"], date_band["y_min"], date_band["y_max"],
                date_band["matched_text"])
    logger.info(">>> DATE BAND FINAL: coords_in=%s  y_min=%.1f  y_max=%.1f  padded_h=%d",
            date_band["coords_in"], date_band["y_min"], date_band["y_max"], ph)

    # 4. Footer block coords
    logger.info("Step 4 — Footer block (image-height anchored) ...")
    footer_y0, footer_y1 = get_footer_coords(ph)
    cv2.imwrite(str(out / "debug_04_footer_crop.jpg"),
                padded[footer_y0:footer_y1, tx0:tx1].copy())
    logger.info("Footer crop saved -> debug_04_footer_crop.jpg")

    # Debug overlay
    save_debug_overlay(padded, table_bbox, date_band, 0,
                       footer_y0, footer_y1, out / "debug_03_bands.jpg")

    # 5. Slice
    logger.info("Step 5 — Slice bands from original image ...")
    date_crop, foot_crop = slice_bands(
        original, date_band, 0,
        footer_y0, footer_y1,
        table_bbox, PAD, BREATHING_ROOM,
    )

    bands = [b for b in [date_crop, foot_crop] if b is not None]
    if not bands:
        logger.error("No bands extracted.")
        sys.exit(1)

    # 6. Stitch
    logger.info("Step 6 — Stitch ...")
    stitched = stitch(bands)
    if stitched is None or stitched.size == 0:
        logger.error("Stitch empty.")
        sys.exit(1)

    final = out / "phi_safe_payload.jpg"
    cv2.imwrite(str(final), stitched)

    logger.info("=" * 64)
    logger.info("SUCCESS  ->  %s", final.resolve())
    logger.info("Bands: DATE + FOOTER (2 total)")
    logger.info("=" * 64)
    logger.info("")
    logger.info("Calibration hints:")
    logger.info("  DATE too high?   decrease DATE_Y_START_FRAC (currently %.3f)", DATE_Y_START_FRAC)
    logger.info("  DATE too low?    increase  DATE_Y_START_FRAC (currently %.3f)", DATE_Y_START_FRAC)
    logger.info("  DATE too narrow? increase  BREATHING_ROOM    (currently %d)", BREATHING_ROOM)
    logger.info("  Footer wrong?    adjust FOOTER_HEIGHT_FRAC   (currently %.2f)", FOOTER_HEIGHT_FRAC)
    logger.info("  Check debug_03_bands.jpg to verify both bands visually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PHI-safe timesheet row extractor (v7).")
    parser.add_argument("image", help="Path to timesheet image or PDF.")
    parser.add_argument("--out-dir", default="./explore/output",
                        help="Output directory (default: ./explore/output)")
    args = parser.parse_args()
    run(args.image, args.out_dir)
