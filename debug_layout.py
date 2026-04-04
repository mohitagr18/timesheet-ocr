"""
Diagnostic script: Render a timesheet page, run PaddleOCR, and annotate every
detected text box with its fractional Y-coordinate.  Outputs:
  1. debug_ocr_boxes.png  — the page image with bounding boxes and y-fractions
  2. debug_ocr_boxes.txt  — a text dump of every box sorted by y-fraction

Usage:
    cd /Users/mohit/Documents/GitHub/timesheet-ocr
    uv run python debug_layout.py "input/C.Ferguson Timesheets - 010726-011326.pdf"
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

# ── Configuration ───────────────────────────────────────────────────
DPI = 400  # match your config.yaml target_dpi
PAGE_INDEX = 0  # 0 = first page
USE_TEXTLINE_ORIENTATION = False

# ── Load ────────────────────────────────────────────────────────────
pdf_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "input/C.Ferguson Timesheets - 010726-011326.pdf"
)
print(f"Loading {pdf_path} at {DPI} DPI ...")
pil_images = convert_from_path(pdf_path, dpi=DPI)
pil_img = pil_images[PAGE_INDEX]
image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
h, w = image.shape[:2]
print(f"Page {PAGE_INDEX + 1}: {w}x{h} pixels")

# ── Run PaddleOCR ──────────────────────────────────────────────────
print("Running PaddleOCR ...")
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_textline_orientation=USE_TEXTLINE_ORIENTATION,
    device="cpu",
    text_det_thresh=0.45,
    text_recognition_batch_size=4,
)
result = ocr.ocr(image)

# ── Collect boxes ──────────────────────────────────────────────────
boxes = []
if result:
    for page in result:
        rec_texts = page.get("rec_texts", [])
        rec_scores = page.get("rec_scores", [])
        rec_polys = page.get("rec_polys", [])
        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
            bbox = poly.tolist()
            y_center = sum(p[1] for p in bbox) / 4
            x_center = sum(p[0] for p in bbox) / 4
            y_frac = y_center / h
            x_frac = x_center / w
            boxes.append(
                {
                    "text": text,
                    "conf": float(score),
                    "bbox": bbox,
                    "y_center": y_center,
                    "x_center": x_center,
                    "y_frac": y_frac,
                    "x_frac": x_frac,
                }
            )

boxes.sort(key=lambda b: (b["y_frac"], b["x_frac"]))
print(f"Detected {len(boxes)} text boxes")

# ── Draw annotated image ───────────────────────────────────────────
annotated = image.copy()

# Draw horizontal band guides at every 5%
for frac in range(0, 100, 5):
    y = int(h * frac / 100)
    color = (200, 200, 200) if frac % 10 else (100, 100, 255)
    cv2.line(annotated, (0, y), (w, y), color, 1)
    cv2.putText(
        annotated,
        f"{frac}%",
        (5, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (100, 100, 255),
        2,
    )

# Draw each OCR box
for b in boxes:
    pts = np.array(b["bbox"], dtype=np.int32)
    cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)
    label = f'{b["y_frac"]:.2f} "{b["text"][:30]}"'
    cv2.putText(
        annotated,
        label,
        (int(pts[0][0]), int(pts[0][1]) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1,
    )

# Save annotated image
out_img = "debug_ocr_boxes.png"
cv2.imwrite(out_img, annotated)
print(f"Saved annotated image → {out_img}")

# ── Dump text‐sorted list ─────────────────────────────────────────
out_txt = "debug_ocr_boxes.txt"
with open(out_txt, "w") as f:
    f.write(f"Page {PAGE_INDEX + 1} of {pdf_path}  ({w}x{h}px at {DPI} DPI)\n")
    f.write(f"{'─' * 80}\n")
    f.write(f"{'Y%':>6}  {'X%':>6}  {'Conf':>5}  Text\n")
    f.write(f"{'─' * 80}\n")
    for b in boxes:
        f.write(
            f"{b['y_frac'] * 100:6.1f}  {b['x_frac'] * 100:6.1f}  {b['conf']:.3f}  {b['text']}\n"
        )

print(f"Saved text dump      → {out_txt}")
print("\nDone!  Open debug_ocr_boxes.png to visually locate TIME IN / TIME OUT rows.")
