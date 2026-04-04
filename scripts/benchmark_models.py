"""
Benchmark: PaddleOCR v3.4.0 Mobile vs Server models on handwritten timesheets.

Compares PP-OCRv5_mobile_det/rec vs PP-OCRv5_server_det/rec on the same images.
Logs detection count, recognition quality, and per-zone accuracy metrics.

Usage:
    uv run python scripts/benchmark_models.py "input/J.Flemming Timesheets - 012826-020326.pdf"

Output:
    output/benchmark_results.json  — full metrics for both models
    output/benchmark_report.txt    — human-readable summary
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress PaddleOCR verbosity
import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ── Configuration ───────────────────────────────────────────────────
DPI = 400
PAGE_INDEX = 0
TEXT_DET_THRESH = 0.45
TEXT_REC_BATCH_SIZE = 4

MODEL_PAIRS = {
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
}


def run_ocr(model_name: str, det_model: str, rec_model: str, image: np.ndarray):
    """Run OCR with specified models and return structured results."""
    start = time.time()
    ocr = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_textline_orientation=False,
        device="cpu",
        text_det_thresh=TEXT_DET_THRESH,
        text_recognition_batch_size=TEXT_REC_BATCH_SIZE,
    )
    result = ocr.ocr(image)
    elapsed = time.time() - start

    h, w = image.shape[:2]
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
                boxes.append(
                    {
                        "text": text,
                        "conf": float(score),
                        "bbox": bbox,
                        "y_center": y_center,
                        "x_center": x_center,
                        "y_frac": y_center / h,
                        "x_frac": x_center / w,
                        "width": max(p[0] for p in bbox) - min(p[0] for p in bbox),
                        "height": max(p[1] for p in bbox) - min(p[1] for p in bbox),
                    }
                )

    return {
        "model_name": model_name,
        "det_model": det_model,
        "rec_model": rec_model,
        "elapsed_seconds": round(elapsed, 2),
        "total_boxes": len(boxes),
        "boxes": boxes,
        "avg_confidence": round(np.mean([b["conf"] for b in boxes]), 4)
        if boxes
        else 0.0,
        "median_confidence": round(float(np.median([b["conf"] for b in boxes])), 4)
        if boxes
        else 0.0,
        "min_confidence": round(min(b["conf"] for b in boxes), 4) if boxes else 0.0,
        "max_confidence": round(max(b["conf"] for b in boxes), 4) if boxes else 0.0,
        "avg_box_width": round(np.mean([b["width"] for b in boxes]), 2)
        if boxes
        else 0.0,
        "avg_box_height": round(np.mean([b["height"] for b in boxes]), 2)
        if boxes
        else 0.0,
        "non_empty_boxes": len([b for b in boxes if b["text"].strip()]),
        "empty_boxes": len([b for b in boxes if not b["text"].strip()]),
        "high_conf_boxes": len([b for b in boxes if b["conf"] >= 0.7]),
        "medium_conf_boxes": len([b for b in boxes if 0.3 <= b["conf"] < 0.7]),
        "low_conf_boxes": len([b for b in boxes if b["conf"] < 0.3]),
    }


def analyze_zones(boxes: list, zones: dict[str, tuple[float, float]]) -> dict:
    """Analyze detection quality within predefined Y-axis zones."""
    zone_results = {}
    for zone_name, (y_start_frac, y_end_frac) in zones.items():
        zone_boxes = [b for b in boxes if y_start_frac <= b["y_frac"] <= y_end_frac]
        zone_results[zone_name] = {
            "box_count": len(zone_boxes),
            "non_empty": len([b for b in zone_boxes if b["text"].strip()]),
            "avg_confidence": round(np.mean([b["conf"] for b in zone_boxes]), 4)
            if zone_boxes
            else 0.0,
            "texts": [b["text"] for b in zone_boxes if b["text"].strip()],
        }
    return zone_results


def main():
    pdf_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "input/J.Flemming Timesheets - 012826-020326.pdf"
    )
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return

    logger.info(f"Loading {pdf_path.name} at {DPI} DPI ...")
    pil_images = convert_from_path(pdf_path, dpi=DPI)
    pil_img = pil_images[PAGE_INDEX]
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = image.shape[:2]
    logger.info(f"Page {PAGE_INDEX + 1}: {w}x{h} pixels")

    # Define zones for analysis (based on typical transposed timesheet layout)
    zones = {
        "header (0-20%)": (0.0, 0.20),
        "date_label (10-20%)": (0.10, 0.20),
        "body (20-85%)": (0.20, 0.85),
        "time_in_zone (85-95%)": (0.85, 0.95),
        "hours_zone (95-100%)": (0.95, 1.0),
    }

    results = {
        "benchmark_date": datetime.now().isoformat(),
        "source_file": pdf_path.name,
        "image_size": f"{w}x{h}",
        "dpi": DPI,
        "page_index": PAGE_INDEX,
        "models": {},
    }

    for model_name, (det_model, rec_model) in MODEL_PAIRS.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing {model_name.upper()} model: {det_model} + {rec_model}")
        logger.info(f"{'=' * 60}")

        metrics = run_ocr(model_name, det_model, rec_model, image)
        metrics["zone_analysis"] = analyze_zones(metrics["boxes"], zones)
        results["models"][model_name] = metrics

        logger.info(f"  Time: {metrics['elapsed_seconds']:.2f}s")
        logger.info(f"  Total boxes: {metrics['total_boxes']}")
        logger.info(f"  Non-empty boxes: {metrics['non_empty_boxes']}")
        logger.info(f"  Empty boxes: {metrics['empty_boxes']}")
        logger.info(f"  Avg confidence: {metrics['avg_confidence']:.4f}")
        logger.info(f"  High conf (>=0.7): {metrics['high_conf_boxes']}")
        logger.info(f"  Medium conf (0.3-0.7): {metrics['medium_conf_boxes']}")
        logger.info(f"  Low conf (<0.3): {metrics['low_conf_boxes']}")
        logger.info(
            f"  Avg box size: {metrics['avg_box_width']:.0f}x{metrics['avg_box_height']:.0f}px"
        )

        logger.info(f"\n  Zone analysis:")
        for zone_name, zone_data in metrics["zone_analysis"].items():
            logger.info(
                f"    {zone_name}: {zone_data['box_count']} boxes, "
                f"{zone_data['non_empty']} non-empty, "
                f"avg_conf={zone_data['avg_confidence']:.4f}"
            )
            if zone_data["texts"]:
                logger.info(f"      Texts: {zone_data['texts'][:5]}...")

    # Compute comparison
    mobile = results["models"]["mobile"]
    server = results["models"]["server"]

    comparison = {
        "speed_ratio": round(mobile["elapsed_seconds"] / server["elapsed_seconds"], 2)
        if server["elapsed_seconds"] > 0
        else 0,
        "box_count_diff": server["total_boxes"] - mobile["total_boxes"],
        "non_empty_diff": server["non_empty_boxes"] - mobile["non_empty_boxes"],
        "avg_confidence_diff": round(
            server["avg_confidence"] - mobile["avg_confidence"], 4
        ),
        "high_conf_diff": server["high_conf_boxes"] - mobile["high_conf_boxes"],
    }
    results["comparison"] = comparison

    logger.info(f"\n{'=' * 60}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Speed ratio (mobile/server): {comparison['speed_ratio']}x")
    logger.info(f"  Server detected {comparison['box_count_diff']:+d} more boxes")
    logger.info(
        f"  Server detected {comparison['non_empty_diff']:+d} more non-empty boxes"
    )
    logger.info(f"  Server avg confidence: {comparison['avg_confidence_diff']:+.4f}")
    logger.info(f"  Server high-conf boxes: {comparison['high_conf_diff']:+d}")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull results saved to: {json_path}")

    # Save human-readable report
    report_path = output_dir / "benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(f"PaddleOCR v3.4.0 Model Benchmark Report\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Date: {results['benchmark_date']}\n")
        f.write(f"Source: {results['source_file']}\n")
        f.write(f"Image: {results['image_size']}px at {results['dpi']} DPI\n\n")

        for model_name in ["mobile", "server"]:
            m = results["models"][model_name]
            f.write(f"\n{'─' * 40}\n")
            f.write(f"{model_name.upper()} MODEL\n")
            f.write(f"{'─' * 40}\n")
            f.write(f"  Detection: {m['det_model']}\n")
            f.write(f"  Recognition: {m['rec_model']}\n")
            f.write(f"  Time: {m['elapsed_seconds']:.2f}s\n")
            f.write(f"  Total boxes: {m['total_boxes']}\n")
            f.write(f"  Non-empty boxes: {m['non_empty_boxes']}\n")
            f.write(f"  Empty boxes: {m['empty_boxes']}\n")
            f.write(f"  Avg confidence: {m['avg_confidence']:.4f}\n")
            f.write(f"  Median confidence: {m['median_confidence']:.4f}\n")
            f.write(f"  High conf (>=0.7): {m['high_conf_boxes']}\n")
            f.write(f"  Medium conf (0.3-0.7): {m['medium_conf_boxes']}\n")
            f.write(f"  Low conf (<0.3): {m['low_conf_boxes']}\n")
            f.write(
                f"  Avg box size: {m['avg_box_width']:.0f}x{m['avg_box_height']:.0f}px\n\n"
            )

            f.write(f"  Zone Analysis:\n")
            for zone_name, zone_data in m["zone_analysis"].items():
                f.write(f"    {zone_name}:\n")
                f.write(
                    f"      Boxes: {zone_data['box_count']} (non-empty: {zone_data['non_empty']})\n"
                )
                f.write(f"      Avg confidence: {zone_data['avg_confidence']:.4f}\n")
                if zone_data["texts"]:
                    f.write(
                        f"      Detected texts: {', '.join(zone_data['texts'][:10])}\n"
                    )
                f.write(f"\n")

        f.write(f"\n{'─' * 40}\n")
        f.write(f"COMPARISON\n")
        f.write(f"{'─' * 40}\n")
        f.write(f"  Speed ratio (mobile/server): {comparison['speed_ratio']}x\n")
        f.write(f"  Server detected {comparison['box_count_diff']:+d} more boxes\n")
        f.write(
            f"  Server detected {comparison['non_empty_diff']:+d} more non-empty boxes\n"
        )
        f.write(f"  Server avg confidence: {comparison['avg_confidence_diff']:+.4f}\n")
        f.write(f"  Server high-conf boxes: {comparison['high_conf_diff']:+d}\n")

    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
