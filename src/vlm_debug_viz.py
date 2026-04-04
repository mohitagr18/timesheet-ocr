"""VLM debug visualization — generates publication-quality images with extracted text annotations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .models import TimesheetRow, TimesheetRecord


def render_vlm_page(
    image: np.ndarray,
    records: list["TimesheetRecord"],
    page_number: int,
    source_file: str,
    output_dir: Path,
) -> Path:
    """Render annotated page image with VLM-extracted text annotations.

    Since VLM mode doesn't produce bounding boxes, this overlays extracted
    text at approximate positions with color-coded status indicators.

    Returns path to saved PNG.
    """
    import cv2

    viz = image.copy()
    if len(viz.shape) == 2:
        viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    h, w = viz.shape[:2]

    # 1. Header zone redaction (PHI protection) — black rectangle
    header_fraction = 0.16
    header_h = int(h * header_fraction)
    cv2.rectangle(viz, (0, 0), (w, header_h), (0, 0, 0), -1)

    # 2. Get rows for this page
    page_records = [r for r in records if r.page_number == page_number]
    if not page_records:
        out_path = _save(viz, source_file, page_number, output_dir)
        return out_path

    all_rows = []
    for rec in page_records:
        all_rows.extend(rec.rows)

    if not all_rows:
        out_path = _save(viz, source_file, page_number, output_dir)
        return out_path

    # 3. Draw row zones (evenly spaced across table area)
    table_y_start = header_h
    table_y_end = int(h * 0.98)
    table_h = table_y_end - table_y_start
    n_rows = len(all_rows)
    row_h = table_h // max(n_rows, 1)

    for i, row in enumerate(all_rows):
        y_top = table_y_start + i * row_h
        y_bottom = table_y_start + (i + 1) * row_h

        # Status color for row tint
        status = row.status.value if hasattr(row, "status") else "flagged"
        if status == "accepted":
            tint = (0, 255, 0)  # Green
        elif status == "flagged":
            tint = (0, 255, 255)  # Yellow
        else:
            tint = (0, 0, 255)  # Red

        # Semi-transparent tint
        overlay = viz.copy()
        cv2.rectangle(overlay, (0, y_top), (w, y_bottom), tint, -1)
        cv2.addWeighted(overlay, 0.10, viz, 0.90, 0, viz)

        # Row border
        cv2.rectangle(viz, (0, y_top), (w, y_bottom), (0, 255, 0), 2)

        # Row number label
        cv2.putText(
            viz,
            f"Row {i + 1}",
            (10, y_top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Extracted values overlay
        date_str = str(row.date_parsed) if row.date_parsed else row.date_text
        time_in_str = (
            row.time_in_parsed.strftime("%H:%M")
            if row.time_in_parsed
            else row.time_in_text
        )
        time_out_str = (
            row.time_out_parsed.strftime("%H:%M")
            if row.time_out_parsed
            else row.time_out_text
        )
        hours_str = (
            str(row.total_hours_parsed)
            if row.total_hours_parsed is not None
            else row.total_hours_text
        )

        # Column positions for text (spread across width)
        col_w = w // 4
        annotations = [
            (f"Date: {date_str or '—'}", col_w * 0 + 10),
            (f"Time In: {time_in_str or '—'}", col_w * 1 + 10),
            (f"Time Out: {time_out_str or '—'}", col_w * 2 + 10),
            (f"Hours: {hours_str or '—'}", col_w * 3 + 10),
        ]

        text_y = y_top + 35
        for text, x_pos in annotations:
            # Background for text readability
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(
                viz,
                (int(x_pos) - 2, text_y - th - 2),
                (int(x_pos) + tw + 2, text_y + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                viz,
                text,
                (int(x_pos), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    # 4. Legend
    _draw_legend(viz)

    out_path = _save(viz, source_file, page_number, output_dir)
    return out_path


def _save(
    viz: np.ndarray, source_file: str, page_number: int, output_dir: Path
) -> Path:
    """Save the visualization to disk."""
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_file).stem
    filename = f"vlm_{stem}_page{page_number}.png"
    out_path = output_dir / filename
    cv2.imwrite(str(out_path), viz)
    logger.info(f"VLM debug visualization saved: {out_path}")
    return out_path


def _draw_legend(viz: np.ndarray) -> None:
    """Draw a color legend in the top-right corner."""
    import cv2

    h, w = viz.shape[:2]
    legend_x = w - 200
    legend_y = 10
    line_h = 22

    items = [
        ("Accepted row", (0, 255, 0)),
        ("Flagged row", (0, 255, 255)),
        ("Failed row", (0, 0, 255)),
        ("PHI redacted", (0, 0, 0)),
    ]

    # Semi-transparent background
    overlay = viz.copy()
    bg_h = len(items) * line_h + 30
    cv2.rectangle(
        overlay,
        (legend_x - 10, legend_y - 5),
        (w - 10, legend_y + bg_h),
        (50, 50, 50),
        -1,
    )
    cv2.addWeighted(overlay, 0.7, viz, 0.3, 0, viz)

    cv2.putText(
        viz,
        "VLM MODE",
        (legend_x, legend_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    for i, (label, color) in enumerate(items):
        y = legend_y + 25 + i * line_h
        cv2.rectangle(viz, (legend_x, y - 10), (legend_x + 20, y + 5), color, -1)
        cv2.putText(
            viz,
            label,
            (legend_x + 28, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
        )
