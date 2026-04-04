"""Debug visualization — generates publication-quality images with overlaid bounding boxes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .ocr_engine import OcrBox
    from .layout import LayoutResult


@dataclass
class VlmFallbackCell:
    """Tracks a single cell that fell back to VLM."""

    row_idx: int
    field_name: str
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    vlm_text: str
    vlm_conf: float


# ── Color constants ──────────────────────────────────────────────────

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

CONF_THRESHOLD = 0.7


def render_page(
    image: np.ndarray,
    ocr_boxes: list["OcrBox"],
    layout: "LayoutResult",
    field_bands: dict[str, tuple[int, int]],
    vlm_fallbacks: list[VlmFallbackCell],
    page_number: int,
    source_file: str,
    output_dir: Path,
) -> Path:
    """Render annotated page image with all visualization layers.

    Returns path to saved PNG.
    """
    import cv2

    # Work on a copy
    viz = image.copy()
    if len(viz.shape) == 2:
        viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)

    # 1. Layout zones (blue dashed rectangles)
    _draw_zone(viz, layout.header_zone, BLUE, "HEADER")
    _draw_zone(viz, layout.table_zone, BLUE, "TABLE")
    _draw_zone(viz, layout.footer_zone, BLUE, "FOOTER")

    # 2. Column zones (orange dashed)
    for col_idx, col_zone in enumerate(layout.row_zones):
        _draw_dashed_rect(
            viz,
            col_zone.x_start,
            col_zone.y_start,
            col_zone.x_end,
            col_zone.y_end,
            ORANGE,
        )

    # 3. Field bands (purple dashed horizontal lines)
    h, w = viz.shape[:2]
    for band_name, (y_start, y_end) in field_bands.items():
        y_mid = (y_start + y_end) // 2
        _draw_dashed_line(viz, 0, y_mid, w, y_mid, PURPLE)
        label = f"{band_name} y={y_mid}"
        cv2.putText(
            viz,
            label,
            (5, y_mid - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            PURPLE,
            1,
        )

    # 3b. Header zone redaction (black rectangle to hide PHI)
    hx1 = int(layout.header_zone.x_start)
    hy1 = int(layout.header_zone.y_start)
    hx2 = int(layout.header_zone.x_end)
    hy2 = int(layout.header_zone.y_end)
    cv2.rectangle(viz, (hx1, hy1), (hx2, hy2), BLACK, -1)

    # 4. OCR boxes (green/red based on confidence)
    for box in ocr_boxes:
        color = GREEN if box.confidence > CONF_THRESHOLD else RED
        pts = np.array(box.bbox, dtype=np.int32)
        cv2.polylines(viz, [pts], True, color, 2)

        # Text label: truncate if too long
        text = box.text[:20]
        conf_str = f"{box.confidence:.2f}"
        label = f"{text} ({conf_str})"

        # Place text above the box
        min_y = int(min(p[1] for p in box.bbox))
        label_y = max(min_y - 5, 15)
        label_x = int(box.bbox[0][0])
        cv2.putText(
            viz,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )

    # 5. VLM fallback cells (yellow semi-transparent fill + text)
    for cell in vlm_fallbacks:
        # Semi-transparent yellow overlay
        overlay = viz.copy()
        cv2.rectangle(
            overlay,
            (cell.x_start, cell.y_start),
            (cell.x_end, cell.y_end),
            YELLOW,
            -1,
        )
        cv2.addWeighted(overlay, 0.25, viz, 0.75, 0, viz)

        # Yellow border
        cv2.rectangle(
            viz,
            (cell.x_start, cell.y_start),
            (cell.x_end, cell.y_end),
            YELLOW,
            2,
        )

        # VLM text overlay
        if cell.vlm_text:
            cx = cell.x_start + 5
            cy = cell.y_start + 15
            label = f"[VLM] {cell.field_name}: {cell.vlm_text}"
            # Truncate long text
            if len(label) > 40:
                label = label[:37] + "..."
            cv2.putText(
                viz,
                label,
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                WHITE,
                1,
            )

    # 6. Legend (top-right corner)
    _draw_legend(viz)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_file).stem
    filename = f"{stem}_page{page_number}.png"
    out_path = output_dir / filename
    cv2.imwrite(str(out_path), viz)
    logger.info(f"Debug visualization saved: {out_path}")
    return out_path


def _draw_zone(
    viz: np.ndarray,
    zone: object,
    color: tuple[int, int, int],
    label: str,
) -> None:
    """Draw a dashed rectangle for a layout zone."""
    import cv2

    x1 = int(zone.x_start)
    y1 = int(zone.y_start)
    x2 = int(zone.x_end)
    y2 = int(zone.y_end)

    _draw_dashed_rect(viz, x1, y1, x2, y2, color)

    # Label
    cv2.putText(
        viz,
        f"{label} ({x1},{y1})-({x2},{y2})",
        (x1 + 3, y1 + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
    )


def _draw_dashed_rect(
    viz: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    gap: int = 12,
) -> None:
    """Draw a dashed rectangle."""
    import cv2

    # Top edge
    _draw_dashed_line(viz, x1, y1, x2, y1, color, gap)
    # Bottom edge
    _draw_dashed_line(viz, x1, y2, x2, y2, color, gap)
    # Left edge
    _draw_dashed_line(viz, x1, y1, x1, y2, color, gap)
    # Right edge
    _draw_dashed_line(viz, x2, y1, x2, y2, color, gap)


def _draw_dashed_line(
    viz: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    gap: int = 12,
) -> None:
    """Draw a dashed line."""
    import cv2

    length = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if length == 0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length

    pos = 0.0
    while pos < length:
        end = min(pos + gap * 0.6, length)
        px1 = int(x1 + dx * pos)
        py1 = int(y1 + dy * pos)
        px2 = int(x1 + dx * end)
        py2 = int(y1 + dy * end)
        cv2.line(viz, (px1, py1), (px2, py2), color, 2)
        pos += gap


def _draw_legend(viz: np.ndarray) -> None:
    """Draw a color legend in the top-right corner."""
    import cv2

    h, w = viz.shape[:2]
    legend_x = w - 220
    legend_y = 10
    line_h = 22

    items = [
        ("OCR box (conf>0.7)", GREEN),
        ("OCR box (conf<=0.7)", RED),
        ("Layout zones", BLUE),
        ("Column zones", ORANGE),
        ("Field bands", PURPLE),
        ("VLM fallback cell", YELLOW),
    ]

    # Semi-transparent background
    overlay = viz.copy()
    bg_h = len(items) * line_h + 30
    cv2.rectangle(
        overlay, (legend_x - 10, legend_y - 5), (w - 10, legend_y + bg_h), BLACK, -1
    )
    cv2.addWeighted(overlay, 0.6, viz, 0.4, 0, viz)

    cv2.putText(
        viz,
        "LEGEND",
        (legend_x, legend_y + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        WHITE,
        1,
    )

    for i, (label, color) in enumerate(items):
        y = legend_y + 25 + i * line_h
        # Color swatch
        cv2.rectangle(viz, (legend_x, y - 10), (legend_x + 20, y + 5), color, -1)
        # Label
        cv2.putText(
            viz,
            label,
            (legend_x + 28, y + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            WHITE,
            1,
        )
