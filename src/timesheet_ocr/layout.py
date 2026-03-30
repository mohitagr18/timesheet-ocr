"""Layout detection — zone extraction and grid row/column mapping."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """A rectangular region of an image."""

    x_start: int
    y_start: int
    x_end: int
    y_end: int
    label: str = ""

    @property
    def width(self) -> int:
        return self.x_end - self.x_start

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop this zone from an image."""
        return image[self.y_start : self.y_end, self.x_start : self.x_end]


@dataclass
class GridCell:
    """A single cell in the timesheet grid."""

    row_idx: int
    col_name: str
    zone: Zone


@dataclass
class LayoutResult:
    """Result of layout detection for one timesheet page."""

    header_zone: Zone
    table_zone: Zone
    footer_zone: Zone
    row_zones: list[Zone] = field(default_factory=list)
    grid_cells: list[GridCell] = field(default_factory=list)
    image_height: int = 0
    image_width: int = 0


def detect_layout(image: np.ndarray, config: AppConfig) -> LayoutResult:
    """Detect the layout zones and grid structure of a timesheet image.

    Uses the fractional zone coordinates from config to define header, table,
    and footer zones. Then detects horizontal lines within the table zone to
    identify individual rows, and maps columns by x-coordinate ranges.
    """
    h, w = image.shape[:2]
    layout_cfg = config.layout

    # Convert fractional coordinates to pixel coordinates
    header_zone = _frac_to_zone(layout_cfg.header_zone, w, h, "header")
    table_zone = _frac_to_zone(layout_cfg.table_zone, w, h, "table")
    footer_zone = _frac_to_zone(layout_cfg.footer_zone, w, h, "footer")

    logger.info(f"Layout zones — header: {header_zone.height}px, table: {table_zone.height}px, footer: {footer_zone.height}px")

    # Detect rows or columns within the table zone
    table_crop = table_zone.crop(image)
    row_zones = []
    
    if layout_cfg.transposed:
        col_boundaries = _detect_col_boundaries(table_crop)
        for i, (x_left, x_right) in enumerate(col_boundaries):
            row_zone = Zone(
                x_start=table_zone.x_start + x_left,
                y_start=table_zone.y_start,
                x_end=table_zone.x_start + x_right,
                y_end=table_zone.y_end,
                label=f"shift_col_{i}"
            )
            row_zones.append(row_zone)
        logger.info(f"Detected {len(row_zones)} transposed table shifts (columns)")
    else:
        row_boundaries = _detect_row_boundaries(table_crop)
        for i, (y_top, y_bottom) in enumerate(row_boundaries):
            row_zone = Zone(
                x_start=table_zone.x_start,
                y_start=table_zone.y_start + y_top,
                x_end=table_zone.x_end,
                y_end=table_zone.y_start + y_bottom,
                label=f"row_{i}",
            )
            row_zones.append(row_zone)
        logger.info(f"Detected {len(row_zones)} table rows")

    # Build grid cells (row × column)
    columns = layout_cfg.columns
    col_map = {
        "date": columns.date,
        "time_in": columns.time_in,
        "time_out": columns.time_out,
        "total_hours": columns.total_hours,
        "notes": columns.notes,
    }

    grid_cells = []
    for row_idx, row_zone in enumerate(row_zones):
        for col_name, (frac_start, frac_end) in col_map.items():
            if layout_cfg.transposed:
                cell = GridCell(
                    row_idx=row_idx,
                    col_name=col_name,
                    zone=Zone(
                        x_start=row_zone.x_start,
                        y_start=int(h * frac_start),
                        x_end=row_zone.x_end,
                        y_end=int(h * frac_end),
                        label=f"shift_{row_idx}_{col_name}",
                    ),
                )
            else:
                cell = GridCell(
                    row_idx=row_idx,
                    col_name=col_name,
                    zone=Zone(
                        x_start=int(w * frac_start),
                        y_start=row_zone.y_start,
                        x_end=int(w * frac_end),
                        y_end=row_zone.y_end,
                        label=f"row_{row_idx}_{col_name}",
                    ),
                )
            grid_cells.append(cell)

    return LayoutResult(
        header_zone=header_zone,
        table_zone=table_zone,
        footer_zone=footer_zone,
        row_zones=row_zones,
        grid_cells=grid_cells,
        image_height=h,
        image_width=w,
    )


def _frac_to_zone(fracs: list[float], width: int, height: int, label: str = "") -> Zone:
    """Convert fractional [x_start, y_start, x_end, y_end] to pixel Zone."""
    return Zone(
        x_start=int(width * fracs[0]),
        y_start=int(height * fracs[1]),
        x_end=int(width * fracs[2]),
        y_end=int(height * fracs[3]),
        label=label,
    )


def _detect_row_boundaries(table_image: np.ndarray) -> list[tuple[int, int]]:
    """Detect horizontal row boundaries in the table zone.

    Uses horizontal morphological operations and contour detection to find
    row separators, then returns (y_top, y_bottom) pairs for each row.
    Falls back to evenly-spaced row slicing if line detection fails.
    """
    h, w = table_image.shape[:2]

    # Ensure grayscale
    if len(table_image.shape) == 3:
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = table_image.copy()

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal lines using morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Find the y-coordinates of horizontal lines
    line_y_positions = []
    row_sums = np.sum(horizontal_lines, axis=1)
    threshold = w * 128  # At least half the width should be filled

    in_line = False
    line_start = 0
    for y in range(h):
        if row_sums[y] > threshold:
            if not in_line:
                line_start = y
                in_line = True
        else:
            if in_line:
                line_y_positions.append((line_start + y) // 2)
                in_line = False

    # Build rows from line boundaries
    if len(line_y_positions) >= 2:
        rows = []
        for i in range(len(line_y_positions) - 1):
            y_top = line_y_positions[i]
            y_bottom = line_y_positions[i + 1]
            if y_bottom - y_top > 10:  # Skip tiny rows
                rows.append((y_top, y_bottom))
        if rows:
            logger.info(f"Detected {len(rows)} rows from horizontal lines")
            return rows

    # Fallback: evenly-spaced rows (assume ~14 rows per table, typical for timesheets)
    n_rows = 14
    row_height = h // n_rows
    rows = []
    for i in range(n_rows):
        y_top = i * row_height
        y_bottom = min((i + 1) * row_height, h)
        rows.append((y_top, y_bottom))

    logger.info(f"Fallback: using {n_rows} evenly-spaced rows")
    return rows


def _detect_col_boundaries(table_image: np.ndarray) -> list[tuple[int, int]]:
    """Detect vertical column boundaries in the table zone."""
    h, w = table_image.shape[:2]

    # Ensure grayscale
    if len(table_image.shape) == 3:
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = table_image.copy()

    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect vertical lines using morphological operations
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Find the x-coordinates of vertical lines
    line_x_positions = []
    col_sums = np.sum(vertical_lines, axis=0) # sum over Y
    threshold = h * 128  # At least half the height should be filled

    in_line = False
    line_start = 0
    for x in range(w):
        if col_sums[x] > threshold:
            if not in_line:
                line_start = x
                in_line = True
        else:
            if in_line:
                line_x_positions.append((line_start + x) // 2)
                in_line = False

    # Build cols from line boundaries
    if len(line_x_positions) >= 2:
        cols = []
        for i in range(len(line_x_positions) - 1):
            x_left = line_x_positions[i]
            x_right = line_x_positions[i + 1]
            if x_right - x_left > 10:  # Skip tiny cols
                cols.append((x_left, x_right))
        if cols:
            logger.info(f"Detected {len(cols)} columns from vertical lines")
            return cols

    # Fallback: evenly-spaced columns
    n_cols = 7 # 7 days a week
    col_width = w // n_cols
    cols = []
    for i in range(n_cols):
        x_left = i * col_width
        x_right = min((i + 1) * col_width, w)
        cols.append((x_left, x_right))

    logger.info(f"Fallback: using {n_cols} evenly-spaced columns")
    return cols
