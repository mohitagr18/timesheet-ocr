"""Confidence scoring and routing logic for OCR results."""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AppConfig
    from .layout import GridCell
    from .ocr_engine import OcrBox

logger = logging.getLogger(__name__)


class Route(str, Enum):
    """Routing decision based on confidence score."""

    ACCEPT = "accept"      # Use PP-OCRv5 result as-is
    FALLBACK = "fallback"  # Send to Qwen2.5-VL for re-extraction
    REVIEW = "review"      # Route to human review queue


def route_by_confidence(confidence: float, config: AppConfig) -> Route:
    """Determine the routing action for a given confidence score.

    Args:
        confidence: OCR confidence score (0.0 to 1.0).
        config: Application config with threshold values.

    Returns:
        Route.ACCEPT if confidence >= accept_threshold
        Route.FALLBACK if confidence >= fallback_threshold
        Route.REVIEW if confidence < fallback_threshold
    """
    if confidence >= config.confidence.accept_threshold:
        return Route.ACCEPT
    elif confidence >= config.confidence.fallback_threshold:
        return Route.FALLBACK
    else:
        return Route.REVIEW


def aggregate_cell_confidence(boxes: list[OcrBox]) -> float:
    """Aggregate confidence for a cell from multiple OCR boxes.

    Uses the minimum confidence across all boxes in the cell,
    as the weakest link determines reliability.
    Returns 0.0 if no boxes are present.
    """
    if not boxes:
        return 0.0
    return min(box.confidence for box in boxes)


def boxes_in_zone(
    all_boxes: list[OcrBox],
    x_start: int,
    y_start: int,
    x_end: int,
    y_end: int,
) -> list[OcrBox]:
    """Filter OCR boxes that fall within a rectangular zone.

    A box is considered "in" the zone if its center falls within the zone boundaries.
    """
    result = []
    for box in all_boxes:
        cx, cy = box.x_center, box.y_center
        if x_start <= cx <= x_end and y_start <= cy <= y_end:
            result.append(box)
    return result


def should_fallback_entire_row(
    cell_confidences: dict[str, float],
    config: AppConfig,
    min_low_cells: int = 3,
) -> bool:
    """Determine if an entire row should be sent to VLM fallback.

    If >= min_low_cells cells in the row have confidence below the accept threshold,
    send the entire row instead of individual cells for better context.
    """
    low_count = sum(
        1 for conf in cell_confidences.values()
        if conf < config.confidence.accept_threshold
    )
    return low_count >= min_low_cells
