"""Review queue — tracks fields and rows flagged for human inspection."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import ExtractionResult, ReviewItem, RowStatus, TimesheetRecord

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


def build_review_queue(result: ExtractionResult, config: AppConfig) -> list[ReviewItem]:
    """Build review items from all flagged rows across all records.

    Creates one ReviewItem per flagged field, with reason codes and
    any available fallback values from VLM re-extraction.
    """
    items: list[ReviewItem] = []

    for record in result.records:
        for row in record.rows:
            if row.status != RowStatus.FLAGGED:
                continue

            # Create review items for each validation error
            for error in row.validation_errors:
                field = _error_to_field(error)
                item = ReviewItem(
                    source_file=record.source_file,
                    page_number=record.page_number,
                    row_index=row.row_index,
                    field=field,
                    ocr_value=_get_field_value(row, field),
                    confidence=_get_field_confidence(row, field),
                    reason=error,
                )
                items.append(item)

    logger.info(f"Review queue: {len(items)} items from {result.flagged_count} flagged rows")
    return items


def _error_to_field(error: str) -> str:
    """Map a validation error code to its corresponding field name."""
    if "date" in error:
        return "date"
    elif "time_in" in error:
        return "time_in"
    elif "time_out" in error:
        return "time_out"
    elif "hours" in error or "shift" in error:
        return "total_hours"
    elif "duplicate" in error:
        return "row"
    elif "missing" in error:
        # Extract field name from "missing_X" errors
        parts = error.split("_", 1)
        return parts[1] if len(parts) > 1 else "unknown"
    return "unknown"


def _get_field_value(row, field: str) -> str:
    """Get the raw text value for a field."""
    field_map = {
        "date": row.date_text,
        "time_in": row.time_in_text,
        "time_out": row.time_out_text,
        "total_hours": row.total_hours_text,
        "notes": row.notes,
    }
    return field_map.get(field, "")


def _get_field_confidence(row, field: str) -> float:
    """Get the confidence score for a field."""
    conf_map = {
        "date": row.date_confidence,
        "time_in": row.time_in_confidence,
        "time_out": row.time_out_confidence,
        "total_hours": row.hours_confidence,
    }
    return conf_map.get(field, 0.0)
