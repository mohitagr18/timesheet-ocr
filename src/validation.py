"""Validation engine — business rules for extracted timesheet data."""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

from .models import RowStatus, TimesheetRecord, TimesheetRow

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


def validate_record(record: TimesheetRecord, config: AppConfig) -> list[str]:
    """Validate an entire timesheet record. Returns list of warnings."""
    warnings = []

    # Validate each row
    for row in record.rows:
        row_warnings = validate_row(row, config)
        warnings.extend(row_warnings)

    # Check for duplicate rows
    dup_warnings = _check_duplicates(record)
    warnings.extend(dup_warnings)

    # Check for daily 24h limit
    limit_warnings = _check_daily_hours(record)
    warnings.extend(limit_warnings)

    return warnings


def validate_row(row: TimesheetRow, config: AppConfig) -> list[str]:
    """Validate a single timesheet row. Updates row.status and row.validation_errors."""
    errors = []
    val_cfg = config.validation

    # 1. Required fields check
    if not row.date_text and row.date_parsed is None:
        errors.append("missing_date")
    if not row.time_in_text and row.time_in_parsed is None:
        errors.append("missing_time_in")
    if not row.time_out_text and row.time_out_parsed is None:
        errors.append("missing_time_out")

    # 2. Date validation
    if row.date_parsed is not None:
        today = date.today()

        if not val_cfg.allow_future_dates and row.date_parsed > today:
            errors.append("date_in_future")

        days_ago = (today - row.date_parsed).days
        if days_ago > val_cfg.max_days_in_past:
            errors.append("date_too_old")

    elif row.date_text:
        errors.append("date_not_parseable")

    # 3. Time validation
    if row.time_in_parsed is not None and row.time_out_parsed is not None:
        # Calculate actual hours
        calculated = row.calculated_hours()

        if calculated is not None:
            # Check for unreasonably long shift
            if calculated > val_cfg.max_shift_hours:
                errors.append(f"shift_too_long_{calculated:.1f}h")

            # Cross-check with written total hours
            if row.total_hours_parsed is not None:
                diff = abs(calculated - row.total_hours_parsed)
                if diff > val_cfg.hours_mismatch_tolerance:
                    errors.append(
                        f"hours_mismatch_calc={calculated:.2f}_written={row.total_hours_parsed:.2f}"
                    )

    elif row.time_in_text and not row.time_in_parsed:
        errors.append("time_in_not_parseable")

    if row.time_out_text and not row.time_out_parsed:
        errors.append("time_out_not_parseable")

    # Update row status
    if errors:
        row.validation_errors = errors
        row.status = RowStatus.FLAGGED
        logger.info(f"Row {row.row_index} flagged: {errors}")
    else:
        row.status = RowStatus.ACCEPTED

    return [f"row_{row.row_index}: {e}" for e in errors]


def _check_duplicates(record: TimesheetRecord) -> list[str]:
    """Flag duplicate rows (same date + time_in)."""
    warnings = []
    seen: dict[tuple, int] = {}

    for row in record.rows:
        if row.date_parsed and row.time_in_parsed:
            key = (row.date_parsed, row.time_in_parsed)
            if key in seen:
                dup_msg = f"duplicate_row_{row.row_index}_matches_row_{seen[key]}"
                warnings.append(dup_msg)
                row.validation_errors.append("duplicate_row")
                row.status = RowStatus.FLAGGED
            else:
                seen[key] = row.row_index

    return warnings


def _check_daily_hours(record: TimesheetRecord) -> list[str]:
    """Flag rows if daily total hours exceed 24.0."""
    warnings = []
    daily_sums: dict[date, float] = {}

    for row in record.rows:
        calculated = row.calculated_hours()
        if row.date_parsed and calculated is not None:
            current_sum = daily_sums.get(row.date_parsed, 0.0)
            if current_sum + calculated > 24.0:
                row.is_over_24h_limit = True
                row.status = RowStatus.FLAGGED
                row.validation_errors.append("duplicate_shift_exceeds_24h")
                warnings.append(f"row_{row.row_index}: duplicate_shift_exceeds_24h sum={current_sum + calculated}")
            else:
                daily_sums[row.date_parsed] = current_sum + calculated

    return warnings
