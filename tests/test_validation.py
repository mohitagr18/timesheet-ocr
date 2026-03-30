"""Tests for validation rules."""

from datetime import date, time, timedelta

import pytest

from config import load_config
from models import RowStatus, TimesheetRecord, TimesheetRow
from validation import validate_record, validate_row


@pytest.fixture
def config():
    """Load default config for testing."""
    return load_config()


class TestValidateRow:
    def test_valid_row(self, config):
        row = TimesheetRow(
            row_index=0,
            date_text="03/15/2026",
            date_parsed=date(2026, 3, 15),
            time_in_text="09:00",
            time_in_parsed=time(9, 0),
            time_out_text="17:00",
            time_out_parsed=time(17, 0),
            total_hours_text="8",
            total_hours_parsed=8.0,
        )
        errors = validate_row(row, config)
        assert errors == []
        assert row.status == RowStatus.ACCEPTED

    def test_missing_date(self, config):
        row = TimesheetRow(
            row_index=0,
            time_in_text="09:00",
            time_in_parsed=time(9, 0),
            time_out_text="17:00",
            time_out_parsed=time(17, 0),
        )
        errors = validate_row(row, config)
        assert any("missing_date" in e for e in errors)
        assert row.status == RowStatus.FLAGGED

    def test_future_date(self, config):
        future = date.today() + timedelta(days=30)
        row = TimesheetRow(
            row_index=0,
            date_text=future.strftime("%m/%d/%Y"),
            date_parsed=future,
            time_in_text="09:00",
            time_in_parsed=time(9, 0),
            time_out_text="17:00",
            time_out_parsed=time(17, 0),
        )
        errors = validate_row(row, config)
        assert any("date_in_future" in e for e in errors)

    def test_shift_too_long(self, config):
        row = TimesheetRow(
            row_index=0,
            date_text="03/15/2026",
            date_parsed=date(2026, 3, 15),
            time_in_text="00:00",
            time_in_parsed=time(0, 0),
            time_out_text="23:00",
            time_out_parsed=time(23, 0),
        )
        errors = validate_row(row, config)
        assert any("shift_too_long" in e for e in errors)

    def test_hours_mismatch(self, config):
        row = TimesheetRow(
            row_index=0,
            date_text="03/15/2026",
            date_parsed=date(2026, 3, 15),
            time_in_text="09:00",
            time_in_parsed=time(9, 0),
            time_out_text="17:00",
            time_out_parsed=time(17, 0),
            total_hours_text="6",
            total_hours_parsed=6.0,  # Should be 8.0
        )
        errors = validate_row(row, config)
        assert any("hours_mismatch" in e for e in errors)


class TestDuplicateDetection:
    def test_duplicate_rows_flagged(self, config):
        record = TimesheetRecord(
            source_file="test.png",
            rows=[
                TimesheetRow(
                    row_index=0,
                    date_parsed=date(2026, 3, 15),
                    time_in_parsed=time(9, 0),
                    time_out_parsed=time(17, 0),
                ),
                TimesheetRow(
                    row_index=1,
                    date_parsed=date(2026, 3, 15),
                    time_in_parsed=time(9, 0),  # Duplicate!
                    time_out_parsed=time(17, 0),
                ),
            ],
        )
        warnings = validate_record(record, config)
        assert any("duplicate" in w for w in warnings)
