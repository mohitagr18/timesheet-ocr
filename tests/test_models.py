"""Tests for Pydantic data models."""

from datetime import date, time

import pytest

from timesheet_ocr.models import (
    CellResult,
    ExtractionResult,
    OcrSource,
    ReviewItem,
    RowStatus,
    TimesheetRecord,
    TimesheetRow,
)


class TestTimesheetRow:
    """Tests for TimesheetRow model."""

    def test_basic_row_creation(self):
        row = TimesheetRow(row_index=0)
        assert row.row_index == 0
        assert row.status == RowStatus.ACCEPTED
        assert row.validation_errors == []

    def test_min_confidence(self):
        row = TimesheetRow(
            row_index=0,
            date_confidence=0.95,
            time_in_confidence=0.80,
            time_out_confidence=0.90,
        )
        assert row.min_confidence == 0.80

    def test_calculated_hours_normal(self):
        row = TimesheetRow(
            row_index=0,
            time_in_parsed=time(9, 0),
            time_out_parsed=time(17, 0),
        )
        assert row.calculated_hours() == 8.0
        assert not row.is_overnight

    def test_calculated_hours_overnight(self):
        row = TimesheetRow(
            row_index=0,
            time_in_parsed=time(23, 0),
            time_out_parsed=time(7, 0),
        )
        assert row.calculated_hours() == 8.0
        assert row.is_overnight

    def test_calculated_hours_none_when_missing(self):
        row = TimesheetRow(row_index=0, time_in_parsed=time(9, 0))
        assert row.calculated_hours() is None

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            TimesheetRow(row_index=0, date_confidence=1.5)


class TestTimesheetRecord:
    """Tests for TimesheetRecord model."""

    def test_row_filtering(self):
        record = TimesheetRecord(
            source_file="test.png",
            rows=[
                TimesheetRow(row_index=0, status=RowStatus.ACCEPTED),
                TimesheetRow(row_index=1, status=RowStatus.FLAGGED),
                TimesheetRow(row_index=2, status=RowStatus.ACCEPTED),
                TimesheetRow(row_index=3, status=RowStatus.FAILED),
            ],
        )
        assert len(record.accepted_rows) == 2
        assert len(record.flagged_rows) == 1
        assert len(record.failed_rows) == 1


class TestExtractionResult:
    """Tests for ExtractionResult model."""

    def test_aggregate_counts(self):
        result = ExtractionResult(
            source_file="test.pdf",
            records=[
                TimesheetRecord(
                    source_file="test.pdf",
                    page_number=1,
                    rows=[
                        TimesheetRow(row_index=0, status=RowStatus.ACCEPTED),
                        TimesheetRow(row_index=1, status=RowStatus.FLAGGED),
                    ],
                ),
                TimesheetRecord(
                    source_file="test.pdf",
                    page_number=2,
                    rows=[
                        TimesheetRow(row_index=0, status=RowStatus.ACCEPTED),
                    ],
                ),
            ],
        )
        assert result.total_rows == 3
        assert result.accepted_count == 2
        assert result.flagged_count == 1
