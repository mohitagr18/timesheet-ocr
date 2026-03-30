"""Pydantic data models for timesheet extraction results."""

from __future__ import annotations

from datetime import date, datetime, time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RowStatus(str, Enum):
    """Status of an extracted timesheet row."""

    ACCEPTED = "accepted"
    FLAGGED = "flagged"
    FAILED = "failed"


class OcrSource(str, Enum):
    """Which OCR engine produced the value."""

    PPOCR = "ppocr"
    VLM = "vlm"
    MANUAL = "manual"


class CellResult(BaseModel):
    """Result of OCR for a single cell."""

    raw_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: OcrSource = OcrSource.PPOCR
    bbox: list[float] = Field(default_factory=list)


class TimesheetRow(BaseModel):
    """A single row extracted from a timesheet (one day/shift)."""

    row_index: int = Field(ge=0)
    date_text: str = ""
    date_parsed: Optional[date] = None
    time_in_text: str = ""
    time_in_parsed: Optional[time] = None
    time_out_text: str = ""
    time_out_parsed: Optional[time] = None
    total_hours_text: str = ""
    total_hours_parsed: Optional[float] = None
    notes: str = ""

    # Confidence scores per cell
    date_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    time_in_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    time_out_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hours_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Source tracking
    date_source: OcrSource = OcrSource.PPOCR
    time_in_source: OcrSource = OcrSource.PPOCR
    time_out_source: OcrSource = OcrSource.PPOCR
    hours_source: OcrSource = OcrSource.PPOCR

    # Validation
    status: RowStatus = RowStatus.ACCEPTED
    validation_errors: list[str] = Field(default_factory=list)
    is_overnight: bool = False

    @property
    def min_confidence(self) -> float:
        """Minimum confidence across all required cells."""
        scores = [self.date_confidence, self.time_in_confidence, self.time_out_confidence]
        return min(scores) if scores else 0.0

    def calculated_hours(self) -> Optional[float]:
        """Calculate hours from parsed time_in and time_out, supporting overnight shifts."""
        if self.time_in_parsed is None or self.time_out_parsed is None:
            return None

        dt_in = datetime.combine(date.today(), self.time_in_parsed)
        dt_out = datetime.combine(date.today(), self.time_out_parsed)

        # Overnight shift: time_out is before time_in → add 24 hours
        if dt_out <= dt_in:
            from datetime import timedelta
            dt_out += timedelta(days=1)
            self.is_overnight = True

        delta = dt_out - dt_in
        return round(delta.total_seconds() / 3600, 2)


class TimesheetRecord(BaseModel):
    """A complete timesheet document (one employee, one period)."""

    source_file: str
    page_number: int = 1
    employee_name: str = ""
    employee_name_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    patient_name: str = ""
    patient_name_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rows: list[TimesheetRow] = Field(default_factory=list)

    @property
    def accepted_rows(self) -> list[TimesheetRow]:
        return [r for r in self.rows if r.status == RowStatus.ACCEPTED]

    @property
    def flagged_rows(self) -> list[TimesheetRow]:
        return [r for r in self.rows if r.status == RowStatus.FLAGGED]

    @property
    def failed_rows(self) -> list[TimesheetRow]:
        return [r for r in self.rows if r.status == RowStatus.FAILED]


class ReviewItem(BaseModel):
    """An item flagged for human review."""

    source_file: str
    page_number: int = 1
    row_index: int
    field: str
    ocr_value: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    fallback_value: str = ""
    fallback_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Top-level result container for a full pipeline run."""

    source_file: str
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_time_seconds: float = 0.0
    total_pages: int = 0
    records: list[TimesheetRecord] = Field(default_factory=list)
    review_items: list[ReviewItem] = Field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return sum(len(r.rows) for r in self.records)

    @property
    def accepted_count(self) -> int:
        return sum(len(r.accepted_rows) for r in self.records)

    @property
    def flagged_count(self) -> int:
        return sum(len(r.flagged_rows) for r in self.records)

    @property
    def failed_count(self) -> int:
        return sum(len(r.failed_rows) for r in self.records)
