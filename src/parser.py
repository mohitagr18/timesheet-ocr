"""Structured parser — converts raw OCR output into typed timesheet records."""

from __future__ import annotations

import logging
import re
from datetime import date, time, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def parse_date(text: str, expected_year: int | None = None) -> Optional[date]:
    """Parse a date string from OCR output.

    Handles common formats:
    - MM/DD/YYYY, MM-DD-YYYY
    - MM/DD/YY, MM-DD-YY
    - MM/DD (no year — uses expected_year if available)

    If expected_year is provided:
    - 4-digit year that doesn't match → override with expected_year
    - 2-digit year → use expected_year's century
    - No year → apply expected_year
    """
    text = text.strip().replace(" ", "")

    # Normalize separators
    text = text.replace("-", "/").replace(".", "/")

    patterns = [
        # MM/DD/YYYY
        (
            r"(\d{1,2})/(\d{1,2})/(\d{4})",
            lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3))),
        ),
        # MM/DD/YY
        (
            r"(\d{1,2})/(\d{1,2})/(\d{2})$",
            lambda m: (int(m.group(1)), int(m.group(2)), 2000 + int(m.group(3))),
        ),
        # MM/DD (no year)
        (
            r"(\d{1,2})/(\d{1,2})$",
            lambda m: (int(m.group(1)), int(m.group(2)), None),
        ),
    ]

    for pattern, extractor in patterns:
        match = re.match(pattern, text)
        if match:
            try:
                month, day, year = extractor(match)
                if year is None and expected_year is not None:
                    year = expected_year
                elif (
                    year is not None
                    and expected_year is not None
                    and year != expected_year
                ):
                    # 4-digit year doesn't match — VLM hallucination likely
                    logger.info(
                        f"Date year mismatch: parsed={year}, expected={expected_year}. "
                        f"Using expected year. Original text: '{text}'"
                    )
                    year = expected_year
                if year is not None:
                    return date(year, month, day)
                else:
                    logger.warning(
                        f"Could not parse date (no year available): '{text}'"
                    )
                    return None
            except ValueError:
                continue

    logger.warning(f"Could not parse date: '{text}'")
    return None


def parse_time(text: str) -> Optional[time]:
    """Parse a time string from OCR output.

    Handles:
    - HH:MM, H:MM
    - HH:MM AM/PM, H:MM AM/PM
    - HHMM (no separator)
    - Various OCR artifacts (e.g., "9:OO" → "9:00", "Q:00" → "2:00")
    """
    text = text.strip().upper()

    # Common OCR substitutions
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace("S", "5").replace("s", "5")
    text = text.replace("B", "8")
    text = text.replace("Q", "2")
    text = text.replace("Z", "2")
    text = text.replace("?", "2")

    # Fix dots between digits (e.g., 5.30 -> 5:30)
    text = re.sub(r"(\d)\.(\d)", r"\1:\2", text)

    # Remove spaces around the colon
    text = re.sub(r"\s*:\s*", ":", text)

    # Often OCR reads "AM" or "PM" as "0M", ".0M", "CM"
    text = re.sub(r"[0\.C]M\b", "PM", text)  # best guess fallback

    # Handle standalone 'P' or 'A' at the end like '4:30P' or '4:30 P'
    text = re.sub(r"([0-9])\s*P$", r"\1 PM", text)
    text = re.sub(r"([0-9])\s*A$", r"\1 AM", text)

    # 1. Try 12-hour format with AM/PM (or A/P)
    # Allows optional colon or space separator. The minute group is optional.
    match = re.search(r"\b(\d{1,2})[: ]?(\d{2})?\s*([AP]\.?M?\.?)", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        period = match.group(3).replace(".", "")

        if period.startswith("P") and hour != 12:
            hour += 12
        elif period.startswith("A") and hour == 12:
            hour = 0

        try:
            return time(hour, minute)
        except ValueError:
            pass

    # 2. Try 24-hour format HH:MM or H:MM anywhere in the string
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        try:
            return time(hour, minute)
        except ValueError:
            pass

    # 3. Try HHMM or HMM (3 or 4 digits at the START, ignore trailing junk)
    # Be careful not to match random digits; ensure it matches time format
    match = re.match(r"(\d{1,2})(\d{2})(?!\d)", text.strip())
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))

        # OCR artifact: if minute is 80-89, it frequently is a misread "30" or "00".
        if 80 <= minute <= 89:
            minute = 30  # "8" is usually "3" in cursive OCR

        try:
            return time(hour, minute)
        except ValueError:
            pass

    logger.warning(f"Could not parse time: '{text}'")
    return None


def parse_hours(text: str) -> Optional[float]:
    """Parse a total-hours value from OCR output.

    Handles:
    - Decimal: "3.5", "7.25"
    - Fraction-like: "3 1/2" → 3.5
    - Integer: "8"
    - With unit suffixes: "8hrs", "8hr", "12 hrs", "4 hours"
    - Hours+minutes: "8hr 15min", "15h 30m", "8hr 05min"
    - H:MM as hours:minutes: "8:15" → 8.25 (when used as total_hours)
    """
    text = text.strip()

    if not text:
        return None

    # Common OCR substitutions
    text = text.replace("O", "0").replace("o", "0")

    # Try decimal
    match = re.match(r"(\d+\.?\d*)$", text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try "N 1/2" format
    match = re.match(r"(\d+)\s+1/2$", text)
    if match:
        return float(match.group(1)) + 0.5

    # Try "N 1/4" format
    match = re.match(r"(\d+)\s+1/4$", text)
    if match:
        return float(match.group(1)) + 0.25

    # Try "N 3/4" format
    match = re.match(r"(\d+)\s+3/4$", text)
    if match:
        return float(match.group(1)) + 0.75

    # Try hours+minutes: "8hr 15min", "15h 30m", "8hr 05min", "6hr 55min"
    match = re.match(
        r"(\d+\.?\d*)\s*(?:hr|h)(?:s|R)?\.?\s+(\d+)\s*(?:min|m)(?:s|utes)?\.?$",
        text,
        re.IGNORECASE,
    )
    if match:
        h = float(match.group(1))
        m = int(match.group(2))
        return round(h + m / 60.0, 2)

    # Try with unit suffix: "8hrs", "8hr", "12 hrs", "4 hours", "7hR"
    match = re.match(
        r"(\d+\.?\d*)\s*(?:hrs?|h0?urs?|hR)\.?$", text, re.IGNORECASE
    )
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try "N HRS." with period
    match = re.match(r"(\d+\.?\d*)\s*HRS?\.?$", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    # Try H:MM as hours:minutes (e.g., "8:15" → 8.25)
    match = re.match(r"(\d{1,2}):(\d{2})(?:\s*(?:min)?)?$", text)
    if match:
        h = int(match.group(1))
        m = int(match.group(2))
        if h <= 24 and m < 60:
            return round(h + m / 60.0, 2)

    # Try "Nh Mm" shorthand: "15h 30m"
    match = re.match(r"(\d+)\s*h\s+(\d+)\s*m$", text, re.IGNORECASE)
    if match:
        h = int(match.group(1))
        m = int(match.group(2))
        return round(h + m / 60.0, 2)

    logger.warning(f"Could not parse hours: '{text}'")
    return None


def _has_period_marker(text: str) -> bool:
    """Check if a time string contains an AM/PM marker."""
    return bool(re.search(r"[AP]\.?M\.?", text, re.IGNORECASE))


def _try_parse_with_period(text: str, period: str) -> Optional[time]:
    """Parse a time string with an explicit AM or PM suffix."""
    if not text.strip():
        return None
    text = text.strip().upper()
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace("S", "5").replace("s", "5")
    text = text.replace("B", "8")
    text = text.replace("Q", "2")
    text = text.replace("Z", "2")
    text = text.replace("?", "2")
    text = re.sub(r"(\d)\.(\d)", r"\1:\2", text)
    text = re.sub(r"\s*:\s*", ":", text)

    match = re.search(r"(\d{1,2}):?(\d{2})", text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))

    if period.upper().startswith("P") and hour != 12:
        hour += 12
    elif period.upper().startswith("A") and hour == 12:
        hour = 0

    try:
        return time(hour, minute)
    except ValueError:
        return None


def _calc_hours(t_in: Optional[time], t_out: Optional[time]) -> Optional[float]:
    """Calculate hours between two times, supporting overnight shifts."""
    if t_in is None or t_out is None:
        return None
    from datetime import datetime, timedelta

    dt_in = datetime.combine(date.today(), t_in)
    dt_out = datetime.combine(date.today(), t_out)
    if dt_out <= dt_in:
        dt_out += timedelta(days=1)
    delta = dt_out - dt_in
    return round(delta.total_seconds() / 3600, 2)


def disambiguate_times(
    time_in_text: str,
    time_out_text: str,
    total_hours_text: str,
) -> tuple[Optional[time], Optional[time]]:
    """Parse time_in and time_out, using total_hours to resolve AM/PM ambiguity.

    When the VLM returns times without AM/PM markers (e.g., "8:00" and "4:30"),
    this function tries all AM/PM combinations and picks the one whose calculated
    hours is closest to the written total_hours.

    Returns (time_in_parsed, time_out_parsed).
    """
    time_in_parsed = parse_time(time_in_text)
    time_out_parsed = parse_time(time_out_text)

    has_in_period = _has_period_marker(time_in_text)
    has_out_period = _has_period_marker(time_out_text)

    if has_in_period or has_out_period:
        return time_in_parsed, time_out_parsed

    total_hours = parse_hours(total_hours_text)
    if total_hours is None or total_hours <= 0:
        return time_in_parsed, time_out_parsed

    in_bare = time_in_text.strip()
    out_bare = time_out_text.strip()

    if not in_bare or not out_bare:
        return time_in_parsed, time_out_parsed

    periods = ["AM", "PM"]
    best_in = time_in_parsed
    best_out = time_out_parsed
    best_diff = (
        abs(_calc_hours(time_in_parsed, time_out_parsed) - total_hours)
        if time_in_parsed and time_out_parsed
        else float("inf")
    )

    for in_p in periods:
        for out_p in periods:
            candidate_in = _try_parse_with_period(in_bare, in_p)
            candidate_out = _try_parse_with_period(out_bare, out_p)
            if candidate_in is None or candidate_out is None:
                continue
            calc = _calc_hours(candidate_in, candidate_out)
            if calc is None:
                continue
            diff = abs(calc - total_hours)
            if diff < best_diff:
                best_diff = diff
                best_in = candidate_in
                best_out = candidate_out
            elif diff == best_diff and diff == 0:
                if (in_p == "AM" and out_p == "PM") and (
                    best_in != candidate_in or best_out != candidate_out
                ):
                    best_in = candidate_in
                    best_out = candidate_out

    return best_in, best_out


def clean_name(text: str) -> str:
    """Clean up an employee or patient name from OCR output."""
    # Remove leading/trailing whitespace and normalize internal spaces
    text = re.sub(r"\s+", " ", text.strip())

    # Remove common prefixes/suffixes that aren't part of the name
    text = re.sub(
        r"^(name|employee|patient|client)\s*:?\s*", "", text, flags=re.IGNORECASE
    )

    return text.strip()


def extract_week_dates(
    filename: str, week_start_day: int = 2, week_length: int = 7
) -> list[date] | None:
    """Parse filename date range and generate consecutive week dates.

    Expected filename pattern: 'Name - MMDDYY-MMDDYY.pdf'
    Example: '<patient_2> Timesheets - 012826-020326.pdf'
        → start=2026-01-28, end=2026-02-03
        → returns [2026-01-28, 2026-01-29, ..., 2026-02-03]

    Args:
        filename: Source filename (e.g., '<patient_2> Timesheets - 012826-020326.pdf')
        week_start_day: ISO weekday for expected start (0=Mon, 2=Wed, 6=Sun)
        week_length: Number of consecutive days to generate

    Returns:
        List of `week_length` consecutive dates, or None if pattern not found.
    """
    match = re.search(r"(\d{2})(\d{2})(\d{2})[-–](\d{2})(\d{2})(\d{2})", filename)
    if not match:
        return None

    start_month = int(match.group(1))
    start_day = int(match.group(2))
    start_year = 2000 + int(match.group(3))

    try:
        start_date = date(start_year, start_month, start_day)
    except ValueError:
        return None

    # Warn if start day doesn't match expected week_start_day
    # Python isoweekday(): Mon=1, Tue=2, Wed=3, ..., Sun=7
    # Config uses 0=Mon, 1=Tue, 2=Wed, ..., 6=Sun
    actual_weekday = start_date.weekday()  # Mon=0, ..., Sun=6
    if actual_weekday != week_start_day:
        logger.warning(
            f"Filename start date {start_date} is weekday {actual_weekday + 1} "
            f"(expected {week_start_day + 1}). Proceeding anyway."
        )

    return [start_date + timedelta(days=i) for i in range(week_length)]


def extract_expected_year(filename: str) -> int | None:
    """Extract the expected year from a timesheet filename.

    Handles common patterns:
    - "<patient_2> Timesheets - 012826-020326.pdf" → 2026
    - "<patient_1> Timesheets - 010726-011326.pdf" → 2026
    - "Timesheet_2026-01-28.pdf" → 2026
    - "timesheet_01-28-2026.pdf" → 2026
    """
    # Pattern 1: MMDDYY-MMDDYY or MMDDYYYY-MMDDYYYY in filename
    match = re.search(r"(\d{2})(\d{2})(\d{2,4})[-–](\d{2})(\d{2})(\d{2,4})", filename)
    if match:
        year_str = match.group(3)
        if len(year_str) == 2:
            return 2000 + int(year_str)
        return int(year_str)

    # Pattern 2: YYYY-MM-DD or YYYYMMDD
    match = re.search(r"(\d{4})[-_](\d{1,2})[-_](\d{1,2})", filename)
    if match:
        return int(match.group(1))

    # Pattern 3: MM-DD-YYYY or MM/DD/YYYY
    match = re.search(r"(\d{1,2})[-_/](\d{1,2})[-_/](\d{4})", filename)
    if match:
        return int(match.group(3))

    # Pattern 4: Just a 4-digit year anywhere
    match = re.search(r"\b(20\d{2})\b", filename)
    if match:
        return int(match.group(1))

    return None
