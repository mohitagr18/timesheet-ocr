"""Structured parser — converts raw OCR output into typed timesheet records."""

from __future__ import annotations

import logging
import re
from datetime import date, time
from typing import Optional

logger = logging.getLogger(__name__)


def parse_date(text: str) -> Optional[date]:
    """Parse a date string from OCR output.

    Handles common formats:
    - MM/DD/YYYY, MM-DD-YYYY
    - MM/DD/YY, MM-DD-YY
    - M/D/YYYY, M/D/YY
    """
    text = text.strip().replace(" ", "")

    # Normalize separators
    text = text.replace("-", "/").replace(".", "/")

    patterns = [
        # MM/DD/YYYY
        (r"(\d{1,2})/(\d{1,2})/(\d{4})", lambda m: (int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        # MM/DD/YY
        (r"(\d{1,2})/(\d{1,2})/(\d{2})$", lambda m: (int(m.group(1)), int(m.group(2)), 2000 + int(m.group(3)))),
    ]

    for pattern, extractor in patterns:
        match = re.match(pattern, text)
        if match:
            try:
                month, day, year = extractor(match)
                return date(year, month, day)
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
    - Various OCR artifacts (e.g., "9:OO" → "9:00")
    """
    text = text.strip().upper()

    # Common OCR substitutions
    text = text.replace("O", "0").replace("o", "0")
    text = text.replace("I", "1").replace("l", "1")
    text = text.replace("S", "5").replace("s", "5")
    text = text.replace("B", "8")

    # Remove spaces around the colon
    text = re.sub(r"\s*:\s*", ":", text)

    # Try 12-hour format with AM/PM
    match = re.match(r"(\d{1,2}):?(\d{2})\s*(AM|PM)", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        period = match.group(3)

        if period == "PM" and hour != 12:
            hour += 12
        elif period == "AM" and hour == 12:
            hour = 0

        try:
            return time(hour, minute)
        except ValueError:
            pass

    # Try 24-hour format HH:MM
    match = re.match(r"(\d{1,2}):(\d{2})$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        try:
            return time(hour, minute)
        except ValueError:
            pass

    # Try HHMM (no separator)
    match = re.match(r"(\d{2})(\d{2})$", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
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

    logger.warning(f"Could not parse hours: '{text}'")
    return None


def clean_name(text: str) -> str:
    """Clean up an employee or patient name from OCR output."""
    # Remove leading/trailing whitespace and normalize internal spaces
    text = re.sub(r"\s+", " ", text.strip())

    # Remove common prefixes/suffixes that aren't part of the name
    text = re.sub(r"^(name|employee|patient|client)\s*:?\s*", "", text, flags=re.IGNORECASE)

    return text.strip()
