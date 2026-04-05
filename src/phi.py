"""PHI/PII anonymization — consistent, deterministic redaction across all outputs."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class PhiAnonymizer:
    """Consistent PHI/PII anonymization across all outputs.

    Deterministic: filenames are sorted alphabetically and mapped to
    Patient_A, Patient_B, etc. so the same file always gets the same
    anonymized name across runs.
    """

    def __init__(self, filenames: list[str]) -> None:
        self._patient_map: dict[str, str] = {}
        self._employee_map: dict[str, str] = {}
        self._filename_map: dict[str, str] = {}
        self._employee_counter = 0
        self._known_employees: set[str] = set()

        # Build deterministic mappings from sorted filenames
        sorted_files = sorted(set(filenames))
        for idx, fname in enumerate(sorted_files):
            letter = chr(ord("A") + idx)
            patient_name = _extract_patient_name(fname)
            anon_patient = f"Patient_{letter}"
            anon_filename = f"patient_{letter.lower()}_week{idx + 1}.pdf"

            self._patient_map[patient_name] = anon_patient
            self._filename_map[fname] = anon_filename
            self._filename_map[Path(fname).stem] = (
                f"patient_{letter.lower()}_week{idx + 1}"
            )

        logger.info(
            f"PHI anonymizer initialized: {len(self._patient_map)} patients mapped"
        )

    def anonymize_patient(self, name: str) -> str:
        if not name:
            return name
        return self._patient_map.get(name, name)

    def anonymize_employee(self, name: str) -> str:
        if not name:
            return name
        if name not in self._employee_map:
            self._employee_counter += 1
            letter = chr(ord("A") + self._employee_counter - 1)
            self._employee_map[name] = f"Employee_{letter}"
        return self._employee_map[name]

    def anonymize_filename(self, filename: str) -> str:
        if not filename:
            return filename
        return self._filename_map.get(filename, filename)

    @staticmethod
    def is_signature_page(ocr_box_count: int, threshold: int) -> bool:
        """Determine if a page is a signature page based on OCR box count."""
        return ocr_box_count < threshold


def _extract_patient_name(filename: str) -> str:
    """Extract patient name from filename.

    Examples:
        "<patient_1> Timesheets - 010726-011326.pdf" → "<patient_1>"
        "<patient_2> Timesheets - 012826-020326.pdf" → "<patient_2>"
        "<patient_3> Timesheets 020426-021026.pdf" → "<patient_3>"
    """
    # Try "Timesheet" split first
    if "Timesheet" in filename:
        return filename.split("Timesheet")[0].strip(" -_")
    # Fallback: use the filename stem without dates
    stem = Path(filename).stem
    # Remove trailing date patterns like "020426-021026"
    stem = re.sub(r"\s*\d{6}-\d{6}$", "", stem)
    return stem.strip(" -_")
