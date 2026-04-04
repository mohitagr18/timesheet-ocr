"""Qwen2.5-VL fallback — Vision-LLM re-extraction for low-confidence cells via Ollama."""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


class VlmFallback:
    """Qwen2.5-VL 7B fallback via Ollama for re-extracting low-confidence cells.

    Used only when PP-OCRv5 confidence is below the fallback threshold.
    Sends cropped cell/row images with targeted prompts to get structured JSON output.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = None
        self._available: Optional[bool] = None
        # Metrics tracking
        self._total_calls: int = 0
        self._empty_responses: int = 0
        self._json_parse_failures: int = 0
        self._total_fields_queried: int = 0

    def _ensure_client(self) -> bool:
        """Initialize Ollama client and check availability."""
        if self._available is not None:
            return self._available

        try:
            import ollama

            self._client = ollama.Client(host=self.config.ollama.host)
            # Quick availability check
            self._client.list()
            self._available = True
            logger.info(f"Ollama connected at {self.config.ollama.host}")
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. VLM fallback disabled.")
            self._available = False
            return False

    def extract_cell_value(
        self, image: np.ndarray, field_name: str, expected_year: int | None = None
    ) -> tuple[str, float]:
        """Extract a single cell value from a cropped image.

        Args:
            image: Cropped image of the cell (numpy array, BGR or grayscale).
            field_name: Name of the field (e.g., "date", "time_in").
            expected_year: The expected year for this timesheet (from filename).

        Returns:
            (extracted_value, confidence) — confidence is estimated from the VLM response.
        """
        if not self._ensure_client():
            return ("", 0.0)

        self._total_calls += 1
        self._total_fields_queried += 1

        prompt = self._build_cell_prompt(field_name, expected_year)
        img_b64 = self._image_to_base64(image)

        try:
            response = self._client.chat(
                model=self.config.ollama.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                options={"temperature": 0.1},
            )

            reply = response["message"]["content"].strip()
            logger.debug(f"VLM cell response for {field_name}: {reply}")
            value, confidence = self._parse_cell_response(reply, field_name)
            if not value:
                self._empty_responses += 1
            return (value, confidence)

        except Exception as e:
            logger.error(f"VLM extraction failed for {field_name}: {e}")
            self._empty_responses += 1
            return ("", 0.0)

    def extract_row(
        self, image: np.ndarray, expected_year: int | None = None
    ) -> dict[str, str]:
        """Extract all fields from a full row image.

        Args:
            image: Full row image.
            expected_year: The expected year for this timesheet (from filename).

        Returns:
            A dict mapping field names to extracted values.
        """
        if not self._ensure_client():
            return {}

        prompt = self._build_row_prompt(expected_year)
        img_b64 = self._image_to_base64(image)

        try:
            response = self._client.chat(
                model=self.config.ollama.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                options={"temperature": 0.1},
            )

            reply = response["message"]["content"].strip()
            logger.debug(f"VLM row response: {reply}")
            return self._parse_row_response(reply)

        except Exception as e:
            logger.error(f"VLM row extraction failed: {e}")
            return {}

    def extract_full_page(self, image: np.ndarray) -> dict:
        """Extract all shift records and header information from a full page image.

        Returns a dict containing 'shifts' (list of dicts) and header strings.
        """
        if not self._ensure_client():
            return {"shifts": [], "recipient_name": "", "rn_lpn_name": ""}

        prompt = self._build_full_page_prompt()
        img_b64 = self._image_to_base64(image)

        try:
            logger.info("Sending full page to VLM. Processing...")
            response = self._client.chat(
                model=self.config.ollama.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }
                ],
                options={"temperature": 0.1},
            )

            reply = response["message"]["content"].strip()
            logger.debug(f"VLM full page response: {reply[:200]}...")
            return self._parse_full_page_response(reply)

        except Exception as e:
            logger.error(f"VLM full page extraction failed: {e}")
            return {"shifts": [], "recipient_name": "", "rn_lpn_name": ""}

    def _build_cell_prompt(
        self, field_name: str, expected_year: int | None = None
    ) -> str:
        """Build a targeted prompt for single-cell extraction."""
        field_hints = {
            "date": "Read the handwritten date in this cell. Format: MM/DD. If a year is visible, include it.",
            "time_in": "Read the handwritten clock-in time exactly as written in this cell (e.g., '8:00 AM', '4:30p'). Do not convert to 24-hour time prematurely.",
            "time_out": "Read the handwritten clock-out time exactly as written in this cell (e.g., '8:00 AM', '4:30p'). Do not convert to 24-hour time prematurely.",
            "total_hours": "Read the handwritten total hours in this cell. Format: decimal number (e.g., 3.5).",
            "notes": "Read any handwritten notes or text in this cell.",
        }

        hint = field_hints.get(field_name, "Read the handwritten text in this cell.")

        year_note = ""
        if expected_year is not None:
            year_note = f"\nNOTE: This timesheet is from the year {expected_year}. If the year is not clearly visible in the cell, do NOT include a year in your answer — just return the month and day (MM/DD)."

        return (
            f"{hint}{year_note}\n\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            '{"value": "<your reading>", "confidence": "<high|medium|low>"}\n\n'
            "If the cell is empty or illegible, return:\n"
            '{"value": "", "confidence": "low"}'
        )

    def _build_row_prompt(self, expected_year: int | None = None) -> str:
        """Build a prompt for full-row extraction."""
        year_note = ""
        if expected_year is not None:
            year_note = f"\nNOTE: This timesheet is from the year {expected_year}. For the date field, return MM/DD format only (no year) unless the year is clearly visible."

        return (
            f"This image shows one row of a handwritten timesheet. "
            f"Extract the following fields from left to right:\n\n"
            f"1. Date (MM/DD format{year_note})\n"
            f"2. Time In (exact text written on page, e.g., '8:00 AM', '4:30p')\n"
            f"3. Time Out (exact text written on page, e.g., '8:00 AM', '4:30p')\n"
            f"4. Total Hours (decimal number)\n"
            f"5. Notes (any additional text)\n\n"
            f"Return ONLY a JSON object:\n"
            "{"
            '"date": "...", '
            '"time_in": "...", '
            '"time_out": "...", '
            '"total_hours": "...", '
            '"notes": "..."'
            "}\n\n"
            "Use empty string for any field you cannot read."
        )

    def _build_full_page_prompt(self) -> str:
        """Build a prompt for full-page matrix extraction."""
        return (
            "This image is a handwritten timesheet. Please extract the shifts for all the listed days, "
            "as well as the Recipient's Name (Patient) and the Print RN/LPN Name (Employee).\n"
            "CRITICAL: Do NOT extract the Agency or Provider Name (e.g., do not extract 'Yours Truly, Inc' as the RN/LPN Name).\n"
            "Do NOT extract notes, comments, or other care tasks.\n"
            "CRITICAL WARNING: If this image is purely a signature page, or does NOT contain a grid/table of handwritten shifts, "
            "you MUST return an empty shifts array: []. Do NOT invent, guess, or sequentially generate dates or names.\n\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "recipient_name": "...",\n'
            '  "rn_lpn_name": "...",\n'
            '  "shifts": [\n'
            '    {"date": "...", "time_in": "...", "time_out": "...", "total_hours": "..."}\n'
            "  ]\n"
            "}\n\n"
            "If a field is missing or illegible, use an empty string.\n"
            "CRITICAL: For dates, prioritize extracting the numerical Date (Month/Day/Year) over the Day of the week. "
            "If both exist, extract the numerical date (e.g. '01/07/2026').\n"
            "CRITICAL: For times, strictly extract the exact text written on the page (e.g., '8:00 AM', '4:30p'). "
            "Do NOT convert the times to 24-hour format prematurely.\n"
            "CRITICAL: Only extract rows that contain visible, handwritten times. "
            "If a day or row is blank, or if the page does not contain a timesheet grid, DO NOT output it. "
            "Return an empty list for the shifts array. Do not infer, guess, or carry over hours for empty cells."
        )

    def _parse_full_page_response(self, reply: str) -> dict:
        """Parse VLM response for a full page extraction."""
        try:
            data = self._extract_json(reply)
            result = {
                "recipient_name": str(data.get("recipient_name", "")).strip(),
                "rn_lpn_name": str(data.get("rn_lpn_name", "")).strip(),
                "shifts": [],
            }

            shifts_data = data.get("shifts", [])

            if not isinstance(shifts_data, list):
                logger.warning("VLM full page shifts is not a JSON array")
                return result

            for item in shifts_data:
                if isinstance(item, dict):
                    row = {}
                    for key in ["date", "time_in", "time_out", "total_hours"]:
                        row[key] = str(item.get(key, "")).strip()
                    row["notes"] = ""  # Omit notes

                    if row.get("time_in") or row.get("time_out"):
                        result["shifts"].append(row)

            return result
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(f"VLM full page response not valid JSON: {e}")
            return {"shifts": [], "recipient_name": "", "rn_lpn_name": ""}

    def _parse_cell_response(self, reply: str, field_name: str) -> tuple[str, float]:
        """Parse VLM response for a single cell extraction."""
        try:
            # Try to extract JSON from the response
            data = self._extract_json(reply)
            value = str(data.get("value", "")).strip()

            # Map confidence string to float
            conf_str = str(data.get("confidence", "low")).lower()
            conf_map = {"high": 0.90, "medium": 0.75, "low": 0.50}
            confidence = conf_map.get(conf_str, 0.50)

            logger.info(f"VLM extracted {field_name}: '{value}' (conf={confidence})")
            return (value, confidence)

        except (json.JSONDecodeError, ValueError) as e:
            # If JSON parsing fails, treat the raw reply as the value
            logger.warning(f"VLM response not valid JSON for {field_name}: {e}")
            return (reply.strip(), 0.50)

    def _parse_row_response(self, reply: str) -> dict[str, str]:
        """Parse VLM response for a full row extraction."""
        try:
            data = self._extract_json(reply)
            result = {}
            for key in ["date", "time_in", "time_out", "total_hours", "notes"]:
                result[key] = str(data.get(key, "")).strip()
            return result
        except (json.JSONDecodeError, ValueError):
            logger.warning("VLM row response not valid JSON")
            return {}

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON object from a text response (handles markdown code blocks and thinking text)."""
        import re

        text = text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        # Try direct parse first
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Qwen3-VL "thinking" mode: find JSON object in text
        # Look for the last { ... } block in the response
        brace_start = text.rfind("{")
        if brace_start != -1:
            json_text = text[brace_start:]
            try:
                return json.loads(json_text)
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: use regex to find JSON-like pattern
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)

    @staticmethod
    def _image_to_base64(image: np.ndarray, max_dim: int = 2048) -> str:
        """Convert a numpy image to base64 string for Ollama API."""
        if len(image.shape) == 2:
            # Grayscale -> convert to BGR for PIL
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_stats(self) -> dict:
        """Return VLM metrics for benchmarking."""
        total_fields = max(self._total_fields_queried, 1)
        return {
            "total_calls": self._total_calls,
            "empty_responses": self._empty_responses,
            "json_parse_failures": self._json_parse_failures,
            "total_fields_queried": self._total_fields_queried,
            "empty_response_rate": self._empty_responses / total_fields,
            "json_parse_success_rate": (1.0 - self._json_parse_failures / total_fields),
        }

    def reset_stats(self) -> None:
        """Reset metrics for a new run."""
        self._total_calls = 0
        self._empty_responses = 0
        self._json_parse_failures = 0
        self._total_fields_queried = 0
