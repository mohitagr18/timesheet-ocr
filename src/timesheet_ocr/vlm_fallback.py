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

    def extract_cell_value(self, image: np.ndarray, field_name: str) -> tuple[str, float]:
        """Extract a single cell value from a cropped image.

        Args:
            image: Cropped image of the cell (numpy array, BGR or grayscale).
            field_name: Name of the field (e.g., "date", "time_in").

        Returns:
            (extracted_value, confidence) — confidence is estimated from the VLM response.
        """
        if not self._ensure_client():
            return ("", 0.0)

        prompt = self._build_cell_prompt(field_name)
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
            return self._parse_cell_response(reply, field_name)

        except Exception as e:
            logger.error(f"VLM extraction failed for {field_name}: {e}")
            return ("", 0.0)

    def extract_row(self, image: np.ndarray) -> dict[str, str]:
        """Extract all fields from a full row image.

        Returns a dict mapping field names to extracted values.
        """
        if not self._ensure_client():
            return {}

        prompt = self._build_row_prompt()
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
            return self._parse_row_response(reply)

        except Exception as e:
            logger.error(f"VLM row extraction failed: {e}")
            return {}

    def _build_cell_prompt(self, field_name: str) -> str:
        """Build a targeted prompt for single-cell extraction."""
        field_hints = {
            "date": "Read the handwritten date in this cell. Format: MM/DD/YYYY.",
            "time_in": "Read the handwritten clock-in time in this cell. Format: HH:MM (use 12-hour with AM/PM if visible, otherwise 24-hour).",
            "time_out": "Read the handwritten clock-out time in this cell. Format: HH:MM (use 12-hour with AM/PM if visible, otherwise 24-hour).",
            "total_hours": "Read the handwritten total hours in this cell. Format: decimal number (e.g., 3.5).",
            "notes": "Read any handwritten notes or text in this cell.",
        }

        hint = field_hints.get(field_name, "Read the handwritten text in this cell.")

        return (
            f"{hint}\n\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            '{"value": "<your reading>", "confidence": "<high|medium|low>"}\n\n'
            "If the cell is empty or illegible, return:\n"
            '{"value": "", "confidence": "low"}'
        )

    def _build_row_prompt(self) -> str:
        """Build a prompt for full-row extraction."""
        return (
            "This image shows one row of a handwritten timesheet. "
            "Extract the following fields from left to right:\n\n"
            "1. Date (MM/DD/YYYY format)\n"
            "2. Time In (HH:MM format)\n"
            "3. Time Out (HH:MM format)\n"
            "4. Total Hours (decimal number)\n"
            "5. Notes (any additional text)\n\n"
            "Return ONLY a JSON object:\n"
            "{"
            '"date": "...", '
            '"time_in": "...", '
            '"time_out": "...", '
            '"total_hours": "...", '
            '"notes": "..."'
            "}\n\n"
            "Use empty string for any field you cannot read."
        )

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
        """Extract JSON object from a text response (handles markdown code blocks)."""
        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        return json.loads(text)

    @staticmethod
    def _image_to_base64(image: np.ndarray) -> str:
        """Convert a numpy image to base64 string for Ollama API."""
        if len(image.shape) == 2:
            # Grayscale -> convert to BGR for PIL
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
