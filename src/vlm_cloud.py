"""Cloud VLM extractor — Gemini API for table crop extraction."""

from __future__ import annotations

import base64
import io
import json
import time
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .config import AppConfig

logger = logging.getLogger(__name__)

MAX_DIM = 2048

# Media resolution mapping for Gemini API
MEDIA_RESOLUTION_MAP = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
    "ultra_high": "MEDIA_RESOLUTION_ULTRA_HIGH",
}


def _load_dotenv() -> None:
    """Load .env file from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        # Fallback: manual .env parsing
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value


class CloudVlmExtractor:
    """Cloud VLM extractor using Google Gemini API for table crop extraction.

    Sends only the cropped table region (no PHI) to Gemini for structured
    extraction. Returns the same dict format as the local VLM fallback.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._client = None
        self._available: Optional[bool] = None

    def _ensure_client(self) -> bool:
        """Initialize Google AI client and check availability."""
        if self._available is not None:
            return self._available

        _load_dotenv()

        try:
            from google import genai

            is_vertex = getattr(self.config.cloud_vlm, "vertexai", False)
            api_key = os.environ.get(self.config.cloud_vlm.api_key_env, "")
            
            if not is_vertex and not api_key:
                logger.warning(
                    f"Cloud VLM API key not found in env var "
                    f"{self.config.cloud_vlm.api_key_env}. "
                    f"Cloud VLM extraction disabled."
                )
                self._available = False
                return False

            if is_vertex:
                self._client = genai.Client(
                    vertexai=True,
                    project=self.config.cloud_vlm.project,
                    location=self.config.cloud_vlm.location,
                )
            else:
                self._client = genai.Client(api_key=api_key)
                
            self._available = True
            logger.info(
                f"Google Gemini client initialized "
                f"(model={self.config.cloud_vlm.model})"
            )
            return True
        except Exception as e:
            logger.warning(f"Google Gemini client unavailable: {e}")
            self._available = False
            return False

    def extract_table_crop(self, image: np.ndarray) -> dict:
        """Extract shifts from a cropped table image via Gemini API.

        The image should contain only the timesheet grid (no headers,
        signatures, footers). Returns dict with 'shifts' list.
        """
        if not self._ensure_client():
            return {"shifts": []}

        prompt = self._build_table_prompt()
        img_bytes = self._image_to_bytes(image)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Sending table crop to Gemini ({self.config.cloud_vlm.model})... (Attempt {attempt+1}/{max_retries})"
                )
                from google.genai import types

                image_part = types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg",
                )

                # Set media resolution for better accuracy
                resolution_str = getattr(self.config.cloud_vlm, "media_resolution", "high")
                media_resolution = types.MediaResolution(
                    MEDIA_RESOLUTION_MAP.get(resolution_str, "MEDIA_RESOLUTION_HIGH")
                )

                response = self._client.models.generate_content(
                    model=self.config.cloud_vlm.model,
                    contents=[prompt, image_part],
                    config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                        "media_resolution": media_resolution,
                    },
                )

                reply = response.text.strip()
                logger.debug(f"Cloud VLM response: {reply[:300]}...")
                return self._parse_response(reply)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg and attempt < max_retries - 1:
                    logger.warning(f"Rate limit hit (429). Retrying in 60 seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(60)
                else:
                    logger.error(f"Cloud VLM table crop extraction failed: {e}")
                    return {"shifts": []}
                    
        return {"shifts": []}


    def _build_table_prompt(self) -> str:
        """Build prompt optimized for cropped table images (no PHI context)."""
        return (
            "This image shows a handwritten timesheet grid. "
            "Extract all shift entries as a JSON array.\n\n"
            "Each row has: Date, Time In, Time Out, Total Hours.\n"
            "Return ONLY a JSON object with this exact structure:\n"
            "{\n"
            '  "shifts": [\n'
            '    {"date": "...", "time_in": "...", "time_out": "...", "total_hours": "..."}\n'
            "  ]\n"
            "}\n\n"
            "If a field is missing or illegible, use an empty string.\n"
            "CRITICAL: For dates, extract the numeric date written on the page "
            "(e.g., '2/18/26', '02/18/2026', '1/4'). Do NOT return day-of-week names "
            "like 'Wednesday' or 'Thursday'. Always return the numeric date.\n"
            "CRITICAL: For times, strictly extract the exact text written on the page "
            "(e.g., '8:00 AM', '4:30p', '10:30 PM'). Do NOT convert to 24-hour format.\n"
            "CRITICAL: Only extract rows that contain visible, handwritten times. "
            "If a row is blank, DO NOT include it.\n"
            "Do not invent, guess, or sequentially generate dates or times."
        )

    def _parse_response(self, reply: str) -> dict:
        """Parse Gemini response into standard shifts dict."""
        try:
            data = self._extract_json(reply)
            result = {"shifts": []}

            shifts_data = data.get("shifts", [])
            if not isinstance(shifts_data, list):
                logger.warning("Cloud VLM shifts is not a JSON array")
                return result

            for item in shifts_data:
                if isinstance(item, dict):
                    row = {}
                    for key in ["date", "time_in", "time_out", "total_hours"]:
                        row[key] = str(item.get(key, "")).strip()
                    row["notes"] = ""

                    if row.get("time_in") or row.get("time_out"):
                        result["shifts"].append(row)

            return result
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning(f"Cloud VLM response not valid JSON: {e}")
            return {"shifts": []}

    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from response text."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 1)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.rfind("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start : end + 1])
            raise

    def _image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy array to JPEG bytes.
        
        For cloud VLM, we prefer color images with higher quality
        to maximize handwriting recognition accuracy.
        """
        # Get configurable JPEG quality
        quality = getattr(self.config.cloud_vlm, "image_quality", 92)
        
        # Check if we should preserve color or convert to grayscale
        use_color = getattr(self.config.cloud_vlm, "use_color_images", True)
        
        if len(image.shape) == 2:
            # Grayscale input - convert to BGR for consistency
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif len(image.shape) == 3 and not use_color:
            # Convert color to grayscale if configured
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        h, w = image.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()
