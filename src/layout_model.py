"""PP-DocLayoutV3 layout detector — learned table region detection."""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from .config import AppConfig
from .layout import Zone

logger = logging.getLogger(__name__)


class DocLayoutDetector:
    """PP-DocLayoutV3 wrapper for table/grid region detection.

    Uses a learned document layout model to detect the table region
    in a timesheet image. Returns a Zone with the table bounding box plus
    padding to ensure header rows and column labels are included.

    Fully local — no data leaves the machine.
    """

    TABLE_PADDING = 30  # pixels added on all sides of detected bbox

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model = None
        self._available: Optional[bool] = None

    def _ensure_initialized(self) -> bool:
        """Lazy-load PP-DocLayoutV3 model."""
        if self._available is not None:
            return self._available

        try:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddlex import create_model

            logger.info("Loading PP-DocLayoutV3 model...")
            self._model = create_model("PP-DocLayoutV3")
            self._available = True
            logger.info("PP-DocLayoutV3 model loaded")
            return True
        except Exception as e:
            logger.warning(
                f"PP-DocLayoutV3 unavailable: {e}. Falling back to full-page VLM."
            )
            self._available = False
            return False

    def detect_table(self, image: np.ndarray) -> Optional[Zone]:
        """Detect the table/grid region in a timesheet image.

        Args:
            image: Full page image (numpy array, BGR or grayscale).

        Returns:
            Zone with table bounding box + padding, or None if not detected.
        """
        if not self._ensure_initialized():
            return None

        try:
            # Convert grayscale to BGR if needed
            if len(image.shape) == 2:
                import cv2

                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            result_gen = self._model.predict(image, batch_size=1)
            result = next(result_gen)

            boxes = result.get("boxes", [])
            if not boxes:
                logger.info("No layout elements detected by PP-DocLayoutV3")
                return None

            # Find table/grid box
            # PP-DocLayoutV3 returns boxes with label and coordinate
            # Labels include: 'title', 'text', 'table', 'figure', 'header', 'footer', etc.
            table_bbox = None
            for box in boxes:
                label = box.get("label", "").lower()
                if label in ("table", "grid"):
                    coord = box.get("coordinate", None)
                    if coord is not None and len(coord) == 4:
                        table_bbox = coord
                        break

            if table_bbox is None:
                # Log what was detected for debugging
                labels = [b.get("label", "unknown") for b in boxes]
                logger.info(
                    f"No table/grid region detected by PP-DocLayoutV3. "
                    f"Detected: {labels}"
                )
                return None

            x1, y1, x2, y2 = table_bbox
            h, w = image.shape[:2]

            # Add padding, clamped to image boundaries
            pad = self.TABLE_PADDING
            x1 = max(0, int(x1) - pad)
            y1 = max(0, int(y1) - pad)
            x2 = min(w, int(x2) + pad)
            y2 = min(h, int(y2) + pad)

            logger.info(
                f"Table detected by PP-DocLayoutV3: "
                f"({x1}, {y1}) → ({x2}, {y2}) "
                f"size={x2 - x1}x{y2 - y1}"
            )

            return Zone(x_start=x1, y_start=y1, x_end=x2, y_end=y2, label="table")

        except Exception as e:
            logger.warning(f"PP-DocLayoutV3 detection failed: {e}")
            return None
