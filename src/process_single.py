"""Standalone subprocess entry point — processes a single file and writes results to JSON.

Usage:
    python -m src.process_single <file_path> <output_dir> <config_path>

Outputs:
    <output_dir>/.tmp/<file_name>.result.json  — ExtractionResult as JSON
    <output_dir>/.tmp/<file_name>.error.json   — Error details (if failed)
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline
from .phi import PhiAnonymizer

logger = logging.getLogger(__name__)


def process_single_file(
    file_path: str,
    output_dir: str,
    all_filenames: list[str] | None = None,
    config_path: str | None = None,
) -> dict:
    """Process a single file in isolation and return result as a dict.

    This function is designed to run in its own subprocess so that
    PaddleOCR's memory is fully released when the process exits.
    """
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(config_path)
    # Override output directory via the paths config (output_path is a computed property)
    config.paths.output_dir = str(output_dir)

    # Use the same anonymizer as the parent (all filenames for consistent mapping)
    if all_filenames is None:
        all_filenames = [file_path.name]
    anonymizer = PhiAnonymizer(all_filenames)

    # Set up logging for this subprocess
    log_level = logging.INFO
    if getattr(config, "debug", None) and getattr(config.debug, "verbose", False):
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Replace any existing handlers
    )

    # Run pipeline
    pipeline = Pipeline(config)
    result = pipeline.process_file(file_path, anonymizer)

    # Clean up models in this subprocess
    pipeline.cleanup()

    # Convert result to serializable dict
    result_dict = {
        "source_file": result.source_file,
        "processing_time_seconds": result.processing_time_seconds,
        "total_pages": result.total_pages,
        "total_rows": result.total_rows,
        "accepted_count": result.accepted_count,
        "flagged_count": result.flagged_count,
        "failed_count": result.failed_count,
        "records": [
            {
                "page_number": r.page_number,
                "patient_name": r.patient_name,
                "employee_name": r.employee_name,
                "row_count": len(r.rows),
            }
            for r in result.records
        ],
    }

    return result_dict


def main() -> int:
    """CLI entry point for subprocess execution."""
    if len(sys.argv) < 3:
        print(
            "Usage: python -m src.process_single <file_path> <output_dir> [config_path] [--filenames JSON]",
            file=sys.stderr,
        )
        return 1

    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else None

    # Parse optional filenames (JSON list of all input filenames for consistent anonymization)
    all_filenames = None
    if len(sys.argv) > 4 and sys.argv[4] == "--filenames" and len(sys.argv) > 5:
        all_filenames = json.loads(sys.argv[5])

    # Create temp output directory
    tmp_dir = Path(output_dir) / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    file_name = Path(file_path).stem
    result_file = tmp_dir / f"{file_name}.result.json"
    error_file = tmp_dir / f"{file_name}.error.json"

    try:
        result = process_single_file(file_path, output_dir, all_filenames, config_path)
        result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"SUCCESS: {result['source_file']} — {result['total_rows']} rows in {result['processing_time_seconds']:.1f}s")
        return 0
    except Exception as e:
        error_data = {
            "file": str(file_path),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        error_file.write_text(json.dumps(error_data, indent=2, ensure_ascii=False))
        print(f"FAILED: {Path(file_path).name} — {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
