"""CLI entry point for the timesheet OCR pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="timesheet-ocr",
        description="Extract structured data from scanned handwritten timesheets.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory containing timesheet scans (default: from config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: from config.yaml)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Process a single file instead of the entire input directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING
    if args.verbose:
        log_level = logging.INFO
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_config(args.config)

    # Override paths from CLI if provided
    if args.input_dir:
        config.paths.input_dir = args.input_dir
    if args.output_dir:
        config.paths.output_dir = args.output_dir

    # Ensure output directory exists
    config.output_path.mkdir(parents=True, exist_ok=True)

    # Run pipeline
    pipeline = Pipeline(config)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            return 1
        results = [pipeline.process_file(file_path)]
    else:
        results = pipeline.process_directory()

    if not results:
        print("No files processed.", file=sys.stderr)
        return 1

    # Print summary
    print(f"\n{'=' * 50}")
    print("EXTRACTION SUMMARY")
    print(f"{'=' * 50}")
    for result in results:
        # Results may be ExtractionResult objects or dicts (from subprocess)
        if isinstance(result, dict):
            source = result["source_file"]
            pages = result["total_pages"]
            rows = result["total_rows"]
            accepted = result["accepted_count"]
            flagged = result["flagged_count"]
            failed = result["failed_count"]
            time_s = result["processing_time_seconds"]
        else:
            source = result.source_file
            pages = result.total_pages
            rows = result.total_rows
            accepted = result.accepted_count
            flagged = result.flagged_count
            failed = result.failed_count
            time_s = result.processing_time_seconds

        print(f"\n  {source}:")
        print(f"    Pages:    {pages}")
        print(f"    Rows:     {rows}")
        print(f"    Accepted: {accepted}")
        print(f"    Flagged:  {flagged}")
        print(f"    Failed:   {failed}")
        print(f"    Time:     {time_s:.1f}s")

    if isinstance(results[0], dict):
        total_rows = sum(r["total_rows"] for r in results)
        total_accepted = sum(r["accepted_count"] for r in results)
        total_flagged = sum(r["flagged_count"] for r in results)
    else:
        total_rows = sum(r.total_rows for r in results)
        total_accepted = sum(r.accepted_count for r in results)
        total_flagged = sum(r.flagged_count for r in results)
    print(f"\n  TOTAL: {total_rows} rows ({total_accepted} accepted, {total_flagged} flagged)")
    print(f"  Output: {config.output_path}")
    print(f"{'=' * 50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
