"""Batch runner: execute all 5 extraction approaches on all input files.

This script:
1. Iterates through all 5 approaches sequentially
2. For each approach: processes all input files with memory cleanup between files
3. Moves outputs to output/{approach}/ directories
4. Generates combined comparison at the end

Usage:
    uv run python scripts/run_all_approaches.py
"""

import gc
import logging
import shutil
import subprocess
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

APPROACHES = [
    ("ocr_only", "OCR Only (Baseline)"),
    ("ppocr_grid", "OCR + VLM Fallback"),
    ("layout_guided_vlm_local", "Layout-Guided VLM (Local)"),
    ("layout_guided_vlm_cloud", "Layout-Guided VLM (Cloud)"),
    ("vlm_full_page", "VLM Full Page"),
]


def get_input_files():
    """Get all supported input files."""
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    files = sorted(
        f
        for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in supported and not f.name.startswith(".")
    )
    return files


def load_config():
    """Load current config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(config):
    """Write config back to config.yaml."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def clean_output_dir():
    """Remove all approach output directories and combined results."""
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink()
    OUTPUT_DIR.mkdir(exist_ok=True)
    logger.info("Cleaned output directory.")


def setup_approach_output(approach_id):
    """Create output directory structure for an approach."""
    approach_dir = OUTPUT_DIR / approach_id
    approach_dir.mkdir(exist_ok=True)
    (approach_dir / "debug").mkdir(exist_ok=True)
    return approach_dir


def move_outputs_to_approach_dir(approach_id):
    """Move generated output files from output/ root to output/{approach_id}/."""
    approach_dir = OUTPUT_DIR / approach_id
    for item in OUTPUT_DIR.iterdir():
        if item.is_file() or (item.is_dir() and item.name == "debug"):
            dest = approach_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if item.is_dir():
                shutil.move(str(item), str(dest))
            else:
                shutil.move(str(item), str(dest))


def run_approach(approach_id, approach_label, input_files):
    """Run a single extraction approach on all input files using process_directory."""
    logger.info(f"\n{'=' * 70}")
    logger.info(f"APPROACH: {approach_label} ({approach_id})")
    logger.info(f"{'=' * 70}")

    # Update config
    config = load_config()
    config["extraction_mode"] = approach_id
    config["debug"]["visualize_ocr"] = False
    save_config(config)
    logger.info(f"Config updated: extraction_mode={approach_id}, visualize_ocr=false")

    # Setup output directory
    setup_approach_output(approach_id)

    # Import pipeline
    from src.config import load_config as load_app_config
    from src.pipeline import Pipeline

    app_config = load_app_config(str(CONFIG_PATH))
    pipeline = Pipeline(app_config)

    results = []
    errors = []

    # Use process_directory which handles all files and combined benchmark
    file_start = time.time()
    try:
        all_results = pipeline.process_directory(INPUT_DIR)
        elapsed = time.time() - file_start

        total_rows = sum(r.total_rows for r in all_results)
        total_accepted = sum(r.accepted_count for r in all_results)
        total_flagged = sum(r.flagged_count for r in all_results)
        total_failed = sum(r.failed_count for r in all_results)

        logger.info(
            f"  DONE: {len(all_results)} file(s) in {elapsed:.1f}s "
            f"({total_rows} rows, {total_accepted} accepted)"
        )

        for r in all_results:
            results.append(
                {
                    "file": r.source_file,
                    "rows": r.total_rows,
                    "accepted": r.accepted_count,
                    "flagged": r.flagged_count,
                    "failed": r.failed_count,
                }
            )

    except Exception as e:
        elapsed = time.time() - file_start
        logger.error(f"  FAILED after {elapsed:.1f}s: {e}")
        errors.append(
            {
                "file": "all",
                "error": str(e),
            }
        )

    # Move outputs to approach directory
    move_outputs_to_approach_dir(approach_id)
    logger.info(f"Outputs moved to {OUTPUT_DIR / approach_id}/")

    # Destroy pipeline to release any remaining references
    del pipeline
    gc.collect()

    return results, errors


def run_combined_comparison():
    """Generate combined comparison across all approaches."""
    logger.info(f"\n{'=' * 70}")
    logger.info("GENERATING COMBINED COMPARISON")
    logger.info(f"{'=' * 70}")

    result = subprocess.run(
        ["python", "scripts/create_combined_results.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.warning(result.stderr.strip())


def print_summary(all_results, all_errors, total_start):
    """Print final summary table."""
    total_elapsed = time.time() - total_start

    logger.info(f"\n{'=' * 70}")
    logger.info("BATCH RUN COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total time: {total_elapsed / 60:.1f} minutes")
    logger.info("")

    for approach_id, approach_label in APPROACHES:
        results = all_results.get(approach_id, [])
        errors = all_errors.get(approach_id, [])

        logger.info(f"{'─' * 50}")
        logger.info(f"{approach_label} ({approach_id})")
        logger.info(f"{'─' * 50}")

        if results:
            total_rows = sum(r["rows"] for r in results)
            total_accepted = sum(r["accepted"] for r in results)
            total_flagged = sum(r["flagged"] for r in results)
            total_failed = sum(r["failed"] for r in results)

            logger.info(f"  Files processed: {len(results)}/{len(get_input_files())}")
            logger.info(f"  Total rows:      {total_rows}")
            logger.info(f"  Accepted:        {total_accepted}")
            logger.info(f"  Flagged:         {total_flagged}")
            logger.info(f"  Failed:          {total_failed}")

            for r in results:
                logger.info(
                    f"    {r['file']}: {r['rows']} rows "
                    f"({r['accepted']} accepted, {r['flagged']} flagged)"
                )

        if errors:
            logger.info(f"  ERRORS:")
            for e in errors:
                logger.info(f"    {e['file']}: {e['error']}")

        logger.info("")

    combined_bench = OUTPUT_DIR / "combined" / "benchmark_combined.xlsx"
    combined_merged = OUTPUT_DIR / "combined" / "merged_combined.xlsx"

    logger.info(f"Output files:")
    logger.info(
        f"  Combined benchmark: {combined_bench} ({'✓' if combined_bench.exists() else '✗'})"
    )
    logger.info(
        f"  Combined merged:    {combined_merged} ({'✓' if combined_merged.exists() else '✗'})"
    )
    logger.info("")

    for approach_id, _ in APPROACHES:
        approach_dir = OUTPUT_DIR / approach_id
        if approach_dir.exists():
            files = list(approach_dir.glob("*"))
            logger.info(f"  {approach_id}/: {len(files)} files")


def main():
    total_start = time.time()

    input_files = get_input_files()
    if not input_files:
        logger.error("No input files found!")
        return

    logger.info(f"Found {len(input_files)} input file(s):")
    for f in input_files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name} ({size_kb:.0f} KB)")
    logger.info("")

    # Clean output directory
    clean_output_dir()

    all_results = {}
    all_errors = {}

    for approach_id, approach_label in APPROACHES:
        results, errors = run_approach(approach_id, approach_label, input_files)
        all_results[approach_id] = results
        all_errors[approach_id] = errors

    # Generate combined comparison
    run_combined_comparison()

    # Print summary
    print_summary(all_results, all_errors, total_start)


if __name__ == "__main__":
    main()
