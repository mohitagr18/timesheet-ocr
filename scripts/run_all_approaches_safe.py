"""Resilient batch runner with persistent logging and crash recovery.

This script:
1. Cleans output directory
2. Runs ONE approach at a time, saving results immediately after each
3. Logs everything to a file (survives terminal crashes)
4. Saves progress state to JSON (can resume if interrupted)

Usage:
    # Run all approaches from scratch
    uv run python scripts/run_all_approaches_safe.py

    # Resume from where it left off (skip completed approaches)
    uv run python scripts/run_all_approaches_safe.py --resume
"""

import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.phi import PhiAnonymizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_FILE = OUTPUT_DIR / ".run_state.json"

APPROACHES = [
    ("ocr_only", "OCR Only (Baseline)"),
    ("ppocr_grid", "OCR + VLM Fallback"),  # SLOW: local Ollama per-field calls
    ("layout_guided_vlm_local", "Layout-Guided VLM (Local)"),  # SLOW: local Ollama
    ("layout_guided_vlm_cloud", "Layout-Guided VLM (Cloud)"),
    ("vlm_full_page", "VLM Full Page"),
    ("band_crop_vlm_cloud", "Band-Crop VLM (Cloud)"),
]


def setup_logging():
    """Setup logging to both file and console."""
    LOG_DIR.mkdir(exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"batch_run_{timestamp}.log"
    
    # Also create a symlink for easy access to latest log
    latest_log = LOG_DIR / "latest.log"
    
    # File handler - detailed format
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    
    # Console handler - simpler format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create symlink to latest log
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
    latest_log.symlink_to(log_file)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Latest log symlink: {latest_log}")
    
    return logger


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
    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.iterdir():
            if item.name.startswith("."):
                continue  # Skip hidden files like .gitkeep
            if item.is_dir():
                shutil.rmtree(item)
            elif item.is_file():
                item.unlink()
    OUTPUT_DIR.mkdir(exist_ok=True)
    logging.getLogger(__name__).info("Cleaned output directory.")


def save_state(state):
    """Save progress state to JSON file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    logging.getLogger(__name__).info(f"State saved: {len(state['completed'])} approaches completed")


def load_state():
    """Load progress state if it exists."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": [], "all_results": {}, "all_errors": {}, "total_start": None}


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
        if item.name.startswith("."):
            continue  # Skip hidden files
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


def run_approach(approach_id, approach_label, input_files, logger):
    """Run a single extraction approach on all input files."""
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

    # Use process_directory which handles all files via subprocess
    # generate_combined=False: combined results generated once at the end
    file_start = time.time()
    try:
        all_results = pipeline.process_directory(INPUT_DIR, generate_combined=False)
        elapsed = time.time() - file_start

        # Handle both dict (subprocess) and ExtractionResult (in-process) returns
        def _get(r, attr, default=0):
            if isinstance(r, dict):
                return r.get(attr, default)
            return getattr(r, attr, default)

        total_rows = sum(_get(r, "total_rows") for r in all_results)
        total_accepted = sum(_get(r, "accepted_count") for r in all_results)
        total_flagged = sum(_get(r, "flagged_count") for r in all_results)
        total_failed = sum(_get(r, "failed_count") for r in all_results)

        logger.info(
            f"  DONE: {len(all_results)} file(s) in {elapsed:.1f}s "
            f"({total_rows} rows, {total_accepted} accepted)"
        )

        for r in all_results:
            results.append(
                {
                    "file": _get(r, "source_file", "unknown"),
                    "rows": _get(r, "total_rows"),
                    "accepted": _get(r, "accepted_count"),
                    "flagged": _get(r, "flagged_count"),
                    "failed": _get(r, "failed_count"),
                    "status": "success",
                }
            )

    except Exception as e:
        elapsed = time.time() - file_start
        logger.error(f"  FAILED after {elapsed:.1f}s: {e}")
        import traceback
        logger.error(traceback.format_exc())
        errors.append(
            {
                "file": "all",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
        # Mark all input files as failed for this approach
        for f in input_files:
            results.append(
                {
                    "file": f.name,
                    "rows": 0,
                    "accepted": 0,
                    "flagged": 0,
                    "failed": 0,
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Move outputs to approach directory
    move_outputs_to_approach_dir(approach_id)
    logger.info(f"Outputs moved to {OUTPUT_DIR / approach_id}/")

    # Verify files were saved
    approach_dir = OUTPUT_DIR / approach_id
    saved_files = list(approach_dir.rglob("*"))
    logger.info(f"  Verified: {len(saved_files)} files/dirs in {approach_id}/")

    # Destroy pipeline to release any remaining references
    del pipeline
    gc.collect()

    return results, errors


def run_combined_comparison(logger):
    """Generate combined comparison across all approaches."""
    logger.info(f"\n{'=' * 70}")
    logger.info("GENERATING COMBINED COMPARISON")
    logger.info(f"{'=' * 70}")

    result = subprocess.run(
        ["python", "scripts/build_combined_report.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.stderr:
        logger.warning(result.stderr.strip())


REPORTS_DIR = PROJECT_ROOT / "reports"


def is_file_processed_in_approach(approach_id: str, anonymized_stem: str) -> bool:
    """Check if a file was successfully processed by looking for its anonymized _report.json.
    
    The pipeline saves results as <anonymized_stem>_report.json in the approach directory.
    """
    approach_dir = OUTPUT_DIR / approach_id
    report_file = approach_dir / f"{anonymized_stem}_report.json"
    return report_file.exists()


def get_anonymizer_for_files(input_files: list) -> PhiAnonymizer:
    """Create a PhiAnonymizer instance with all input filenames for consistent mapping."""
    filenames = [f.name for f in input_files]
    return PhiAnonymizer(filenames)


def get_files_still_needed(approach_id: str, input_files: list, anonymizer: PhiAnonymizer) -> list:
    """Get list of input files that haven't been successfully processed yet for this approach.
    
    Returns the list of files that still need processing (i.e., their anonymized
    _report.json doesn't exist in the approach directory).
    """
    approach_dir = OUTPUT_DIR / approach_id
    needed = []
    for f in input_files:
        anon_name = anonymizer.anonymize_filename(f.name)
        anon_stem = Path(anon_name).stem
        if not (approach_dir / f"{anon_stem}_report.json").exists():
            needed.append(f)
    return needed


def report_missing_files(all_results, logger):
    """Check which input files are missing from each approach's output.
    
    Uses PhiAnonymizer to map real filenames to anonymized names when
    checking for output files, since the pipeline saves results under
    anonymized names (e.g., patient_a_week1_report.json).
    """
    input_files = get_input_files()
    anonymizer = get_anonymizer_for_files(input_files)
    
    logger.info(f"\n{'=' * 70}")
    logger.info("MISSING / FAILED FILES REPORT")
    logger.info(f"{'=' * 70}")
    any_missing = False
    report_lines = [
        f"Session Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input files: {[f.name for f in input_files]}",
        "",
    ]
    for approach_id, approach_label in APPROACHES:
        results = all_results.get(approach_id, [])
        
        # Check actual output files using anonymized names
        processed = set()
        failed = {}
        for r in results:
            original_name = r["file"]
            anon_name = anonymizer.anonymize_filename(original_name)
            anon_stem = Path(anon_name).stem
            if r["status"] == "success" and is_file_processed_in_approach(approach_id, anon_stem):
                processed.add(original_name)
            elif r["status"] == "failed":
                failed[original_name] = r.get("error", "unknown")
        
        missing = [
            f.name for f in input_files
            if f.name not in processed and f.name not in failed
        ]

        if missing or failed:
            any_missing = True
            report_lines.append(f"⚠ {approach_label}:")
            logger.info(f"\n  ⚠ {approach_label}:")
            for fname, err in failed.items():
                line = f"    ✗ {fname}: FAILED — {err}"
                report_lines.append(line)
                logger.info(line)
            for fname in missing:
                line = f"    ✗ {fname}: MISSING (never processed)"
                report_lines.append(line)
                logger.info(line)
        else:
            line = f"  ✓ {approach_label}: all {len(input_files)} files present"
            report_lines.append(f"✓ {approach_label}: all {len(input_files)} files present")
            logger.info(line)

    if not any_missing:
        report_lines.append("\nAll approaches completed successfully for all input files.")
        logger.info(
            "\n  All approaches have all input files processed successfully."
        )

    # Save single report file for this session
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"report_{timestamp}.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    logger.info(f"\n  Report saved: {report_path}")

    return any_missing


def print_summary(all_results, all_errors, total_start, logger):
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
                status_icon = "✓" if r.get("status") == "success" else "✗"
                logger.info(
                    f"    {status_icon} {r['file']}: {r['rows']} rows "
                    f"({r['accepted']} accepted, {r['flagged']} flagged)"
                    + (f" — ERROR: {r['error']}" if r.get("error") else "")
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
    # Setup logging first
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Run all OCR approaches safely")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved state (skip already-processed files)")
    args = parser.parse_args()

    # Load or initialize state
    if args.resume:
        state = load_state()
        logger.info("Resuming from saved state")
    else:
        state = {
            "completed": [],
            "all_results": {},
            "all_errors": {},
            "total_start": time.time()
        }
        # Clean output directory (but not the state file if it exists)
        clean_output_dir()

    input_files = get_input_files()
    if not input_files:
        logger.error("No input files found!")
        return

    anonymizer = get_anonymizer_for_files(input_files)

    logger.info(f"Found {len(input_files)} input file(s):")
    for f in input_files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name} ({size_kb:.0f} KB)")
    logger.info("")

    all_results = state["all_results"]
    all_errors = state["all_errors"]
    completed = set(state["completed"])
    total_start = state.get("total_start") or time.time()

    # Track if we completed all approaches
    all_done = True

    for approach_id, approach_label in APPROACHES:
        # Check per-file completion when --resume is used
        if args.resume:
            files_needed = get_files_still_needed(approach_id, input_files, anonymizer)
            if not files_needed:
                logger.info(f"⏭  SKIPPING {approach_label} (already completed)")
                if approach_id not in completed:
                    completed.add(approach_id)
                continue
            else:
                logger.info(f"▶ {approach_label}: {len(files_needed)} file(s) still need processing "
                           f"({len(input_files) - len(files_needed)} already done)")
        else:
            # Not resuming: only skip if already marked completed in state
            if approach_id in completed:
                logger.info(f"⏭  SKIPPING {approach_label} (already completed)")
                continue

        # Run this approach
        approach_start = time.time()
        try:
            results, errors = run_approach(approach_id, approach_label, input_files, logger)
            all_results[approach_id] = results
            all_errors[approach_id] = errors
            
            # Only mark as completed if ALL files have successful output
            files_still_needed = get_files_still_needed(approach_id, input_files, anonymizer)
            if not files_still_needed:
                completed.add(approach_id)

            # Save state immediately after each approach
            state = {
                "completed": list(completed),
                "all_results": all_results,
                "all_errors": all_errors,
                "total_start": total_start
            }
            save_state(state)

            approach_elapsed = time.time() - approach_start
            logger.info(f"✅ {approach_label} completed in {approach_elapsed:.1f}s")

        except KeyboardInterrupt:
            logger.warning(f"\n⚠️  Interrupted during {approach_label}")
            logger.info("Progress has been saved. Run with --resume to continue.")
            all_done = False
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error in {approach_label}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            all_errors[approach_id] = [{"file": "all", "error": str(e)}]
            all_done = False
            break

    # Only generate combined results if all approaches completed
    if all_done and len(completed) == len(APPROACHES):
        run_combined_comparison(logger)
        print_summary(all_results, all_errors, total_start, logger)
        report_missing_files(all_results, logger)
        logger.info("\n🎉 All approaches completed successfully!")
    else:
        logger.info(f"\n⏸  Partial completion: {len(completed)}/{len(APPROACHES)} approaches done")
        logger.info("Run with --resume to continue from where we left off")
        report_missing_files(all_results, logger)

        # Print partial summary
        logger.info(f"\nCompleted so far:")
        for approach_id in completed:
            approach_label = dict(APPROACHES)[approach_id]
            results = all_results.get(approach_id, [])
            if results:
                success_count = sum(1 for r in results if r.get("status") == "success")
                failed_count = sum(1 for r in results if r.get("status") == "failed")
                total_rows = sum(r["rows"] for r in results)
                logger.info(
                    f"  ✓ {approach_label}: {total_rows} rows extracted "
                    f"({success_count} files ok, {failed_count} failed)"
                )


if __name__ == "__main__":
    main()
