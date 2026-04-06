"""Test cloud approach with parallel processing and compare with previous run times.

This script:
1. Runs ONLY the layout_guided_vlm_cloud approach on all input files
2. Captures detailed timing metrics
3. Compares with previous benchmark data (if available)
4. Writes comparison to a markdown file

Usage:
    uv run python scripts/test_cloud_parallel.py
"""

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
COMPARISON_FILE = PROJECT_ROOT / "docs" / "cloud_parallel_comparison.md"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "cloud_parallel_test.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)


def load_config():
    """Load current config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def save_config(config):
    """Write config back to config.yaml."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_input_files():
    """Get all supported input files."""
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    files = sorted(
        f
        for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in supported and not f.name.startswith(".")
    )
    return files


def clean_cloud_output():
    """Clean only the cloud approach output directory."""
    cloud_dir = OUTPUT_DIR / "layout_guided_vlm_cloud"
    if cloud_dir.exists():
        shutil.rmtree(cloud_dir)
    cloud_dir.mkdir(exist_ok=True)
    (cloud_dir / "debug").mkdir(exist_ok=True)


def run_cloud_approach():
    """Run the cloud approach with parallel processing and capture timing."""
    from src.config import load_config as load_app_config
    from src.pipeline import Pipeline

    # Update config to use cloud approach
    config = load_config()
    config["extraction_mode"] = "layout_guided_vlm_cloud"
    config["debug"]["visualize_ocr"] = False
    save_config(config)
    logger.info("Config updated: extraction_mode=layout_guided_vlm_cloud")

    # Load app config
    app_config = load_app_config(str(CONFIG_PATH))
    pipeline = Pipeline(app_config)

    input_files = get_input_files()
    if not input_files:
        logger.error("No input files found!")
        return None

    logger.info(f"\n{'=' * 70}")
    logger.info(f"STARTING CLOUD APPROACH (PARALLEL PROCESSING)")
    logger.info(f"{'=' * 70}")
    logger.info(f"Files to process: {len(input_files)}")
    for f in input_files:
        size_kb = f.stat().st_size / 1024
        logger.info(f"  - {f.name} ({size_kb:.0f} KB)")

    # Clean output and run
    clean_cloud_output()

    overall_start = time.time()
    file_timings = []

    try:
        # Process each file individually to capture per-file timing
        for file_idx, file_path in enumerate(input_files):
            # Add delay between files to avoid rate limiting
            if file_idx > 0:
                delay = getattr(app_config.cloud_vlm, "inter_file_delay", 5)
                if delay > 0:
                    logger.info(f"\n⏳ Waiting {delay}s before next file (rate limit avoidance)...")
                    time.sleep(delay)
            
            logger.info(f"\n{'─' * 70}")
            logger.info(f"Processing: {file_path.name}")
            logger.info(f"{'─' * 70}")

            file_start = time.time()
            result = pipeline.process_file(file_path)
            file_elapsed = time.time() - file_start

            file_timings.append({
                "file": file_path.name,
                "elapsed_seconds": file_elapsed,
                "total_rows": result.total_rows,
                "accepted_count": result.accepted_count,
                "flagged_count": result.flagged_count,
                "failed_count": result.failed_count,
            })

            logger.info(
                f"  ✓ Completed in {file_elapsed:.1f}s "
                f"({result.total_rows} rows, {result.accepted_count} accepted)"
            )

            # Check if benchmark has page-level timing details
            if hasattr(pipeline, 'benchmark') and pipeline.benchmark:
                bench = pipeline.benchmark
                if hasattr(bench, 'page_metrics') and bench.page_metrics:
                    for pm in bench.page_metrics:
                        logger.info(
                            f"    Page {pm.page_number}: "
                            f"OCR={pm.ocr_inference_time_s:.2f}s, "
                            f"Layout={pm.layout_detection_time_s:.2f}s, "
                            f"VLM={pm.vlm_inference_time_s:.2f}s"
                        )

        overall_elapsed = time.time() - overall_start

        logger.info(f"\n{'=' * 70}")
        logger.info(f"CLOUD APPROACH COMPLETE")
        logger.info(f"{'=' * 70}")
        logger.info(f"Total time: {overall_elapsed:.1f}s")
        logger.info(f"Files processed: {len(file_timings)}")
        total_rows = sum(ft["total_rows"] for ft in file_timings)
        logger.info(f"Total rows: {total_rows}")

        # Reset benchmark for next run if needed
        if hasattr(pipeline, 'benchmark'):
            pipeline.benchmark = None

        return {
            "approach": "layout_guided_vlm_cloud_parallel",
            "total_time_seconds": overall_elapsed,
            "file_count": len(file_timings),
            "files": file_timings,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        logger.error(f"❌ Cloud approach failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def find_previous_benchmark():
    """Find previous benchmark/timing data from logs or output."""
    # Check for combined benchmark files
    combined_bench = OUTPUT_DIR / "combined" / "benchmark_combined.xlsx"
    combined_merged = OUTPUT_DIR / "combined" / "merged_combined.xlsx"

    # Check for individual approach report files
    cloud_report = OUTPUT_DIR / "layout_guided_vlm_cloud" / "patient_a_week1_report.json"

    # Try to extract timing from log files
    log_dir = PROJECT_ROOT / "logs"
    log_files = sorted(log_dir.glob("batch_run_*.log"), reverse=True)

    previous_runs = []

    # Parse recent log files for timing information
    for log_file in log_files[:3]:  # Check last 3 runs
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Look for approach completion times
            import re
            # Pattern: "APPROACH: ... (layout_guided_vlm_cloud)" followed by timing
            approach_pattern = r'APPROACH:.*?layout_guided_vlm_cloud.*?\n.*?\n.*?DONE:.*?in ([\d.]+)s'
            matches = re.findall(approach_pattern, content, re.DOTALL)

            if matches:
                # Extract date from filename
                date_match = re.search(r'batch_run_(\d{8}_\d{6})', log_file.name)
                date_str = date_match.group(1) if date_match else "unknown"

                previous_runs.append({
                    "date": date_str,
                    "log_file": log_file.name,
                    "total_time": float(matches[0]),
                    "details": f"Found in {log_file.name}"
                })

        except Exception as e:
            logger.debug(f"Could not parse {log_file}: {e}")

    return previous_runs


def generate_comparison(new_run, previous_runs):
    """Generate markdown comparison file."""
    md_lines = []
    md_lines.append("# Cloud Approach: Parallel Processing Performance Comparison")
    md_lines.append("")
    md_lines.append(f"**Generated:** {new_run['timestamp']}")
    md_lines.append(f"**Approach:** `layout_guided_vlm_cloud` (with parallel VLM API requests)")
    md_lines.append("")

    # New run summary
    md_lines.append("## New Run (Parallel Processing Enabled)")
    md_lines.append("")
    md_lines.append(f"- **Total Time:** {new_run['total_time_seconds']:.1f}s")
    md_lines.append(f"- **Files Processed:** {new_run['file_count']}")
    md_lines.append(f"- **Total Rows Extracted:** {sum(f['total_rows'] for f in new_run['files'])}")
    md_lines.append("")

    md_lines.append("### Per-File Breakdown")
    md_lines.append("")
    md_lines.append("| File | Time (s) | Rows | Accepted | Flagged | Failed |")
    md_lines.append("|------|----------|------|----------|---------|--------|")
    for f in new_run['files']:
        md_lines.append(
            f"| {f['file']} | {f['elapsed_seconds']:.1f} | "
            f"{f['total_rows']} | {f['accepted_count']} | "
            f"{f['flagged_count']} | {f['failed_count']} |"
        )
    md_lines.append("")

    # Previous runs
    if previous_runs:
        md_lines.append("## Previous Runs (Sequential Processing)")
        md_lines.append("")
        md_lines.append("| Date/Log | Total Time (s) | Notes |")
        md_lines.append("|----------|---------------|-------|")
        for run in previous_runs:
            md_lines.append(
                f"| {run['date']} | {run['total_time']:.1f} | {run['details']} |"
            )
        md_lines.append("")

        # Calculate improvement
        if previous_runs:
            avg_previous = sum(r['total_time'] for r in previous_runs) / len(previous_runs)
            new_time = new_run['total_time_seconds']
            improvement = ((avg_previous - new_time) / avg_previous * 100) if avg_previous > 0 else 0

            md_lines.append("## Performance Improvement")
            md_lines.append("")
            md_lines.append(f"- **Previous Average:** {avg_previous:.1f}s")
            md_lines.append(f"- **New (Parallel):** {new_time:.1f}s")
            md_lines.append(f"- **Time Saved:** {avg_previous - new_time:.1f}s")
            md_lines.append(f"- **Improvement:** {improvement:+.1f}%")
            md_lines.append("")

            if improvement > 0:
                md_lines.append("✅ **Parallel processing is faster!**")
            else:
                md_lines.append("⚠️ **Parallel processing did not show improvement in this run.**")
                md_lines.append("")
                md_lines.append("Possible reasons:")
                md_lines.append("- API rate limiting may be throttling parallel requests")
                md_lines.append("- Network latency is not the bottleneck")
                md_lines.append("- OCR/layout detection is the dominant cost")
                md_lines.append("- Small number of files reduces the benefit of parallelism")
            md_lines.append("")
    else:
        md_lines.append("## Previous Runs")
        md_lines.append("")
        md_lines.append("⚠️ No previous timing data found in logs.")
        md_lines.append("")
        md_lines.append("The per-file breakdown above shows the current run's performance.")
        md_lines.append("Run the full benchmark suite (`scripts/run_all_approaches_safe.py`) to get comparison data.")
        md_lines.append("")

    # Technical details
    md_lines.append("## Technical Details")
    md_lines.append("")
    md_lines.append("### Parallel Processing Strategy")
    md_lines.append("")
    md_lines.append("The parallel implementation works as follows:")
    md_lines.append("")
    md_lines.append("1. **Phase 1 (Sequential, CPU-bound):** OCR + layout detection on all pages")
    md_lines.append("2. **Phase 2 (Parallel, I/O-bound):** All VLM API requests sent concurrently via `ThreadPoolExecutor`")
    md_lines.append("3. **Phase 3 (Sequential, CPU-bound):** Process results into structured records")
    md_lines.append("")
    md_lines.append("### Configuration")
    md_lines.append("")
    md_lines.append("```yaml")
    md_lines.append("cloud_vlm:")
    md_lines.append("  parallel_workers: 3")
    md_lines.append("  model: gemini-3-flash-preview")
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("### Implementation Files")
    md_lines.append("")
    md_lines.append("- `src/vlm_cloud.py`: Added `batch_extract_table_crops()` with `ThreadPoolExecutor`")
    md_lines.append("- `src/pipeline.py`: Added `_process_file_cloud_batch()` for parallel routing")
    md_lines.append("- `config.yaml`: Added `parallel_workers: 3` setting")
    md_lines.append("")

    # Write to file
    comparison_file = COMPARISON_FILE
    comparison_file.parent.mkdir(parents=True, exist_ok=True)

    with open(comparison_file, 'w') as f:
        f.write('\n'.join(md_lines))

    return comparison_file


def main():
    logger.info("=" * 70)
    logger.info("CLOUD APPROACH PARALLEL PROCESSING TEST")
    logger.info("=" * 70)

    # Run cloud approach
    new_run = run_cloud_approach()

    if not new_run:
        logger.error("❌ Failed to run cloud approach")
        return

    # Find previous benchmark data
    previous_runs = find_previous_benchmark()

    if previous_runs:
        logger.info(f"\nFound {len(previous_runs)} previous run(s) for comparison")
    else:
        logger.info("\n⚠️ No previous timing data found")

    # Generate comparison markdown
    comparison_file = generate_comparison(new_run, previous_runs)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"COMPARISON SAVED")
    logger.info(f"{'=' * 70}")
    logger.info(f"File: {comparison_file}")
    logger.info(f"")
    logger.info(f"Open with: code {comparison_file}")
    logger.info(f"Or view: cat {comparison_file}")


if __name__ == "__main__":
    main()
