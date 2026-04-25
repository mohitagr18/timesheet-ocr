#!/usr/bin/env python
"""Generate band crops for issue files to test the fixed band_crop logic.

This script processes only the files mentioned in the issues/ directory,
comparing the new (fixed) extraction against the problematic outputs.

Usage:
    uv run python scripts/generate_band_crops_for_issues.py
"""

import re
from pathlib import Path

import cv2
import numpy as np
from pdf2image import convert_from_path

from src.band_crop_extractor import BandCropExtractor
from src.config import load_config


def parse_issue_filename(issue_path: Path) -> tuple[str, int]:
    """Parse issue filename to get PDF name and page number.

    Examples:
        S.Hanton_Timesheet_-_012826-020326_page_1.png
            -> ("S.Hanton Timesheet - 012826-020326.pdf", 1)

        N.Rivera_Timesheets_-021826-022426_page_1.png
            -> ("N.Rivera Timesheets -021826-022426.pdf", 1)
    """
    stem = issue_path.stem

    page_match = re.search(r"_page_(\d+)$", stem)
    if page_match:
        page_num = int(page_match.group(1))
        stem = stem[: page_match.start()]
    else:
        page_num = 1

    pdf_name = stem.replace("_", " ").replace("\\-", "-") + ".pdf"

    return pdf_name, page_num


def find_pdf_in_input(pdf_name: str, input_dir: Path) -> Path | None:
    """Find PDF file in input directory (handles slight name variations)."""
    exact = input_dir / pdf_name
    if exact.exists():
        return exact

    for f in input_dir.glob("*.pdf"):
        if f.name.lower() == pdf_name.lower():
            return f
        f_normalized = f.name.replace(" ", "_").replace("-", "")
        n_normalized = pdf_name.replace(" ", "_").replace("-", "")
        if f_normalized == n_normalized:
            return f

    return None


def main() -> None:
    """Generate band crops for issue files."""
    print("Band Crop Generator for Issues")
    print("=" * 50)

    issues_dir = Path("issues")
    input_dir = Path("input")
    output_dir = Path("output/band_crop_debug_issues")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    config.debug.visualize_ocr = True

    extractor = BandCropExtractor(config)

    issue_files = sorted(issues_dir.glob("*.png"))
    print(f"Found {len(issue_files)} issue files")

    results: dict[str, list[str]] = {"success": [], "failed": [], "not_found": []}

    for issue_path in issue_files:
        pdf_name, page_num = parse_issue_filename(issue_path)
        pdf_path = find_pdf_in_input(pdf_name, input_dir)

        if pdf_path is None:
            print(f"  [NOT FOUND] {pdf_name}")
            results["not_found"].append((issue_path.name, pdf_name))
            continue

        print(f"Processing: {pdf_path.name} (page {page_num})")

        try:
            pages = convert_from_path(pdf_path, dpi=300)
            target_page = pages[page_num - 1]

            image = np.array(target_page)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            payload_img, is_sig = extractor.build_phi_safe_payload(image)

            if payload_img is not None:
                out_name = issue_path.stem + "_NEW.png"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), cv2.cvtColor(payload_img, cv2.COLOR_RGB2BGR))
                print(f"  -> Saved: {out_path.name}")
                results["success"].append(out_path.name)
            else:
                print(f"  -> FAILED: No payload (signature page)")
                results["failed"].append(issue_path.name)

        except Exception as e:
            print(f"  -> FAILED: {e}")
            results["failed"].append(issue_path.name)

    print("")
    print("=" * 50)
    print("SUMMARY")
    print(f"  Success:   {len(results['success'])}")
    print(f"  Failed:    {len(results['failed'])}")
    print(f"  Not Found: {len(results['not_found'])}")
    print(f"  Output:    {output_dir}/")
    print("")
    print("Comparison:")
    print(f"  Issues (OLD): {issues_dir}/")
    print(f"  Output (NEW): {output_dir}/")
    if results["not_found"]:
        print("")
        print("NOT FOUND FILES:")
        for issue_name, pdf_name in results["not_found"]:
            print(f"  - {pdf_name}")


if __name__ == "__main__":
    main()