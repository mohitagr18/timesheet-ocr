"""Compare extraction results against ground truth data.

Reads ground truth from output/ground_truth.xlsx, compares against
all approach outputs, and adds a 'Human-Verified Results' sheet to
benchmark_combined.xlsx.

Usage:
    uv run python scripts/compare_ground_truth.py
"""

import glob
import os
import re
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

GROUND_TRUTH_PATH = "output/ground_truth.xlsx"
COMBINED_PATH = "output/combined/benchmark_combined.xlsx"
NAME_DB_PATH = "output/name_mapping.db"

APPROACHES = [
    ("ocr_only", "OCR Only (Baseline)"),
    ("ppocr_grid", "OCR + VLM Fallback"),
    ("vlm_full_page", "VLM Full Page"),
    ("layout_guided_vlm_local", "Layout-Guided VLM (Local)"),
    ("layout_guided_vlm_cloud", "Layout-Guided VLM (Cloud)"),
]

HEADER_FONT = Font(bold=True, size=11, color="FFFFFF")
HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
SECTION_FILL = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
MATCH_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
MISMATCH_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)

HOURS_TOLERANCE = 0.25  # ±15 minutes


def parse_time(time_str):
    """Parse time string like '7:00 AM' to minutes since midnight."""
    if not time_str or not isinstance(time_str, str):
        return None
    time_str = time_str.strip()
    match = re.match(r"(\d{1,2}):?(\d{2})?\s*(AM|PM|am|pm|Am|Pm)?", time_str)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3)
    if ampm and ampm.lower() == "pm" and hour != 12:
        hour += 12
    elif ampm and ampm.lower() == "am" and hour == 12:
        hour = 0
    return hour * 60 + minute


def parse_hours(hours_val):
    """Parse hours to float."""
    if hours_val is None:
        return None
    if isinstance(hours_val, (int, float)):
        return float(hours_val)
    try:
        return float(str(hours_val).strip())
    except (ValueError, TypeError):
        return None


def hours_match(extracted, ground_truth, tolerance=HOURS_TOLERANCE):
    """Check if extracted hours match ground truth within tolerance."""
    ext = parse_hours(extracted)
    gt = parse_hours(ground_truth)
    if ext is None or gt is None:
        return False
    return abs(ext - gt) <= tolerance


def time_match(extracted, ground_truth, tolerance_minutes=30):
    """Check if extracted time matches ground truth within tolerance."""
    ext = parse_time(str(extracted) if extracted else "")
    gt = parse_time(str(ground_truth) if ground_truth else "")
    if ext is None or gt is None:
        return False
    return abs(ext - gt) <= tolerance_minutes


def load_ground_truth():
    """Load ground truth data from Excel."""
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"ERROR: Ground truth file not found: {GROUND_TRUTH_PATH}")
        print("Please fill in output/ground_truth.xlsx first.")
        return None

    wb = openpyxl.load_workbook(GROUND_TRUTH_PATH, read_only=True)
    ws = wb.active
    rows = []
    header = None
    for row in ws.iter_rows(values_only=True):
        if header is None:
            header = [str(c).strip() if c else "" for c in row]
            continue
        if not any(row):
            continue
        record = dict(zip(header, row))
        if record.get("source_file") and record.get("date"):
            rows.append(record)
    wb.close()
    print(f"Loaded {len(rows)} ground truth rows")
    return rows


def load_name_mapping():
    """Load name mapping from SQLite DB."""
    mapping = {}
    if not os.path.exists(NAME_DB_PATH):
        print(f"WARNING: Name mapping DB not found: {NAME_DB_PATH}")
        return mapping

    import sqlite3

    conn = sqlite3.connect(NAME_DB_PATH)
    conn.row_factory = sqlite3.Row

    for table, label in [("patients", "patient"), ("employees", "employee")]:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        for row in rows:
            mapping[row["anonymized_id"]] = {
                "real_name": row["real_name"],
                "label": label,
                "source_files": (row["source_files"] or "").split(","),
            }

    conn.close()
    print(f"Loaded {len(mapping)} name mappings from DB")
    return mapping


def get_anonymized_name(source_file, name_mapping):
    """Map original filename to anonymized name via SQLite DB."""
    stem = Path(source_file).stem
    for anon_id, info in name_mapping.items():
        if source_file in info["source_files"] or stem in info["source_files"]:
            return anon_id
    return None


def load_approach_data(approach_id):
    """Load all benchmark data for an approach."""
    bench_files = sorted(
        f
        for f in glob.glob(f"output/{approach_id}/benchmark_*.xlsx")
        if "combined" not in f
    )

    all_rows = []
    for path in bench_files:
        wb = openpyxl.load_workbook(path, read_only=True)
        if "Row-Level" in wb.sheetnames:
            ws = wb["Row-Level"]
            header = None
            for row in ws.iter_rows(values_only=True):
                if header is None:
                    header = [str(c).strip() if c else "" for c in row]
                    continue
                if not any(row):
                    continue
                record = dict(zip(header, row))
                all_rows.append(record)
        wb.close()

    # Also load merged results for employee name
    merged_path = f"output/{approach_id}/merged_results.xlsx"
    if os.path.exists(merged_path):
        wb = openpyxl.load_workbook(merged_path, read_only=True)
        ws = wb.active
        header = None
        for row in ws.iter_rows(values_only=True):
            if header is None:
                header = [str(c).strip() if c else "" for c in row]
                continue
            if not any(row):
                continue
            record = dict(zip(header, row))
            # Merge employee name into benchmark rows
            all_rows.append(record)
        wb.close()

    print(f"  {approach_id}: {len(all_rows)} rows loaded")
    return all_rows


def compare_approach(ground_truth, approach_id, name_mapping):
    """Compare approach output against ground truth."""
    approach_rows = load_approach_data(approach_id)

    results = []
    for gt_row in ground_truth:
        source_file = gt_row["source_file"]
        gt_date = str(gt_row["date"]).strip()
        gt_hours = gt_row.get("total_hours")
        gt_time_in = gt_row.get("time_in")
        gt_time_out = gt_row.get("time_out")
        gt_employee = gt_row.get("employee_name", "")

        # Find matching row in approach output
        matched = None
        for ar in approach_rows:
            ar_date = str(ar.get("Date", "")).strip()
            if ar_date == gt_date or ar_date.endswith(gt_date):
                matched = ar
                break

        if matched:
            ext_hours = matched.get("Hours")
            ext_status = matched.get("Status", "")
            ext_time_in = matched.get("Time In", "")
            ext_time_out = matched.get("Time Out", "")
            ext_employee = matched.get("Employee", "")

            hours_ok = hours_match(ext_hours, gt_hours)
            time_in_ok = time_match(ext_time_in, gt_time_in)
            time_out_ok = time_match(ext_time_out, gt_time_out)
            status_ok = ext_status.lower() == "accepted" if hours_ok else False

            results.append(
                {
                    "source_file": source_file,
                    "date": gt_date,
                    "gt_hours": gt_hours,
                    "gt_time_in": gt_time_in,
                    "gt_time_out": gt_time_out,
                    "gt_employee": gt_employee,
                    "ext_hours": ext_hours,
                    "ext_time_in": ext_time_in,
                    "ext_time_out": ext_time_out,
                    "ext_status": ext_status,
                    "ext_employee": ext_employee,
                    "hours_correct": hours_ok,
                    "time_in_correct": time_in_ok,
                    "time_out_correct": time_out_ok,
                    "status_correct": status_ok,
                    "all_correct": hours_ok and time_in_ok and time_out_ok,
                }
            )
        else:
            results.append(
                {
                    "source_file": source_file,
                    "date": gt_date,
                    "gt_hours": gt_hours,
                    "gt_time_in": gt_time_in,
                    "gt_time_out": gt_time_out,
                    "gt_employee": gt_employee,
                    "ext_hours": None,
                    "ext_time_in": None,
                    "ext_time_out": None,
                    "ext_status": "not extracted",
                    "ext_employee": None,
                    "hours_correct": False,
                    "time_in_correct": False,
                    "time_out_correct": False,
                    "status_correct": False,
                    "all_correct": False,
                }
            )

    return results


def compute_metrics(all_approach_results):
    """Compute accuracy metrics per approach."""
    metrics = {}
    for approach_id, approach_label, approach_results in all_approach_results:
        total = len(approach_results)
        if total == 0:
            continue

        hours_correct = sum(1 for r in approach_results if r["hours_correct"])
        time_in_correct = sum(1 for r in approach_results if r["time_in_correct"])
        time_out_correct = sum(1 for r in approach_results if r["time_out_correct"])
        all_correct = sum(1 for r in approach_results if r["all_correct"])

        # Precision: of rows marked "accepted", how many were actually correct?
        accepted = [
            r for r in approach_results if r["ext_status"].lower() == "accepted"
        ]
        accepted_correct = sum(1 for r in accepted if r["all_correct"])
        precision = accepted_correct / len(accepted) if accepted else 0.0

        # Recall: of all correct rows, how many did the pipeline accept?
        correct_rows = [r for r in approach_results if r["all_correct"]]
        correct_accepted = sum(
            1 for r in correct_rows if r["ext_status"].lower() == "accepted"
        )
        recall = correct_accepted / len(correct_rows) if correct_rows else 0.0

        # F1
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # False positive: accepted but wrong
        false_pos = sum(1 for r in accepted if not r["all_correct"])
        fpr = false_pos / len(accepted) if accepted else 0.0

        # False negative: not accepted but correct
        not_accepted = [
            r for r in approach_results if r["ext_status"].lower() != "accepted"
        ]
        false_neg = sum(1 for r in not_accepted if r["all_correct"])
        fnr = false_neg / len(correct_rows) if correct_rows else 0.0

        metrics[approach_id] = {
            "label": approach_label,
            "total_rows": total,
            "row_accuracy": all_correct / total,
            "hours_accuracy": hours_correct / total,
            "time_in_accuracy": time_in_correct / total,
            "time_out_accuracy": time_out_correct / total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
        }

    return metrics


def style_cell(cell, fill=None):
    cell.border = THIN_BORDER
    cell.alignment = Alignment(horizontal="center", wrap_text=True)
    if fill:
        cell.fill = fill


def write_human_verified_sheet(wb, metrics, all_approach_results, ground_truth):
    """Add Human-Verified Results sheet to the workbook."""
    ws = wb.create_sheet("Human-Verified Results")

    row = 1
    ws.cell(row=row, column=1, value="Human-Verified Accuracy Results").font = Font(
        bold=True, size=14
    )
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    row += 1
    ws.cell(
        row=row,
        column=1,
        value=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ).font = Font(italic=True, size=10)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    row += 2

    # Section 1: Accuracy Metrics
    ws.cell(row=row, column=1, value="ACCURACY METRICS").font = Font(bold=True, size=12)
    ws.cell(row=row, column=1).fill = SECTION_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    row += 1

    metric_headers = ["Metric"] + [m["label"] for m in metrics.values()]
    for c, h in enumerate(metric_headers, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = THIN_BORDER
    row += 1

    metric_rows = [
        ("Total Rows Compared", lambda m: m["total_rows"]),
        ("Row Accuracy", lambda m: f"{m['row_accuracy']:.1%}"),
        ("Hours Accuracy (±0.25h)", lambda m: f"{m['hours_accuracy']:.1%}"),
        ("Time In Accuracy", lambda m: f"{m['time_in_accuracy']:.1%}"),
        ("Time Out Accuracy", lambda m: f"{m['time_out_accuracy']:.1%}"),
        ("Precision", lambda m: f"{m['precision']:.1%}"),
        ("Recall", lambda m: f"{m['recall']:.1%}"),
        ("F1 Score", lambda m: f"{m['f1_score']:.3f}"),
        ("False Positive Rate", lambda m: f"{m['false_positive_rate']:.1%}"),
        ("False Negative Rate", lambda m: f"{m['false_negative_rate']:.1%}"),
    ]

    for label, getter in metric_rows:
        ws.cell(row=row, column=1, value=label).border = THIN_BORDER
        ws.cell(row=row, column=1).alignment = Alignment(horizontal="left")
        for c, (aid, m) in enumerate(metrics.items(), 2):
            val = getter(m)
            ws.cell(row=row, column=c, value=val).border = THIN_BORDER
            ws.cell(row=row, column=c).alignment = Alignment(horizontal="center")
        row += 1

    row += 1

    # Section 2: Per-Row Detail
    ws.cell(row=row, column=1, value="PER-ROW COMPARISON").font = Font(
        bold=True, size=12
    )
    ws.cell(row=row, column=1).fill = SECTION_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
    row += 1

    detail_headers = ["Source File", "Date", "GT Hours"]
    for approach_id, approach_label in APPROACHES:
        detail_headers.append(f"{approach_label} — Hours")
        detail_headers.append(f"{approach_label} — Correct?")

    for c, h in enumerate(detail_headers, 1):
        cell = ws.cell(row=row, column=c, value=h)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = THIN_BORDER
    row += 1

    for gt_row in ground_truth:
        source_file = gt_row["source_file"]
        gt_date = str(gt_row["date"]).strip()
        gt_hours = gt_row.get("total_hours", "")

        ws.cell(row=row, column=1, value=source_file).border = THIN_BORDER
        ws.cell(row=row, column=1).alignment = Alignment(
            horizontal="left", wrap_text=True
        )
        ws.cell(row=row, column=2, value=gt_date).border = THIN_BORDER
        ws.cell(row=row, column=3, value=gt_hours).border = THIN_BORDER

        col = 4
        for approach_id, approach_label in APPROACHES:
            for _, _, approach_results in all_approach_results:
                if approach_id == approach_id:
                    matched = None
                    for r in approach_results:
                        if r["date"] == gt_date and r["source_file"] == source_file:
                            matched = r
                            break

                    if matched:
                        ws.cell(
                            row=row, column=col, value=matched["ext_hours"]
                        ).border = THIN_BORDER
                        col += 1
                        correct = matched["all_correct"]
                        ws.cell(
                            row=row, column=col, value="YES" if correct else "NO"
                        ).border = THIN_BORDER
                        ws.cell(row=row, column=col).fill = (
                            MATCH_FILL if correct else MISMATCH_FILL
                        )
                    else:
                        ws.cell(row=row, column=col, value="").border = THIN_BORDER
                        col += 1
                        ws.cell(row=row, column=col, value="N/A").border = THIN_BORDER
                    col += 1
                    break
        row += 1

    # Column widths
    ws.column_dimensions["A"].width = 50
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 14
    for i in range(4, len(detail_headers) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = 22


def main():
    print("=" * 60)
    print("Ground Truth Comparison")
    print("=" * 60)

    # Load ground truth
    ground_truth = load_ground_truth()
    if ground_truth is None:
        return

    # Load name mapping
    name_mapping = load_name_mapping()

    # Compare each approach
    all_approach_results = []
    for approach_id, approach_label in APPROACHES:
        print(f"\nComparing {approach_label}...")
        results = compare_approach(ground_truth, approach_id, name_mapping)
        all_approach_results.append((approach_id, approach_label, results))

    # Compute metrics
    metrics = compute_metrics(all_approach_results)

    print("\n" + "=" * 60)
    print("ACCURACY METRICS")
    print("=" * 60)
    print(f"{'Metric':<30}", end="")
    for _, m in metrics.items():
        print(f" {m['label'][:20]:<20}", end="")
    print()
    print("-" * 60)

    metric_labels = [
        ("Row Accuracy", lambda m: f"{m['row_accuracy']:.1%}"),
        ("Hours Accuracy (±0.25h)", lambda m: f"{m['hours_accuracy']:.1%}"),
        ("Precision", lambda m: f"{m['precision']:.1%}"),
        ("Recall", lambda m: f"{m['recall']:.1%}"),
        ("F1 Score", lambda m: f"{m['f1_score']:.3f}"),
    ]
    for label, getter in metric_labels:
        print(f"{label:<30}", end="")
        for _, m in metrics.items():
            print(f" {getter(m):<20}", end="")
        print()

    # Write to combined Excel
    if os.path.exists(COMBINED_PATH):
        wb = openpyxl.load_workbook(COMBINED_PATH)
    else:
        wb = openpyxl.Workbook()

    # Remove existing sheet if present
    if "Human-Verified Results" in wb.sheetnames:
        del wb["Human-Verified Results"]

    write_human_verified_sheet(wb, metrics, all_approach_results, ground_truth)
    wb.save(COMBINED_PATH)
    print(f"\nResults saved to {COMBINED_PATH}")


if __name__ == "__main__":
    main()
