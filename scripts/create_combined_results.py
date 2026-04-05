"""Create combined comparison outputs from all 5 approach results."""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime
import os

APPROACHES = [
    ("ppocr_grid", "OCR + VLM Fallback", "E2EFDA"),
    ("vlm_full_page", "VLM Full Page", "D6E4F0"),
    ("layout_guided_vlm_local", "Layout-Guided VLM (Local)", "FFF2CC"),
    ("layout_guided_vlm_cloud", "Layout-Guided VLM (Cloud)", "FCE4EC"),
]

OUTPUT_DIR = "output/combined"
BENCH_OUTPUT = f"{OUTPUT_DIR}/benchmark_combined.xlsx"
MERGED_OUTPUT = f"{OUTPUT_DIR}/merged_combined.xlsx"

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

FILLS = {}
for _, _, color in APPROACHES:
    FILLS[color] = PatternFill(start_color=color, end_color=color, fill_type="solid")


def style_header_row(ws, row, max_col):
    for col in range(1, max_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = THIN_BORDER


def style_data_cell(ws, row, col, fill=None):
    cell = ws.cell(row=row, column=col)
    cell.border = THIN_BORDER
    cell.alignment = Alignment(horizontal="center", wrap_text=True)
    if fill:
        cell.fill = fill


def load_benchmark(folder):
    path = f"output/{folder}/benchmark_patient_a_week1.xlsx"
    if not os.path.exists(path):
        return None
    wb = openpyxl.load_workbook(path)
    summary = {}
    for r in wb["Run Summary"].iter_rows(min_row=1, values_only=False):
        key = r[0].value
        val = r[1].value
        if key:
            summary[key] = val
    pages = list(wb["Page Details"].iter_rows(min_row=2, values_only=True))
    rows = list(wb["Row-Level"].iter_rows(min_row=2, values_only=True))
    return {"summary": summary, "pages": pages, "rows": rows}


def load_merged(folder):
    path = f"output/{folder}/merged_results.xlsx"
    if not os.path.exists(path):
        return None
    wb = openpyxl.load_workbook(path)
    ws = wb.active
    return list(ws.iter_rows(min_row=2, values_only=True))


def create_benchmark_combined():
    """Create benchmark comparison grouped by approach with IEEE-paper-ready format."""
    data = {}
    for folder, label, color in APPROACHES:
        d = load_benchmark(folder)
        if d:
            d["_color"] = color
            data[folder] = {"label": label, **d}

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Approach Comparison"

    row = 1
    ws.cell(
        row=row, column=1, value="Approach Comparison: Handwritten Timesheet OCR"
    ).font = Font(bold=True, size=14)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 1
    ws.cell(
        row=row,
        column=1,
        value=f"File: C.Ferguson Timesheets - 010726-011326.pdf | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ).font = Font(italic=True, size=10)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 2

    # SUMMARY METRICS
    ws.cell(row=row, column=1, value="SUMMARY METRICS").font = Font(bold=True, size=12)
    ws.cell(row=row, column=1).fill = SECTION_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 1

    headers = ["Metric"] + [d["label"] for d in data.values()] + ["Best"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=row, column=c, value=h)
    style_header_row(ws, row, len(headers))
    row += 1

    metrics = [
        ("Total Processing Time (s)", "Total Processing Time (s)", True),
        ("Pages Processed", "Number of Pages", False),
        ("Rows Extracted", "Total Rows Extracted", False),
        ("Accepted Rows", "Accepted Rows", False, True),
        ("Flagged Rows", "Flagged Rows", False, False),
        ("Failed Rows", "Failed Rows", False, False),
        ("Mean Confidence", "Mean Overall Confidence", False, True),
        ("Min Confidence", "Min Overall Confidence", False, True),
        ("VLM Fallbacks Triggered", "VLM Fallbacks Triggered", False, False),
        ("Hours Mismatch Rate", "Hours Mismatch Rate", False, False),
        ("Field Missing Rate", "Field Missing Rate", False, False),
        ("Mean CER", "Mean Character Error Rate", False, False),
    ]

    for metric_def in metrics:
        label = metric_def[0]
        key = metric_def[1]
        lower_better = metric_def[2] if len(metric_def) > 2 else False
        higher_better = metric_def[3] if len(metric_def) > 3 else False

        ws.cell(row=row, column=1, value=label)
        vals = []
        for i, (folder, d) in enumerate(data.items()):
            val = d["summary"].get(key, "N/A")
            ws.cell(row=row, column=i + 2, value=val)
            style_data_cell(ws, row, i + 2, FILLS.get(d.get("_color", ""), None))
            if isinstance(val, (int, float)):
                vals.append((val, folder))

        best = ""
        if vals:
            if higher_better:
                best_folder = max(vals, key=lambda x: x[0])[1]
                best = data[best_folder]["label"]
            elif lower_better:
                best_folder = min(vals, key=lambda x: x[0])[1]
                best = data[best_folder]["label"]

        ws.cell(row=row, column=len(headers), value=best)
        style_data_cell(ws, row, len(headers))
        row += 1

    row += 1

    # ROW-LEVEL COMPARISON (transposed: dates as columns, metrics as rows)
    ws.cell(
        row=row, column=1, value="ROW-LEVEL COMPARISON (matched by date + time_in)"
    ).font = Font(bold=True, size=12)
    ws.cell(row=row, column=1).fill = SECTION_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 1

    approach_rows = {}
    for folder, d in data.items():
        by_key = {}
        for r in d["rows"]:
            date_val = str(r[6]) if len(r) > 6 and r[6] else ""
            time_in = str(r[10]) if len(r) > 10 and r[10] else ""
            by_key[(date_val, time_in)] = r
        approach_rows[folder] = by_key

    all_keys = set()
    for by_key in approach_rows.values():
        all_keys.update(by_key.keys())
    all_keys = sorted(all_keys)

    # Transposed: first column = metric labels, remaining columns = dates
    transposed_headers = ["Metric"] + [k[0] for k in all_keys]
    for c, h in enumerate(transposed_headers, 1):
        ws.cell(row=row, column=c, value=h)
    style_header_row(ws, row, len(transposed_headers))
    row += 1

    for folder, d in data.items():
        short = d["label"][:15]
        fill_color = FILLS.get(d.get("_color", ""), None)

        # Hours row
        ws.cell(row=row, column=1, value=f"Hours ({short})")
        ws.cell(row=row, column=1).border = THIN_BORDER
        ws.cell(row=row, column=1).alignment = Alignment(
            horizontal="left", wrap_text=True
        )
        ws.cell(row=row, column=1).fill = fill_color
        for col_idx, key in enumerate(all_keys, 2):
            r = approach_rows[folder].get(key)
            hours = str(r[18]) if r and len(r) > 18 and r[18] is not None else ""
            ws.cell(row=row, column=col_idx, value=hours)
            style_data_cell(ws, row, col_idx, fill_color)
        row += 1

        # Status row
        ws.cell(row=row, column=1, value=f"Status ({short})")
        ws.cell(row=row, column=1).border = THIN_BORDER
        ws.cell(row=row, column=1).alignment = Alignment(
            horizontal="left", wrap_text=True
        )
        ws.cell(row=row, column=1).fill = fill_color
        for col_idx, key in enumerate(all_keys, 2):
            r = approach_rows[folder].get(key)
            status = str(r[24]) if r and len(r) > 24 and r[24] else ""
            if not r:
                status = "not extracted"
            fill = fill_color
            if status == "accepted":
                fill = MATCH_FILL
            elif status == "not extracted":
                fill = MISMATCH_FILL
            ws.cell(row=row, column=col_idx, value=status)
            style_data_cell(ws, row, col_idx, fill)
        row += 1

    ws.column_dimensions["A"].width = 32
    for i in range(2, len(transposed_headers) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = 16

    wb.save(BENCH_OUTPUT)
    print(f"Saved: {BENCH_OUTPUT}")


def create_merged_combined():
    """Create simplified merged comparison with Hours + Status only per approach."""
    data = {}
    for folder, label, _ in APPROACHES:
        rows = load_merged(folder)
        if rows:
            data[folder] = {"label": label, "rows": rows}

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Row Comparison"

    row = 1
    ws.cell(
        row=row, column=1, value="Row-Level Comparison: All Approaches"
    ).font = Font(bold=True, size=14)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 1
    ws.cell(
        row=row,
        column=1,
        value=f"File: C.Ferguson Timesheets - 010726-011326.pdf | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ).font = Font(italic=True, size=10)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    row += 2

    # Merged columns (0-indexed):
    # 0:Source, 1:Page, 2:Row#, 3:Employee, 4:Patient, 5:Date, 6:DateSrc,
    # 7:TimeIn, 8:TimeInSrc, 9:TimeOut, 10:TimeOutSrc, 11:Hours, 12:HoursSrc,
    # 13:CalcHours, 14:Overnight, 15:Over24h, 16:Confidence, 17:Status, 18:Issues

    approach_by_date = {}
    for folder, d in data.items():
        by_date = {}
        for r in d["rows"]:
            date_val = str(r[5]) if len(r) > 5 and r[5] else ""
            if date_val:
                by_date[date_val] = r
        approach_by_date[folder] = by_date

    all_dates = set()
    for by_date in approach_by_date.values():
        all_dates.update(by_date.keys())
    all_dates = sorted(all_dates)

    # Transposed: first column = metric labels, remaining columns = dates
    transposed_headers = ["Metric"] + all_dates
    for c, h in enumerate(transposed_headers, 1):
        ws.cell(row=row, column=c, value=h)
    style_header_row(ws, row, len(transposed_headers))
    row += 1

    for folder, d in data.items():
        short = d["label"][:15]

        # Hours row
        ws.cell(row=row, column=1, value=f"Hours ({short})")
        ws.cell(row=row, column=1).border = THIN_BORDER
        ws.cell(row=row, column=1).alignment = Alignment(
            horizontal="left", wrap_text=True
        )
        for col_idx, date_val in enumerate(all_dates, 2):
            r = approach_by_date[folder].get(date_val)
            hours = str(r[11]) if r and len(r) > 11 and r[11] is not None else ""
            ws.cell(row=row, column=col_idx, value=hours)
            style_data_cell(ws, row, col_idx)
        row += 1

        # Status row
        ws.cell(row=row, column=1, value=f"Status ({short})")
        ws.cell(row=row, column=1).border = THIN_BORDER
        ws.cell(row=row, column=1).alignment = Alignment(
            horizontal="left", wrap_text=True
        )
        for col_idx, date_val in enumerate(all_dates, 2):
            r = approach_by_date[folder].get(date_val)
            status = str(r[17]) if r and len(r) > 17 and r[17] else ""
            if not r:
                status = "not extracted"
            fill = None
            if status == "accepted":
                fill = MATCH_FILL
            elif status == "not extracted":
                fill = MISMATCH_FILL
            ws.cell(row=row, column=col_idx, value=status)
            style_data_cell(ws, row, col_idx, fill)
        row += 1

    ws.column_dimensions["A"].width = 32
    for i in range(2, len(transposed_headers) + 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = 16

    wb.save(MERGED_OUTPUT)
    print(f"Saved: {MERGED_OUTPUT}")


def copy_debug_images():
    """Copy debug images to combined folder."""
    import shutil

    for folder, label, _ in APPROACHES:
        debug_dir = f"output/{folder}/debug"
        if os.path.exists(debug_dir):
            for f in os.listdir(debug_dir):
                src = os.path.join(debug_dir, f)
                dst = os.path.join(OUTPUT_DIR, "debug", f"{folder}_{f}")
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"Copied: {dst}")


if __name__ == "__main__":
    os.makedirs(f"{OUTPUT_DIR}/debug", exist_ok=True)
    create_benchmark_combined()
    create_merged_combined()
    copy_debug_images()
    print("Done.")
