"""Create combined comparison outputs from ppocr_grid and vlm_full_page results."""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime

PPOCR_BENCH = "output/ppocr_grid/benchmark_patient_a_week1.xlsx"
VLM_BENCH = "output/vlm_full_page/benchmark_patient_a_week1.xlsx"
PPOCR_MERGED = "output/ppocr_grid/merged_results.xlsx"
VLM_MERGED = "output/vlm_full_page/merged_results.xlsx"

OUTPUT_BENCH = "output/combined/benchmark_combined.xlsx"
OUTPUT_MERGED = "output/combined/merged_combined.xlsx"

# Styles
HEADER_FONT = Font(bold=True, size=11, color="FFFFFF")
HEADER_FILL = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
PPOCR_FILL = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
VLM_FILL = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
SECTION_FILL = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
MATCH_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
MISMATCH_FILL = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)


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


def create_benchmark_combined():
    """Create single-sheet benchmark comparison."""
    wb_ppocr = openpyxl.load_workbook(PPOCR_BENCH)
    wb_vlm = openpyxl.load_workbook(VLM_BENCH)

    ppocr_summary = {
        r[0].value: r[1].value
        for r in wb_ppocr["Run Summary"].iter_rows(min_row=1, values_only=False)
    }
    vlm_summary = {
        r[0].value: r[1].value
        for r in wb_vlm["Run Summary"].iter_rows(min_row=1, values_only=False)
    }

    ppocr_rows = list(wb_ppocr["Row-Level"].iter_rows(min_row=2, values_only=True))
    vlm_rows = list(wb_vlm["Row-Level"].iter_rows(min_row=2, values_only=True))

    ppocr_pages = list(wb_ppocr["Page Details"].iter_rows(min_row=2, values_only=True))
    vlm_pages = list(wb_vlm["Page Details"].iter_rows(min_row=2, values_only=True))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Approach Comparison"

    row = 1

    # --- TITLE ---
    ws.cell(
        row=row,
        column=1,
        value="Four Approaches Comparison: OCR + VLM Fallback vs VLM Full Page",
    ).font = Font(bold=True, size=14)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    row += 1
    ws.cell(
        row=row,
        column=1,
        value=f"File: {ppocr_summary.get('Source File', 'N/A')} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ).font = Font(italic=True, size=10)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    row += 2

    # --- SUMMARY METRICS ---
    ws.cell(row=row, column=1, value="SUMMARY METRICS").font = Font(bold=True, size=12)
    ws.cell(row=row, column=1).fill = SECTION_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    row += 1

    headers = ["Metric", "OCR + VLM Fallback", "VLM Full Page", "Delta (%)", "Winner"]
    for c, h in enumerate(headers, 1):
        ws.cell(row=row, column=c, value=h)
    style_header_row(ws, row, len(headers))
    row += 1

    metrics = [
        ("Total Processing Time (s)", "Total Time (s)", True, False),
        ("Pages Processed", "Total Pages", False, False),
        ("Rows Extracted", "Total Rows Extracted", False, False),
        ("Accepted Rows", "Accepted Count", False, True),
        ("Flagged Rows", "Flagged Count", False, False),
        ("Failed Rows", "Failed Count", False, False),
        ("Mean Confidence", "Mean Confidence", False, True),
        ("Min Confidence", "Min Confidence", False, True),
        ("VLM Fallbacks Triggered", "VLM Fallbacks Triggered", False, False),
        ("Hours Mismatch Rate", "Hours Mismatch Rate", False, False),
        ("Field Missing Rate", "Field Missing Rate", False, False),
        ("Mean CER", "Mean CER", False, False),
    ]

    for label, key, lower_better, higher_better in metrics:
        ppocr_val = ppocr_summary.get(key, "N/A")
        vlm_val = vlm_summary.get(key, "N/A")

        # Compute delta
        delta = ""
        winner = ""
        if isinstance(ppocr_val, (int, float)) and isinstance(vlm_val, (int, float)):
            if ppocr_val != 0:
                delta_pct = ((vlm_val - ppocr_val) / abs(ppocr_val)) * 100
                delta = f"{delta_pct:+.1f}%"
                if higher_better:
                    winner = (
                        "VLM Full Page"
                        if vlm_val > ppocr_val
                        else ("OCR + VLM Fallback" if ppocr_val > vlm_val else "Tie")
                    )
                elif lower_better:
                    winner = (
                        "VLM Full Page"
                        if vlm_val < ppocr_val
                        else ("OCR + VLM Fallback" if ppocr_val < vlm_val else "Tie")
                    )
                else:
                    winner = "—"

        ws.cell(row=row, column=1, value=label)
        ws.cell(row=row, column=2, value=ppocr_val)
        ws.cell(row=row, column=3, value=vlm_val)
        ws.cell(row=row, column=4, value=delta)
        ws.cell(row=row, column=5, value=winner)
        for c in range(1, 6):
            style_data_cell(
                ws, row, c, PPOCR_FILL if c == 2 else VLM_FILL if c == 3 else None
            )
        row += 1

    row += 1

    # --- PAGE-LEVEL TIMING ---
    ws.cell(row=row, column=1, value="PAGE-LEVEL TIMING").font = Font(
        bold=True, size=12
    )
    ws.cell(row=row, column=1).fill = SECTION_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
    row += 1

    page_headers = [
        "Page #",
        "Metric",
        "OCR + VLM Fallback",
        "VLM Full Page",
        "Delta (%)",
        "Winner",
    ]
    for c, h in enumerate(page_headers, 1):
        ws.cell(row=row, column=c, value=h)
    style_header_row(ws, row, len(page_headers))
    row += 1

    page_metrics_labels = [
        (4, "Page Time (s)"),
        (5, "OCR Init Time (s)"),
        (6, "OCR Inference Time (s)"),
        (7, "Layout Detection Time (s)"),
        (8, "Extraction Time (s)"),
        (9, "Validation Time (s)"),
        (10, "Total Boxes Detected"),
        (11, "Rows Extracted"),
        (12, "Empty Rows Skipped"),
        (13, "VLM Fallbacks"),
    ]

    for pp_page, vlm_page in zip(ppocr_pages, vlm_pages):
        page_num = pp_page[1] if pp_page[1] else vlm_page[1]
        ws.cell(row=row, column=1, value=page_num).font = Font(bold=True)
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=2)
        for c in range(1, 7):
            style_data_cell(ws, row, c)
        row += 1

        for col_idx, label in page_metrics_labels:
            pp_val = pp_page[col_idx - 1] if col_idx <= len(pp_page) else 0
            vlm_val = vlm_page[col_idx - 1] if col_idx <= len(vlm_page) else 0

            delta = ""
            winner = ""
            if isinstance(pp_val, (int, float)) and isinstance(vlm_val, (int, float)):
                if pp_val != 0:
                    delta_pct = ((vlm_val - pp_val) / abs(pp_val)) * 100
                    delta = f"{delta_pct:+.1f}%"
                    winner = (
                        "VLM"
                        if vlm_val < pp_val
                        else ("OCR" if pp_val < vlm_val else "Tie")
                    )

            ws.cell(row=row, column=2, value=label)
            ws.cell(row=row, column=3, value=pp_val)
            ws.cell(row=row, column=4, value=vlm_val)
            ws.cell(row=row, column=5, value=delta)
            ws.cell(row=row, column=6, value=winner)
            for c in range(1, 7):
                style_data_cell(
                    ws, row, c, PPOCR_FILL if c == 3 else VLM_FILL if c == 4 else None
                )
            row += 1

    row += 1

    # --- ROW-LEVEL COMPARISON ---
    ws.cell(
        row=row, column=1, value="ROW-LEVEL COMPARISON (matched by date + time_in)"
    ).font = Font(bold=True, size=12)
    ws.cell(row=row, column=1).fill = SECTION_FILL
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=18)
    row += 1

    comp_headers = [
        "Date",
        "Time In (ppocr)",
        "Time In (vlm)",
        "Time In Match",
        "Time Out (ppocr)",
        "Time Out (vlm)",
        "Time Out Match",
        "Hours Written (ppocr)",
        "Hours Written (vlm)",
        "Hours Match",
        "Calc Hours (ppocr)",
        "Calc Hours (vlm)",
        "ppocr Status",
        "vlm Status",
        "Both Accepted",
        "ppocr Validation Errors",
        "vlm Validation Errors",
        "Notes",
    ]
    for c, h in enumerate(comp_headers, 1):
        ws.cell(row=row, column=c, value=h)
    style_header_row(ws, row, len(comp_headers))
    row += 1

    # Build lookup by (date, time_in)
    def row_key(r):
        date_val = str(r[6]) if r[6] else ""  # parsed date
        time_in = str(r[10]) if r[10] else ""  # parsed time in
        return (date_val, time_in)

    ppocr_by_key = {row_key(r): r for r in ppocr_rows}
    vlm_by_key = {row_key(r): r for r in vlm_rows}

    all_keys = sorted(set(list(ppocr_by_key.keys()) + list(vlm_by_key.keys())))

    for key in all_keys:
        pp = ppocr_by_key.get(key)
        vv = vlm_by_key.get(key)

        date_val = key[0]
        pp_time_in = str(pp[10]) if pp and pp[10] else ""
        vlm_time_in = str(vv[10]) if vv and vv[10] else ""
        pp_time_out = str(pp[14]) if pp and pp[14] else ""
        vlm_time_out = str(vv[14]) if vv and vv[14] else ""
        pp_hours = str(pp[18]) if pp and pp[18] else ""
        vlm_hours = str(vv[18]) if vv and vv[18] else ""
        pp_written = str(pp[22]) if pp and pp[22] is not None else ""
        vlm_written = str(vv[22]) if vv and vv[22] is not None else ""
        pp_calc = str(pp[21]) if pp and pp[21] is not None else ""
        vlm_calc = str(vv[21]) if vv and vv[21] is not None else ""
        pp_status = str(pp[24]) if pp and pp[24] else ""
        vlm_status = str(vv[24]) if vv and vv[24] else ""
        pp_errors = str(pp[25]) if pp and pp[25] else ""
        vlm_errors = str(vv[25]) if vv and vv[25] else ""

        time_in_match = pp_time_in == vlm_time_in and pp_time_in != ""
        time_out_match = pp_time_out == vlm_time_out and pp_time_out != ""
        hours_match = pp_hours == vlm_hours and pp_hours != ""
        both_accepted = pp_status == "accepted" and vlm_status == "accepted"

        notes = []
        if not pp:
            notes.append("ppocr: row not extracted")
        if not vv:
            notes.append("vlm: row not extracted")
        if pp and not vv:
            notes.append("VLM missed this row")
        if vv and not pp:
            notes.append("ppocr missed this row")

        ws.cell(row=row, column=1, value=date_val)
        ws.cell(row=row, column=2, value=pp_time_in)
        ws.cell(row=row, column=3, value=vlm_time_in)
        ws.cell(row=row, column=4, value="YES" if time_in_match else "NO")
        ws.cell(row=row, column=5, value=pp_time_out)
        ws.cell(row=row, column=6, value=vlm_time_out)
        ws.cell(row=row, column=7, value="YES" if time_out_match else "NO")
        ws.cell(row=row, column=8, value=pp_written)
        ws.cell(row=row, column=9, value=vlm_written)
        ws.cell(row=row, column=10, value="YES" if hours_match else "NO")
        ws.cell(row=row, column=11, value=pp_calc)
        ws.cell(row=row, column=12, value=vlm_calc)
        ws.cell(row=row, column=13, value=pp_status)
        ws.cell(row=row, column=14, value=vlm_status)
        ws.cell(row=row, column=15, value="YES" if both_accepted else "NO")
        ws.cell(row=row, column=16, value=pp_errors)
        ws.cell(row=row, column=17, value=vlm_errors)
        ws.cell(row=row, column=18, value="; ".join(notes) if notes else "")

        for c in range(1, len(comp_headers) + 1):
            style_data_cell(ws, row, c)

        # Highlight match/mismatch cells
        if time_in_match:
            ws.cell(row=row, column=4).fill = MATCH_FILL
        else:
            ws.cell(row=row, column=4).fill = MISMATCH_FILL

        if time_out_match:
            ws.cell(row=row, column=7).fill = MATCH_FILL
        else:
            ws.cell(row=row, column=7).fill = MISMATCH_FILL

        if hours_match:
            ws.cell(row=row, column=10).fill = MATCH_FILL
        else:
            ws.cell(row=row, column=10).fill = MISMATCH_FILL

        if both_accepted:
            ws.cell(row=row, column=15).fill = MATCH_FILL
        else:
            ws.cell(row=row, column=15).fill = MISMATCH_FILL

        row += 1

    # Column widths
    col_widths = [
        14,
        16,
        16,
        12,
        16,
        16,
        12,
        16,
        16,
        12,
        16,
        16,
        14,
        14,
        12,
        30,
        30,
        30,
    ]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    wb.save(OUTPUT_BENCH)
    print(f"Saved: {OUTPUT_BENCH}")


def create_merged_combined():
    """Create single-sheet merged comparison."""
    wb_ppocr = openpyxl.load_workbook(PPOCR_MERGED)
    wb_vlm = openpyxl.load_workbook(VLM_MERGED)

    ws_ppocr = wb_ppocr.active
    ws_vlm = wb_vlm.active

    ppocr_data = list(ws_ppocr.iter_rows(min_row=2, values_only=True))
    vlm_data = list(ws_vlm.iter_rows(min_row=2, values_only=True))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Row Comparison"

    row = 1

    # --- TITLE ---
    ws.cell(
        row=row,
        column=1,
        value="Row-Level Comparison: OCR + VLM Fallback vs VLM Full Page",
    ).font = Font(bold=True, size=14)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=16)
    row += 1
    ws.cell(
        row=row,
        column=1,
        value=f"File: C.Ferguson Timesheets - 010726-011326.pdf | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ).font = Font(italic=True, size=10)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=16)
    row += 2

    headers = [
        "Date",
        "Time In (ppocr)",
        "Time In (vlm)",
        "Time In Match",
        "Time Out (ppocr)",
        "Time Out (vlm)",
        "Time Out Match",
        "Hours Written (ppocr)",
        "Hours Written (vlm)",
        "Calc Hours (ppocr)",
        "Calc Hours (vlm)",
        "ppocr Status",
        "vlm Status",
        "Both Accepted",
        "ppocr Validation Errors",
        "vlm Validation Errors",
    ]
    for c, h in enumerate(headers, 1):
        ws.cell(row=row, column=c, value=h)
    style_header_row(ws, row, len(headers))
    row += 1

    # Merged columns (0-indexed):
    # 0:Source, 1:Page, 2:Row#, 3:Employee, 4:Patient, 5:Date, 6:DateSrc,
    # 7:TimeIn, 8:TimeInSrc, 9:TimeOut, 10:TimeOutSrc, 11:Hours, 12:HoursSrc,
    # 13:CalcHours, 14:Overnight, 15:Over24h, 16:Confidence, 17:Status, 18:Issues

    def get_date(r):
        return str(r[5]) if len(r) > 5 and r[5] else ""

    ppocr_by_date = {}
    for r in ppocr_data:
        d = get_date(r)
        if d:
            ppocr_by_date[d] = r

    vlm_by_date = {}
    for r in vlm_data:
        d = get_date(r)
        if d:
            vlm_by_date[d] = r

    all_dates = sorted(set(list(ppocr_by_date.keys()) + list(vlm_by_date.keys())))

    for date_val in all_dates:
        pp = ppocr_by_date.get(date_val)
        vv = vlm_by_date.get(date_val)

        pp_time_in = str(pp[7]) if pp and len(pp) > 7 and pp[7] else ""
        vlm_time_in = str(vv[7]) if vv and len(vv) > 7 and vv[7] else ""
        pp_time_out = str(pp[9]) if pp and len(pp) > 9 and pp[9] else ""
        vlm_time_out = str(vv[9]) if vv and len(vv) > 9 and vv[9] else ""
        pp_hours = str(pp[11]) if pp and len(pp) > 11 and pp[11] is not None else ""
        vlm_hours = str(vv[11]) if vv and len(vv) > 11 and vv[11] is not None else ""
        pp_calc = (
            str(pp[13]) if pp and len(pp) > 13 and pp[13] not in ("", None) else ""
        )
        vlm_calc = (
            str(vv[13]) if vv and len(vv) > 13 and vv[13] not in ("", None) else ""
        )
        pp_status = str(pp[17]) if pp and len(pp) > 17 and pp[17] else ""
        vlm_status = str(vv[17]) if vv and len(vv) > 17 and vv[17] else ""
        pp_errors = str(pp[18]) if pp and len(pp) > 18 and pp[18] else ""
        vlm_errors = str(vv[18]) if vv and len(vv) > 18 and vv[18] else ""

        time_in_match = pp_time_in == vlm_time_in and pp_time_in != ""
        time_out_match = pp_time_out == vlm_time_out and pp_time_out != ""
        both_accepted = pp_status == "accepted" and vlm_status == "accepted"

        ws.cell(row=row, column=1, value=date_val)
        ws.cell(row=row, column=2, value=pp_time_in)
        ws.cell(row=row, column=3, value=vlm_time_in)
        ws.cell(row=row, column=4, value="YES" if time_in_match else "NO")
        ws.cell(row=row, column=5, value=pp_time_out)
        ws.cell(row=row, column=6, value=vlm_time_out)
        ws.cell(row=row, column=7, value="YES" if time_out_match else "NO")
        ws.cell(row=row, column=8, value=pp_hours)
        ws.cell(row=row, column=9, value=vlm_hours)
        ws.cell(row=row, column=10, value=pp_calc)
        ws.cell(row=row, column=11, value=vlm_calc)
        ws.cell(row=row, column=12, value=pp_status)
        ws.cell(row=row, column=13, value=vlm_status)
        ws.cell(row=row, column=14, value="YES" if both_accepted else "NO")
        ws.cell(row=row, column=15, value=pp_errors)
        ws.cell(row=row, column=16, value=vlm_errors)

        for c in range(1, len(headers) + 1):
            style_data_cell(ws, row, c)

        if time_in_match:
            ws.cell(row=row, column=4).fill = MATCH_FILL
        else:
            ws.cell(row=row, column=4).fill = MISMATCH_FILL

        if time_out_match:
            ws.cell(row=row, column=7).fill = MATCH_FILL
        else:
            ws.cell(row=row, column=7).fill = MISMATCH_FILL

        if both_accepted:
            ws.cell(row=row, column=14).fill = MATCH_FILL
        else:
            ws.cell(row=row, column=14).fill = MISMATCH_FILL

        row += 1

    col_widths = [14, 16, 16, 12, 16, 16, 12, 18, 18, 16, 16, 14, 14, 12, 30, 30]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    wb.save(OUTPUT_MERGED)
    print(f"Saved: {OUTPUT_MERGED}")


if __name__ == "__main__":
    create_benchmark_combined()
    create_merged_combined()
    print("Done.")
