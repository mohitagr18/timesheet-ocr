"""Pipeline orchestrator — ties all components together for end-to-end extraction."""

from __future__ import annotations

import logging
import time as time_module
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .confidence import Route, boxes_in_zone, route_by_confidence, should_fallback_entire_row
from .exporter import export_results
from .layout import detect_layout
from .models import (
    ExtractionResult,
    OcrSource,
    RowStatus,
    TimesheetRecord,
    TimesheetRow,
)
from .ocr_engine import OcrEngine
from .parser import clean_name, parse_date, parse_hours, parse_time
from .preprocessing import load_image, pdf_to_images, preprocess_image
from .review_queue import build_review_queue
from .validation import validate_record
from .vlm_fallback import VlmFallback

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)


class Pipeline:
    """End-to-end extraction pipeline for timesheet documents.

    Orchestrates: load → preprocess → layout → OCR → fallback → parse → validate → export.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ocr_engine = OcrEngine(config)
        self.vlm = VlmFallback(config)

    def process_file(self, file_path: str | Path) -> ExtractionResult:
        """Process a single file (PDF or image) through the full pipeline."""
        file_path = Path(file_path)
        start_time = time_module.time()

        logger.info(f"{'=' * 60}")
        logger.info(f"Processing: {file_path.name}")
        logger.info(f"{'=' * 60}")

        # 1. Load file → list of images (one per page)
        images = self._load_file(file_path)

        # 2. Process each page
        records: list[TimesheetRecord] = []
        for page_idx, image in enumerate(images):
            logger.info(f"\n--- Page {page_idx + 1}/{len(images)} ---")
            
            page_start_time = time_module.time()
            record = self._process_page(image, file_path.name, page_idx + 1)
            page_elapsed = time_module.time() - page_start_time
            
            logger.info(f"Page {page_idx + 1} processing complete in {page_elapsed:.1f}s")
            records.append(record)

        # 2.5 Aggregate Document-Level Metadata
        # (E.g. Page 1 has shifts/patient name, Page 2 has the employee signature)
        best_emp = max(records, key=lambda r: (r.employee_name_confidence, len(r.employee_name))) if records else None
        best_pat = max(records, key=lambda r: (r.patient_name_confidence, len(r.patient_name))) if records else None
        
        for r in records:
            if not r.employee_name and best_emp and best_emp.employee_name:
                r.employee_name = best_emp.employee_name
                r.employee_name_confidence = best_emp.employee_name_confidence
            if not r.patient_name and best_pat and best_pat.patient_name:
                r.patient_name = best_pat.patient_name
                r.patient_name_confidence = best_pat.patient_name_confidence

        # 3. Build result
        elapsed = time_module.time() - start_time
        result = ExtractionResult(
            source_file=file_path.name,
            processing_time_seconds=elapsed,
            total_pages=len(images),
            records=records,
        )

        # 4. Build review queue
        result.review_items = build_review_queue(result, self.config)

        # 5. Export
        output_files = export_results(result, self.config)

        # 6. Summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"DONE: {file_path.name}")
        logger.info(f"  Pages: {result.total_pages}")
        logger.info(f"  Rows:  {result.total_rows} (accepted={result.accepted_count}, flagged={result.flagged_count}, failed={result.failed_count})")
        logger.info(f"  Time:  {elapsed:.1f}s")
        logger.info(f"  Files: {[f.name for f in output_files]}")
        logger.info(f"{'=' * 60}")

        return result

    def process_directory(self, input_dir: str | Path | None = None) -> list[ExtractionResult]:
        """Process all supported files in a directory."""
        if input_dir is None:
            input_dir = self.config.input_path
        else:
            input_dir = Path(input_dir)

        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return []

        # Find all supported files
        supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        files = sorted(
            f for f in input_dir.iterdir()
            if f.suffix.lower() in supported and not f.name.startswith(".")
        )

        if not files:
            logger.warning(f"No supported files found in {input_dir}")
            return []

        # Find already processed files in the merged Excel sheet to skip them
        processed_files = set()
        xlsx_path = self.config.output_path / "merged_results.xlsx"
        if xlsx_path.exists():
            try:
                from openpyxl import load_workbook
                wb = load_workbook(xlsx_path, read_only=True)
                if self.config.export.excel_sheet_name in wb.sheetnames:
                    ws = wb[self.config.export.excel_sheet_name]
                else:
                    ws = wb.active
                
                # Source File is column A (1-indexed)
                for row in ws.iter_rows(min_row=2, max_col=1, values_only=True):
                    if row[0]:
                        processed_files.add(str(row[0]))
                if processed_files:
                    logger.info(f"Found {len(processed_files)} previously processed file(s) in Excel.")
            except Exception as e:
                logger.warning(f"Could not read existing Excel to find processed files: {e}")

        logger.info(f"Found {len(files)} file(s) in input directory")

        results = []
        for file_path in files:
            if file_path.name in processed_files:
                logger.info(f"Skipping {file_path.name}: already processed and in Excel.")
                continue
                
            try:
                result = self.process_file(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)

        return results

    def _load_file(self, file_path: Path) -> list[np.ndarray]:
        """Load a file as a list of images (handles both PDFs and images)."""
        if file_path.suffix.lower() == ".pdf":
            return pdf_to_images(file_path, dpi=self.config.preprocessing.target_dpi)
        else:
            return [load_image(file_path)]

    def _process_page(self, image: np.ndarray, source_file: str, page_number: int) -> TimesheetRecord:
        """Process a single page image through OCR + validation."""
        preprocessed = preprocess_image(image, self.config)

        employee_name = ""
        employee_conf = 0.0
        patient_name = ""
        patient_conf = 0.0
        rows = []

        if getattr(self.config, "extraction_mode", "ppocr_grid") == "vlm_full_page":
            vlm_results = self.vlm.extract_full_page(image)
            
            patient_name = vlm_results.get("recipient_name", "")
            patient_conf = 0.9 if patient_name else 0.0
            
            employee_name = vlm_results.get("rn_lpn_name", "") # RN/LPN mapped to employee
            employee_conf = 0.9 if employee_name else 0.0
            
            valid_row_idx = 0
            for row_data in vlm_results.get("shifts", []):
                date_text = row_data.get("date", "").strip()
                time_in_text = row_data.get("time_in", "").strip()
                time_out_text = row_data.get("time_out", "").strip()
                hours_text = row_data.get("total_hours", "").strip()
                notes_text = row_data.get("notes", "").strip()
                
                # Robust Post-Processing: Drop hallucinated empty shifts
                if not time_in_text and not time_out_text and not hours_text:
                    continue
                
                rows.append(TimesheetRow(
                    row_index=valid_row_idx,
                    date_text=date_text,
                    date_parsed=parse_date(date_text),
                    time_in_text=time_in_text,
                    time_in_parsed=parse_time(time_in_text),
                    time_out_text=time_out_text,
                    time_out_parsed=parse_time(time_out_text),
                    total_hours_text=hours_text,
                    total_hours_parsed=parse_hours(hours_text),
                    notes=notes_text.strip(),
                    date_confidence=0.9, 
                    time_in_confidence=0.9,
                    time_out_confidence=0.9,
                    hours_confidence=0.9,
                    date_source=OcrSource.VLM,
                    time_in_source=OcrSource.VLM,
                    time_out_source=OcrSource.VLM,
                    hours_source=OcrSource.VLM,
                ))
                valid_row_idx += 1
        else:
            layout = detect_layout(image, self.config)
            ocr_result = self.ocr_engine.run(preprocessed)

            header_boxes = boxes_in_zone(
                ocr_result.boxes,
                layout.header_zone.x_start,
                layout.header_zone.y_start,
                layout.header_zone.x_end,
                layout.header_zone.y_end,
            )

            if header_boxes:
                sorted_header = sorted(header_boxes, key=lambda b: (b.y_center, b.x_center))
                if sorted_header:
                    employee_name = clean_name(sorted_header[0].text)
                    employee_conf = sorted_header[0].confidence
                if len(sorted_header) > 1:
                    patient_name = clean_name(sorted_header[1].text)
                    patient_conf = sorted_header[1].confidence

            for row_idx, row_zone in enumerate(layout.row_zones):
                row = self._extract_row(
                    image=preprocessed,
                    preprocessed=preprocessed,
                    all_boxes=ocr_result.boxes,
                    row_zone=row_zone,
                    row_idx=row_idx,
                    layout=layout,
                )
                if row is not None:
                    rows.append(row)

        # 6. Build record
        record = TimesheetRecord(
            source_file=source_file,
            page_number=page_number,
            employee_name=employee_name,
            employee_name_confidence=employee_conf,
            patient_name=patient_name,
            patient_name_confidence=patient_conf,
            rows=rows,
        )

        # 7. Validate
        validate_record(record, self.config)

        return record

    def _extract_row(
        self, image, preprocessed, all_boxes, row_zone, row_idx, layout
    ) -> TimesheetRow | None:
        """Extract data from a single table row using OCR boxes + fallback."""
        columns = self.config.layout.columns
        w = layout.image_width
        h = layout.image_height

        # Map column names to pixel boundaries
        if self.config.layout.transposed:
            field_bounds = {
                "date": (int(h * columns.date[0]), int(h * columns.date[1])),
                "time_in": (int(h * columns.time_in[0]), int(h * columns.time_in[1])),
                "time_out": (int(h * columns.time_out[0]), int(h * columns.time_out[1])),
                "total_hours": (int(h * columns.total_hours[0]), int(h * columns.total_hours[1])),
                "notes": (int(h * columns.notes[0]), int(h * columns.notes[1])),
            }
        else:
            field_bounds = {
                "date": (int(w * columns.date[0]), int(w * columns.date[1])),
                "time_in": (int(w * columns.time_in[0]), int(w * columns.time_in[1])),
                "time_out": (int(w * columns.time_out[0]), int(w * columns.time_out[1])),
                "total_hours": (int(w * columns.total_hours[0]), int(w * columns.total_hours[1])),
                "notes": (int(w * columns.notes[0]), int(w * columns.notes[1])),
            }

        # Find OCR boxes in each column of this row
        cell_data: dict[str, tuple[str, float]] = {}
        cell_confidences: dict[str, float] = {}

        for col_name, (start_pix, end_pix) in field_bounds.items():
            if self.config.layout.transposed:
                col_boxes = boxes_in_zone(
                    all_boxes, row_zone.x_start, start_pix, row_zone.x_end, end_pix
                )
            else:
                col_boxes = boxes_in_zone(
                    all_boxes, start_pix, row_zone.y_start, end_pix, row_zone.y_end
                )

            if col_boxes:
                if self.config.layout.transposed:
                    text = " ".join(b.text for b in sorted(col_boxes, key=lambda b: b.y_center))
                else:
                    text = " ".join(b.text for b in sorted(col_boxes, key=lambda b: b.x_center))
                conf = min(b.confidence for b in col_boxes)
            else:
                text = ""
                conf = 0.0

            # Per-cell re-OCR for time fields: PaddleOCR often merges adjacent
            # columns' handwritten times into one text box (e.g. "200pm5:00om 8:00om").
            # If the cell is empty (box center fell in a neighbor) or suspiciously
            # long (merged), crop just this cell and re-run OCR on it.
            _CELL_OCR_FIELDS = {"time_in", "time_out", "total_hours"}
            if col_name in _CELL_OCR_FIELDS and (text == "" or len(text) > 10):
                # Use the original color image (not preprocessed) — grayscale
                # denoising loses color cues that help OCR read handwriting.
                if self.config.layout.transposed:
                    cell_crop = image[start_pix:end_pix, row_zone.x_start:row_zone.x_end]
                else:
                    cell_crop = image[row_zone.y_start:row_zone.y_end, start_pix:end_pix]

                if cell_crop.size > 0:
                    cell_result = self.ocr_engine.run(cell_crop)
                    if cell_result.boxes:
                        text = cell_result.full_text
                        conf = min(b.confidence for b in cell_result.boxes)
                        logger.debug(f"Row {row_idx} {col_name}: per-cell re-OCR → '{text}' (conf={conf:.3f})")

            cell_data[col_name] = (text, conf)
            cell_confidences[col_name] = conf

        # Check if this row has any meaningful content
        if all(text == "" for text, _ in cell_data.values()):
            return None  # Empty row, skip

        # Route through confidence check
        row_data = dict(cell_data)  # Copy for potential VLM updates
        sources: dict[str, OcrSource] = {k: OcrSource.PPOCR for k in field_bounds}

        # Check if entire row needs VLM fallback
        if should_fallback_entire_row(cell_confidences, self.config):
            logger.info(f"Row {row_idx}: sending entire row to VLM fallback")
            if self.config.layout.transposed:
                row_crop = image[row_zone.y_start:row_zone.y_end, row_zone.x_start:row_zone.x_end]
            else:
                row_crop = image[row_zone.y_start:row_zone.y_end, :]
                
            vlm_result = self.vlm.extract_row(row_crop)
            if vlm_result:
                for field, value in vlm_result.items():
                    if value and field in row_data:
                        row_data[field] = (value, 0.75)
                        sources[field] = OcrSource.VLM
        else:
            # Per-cell fallback for low-confidence cells
            for col_name, conf in cell_confidences.items():
                route = route_by_confidence(conf, self.config)
                if route == Route.FALLBACK:
                    start_pix, end_pix = field_bounds[col_name]
                    if self.config.layout.transposed:
                        cell_crop = image[start_pix:end_pix, row_zone.x_start:row_zone.x_end]
                    else:
                        cell_crop = image[row_zone.y_start:row_zone.y_end, start_pix:end_pix]
                        
                    vlm_value, vlm_conf = self.vlm.extract_cell_value(cell_crop, col_name)
                    if vlm_value:
                        row_data[col_name] = (vlm_value, vlm_conf)
                        sources[col_name] = OcrSource.VLM

        # Build TimesheetRow with parsed values
        date_text, date_conf = row_data["date"]
        time_in_text, time_in_conf = row_data["time_in"]
        time_out_text, time_out_conf = row_data["time_out"]
        hours_text, hours_conf = row_data["total_hours"]
        notes_text, _ = row_data["notes"]

        row = TimesheetRow(
            row_index=row_idx,
            date_text=date_text,
            date_parsed=parse_date(date_text),
            time_in_text=time_in_text,
            time_in_parsed=parse_time(time_in_text),
            time_out_text=time_out_text,
            time_out_parsed=parse_time(time_out_text),
            total_hours_text=hours_text,
            total_hours_parsed=parse_hours(hours_text),
            notes=notes_text.strip(),
            date_confidence=date_conf,
            time_in_confidence=time_in_conf,
            time_out_confidence=time_out_conf,
            hours_confidence=hours_conf,
            date_source=sources["date"],
            time_in_source=sources["time_in"],
            time_out_source=sources["time_out"],
            hours_source=sources["total_hours"],
        )

        return row
