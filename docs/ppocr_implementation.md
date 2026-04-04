# PP-OCRv5 Implementation: Architecture & Pipeline

## Overview

This document describes how PaddleOCR (PP-OCRv5 mobile) is integrated into the timesheet-OCR pipeline. The system extracts structured shift data from scanned weekly timesheet PDFs using a hybrid approach: PaddleOCR detects printed form labels and any handwritten text it can find, while a VLM (Qwen2.5-VL via Ollama) fills in the gaps where OCR fails.

---

## 1. System Architecture

```
PDF Input
    │
    ▼
┌─────────────────────────────────────────┐
│  1. Preprocessing (preprocessing.py)    │
│     • PDF → Images (400 DPI)            │
│     • Grayscale → Denoise → Binarize    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  2. OCR Engine (ocr_engine.py)          │
│     • PP-OCRv5_mobile_det (detection)   │
│     • PP-OCRv5_mobile_rec (recognition) │
│     • Returns list of OcrBox objects    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  3. Layout Detection (layout.py)        │
│     • Header / Table / Footer zones     │
│     • Row/Column boundaries             │
│     • Dynamic field band detection      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  4. Spatial Matching (pipeline.py)      │
│     • Match OCR boxes to cell zones     │
│     • Concatenate text per cell         │
│     • Compute confidence per cell       │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  5. Confidence Routing (confidence.py)  │
│     • Accept (conf ≥ 0.90)              │
│     • Fallback to VLM (conf < 0.75)     │
│     • Accept with flag (0.75–0.90)      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  6. Parser & Validator (parser.py,      │
│     validation.py)                      │
│     • Parse dates, times, hours         │
│     • Validate rules                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  7. Export (exporter.py)                │
│     • Excel (merged_results.xlsx)       │
│     • Benchmark (benchmark_*.xlsx)      │
│     • Review queue / Report (JSON)      │
└─────────────────────────────────────────┘
```

---

## 2. Preprocessing Pipeline

**File:** `src/preprocessing.py`

### 2.1 PDF to Image Conversion

```python
def pdf_to_images(pdf_path, dpi=400) -> list[np.ndarray]
```

- Uses `pdf2image` (requires `poppler`) to convert each PDF page to a PIL Image
- Converts PIL Image → OpenCV BGR numpy array
- At 400 DPI, a standard letter-size page becomes ~3297×4292 pixels

### 2.2 Image Preprocessing

```python
def preprocess_image(img, config) -> np.ndarray
```

| Step | Operation | OpenCV Function | Purpose |
|------|-----------|-----------------|---------|
| 1. Grayscale | BGR → Gray | `cv2.cvtColor(BGR2GRAY)` | Reduce to single channel |
| 2. Deskew | Detect & correct rotation | `cv2.Canny` + `cv2.HoughLinesP` + `cv2.warpAffine` | Straighten scanned pages |
| 3. Denoise | Remove scan artifacts | `cv2.fastNlMeansDenoising(h=10)` | Clean noise from scanner |
| 4. Binarize | Adaptive threshold | `cv2.adaptiveThreshold(GAUSSIAN_C, block=15, C=3)` | Isolate text from background |

**Current config:** `denoise: true`, `deskew: false`, `binarize: false`

The binarization is disabled because the handwritten ink on these timesheets is faint and adaptive thresholding can wash it out. The denoising step helps clean scanner noise while preserving handwritten strokes.

---

## 3. OCR Engine: PP-OCRv5

**File:** `src/ocr_engine.py`

### 3.1 Model Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Detection model | `PP-OCRv5_mobile_det` | Lightweight text detection |
| Recognition model | `PP-OCRv5_mobile_rec` | Lightweight text recognition |
| Text orientation | `false` | No textline orientation correction |
| Device | `cpu` | CPU-only inference (no Apple Silicon GPU support) |
| Detection threshold | `0.45` | Lower threshold to catch faint handwriting |
| Recognition batch size | `4` | Batch size for recognition |

### 3.2 Why Mobile Models?

The server models (`PP-OCRv5_server_det/rec`) are 3-5x slower on CPU with marginal accuracy gains. Mobile models are optimized for speed while maintaining sufficient accuracy for printed form labels. Handwritten text is primarily handled by the VLM fallback layer.

### 3.3 Data Structures

```python
@dataclass
class OcrBox:
    text: str                        # Recognized text
    confidence: float                # Recognition confidence (0.0–1.0)
    bbox: list[list[float]]          # 4 corners: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

    # Computed properties
    x_center, y_center               # Center of bounding box
    x_min, y_min, x_max, y_max       # Axis-aligned bounding box bounds
```

```python
@dataclass
class OcrResult:
    boxes: list[OcrBox]              # All detected text boxes
    raw_output: Any                  # Raw PaddleOCR output (for debugging)

    @property
    def full_text: str               # Concatenated text, sorted top-to-bottom, left-to-right
```

### 3.4 OCR Execution Flow

```python
def run(image: np.ndarray) -> OcrResult:
    1. Ensure model is initialized (lazy loading)
    2. Convert grayscale → BGR if needed (PaddleOCR v3.4.0 requirement)
    3. Run self._ocr.ocr(image)
    4. Parse results:
       a. Extract rec_texts, rec_scores, rec_polys from each page
       b. Check if PaddleOCR internally resized the image
       c. If resized, compute scale factors (sx, sy) from output_img shape
       d. Scale all bbox coordinates back to original image space
       e. Create OcrBox for each (text, score, scaled_poly) tuple
    5. Return OcrResult(boxes, raw_output)
```

### 3.5 Coordinate Scaling

PaddleOCR may internally resize large images to fit its model's max side limit (4000px). The pipeline detects this by checking `page["doc_preprocessor_res"]["output_img"]` and computes scale factors:

```python
sx = orig_w / out_w   # horizontal scale
sy = orig_h / out_h   # vertical scale
bbox_point = [[float(p[0]) * sx, float(p[1]) * sy] for p in poly]
```

### 3.6 Cropped Region OCR

```python
def run_on_crop(image, x1, y1, x2, y2) -> OcrResult:
    1. Crop: image[y1:y2, x1:x2]
    2. Run OCR on crop
    3. Adjust all bbox coordinates back to original image space:
       point[0] += x1
       point[1] += y1
    4. Return adjusted OcrResult
```

This is used by `extract_text_from_zone()` to run OCR on specific regions (e.g., header zone for employee name).

---

## 4. Layout Detection

**File:** `src/layout.py`

### 4.1 Zone System

The layout is divided into three zones defined as fractional coordinates in `config.yaml`:

| Zone | Fractional Coords | Pixel Range (3297×4292) | Purpose |
|------|-------------------|------------------------|---------|
| Header | `[0.0, 0.0, 1.0, 0.16]` | y: 0–686px | Employee name, patient info |
| Table | `[0.24, 0.16, 1.0, 0.98]` | x: 791–3297, y: 686–4207 | Main timesheet grid |
| Footer | `[0.0, 0.98, 1.0, 1.0]` | y: 4207–4292 | Signatures, dates |

### 4.2 Row/Column Detection

For **transposed layouts** (columns = days of the week):

```python
def _detect_col_boundaries(table_image) -> list[tuple[int, int]]:
    1. Binarize with Otsu's threshold (inverted)
    2. Morphological open with vertical kernel (1 × h/3)
    3. Sum pixel values along Y-axis → col_sums
    4. Find x-positions where col_sums > threshold (h × 128)
    5. Build columns from consecutive line positions
    6. Fallback: 7 evenly-spaced columns (one per day)
```

For **standard layouts** (rows = days of the week):

```python
def _detect_row_boundaries(table_image) -> list[tuple[int, int]]:
    1. Binarize with Otsu's threshold (inverted)
    2. Morphological open with horizontal kernel (w/3 × 1)
    3. Sum pixel values along X-axis → row_sums
    4. Find y-positions where row_sums > threshold (w × 128)
    5. Build rows from consecutive line positions
    6. Fallback: 14 evenly-spaced rows
```

### 4.3 Dynamic Field Band Detection

The system dynamically discovers where each field (date, time_in, time_out, total_hours) is located by searching for printed labels in the OCR output:

```python
FIELD_LABELS = {
    "date":        ["date", "day", "month"],
    "time_in":     ["time in", "time-in", "clock in", "start"],
    "time_out":    ["time out", "time-out", "clock out", "end"],
    "total_hours": ["hours", "total hours", "total", "number of hours"],
}

def _detect_field_bands(boxes, image_height) -> dict[str, tuple[int, int]]:
    1. For each OCR box, check if its text matches any known label keyword
    2. Record the y_center position of each matched label
    3. If ≥2 labels found, build bands around each label position (±100px margin)
    4. Fill missing fields with full-range bands (0, image_height)
    5. If <2 labels found, fallback to evenly-spaced bands
```

**Why ±100px margin?** Handwritten values are typically written within 100 pixels of their printed labels on these timesheet forms.

### 4.4 LayoutResult Output

```python
@dataclass
class LayoutResult:
    header_zone: Zone                    # Header region
    table_zone: Zone                     # Table region
    footer_zone: Zone                    # Footer region
    row_zones: list[Zone]                # 7 column zones (transposed) or row zones
    grid_cells: list[GridCell]           # row × column cell matrix
    image_height: int
    image_width: int
    field_bands: dict[str, tuple[int, int]]  # e.g., {"date": (488, 688), "time_in": (3806, 4006), ...}
```

---

## 5. Pipeline: Spatial Matching & Extraction

**File:** `src/pipeline.py`

### 5.1 Signature Page Detection

Before attempting row extraction, the pipeline checks if a page is a signature page:

```python
is_sig_page = len(ocr_result.boxes) < signature_ocr_threshold  # default: 100
```

**Rationale:** Grid pages have 122–201 OCR boxes (form labels + handwritten data + printed text). Signature pages have only 34–59 boxes (signatures, footer text, dates). Signature pages skip row extraction entirely — only the employee name is extracted from the header zone.

### 5.2 Transposed Row Extraction

For transposed layouts, each "row" is actually a column representing one day of the week:

```python
def _extract_transposed_row(...) -> tuple[TimesheetRow | None, int, list[VlmFallbackCell]]:
    For each column (day):
        1. DATE: Use generated_dates[row_idx] from filename (source=GENERATED, conf=1.0)
        2. TIME IN / TIME OUT / HOURS:
           a. Find OCR boxes whose centers fall within column × field_band zone
           b. If boxes found → concatenate text, use min confidence (source=PPOCR)
           c. If no boxes → VLM fallback on cropped cell (source=VLM, conf=0.9)
        3. Skip if all time fields are empty
        4. Return (row, vlm_count, vlm_cells)
```

### 5.3 VLM Fallback Tracking

Every time a cell falls back to VLM, a `VlmFallbackCell` is recorded:

```python
@dataclass
class VlmFallbackCell:
    row_idx: int
    field_name: str          # "time_in", "time_out", "total_hours"
    x_start, y_start, x_end, y_end: int  # Cell boundaries in original image
    vlm_text: str            # What VLM extracted
    vlm_conf: float          # VLM confidence (0.9 for success, 0.5 for empty)
```

These are used for:
- Benchmark metrics (count of VLM fallbacks per page)
- Debug visualization (yellow highlights on fallback cells)

---

## 6. Confidence Routing

**File:** `src/confidence.py`

After spatial matching, each cell's confidence determines its fate:

| Confidence | Route | Action |
|------------|-------|--------|
| ≥ 0.90 | ACCEPT | Use PPOCR value directly |
| 0.75–0.89 | FLAG | Use PPOCR value, flag for review |
| < 0.75 | FALLBACK | Call VLM on cropped cell |

**Configurable thresholds:**
```yaml
confidence:
  accept_threshold: 0.90
  fallback_threshold: 0.75
```

---

## 7. Parser & Validation

**File:** `src/parser.py`, `src/validation.py`

### 7.1 Parsing

| Field | Parser | Handles |
|-------|--------|---------|
| Date | `parse_date(text, expected_year)` | MM/DD/YY, MM/DD/YYYY, MMDDYY, day names (with generated dates) |
| Time | `parse_time(text)` | 8:00 AM, 0800, 8:00, 8am, 8:00pm, 17:00 |
| Hours | `parse_hours(text)` | 8.0, 8, 8.5, 7.25 |

### 7.2 Validation Rules

| Rule | Condition | Status |
|------|-----------|--------|
| `missing_time_in` | time_in_parsed is None | FLAGGED |
| `missing_time_out` | time_out_parsed is None | FLAGGED |
| `time_in_not_parseable` | time_in_text exists but parse failed | FLAGGED |
| `time_out_not_parseable` | time_out_text exists but parse failed | FLAGGED |
| `hours_mismatch` | |calculated - written| > 0.25 | FLAGGED |
| `shift_too_long` | calculated > max_shift_hours (16) | FLAGGED |
| `future_date` | date > today + allow_future_dates | FLAGGED |
| `stale_date` | date < today - max_days_in_past (365) | FLAGGED |

---

## 8. Benchmark System

**File:** `src/benchmark.py`

### 8.1 Metrics Collected

**Per-Run (Run Summary):**
- Total time, avg page time, OCR init/inference/layout/extraction/validation times
- Rows extracted, accepted/flagged/failed counts, acceptance rate
- Mean/min/max confidence across all fields
- Total OCR boxes, VLM fallbacks, empty rows skipped
- Hours mismatch rate, field missing rate, mean CER

**Per-Page (Page Details):**
- Image dimensions, per-phase timing
- Boxes detected, rows extracted, empty rows skipped, VLM fallbacks

**Per-Row (Row-Level):**
- Raw OCR text vs parsed value for each field
- Confidence scores and source (PPOCR/VLM/GENERATED)
- Calculated hours vs written hours
- Validation status, errors, corrections applied

**Corrections Log:**
- Every parser correction: field name, raw OCR value, corrected value, correction type

### 8.2 Combined Benchmark

When processing multiple files, `benchmark_combined.xlsx` is generated with:
- **Per-File Results** — paper-ready table: rows=metrics, columns=files + Combined
- **Page Details** — all pages from all runs
- **Row-Level** — all rows from all runs
- **Corrections** — all corrections from all runs

---

## 9. PHI/PII Protection

**File:** `src/phi.py`

### 9.1 Anonymization Strategy

Deterministic mapping based on sorted filenames:

| Original | Anonymized |
|----------|------------|
| C.Ferguson | Patient_A |
| J.Flemming | Patient_B |
| K.Drewry | Patient_C |
| DAY: (employee) | Employee_A |
| Skilled Record... (employee) | Employee_B |
| C.Ferguson Timesheets - 010726-011326.pdf | patient_a_week1.pdf |

### 9.2 Applied To

| Output | Anonymization |
|--------|---------------|
| merged_results.xlsx | Patient/employee names, source filenames |
| Benchmark sheets | Source filenames in all sheets |
| Review/Report JSON | Source filenames |
| Debug PNGs | Header zone blacked out (solid black rectangle) |

### 9.3 Config

```yaml
debug:
  visualize_ocr: true
  anonymize_phi: true
  signature_ocr_threshold: 100    # Pages with <100 boxes = signature pages
```

---

## 10. Debug Visualization

**File:** `src/debug_viz.py`

### 10.1 Color Scheme

| Element | Color | Style |
|---------|-------|-------|
| OCR box (conf > 0.7) | Green (#00FF00) | Rectangle border + text label |
| OCR box (conf ≤ 0.7) | Red (#FF0000) | Rectangle border + text label |
| Header/Table/Footer zones | Blue (#0000FF) | Dashed rectangle |
| Column zones | Orange (#FFA500) | Dashed rectangle |
| Field bands | Purple (#800080) | Dashed horizontal line + label |
| VLM fallback cell | Yellow (#FFFF00) | Semi-transparent fill + white text |
| Header redaction | Black (#000000) | Solid rectangle (PHI protection) |

### 10.2 Rendering Order

1. Base image (original page)
2. Layout zones (blue dashed)
3. Column zones (orange dashed)
4. Field bands (purple dashed)
5. **Header redaction** (black solid — PHI protection)
6. OCR boxes (green/red rectangles + text)
7. VLM fallback cells (yellow fill + white text)
8. Legend (top-right corner)

### 10.3 Output

```
output/debug/{anonymized_filename}_page{N}.png
```

Only grid pages are visualized. Signature pages are skipped entirely.

---

## 11. Performance Characteristics

### 11.1 Timing Breakdown (3 files, 8 pages total)

| Phase | Total Time | % of Total |
|-------|-----------|------------|
| OCR Inference | 223s | 42% |
| VLM Fallbacks | ~283s | 53% |
| Layout Detection | 3.6s | 1% |
| Validation | 0.05s | <1% |
| Export | ~2s | <1% |

### 11.2 OCR Box Counts

| File | Page | Boxes | Type |
|------|------|-------|------|
| patient_a_week1 | 1 | 260 | Grid |
| patient_a_week1 | 2 | 59 | Signature (skipped) |
| patient_b_week2 | 1 | 177 | Grid |
| patient_b_week2 | 2 | 55 | Signature (skipped) |
| patient_c_week3 | 1 | 147 | Grid |
| patient_c_week3 | 2 | 34 | Signature (skipped) |
| patient_c_week3 | 3 | 157 | Grid |
| patient_c_week3 | 4 | 36 | Signature (skipped) |

### 11.3 VLM Fallback Rate

| File | Grid Pages | VLM Fallbacks | Rows Extracted | Fallbacks/Row |
|------|-----------|---------------|----------------|---------------|
| patient_a_week1 | 1 | 6 | 6 | 1.0 |
| patient_b_week2 | 1 | 17 | 7 | 2.4 |
| patient_c_week3 | 2 | 26 | 11 | 2.4 |

**Total:** 49 VLM fallbacks across 24 rows (2.0 per row on average).

---

## 12. Configuration Reference

```yaml
extraction_mode: "ppocr_grid"     # 'ppocr_grid' or 'vlm_full_page'

paths:
  input_dir: "input"
  output_dir: "output"

confidence:
  accept_threshold: 0.90          # Above this: accept PPOCR value
  fallback_threshold: 0.75        # Below this: VLM fallback

ppocr:
  lang: "en"
  use_textline_orientation: false
  device: "cpu"
  text_det_thresh: 0.45           # Lower = more detections (catches faint handwriting)
  text_rec_batch_size: 4

ollama:
  host: "http://localhost:11434"
  model: "qwen2.5vl:7b"
  timeout_seconds: 60
  max_retries: 2

preprocessing:
  target_dpi: 400
  denoise: true
  deskew: false
  binarize: false
  adaptive_block_size: 15
  adaptive_c: 3

layout:
  transposed: true                # Columns = days of week
  header_zone: [0.0, 0.0, 1.0, 0.16]
  table_zone: [0.24, 0.16, 1.0, 0.98]
  footer_zone: [0.0, 0.98, 1.0, 1.0]

validation:
  max_shift_hours: 16
  hours_mismatch_tolerance: 0.25
  allow_future_dates: false
  max_days_in_past: 365
  week_start_day: 2               # 0=Mon, 1=Tue, 2=Wed, ..., 6=Sun
  week_length: 7

export:
  formats: ["xlsx"]
  excel_sheet_name: "Timesheet Data"
  include_review_json: true
  include_report_json: true

debug:
  visualize_ocr: true
  output_dir: "output/debug"
  anonymize_phi: true
  signature_zone_fraction: 0.30
  signature_ocr_threshold: 100    # Pages with <100 boxes = signature pages
```

---

## 13. Key Design Decisions

### 13.1 Why Not Use PPOCR Alone?

PaddleOCR mobile detects printed form labels reliably but misses most handwritten text. On these timesheets:
- **Printed labels** (Date, Time In, Time Out, Hours): ~80-90% detection rate
- **Handwritten values** (8:00 AM, 5:00 PM, 8.0): ~10-30% detection rate

The VLM fallback layer compensates for this by reading cropped cells that PPOCR couldn't detect text in.

### 13.2 Why Transposed Layout?

The timesheet forms have days of the week as columns (left to right) and fields as rows (top to bottom). This is the opposite of a standard table where rows = records and columns = fields. The `transposed: true` config tells the layout detector to search for vertical column boundaries instead of horizontal row boundaries.

### 13.3 Why Generated Dates?

The filename contains the week's date range (e.g., `012826-020326` = Jan 28 – Feb 3, 2026). The pipeline generates the 7 dates for that week and assigns them to columns. This is more reliable than trying to OCR handwritten dates, which are often illegible.

### 13.4 Why Mobile Models?

Server models are 3-5x slower on CPU with marginal accuracy improvements. Since handwritten text is handled by VLM anyway, the mobile models are sufficient for detecting printed labels and any clear handwritten text.

---

## 14. File Structure

```
src/
├── ocr_engine.py        # PP-OCRv5 wrapper, OcrBox, OcrResult
├── preprocessing.py     # PDF→images, grayscale, denoise, binarize, deskew
├── layout.py            # Zone detection, field band detection, row/column boundaries
├── pipeline.py          # Main orchestrator: load → OCR → layout → extract → validate → export
├── parser.py            # Date/time/hours parsing
├── validation.py        # Validation rules (missing fields, hours mismatch, etc.)
├── confidence.py        # Confidence routing (accept/flag/fallback)
├── vlm_fallback.py      # Ollama/Qwen2.5-VL cell extraction
├── benchmark.py         # Metrics collection and export
├── phi.py               # PHI/PII anonymization
├── debug_viz.py         # Debug visualization with bounding boxes
├── exporter.py          # Excel, CSV, JSON export
├── review_queue.py      # Build review queue for flagged rows
├── models.py            # Pydantic data models (TimesheetRow, TimesheetRecord, etc.)
├── config.py            # Configuration loading from config.yaml
└── main.py              # CLI entry point
```
