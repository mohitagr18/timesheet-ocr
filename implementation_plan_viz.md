# Implementation Plan: Debug Visualization + Benchmark Bug Fix

## 1. Bug Fix: Benchmark Accumulation

**Problem:** `start_run()` in `benchmark.py` creates a new `RunMetrics` but doesn't reset `self.pages` and `self.rows`. Each file's data accumulates on top of previous files:
- Ferguson: 2 pages → snapshot captures 2 ✅
- Fleming: adds 2 more → `self.pages` has 4 → snapshot captures 4 ❌
- Drewry: adds 4 more → `self.pages` has 8 → snapshot captures 8 ❌

**Fix:** In `benchmark.py:122-136`, add resets:
```python
def start_run(self, ...) -> None:
    self.run = RunMetrics(...)
    self.pages = []   # ← add
    self.rows = []    # ← add
```

---

## 2. Config: Debug Visualization Toggle

**File:** `config.yaml` — add new `debug` section:

```yaml
# --- Debug / Visualization ---
debug:
  visualize_ocr: false       # Set to true to generate annotated page images
  output_dir: "output/debug" # Where to save debug images
```

**Behavior:**
- `visualize_ocr: false` (default) — no overhead, no images generated
- `visualize_ocr: true` — generates one annotated PNG per page during pipeline execution
- User toggles manually in `config.yaml` before running

**File:** `src/config.py` — add `DebugConfig` dataclass:
```python
@dataclass
class DebugConfig:
    visualize_ocr: bool = False
    output_dir: str = "output/debug"
```

---

## 3. New Module: `src/debug_viz.py`

**Purpose:** Generate publication-quality images with overlaid bounding boxes, layout zones, and VLM fallback annotations.

### Data Structures

```python
@dataclass
class VlmFallbackCell:
    """Tracks a single cell that fell back to VLM."""
    row_idx: int
    field_name: str          # "time_in", "time_out", "total_hours"
    x_start: int
    y_start: int
    x_end: int
    y_end: int
    vlm_text: str            # What VLM extracted
    vlm_conf: float
```

### Color Scheme (all on same image, differentiated by color)

| Element | Color | Style |
|---|---|---|
| OCR text box (conf > 0.7) | **Green** `#00FF00` | Rectangle border + text label |
| OCR text box (conf ≤ 0.7) | **Red** `#FF0000` | Rectangle border + text label |
| Header zone | **Blue** `#0000FF` | Dashed rectangle |
| Table zone | **Blue** `#0000FF` | Dashed rectangle |
| Column zones (transposed) | **Orange** `#FFA500` | Dashed rectangle |
| Field bands | **Purple** `#800080` | Dashed horizontal lines |
| VLM fallback cell | **Yellow** `#FFFF00` | Semi-transparent fill (alpha=0.3) |
| VLM extracted text | **White** on yellow | Text overlay inside fallback cell |

### Function Signature

```python
def render_page(
    image: np.ndarray,
    ocr_boxes: list[TextBox],
    layout: LayoutResult,
    field_bands: dict[str, tuple[int, int]],
    vlm_fallbacks: list[VlmFallbackCell],
    page_number: int,
    source_file: str,
    output_dir: Path,
) -> Path:
    """Render annotated page image with all visualization layers.
    
    Returns path to saved PNG.
    """
```

### Rendering Order (bottom to top)

1. **Base image** — original page (full resolution)
2. **Layout zones** — header, table, footer (blue dashed rectangles)
3. **Column zones** — transposed column boundaries (orange dashed)
4. **Field bands** — date/time_in/time_out/hours horizontal bands (purple dashed)
5. **OCR boxes** — green/red rectangles with text labels
6. **VLM fallback cells** — yellow semi-transparent fills with VLM text overlay

### Output Naming

```
output/debug/{filename}_page{N}.png
```

Example: `output/debug/J.Flemming Timesheets - 012826-020326_page1.png`

---

## 4. Pipeline Integration

### 4a. Track VLM Fallback Cell Details

**Current:** `_extract_transposed_row` returns `tuple[TimesheetRow | None, int]` (row + count)

**New:** Returns `tuple[TimesheetRow | None, int, list[VlmFallbackCell]]`

When a VLM fallback occurs, record:
```python
vlm_fallbacks.append(VlmFallbackCell(
    row_idx=row_idx,
    field_name=field_name,
    x_start=x_left,
    y_start=y_start,
    x_end=x_right,
    y_end=y_end,
    vlm_text=vlm_value,
    vlm_conf=vlm_conf,
))
```

Same pattern for `_extract_row` (non-transposed mode).

### 4b. Page Loop Aggregation

```python
row, vlm_count, vlm_cells = self._extract_row(...)
total_vlm_fallbacks += vlm_count
all_vlm_fallbacks.extend(vlm_cells)
```

### 4c. Visualization Hook

After extraction loop in `_process_page`, before returning:

```python
# 8. Debug visualization (if enabled)
if getattr(self.config, "debug", None) and self.config.debug.visualize_ocr:
    debug_viz.render_page(
        image=image,
        ocr_boxes=ocr_result.boxes,
        layout=layout,
        field_bands=layout.field_bands,
        vlm_fallbacks=all_vlm_fallbacks,
        page_number=page_number,
        source_file=source_file,
        output_dir=self.config.debug.output_dir,
    )
```

---

## 5. Files Modified

| File | Change |
|---|---|
| `config.yaml` | Add `debug:` section with `visualize_ocr: false` |
| `src/config.py` | Add `DebugConfig` dataclass, add to `AppConfig` |
| `src/benchmark.py` | Fix `start_run()` to reset `self.pages` and `self.rows` |
| `src/pipeline.py` | Update `_extract_row` and `_extract_transposed_row` to return VLM fallback details; add visualization hook after extraction |
| `src/debug_viz.py` | **NEW** — visualization module with `render_page()` function |

---

## 6. Execution Order

1. Fix `benchmark.py` accumulation bug
2. Add `debug` config section to `config.yaml` and `src/config.py`
3. Create `src/debug_viz.py` with full visualization logic
4. Update `src/pipeline.py` to track VLM fallback details and hook visualization
5. Clean `output/` directory
6. Run pipeline on all 3 files with `visualize_ocr: true`
7. Verify:
   - `output/benchmark_combined.xlsx` has correct per-file counts (Ferguson=2 pages, Fleming=2 pages, Drewry=4 pages)
   - `output/debug/` contains annotated PNGs for all 8 pages
   - Each PNG shows OCR boxes (green/red), layout zones (blue), columns (orange), field bands (purple), VLM fallbacks (yellow with text)

---

## 7. Notes on VLM-Only Mode Visualization

The visualization described above is for **`ppocr_grid` mode only** — it shows PaddleOCR boxes + VLM fallback cells. 

For **`vlm_full_page` mode** (future), there are no PaddleOCR boxes. A separate visualization path would show:
- Full page image with extracted text annotations at approximate positions
- No bounding boxes (VLM doesn't produce them)
- Different output naming: `debug_{filename}_page{N}_vlm.png`

This separation allows direct side-by-side comparison in the IEEE paper:
- **Image A**: `ppocr_grid` mode → shows OCR failures (red boxes) and VLM fallbacks (yellow)
- **Image B**: `vlm_full_page` mode → shows what VLM extracted directly from full page

---

## 8. PHI/PII Protection + Signature Page Handling + Notes Removal

### 8a. Config Additions

**`config.yaml`** — new sections:
```yaml
export:
  formats: ["xlsx"]
  excel_sheet_name: "Timesheet Data"
  include_review_json: true    # Toggle *_review.json output
  include_report_json: true    # Toggle *_report.json output

debug:
  visualize_ocr: true
  output_dir: "output/debug"
  anonymize_phi: true
  signature_zone_fraction: 0.30   # Bottom 30% of signature pages for redaction
  signature_ocr_threshold: 100    # Pages with <100 OCR boxes = signature pages
```

**`src/config.py`** — add to `ExportConfig`:
```python
include_review_json: bool = True
include_report_json: bool = True
```

Add to `DebugConfig`:
```python
anonymize_phi: bool = True
signature_zone_fraction: float = 0.30
signature_ocr_threshold: int = 100
```

### 8b. New Module: `src/phi.py`

```python
class PhiAnonymizer:
    """Consistent PHI/PII anonymization across all outputs."""
    
    def __init__(self, filenames: list[str]) -> None:
        # Deterministic: sort filenames alphabetically, map to Patient_A, Patient_B...
        # Extract patient name from filename (before "Timesheet")
        # Extract employee name from first occurrence in extraction results
        self._patient_map: dict[str, str]   # "C.Ferguson" → "Patient_A"
        self._employee_map: dict[str, str]  # "DAY:" → "Employee_A"
        self._filename_map: dict[str, str]  # "C.Ferguson Timesheets..." → "patient_a_week1.pdf"
    
    def anonymize_patient(self, name: str) -> str
    def anonymize_employee(self, name: str) -> str
    def anonymize_filename(self, filename: str) -> str
    def is_signature_page(self, ocr_box_count: int, threshold: int) -> bool
```

**Anonymization mapping (deterministic, sorted by filename):**

| Original Patient | Anonymized | Original Filename → |
|---|---|---|
| C.Ferguson | Patient_A | `patient_a_week1.pdf` |
| J.Flemming | Patient_B | `patient_b_week2.pdf` |
| K.Drewry | Patient_C | `patient_c_week3.pdf` |

Employee names mapped similarly: first unique employee → Employee_A, second → Employee_B, etc.

### 8c. Signature Page Skip Logic

**In `pipeline.py` `_process_page`:**

After OCR runs, check box count:
```python
is_sig_page = len(ocr_result.boxes) < self.config.debug.signature_ocr_threshold
if is_sig_page:
    # Only extract employee name from header zone
    # Skip all row extraction, VLM fallbacks, visualization
    employee_name = _extract_employee_from_header(ocr_result.boxes, layout)
    record = TimesheetRecord(
        source_file=anonymized_filename,
        page_number=page_number,
        employee_name=anonymizer.anonymize_employee(employee_name),
        rows=[],  # No rows from signature pages
    )
    return record, page_metrics
```

**Expected impact:**
- Eliminates ~21 garbage rows (Ferguson pg2: 7, Fleming pg2: 7, Drewry pg2+pg4: 7)
- Eliminates ~40 VLM fallbacks on blank signature pages
- Dramatically improves acceptance rate and field missing rate

### 8d. Notes Column Removal

**Remove from:**
- `pipeline.py` — stop extracting notes field in `_extract_transposed_row` and `_extract_row`
- `src/models.py` — remove `notes` field from `TimesheetRow` (or keep but don't populate)
- `src/exporter.py` — remove Notes column from Excel output
- `src/benchmark.py` — remove any notes-related columns from Row-Level sheet

### 8e. Debug Visualization Updates

**Changes to `src/debug_viz.py`:**

1. **Header zone redaction** — Draw solid black rectangle over `layout.header_zone` on all grid pages
2. **Skip signature pages entirely** — If `is_signature_page` is true, don't call `render_page()` at all
3. **No visualization for signature pages** — No output files generated

### 8f. Exporter Updates

**Changes to `src/exporter.py`:**

1. **Respect JSON toggles:**
   ```python
   if self.config.export.include_review_json:
       export_review_json(...)
   if self.config.export.include_report_json:
       export_report_json(...)
   ```

2. **Anonymize names and filenames:**
   - `Patient Name` column → anonymized via `PhiAnonymizer`
   - `Employee Name` column → anonymized via `PhiAnonymizer`
   - `Source File` column → anonymized filename
   - Review JSON → anonymize all patient/employee names and source filenames
   - Report JSON → anonymize source filename

### 8g. Benchmark Updates

**Changes to `src/benchmark.py`:**

1. **Remove notes-related columns** from Row-Level sheet:
   - Remove `raw_ocr_notes`, `parsed_notes`, `notes_confidence`, `notes_source`

2. **Anonymize filenames** in all sheets:
   - `Source File` column → use `PhiAnonymizer.anonymize_filename()`
   - Combined sheet column headers → anonymized filenames

### 8h. Files Modified (Updated)

| File | Change |
|---|---|
| `config.yaml` | Add export toggles + debug PHI settings |
| `src/config.py` | Add new fields to `ExportConfig` and `DebugConfig` |
| `src/phi.py` | **NEW** — `PhiAnonymizer` class |
| `src/pipeline.py` | Skip signature page row extraction; remove notes extraction; integrate `PhiAnonymizer`; skip debug viz for signature pages |
| `src/exporter.py` | Respect JSON toggles; remove Notes column; anonymize names/filenames |
| `src/benchmark.py` | Remove notes columns; anonymize filenames |
| `src/debug_viz.py` | Add header redaction; skip signature pages |

### 8i. Execution Order

1. Create `src/phi.py`
2. Update `config.yaml` and `src/config.py`
3. Update `src/pipeline.py` — signature page skip + notes removal + PHI integration
4. Update `src/exporter.py` — JSON toggles + Notes removal + anonymization
5. Update `src/benchmark.py` — Notes removal + filename anonymization
6. Update `src/debug_viz.py` — header redaction + signature page skip
7. **TEST ON ONE FILE FIRST:** Run on Ferguson only, verify:
   - No PHI/PII in any output (patient names, employee names, filenames)
   - Signature page (page 2) skipped — no rows extracted, no visualization
   - Only grid page (page 1) visualized with header blacked out
   - Notes column absent from Excel
   - JSON files generated (if toggles on) and anonymized
   - Benchmark has correct counts (2 pages, but only 7 rows from page 1)
8. **CLEAN AND FULL RUN:** Clean `output/`, run on all 3 files
9. Verify combined benchmark has correct per-file counts and no PHI

### 8j. Expected Output After Changes

**merged_results.xlsx:**
- No Notes column
- Patient names → Patient_A, Patient_B, Patient_C
- Employee names → Employee_A, Employee_B
- Source filenames → patient_a_week1.pdf, etc.
- Only grid page rows (~20 rows vs 41 before)

**Debug PNGs (4 files):**
- `patient_a_week1_page1.png` — grid page with header blacked out
- `patient_b_week2_page1.png` — grid page with header blacked out
- `patient_c_week3_page1.png` — grid page with header blacked out
- `patient_c_week3_page3.png` — grid page with header blacked out
- NO signature page images

**Benchmark sheets:**
- Anonymized filenames throughout
- No notes-related metrics
- Correct per-file counts (no accumulation, no signature page garbage)

**JSON files (if enabled):**
- Anonymized patient/employee names
- Anonymized source filenames

---

## 9. VLM Mode Enhancement: Model Upgrade + Visualization + Combined Benchmark

### 9a. Model Upgrade

**Current:** `qwen2.5vl:7b`
**New:** `qwen3-vl:8b`

| Aspect | qwen2.5vl:7b | qwen3-vl:8b |
|--------|-------------|-------------|
| Size | ~5GB | 6.1GB |
| OCR quality | Good | Significantly better (32 languages, complex scenes) |
| Handwriting recognition | Struggles with cursive/faint | Much improved |
| Structured JSON output | Basic compliance | Better JSON compliance |
| Context window | Standard | 256K tokens |

**Config change:** `config.yaml` → `model: "qwen3-vl:8b"`
**Pull command:** `ollama pull qwen3-vl:8b`

### 9b. VLM Debug Visualization

**New file:** `src/vlm_debug_viz.py`

Since VLM mode doesn't produce bounding boxes, the visualization overlays extracted text annotations directly on the full page image:

| Element | Color | Style |
|---------|-------|-------|
| Header zone (PHI redaction) | Black | Solid rectangle |
| Extracted shift rows | Green border | Numbered rectangles around each row area (evenly spaced across table zone) |
| Extracted values | White text on colored bg | Text overlay at approximate position |
| Accepted rows | Green tint | Semi-transparent fill (alpha=0.15) |
| Flagged rows | Yellow tint | Semi-transparent fill (alpha=0.15) |
| Failed rows | Red tint | Semi-transparent fill (alpha=0.15) |
| Legend | White on black | Top-right corner |

**Output naming:** `output/debug/vlm_{anonymized_filename}_page{N}.png`

**Integration:** Hook into `_process_page` when `extraction_mode == "vlm_full_page"` and `debug.visualize_ocr == true`.

### 9c. VLM Metrics Tracking

**New fields in `vlm_fallback.py`:**
- Track total VLM calls, empty responses, JSON parse failures
- Expose counters via `get_stats()` method

**New fields in `benchmark.py` RunMetrics:**
```python
vlm_inference_time_s: float = 0.0        # Total VLM inference time
hallucination_rate: float = 0.0          # % rows discarded (>7 rows)
empty_response_rate: float = 0.0         # % fields where VLM returned ""
json_parse_success_rate: float = 0.0     # % VLM responses parsed as valid JSON
```

**New fields in `benchmark.py` PageMetrics:**
```python
vlm_inference_time_s: float = 0.0        # VLM time for this page
hallucinated_rows: int = 0               # Rows discarded by anti-hallucination
empty_fields: int = 0                    # Fields where VLM returned empty
total_vlm_fields: int = 0                # Total VLM fields queried
json_parse_failures: int = 0             # VLM responses that failed JSON parsing
```

### 9d. Combined Benchmark: Side-by-Side PP-OCR vs VLM Columns

The `benchmark_combined.xlsx` Per-File Results sheet will have columns like:

| Metric | patient_a_ppocr | patient_a_vlm | patient_b_ppocr | patient_b_vlm | Combined PP-OCR | Combined VLM |
|--------|----------------|---------------|-----------------|---------------|-----------------|-------------|
| Model | PP-OCRv5_mobile | qwen3-vl:8b | — | — | — | — |
| Pages | 2 | 2 | 8 | 8 | 8 | 8 |
| Total Time (s) | 100.4 | ... | 461.7 | ... | 461.7 | ... |
| OCR Inference (s) | 65.5 | NA | 223.4 | NA | 223.4 | NA |
| VLM Inference (s) | 32.8 | 95.0 | 282.9 | ... | 282.9 | ... |
| VLM Fallbacks | 6 | NA | 49 | NA | 49 | NA |
| Hallucinated Rows | NA | ... | NA | ... | NA | ... |
| Empty Response Rate | NA | ... | NA | ... | NA | ... |
| JSON Parse Success | NA | ... | NA | ... | NA | ... |
| Rows Extracted | 6 | ... | 23 | ... | 23 | ... |
| Acceptance Rate | 16.7% | ... | 8.7% | ... | 8.7% | ... |
| Mean Confidence | 0.857 | ... | 0.753 | ... | 0.753 | ... |
| Mean CER | 0.369 | NA | 0.603 | NA | 0.603 | NA |

**PP-OCR-only metrics** (marked NA for VLM):
- OCR Inference Time
- VLM Fallbacks
- Mean CER (compares raw OCR text vs parsed value — no raw OCR in VLM mode)

**VLM-only metrics** (marked NA for PP-OCR):
- VLM Inference Time (in VLM mode, this IS the extraction time)
- Hallucination Rate (% rows discarded by anti-hallucination hammer)
- Empty Response Rate (% fields where VLM returned empty string)
- JSON Parse Success Rate (% of VLM responses that parsed as valid JSON)

### 9e. BenchmarkCollector Enhancement

The `BenchmarkCollector` needs to track which mode each run was in. Add:
```python
def snapshot_run(self, mode: str = "ppocr") -> None:
    """Snapshot with mode tag."""
    self.run.extraction_mode = mode
    # ... existing snapshot logic
```

The `export_combined` method will group runs by mode and create side-by-side columns:
```python
# Column order: patient_a_ppocr, patient_a_vlm, patient_b_ppocr, patient_b_vlm, ...
```

### 9f. Files Modified

| File | Change |
|---|---|
| `config.yaml` | `model: "qwen3-vl:8b"` |
| `src/vlm_fallback.py` | Add metrics tracking (counters for empty responses, JSON failures, total calls); expose `get_stats()` method |
| `src/pipeline.py` | Add VLM debug visualization hook for `vlm_full_page` mode; pass extraction mode to benchmark snapshot; collect VLM metrics from `vlm.get_stats()` |
| `src/benchmark.py` | Add `extraction_mode` to RunMetrics; add VLM-specific metrics; update `snapshot_run` to accept mode; update `export_combined` to create side-by-side PP-OCR vs VLM columns with NA handling |
| `src/vlm_debug_viz.py` | **NEW** — VLM-mode visualization with text annotations overlaid on full page image |

### 9g. Execution Order

1. Pull model: `ollama pull qwen3-vl:8b`
2. Update `config.yaml` → `model: "qwen3-vl:8b"`
3. Update `src/vlm_fallback.py` → add metrics tracking
4. Create `src/vlm_debug_viz.py` → VLM visualization
5. Update `src/pipeline.py` → VLM viz hook + metrics collection + mode tagging
6. Update `src/benchmark.py` → VLM metrics + side-by-side combined columns
7. **TEST ON FERGUSON ONLY:**
   a. Clean `output/`, set `extraction_mode: "ppocr_grid"`, run on Ferguson
   b. Clean `output/`, set `extraction_mode: "vlm_full_page"`, run on Ferguson
   c. Verify combined benchmark has both `patient_a_ppocr` and `patient_a_vlm` columns
   d. Verify VLM debug image generated with text annotations
   e. Verify PHI anonymization on VLM debug image
8. **FULL RUN (after verification):**
   a. Clean `output/`, run PP-OCR mode on all 3 files
   b. Clean `output/`, run VLM mode on all 3 files
   c. Verify combined benchmark has all 6 columns (3 files × 2 modes)

### 9h. Expected Output After Full Run

**benchmark_combined.xlsx — Per-File Results sheet:**
| Metric | patient_a_ppocr | patient_a_vlm | patient_b_ppocr | patient_b_vlm | patient_c_ppocr | patient_c_vlm | Combined PP-OCR | Combined VLM |
|--------|----------------|---------------|-----------------|---------------|-----------------|---------------|-----------------|-------------|
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Debug images (8 total):**
- `output/debug/patient_a_week1_page1.png` (PP-OCR)
- `output/debug/patient_b_week2_page1.png` (PP-OCR)
- `output/debug/patient_c_week3_page1.png` (PP-OCR)
- `output/debug/patient_c_week3_page3.png` (PP-OCR)
- `output/debug/vlm_patient_a_week1_page1.png` (VLM)
- `output/debug/vlm_patient_b_week2_page1.png` (VLM)
- `output/debug/vlm_patient_c_week3_page1.png` (VLM)
- `output/debug/vlm_patient_c_week3_page3.png` (VLM)
