# Timesheet OCR

<div align="center">
  <h3>A privacy-first pipeline to extract, validate, and structure data from scanned handwritten home-health timesheets using 6 distinct approaches.</h3>
</div>

<br/>

Designed to convert messy, handwritten PDF uploads into structured, validated Excel databases while benchmarking extraction quality across OCR-only, hybrid, and full VLM approaches.

---

## 🧠 System Architecture

```mermaid
graph TD
    A[Upload Timesheet PDFs] --> B[Pipeline Orchestrator]
    B --> C[Preprocess & Deskew]
    C --> D{extraction_mode}

    D -->|ocr_only| E1[PaddleOCR Grid — No Confidence Routing, No VLM]
    E1 --> F1[Parse & Validate — No VLM]

    D -->|ppocr_grid| E2[PaddleOCR Grid + Confidence-Gated VLM Fallback]
    E2 --> F2{Cell Confidence Check}
    F2 -->|Low| G2[VLM Fallback via Ollama]
    F2 -->|OK| H2[Parse & Validate]
    G2 --> H2

    D -->|vlm_full_page| E3[Full Page → VLM via Ollama]
    E3 --> F3[Parse JSON Shifts & Validate]

    D -->|layout_guided_vlm_local| E4[PP-DocLayoutV3 Table Crop]
    E4 --> F4[Local VLM via Ollama]
    F4 --> G4[Parse JSON Shifts & Validate]

    D -->|layout_guided_vlm_cloud| E5[PP-DocLayoutV3 Table Crop]
    E5 --> F5[Cloud VLM API Gemini]
    F5 --> G5[Parse JSON Shifts & Validate]

    D -->|band_crop_vlm_cloud| E6[Surgical Two-Band Crop]
    E6 --> F6[Cloud VLM API Gemini]
    F6 --> G6[Parse JSON Shifts & Validate]

    F1 & H2 & F3 & G4 & G5 & G6 --> I[Validation Engine]
    I -->|Errors| J[Review Queue]
    I -->|Clean| K[Accepted]
    J & K --> L[Export: Excel + JSON + Benchmark]
```

---

## 🔬 6 Extraction Approaches

| # | Approach | Mode | Description | Speed | Best For |
|---|----------|------|-------------|-------|----------|
| 1 | **OCR Only** | `ocr_only` | PaddleOCR grid extraction with zero VLM involvement. Empty cells stay empty. | ⚡⚡ Fast | Baseline comparison, printed forms |
| 2 | **OCR + VLM Fallback** | `ppocr_grid` | PaddleOCR grid extraction with per-cell VLM fallback on low confidence. | ⚡⚡ Moderate | Standardized forms, privacy-first |
| 3 | **VLM Full Page** | `vlm_full_page` | Entire page sent to local VLM for structured JSON extraction. | ⚡ Slow | Messy layouts, cursive handwriting |
| 4 | **Layout-Guided VLM (Local)** | `layout_guided_vlm_local` | PP-DocLayoutV3 detects table zone, crops it, sends to local VLM. | ⚡ Slow | Balance of accuracy + privacy |
| 5 | **Layout-Guided VLM (Cloud)** | `layout_guided_vlm_cloud` | PP-DocLayoutV3 detects table zone, crops it, sends to cloud VLM (Gemini). | ⚡⚡ Moderate | Maximum accuracy, API available |
| 6 | **Band-Crop VLM (Cloud)** | `band_crop_vlm_cloud` | PP-DocLayoutV3 detects table, then surgical two-band crop (DATE row + footer block) — only billing fields transmitted to Gemini. Zero clinical PHI sent to cloud. | ⚡⚡⚡ Fastest | Maximum accuracy + strongest PHI compliance |

### Workflow Diagrams

Detailed Mermaid diagrams for each approach are in the [`workflows/`](workflows/) directory:

- [`workflows/ocr_only_flow.md`](workflows/ocr_only_flow.md)
- [`workflows/ppocr_grid_flow.md`](workflows/ppocr_grid_flow.md)
- [`workflows/vlm_full_page_flow.md`](workflows/vlm_full_page_flow.md)
- [`workflows/layout_guided_vlm_local_flow.md`](workflows/layout_guided_vlm_local_flow.md)
- [`workflows/layout_guided_vlm_cloud_flow.md`](workflows/layout_guided_vlm_cloud_flow.md)
- [`workflows/band_crop_vlm_cloud_flow.md`](workflows/band_crop_vlm_cloud_flow.md)
- [`workflows/ground_truth_comparison.md`](workflows/ground_truth_comparison.md) — Ground truth comparison workflow

---

## 📊 Benchmark Results

Results from processing **120 timesheet pages** with ground truth comparison:

| Metric | OCR Only | OCR + VLM Fallback | VLM Full Page | Layout-Guided VLM (Local) | Layout-Guided VLM (Cloud) | Band-Crop VLM (Cloud) |
|--------|---------|-------------------|---------------|--------------------------|--------------------------|----------------------|
| **Processing Time (s)** | 3588.37 | 6140.04 | 25339.01 | 24684.40 | 7340.12 | **4540.60** |
| **Pages Processed** | 120 | 120 | 120 | 120 | 120 | 120 |
| **Rows Extracted** | 278 | 346 | 222 | 240 | 235 | 225 |
| **Accepted Rows** | 28 | 34 | 140 | 170 | 182 | **183** |
| **Flagged Rows** | 250 | 312 | 82 | 70 | 53 | 42 |
| **Hours Mismatch Rate** | 79.5% | 80.6% | 11.9% | 22.2% | 17.4% | **9.0%** |
| **GT Hours Accuracy (±15min)** | 20.5% | 19.4% | 88.1% | 77.8% | 82.6% | **91.0%** |
| **GT Time-In Accuracy (±30min)** | 26.1% | 21.5% | 77.6% | 69.1% | 82.6% | **89.9%** |
| **GT Time-Out Accuracy (±30min)** | 34.1% | 30.1% | 79.1% | 77.8% | 80.4% | **87.6%** |

### Key Findings

- **Band-Crop VLM (Cloud)** achieves the best overall accuracy: 91.0% GT hours accuracy (±15min), lowest hours mismatch (9.0%), and highest accepted rows (183)
- **Band-Crop VLM (Cloud)** is also the fastest cloud approach (4540.60s for 120 pages) — 1.6× faster than layout-guided cloud (7340.12s)
- **Zero clinical PHI** transmitted — only billing fields (DATE row + footer) sent to Gemini
- **Layout-Guided VLM (Cloud)** is second best: 82.6% hours accuracy, 17.4% mismatch
- **VLM Full Page** has highest local accuracy (88.1% hours) but is slowest (25339s) due to full-page inference
- **OCR Only** and **OCR+VLM Fallback** have poor accuracy (~20%) due to handwritten text recognition challenges

---

## 📏 Ground Truth Comparison & Extraction Accuracy

The pipeline includes a ground truth comparison workflow that evaluates extraction accuracy against manually-annotated reference data:

### How It Works

1. **Fill in ground truth**: Manually enter expected values in `ground_truth.xlsx` (project root)
   - Columns: `source_file`, `date`, `total_hours`, `time_in`, `time_out`, `employee_name`
2. **Run all 6 approaches**: Use `scripts/run_all_approaches_safe.py` to generate benchmark data
3. **Automatic**: Combined metrics and ground truth comparison are generated automatically after all approaches complete:
   - `output/combined/benchmark_combined.xlsx` — approach comparison + Human-Verified Results + IEEE Paper Results

### Metrics Computed

The comparison script computes two categories of metrics:

#### 1. Field-Level Accuracy

Each extracted row is evaluated against ground truth on a per-field basis:

| Field | Tolerance | Definition |
|-------|-----------|------------|
| **Date** | Exact match | Parsed date must equal ground truth date |
| **Hours** | ±0.25 hours (15 min) | Computed or extracted hours within tolerance |
| **Time In** | ±30 minutes | Clock-in time within tolerance |
| **Time Out** | ±30 minutes | Clock-out time within tolerance |

**Composite metrics:**
- **Fully Correct**: All 3 time/hour fields match
- **Partial or Full Match**: At least one field matches
- **Not Extracted**: No matching row found in the approach's output

#### 2. Pipeline Validation Quality

Measures how well the pipeline's *internal* validation status (accepted/flagged/failed) correlates with *actual* correctness:

| Metric | Definition | Ideal |
|--------|------------|-------|
| **Validation Precision** | Of rows marked "accepted", % that are fully correct | 100% |
| **Validation Recall** | Of all fully correct rows, % that were accepted | 100% |
| **Validation F1** | Harmonic mean of precision and recall | 1.000 |
| **False Accept Rate** | Of accepted rows, % that are actually wrong | 0% |
| **Missed Detection Rate** | Of correct rows, % that were flagged/failed | 0% |

### Output

Results are written to the `Human-Verified Results` sheet in `output/combined/benchmark_combined.xlsx` with these sections:

- **Section 1**: Extraction Coverage & Field-Level Accuracy (per-field rates, duplicate/hallucinated row counts, fully correct counts)
- **Section 2**: Pipeline Validation Quality (precision, recall, false accept rate)
- **Section 3**: Per-Row Detailed Comparison (side-by-side hours, correctness checkmarks, and status for all 5 approaches)
- **Section 4**: Duplicate Rows (approach extracted same date but not best match — source timesheet had duplicates)
- **Section 5**: Extra Rows (approach extracted dates not in ground truth — true hallucinations)

---

## 💻 Quick Start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) for Python dependency management
- [Ollama](https://ollama.com/) running locally (for local VLM modes)
- Google API key (for cloud VLM mode)

```bash
# Install dependencies
uv sync

# Pull the local VLM model (required for ppocr_grid, vlm_full_page, layout_guided_vlm_local)
ollama pull qwen2.5vl:7b

# Set up environment variables (for cloud VLM mode)
cp .env.example .env
# Edit .env and add your API keys
```

### Run the Pipeline

```bash
# Place PDFs/images in input/
# Set extraction_mode in config.yaml
uv run timesheet-ocr --verbose
```

### Run All 6 Approaches for Benchmarking

```bash
# Run all 6 approaches (uses config.yaml extraction_mode internally)
uv run python scripts/run_all_approaches_safe.py

# If interrupted, resume from last completed approach:
uv run python scripts/run_all_approaches_safe.py --resume
```

---

## ⚙️ Configuration

Key settings in `config.yaml`:

```yaml
extraction_mode: "ppocr_grid"   # ocr_only | ppocr_grid | vlm_full_page | layout_guided_vlm_local | layout_guided_vlm_cloud | band_crop_vlm_cloud

confidence:
  accept_threshold: 0.90        # Accept OCR results above this confidence
  fallback_threshold: 0.75      # Between this and accept_threshold → VLM fallback; below this → flag for review

layout:
  transposed: true              # Timesheet has dates as columns
  header_zone: [0.0, 0.0, 1.0, 0.16]
  table_zone: [0.24, 0.16, 1.0, 0.98]

debug:
  visualize_ocr: false
  signature_ocr_threshold: 100  # OCR box count below which a page is treated as a signature/summary page

ollama:
  model: "qwen2.5vl:7b"         # Local VLM model
  timeout_seconds: 60

cloud_vlm:
  provider: "google"
  model: "gemini-2.5-flash"
  media_resolution: "high"      # "low" | "medium" | "high" | "ultra_high"
  image_quality: 92             # JPEG quality sent to VLM (85–95 recommended)
  inter_file_delay: 5           # Seconds between files (free-tier rate limiting)
  inter_page_delay: 4           # Seconds between pages within a file
```

> **Note:** The code-level defaults in `src/config.py` are `accept_threshold: 0.85` and `fallback_threshold: 0.60`. The values above (0.90/0.75) reflect the `config.yaml` overrides for this project's timesheet template.

---

## 📁 Output Structure

```
output/
├── ocr_only/                    # Approach 1 results
│   ├── benchmark_patient_a_week1.xlsx
│   ├── merged_results.xlsx
│   └── patient_a_week1_report.json
├── ppocr_grid/                  # Approach 2 results
├── vlm_full_page/               # Approach 3 results
├── layout_guided_vlm_local/     # Approach 4 results
├── layout_guided_vlm_cloud/     # Approach 5 results
├── band_crop_vlm_cloud/         # Approach 6 results
├── combined/                    # Combined benchmark across all approaches
│   ├── benchmark_combined.xlsx  # Summary + Approach Comparison + Human-Verified Results + IEEE Paper Results
│   └── debug/                   # Debug images from all approaches
├── debug/                       # Shared debug images
└── reports/
    └── report_YYYYMMDD_HHMMSS.txt  # Per-file success/failure audit trail

ground_truth.xlsx                # Manually-filled reference data (git-ignored, at project root)
```

Each approach directory contains:
- `benchmark_*.xlsx` — Per-run benchmark with Run Summary, Page Details, and Row-Level sheets
- `merged_results.xlsx` — Consolidated extraction results
- `*_report.json` — Technical audit log
- `*_review.json` — Flagged rows for human review

The `benchmark_combined.xlsx` file contains:
- **Approach Comparison** — Summary metrics + row-level comparison across all approaches
- **Human-Verified Results** — Ground truth comparison with 3 sections:
  - Extraction Coverage & Field-Level Accuracy (Date, Hours, Time In/Out rates, plus duplicate and hallucinated row counts)
  - Pipeline Validation Quality (precision, recall, false accept rate)
  - Per-Row Detailed Comparison (side-by-side values with ✓/✗ checkmarks)
- **IEEE Paper Results** — Formatted for academic paper: Performance Metrics + Accuracy Metrics tables

---

## 🔒 PHI/PII Anonymization

The pipeline includes automatic PHI anonymization for benchmarking:

- **Patient names** → `Patient_A`, `Patient_B`, `Patient_C` (deterministic, sorted by filename)
- **Employee names** → `Employee_A`, `Employee_B`, etc.
- **Filenames** → `patient_a_week1.pdf`, etc.
- **Signature pages** — Detected and skipped (no row extraction, no visualization)

All benchmark outputs use anonymized names. Original data remains in `output/{approach}/` directories.

---

## 📐 Adding Support for New Timesheet Templates

1. Place a sample PDF in `input/`
2. Enable debug visualization in `config.yaml`:
   ```yaml
   debug:
     visualize_ocr: true
   ```
3. Run the pipeline and inspect debug images in `output/debug/`
4. Adjust `layout:` zone fractions in `config.yaml` to match the new template's structure
5. Verify extraction quality and iterate

---

## 📋 Monitoring & Logs

All batch runs write logs to the `logs/` directory:

| File | Contents | When to Use |
|------|----------|-------------|
| `logs/batch_run_YYYYMMDD_HHMMSS.log` | Full detailed log of every page, file, and approach | Post-run debugging |
| `logs/latest.log` | Symlink that always points to the most recent log | Live monitoring during a run |

### Watch a Live Run

```bash
tail -f logs/latest.log
```

### Check for Errors After a Run

```bash
grep -E "ERROR|FAILED" logs/latest.log
```

### Run State File

After every completed approach, the script writes progress to `output/.run_state.json`. This file tracks which approaches are fully done. If the machine shuts down mid-run, use `--resume` to pick up from the last completed approach:

```bash
uv run python scripts/run_all_approaches_safe.py --resume
```

> **Note:** Resume is per-approach, not per-file. If a run is interrupted mid-approach, that approach will restart from its first file.

### Per-File Status

At the end of each approach run, the terminal prints a per-file summary showing `✓` (success) or `✗` (failed) for every input file. The same information is written to `reports/report_YYYYMMDD_HHMMSS.txt` for a persistent audit trail.

---

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
