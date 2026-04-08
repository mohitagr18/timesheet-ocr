# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- 5 extraction approaches for comprehensive benchmarking:
  - **OCR Only (Baseline)** — PaddleOCR without any VLM fallback
  - **OCR + VLM Fallback** (`ppocr_grid`) — PaddleOCR with per-cell VLM fallback on low confidence
  - **VLM Full Page** (`vlm_full_page`) — Full page VLM extraction (no PaddleOCR parsing)
  - **Layout-Guided VLM (Local)** — PP-DocLayoutV3 table detection + local VLM on cropped table
  - **Layout-Guided VLM (Cloud)** — PP-DocLayoutV3 table detection + cloud VLM (Gemini) on cropped table
- Combined benchmark comparison across all 5 approaches (`output/combined/benchmark_combined.xlsx`)
- Transposed row-level comparison table (dates as rows, approaches as columns)
- PHI/PII anonymization system (deterministic patient/employee name mapping)
- Signature page detection and skip logic
- Debug visualization with approach-specific prefixes
- `CHANGELOG.md`, `.env.example`, `samples/` directory structure

### Changed
- Row-level comparison table redesigned: dates as rows, Hours+Status per approach as columns
- Debug images now use approach-specific prefixes for easy identification
- Implementation plans moved to `docs/` directory

### Fixed
- Summary metrics NA issue — corrected metric key mappings between benchmark.py and rebuild_combined_report.py
- Corrupted `accept_threshold` config value
- Benchmark accumulation bug — pages/rows now reset per file
- Combined report column index bugs — rebuilt from scratch in `rebuild_combined_report.py`
- Hours accuracy metric inflated by partial matches — now uses actual hours tolerance check

### Removed
- `debug_layout.py` scratch script
- Notes column from extraction and output
- Redundant debug files from repository root

---

## [0.1.0] — Initial Release

- Basic PaddleOCR + VLM fallback pipeline
- PDF/image input support
- Excel, CSV, JSON export
- Confidence-based routing (accept/fallback/review)
- Validation engine with overnight shift support
- Review queue for flagged rows
- PHI anonymization
- Benchmark system with combined export
