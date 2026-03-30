# Timesheet OCR — Task Tracker

- [x] Phase 1: Project Scaffolding & Dependencies
  - [x] Initialize uv project with pyproject.toml
  - [x] Add runtime dependencies
  - [x] Add dev dependencies
  - [x] Create src/timesheet_ocr/ package structure
  - [x] Create config.yaml
  - [x] Create output/ and samples/ directories
  - [x] Run `uv sync` to verify

- [x] Phase 2: Data Models, Config & Preprocessing
  - [x] Pydantic models (models.py)
  - [x] Config loader (config.py + config.yaml)
  - [x] Image preprocessing (preprocessing.py)
  - [x] Unit tests for models

- [x] Phase 3: PP-OCRv5 Engine & Layout Detection
  - [x] PP-OCRv5 wrapper (ocr_engine.py)
  - [x] Layout zone config (layout.py)
  - [x] Grid row detection
  - [x] Cell-to-field mapping
  - [x] Confidence aggregation (confidence.py)

- [x] Phase 4: Qwen2.5-VL Fallback & Confidence Router
  - [x] Ollama client wrapper (vlm_fallback.py)
  - [x] Prompt engineering
  - [x] Confidence router

- [x] Phase 5: Structured Parsing, Validation & Review Queue
  - [x] Parser (parser.py)
  - [x] Validation rules with overnight shift support (validation.py)
  - [x] Review queue builder (review_queue.py)

- [ ] Phase 6: Export, CLI & Documentation
  - [x] Excel exporter (exporter.py)
  - [x] CSV + JSON exporters
  - [x] CLI entry point (__main__.py)
  - [x] Pipeline orchestrator (pipeline.py)
  - [x] README documentation
  - [ ] End-to-end test with real PDFs
