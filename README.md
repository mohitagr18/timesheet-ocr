# Timesheet OCR

A fully local pipeline designed to extract structured data from scanned handwritten home-health timesheets. 
Uses PP-OCRv5 for primary extraction and Qwen2.5-VL via Ollama for low-confidence cell fallback.

## Requirements

### 1. System Dependencies 
Since the pipeline processes `.pdf` documents directly, you need the system-level C++ library `poppler` installed. This provides the tools necessary to convert PDFs into useable images.
- **macOS (via Homebrew):**
  ```bash
  brew install poppler
  ```
- **Ubuntu/Debian:** 
  ```bash
  sudo apt-get install poppler-utils
  ```

### 2. Ollama
Used as the VLM fallback for unreadable handwritten cells.
- Install Ollama from [ollama.com](https://ollama.com/)
- Pull the model: `ollama run qwen2.5vl:7b`
- **Make sure the Ollama app is running before executing the pipeline.**

### 3. Python Environment
This project uses `uv` for lightning-fast dependency management.
```bash
# Sync all dependencies
uv sync
```

## Quick Start

1. Drop your timesheet PDF files into the `input/` folder.
2. Run the pipeline:
   ```bash
   uv run python -m timesheet_ocr --verbose
   ```
3. Check the `output/` folder for your results! You will see:
   - `_results.xlsx` / `.csv` / `.json` (Extracted usable data)
   - `_review.json` (Flagged cells that require human intervention)
   - `_report.json` (Processing metrics)

## Configuration
The extraction rules, confidence thresholds, and zone layout pixel boundaries are defined in `config.yaml`. Feel free to adjust these to precisely match your specific timesheet forms.