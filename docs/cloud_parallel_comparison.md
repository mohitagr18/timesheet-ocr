# Cloud Approach: Parallel Processing Performance Comparison

**Generated:** 2026-04-06 10:59:18
**Approach:** `layout_guided_vlm_cloud` (with parallel VLM API requests)

## New Run (Parallel Processing Enabled)

- **Total Time:** 538.5s
- **Files Processed:** 3
- **Total Rows Extracted:** 18

### Per-File Breakdown

| File | Time (s) | Rows | Accepted | Flagged | Failed |
|------|----------|------|----------|---------|--------|
| C.Ferguson Timesheets - 010726-011326.pdf | 108.9 | 5 | 5 | 0 | 0 |
| J.Flemming Timesheets - 012826-020326.pdf | 295.2 | 7 | 7 | 0 | 0 |
| K.Drewry Timesheets 020426-021026.pdf | 134.4 | 6 | 6 | 0 | 0 |

## Previous Runs (Sequential Processing)

| Date/Log | Total Time (s) | Notes |
|----------|---------------|-------|
| 20260405_183547 | 249.3 | Found in batch_run_20260405_183547.log |

## Performance Improvement

- **Previous Average:** 249.3s
- **New (Parallel):** 538.5s
- **Time Saved:** -289.2s
- **Improvement:** -116.0%

⚠️ **Parallel processing did not show improvement in this run.**

Possible reasons:
- API rate limiting may be throttling parallel requests
- Network latency is not the bottleneck
- OCR/layout detection is the dominant cost
- Small number of files reduces the benefit of parallelism

## Technical Details

### Parallel Processing Strategy

The parallel implementation works as follows:

1. **Phase 1 (Sequential, CPU-bound):** OCR + layout detection on all pages
2. **Phase 2 (Parallel, I/O-bound):** All VLM API requests sent concurrently via `ThreadPoolExecutor`
3. **Phase 3 (Sequential, CPU-bound):** Process results into structured records

### Configuration

```yaml
cloud_vlm:
  parallel_workers: 3
  model: gemini-3-flash-preview
```

### Implementation Files

- `src/vlm_cloud.py`: Added `batch_extract_table_crops()` with `ThreadPoolExecutor`
- `src/pipeline.py`: Added `_process_file_cloud_batch()` for parallel routing
- `config.yaml`: Added `parallel_workers: 3` setting
