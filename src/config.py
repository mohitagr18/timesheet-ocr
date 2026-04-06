"""Configuration loader — reads config.yaml and merges with env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ── Config sub-models ───────────────────────────────────────────────


class PathsConfig(BaseModel):
    input_dir: str = "input"
    output_dir: str = "output"
    samples_dir: str = "samples"


class ConfidenceConfig(BaseModel):
    accept_threshold: float = 0.85
    fallback_threshold: float = 0.60


class PpocrConfig(BaseModel):
    lang: str = "en"
    use_textline_orientation: bool = True
    device: str = "cpu"
    text_det_thresh: float = 0.3
    text_rec_batch_size: int = 6


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen2.5vl:7b"
    timeout_seconds: int = 60
    max_retries: int = 2


class PreprocessingConfig(BaseModel):
    target_dpi: int = 300
    denoise: bool = True
    deskew: bool = True
    binarize: bool = True
    adaptive_block_size: int = 11
    adaptive_c: int = 2


class LayoutConfig(BaseModel):
    transposed: bool = False
    header_zone: list[float] = [0.0, 0.0, 1.0, 0.16]
    table_zone: list[float] = [0.0, 0.16, 1.0, 0.98]
    footer_zone: list[float] = [0.0, 0.98, 1.0, 1.0]


class ValidationConfig(BaseModel):
    max_shift_hours: int = 16
    hours_mismatch_tolerance: float = 0.25
    allow_future_dates: bool = False
    max_days_in_past: int = 365
    week_start_day: int = 2  # 0=Mon, 1=Tue, 2=Wed, ..., 6=Sun
    week_length: int = 7


class ExportConfig(BaseModel):
    formats: list[str] = ["xlsx", "csv", "json"]
    excel_sheet_name: str = "Timesheet Data"
    include_review_json: bool = True
    include_report_json: bool = True


class CloudVlmConfig(BaseModel):
    provider: str = "google"
    api_key_env: str = "GOOGLE_API_KEY"
    model: str = "gemini-3-flash-preview"
    timeout_seconds: int = 30
    parallel_workers: int = 2
    media_resolution: str = "high"  # "low" | "medium" | "high" | "ultra_high"
    image_quality: int = 92  # JPEG quality (85-95 recommended)
    use_color_images: bool = True  # Send color images to VLM for better accuracy
    inter_file_delay: int = 5  # Seconds to wait between files (avoid rate limits)


class DebugConfig(BaseModel):
    visualize_ocr: bool = False
    output_dir: str = "output/debug"
    anonymize_phi: bool = True
    signature_zone_fraction: float = 0.30
    signature_ocr_threshold: int = 100


# ── Top-level config ────────────────────────────────────────────────


class AppConfig(BaseModel):
    extraction_mode: str = "vlm_full_page"  # "ppocr_grid" | "vlm_full_page" | "ocr_only" | "layout_guided_vlm_local" | "layout_guided_vlm_cloud"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    ppocr: PpocrConfig = Field(default_factory=PpocrConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    cloud_vlm: CloudVlmConfig = Field(default_factory=CloudVlmConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    # Resolved absolute paths (set during loading)
    project_root: Path = Field(default_factory=lambda: Path.cwd())

    @property
    def input_path(self) -> Path:
        return self.project_root / self.paths.input_dir

    @property
    def output_path(self) -> Path:
        return self.project_root / self.paths.output_dir

    @property
    def samples_path(self) -> Path:
        return self.project_root / self.paths.samples_dir

    @property
    def debug_output_path(self) -> Path:
        return self.project_root / self.debug.output_dir


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file with environment variable overrides.

    Resolution order:
    1. Default values in Pydantic models
    2. Values from config.yaml
    3. Environment variable overrides (TIMESHEET_OCR_*)
    """
    project_root = _find_project_root()

    if config_path is None:
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)

    data: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    _apply_env_overrides(data)

    config = AppConfig(**data, project_root=project_root)

    # Ensure output directory exists
    config.output_path.mkdir(parents=True, exist_ok=True)

    return config


def _find_project_root() -> Path:
    """Walk upward to find pyproject.toml as root marker."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Override config values from environment variables.

    Convention: TIMESHEET_OCR_<SECTION>_<KEY>=value
    Example: TIMESHEET_OCR_CONFIDENCE_ACCEPT_THRESHOLD=0.90
    """
    prefix = "TIMESHEET_OCR_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field = parts
        if section not in data:
            data[section] = {}
        # Attempt type coercion
        try:
            data[section][field] = float(value)
        except ValueError:
            data[section][field] = value
