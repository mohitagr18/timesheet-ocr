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
    use_angle_cls: bool = True
    use_gpu: bool = False
    det_db_thresh: float = 0.3
    rec_batch_num: int = 6


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


class ColumnBounds(BaseModel):
    date: list[float] = [0.0, 0.20]
    time_in: list[float] = [0.20, 0.40]
    time_out: list[float] = [0.40, 0.60]
    total_hours: list[float] = [0.60, 0.75]
    notes: list[float] = [0.75, 1.0]


class LayoutConfig(BaseModel):
    transposed: bool = False
    header_zone: list[float] = [0.0, 0.0, 1.0, 0.16]
    table_zone: list[float] = [0.0, 0.16, 1.0, 0.94]
    footer_zone: list[float] = [0.0, 0.94, 1.0, 1.0]
    columns: ColumnBounds = Field(default_factory=ColumnBounds)


class ValidationConfig(BaseModel):
    max_shift_hours: int = 16
    hours_mismatch_tolerance: float = 0.25
    allow_future_dates: bool = False
    max_days_in_past: int = 365


class ExportConfig(BaseModel):
    formats: list[str] = ["xlsx", "csv", "json"]
    excel_sheet_name: str = "Timesheet Data"


# ── Top-level config ────────────────────────────────────────────────

class AppConfig(BaseModel):
    extraction_mode: str = "vlm_full_page"  # "ppocr_grid" or "vlm_full_page"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    ppocr: PpocrConfig = Field(default_factory=PpocrConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

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
        parts = key[len(prefix):].lower().split("_", 1)
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
