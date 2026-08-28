"""Application configuration and filesystem path handling."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable


MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024

DEFAULT_FUSION_ALPHA = 0.75
DEFAULT_FUSION_UNCERTAINTY_MARGIN = 0.10
DEFAULT_PNEUMONIA_DDX_LABEL = "Pneumonia"


class Settings:
    """Resolve runtime paths from environment variables with safe defaults."""

    def __init__(self) -> None:
        self.api_dir = Path(__file__).resolve().parent
        self.project_root = self.api_dir.parent

        self.data_raw_dir = self._env_path(
            "DDXPLUS_DATA_DIR",
            self.project_root / "data" / "raw",
        )
        self.model_dir = self._env_path("MODEL_DIR", self.api_dir / "artifacts")

        self.metadata_dir = self.api_dir / "metadata"
        self.evidences_json_path = self._metadata_path(
            env_name="EVIDENCES_JSON_PATH",
            default_path=self.data_raw_dir / "release_evidences.json",
            fallback_path=self.metadata_dir / "release_evidences.json",
        )
        self.evidence_display_en_path = self._metadata_path(
            env_name="EVIDENCE_DISPLAY_EN_PATH",
            default_path=self.data_raw_dir / "evidence_display_en.json",
            fallback_path=self.metadata_dir / "evidence_display_en.json",
        )
        self.conditions_json_path = self._metadata_path(
            env_name="CONDITIONS_JSON_PATH",
            default_path=self.data_raw_dir / "release_conditions.json",
            fallback_path=self.metadata_dir / "release_conditions.json",
        )

        self.model_path = self.model_dir / "best_model.pkl"
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"
        self.label_encoder_path = self.model_dir / "label_encoder.pkl"
        self.metrics_path = self.model_dir / "model_metrics.json"

        # The chest X-ray model lives in its own directory and never reuses,
        # renames or overwrites the DDXPlus artifacts above.
        self.pneumonia_model_dir = self._env_path(
            "PNEUMONIA_MODEL_DIR",
            self.model_dir / "advanced_pneumonia",
        )
        self.pneumonia_model_path = self.pneumonia_model_dir / "advanced_pneumonia_model.onnx"
        self.pneumonia_config_path = self.pneumonia_model_dir / "advanced_pneumonia_config.json"
        self.pneumonia_metrics_path = self.pneumonia_model_dir / "advanced_pneumonia_metrics.json"
        self.pneumonia_manifest_path = self.pneumonia_model_dir / "advanced_pneumonia_manifest.json"

        self.max_image_upload_bytes = MAX_IMAGE_UPLOAD_BYTES

        # Demonstration fusion parameters. They are not clinically validated.
        self.pneumonia_fusion_alpha = self._env_float(
            "PNEUMONIA_FUSION_ALPHA", DEFAULT_FUSION_ALPHA
        )
        self.pneumonia_fusion_uncertainty_margin = self._env_float(
            "PNEUMONIA_FUSION_UNCERTAINTY_MARGIN", DEFAULT_FUSION_UNCERTAINTY_MARGIN
        )
        self.pneumonia_ddx_label = (
            os.getenv("PNEUMONIA_DDX_LABEL") or DEFAULT_PNEUMONIA_DDX_LABEL
        ).strip() or DEFAULT_PNEUMONIA_DDX_LABEL

    @staticmethod
    def _env_float(env_name: str, default_value: float) -> float:
        """Read a float setting, falling back to the default when unparseable.

        Range validation belongs to the consuming service so that a bad value
        disables fusion instead of preventing application startup.
        """
        raw = os.getenv(env_name)
        if raw is None or not raw.strip():
            return default_value
        try:
            return float(raw.strip())
        except ValueError:
            return default_value

    @staticmethod
    def _env_path(env_name: str, default_path: Path) -> Path:
        value = os.getenv(env_name)
        if value:
            return Path(value).expanduser().resolve()
        return default_path.resolve()

    def _metadata_path(self, env_name: str, default_path: Path, fallback_path: Path) -> Path:
        env_value = os.getenv(env_name)
        if env_value:
            return Path(env_value).expanduser().resolve()
        if default_path.exists():
            return default_path.resolve()
        return fallback_path.resolve()

    def detect_data_raw_files(self) -> list[str]:
        """Return detected files in data/raw without reading large datasets."""
        if not self.data_raw_dir.exists():
            return []
        return sorted(path.name for path in self.data_raw_dir.iterdir() if path.is_file())

    def display_path(self, path: Path) -> str:
        """Show project-relative paths when possible for clearer API responses."""
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def missing_paths(self, paths: Iterable[Path]) -> list[str]:
        return [self.display_path(path) for path in paths if not path.exists()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
