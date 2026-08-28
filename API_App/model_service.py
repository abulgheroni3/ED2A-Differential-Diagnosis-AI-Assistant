"""Generic model artifact loading and prediction orchestration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import Settings
from metadata_service import MetadataService
from schemas import PredictionRequest


DISCLAIMER = "Educational decision-support prototype. Not a real medical diagnosis."


class ModelNotReadyError(RuntimeError):
    def __init__(self, message: str, missing_files: list[str]) -> None:
        super().__init__(message)
        self.message = message
        self.missing_files = missing_files


class PreprocessingError(RuntimeError):
    pass


class PredictionError(RuntimeError):
    pass


class ModelService:
    """Load sklearn-compatible artifacts and keep prediction logic model-agnostic."""

    def __init__(self, settings: Settings, metadata_service: MetadataService) -> None:
        self.settings = settings
        self.metadata_service = metadata_service
        self.model: Any | None = None
        self.preprocessor: Any | None = None
        self.label_encoder: Any | None = None
        self.load_errors: list[str] = []
        self.load_artifacts()

    def _artifact_paths(self) -> list[Path]:
        return [
            self.settings.model_path,
            self.settings.preprocessor_path,
            self.settings.label_encoder_path,
        ]

    def missing_artifacts(self) -> list[str]:
        return self.settings.missing_paths(self._artifact_paths())

    def load_artifacts(self) -> None:
        self.load_errors = []
        self.model = self._load_optional(self.settings.model_path, "model")
        self.preprocessor = self._load_optional(self.settings.preprocessor_path, "preprocessor")
        self.label_encoder = self._load_optional(self.settings.label_encoder_path, "label encoder")

    def _load_optional(self, path: Path, label: str) -> Any | None:
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - convert artifact failures into status messages
            self.load_errors.append(f"Could not load {label} from {self.settings.display_path(path)}: {exc}")
            return None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def preprocessor_loaded(self) -> bool:
        return self.preprocessor is not None

    @property
    def label_encoder_loaded(self) -> bool:
        return self.label_encoder is not None

    def ready_for_prediction(self) -> bool:
        return self.model_loaded and self.preprocessor_loaded

    def _loaded_artifact_paths(self) -> list[Path]:
        """Return the inference artifacts that exist and were loaded successfully."""
        loaded = [
            (self.settings.model_path, self.model),
            (self.settings.preprocessor_path, self.preprocessor),
            (self.settings.label_encoder_path, self.label_encoder),
        ]
        return [path for path, artifact in loaded if artifact is not None]

    def model_freshness(self) -> dict[str, str] | None:
        """Return the newest modification time among the loaded inference artifacts.

        The timestamp is derived only from the filesystem, never from the current
        time, the metrics file or the application start time.
        """
        newest_mtime: float | None = None
        newest_name: str | None = None

        for path in self._loaded_artifact_paths():
            try:
                mtime = path.stat().st_mtime
            except OSError:  # artifact vanished or is unreadable between load and stat
                continue
            if newest_mtime is None or mtime > newest_mtime:
                newest_mtime = mtime
                newest_name = path.name

        if newest_mtime is None or newest_name is None:
            return None

        return {
            "timestamp_utc": datetime.fromtimestamp(newest_mtime, tz=timezone.utc).isoformat(),
            "latest_artifact": newest_name,
        }

    def _class_labels(self) -> list[str]:
        if self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
            return [str(label) for label in self.label_encoder.classes_]
        if self.model is not None and hasattr(self.model, "classes_"):
            return [str(label) for label in self.model.classes_]
        return []

    def info(self) -> dict[str, Any]:
        self.load_artifacts()
        missing = self.missing_artifacts()

        if not self.model_loaded:
            return {
                "model_loaded": False,
                "message": "Model artifacts are not available yet. Run the training pipeline and export artifacts first.",
                "missing_files": missing,
                "load_errors": self.load_errors,
            }

        classes = self._class_labels()
        return {
            "model_loaded": self.model_loaded,
            "model_type": type(self.model).__name__,
            "preprocessor_loaded": self.preprocessor_loaded,
            "label_encoder_loaded": self.label_encoder_loaded,
            "ready_for_prediction": self.ready_for_prediction(),
            "n_classes": len(classes) if classes else None,
            "classes": classes,
            "missing_files": missing,
            "load_errors": self.load_errors,
        }

    def metrics(self) -> dict[str, Any]:
        if not self.settings.metrics_path.exists():
            return {
                "metrics_available": False,
                "message": "model_metrics.json is not available yet. Export metrics after model evaluation.",
                "missing_files": [self.settings.display_path(self.settings.metrics_path)],
            }
        with self.settings.metrics_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def build_feature_frame(request: PredictionRequest) -> pd.DataFrame:
        """Build the non-leaking runtime feature frame expected by future preprocessors."""
        return pd.DataFrame(
            [
                {
                    "AGE": request.age,
                    "SEX": request.sex,
                    "EVIDENCES": list(request.evidences),
                    "INITIAL_EVIDENCE": request.initial_evidence,
                }
            ]
        )

    def _transform(self, feature_frame: pd.DataFrame) -> Any:
        if self.preprocessor is None:
            raise PreprocessingError("Preprocessor artifact is not loaded.")
        try:
            if hasattr(self.preprocessor, "transform"):
                return self.preprocessor.transform(feature_frame)
            if callable(self.preprocessor):
                return self.preprocessor(feature_frame)
        except Exception as exc:  # noqa: BLE001 - expose a controlled API error
            raise PreprocessingError(f"Preprocessing failed: {exc}") from exc
        raise PreprocessingError("Preprocessor must provide a transform method or be callable.")

    def _decode_prediction_label(self, label: Any) -> str:
        if isinstance(label, np.generic):
            label = label.item()
        if self.label_encoder is not None and hasattr(self.label_encoder, "inverse_transform"):
            try:
                decoded = self.label_encoder.inverse_transform([label])
                return str(decoded[0])
            except Exception:
                pass
        return str(label)

    def _aligned_distribution(self, probabilities: Any) -> tuple[list[str], np.ndarray]:
        """Return the complete class distribution aligned with its labels.

        A single inference feeds both the Top-k response and the multimodal
        fusion layer, so the model is never evaluated twice for one request.
        """
        proba_array = np.asarray(probabilities)
        if proba_array.ndim == 2:
            proba_array = proba_array[0]
        if proba_array.ndim != 1:
            raise PredictionError("Model returned probabilities with an unsupported shape.")

        labels = self._class_labels()
        if labels and len(labels) != len(proba_array):
            raise PredictionError("Number of class labels does not match probability output.")
        if not labels:
            labels = [f"class_index_{idx}" for idx in range(len(proba_array))]
        return labels, proba_array

    def _top_k_from_distribution(
        self, labels: list[str], proba_array: np.ndarray, k: int
    ) -> list[dict[str, Any]]:
        top_indices = np.argsort(proba_array)[::-1][: min(k, len(proba_array))]
        diagnoses: list[dict[str, Any]] = []
        for index in top_indices:
            diagnosis = labels[int(index)]
            condition_details = self.metadata_service.condition_summary(diagnosis)
            diagnoses.append(
                {
                    "diagnosis": diagnosis,
                    "probability": float(proba_array[int(index)]),
                    "icd10_id": condition_details.get("icd10_id"),
                    "severity": condition_details.get("severity"),
                }
            )
        return diagnoses

    def predict(
        self,
        request: PredictionRequest,
        k: int = 5,
        include_distribution: bool = False,
    ) -> dict[str, Any]:
        self.load_artifacts()
        missing = self.missing_artifacts()
        if self.model is None or self.preprocessor is None:
            raise ModelNotReadyError(
                "Model and preprocessor artifacts are not available yet. Run training and export artifacts first.",
                missing,
            )

        feature_frame = self.build_feature_frame(request)
        transformed_features = self._transform(feature_frame)

        class_distribution: dict[str, Any] | None = None
        try:
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(transformed_features)
                labels, proba_array = self._aligned_distribution(probabilities)
                top_k = self._top_k_from_distribution(labels, proba_array, k)
                predicted_diagnosis = top_k[0]["diagnosis"]
                confidence = top_k[0]["probability"]
                if include_distribution:
                    class_distribution = {
                        "labels": list(labels),
                        "probabilities": [float(value) for value in proba_array],
                    }
            else:
                raw_prediction = self.model.predict(transformed_features)
                predicted_diagnosis = self._decode_prediction_label(raw_prediction[0])
                condition_details = self.metadata_service.condition_summary(predicted_diagnosis)
                top_k = [
                    {
                        "diagnosis": predicted_diagnosis,
                        "probability": None,
                        "icd10_id": condition_details.get("icd10_id"),
                        "severity": condition_details.get("severity"),
                    }
                ]
                confidence = None
        except PreprocessingError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert unknown model errors into controlled response
            raise PredictionError(f"Prediction failed: {exc}") from exc

        payload: dict[str, Any] = {
            "predicted_diagnosis": predicted_diagnosis,
            "confidence": confidence,
            "top_k_diagnoses": top_k,
            "input_summary": {
                "age": request.age,
                "sex": request.sex,
                "n_evidences": len(request.evidences),
                "initial_evidence": request.initial_evidence,
            },
            "interpreted_evidences": self.metadata_service.interpret_evidences(request.evidences),
            "disclaimer": DISCLAIMER,
        }
        if include_distribution:
            payload["class_distribution"] = class_distribution
        return payload
