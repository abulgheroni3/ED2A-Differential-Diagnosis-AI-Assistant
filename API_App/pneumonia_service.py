"""Chest X-ray pneumonia service.

This module is completely independent from the DDXPlus ``ModelService``: it owns
its own artifacts, its own configuration contract and its own failure modes. A
missing or malformed artifact only makes this service unavailable, it never
prevents FastAPI from starting.

Inference runs on ONNX Runtime. PyTorch and torchvision are training-only and
are never imported here.
"""

from __future__ import annotations

import io
import json
import logging
import math
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import Settings


logger = logging.getLogger(__name__)

MODEL_NAME = "advanced_pneumonia"
EXECUTION_PROVIDER = "CPUExecutionProvider"

SUPPORTED_UPLOAD_FORMATS: frozenset[str] = frozenset({"JPEG", "PNG", "WEBP"})
SUPPORTED_FORMAT_LABEL = "JPEG, PNG, WebP"

#: Hard ceiling on decoded pixels, well below Pillow's own bomb threshold.
MAX_DECODED_PIXELS = 64_000_000

FINDING_POSITIVE = "Pneumonia-compatible pulmonary opacity detected"
FINDING_NEGATIVE = "No pneumonia-compatible pulmonary opacity detected"
FINDING_INCONCLUSIVE = "Radiographic assessment inconclusive"

CALIBRATION_NOT_VALIDATED = "not_validated"

IMAGING_DISCLAIMER = (
    "The radiographic model identifies a pulmonary opacity compatible with pneumonia. "
    "It does not establish a complete clinical diagnosis."
)


class PneumoniaServiceUnavailableError(RuntimeError):
    """The ONNX model or its configuration could not be loaded or validated."""


class ImageValidationError(ValueError):
    """The upload is missing, unreadable or violates the deployment contract.

    The message is safe to show in the browser: it never contains filesystem
    paths, stack traces or image content.
    """


class PneumoniaInferenceError(RuntimeError):
    """ONNX Runtime returned an unusable result."""


class InferenceSessionLike(Protocol):
    """Minimal ONNX Runtime surface used by this service."""

    def get_inputs(self) -> Sequence[Any]: ...

    def get_outputs(self) -> Sequence[Any]: ...

    def run(self, output_names: Any, input_feed: dict[str, Any]) -> Sequence[Any]: ...


SessionFactory = Callable[[Path], InferenceSessionLike]


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PneumoniaModelConfig:
    """Validated deployment contract read from ``advanced_pneumonia_config.json``."""

    input_name: str
    output_name: str
    image_height: int
    image_width: int
    channels: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    threshold: float
    class_mapping: dict[int, str]
    positive_class_index: int
    calibration_status: str
    supported_formats: frozenset[str]

    def label_for(self, class_index: int) -> str:
        return self.class_mapping.get(class_index, f"class_index_{class_index}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PneumoniaServiceUnavailableError(message)


def _as_float_triplet(values: Any, field: str) -> tuple[float, float, float]:
    _require(
        isinstance(values, (list, tuple)) and len(values) == 3,
        f"Configuration field '{field}' must contain three values.",
    )
    numbers: list[float] = []
    for value in values:
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"Configuration field '{field}' must contain finite numbers.",
        )
        numbers.append(float(value))
    return (numbers[0], numbers[1], numbers[2])


def _resolve_calibration_status(config: dict[str, Any]) -> str:
    """Return ``not_validated`` unless the config carries validated calibration data."""
    calibration = config.get("calibration")
    if not isinstance(calibration, dict):
        return CALIBRATION_NOT_VALIDATED
    if calibration.get("validated") is not True:
        return CALIBRATION_NOT_VALIDATED
    method = calibration.get("method")
    if not isinstance(method, str) or not method.strip():
        return CALIBRATION_NOT_VALIDATED
    return f"validated:{method.strip()}"


def parse_model_config(config: dict[str, Any]) -> PneumoniaModelConfig:
    """Validate the required configuration fields and freeze them into a contract."""
    _require(isinstance(config, dict), "The pneumonia configuration must be a JSON object.")

    onnx_section = config.get("onnx")
    input_section = config.get("input")
    output_section = config.get("output")
    _require(isinstance(onnx_section, dict), "Configuration section 'onnx' is missing.")
    _require(isinstance(input_section, dict), "Configuration section 'input' is missing.")
    _require(isinstance(output_section, dict), "Configuration section 'output' is missing.")

    input_name = onnx_section.get("input_name")
    output_name = onnx_section.get("output_name")
    _require(
        isinstance(input_name, str) and bool(input_name.strip()),
        "Configuration field 'onnx.input_name' is missing.",
    )
    _require(
        isinstance(output_name, str) and bool(output_name.strip()),
        "Configuration field 'onnx.output_name' is missing.",
    )

    input_size = input_section.get("input_size")
    _require(
        isinstance(input_size, (list, tuple)) and len(input_size) == 2,
        "Configuration field 'input.input_size' must contain [height, width].",
    )
    height, width = input_size
    _require(
        isinstance(height, int) and isinstance(width, int) and not isinstance(height, bool)
        and not isinstance(width, bool) and height > 0 and width > 0,
        "Configuration field 'input.input_size' must contain positive integers.",
    )

    channels = input_section.get("channels", 3)
    _require(
        isinstance(channels, int) and not isinstance(channels, bool) and channels == 3,
        "Only three-channel input is supported by this service.",
    )

    normalization = input_section.get("normalization")
    _require(isinstance(normalization, dict), "Configuration section 'input.normalization' is missing.")
    mean = _as_float_triplet(normalization.get("mean"), "input.normalization.mean")
    std = _as_float_triplet(normalization.get("std"), "input.normalization.std")
    _require(all(value != 0.0 for value in std), "Normalization standard deviations must be non-zero.")

    _require(
        output_section.get("is_logit") is True,
        "This service only supports models that emit a raw logit.",
    )
    activation = output_section.get("activation_required")
    _require(
        isinstance(activation, str) and activation.strip().lower() == "sigmoid",
        "Configuration field 'output.activation_required' must be 'sigmoid'.",
    )

    threshold = output_section.get("probability_threshold")
    _require(
        isinstance(threshold, (int, float)) and not isinstance(threshold, bool),
        "Configuration field 'output.probability_threshold' is missing.",
    )
    threshold = float(threshold)
    _require(
        math.isfinite(threshold) and 0.0 < threshold < 1.0,
        "Configuration field 'output.probability_threshold' must be strictly between 0 and 1.",
    )

    raw_mapping = config.get("class_mapping")
    _require(isinstance(raw_mapping, dict) and bool(raw_mapping), "Configuration field 'class_mapping' is missing.")
    class_mapping: dict[int, str] = {}
    for key, value in raw_mapping.items():
        try:
            index = int(key)
        except (TypeError, ValueError) as exc:
            raise PneumoniaServiceUnavailableError(
                "Configuration field 'class_mapping' must use integer class indices."
            ) from exc
        _require(
            isinstance(value, str) and bool(value.strip()),
            "Configuration field 'class_mapping' must map indices to non-empty labels.",
        )
        class_mapping[index] = value.strip()
    _require(
        {0, 1}.issubset(class_mapping.keys()),
        "Configuration field 'class_mapping' must define class indices 0 and 1.",
    )

    positive_index = config.get("positive_class_index")
    _require(
        isinstance(positive_index, int) and not isinstance(positive_index, bool)
        and positive_index in class_mapping,
        "Configuration field 'positive_class_index' must reference a mapped class.",
    )

    declared_formats = config.get("supported_upload_formats")
    if isinstance(declared_formats, (list, tuple)) and declared_formats:
        formats = frozenset(
            str(item).strip().upper() for item in declared_formats if str(item).strip()
        )
        supported = formats & SUPPORTED_UPLOAD_FORMATS
        _require(bool(supported), "No supported upload format is declared in the configuration.")
    else:
        supported = SUPPORTED_UPLOAD_FORMATS

    return PneumoniaModelConfig(
        input_name=input_name.strip(),
        output_name=output_name.strip(),
        image_height=int(height),
        image_width=int(width),
        channels=int(channels),
        mean=mean,
        std=std,
        threshold=threshold,
        class_mapping=class_mapping,
        positive_class_index=int(positive_index),
        calibration_status=_resolve_calibration_status(config),
        supported_formats=supported,
    )


# --------------------------------------------------------------------- preprocessing


def stable_sigmoid(value: float) -> float:
    """Numerically stable logistic function for a single raw logit."""
    if not math.isfinite(value):
        raise PneumoniaInferenceError("The model returned a non-finite logit.")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _open_validated_image(image_bytes: bytes, config: PneumoniaModelConfig) -> Image.Image:
    """Decode the upload with Pillow and reject anything outside the contract."""
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise ImageValidationError("The uploaded chest X-ray could not be read.")
    if not image_bytes:
        raise ImageValidationError("The uploaded chest X-ray is empty.")

    try:
        with warnings.catch_warnings():
            # Turn Pillow's decompression-bomb warning into a hard failure.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(bytes(image_bytes)))
            image_format = (image.format or "").upper()
            image.load()
    except Image.DecompressionBombError as exc:
        raise ImageValidationError("The uploaded image is too large to be processed safely.") from exc
    except Image.DecompressionBombWarning as exc:
        raise ImageValidationError("The uploaded image is too large to be processed safely.") from exc
    except UnidentifiedImageError as exc:
        raise ImageValidationError(
            f"The uploaded file is not a readable image. Accepted formats: {SUPPORTED_FORMAT_LABEL}."
        ) from exc
    except ImageValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - corrupted uploads must fail in a controlled way
        raise ImageValidationError("The uploaded chest X-ray could not be decoded.") from exc

    if image_format not in config.supported_formats:
        raise ImageValidationError(
            f"Unsupported image format. Accepted formats: {SUPPORTED_FORMAT_LABEL}."
        )

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageValidationError("The uploaded image has invalid dimensions.")
    if width * height > MAX_DECODED_PIXELS:
        raise ImageValidationError("The uploaded image is too large to be processed safely.")

    return image


def preprocess_image(image_bytes: bytes, config: PneumoniaModelConfig) -> np.ndarray:
    """Return the deterministic ``[1, 3, H, W]`` float32 tensor expected by the graph.

    The steps mirror the notebook's deployment transform exactly: EXIF
    normalisation, grayscale conversion, three-channel replication, bilinear
    resize without cropping, ``[0, 1]`` scaling, ImageNet normalisation and a
    HWC to CHW transposition.
    """
    image = _open_validated_image(image_bytes, config)

    try:
        oriented = ImageOps.exif_transpose(image) or image
        grayscale = oriented.convert("L")
        rgb_image = grayscale.convert("RGB")
        resized = rgb_image.resize(
            (config.image_width, config.image_height),
            resample=Image.BILINEAR,
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
    except ImageValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - keep decoding failures user-safe
        raise ImageValidationError("The uploaded chest X-ray could not be processed.") from exc

    if pixels.shape != (config.image_height, config.image_width, config.channels):
        raise ImageValidationError("The uploaded image could not be converted to the expected shape.")

    mean = np.asarray(config.mean, dtype=np.float32)
    std = np.asarray(config.std, dtype=np.float32)
    normalized = (pixels - mean) / std
    tensor = np.ascontiguousarray(
        np.transpose(normalized, (2, 0, 1))[np.newaxis, ...],
        dtype=np.float32,
    )

    if not np.isfinite(tensor).all():
        raise ImageValidationError("The uploaded image produced non-finite pixel values.")
    return tensor


# ------------------------------------------------------------------------ assessment


@dataclass(frozen=True)
class PneumoniaAssessment:
    """Physician-facing result of one chest X-ray inference."""

    status: str
    finding: str
    raw_logit: float
    image_model_score: float
    threshold: float
    class_index: int
    class_label: str
    model_name: str
    calibration_status: str
    disclaimer: str = IMAGING_DISCLAIMER

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_session_factory(model_path: Path) -> InferenceSessionLike:
    import onnxruntime as ort  # imported lazily so the API starts without the model

    return ort.InferenceSession(str(model_path), providers=[EXECUTION_PROVIDER])


def _dimension_matches(dimension: Any, expected: int) -> bool:
    """Accept a fixed matching dimension or a dynamic/symbolic one."""
    if isinstance(dimension, int):
        return dimension == expected or dimension < 0
    return dimension is None or isinstance(dimension, str)


def _is_dynamic(dimension: Any) -> bool:
    if isinstance(dimension, int):
        return dimension < 0
    return dimension is None or isinstance(dimension, str)


class PneumoniaService:
    """Load and serve the RSNA chest X-ray model exported to ONNX."""

    def __init__(
        self,
        settings: Settings,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.settings = settings
        self._session_factory: SessionFactory = session_factory or _default_session_factory
        self._session: InferenceSessionLike | None = None
        self._config: PneumoniaModelConfig | None = None
        self._unavailable_reason: str | None = None
        self._last_attempt_signature: tuple[Any, ...] | None = None
        self.model_name = MODEL_NAME
        self.load()

    # ------------------------------------------------------------------ loading

    def _artifact_signature(self) -> tuple[Any, ...]:
        """Cheap fingerprint of the artifact set, used to avoid reload storms."""
        parts: list[Any] = []
        for path in (self.settings.pneumonia_model_path, self.settings.pneumonia_config_path):
            try:
                stat = path.stat()
                parts.append((path.name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                parts.append((path.name, None, None))
        return tuple(parts)

    def refresh_if_changed(self) -> None:
        """Re-attempt loading only when the artifacts changed since the last try.

        Reporting a real readiness state must not cost a 90 MB session load on
        every health check.
        """
        if self._artifact_signature() != self._last_attempt_signature:
            self.load()

    def load(self) -> None:
        """Load configuration and ONNX session; record a safe reason on failure."""
        self._last_attempt_signature = self._artifact_signature()
        self._session = None
        self._config = None
        self._unavailable_reason = None
        try:
            config = self._load_config()
            session = self._load_session()
            self._validate_session_contract(session, config)
        except PneumoniaServiceUnavailableError as exc:
            self._unavailable_reason = str(exc)
            logger.warning("Advanced Pneumonia Model unavailable: %s", exc)
            return
        except Exception as exc:  # noqa: BLE001 - never let artifact issues break startup
            self._unavailable_reason = "The Advanced Pneumonia Model could not be initialised."
            logger.warning("Advanced Pneumonia Model failed to load: %s", exc)
            return
        self._config = config
        self._session = session

    def _load_config(self) -> PneumoniaModelConfig:
        path = self.settings.pneumonia_config_path
        if not path.exists():
            raise PneumoniaServiceUnavailableError(
                f"Missing configuration file: {path.name}."
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                raw_config = json.load(handle)
        except json.JSONDecodeError as exc:
            raise PneumoniaServiceUnavailableError(
                f"Configuration file {path.name} is not valid JSON."
            ) from exc
        except OSError as exc:
            raise PneumoniaServiceUnavailableError(
                f"Configuration file {path.name} could not be read."
            ) from exc
        return parse_model_config(raw_config)

    def _load_session(self) -> InferenceSessionLike:
        path = self.settings.pneumonia_model_path
        if not path.exists():
            raise PneumoniaServiceUnavailableError(f"Missing ONNX model file: {path.name}.")
        try:
            return self._session_factory(path)
        except PneumoniaServiceUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises provider-specific errors
            logger.warning("ONNX Runtime could not open the pneumonia model: %s", exc)
            raise PneumoniaServiceUnavailableError(
                f"ONNX Runtime could not load {path.name}."
            ) from exc

    @staticmethod
    def _validate_session_contract(
        session: InferenceSessionLike, config: PneumoniaModelConfig
    ) -> None:
        """Check that the graph really matches the declared deployment contract."""
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
        _require(len(inputs) == 1, "The ONNX graph must expose exactly one input.")
        _require(len(outputs) == 1, "The ONNX graph must expose exactly one output.")

        graph_input, graph_output = inputs[0], outputs[0]
        _require(
            getattr(graph_input, "name", None) == config.input_name,
            "The ONNX input name does not match the configuration.",
        )
        _require(
            getattr(graph_output, "name", None) == config.output_name,
            "The ONNX output name does not match the configuration.",
        )

        input_shape = list(getattr(graph_input, "shape", []) or [])
        _require(len(input_shape) == 4, "The ONNX input must have four dimensions.")
        _require(_is_dynamic(input_shape[0]), "The ONNX input must accept a dynamic batch axis.")
        _require(
            _dimension_matches(input_shape[1], config.channels),
            "The ONNX input must accept three channels.",
        )
        _require(
            _dimension_matches(input_shape[2], config.image_height)
            and _dimension_matches(input_shape[3], config.image_width),
            "The ONNX input size does not match the configured image size.",
        )

        output_shape = list(getattr(graph_output, "shape", []) or [])
        _require(
            len(output_shape) in {1, 2},
            "The ONNX output must expose one raw logit per image.",
        )
        _require(_is_dynamic(output_shape[0]), "The ONNX output must keep a dynamic batch axis.")
        if len(output_shape) == 2:
            _require(
                _dimension_matches(output_shape[1], 1),
                "The ONNX output must expose one raw logit per image.",
            )

    # ------------------------------------------------------------------ readiness

    @property
    def ready(self) -> bool:
        return self._session is not None and self._config is not None

    @property
    def unavailable_reason(self) -> str | None:
        if self.ready:
            return None
        return self._unavailable_reason or "The Advanced Pneumonia Model is not available."

    @property
    def config(self) -> PneumoniaModelConfig | None:
        return self._config

    @property
    def threshold(self) -> float | None:
        return self._config.threshold if self._config else None

    @property
    def model_path(self) -> str:
        return self.settings.display_path(self.settings.pneumonia_model_path)

    @property
    def config_path(self) -> str:
        return self.settings.display_path(self.settings.pneumonia_config_path)

    def model_timestamp(self) -> dict[str, str] | None:
        """Newest modification time among the loaded pneumonia artifacts.

        This is deliberately separate from the DDXPlus model freshness.
        """
        if not self.ready:
            return None
        newest_mtime: float | None = None
        newest_name: str | None = None
        for path in (self.settings.pneumonia_model_path, self.settings.pneumonia_config_path):
            try:
                mtime = path.stat().st_mtime
            except OSError:
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

    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ready": self.ready,
            "reason": self.unavailable_reason
            or "ONNX model and configuration are loaded and validated.",
            "execution_provider": EXECUTION_PROVIDER,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "threshold": self.threshold,
            "calibration_status": self._config.calibration_status
            if self._config
            else CALIBRATION_NOT_VALIDATED,
            "model_timestamp": self.model_timestamp(),
        }

    # ------------------------------------------------------------------ inference

    def predict(self, image_bytes: bytes) -> PneumoniaAssessment:
        """Run one chest X-ray through the ONNX graph.

        Raises:
            PneumoniaServiceUnavailableError: The model is not loaded.
            ImageValidationError: The upload violates the deployment contract.
            PneumoniaInferenceError: ONNX Runtime returned an unusable result.
        """
        if self._session is None or self._config is None:
            raise PneumoniaServiceUnavailableError(
                self.unavailable_reason or "The Advanced Pneumonia Model is not available."
            )
        config = self._config

        if len(image_bytes) > self.settings.max_image_upload_bytes:
            raise ImageValidationError(
                "The uploaded chest X-ray exceeds the 10 MB limit."
            )

        tensor = preprocess_image(image_bytes, config)
        raw_logit = self._run_session(tensor, config)
        return self._build_assessment(raw_logit, config)

    def _run_session(self, tensor: np.ndarray, config: PneumoniaModelConfig) -> float:
        assert self._session is not None  # guarded by predict()
        try:
            outputs = self._session.run([config.output_name], {config.input_name: tensor})
        except Exception as exc:  # noqa: BLE001 - runtime errors must not leak internals
            logger.warning("ONNX Runtime inference failed: %s", exc)
            raise PneumoniaInferenceError("The chest X-ray model could not process the image.") from exc

        if not outputs:
            raise PneumoniaInferenceError("The chest X-ray model returned no output.")
        array = np.asarray(outputs[0], dtype=np.float64)
        if array.size != 1:
            raise PneumoniaInferenceError(
                "The chest X-ray model returned an unexpected output shape."
            )
        raw_logit = float(array.reshape(-1)[0])
        if not math.isfinite(raw_logit):
            raise PneumoniaInferenceError("The chest X-ray model returned a non-finite logit.")
        return raw_logit

    @staticmethod
    def _build_assessment(raw_logit: float, config: PneumoniaModelConfig) -> PneumoniaAssessment:
        image_model_score = stable_sigmoid(raw_logit)
        is_positive = image_model_score >= config.threshold
        class_index = config.positive_class_index if is_positive else int(not config.positive_class_index)
        return PneumoniaAssessment(
            status="analysed",
            finding=FINDING_POSITIVE if is_positive else FINDING_NEGATIVE,
            raw_logit=raw_logit,
            image_model_score=image_model_score,
            threshold=config.threshold,
            class_index=class_index,
            class_label=config.label_for(class_index),
            model_name=MODEL_NAME,
            calibration_status=config.calibration_status,
        )
