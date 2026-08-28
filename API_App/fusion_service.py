"""Experimental decision-level late fusion between DDXPlus and the RSNA model.

The service is deterministic, has no I/O and never touches either trained model.
It receives the complete DDXPlus class distribution and one chest X-ray score,
and returns a bounded image-assisted re-ranking.

The result is **not** a calibrated combined diagnostic probability: DDXPlus and
RSNA are separate, unpaired datasets, and the fusion weight is a demonstration
parameter rather than a clinically validated value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


FUSION_METHOD = "experimental_bounded_late_fusion"
LIMITATION = (
    "Experimental image-assisted re-ranking; not a calibrated combined diagnostic probability."
)
EPSILON = 1e-12

INTERPRETATION_SUPPORTS = "supports_clinical_pneumonia_hypothesis"
INTERPRETATION_DOES_NOT_SUPPORT = "does_not_support_clinical_pneumonia_hypothesis"
INTERPRETATION_INCONCLUSIVE = "inconclusive"
INTERPRETATION_NOT_APPLIED = "not_applied"

REASON_NO_IMAGE = "No chest X-ray was supplied, so only the clinical model was used."
REASON_SUPPORTS = "The radiographic model score supports the clinical pneumonia hypothesis."
REASON_DOES_NOT_SUPPORT = (
    "The radiographic model score does not support the clinical pneumonia hypothesis."
)

ALPHA_MIN, ALPHA_MAX = 0.0, 2.0
MARGIN_MIN, MARGIN_MAX = 0.0, 0.5


class InvalidFusionParametersError(ValueError):
    """The configured fusion weight or uncertainty margin is out of range."""


class InvalidFusionInputError(ValueError):
    """The supplied class distribution or image score cannot be fused."""


@dataclass(frozen=True)
class RankedDiagnosis:
    """One entry of a ranked list, ordered from the most to the least likely."""

    rank: int
    diagnosis: str
    score: float


@dataclass(frozen=True)
class FusionOutcome:
    """Complete, transparent description of one fusion decision."""

    applied: bool
    method: str
    interpretation: str
    target_diagnosis: str
    alpha: float
    uncertainty_margin: float
    reason: str
    limitation: str
    integrated_diagnoses: list[RankedDiagnosis]
    image_signal: float | None = None
    rank_before: int | None = None
    rank_after: int | None = None
    clinical_score_before: float | None = None
    integrated_score_after: float | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the decision without the ranking, which has its own schema."""
        return {
            "applied": self.applied,
            "method": self.method,
            "interpretation": self.interpretation,
            "target_diagnosis": self.target_diagnosis,
            "alpha": self.alpha,
            "uncertainty_margin": self.uncertainty_margin,
            "image_signal": self.image_signal,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "clinical_score_before": self.clinical_score_before,
            "integrated_score_after": self.integrated_score_after,
            "reason": self.reason,
            "limitation": self.limitation,
        }


def normalize_label(label: str) -> str:
    """Case-insensitive, whitespace-insensitive label key used for class matching."""
    return " ".join(str(label).split()).casefold()


def _stable_softmax(log_scores: np.ndarray) -> np.ndarray:
    shifted = log_scores - float(np.max(log_scores))
    exponentials = np.exp(shifted)
    total = float(np.sum(exponentials))
    if not math.isfinite(total) or total <= 0.0:
        raise InvalidFusionInputError("The fused distribution could not be normalised.")
    return exponentials / total


def _descending_order(scores: np.ndarray) -> np.ndarray:
    """Ranking rule shared by the clinical and the integrated distribution."""
    return np.argsort(scores)[::-1]


class FusionService:
    """Bounded, decision-level late fusion targeting a single DDXPlus class."""

    def __init__(
        self,
        alpha: float = 0.75,
        uncertainty_margin: float = 0.10,
        target_label: str = "Pneumonia",
    ) -> None:
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(float(alpha)):
            raise InvalidFusionParametersError("The fusion weight alpha must be a finite number.")
        if not ALPHA_MIN <= float(alpha) <= ALPHA_MAX:
            raise InvalidFusionParametersError(
                f"The fusion weight alpha must satisfy {ALPHA_MIN} <= alpha <= {ALPHA_MAX}."
            )
        if (
            not isinstance(uncertainty_margin, (int, float))
            or isinstance(uncertainty_margin, bool)
            or not math.isfinite(float(uncertainty_margin))
        ):
            raise InvalidFusionParametersError("The uncertainty margin must be a finite number.")
        if not MARGIN_MIN <= float(uncertainty_margin) < MARGIN_MAX:
            raise InvalidFusionParametersError(
                f"The uncertainty margin must satisfy {MARGIN_MIN} <= margin < {MARGIN_MAX}."
            )
        target = str(target_label).strip()
        if not target:
            raise InvalidFusionParametersError("The target diagnosis label must not be empty.")

        self.alpha = float(alpha)
        self.uncertainty_margin = float(uncertainty_margin)
        self.target_label = target

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _validate_distribution(
        labels: Sequence[str], probabilities: Sequence[float]
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return the labels, the validated scores and their normalised copy.

        The validated scores are reported verbatim whenever fusion does not
        apply, so an untouched clinical ranking stays bit-identical to the one
        produced by the differential model.
        """
        clean_labels = [str(label) for label in labels]
        if not clean_labels:
            raise InvalidFusionInputError("The clinical distribution is empty.")
        array = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        if array.size != len(clean_labels):
            raise InvalidFusionInputError(
                "The clinical distribution and its labels have different lengths."
            )
        if not np.isfinite(array).all():
            raise InvalidFusionInputError("The clinical distribution contains non-finite scores.")

        clipped = np.clip(array, 0.0, None)
        total = float(np.sum(clipped))
        if not math.isfinite(total) or total <= 0.0:
            raise InvalidFusionInputError("The clinical distribution sums to zero.")
        normalized = clipped if abs(total - 1.0) <= 1e-9 else clipped / total
        return clean_labels, clipped, normalized

    def _resolve_target_index(self, labels: Sequence[str]) -> tuple[int | None, str | None]:
        """Locate the DDXPlus pneumonia class with a normalised comparison."""
        wanted = normalize_label(self.target_label)
        matches = [index for index, label in enumerate(labels) if normalize_label(label) == wanted]
        if not matches:
            return None, (
                f"The '{self.target_label}' class is not present in the clinical model, "
                "so the radiographic result was not fused."
            )
        if len(matches) > 1:
            return None, (
                f"The '{self.target_label}' class is ambiguous in the clinical model, "
                "so the radiographic result was not fused."
            )
        return matches[0], None

    @staticmethod
    def _rank_of(order: np.ndarray, index: int) -> int:
        return int(np.flatnonzero(order == index)[0]) + 1

    @staticmethod
    def _ranked_list(
        labels: Sequence[str], scores: np.ndarray, order: np.ndarray, k: int
    ) -> list[RankedDiagnosis]:
        limit = max(1, int(k))
        selected = order[: min(limit, len(order))]
        return [
            RankedDiagnosis(rank=position + 1, diagnosis=labels[int(index)], score=float(scores[int(index)]))
            for position, index in enumerate(selected)
        ]

    def _image_signal(self, image_score: float, threshold: float) -> float:
        """Bounded, threshold-relative signal in ``[-1, 1]``."""
        if image_score >= threshold:
            signal = (image_score - threshold) / (1.0 - threshold)
        else:
            signal = (image_score - threshold) / threshold
        return float(min(1.0, max(-1.0, signal)))

    # ------------------------------------------------------------------- results

    def clinical_only(
        self,
        labels: Sequence[str],
        probabilities: Sequence[float],
        k: int,
        reason: str = REASON_NO_IMAGE,
        interpretation: str = INTERPRETATION_NOT_APPLIED,
        warnings: Sequence[str] | None = None,
    ) -> FusionOutcome:
        """Return the untouched clinical ranking as the integrated ranking."""
        clean_labels, clinical, _ = self._validate_distribution(labels, probabilities)
        order = _descending_order(clinical)
        target_index, _ = self._resolve_target_index(clean_labels)
        rank_before = self._rank_of(order, target_index) if target_index is not None else None
        clinical_score = float(clinical[target_index]) if target_index is not None else None
        return FusionOutcome(
            applied=False,
            method=FUSION_METHOD,
            interpretation=interpretation,
            target_diagnosis=self.target_label,
            alpha=self.alpha,
            uncertainty_margin=self.uncertainty_margin,
            reason=reason,
            limitation=LIMITATION,
            integrated_diagnoses=self._ranked_list(clean_labels, clinical, order, k),
            image_signal=None,
            rank_before=rank_before,
            rank_after=rank_before,
            clinical_score_before=clinical_score,
            integrated_score_after=clinical_score,
            warnings=list(warnings or []),
        )

    def fuse(
        self,
        labels: Sequence[str],
        probabilities: Sequence[float],
        image_model_score: float,
        threshold: float,
        k: int,
    ) -> FusionOutcome:
        """Apply the bounded late-fusion algorithm to the complete distribution.

        Args:
            labels: Every DDXPlus class label, before Top-k truncation.
            probabilities: The aligned complete class distribution.
            image_model_score: ``sigmoid(logit)`` from the chest X-ray model.
            threshold: The pneumonia decision threshold from its configuration.
            k: The Top-k value requested by the physician.
        """
        clean_labels, clinical, normalized = self._validate_distribution(labels, probabilities)

        if (
            not isinstance(image_model_score, (int, float))
            or isinstance(image_model_score, bool)
            or not math.isfinite(float(image_model_score))
            or not 0.0 <= float(image_model_score) <= 1.0
        ):
            raise InvalidFusionInputError("The image model score must be a finite value in [0, 1].")
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or not 0.0 < float(threshold) < 1.0
        ):
            raise InvalidFusionInputError("The decision threshold must be strictly between 0 and 1.")

        image_score = float(image_model_score)
        decision_threshold = float(threshold)

        target_index, missing_reason = self._resolve_target_index(clean_labels)
        if target_index is None:
            assert missing_reason is not None
            return self.clinical_only(
                clean_labels,
                clinical,
                k,
                reason=missing_reason,
                warnings=[missing_reason],
            )

        clinical_order = _descending_order(clinical)
        rank_before = self._rank_of(clinical_order, target_index)
        clinical_score = float(clinical[target_index])

        if abs(image_score - decision_threshold) <= self.uncertainty_margin:
            reason = (
                "The image model score falls inside the configured uncertainty interval, "
                "so the clinical ranking was left unchanged."
            )
            return FusionOutcome(
                applied=False,
                method=FUSION_METHOD,
                interpretation=INTERPRETATION_INCONCLUSIVE,
                target_diagnosis=self.target_label,
                alpha=self.alpha,
                uncertainty_margin=self.uncertainty_margin,
                reason=reason,
                limitation=LIMITATION,
                integrated_diagnoses=self._ranked_list(clean_labels, clinical, clinical_order, k),
                image_signal=self._image_signal(image_score, decision_threshold),
                rank_before=rank_before,
                rank_after=rank_before,
                clinical_score_before=clinical_score,
                integrated_score_after=clinical_score,
                warnings=[],
            )

        image_signal = self._image_signal(image_score, decision_threshold)
        log_scores = np.log(np.maximum(normalized, EPSILON))
        log_scores[target_index] += self.alpha * image_signal
        integrated = _stable_softmax(log_scores)

        if not np.isfinite(integrated).all() or float(np.min(integrated)) < 0.0:
            raise InvalidFusionInputError("The fused distribution is not a valid score vector.")

        integrated_order = _descending_order(integrated)
        supports = image_score >= decision_threshold
        return FusionOutcome(
            applied=True,
            method=FUSION_METHOD,
            interpretation=INTERPRETATION_SUPPORTS if supports else INTERPRETATION_DOES_NOT_SUPPORT,
            target_diagnosis=self.target_label,
            alpha=self.alpha,
            uncertainty_margin=self.uncertainty_margin,
            reason=REASON_SUPPORTS if supports else REASON_DOES_NOT_SUPPORT,
            limitation=LIMITATION,
            integrated_diagnoses=self._ranked_list(clean_labels, integrated, integrated_order, k),
            image_signal=image_signal,
            rank_before=rank_before,
            rank_after=self._rank_of(integrated_order, target_index),
            clinical_score_before=clinical_score,
            integrated_score_after=float(integrated[target_index]),
            warnings=[],
        )
