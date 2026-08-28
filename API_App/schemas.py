"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=130)
    sex: str
    evidences: list[str]
    initial_evidence: str | None = None

    @validator("sex")
    def validate_sex(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"M", "F"}:
            raise ValueError("sex must be 'M' or 'F'.")
        return normalized

    @validator("evidences")
    def validate_evidences(cls, value: list[str]) -> list[str]:
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("evidences must contain at least one evidence token.")
        return cleaned

    @validator("initial_evidence")
    def clean_initial_evidence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class TopKPredictionRequest(PredictionRequest):
    k: int = Field(5, ge=1, le=20)


class TopKDiagnosis(BaseModel):
    diagnosis: str
    probability: float | None = None
    icd10_id: str | None = None
    severity: int | None = None


class InputSummary(BaseModel):
    age: int
    sex: str
    n_evidences: int
    initial_evidence: str | None = None


class InterpretedEvidence(BaseModel):
    raw: str
    evidence_id: str
    value: int | str
    question_en: str | None = None
    data_type: str | None = None
    is_antecedent: bool | None = None
    value_meaning_en: str | None = None


class Urgency(BaseModel):
    level: str
    dataset_severity: int | None = None
    source: str
    message: str


class PredictionResponse(BaseModel):
    predicted_diagnosis: str
    confidence: float | None = None
    top_k_diagnoses: list[TopKDiagnosis]
    input_summary: InputSummary
    interpreted_evidences: list[InterpretedEvidence]
    disclaimer: str
    urgency: Urgency | None = None


class MessageResponse(BaseModel):
    message: str
    details: dict[str, Any] | None = None


class ClinicalEvidenceArticle(BaseModel):
    pmid: str
    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    publication_date: str | None = None
    publication_type: str | None = None
    abstract_excerpt: str | None = None
    url: str


class ClinicalEvidenceResponse(BaseModel):
    query: str
    total_matches: int
    articles: list[ClinicalEvidenceArticle]


# ------------------------------------------------------- optional multimodal workflow

ASSESSMENT_MODE_CLINICAL_ONLY = "clinical_only"
ASSESSMENT_MODE_CLINICAL_PLUS_XRAY = "clinical_plus_xray"


class ClinicalRankedDiagnosis(TopKDiagnosis):
    """A clinical Top-k row, extended with its explicit rank and score alias."""

    rank: int
    model_score: float | None = None


class IntegratedRankedDiagnosis(BaseModel):
    """One entry of the experimental image-assisted re-ranking."""

    rank: int
    diagnosis: str
    integrated_model_score: float | None = None
    icd10_id: str | None = None
    severity: int | None = None


class ImagingAssessment(BaseModel):
    """Chest X-ray outcome, always described as a radiographic finding."""

    status: str
    finding: str
    model_name: str
    calibration_status: str
    raw_logit: float | None = None
    image_model_score: float | None = None
    threshold: float | None = None
    class_index: int | None = None
    class_label: str | None = None
    detail: str | None = None
    disclaimer: str | None = None


class FusionSummary(BaseModel):
    """Transparent description of whether and why the ranking changed."""

    applied: bool
    method: str
    interpretation: str
    target_diagnosis: str
    alpha: float
    uncertainty_margin: float
    reason: str
    limitation: str
    image_signal: float | None = None
    rank_before: int | None = None
    rank_after: int | None = None
    clinical_score_before: float | None = None
    integrated_score_after: float | None = None


class IntegratedPredictionResponse(BaseModel):
    """Stable schema returned by ``POST /predict-integrated``."""

    assessment_mode: str
    requested_k: int
    predicted_diagnosis: str
    confidence: float | None = None
    clinical_diagnoses: list[ClinicalRankedDiagnosis]
    imaging_assessment: ImagingAssessment | None = None
    fusion: FusionSummary
    integrated_diagnoses: list[IntegratedRankedDiagnosis]
    input_summary: InputSummary
    interpreted_evidences: list[InterpretedEvidence]
    urgency: Urgency | None = None
    disclaimer: str
    warnings: list[str] = Field(default_factory=list)
