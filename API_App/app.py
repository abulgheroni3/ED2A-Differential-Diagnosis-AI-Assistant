"""FastAPI routes for the DDXPlus differential diagnosis assistant."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from clinical_evidence_service import (
    ArticleType,
    ClinicalEvidenceTimeoutError,
    ClinicalEvidenceUpstreamError,
    InvalidSearchParametersError,
    YearsFilter,
    search_clinical_evidence,
)
from config import get_settings
from fusion_service import (
    FusionOutcome,
    FusionService,
    INTERPRETATION_DOES_NOT_SUPPORT,
    INTERPRETATION_INCONCLUSIVE,
    INTERPRETATION_NOT_APPLIED,
    INTERPRETATION_SUPPORTS,
    InvalidFusionInputError,
    InvalidFusionParametersError,
    LIMITATION as FUSION_LIMITATION,
    FUSION_METHOD,
    RankedDiagnosis,
)
from metadata_service import MetadataService
from model_service import (
    ModelNotReadyError,
    ModelService,
    PredictionError,
    PreprocessingError,
)
from pneumonia_service import (
    FINDING_INCONCLUSIVE,
    IMAGING_DISCLAIMER,
    CALIBRATION_NOT_VALIDATED,
    ImageValidationError,
    PneumoniaInferenceError,
    PneumoniaService,
    PneumoniaServiceUnavailableError,
)
from schemas import (
    ASSESSMENT_MODE_CLINICAL_ONLY,
    ASSESSMENT_MODE_CLINICAL_PLUS_XRAY,
    ClinicalEvidenceResponse,
    IntegratedPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    TopKPredictionRequest,
)
from utils import EvidenceParseError, parse_evidence_token, safe_parse_evidence_list


settings = get_settings()
metadata_service = MetadataService(settings)
model_service = ModelService(settings, metadata_service)

logger = logging.getLogger(__name__)

# The chest X-ray model and the fusion layer are optional: a missing artifact or
# an out-of-range setting only disables the multimodal path.
pneumonia_service = PneumoniaService(settings)

try:
    fusion_service: FusionService | None = FusionService(
        alpha=settings.pneumonia_fusion_alpha,
        uncertainty_margin=settings.pneumonia_fusion_uncertainty_margin,
        target_label=settings.pneumonia_ddx_label,
    )
    fusion_unavailable_reason: str | None = None
except InvalidFusionParametersError as exc:
    fusion_service = None
    fusion_unavailable_reason = str(exc)
    logger.warning("Image-assisted fusion is disabled: %s", exc)

app = FastAPI(
    title="Explainable Differential Diagnosis Assistant",
    description="Educational FastAPI application layer for DDXPlus telemedicine triage experiments.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=settings.api_dir / "static"), name="static")
app.mount("/imgs", StaticFiles(directory=settings.api_dir / "imgs"), name="imgs")

templates = Jinja2Templates(directory=settings.api_dir / "templates")

URGENCY_SOURCE = "DDXPlus condition severity"
URGENCY_MESSAGES = {
    "high": "This condition may require prompt medical evaluation.",
    "moderate": "Medical evaluation is recommended.",
    "low": "Consider medical evaluation, especially if symptoms persist or worsen.",
    "unavailable": "Urgency information is unavailable.",
}


def _metadata_unavailable(detail: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "message": detail,
            "load_errors": metadata_service.load_errors,
            "missing_files": metadata_service.missing_metadata_files(),
        },
    )


def _validate_request_evidences(request: PredictionRequest) -> None:
    for token in request.evidences:
        parse_evidence_token(token)
    if request.initial_evidence:
        parse_evidence_token(request.initial_evidence)

    invalid_ids = metadata_service.validate_evidences(request.evidences)
    if request.initial_evidence and metadata_service.evidences_loaded:
        initial_id = parse_evidence_token(request.initial_evidence)["evidence_id"]
        if initial_id not in metadata_service.evidences:
            invalid_ids.append(initial_id)

    if invalid_ids:
        unique_invalid = list(dict.fromkeys(invalid_ids))
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Unknown evidence ID found in request.",
                "invalid_evidence_ids": unique_invalid,
            },
        )


def _urgency_for_diagnosis(diagnosis: Any) -> dict[str, Any]:
    unavailable = {
        "level": "unavailable",
        "dataset_severity": None,
        "source": URGENCY_SOURCE,
        "message": URGENCY_MESSAGES["unavailable"],
    }
    try:
        if not isinstance(diagnosis, str) or not diagnosis.strip():
            logger.warning("Cannot resolve urgency: prediction returned no diagnosis.")
            return unavailable

        condition = metadata_service.get_condition(diagnosis.strip())
        if not condition:
            logger.warning("Cannot resolve urgency: diagnosis '%s' was not found in conditions metadata.", diagnosis)
            return unavailable

        severity = condition.get("severity")
        if isinstance(severity, bool) or not isinstance(severity, int) or not 1 <= severity <= 5:
            logger.warning("Cannot resolve urgency: invalid severity for diagnosis '%s'.", diagnosis)
            return unavailable

        if severity <= 2:
            level = "high"
        elif severity == 3:
            level = "moderate"
        else:
            level = "low"
        return {
            "level": level,
            "dataset_severity": severity,
            "source": URGENCY_SOURCE,
            "message": URGENCY_MESSAGES[level],
        }
    except Exception as exc:  # noqa: BLE001 - metadata failures must not break prediction responses
        logger.warning("Cannot resolve urgency from conditions metadata: %s", exc)
        return unavailable


def _prediction_payload(
    request: PredictionRequest, k: int, include_distribution: bool = False
) -> dict[str, Any]:
    try:
        _validate_request_evidences(request)
        payload = model_service.predict(request, k=k, include_distribution=include_distribution)
        payload["urgency"] = _urgency_for_diagnosis(payload.get("predicted_diagnosis"))
        return payload
    except EvidenceParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=503,
            detail={"message": exc.message, "missing_files": exc.missing_files},
        ) from exc
    except PreprocessingError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _health_payload() -> dict[str, Any]:
    differential_info = model_service.info()
    differential_ready = bool(differential_info.get("ready_for_prediction"))
    if differential_ready:
        differential_reason = "Model and preprocessor artifacts are loaded and ready."
    elif differential_info.get("load_errors"):
        differential_reason = "; ".join(differential_info["load_errors"])
    elif differential_info.get("missing_files"):
        differential_reason = (
            "Missing artifacts: " + ", ".join(differential_info["missing_files"])
        )
    else:
        differential_reason = "Differential Diagnosis Model is not ready."

    # The real readiness state: the ONNX graph and its configuration must both
    # load and pass contract validation, existing on disk is not enough.
    pneumonia_service.refresh_if_changed()
    pneumonia_ready = pneumonia_service.ready
    pneumonia_reason = (
        pneumonia_service.unavailable_reason
        or "ONNX model and configuration are loaded and validated."
    )

    model_status = {
        "differential_diagnosis": {
            "ready": differential_ready,
            "reason": differential_reason,
        },
        "advanced_pneumonia": {
            "ready": pneumonia_ready,
            "reason": pneumonia_reason,
        },
    }
    missing_files = (
        metadata_service.missing_metadata_files()
        + model_service.missing_artifacts()
    )
    return {
        "status": "ok",
        "metadata_loaded": metadata_service.evidences_loaded,
        "conditions_loaded": metadata_service.conditions_loaded,
        "model_loaded": differential_ready,
        "models": model_status,
        "missing_files": missing_files,
        "model_freshness": model_service.model_freshness(),
        "pneumonia_model_freshness": pneumonia_service.model_timestamp(),
        "fusion": {
            "available": fusion_service is not None,
            "method": FUSION_METHOD,
            "alpha": fusion_service.alpha if fusion_service else None,
            "uncertainty_margin": fusion_service.uncertainty_margin if fusion_service else None,
            "target_diagnosis": fusion_service.target_label if fusion_service else None,
            "reason": fusion_unavailable_reason,
            "limitation": FUSION_LIMITATION,
        },
        "data_raw_files_detected": settings.detect_data_raw_files(),
        "metadata_paths": {
            "evidences": settings.display_path(settings.evidences_json_path),
            "conditions": settings.display_path(settings.conditions_json_path),
        },
    }


def _template_context(
    request: Request,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    form_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request": request,
        "health": _health_payload(),
        "model_info": model_service.info(),
        "sample_evidences": metadata_service.list_evidences(limit=8)
        if metadata_service.evidences_loaded
        else [],
        "evidence_options_source": metadata_service.evidence_options_source,
        "evidence_options": (
            metadata_service.list_evidence_options()
            if metadata_service.evidences_loaded
            else []
        ),
        "all_evidences": (
            metadata_service.list_evidences(limit=len(metadata_service.evidences))
            if metadata_service.evidences_loaded
            else []
        ),
        "result": result,
        "error": error,
        "form": form_values
        or {
            "age": "",
            "sex": "M",
            "evidences": "E_91, E_201, E_66, E_56_@_4",
            "initial_evidence": "E_91",
            "k": 5,
        },
    }


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Unexpected server error. Check server logs for details.",
            "error_type": type(exc).__name__,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _template_context(request))


@app.post("/web-predict", response_class=HTMLResponse)
async def web_predict(
    request: Request,
    age: str = Form(""),
    sex: str = Form(""),
    evidences: str = Form(""),
    initial_evidence: str = Form(""),
    k: str = Form("5"),
) -> HTMLResponse:
    form_values = {
        "age": age,
        "sex": sex,
        "evidences": evidences,
        "initial_evidence": initial_evidence,
        "k": k,
    }
    try:
        evidence_list = safe_parse_evidence_list(evidences)
        payload = TopKPredictionRequest(
            age=int(age),
            sex=sex,
            evidences=evidence_list,
            initial_evidence=initial_evidence or None,
            k=int(k),
        )
        result = _prediction_payload(payload, payload.k)
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(request, result=jsonable_encoder(result), form_values=form_values),
        )
    except (ValueError, ValidationError, HTTPException, EvidenceParseError) as exc:
        if isinstance(exc, HTTPException):
            detail = exc.detail
            if isinstance(detail, dict):
                error = detail.get("message", str(detail))
                if detail.get("missing_files"):
                    error = f"{error} Missing files: {', '.join(detail['missing_files'])}"
            else:
                error = str(detail)
        else:
            error = str(exc)
        return templates.TemplateResponse(
            request,
            "index.html",
            _template_context(request, error=error, form_values=form_values),
            status_code=200,
        )


@app.get("/health")
async def health() -> dict[str, Any]:
    return _health_payload()


@app.get("/metadata/evidences")
async def evidences(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    if not metadata_service.evidences_loaded:
        raise _metadata_unavailable("Evidence metadata is not available.")
    return metadata_service.list_evidences(limit=limit)


@app.get("/metadata/evidences/{evidence_id}")
async def evidence_detail(evidence_id: str) -> dict[str, Any]:
    if not metadata_service.evidences_loaded:
        raise _metadata_unavailable("Evidence metadata is not available.")
    evidence = metadata_service.get_evidence(evidence_id)
    if not evidence:
        raise HTTPException(status_code=404, detail=f"Evidence ID '{evidence_id}' was not found.")
    return evidence


@app.get("/metadata/conditions")
async def conditions(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    if not metadata_service.conditions_loaded:
        raise _metadata_unavailable("Condition metadata is not available.")
    return metadata_service.list_conditions(limit=limit)


@app.get("/metadata/conditions/{condition_name}")
async def condition_detail(condition_name: str) -> dict[str, Any]:
    if not metadata_service.conditions_loaded:
        raise _metadata_unavailable("Condition metadata is not available.")
    condition = metadata_service.get_condition(condition_name)
    if not condition:
        raise HTTPException(status_code=404, detail=f"Condition '{condition_name}' was not found.")
    return condition


@app.get("/model-info")
async def model_info() -> dict[str, Any]:
    return model_service.info()


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    try:
        return model_service.metrics()
    except Exception as exc:  # noqa: BLE001 - malformed metrics should be a clear API response
        raise HTTPException(status_code=500, detail=f"Could not read metrics file: {exc}") from exc


@app.get("/api/clinical-evidence", response_model=ClinicalEvidenceResponse)
async def clinical_evidence(
    condition: str = Query(..., max_length=400, description="Free-text PubMed query."),
    article_type: ArticleType = Query("all"),
    years: YearsFilter = Query("5"),
) -> dict[str, Any]:
    """Search PubMed for literature related to an editable clinical query.

    Only the search terms reach NCBI: no patient demographics or evidences are
    forwarded, and failures stay contained in this endpoint.
    """
    try:
        return await search_clinical_evidence(condition, article_type=article_type, years=years)
    except InvalidSearchParametersError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ClinicalEvidenceTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="PubMed did not respond in time. Try again in a moment.",
        ) from exc
    except ClinicalEvidenceUpstreamError as exc:
        raise HTTPException(
            status_code=502,
            detail="PubMed is currently unavailable. Try again later.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - literature search must never leak internals
        logger.warning("Clinical evidence search failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The literature search could not be completed.",
        ) from exc


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> dict[str, Any]:
    return _prediction_payload(request, k=5)


@app.post("/predict-topk", response_model=PredictionResponse)
async def predict_topk(request: TopKPredictionRequest) -> dict[str, Any]:
    return _prediction_payload(request, k=request.k)


# ------------------------------------------------- optional multimodal orchestration
#
# The two models stay independent services: neither calls the other. This layer
# runs them and hands their outputs to the fusion service.

DISCORDANCE_WARNING = (
    "Discordant evidence: the clinical and radiographic models do not support the same "
    "conclusion. Physician review is required."
)
IMAGE_UNAVAILABLE_FINDING = "The chest X-ray was not analysed."
FUSION_UNAVAILABLE_REASON = (
    "Image-assisted re-ranking is disabled, so the clinical ranking was left unchanged."
)


def _clinical_ranking(top_k_diagnoses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add an explicit rank to the untouched DDXPlus Top-k rows."""
    return [
        {
            **item,
            "rank": position,
            "model_score": item.get("probability"),
        }
        for position, item in enumerate(top_k_diagnoses, start=1)
    ]


def _integrated_ranking(entries: list[RankedDiagnosis]) -> list[dict[str, Any]]:
    """Attach condition metadata to the experimental image-assisted ranking."""
    ranking: list[dict[str, Any]] = []
    for entry in entries:
        condition_details = metadata_service.condition_summary(entry.diagnosis)
        ranking.append(
            {
                "rank": entry.rank,
                "diagnosis": entry.diagnosis,
                "integrated_model_score": entry.score,
                "icd10_id": condition_details.get("icd10_id"),
                "severity": condition_details.get("severity"),
            }
        )
    return ranking


def _clinical_as_integrated(clinical_diagnoses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mirror the clinical ranking when no fusion was performed."""
    return [
        {
            "rank": item["rank"],
            "diagnosis": item["diagnosis"],
            "integrated_model_score": item.get("probability"),
            "icd10_id": item.get("icd10_id"),
            "severity": item.get("severity"),
        }
        for item in clinical_diagnoses
    ]


def _imaging_status(status: str, finding: str, detail: str | None = None) -> dict[str, Any]:
    """Build an imaging block for the cases where no inference was performed."""
    return {
        "status": status,
        "finding": finding,
        "model_name": pneumonia_service.model_name,
        "calibration_status": CALIBRATION_NOT_VALIDATED,
        "raw_logit": None,
        "image_model_score": None,
        "threshold": pneumonia_service.threshold,
        "class_index": None,
        "class_label": None,
        "detail": detail,
        "disclaimer": IMAGING_DISCLAIMER,
    }


def _parse_integrated_payload(payload: str) -> TopKPredictionRequest:
    """Validate the multipart JSON field with the existing Top-k schema."""
    try:
        data = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail="The 'payload' field must contain a valid JSON object.",
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=422,
            detail="The 'payload' field must contain a JSON object.",
        )
    try:
        return TopKPredictionRequest(**data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=jsonable_encoder(exc.errors())) from exc


async def _read_chest_xray(upload: UploadFile) -> bytes:
    """Read at most the configured limit plus one byte, never the whole stream."""
    limit = settings.max_image_upload_bytes
    try:
        data = await upload.read(limit + 1)
    except Exception as exc:  # noqa: BLE001 - a broken upload must not break prediction
        logger.warning("The chest X-ray upload could not be read: %s", type(exc).__name__)
        raise ImageValidationError("The uploaded chest X-ray could not be read.") from exc
    if len(data) > limit:
        raise ImageValidationError("The uploaded chest X-ray exceeds the 10 MB limit.")
    return data


def _run_image_assessment(image_bytes: bytes) -> tuple[dict[str, Any], list[str]]:
    """Run the chest X-ray model, converting every failure into a safe status."""
    try:
        assessment = pneumonia_service.predict(image_bytes)
    except ImageValidationError as exc:
        message = str(exc)
        return _imaging_status("invalid", IMAGE_UNAVAILABLE_FINDING, message), [message]
    except PneumoniaServiceUnavailableError as exc:
        message = str(exc)
        return _imaging_status("unavailable", IMAGE_UNAVAILABLE_FINDING, message), [message]
    except PneumoniaInferenceError as exc:
        message = str(exc)
        return _imaging_status("error", IMAGE_UNAVAILABLE_FINDING, message), [message]
    except Exception as exc:  # noqa: BLE001 - image failures never disable the DDX service
        logger.warning("Chest X-ray assessment failed unexpectedly: %s", type(exc).__name__)
        message = "The chest X-ray could not be analysed."
        return _imaging_status("error", IMAGE_UNAVAILABLE_FINDING, message), [message]

    imaging = assessment.as_dict()
    imaging["detail"] = None
    return imaging, []


def _discordance_warning(
    outcome: FusionOutcome, clinical_top_1: str, requested_k: int
) -> list[str]:
    """Flag the two genuine ways the clinical and radiographic models can disagree."""
    if not outcome.applied:
        return []
    target = outcome.target_diagnosis.strip().casefold()
    leads_clinically = clinical_top_1.strip().casefold() == target
    if outcome.interpretation == INTERPRETATION_DOES_NOT_SUPPORT and leads_clinically:
        return [DISCORDANCE_WARNING]
    if (
        outcome.interpretation == INTERPRETATION_SUPPORTS
        and not leads_clinically
        and outcome.rank_before is not None
        and outcome.rank_before > requested_k
    ):
        return [DISCORDANCE_WARNING]
    return []


def _fallback_fusion(reason: str) -> dict[str, Any]:
    """Fusion block used when the fusion service itself cannot run."""
    return {
        "applied": False,
        "method": FUSION_METHOD,
        "interpretation": INTERPRETATION_NOT_APPLIED,
        "target_diagnosis": settings.pneumonia_ddx_label,
        "alpha": settings.pneumonia_fusion_alpha,
        "uncertainty_margin": settings.pneumonia_fusion_uncertainty_margin,
        "image_signal": None,
        "rank_before": None,
        "rank_after": None,
        "clinical_score_before": None,
        "integrated_score_after": None,
        "reason": reason,
        "limitation": FUSION_LIMITATION,
    }


async def _resolve_imaging_assessment(
    chest_xray: UploadFile,
) -> tuple[dict[str, Any], list[str]]:
    """Read and assess the upload, mapping every failure to a safe status block."""
    if not pneumonia_service.ready:
        reason = (
            pneumonia_service.unavailable_reason
            or "The Advanced Pneumonia Model is unavailable."
        )
        return (
            _imaging_status("unavailable", IMAGE_UNAVAILABLE_FINDING, reason),
            [f"Advanced Pneumonia Model unavailable: {reason}"],
        )
    try:
        image_bytes = await _read_chest_xray(chest_xray)
    except ImageValidationError as exc:
        return _imaging_status("invalid", IMAGE_UNAVAILABLE_FINDING, str(exc)), [str(exc)]
    finally:
        await chest_xray.close()
    return _run_image_assessment(image_bytes)


def _fusion_outcome(
    labels: list[str],
    probabilities: list[float],
    imaging: dict[str, Any] | None,
    image_analysed: bool,
    has_upload: bool,
    k: int,
) -> FusionOutcome:
    """Delegate to the fusion service, which decides whether fusion applies."""
    assert fusion_service is not None  # guarded by the caller
    if not labels or not probabilities:
        raise InvalidFusionInputError(
            "The clinical model did not return a complete class distribution."
        )
    if image_analysed and imaging is not None:
        return fusion_service.fuse(
            labels,
            probabilities,
            float(imaging["image_model_score"]),
            float(imaging["threshold"]),
            k,
        )
    if has_upload:
        return fusion_service.clinical_only(
            labels,
            probabilities,
            k,
            reason="The chest X-ray was not analysed, so the clinical ranking is unchanged.",
        )
    return fusion_service.clinical_only(labels, probabilities, k)


@app.post("/predict-integrated", response_model=IntegratedPredictionResponse)
async def predict_integrated(
    payload: str = Form(..., description="JSON body matching TopKPredictionRequest."),
    chest_xray: UploadFile | None = File(None),
) -> dict[str, Any]:
    """Run the differential model and, when a valid chest X-ray is supplied, fuse it.

    The clinical ranking is always returned untouched. Fusion is an experimental
    image-assisted re-ranking, never a calibrated combined diagnostic probability.
    """
    request = _parse_integrated_payload(payload)

    clinical = _prediction_payload(request, k=request.k, include_distribution=True)
    clinical_diagnoses = _clinical_ranking(clinical["top_k_diagnoses"])
    clinical_top_1 = str(clinical.get("predicted_diagnosis") or "")

    warnings_out: list[str] = []
    imaging: dict[str, Any] | None = None

    has_upload = chest_xray is not None and bool((chest_xray.filename or "").strip())
    if chest_xray is not None:
        if has_upload:
            imaging, image_warnings = await _resolve_imaging_assessment(chest_xray)
            warnings_out.extend(image_warnings)
        else:
            await chest_xray.close()  # an empty part is not an upload

    image_analysed = bool(imaging and imaging.get("status") == "analysed")
    distribution = clinical.get("class_distribution") or {}
    labels = list(distribution.get("labels") or [])
    probabilities = list(distribution.get("probabilities") or [])

    outcome: FusionOutcome | None = None
    if fusion_service is None:
        fusion_block = _fallback_fusion(fusion_unavailable_reason or FUSION_UNAVAILABLE_REASON)
        integrated_diagnoses = _clinical_as_integrated(clinical_diagnoses)
        if has_upload:
            warnings_out.append(FUSION_UNAVAILABLE_REASON)
    else:
        try:
            outcome = _fusion_outcome(
                labels, probabilities, imaging, image_analysed, has_upload, request.k
            )
        except Exception as exc:  # noqa: BLE001 - fusion never breaks the clinical result
            logger.warning("Image-assisted fusion skipped: %s", exc)
            fusion_block = _fallback_fusion(
                "The clinical scores could not be fused, so the clinical ranking is unchanged."
            )
            integrated_diagnoses = _clinical_as_integrated(clinical_diagnoses)
            if image_analysed:
                warnings_out.append(
                    "The radiographic result could not be combined with the clinical ranking."
                )

    if outcome is not None:
        fusion_block = outcome.as_dict()
        integrated_diagnoses = _integrated_ranking(outcome.integrated_diagnoses)
        warnings_out.extend(outcome.warnings)
        warnings_out.extend(_discordance_warning(outcome, clinical_top_1, request.k))
        if imaging is not None and outcome.interpretation == INTERPRETATION_INCONCLUSIVE:
            # The score sits inside the configured uncertainty interval.
            imaging = {**imaging, "finding": FINDING_INCONCLUSIVE}

    assessment_mode = (
        ASSESSMENT_MODE_CLINICAL_PLUS_XRAY if image_analysed else ASSESSMENT_MODE_CLINICAL_ONLY
    )

    return {
        "assessment_mode": assessment_mode,
        "requested_k": request.k,
        "predicted_diagnosis": clinical["predicted_diagnosis"],
        "confidence": clinical.get("confidence"),
        "clinical_diagnoses": clinical_diagnoses,
        "imaging_assessment": imaging,
        "fusion": fusion_block,
        "integrated_diagnoses": integrated_diagnoses,
        "input_summary": clinical["input_summary"],
        "interpreted_evidences": clinical["interpreted_evidences"],
        "urgency": clinical.get("urgency"),
        "disclaimer": clinical["disclaimer"],
        "warnings": list(dict.fromkeys(warnings_out)),
    }
