"""Evidence parsing and validation helpers."""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable, Mapping


EVIDENCE_ID_PATTERN = re.compile(r"^E_\d+$")
EVIDENCE_VALUE_SEPARATOR = "_@_"


class EvidenceParseError(ValueError):
    """Raised when an evidence token cannot be parsed safely."""


def _coerce_evidence_value(value: str) -> int | str:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def parse_evidence_token(token: str) -> dict[str, Any]:
    """Parse a DDXPlus evidence token into base evidence ID and value.

    Examples:
    - E_91 -> {"evidence_id": "E_91", "value": 1}
    - E_56_@_4 -> {"evidence_id": "E_56", "value": 4}
    - E_54_@_V_161 -> {"evidence_id": "E_54", "value": "V_161"}
    """
    if token is None:
        raise EvidenceParseError("Evidence token cannot be null.")

    raw_token = str(token).strip()
    if not raw_token:
        raise EvidenceParseError("Evidence token cannot be empty.")

    if EVIDENCE_VALUE_SEPARATOR in raw_token:
        evidence_id, value = raw_token.split(EVIDENCE_VALUE_SEPARATOR, maxsplit=1)
        evidence_id = evidence_id.strip()
        if not value.strip():
            raise EvidenceParseError(f"Evidence token '{raw_token}' has an empty value.")
        parsed_value: int | str = _coerce_evidence_value(value)
    else:
        evidence_id = raw_token
        parsed_value = 1

    if not EVIDENCE_ID_PATTERN.match(evidence_id):
        raise EvidenceParseError(
            f"Evidence token '{raw_token}' is invalid. Expected formats like E_91 or E_56_@_4."
        )

    return {"raw": raw_token, "evidence_id": evidence_id, "value": parsed_value}


def safe_parse_evidence_list(value: Any) -> list[str]:
    """Safely parse evidence lists from real lists, repr strings, or comma text.

    The DDXPlus CSV stores EVIDENCES as a list-like value. This function uses
    ast.literal_eval for string representations and never uses eval.
    """
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []

        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError) as exc:
                raise EvidenceParseError("Could not parse EVIDENCES list safely.") from exc
            if not isinstance(parsed, (list, tuple, set)):
                raise EvidenceParseError("Parsed EVIDENCES value is not a list.")
            return [str(item).strip() for item in parsed if str(item).strip()]

        return [part.strip() for part in text.split(",") if part.strip()]

    raise EvidenceParseError(f"Unsupported EVIDENCES value type: {type(value).__name__}.")


def validate_evidence_ids(
    evidences: Iterable[str],
    evidences_metadata: Mapping[str, Any],
) -> list[str]:
    """Return unique base evidence IDs that are not present in metadata."""
    invalid_ids: list[str] = []
    seen: set[str] = set()

    for token in evidences:
        parsed = parse_evidence_token(token)
        evidence_id = parsed["evidence_id"]
        if evidence_id not in evidences_metadata and evidence_id not in seen:
            invalid_ids.append(evidence_id)
            seen.add(evidence_id)

    return invalid_ids
