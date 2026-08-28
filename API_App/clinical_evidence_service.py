"""PubMed literature search built on the official NCBI E-utilities.

The integration is deliberately lazy: no network call is issued at import time
or during ordinary predictions. Only the physician-editable search terms are
sent upstream, never patient demographics or evidence tokens.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Final, Iterable, Literal

import httpx


logger = logging.getLogger(__name__)

ArticleType = Literal["all", "guideline", "systematic_review", "diagnostic_study"]
YearsFilter = Literal["3", "5", "10", "all"]

EUTILS_BASE_URL: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL: Final = f"{EUTILS_BASE_URL}/esearch.fcgi"
EFETCH_URL: Final = f"{EUTILS_BASE_URL}/efetch.fcgi"
PUBMED_ARTICLE_URL: Final = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

NCBI_TOOL_NAME: Final = "ED2A"
MAX_RESULTS: Final = 5
REQUEST_TIMEOUT_SECONDS: Final = 12.0
ABSTRACT_EXCERPT_CHARS: Final = 320
CONDITION_MIN_CHARS: Final = 2
CONDITION_MAX_CHARS: Final = 160

#: PubMed publication-type / topic filters appended to the physician query.
ARTICLE_TYPE_FILTERS: Final[dict[str, str]] = {
    "all": "",
    "guideline": '(Guideline[pt] OR "Practice Guideline"[pt] OR "Consensus Development Conference"[pt])',
    "systematic_review": '("Systematic Review"[pt] OR "Meta-Analysis"[pt])',
    "diagnostic_study": (
        '("Sensitivity and Specificity"[MeSH Terms] OR "Diagnosis"[Subheading]'
        ' OR "diagnostic accuracy"[tiab] OR "differential diagnosis"[tiab])'
    ),
}

#: Relative publication windows expressed in days for the ESearch `reldate`.
YEARS_TO_RELATIVE_DAYS: Final[dict[str, int | None]] = {
    "3": 3 * 365,
    "5": 5 * 365,
    "10": 10 * 365,
    "all": None,
}

#: Generic types that carry no useful signal when a specific one is available.
_GENERIC_PUBLICATION_TYPES: Final[frozenset[str]] = frozenset(
    {"journal article", "english abstract", "research support, non-u.s. gov't"}
)

_MONTH_ABBREVIATIONS: Final[dict[str, str]] = {
    "1": "Jan", "2": "Feb", "3": "Mar", "4": "Apr", "5": "May", "6": "Jun",
    "7": "Jul", "8": "Aug", "9": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}


class ClinicalEvidenceError(RuntimeError):
    """Base class for controlled PubMed failures."""


class InvalidSearchParametersError(ClinicalEvidenceError):
    """Raised when the caller supplied parameters PubMed cannot be queried with."""


class ClinicalEvidenceTimeoutError(ClinicalEvidenceError):
    """Raised when NCBI did not answer within the configured timeout."""


class ClinicalEvidenceUpstreamError(ClinicalEvidenceError):
    """Raised for any other upstream transport, status or payload failure."""


def normalise_condition(condition: str) -> str:
    """Trim the query and enforce the accepted length window."""
    trimmed = " ".join(str(condition or "").split())
    if not CONDITION_MIN_CHARS <= len(trimmed) <= CONDITION_MAX_CHARS:
        raise InvalidSearchParametersError(
            f"condition must contain between {CONDITION_MIN_CHARS} and "
            f"{CONDITION_MAX_CHARS} characters once trimmed."
        )
    return trimmed


def build_search_term(condition: str, article_type: str) -> str:
    """Combine the physician query with the selected publication-type filter."""
    article_filter = ARTICLE_TYPE_FILTERS.get(article_type)
    if article_filter is None:
        raise InvalidSearchParametersError(f"Unsupported article_type '{article_type}'.")
    if not article_filter:
        return condition
    return f"({condition}) AND {article_filter}"


def _relative_days(years: str) -> int | None:
    if years not in YEARS_TO_RELATIVE_DAYS:
        raise InvalidSearchParametersError(f"Unsupported years filter '{years}'.")
    return YEARS_TO_RELATIVE_DAYS[years]


def _credential_params() -> dict[str, str]:
    """Add the fixed tool name plus optional NCBI credentials from the environment."""
    params = {"tool": NCBI_TOOL_NAME}
    email = os.getenv("NCBI_EMAIL", "").strip()
    if email:
        params["email"] = email
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    if api_key:
        params["api_key"] = api_key
    return params


def build_esearch_params(condition: str, article_type: str, years: str) -> dict[str, Any]:
    """Build ESearch parameters; values are passed to httpx, never concatenated."""
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": build_search_term(condition, article_type),
        "retmode": "json",
        "retmax": MAX_RESULTS,
        "sort": "pub_date",
        **_credential_params(),
    }
    reldate = _relative_days(years)
    if reldate is not None:
        params["datetype"] = "pdat"
        params["reldate"] = reldate
    return params


def build_efetch_params(pmids: Iterable[str]) -> dict[str, Any]:
    return {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        **_credential_params(),
    }


def _node_text(node: ET.Element | None) -> str:
    """Flatten nested XML markup (``<i>``, ``<sup>``, ...) into plain text."""
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _first_text(parent: ET.Element, path: str) -> str:
    return _node_text(parent.find(path))


def parse_esearch_payload(payload: Any) -> tuple[int, list[str]]:
    """Extract the total match count and the returned PMIDs from ESearch JSON."""
    if not isinstance(payload, dict):
        raise ClinicalEvidenceUpstreamError("PubMed returned an unexpected search payload.")
    result = payload.get("esearchresult")
    if not isinstance(result, dict):
        raise ClinicalEvidenceUpstreamError("PubMed returned an unexpected search payload.")

    try:
        total_matches = int(str(result.get("count", "0")))
    except (TypeError, ValueError):
        total_matches = 0

    raw_ids = result.get("idlist")
    pmids = [str(pmid).strip() for pmid in raw_ids if str(pmid).strip()] if isinstance(raw_ids, list) else []
    return max(total_matches, 0), pmids[:MAX_RESULTS]


def _parse_authors(article: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article.findall("./AuthorList/Author"):
        collective = _first_text(author, "CollectiveName")
        if collective:
            authors.append(collective)
            continue
        last_name = _first_text(author, "LastName")
        given = _first_text(author, "ForeName") or _first_text(author, "Initials")
        full_name = " ".join(part for part in (given, last_name) if part)
        if full_name:
            authors.append(full_name)
    return authors


def _parse_publication_date(article: ET.Element) -> str | None:
    pub_date = article.find("./Journal/JournalIssue/PubDate")
    if pub_date is not None:
        medline_date = _first_text(pub_date, "MedlineDate")
        if medline_date:
            return medline_date
        year = _first_text(pub_date, "Year")
        if year:
            month = _first_text(pub_date, "Month")
            month = _MONTH_ABBREVIATIONS.get(month.lstrip("0"), month)
            day = _first_text(pub_date, "Day")
            return " ".join(part for part in (year, month, day) if part)

    article_date = article.find("./ArticleDate")
    if article_date is not None:
        parts = [_first_text(article_date, tag) for tag in ("Year", "Month", "Day")]
        joined = "-".join(part for part in parts if part)
        if joined:
            return joined
    return None


def _parse_publication_type(article: ET.Element) -> str | None:
    types = [_node_text(node) for node in article.findall("./PublicationTypeList/PublicationType")]
    types = [item for item in types if item]
    if not types:
        return None
    specific = [item for item in types if item.lower() not in _GENERIC_PUBLICATION_TYPES]
    return (specific or types)[0]


def _parse_abstract_excerpt(article: ET.Element) -> str | None:
    segments: list[str] = []
    for node in article.findall("./Abstract/AbstractText"):
        text = _node_text(node)
        if not text:
            continue
        label = (node.get("Label") or "").strip()
        segments.append(f"{label}: {text}" if label else text)

    abstract = " ".join(segments).strip()
    if not abstract:
        return None
    if len(abstract) <= ABSTRACT_EXCERPT_CHARS:
        return abstract

    truncated = abstract[:ABSTRACT_EXCERPT_CHARS]
    cutoff = truncated.rfind(" ")
    if cutoff > ABSTRACT_EXCERPT_CHARS // 2:
        truncated = truncated[:cutoff]
    return f"{truncated.rstrip(' ,;:.')}…"


def _parse_pubmed_article(entry: ET.Element) -> dict[str, Any] | None:
    pmid = _first_text(entry, "./MedlineCitation/PMID")
    article = entry.find("./MedlineCitation/Article")
    if not pmid or article is None:
        return None

    journal = _first_text(article, "./Journal/ISOAbbreviation") or _first_text(article, "./Journal/Title")
    return {
        "pmid": pmid,
        "title": _first_text(article, "ArticleTitle") or "Title unavailable",
        "authors": _parse_authors(article),
        "journal": journal or None,
        "publication_date": _parse_publication_date(article),
        "publication_type": _parse_publication_type(article),
        "abstract_excerpt": _parse_abstract_excerpt(article),
        "url": PUBMED_ARTICLE_URL.format(pmid=pmid),
    }


def parse_efetch_articles(xml_text: str) -> list[dict[str, Any]]:
    """Parse an EFetch PubMed XML document into stable article dictionaries."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ClinicalEvidenceUpstreamError("PubMed returned a malformed XML document.") from exc

    entries = root.findall(".//PubmedArticle")
    articles = [_parse_pubmed_article(entry) for entry in entries]
    return [article for article in articles if article is not None][:MAX_RESULTS]


async def _get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> Any:
    response = await client.get(url, params=params)
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:
        raise ClinicalEvidenceUpstreamError("PubMed returned a malformed search payload.") from exc


async def search_clinical_evidence(
    condition: str,
    article_type: str = "all",
    years: str = "5",
) -> dict[str, Any]:
    """Run ESearch then EFetch and return the normalised evidence payload."""
    query = normalise_condition(condition)
    esearch_params = build_esearch_params(query, article_type, years)

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS),
            headers={"User-Agent": f"{NCBI_TOOL_NAME}/1.0"},
        ) as client:
            total_matches, pmids = parse_esearch_payload(
                await _get_json(client, ESEARCH_URL, esearch_params)
            )
            if not pmids:
                return {"query": query, "total_matches": total_matches, "articles": []}

            fetch_response = await client.get(EFETCH_URL, params=build_efetch_params(pmids))
            fetch_response.raise_for_status()
            articles = parse_efetch_articles(fetch_response.text)
    except httpx.TimeoutException as exc:
        logger.warning("PubMed request timed out: %s", exc)
        raise ClinicalEvidenceTimeoutError("PubMed did not respond in time.") from exc
    except httpx.HTTPStatusError as exc:
        logger.warning("PubMed returned HTTP %s.", exc.response.status_code)
        raise ClinicalEvidenceUpstreamError("PubMed rejected the literature search request.") from exc
    except httpx.HTTPError as exc:
        logger.warning("PubMed request failed: %s", exc)
        raise ClinicalEvidenceUpstreamError("PubMed could not be reached.") from exc

    return {"query": query, "total_matches": total_matches, "articles": articles}
