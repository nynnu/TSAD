import json
import re
from dataclasses import dataclass


VALID_VERDICTS = {"valid", "invalid"}


@dataclass
class CandidateParseResult:
    status: str
    verdict: str | None = None
    reason: str | None = None
    confidence: float | None = None
    failure_reason: str | None = None


def _strip_wrappers(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return text


def parse_candidate_response(raw: str) -> CandidateParseResult:
    text = _strip_wrappers(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception as exc:
            return CandidateParseResult("PARSE_ERROR", failure_reason=f"json_decode_error: {exc}")

    if not isinstance(obj, dict):
        return CandidateParseResult("PARSE_ERROR", failure_reason="json_root_not_object")

    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        return CandidateParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_verdict")

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return CandidateParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_reason")

    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return CandidateParseResult("PARSE_ERROR", failure_reason="confidence_not_float")
    if not 0.0 <= confidence <= 1.0:
        return CandidateParseResult("PARSE_ERROR", failure_reason="confidence_out_of_range")

    return CandidateParseResult("OK", verdict=verdict, reason=reason.strip(), confidence=confidence)
