import json
import re
from dataclasses import dataclass


VALID_ANSWERS = {"maintained", "broken"}


@dataclass
class ParseResult:
    status: str
    answer: str | None = None
    reason: str | None = None
    confidence: float | None = None
    failure_reason: str | None = None


def _strip_wrappers(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return text


def parse_response(raw: str) -> ParseResult:
    text = _strip_wrappers(raw or "")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            result = json.loads(m.group(0)) if m else {}
        except Exception as exc:
            return ParseResult(status="PARSE_ERROR", failure_reason=f"json_decode_error: {exc}")

    if not isinstance(result, dict):
        return ParseResult(status="PARSE_ERROR", failure_reason="json_root_not_object")

    answer = str(result.get("answer", "")).strip().lower()
    if answer not in VALID_ANSWERS:
        return ParseResult(status="PARSE_ERROR", failure_reason="invalid_or_missing_answer")

    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return ParseResult(status="PARSE_ERROR", failure_reason="invalid_or_missing_reason")

    try:
        confidence = float(result.get("confidence"))
    except (TypeError, ValueError):
        return ParseResult(status="PARSE_ERROR", failure_reason="confidence_not_float")
    if not 0.0 <= confidence <= 1.0:
        return ParseResult(status="PARSE_ERROR", failure_reason="confidence_out_of_range")

    return ParseResult(
        status="OK",
        answer=answer,
        reason=reason.strip(),
        confidence=confidence,
    )
