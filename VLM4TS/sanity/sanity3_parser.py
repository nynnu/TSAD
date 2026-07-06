import json
import re
from dataclasses import dataclass

from config import T


VALID_ANSWERS = {"maintained", "broken"}


@dataclass
class LocalizationParseResult:
    status: str
    answer: str | None = None
    break_start: int | None = None
    break_end: int | None = None
    reason: str | None = None
    confidence: float | None = None
    failure_reason: str | None = None


def _strip_wrappers(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return text


def _load_json(raw: str):
    text = _strip_wrappers(raw)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _to_int(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("bool_not_int")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
        return int(value.strip())
    raise ValueError("not_integer")


def parse_localization_response(raw: str, t: int = T) -> LocalizationParseResult:
    try:
        obj = _load_json(raw)
    except Exception as exc:
        return LocalizationParseResult("PARSE_ERROR", failure_reason=f"json_decode_error: {exc}")

    if not isinstance(obj, dict):
        return LocalizationParseResult("PARSE_ERROR", failure_reason="json_root_not_object")

    answer = str(obj.get("answer", "")).strip().lower()
    if answer not in VALID_ANSWERS:
        return LocalizationParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_answer")

    try:
        break_start = _to_int(obj.get("break_start"))
        break_end = _to_int(obj.get("break_end"))
    except ValueError:
        return LocalizationParseResult("PARSE_ERROR", failure_reason="breakpoints_not_integer_or_null")

    if answer == "maintained" and (break_start is not None or break_end is not None):
        return LocalizationParseResult("PARSE_ERROR", failure_reason="maintained_with_interval")
    if answer == "broken":
        if break_start is None or break_end is None:
            return LocalizationParseResult("PARSE_ERROR", failure_reason="broken_missing_interval")
        if not 0 <= break_start < break_end <= t:
            return LocalizationParseResult("PARSE_ERROR", failure_reason="interval_out_of_bounds")

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return LocalizationParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_reason")

    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return LocalizationParseResult("PARSE_ERROR", failure_reason="confidence_not_float")
    if not 0.0 <= confidence <= 1.0:
        return LocalizationParseResult("PARSE_ERROR", failure_reason="confidence_out_of_range")

    return LocalizationParseResult(
        status="OK",
        answer=answer,
        break_start=break_start,
        break_end=break_end,
        reason=reason.strip(),
        confidence=confidence,
    )
