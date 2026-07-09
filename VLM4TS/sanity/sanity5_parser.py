import json
import re
from dataclasses import dataclass


VALID_LEFT = {"L0", "L1", "L2", "L3"}
VALID_RIGHT = {"R0", "R1", "R2", "R3"}


@dataclass
class BoundarySelectionParseResult:
    status: str
    left_option: str | None = None
    right_option: str | None = None
    reason: str | None = None
    confidence: float | None = None
    failure_reason: str | None = None


def _strip_wrappers(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return text


def parse_boundary_response(raw: str) -> BoundarySelectionParseResult:
    text = _strip_wrappers(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception as exc:
            return BoundarySelectionParseResult("PARSE_ERROR", failure_reason=f"json_decode_error: {exc}")

    if not isinstance(obj, dict):
        return BoundarySelectionParseResult("PARSE_ERROR", failure_reason="json_root_not_object")

    left_option = str(obj.get("left_option", "")).strip().upper()
    if left_option not in VALID_LEFT:
        return BoundarySelectionParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_left_option")

    right_option = str(obj.get("right_option", "")).strip().upper()
    if right_option not in VALID_RIGHT:
        return BoundarySelectionParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_right_option")

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return BoundarySelectionParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_reason")

    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return BoundarySelectionParseResult("PARSE_ERROR", failure_reason="confidence_not_float")
    if not 0.0 <= confidence <= 1.0:
        return BoundarySelectionParseResult("PARSE_ERROR", failure_reason="confidence_out_of_range")

    return BoundarySelectionParseResult(
        status="OK",
        left_option=left_option,
        right_option=right_option,
        reason=reason.strip(),
        confidence=confidence,
    )
