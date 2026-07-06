import json
import re
from dataclasses import dataclass


VALID_REASON_TYPES = {"relational", "single_channel", "vague", "hallucinated"}


@dataclass
class ReasonParseResult:
    status: str
    reason_type: str | None = None
    rationale: str | None = None
    failure_reason: str | None = None


def parse_reason_judgment(raw: str) -> ReasonParseResult:
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception as exc:
            return ReasonParseResult("PARSE_ERROR", failure_reason=f"json_decode_error: {exc}")

    if not isinstance(obj, dict):
        return ReasonParseResult("PARSE_ERROR", failure_reason="json_root_not_object")
    reason_type = str(obj.get("reason_type", "")).strip().lower()
    if reason_type not in VALID_REASON_TYPES:
        return ReasonParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_reason_type")
    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return ReasonParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_rationale")
    return ReasonParseResult("OK", reason_type=reason_type, rationale=rationale.strip())
