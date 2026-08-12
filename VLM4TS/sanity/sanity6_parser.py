import json
import re
from dataclasses import dataclass, field


VALID_LABELS = {"maintained", "broken"}


@dataclass
class MultiChannelParseResult:
    status: str
    pair_answers: dict = field(default_factory=dict)
    reason: str | None = None
    confidence: float | None = None
    failure_reason: str | None = None


def _strip_wrappers(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return text


def parse_multichannel_response(raw: str, expected_pairs: list[str]) -> MultiChannelParseResult:
    text = _strip_wrappers(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception as exc:
            return MultiChannelParseResult("PARSE_ERROR", failure_reason=f"json_decode_error: {exc}")

    if not isinstance(obj, dict):
        return MultiChannelParseResult("PARSE_ERROR", failure_reason="json_root_not_object")

    pairs = obj.get("pairs")
    if not isinstance(pairs, dict):
        return MultiChannelParseResult("PARSE_ERROR", failure_reason="missing_or_invalid_pairs_object")

    pair_answers: dict[str, str] = {}
    for name in expected_pairs:
        value = str(pairs.get(name, "")).strip().lower()
        if value not in VALID_LABELS:
            return MultiChannelParseResult("PARSE_ERROR", failure_reason=f"invalid_or_missing_pair_answer:{name}")
        pair_answers[name] = value

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return MultiChannelParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_reason")

    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return MultiChannelParseResult("PARSE_ERROR", failure_reason="confidence_not_float")
    if not 0.0 <= confidence <= 1.0:
        return MultiChannelParseResult("PARSE_ERROR", failure_reason="confidence_out_of_range")

    return MultiChannelParseResult(
        status="OK",
        pair_answers=pair_answers,
        reason=reason.strip(),
        confidence=confidence,
    )
