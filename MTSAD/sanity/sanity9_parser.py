import json
import re
from dataclasses import dataclass, field


VALID_CHANNELS = {"1", "2", "3", "4", "5", "6"}


@dataclass
class CausalParseResult:
    status: str
    root_cause_channel: str | None = None
    affected_channels: list[str] = field(default_factory=list)
    unaffected_channels: list[str] = field(default_factory=list)
    onset_time: int | None = None
    reason: str | None = None
    confidence: float | None = None
    failure_reason: str | None = None


def _strip_wrappers(raw: str) -> str:
    text = (raw or "").strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    return text


def _valid_channel_list(value) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out = []
    for c in value:
        c = str(c).strip()
        if c not in VALID_CHANNELS:
            return None
        out.append(c)
    return out


def parse_causal_response(raw: str) -> CausalParseResult:
    text = _strip_wrappers(raw)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except Exception as exc:
            return CausalParseResult("PARSE_ERROR", failure_reason=f"json_decode_error: {exc}")

    if not isinstance(obj, dict):
        return CausalParseResult("PARSE_ERROR", failure_reason="json_root_not_object")

    root = str(obj.get("root_cause_channel", "")).strip()
    if root.lower() == "none":
        root = "none"
    elif root not in VALID_CHANNELS:
        return CausalParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_root_cause_channel")

    affected = _valid_channel_list(obj.get("affected_channels"))
    if affected is None:
        return CausalParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_affected_channels")

    unaffected = _valid_channel_list(obj.get("unaffected_channels"))
    if unaffected is None:
        return CausalParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_unaffected_channels")

    onset_raw = obj.get("onset_time")
    onset_time = None
    if onset_raw is not None:
        if isinstance(onset_raw, bool):
            return CausalParseResult("PARSE_ERROR", failure_reason="onset_time_not_integer")
        try:
            onset_time = int(onset_raw)
        except (TypeError, ValueError):
            return CausalParseResult("PARSE_ERROR", failure_reason="onset_time_not_integer")

    reason = obj.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return CausalParseResult("PARSE_ERROR", failure_reason="invalid_or_missing_reason")

    try:
        confidence = float(obj.get("confidence"))
    except (TypeError, ValueError):
        return CausalParseResult("PARSE_ERROR", failure_reason="confidence_not_float")
    if not 0.0 <= confidence <= 1.0:
        return CausalParseResult("PARSE_ERROR", failure_reason="confidence_out_of_range")

    return CausalParseResult(
        status="OK",
        root_cause_channel=root,
        affected_channels=affected,
        unaffected_channels=unaffected,
        onset_time=onset_time,
        reason=reason.strip(),
        confidence=confidence,
    )
