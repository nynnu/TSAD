from sanity2_parser import parse_reason_judgment


def test_parse_reason_judgment_ok():
    out = parse_reason_judgment('{"reason_type":"relational","rationale":"It compares both channels."}')
    assert out.status == "OK"
    assert out.reason_type == "relational"


def test_parse_reason_judgment_fenced():
    out = parse_reason_judgment('```json\n{"reason_type":"vague","rationale":"Too generic."}\n```')
    assert out.status == "OK"
    assert out.reason_type == "vague"


def test_parse_reason_judgment_invalid_type():
    out = parse_reason_judgment('{"reason_type":"other","rationale":"x"}')
    assert out.status == "PARSE_ERROR"
