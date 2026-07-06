from sanity4_parser import parse_candidate_response


def test_parse_candidate_response_ok():
    out = parse_candidate_response('{"verdict":"valid","reason":"The channels diverge in the interval.","confidence":0.9}')
    assert out.status == "OK"
    assert out.verdict == "valid"


def test_parse_candidate_response_fenced():
    out = parse_candidate_response('```json\n{"verdict":"invalid","reason":"The interval is aligned.","confidence":0.8}\n```')
    assert out.status == "OK"
    assert out.verdict == "invalid"


def test_parse_candidate_response_invalid_verdict():
    out = parse_candidate_response('{"verdict":"maybe","reason":"x","confidence":0.7}')
    assert out.status == "PARSE_ERROR"


def test_parse_candidate_response_bad_confidence():
    out = parse_candidate_response('{"verdict":"valid","reason":"x","confidence":2.0}')
    assert out.status == "PARSE_ERROR"
