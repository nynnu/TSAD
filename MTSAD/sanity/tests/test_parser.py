from parser import parse_response


def test_parse_well_formed():
    out = parse_response('{"answer":"broken","reason":"They diverge.","confidence":0.8}')
    assert out.status == "OK"
    assert out.answer == "broken"
    assert out.confidence == 0.8


def test_parse_code_fence():
    out = parse_response('```json\n{"answer":"maintained","reason":"They track together.","confidence":1}\n```')
    assert out.status == "OK"
    assert out.answer == "maintained"


def test_parse_missing_key():
    out = parse_response('{"answer":"broken","confidence":0.5}')
    assert out.status == "PARSE_ERROR"


def test_parse_bad_confidence():
    out = parse_response('{"answer":"broken","reason":"x","confidence":2}')
    assert out.status == "PARSE_ERROR"
