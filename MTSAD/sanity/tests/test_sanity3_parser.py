from sanity3_parser import parse_localization_response


def test_parse_localization_response_broken_ok():
    out = parse_localization_response(
        '{"answer":"broken","break_start":100,"break_end":200,"reason":"The channels diverge.","confidence":0.9}'
    )
    assert out.status == "OK"
    assert out.answer == "broken"
    assert out.break_start == 100
    assert out.break_end == 200


def test_parse_localization_response_fenced():
    out = parse_localization_response(
        '```json\n{"answer":"maintained","break_start":null,"break_end":null,"reason":"They stay aligned.","confidence":0.8}\n```'
    )
    assert out.status == "OK"
    assert out.answer == "maintained"


def test_parse_localization_response_invalid_interval():
    out = parse_localization_response(
        '{"answer":"broken","break_start":200,"break_end":100,"reason":"x","confidence":0.7}'
    )
    assert out.status == "PARSE_ERROR"


def test_parse_localization_response_maintained_with_interval():
    out = parse_localization_response(
        '{"answer":"maintained","break_start":10,"break_end":20,"reason":"x","confidence":0.7}'
    )
    assert out.status == "PARSE_ERROR"


def test_parse_localization_response_broken_missing_interval():
    out = parse_localization_response(
        '{"answer":"broken","break_start":null,"break_end":null,"reason":"x","confidence":0.7}'
    )
    assert out.status == "PARSE_ERROR"
