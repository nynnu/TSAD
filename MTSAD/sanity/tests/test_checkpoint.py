from checkpoint import load_checkpoint, should_skip, upsert_checkpoint


def test_checkpoint_upsert_reload(tmp_path):
    path = tmp_path / "checkpoint.json"
    upsert_checkpoint(path, "C0_000", {"status": "OK", "case_type": "C0"})
    upsert_checkpoint(path, "C1_000", {"status": "PARSE_ERROR", "case_type": "C1"})
    data = load_checkpoint(path)
    assert data["C0_000"]["status"] == "OK"
    assert data["C1_000"]["status"] == "PARSE_ERROR"


def test_resume_skip_policy():
    assert should_skip({"status": "OK"})
    assert not should_skip({"status": "PARSE_ERROR"})
    assert not should_skip({"status": "API_ERROR"})
    assert not should_skip(None)
