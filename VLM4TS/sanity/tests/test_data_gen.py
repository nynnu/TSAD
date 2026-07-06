import numpy as np

from data_gen import GENERATORS, middle_third


def test_generators_lengths_and_metadata():
    for case, gen in GENERATORS.items():
        a, b, meta = gen(seed=1, t=300)
        assert len(a) == 300
        assert len(b) == 300
        assert meta["label"] in {"maintained", "broken"}
        if case in {"C1", "C2", "C3", "C5"}:
            assert (meta["break_start"], meta["break_end"]) == middle_third(300)
        if case in {"C0", "C6"}:
            assert meta["break_start"] is None
            assert meta["break_end"] is None


def test_broken_middle_cases_differ_inside_more_than_c0():
    c0_a, c0_b, _ = GENERATORS["C0"](seed=2, t=300)
    base_diff = np.mean(np.abs(c0_a - c0_b))
    s, e = middle_third(300)
    for case in ["C1", "C2", "C3"]:
        a, b, _ = GENERATORS[case](seed=2, t=300)
        assert np.mean(np.abs(a[s:e] - b[s:e])) > base_diff


def test_c6_is_maintained_but_noisier():
    a, b, meta = GENERATORS["C6"](seed=3, t=300)
    assert meta["label"] == "maintained"
    assert np.std(b - a) > 0.15
