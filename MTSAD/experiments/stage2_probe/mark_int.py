"""INT 결과 채점 표시 그림 (나연, 2026-11-10) — VLM 입력 그림과 같고, 채널 색만 채점 결과로
  초록 = GT인데 잡음 (점수 ≥ 5) · 주황 = GT인데 놓침 · 보라 = GT 아닌데 잡음 · 검정 = GT 아니고 안 잡음
  점수 = 그 구간의 여러 장 답 평균 (기준 채널), 채점과 같음
  실행:  python mark_int.py machine-2-3 machine-1-2 machine-2-1   → raw_view_stage/INT_marked/{machine}/{k:03d}_p{n}.png"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from integ_view import prep, draw
from vlm_stage import gt_ivs, THR
HERE = Path(__file__).resolve().parent; C = dict(tp="#16a34a", fn="#f97316", fp="#9333ea")
for e in sys.argv[1:]:
    for k in gt_ivs(e):
        ctx = prep(e, k); sc = {}
        for f in sorted((HERE / "results" / "vlm_stage" / "INT").glob(f"{e}_{k:03d}_p*.json")):
            for c in (json.loads(f.read_text())["parsed"] or {}).get("channels", []):
                sc.setdefault(int(c["channel"]), []).append(float(c.get("score") or 0))
        v = {c: np.mean(sc.get(c, [0])) for c in ctx["s1"]}; G = ctx["G"]
        mark = {c: C["tp"] if c in G and v[c] >= THR else C["fn"] if c in G else C["fp"] for c in ctx["s1"] if c in G or v[c] >= THR}
        tp, fn, fp = ([c for c in sorted(mark) if mark[c] == C[x]] for x in ("tp", "fn", "fp")); miss = sorted(G - set(ctx["s1"]))
        note = (f"[채점 확인용] {e} · Stage1 구간 #{k} [{ctx['a']}, {ctx['b']}) · 초록 = GT 잡음 {tp} · 주황 = GT 놓침 {fn} · "
                f"보라 = GT 아닌데 잡음 {fp} · Stage1 후보 밖 GT {miss}")
        for p, rows in enumerate(ctx["pages"], 1):
            note_p = note + "\n이 장 점수: " + " ".join(f"ch{c}={v[c]:.1f}" for c in rows)
            draw(ctx, rows, HERE / "raw_view_stage" / "INT_marked" / e / f"{k:03d}_p{p}.png", mark=mark, note=note_p)
        print(e, k, "잡음", tp, "놓침", fn, "오탐", fp, flush=True)
