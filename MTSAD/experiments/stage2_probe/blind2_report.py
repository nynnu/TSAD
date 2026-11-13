"""
blind2.py 결과를 사람이 보기 쉬운 PDF로 (나연, 2026-11-05)

  내용  = 전체 결과(Stage1 · 기존 Stage2 · 지금 VLM 실행별 AUC · Recall · Precision, Stage1 같은 개수 비교)
          + 구간마다 표(채널은 VLM에게 보여준 순서 = 번호 순 · GT · Stage1 점수 · 기존 Stage2 · 실행별 판정(확률)) + 실행별 채널 이유
  실행  = 결과 폴더 × 방식. 숫자 있음 = results/blind2/ (채널별 숫자를 텍스트로 줌), 숫자 없음 = results/blind2_nonum/ (그림만)
          방식 (가) 하루 중 같은 시각 · (나) 최고 점수 채널 기준 · (다) 채널마다 가장 비슷한 구간 — 결과 파일이 있는 것만 넣음
  실행:  python blind2_report.py machine-2-9   → results/blind2/{machine}_review.pdf (Chrome 헤드리스로 인쇄)
"""
import html
import json
import subprocess
import sys
from pathlib import Path

from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blind2 import HERE, MODES, setup

NM = {"day": "(가) 하루 중 같은 시각", "top": "(나) 최고 점수 채널 기준", "own": "(다) 채널마다 가장 비슷한 구간"}
RUNS = [("blind2", m, f"{NM[m]} · 숫자 있음") for m in MODES] + [("blind2_nonum", m, f"{NM[m]} · 숫자 없음") for m in MODES]
OLD = HERE.parent / "stage2_link" / "output" / "zoom-b_order-score"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """body{font-family:'Apple SD Gothic Neo','AppleGothic',sans-serif;font-size:10px;color:#111;margin:0}
h1{font-size:17px;margin:0 0 6px} h2{font-size:13px;margin:18px 0 6px;padding-top:8px;border-top:2px solid #2563eb}
p,li{line-height:1.45} table{border-collapse:collapse;margin:6px 0;width:100%} th,td{border:1px solid #d1d5db;padding:3px 5px;text-align:left;vertical-align:top}
th{background:#f3f4f6} .gt{color:#dc2626;font-weight:700} .an{font-weight:700} .muted{color:#6b7280} .mode{margin:8px 0 2px;font-weight:700}
.base td{background:#f9fafb} ul{margin:2px 0 6px 16px;padding:0} li{margin:1px 0} @page{size:A4 landscape;margin:10mm}"""


def main():
    e = sys.argv[1]; tr, te, y, gtr, ivs, ks, sc = setup(e); E = html.escape
    runs = [r for r in RUNS if all((HERE / "results" / r[0] / f"{e}_{r[1]}_{k:03d}.json").exists() for k in ks)]
    D = {(r, k): json.loads((HERE / "results" / r[0] / f"{e}_{r[1]}_{k:03d}.json").read_text()) for r in runs for k in ks}
    A = {(r, k): {int(c["channel"]): c for c in (D[(r, k)]["parsed"] or {}).get("channels", [])} for r in runs for k in ks}
    G = {k: set().union(*[c for s, t, c in gtr if s < ivs[k - 1]["end"] and t > ivs[k - 1]["start"]]) for k in ks}
    S1 = {k: {c["channel"]: c["spike_score"] for c in ivs[k - 1]["channels"]} for k in ks}
    s1 = lambda k, c: -1e9 if S1[k][c] is None else S1[k][c]
    old = {(iv["start"], iv["end"]): {c["channel"] for c in iv["channels"]} for iv in json.loads((OLD / f"{e}.json").read_text())["intervals"]}
    oldp = {k: old.get((ivs[k - 1]["start"], ivs[k - 1]["end"]), set()) for k in ks}
    chs_of = {k: sorted(c["channel"] for c in ivs[k - 1]["channels"]) for k in ks}
    gg = [c in G[k] for k in ks for c in chs_of[k]]; n = sum(gg)

    def line(name, score, sel_of, cls=""):
        tp = sum(len(set(sel_of(k)) & G[k]) for k in ks); pr = sum(len(sel_of(k)) for k in ks)
        tp1 = sum(len(set(sorted(chs_of[k], key=lambda c: -s1(k, c))[:len(sel_of(k))]) & G[k]) for k in ks)
        return (f"<tr class='{cls}'><td>{name}</td><td>{roc_auc_score(gg, score):.3f}</td><td>{pr}</td><td>{tp / n:.3f}</td><td>{tp / max(pr, 1):.3f}</td>"
                f"<td>{tp1 / n:.3f}</td><td>{tp1 / max(pr, 1):.3f}</td></tr>")
    rows = [line("Stage1 (후보 전부 넘김, 점수 = 채널 점수)", [s1(k, c) for k in ks for c in chs_of[k]], lambda k: chs_of[k], "base"),
            line("기존 Stage2 (8채널씩 한 장, 고름/안 고름)", [float(c in oldp[k]) for k in ks for c in chs_of[k]], lambda k: sorted(oldp[k] & set(chs_of[k])), "base")]
    for r in runs:
        rows.append(line(r[2], [float(A[(r, k)].get(c, {}).get("prob", 0) or 0) for k in ks for c in chs_of[k]],
                         lambda k, r=r: [c for c in chs_of[k] if A[(r, k)].get(c, {}).get("anomalous")]))
    H = [f"<h1>{e} · VLM(GPT-4o) 블라인드 이상 채널 판정</h1>",
         "<p>VLM은 GT를 모른 채 Stage1 후보 채널(번호 순)을 보고 채널마다 이상인지·확률·이유를 답함. 그림은 채널마다 [test 창 | 비교 칸] 2칸, 비교 칸만 방식마다 다름.<br>"
         "<b>숫자 있음</b> = 채널마다 test 창 최고값·창 앞부분 중앙값·비교 칸 최고값을 텍스트로 함께 줌. "
         "<b>숫자 없음</b> = 그 텍스트를 빼고 그림만(이유는 '그림에서 읽은 값과 함께' 쓰라고 함). 나머지 프롬프트는 같음.<br>"
         "Stage1 점수 = 그 구간 안에서 그 채널의 (spike + 높이) 보정 점수 최댓값(<b>채널 점수</b>). 탈주기로만 뽑힌 채널은 점수 없음(−).<br>"
         "AUC = GT 채널 하나와 GT 아닌 채널 하나를 무작위로 뽑았을 때 GT 쪽 점수가 더 높을 확률(0.5 = 무작위, 1 = 완벽). 기존 Stage2는 고름(1)/안 고름(0)으로 계산.</p>",
         f"<h2>전체 결과 (GT 구간 {len(ks)}개 · 후보 {len(gg)}개 · GT {n}개)</h2>",
         "<table><tr><th>방법</th><th>AUC</th><th>이상이라고 한 채널</th><th>Recall</th><th>Precision</th>"
         "<th>Stage1 상위 같은 개수: Recall</th><th>Precision</th></tr>" + "".join(rows) + "</table>",
         "<ul><li>Recall · Precision은 후보 안의 GT 기준. Stage1이 놓친 GT 채널은 어느 방법도 볼 수 없어 빠짐</li>"
         "<li>Stage1 상위 같은 개수 = 구간마다 그 방법이 이상이라고 한 개수만큼 Stage1 점수 순으로 고른 경우(같은 개수로 공정 비교)</li></ul>"]
    for k in ks:
        iv = ivs[k - 1]; chs = chs_of[k]; top = iv["channels"][0]["channel"]; miss = sorted(G[k] - set(chs))
        H.append(f"<h2>구간 #{k} [{iv['start']}, {iv['end']}) · 후보 {len(chs)}개 · GT {sorted(G[k] & set(chs))}"
                 + (f" · Stage1이 놓친 GT {miss}" if miss else "") + "</h2>")
        t = ["<table><tr><th>채널</th><th>GT</th><th>Stage1 점수</th><th>기존 Stage2</th>" + "".join(f"<th>{r[2]}</th>" for r in runs) + "</tr>"]
        for c in chs:
            cells = "".join(("<td class='an'>이상" if A[(r, k)].get(c, {}).get("anomalous") else "<td>정상")
                            + f" ({float(A[(r, k)].get(c, {}).get('prob', 0) or 0):.2f})</td>" for r in runs)
            t.append(f"<tr><td{' class=gt' if c in G[k] else ''}>ch {c}{' ★' if c == top else ''}</td><td class='gt'>{'●' if c in G[k] else ''}</td>"
                     f"<td>{'−' if S1[k][c] is None else f'{S1[k][c]:.1f}'}</td><td>{'<b>고름</b>' if c in oldp[k] else '-'}</td>{cells}</tr>")
        H.append("".join(t) + "</table><p class='muted'>★ = Stage1 점수 1위 채널 ((나)의 기준) · ● = GT 채널</p>")
        for r in runs:
            H.append(f"<div class='mode'>{r[2]} — 요약: {E(str((D[(r, k)]['parsed'] or {}).get('summary', '')))}</div><ul>")
            for c in chs:
                x = A[(r, k)].get(c, {})
                rs = "; ".join(f"[{E(str(q.get('type')))}] {E(str(q.get('evidence')))}" for q in x.get("reasons", [])) or "-"
                ni = "; ".join(f"{E(str(q.get('info')))} (왜: {E(str(q.get('why')))})" for q in x.get("needed_info", []))
                H.append(f"<li><span{' class=gt' if c in G[k] else ''}>ch {c}{' ●' if c in G[k] else ''}</span> "
                         f"<b>{'이상' if x.get('anomalous') else '정상'}</b> ({float(x.get('prob', 0) or 0):.2f}): {rs}"
                         + (f" <span class='muted'>· 필요: {ni}</span>" if ni else "") + "</li>")
            H.append("</ul>")
    src = HERE / "results" / "blind2" / f"{e}_review.html"; pdf = src.with_suffix(".pdf")
    src.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(H)}</body></html>", encoding="utf-8")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", src.as_uri()],
                   check=True, capture_output=True)
    print(pdf)


if __name__ == "__main__":
    main()
