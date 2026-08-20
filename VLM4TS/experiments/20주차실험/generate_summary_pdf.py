"""20주차실험 결과를 1페이지 PDF로 정리."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

pdfmetrics.registerFont(TTFont("KR", "/System/Library/Fonts/Supplemental/AppleGothic.ttf"))

OUT = Path(__file__).resolve().parent / "20주차실험_요약.pdf"

styles = {
    "title": ParagraphStyle("title", fontName="KR", fontSize=15, leading=18, spaceAfter=4),
    "h": ParagraphStyle("h", fontName="KR", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2, textColor=colors.HexColor("#1a3a6b")),
    "body": ParagraphStyle("body", fontName="KR", fontSize=8.3, leading=11),
    "concl": ParagraphStyle("concl", fontName="KR", fontSize=8.3, leading=11, textColor=colors.HexColor("#8a1a1a")),
    "small": ParagraphStyle("small", fontName="KR", fontSize=7, leading=9),
}

def tbl(data, col_widths, font_size=7.3, header=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "KR"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf5")),
            ("FONTNAME", (0, 0), (-1, 0), "KR"),
        ]
    t.setStyle(TableStyle(style))
    return t

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                         topMargin=12*mm, bottomMargin=10*mm, leftMargin=14*mm, rightMargin=14*mm)
S = []

S.append(Paragraph("20주차 실험 정리 — top-8 채널 선택 개선 (SMD, 48개 실제 GT 세그먼트)", styles["title"]))

# 1
S.append(Paragraph("1. top-8 고정 선정이 괜찮은가? — GT 채널 수(k) 분포", styles["h"]))
S.append(tbl([
    ["k 구간", "1-3개", "4-8개", "9-15개", "16-38개"],
    ["세그먼트 비율", "35.4%", "33.3%", "22.9%", "8.3% (최대 34/38)"],
], [30*mm, 28*mm, 28*mm, 28*mm, 30*mm]))
S.append(Paragraph("→ <b>결론: k가 1~34로 매우 넓게 퍼져있어 \"항상 8개\"라는 가정 자체가 구조적으로 안 맞음.</b> "
                    "k=16-38 구간은 top-8로 이론상 최대 24%밖에 못 잡음(8/34).", styles["concl"]))

# 2
S.append(Paragraph("2. 대안 설계: residual sum 기반 3단계 selective verification", styles["h"]))
S.append(Paragraph(
    "38채널 각각 z-score(자기 참조, DINOv2 patch-KNN residual sum) 계산 → "
    "<b>① z 매우 큼 → VLM 없이 자동 \"이상\" 확정</b> / "
    "<b>② z 애매함 → lineplot으로 VLM에게 판단시킴</b> / "
    "<b>③ z 매우 작음 → 자동 \"정상\" 확정.</b> "
    "top-8처럼 개수를 고정하지 않고 애매한 것만 가변 개수로 VLM에 보여주므로, "
    "\"lineplot에 8개만 넣을 수 있다\"는 제약도 자연히 완화됨.", styles["body"]))

# 3
S.append(Paragraph("3. residual sum 적응형 선택 실제 결과 (α=0.01) vs top-8(5-ref production)", styles["h"]))
S.append(tbl([
    ["k 구간", "top-8: P / R", "적응형: P / R"],
    ["1-3", "0.191 / 0.706", "0.182 / 0.618"],
    ["4-8", "0.477 / 0.631", "0.432 / 0.629"],
    ["9-15", "0.511 / 0.326", "0.580 / 0.539"],
    ["16-38", "1.000 / 0.293", "0.942 / 0.721"],
], [30*mm, 45*mm, 45*mm]))
S.append(Paragraph("→ <b>k=16-38 구간 recall 0.29→0.72로 급등</b> (k=34 세그먼트 2개는 P=R=1.0). "
                    "단, k=1-3은 top-8이 근소하게 더 나음 — α는 0.01 정도가 precision/recall 균형점.", styles["concl"]))

# 4
S.append(Paragraph("4. 3단계(확실히 이상 / 애매함 / 확실히 정상) 실제 분할 결과", styles["h"]))
S.append(Paragraph("GT 채널 395개가 어느 그룹에 있었나:", styles["body"]))
S.append(tbl([
    ["그룹", "뜻", "GT 채널 수"],
    ["확실히 이상", "VLM 없이 바로 \"이상\"으로 확정", "249개 (63%)"],
    ["애매함", "VLM한테 보여줘서 판단 맡김", "52개 (13%)"],
    ["확실히 정상", "VLM한테 보여주지도 않고 바로 버림", "94개 (24%)"],
], [30*mm, 65*mm, 33*mm]))
S.append(Spacer(1, 3*mm))
S.append(Paragraph("성능:", styles["body"]))
S.append(tbl([
    ["", "맞춘 비율(정밀도)", "놓치지 않은 비율(재현율)"],
    ["VLM 없이, \"확실히 이상\"만으로 판단", "42%", "61%"],
    ["VLM이 \"애매함\"도 다 완벽히 맞췄다고 치면(최선)", "49%", "78%"],
], [72*mm, 28*mm, 28*mm]))
S.append(Paragraph(
    "→ <b>\"확실히 정상\"이라고 버린 94개(전체 정답의 24%)는 VLM한테 보여주지도 않으니, VLM이 제아무리 잘해도 절대 못 잡음</b> — "
    "지금 가장 큰 구멍. line plot으로 직접 확인한 원인은 3가지: "
    "<b>①너무 짧은 스파이크</b>(256패치 sum에 희석) · <b>②224틱보다 긴 이상구간</b>(창 안에 정상 비교기준 없음, 완만한 추세로만 보임) · "
    "<b>③정상일 때도 비슷하게 튀는 채널</b>(크기만으론 이상 시점 구별 불가, 근본적으로 더 어려운 문제).", styles["concl"]))

doc.build(S)
print(f"Saved: {OUT}")
