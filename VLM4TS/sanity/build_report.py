"""Assembles the Oracle-Clean + DINO Relation Mini Sanity Check PDF report.

No pandoc/reportlab/weasyprint available in this environment, so the report
is built as a sequence of matplotlib figures saved via PdfPages (text pages
rendered with ax.text, data pages as normal charts/tables).
"""

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

SANITY_DIR = Path(__file__).resolve().parent
SANITY9_RUN = SANITY_DIR / "results" / "sanity9" / "runs" / "20260716_112728"
ORACLE9_ROOT = SANITY_DIR / "results" / "oracle9" / "runs"
DINO_RESULTS = SANITY_DIR / "results" / "dino_relation9"
OUT_PDF = SANITY_DIR / "results" / "oracle9_dino_relation_report.pdf"

PAGE_SIZE = (11.69, 8.27)  # A4 landscape


def _latest_oracle_run() -> Path:
    runs = sorted(ORACLE9_ROOT.glob("2*"))
    if not runs:
        raise SystemExit("No Oracle-9 run found.")
    return runs[-1]


def _wrap(line: str, width: int = 70) -> str:
    if not line:
        return line
    return "\n".join(
        textwrap.fill(sub, width=width) if sub else ""
        for sub in line.split("\n")
    )


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    ax.text(0.02, 0.96, title, fontsize=18, fontweight="bold", va="top", transform=ax.transAxes)
    y = 0.88
    for line in lines:
        wrapped = _wrap(line)
        ax.text(0.03, y, wrapped, fontsize=11, va="top", transform=ax.transAxes)
        y -= 0.032 * (1 + wrapped.count("\n")) - 0.01
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame, note: str = "", col_widths: list[float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ax.axis("off")
    ax.text(0.02, 0.96, title, fontsize=18, fontweight="bold", va="top", transform=ax.transAxes)
    if col_widths is None:
        col_widths = [1.0 / len(df.columns)] * len(df.columns)
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.4)
    for (row, _col), cell in tbl.get_celld().items():
        cell.set_text_props(wrap=True)
    if note:
        ax.text(0.02, 0.06, note, fontsize=9, va="top", transform=ax.transAxes, color="#444444")
    pdf.savefig(fig)
    plt.close(fig)


def _fmt(x, pct=False, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if pct:
        return f"{x*100:.{digits}f}%"
    return f"{x:.{digits}f}"


def build_experiment1_pages(pdf: PdfPages, oracle_run: Path) -> None:
    s9 = json.loads((SANITY9_RUN / "sanity9_summary.json").read_text(encoding="utf-8"))
    oracle_summary = json.loads((oracle_run / "oracle9_summary.json").read_text(encoding="utf-8"))
    oracle_samples = pd.read_csv(oracle_run / "raw" / "oracle9_samples.csv")
    sanity9_samples = pd.read_csv(SANITY9_RUN / "raw" / "sanity9_samples.csv")

    overlay = s9["by_vis_condition_sample"]["overlay"]
    subplot = s9["by_vis_condition_sample"]["subplot"]
    overlay_ch = s9["by_vis_condition_channel"]["overlay"]
    subplot_ch = s9["by_vis_condition_channel"]["subplot"]
    oracle_sample = oracle_summary["overall_sample"]
    oracle_ch = oracle_summary["overall_channel"]

    # NORMAL FP rate per vis condition, computed directly from the samples CSV
    normal = sanity9_samples[sanity9_samples["scenario"] == "NORMAL"]
    normal = normal[normal["parse_status"] == "OK"]
    normal_fp = {}
    for vis, sub in normal.groupby("vis_condition"):
        normal_fp[vis] = float((sub["pred_root"] != "none").mean())
    oracle_fp = oracle_summary["normal_fp_rate"]

    rows = [
        ["Root cause accuracy", _fmt(overlay["root_accuracy"], pct=True), _fmt(subplot["root_accuracy"], pct=True), _fmt(oracle_sample["root_accuracy"], pct=True)],
        ["Onset MAE (step)", _fmt(overlay["onset_mae"]), _fmt(subplot["onset_mae"]), _fmt(oracle_sample["onset_mae"])],
        ["Affected-set F1", _fmt(overlay_ch["f1"], pct=True), _fmt(subplot_ch["f1"], pct=True), _fmt(oracle_ch["f1"], pct=True)],
        ["NORMAL FP rate", _fmt(normal_fp.get("overlay"), pct=True), _fmt(normal_fp.get("subplot"), pct=True), _fmt(oracle_fp, pct=True)],
        ["n (sample)", str(overlay["n"]), str(subplot["n"]), str(oracle_sample["n"])],
    ]
    df = pd.DataFrame(rows, columns=["Metric", "Overlay", "Subplot", "Oracle-Clean"])
    _table_page(
        pdf, "실험 1 결과 — Overlay / Subplot / Oracle-Clean 비교",
        df,
        note="Oracle-Clean: 이미지 없이 GT(onset/breakdown_type/intensity)를 텍스트로만 제공. n=390(NORMAL 제외 360 + NORMAL 30), Overlay/Subplot과 동일한 390개 원본 샘플 재사용(재생성 없음).",
    )

    # n_affected line chart: overlay vs oracle root accuracy
    overlay_only = sanity9_samples[(sanity9_samples["vis_condition"] == "overlay") & (sanity9_samples["scenario"] != "NORMAL") & (sanity9_samples["parse_status"] == "OK")]
    overlay_by_n = overlay_only.groupby("n_affected")["root_correct"].mean()

    oracle_scored = oracle_samples[(oracle_samples["scenario"] != "NORMAL") & (oracle_samples["parse_status"] == "OK")]
    oracle_by_n = oracle_scored.groupby("n_affected")["root_correct"].mean()

    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    ns = sorted(overlay_by_n.index.tolist())
    ax.plot(ns, [overlay_by_n[n] for n in ns], marker="o", label="Overlay", linewidth=2)
    ax.plot(ns, [oracle_by_n.get(n, np.nan) for n in ns], marker="s", label="Oracle-Clean", linewidth=2)
    ax.set_xlabel("n_affected (영향받는 채널 수)")
    ax.set_ylabel("Root cause accuracy")
    ax.set_title("n_affected별 root cause accuracy — Overlay vs Oracle-Clean")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # Oracle failure cases: confidently wrong, sorted by confidence desc
    all_fails = oracle_scored[oracle_scored["root_correct"] == False].copy()  # noqa: E712
    fails = all_fails.sort_values("model_confidence", ascending=False).head(3)
    lines = [f"전체 {len(oracle_scored)}건 중 root cause 오답은 {len(all_fails)}건.", ""]
    for _, row in fails.iterrows():
        lines.append(
            f"[{row['case_id']}] true_root={row['true_root']} pred_root={row['pred_root']} "
            f"confidence={row['model_confidence']:.2f}\nreason: {row['model_reason']}\n"
        )
    _text_page(pdf, "실험 1 — Oracle-Clean 실패 사례 (confidently wrong)", lines)


def build_experiment2_pages(pdf: PdfPages) -> None:
    summary = json.loads((DINO_RESULTS / "dino_relation9_summary.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv(DINO_RESULTS / "raw" / "pairs.csv")

    role_order = ["cascade", "mild", "unrelated"]
    colors = {"cascade": "#d62728", "mild": "#ff7f0e", "unrelated": "#1f77b4"}

    fig, axes = plt.subplots(1, 2, figsize=PAGE_SIZE)
    for ax, col, title in zip(axes, ["delta_cls", "delta_patch"], ["CLS token", "Patch-mean"]):
        data = [pairs[pairs["role"] == r][col].values for r in role_order]
        bp = ax.boxplot(data, tick_labels=role_order, patch_artist=True)
        for patch, r in zip(bp["boxes"], role_order):
            patch.set_facecolor(colors[r])
            patch.set_alpha(0.6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"{title} — delta by role")
        ax.set_ylabel("delta (post_dist - pre_dist)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("실험 2 — DINO embedding delta 분포 (root vs 나머지 5채널, cascade/mild/unrelated)", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)

    kw_rows = [
        ["CLS / role (casc-mild-unrel)", *_kw_row(summary["kruskal_delta_cls_by_role"])],
        ["Patch / role (casc-mild-unrel)", *_kw_row(summary["kruskal_delta_patch_by_role"])],
        ["CLS / homog. vs heterog.", *_kw_row(summary["kruskal_homogeneous_delta_cls"])],
        ["Patch / homog. vs heterog.", *_kw_row(summary["kruskal_homogeneous_delta_patch"])],
    ]
    df = pd.DataFrame(kw_rows, columns=["Comparison", "H", "p-value", "Group means"])
    _table_page(
        pdf, "실험 2 — Kruskal-Wallis 검정 결과 (1차)", df, col_widths=[0.26, 0.10, 0.14, 0.50],
        note="주의: n=1800으로 커서 p-value가 작아도 효과크기가 노이즈 수준일 수 있음 -> 다음 페이지에서 effect size로 재검증.",
    )

    build_effect_size_pages(pdf)
    _sample_separation_pages(pdf, pairs)


def build_effect_size_pages(pdf: PdfPages) -> None:
    """Effect-size re-analysis: p-value alone is misleading at n=1800.
    Reports Cohen's d / AUC instead, tests breakdown_type-match as a direct
    2-group alternative to the 3-way role split, quantifies how often the
    naive role ordering fails at the sample level, and extends the
    comparison to a mid-layer CLS token and the last-layer attention map."""
    es = json.loads((DINO_RESULTS / "dino_relation9_effect_sizes.json").read_text(encoding="utf-8"))
    layers_path = DINO_RESULTS / "dino_relation9_layers_summary.json"

    rows = [
        ["CLS (final layer)", *_effect_row(es["delta_cls"]["role_pairwise"][0]), *_effect_row(es["delta_cls"]["type_match_affected_only"])],
        ["Patch-mean (final layer)", *_effect_row(es["delta_patch"]["role_pairwise"][0]), *_effect_row(es["delta_patch"]["type_match_affected_only"])],
    ]
    if layers_path.exists():
        lay = json.loads(layers_path.read_text(encoding="utf-8"))
        rows.append(["Mid-layer CLS (block 5)", *_effect_row(lay["mid_layer_cls_block5"]["cascade_vs_unrelated"]), *_effect_row(lay["mid_layer_cls_block5"]["homogeneous_vs_heterogeneous"])])
        rows.append(["Last-layer attention map", *_effect_row(lay["last_layer_attention_map"]["cascade_vs_unrelated"]), *_effect_row(lay["last_layer_attention_map"]["homogeneous_vs_heterogeneous"])])

    df = pd.DataFrame(rows, columns=["Representation", "role d", "role AUC", "role p", "type-match d", "type-match AUC", "type-match p"])
    _table_page(
        pdf, "실험 2 (추가) — 표현/비교축별 Effect Size 비교", df,
        col_widths=[0.24, 0.10, 0.10, 0.12, 0.14, 0.14, 0.16],
        note=(
            "role = cascade vs unrelated. type-match = homogeneous vs heterogeneous (affected 채널만, breakdown_type 일치 여부).\n"
            "|d|<0.2 negligible, 0.2~0.5 small, 0.5~0.8 medium, >0.8 large (Cohen 기준). AUC 0.5=구분력 없음.\n"
            f"실패 패턴 비율(샘플 단위, affected 평균 delta < unrelated 평균 delta): "
            f"CLS {es['failure_rate_cls']['failure_rate']*100:.1f}%, "
            f"patch-mean {es['failure_rate_patch']['failure_rate']*100:.1f}% (n={es['failure_rate_cls']['n_applicable_cases']})."
        ),
    )

    if layers_path.exists():
        lay = json.loads(layers_path.read_text(encoding="utf-8"))
        layer_pairs = pd.read_csv(DINO_RESULTS / "raw" / "layer_pairs.csv")
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
        affected = layer_pairs[layer_pairs["role"].isin(["cascade", "mild"])]
        homog = affected[affected["homogeneous"] == True]["delta_attn"].values  # noqa: E712
        heterog = affected[affected["homogeneous"] == False]["delta_attn"].values  # noqa: E712
        bp = ax.boxplot([homog, heterog], tick_labels=["homogeneous", "heterogeneous"], patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#2ca02c", "#9467bd"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        d = lay["last_layer_attention_map"]["homogeneous_vs_heterogeneous"]["cohens_d"]
        auc = lay["last_layer_attention_map"]["homogeneous_vs_heterogeneous"]["auc"]
        p = lay["last_layer_attention_map"]["homogeneous_vs_heterogeneous"]["p"]
        ax.set_title(f"Last-layer attention map — delta by breakdown_type match (d={d:.2f}, AUC={auc:.2f}, p={p:.2g})")
        ax.set_ylabel("delta_attn (post_dist - pre_dist)")
        ax.grid(True, alpha=0.3)
        pdf.savefig(fig)
        plt.close(fig)


def _effect_row(d: dict) -> list[str]:
    return [f"{d['cohens_d']:.3f}", f"{d['auc']:.3f}", f"{d.get('p', d.get('p_value')):.2g}"]


def _kw_row(d: dict) -> list[str]:
    if "error" in d:
        return ["-", "-", d["error"]]
    means = d.get("means", {})
    means_str = ", ".join(f"{k}={v:.4f}" for k, v in means.items())
    return [f"{d['H']:.3f}", f"{d['p']:.4g}", means_str]


def _sample_separation_pages(pdf: PdfPages, pairs: pd.DataFrame) -> None:
    role_rank = {"cascade": 2, "mild": 1, "unrelated": 0}
    scores = {}
    for case_id, g in pairs.groupby("case_id"):
        affected = g[g["role"].isin(["cascade", "mild"])]
        unrelated = g[g["role"] == "unrelated"]
        if affected.empty or unrelated.empty:
            continue
        scores[case_id] = affected["delta_cls"].mean() - unrelated["delta_cls"].mean()

    if not scores:
        return
    best_case = max(scores, key=scores.get)
    worst_case = min(scores, key=scores.get)

    for case_id, label in [(best_case, "잘 분리된 대표 샘플"), (worst_case, "안 분리된 대표 샘플")]:
        g = pairs[pairs["case_id"] == case_id].sort_values("delta_cls")
        fig, ax = plt.subplots(figsize=PAGE_SIZE)
        colors = [{"cascade": "#d62728", "mild": "#ff7f0e", "unrelated": "#1f77b4"}[r] for r in g["role"]]
        ax.barh([f"ch{c} ({r})" for c, r in zip(g["channel"], g["role"])], g["delta_cls"], color=colors)
        ax.axvline(0, color="gray", linestyle="--")
        ax.set_xlabel("delta_cls (post_dist - pre_dist)")
        ax.set_title(f"{label}: {case_id} (separation score={scores[case_id]:+.4f})")
        pdf.savefig(fig)
        plt.close(fig)


def build_verdict_page(pdf: PdfPages, oracle_run: Path) -> None:
    s9 = json.loads((SANITY9_RUN / "sanity9_summary.json").read_text(encoding="utf-8"))
    oracle_summary = json.loads((oracle_run / "oracle9_summary.json").read_text(encoding="utf-8"))
    dino_summary = json.loads((DINO_RESULTS / "dino_relation9_summary.json").read_text(encoding="utf-8"))

    overlay_acc = s9["by_vis_condition_sample"]["overlay"]["root_accuracy"]
    oracle_acc = oracle_summary["overall_sample"]["root_accuracy"]

    es_path = DINO_RESULTS / "dino_relation9_effect_sizes.json"
    layers_path = DINO_RESULTS / "dino_relation9_layers_summary.json"
    es = json.loads(es_path.read_text(encoding="utf-8")) if es_path.exists() else None
    lay = json.loads(layers_path.read_text(encoding="utf-8")) if layers_path.exists() else None

    lines = [
        f"Oracle-Clean root accuracy = {oracle_acc*100:.1f}%  vs  Overlay = {overlay_acc*100:.1f}%",
        "",
        "-> " + (
            "Oracle이 Overlay보다 유의미하게 높다면, 현재 파이프라인의 병목은 Stage1 시각 인코딩이지 LLM의 "
            "인과 추론 능력 자체가 아니라는 뜻 -> Stage1 이미지 표현을 개선하는 방향이 유효함."
            if oracle_acc > overlay_acc + 0.05 else
            "Oracle이 Overlay와 비슷하거나 낮다면, 문제는 이미지 인코딩이 아니라 LLM의 인과 추론 능력 자체의 "
            "한계일 가능성이 높음 -> Stage1 이미지 개선만으로는 한계가 있고, 추론 절차(예: chain-of-thought, "
            "구조화된 중간 표현) 쪽 개선이 더 유효할 수 있음."
        ),
        "",
    ]

    if es is not None:
        role_d = es["delta_cls"]["role_pairwise"][0]["cohens_d"]
        role_auc = es["delta_cls"]["role_pairwise"][0]["auc"]
        type_d = es["delta_cls"]["type_match_affected_only"]["cohens_d"]
        type_auc = es["delta_cls"]["type_match_affected_only"]["auc"]
        fail_rate = es["failure_rate_cls"]["failure_rate"]
        lines += [
            f"[재검증] p-value 대신 effect size로 보면: role(cascade vs unrelated) d={role_d:.2f}, "
            f"AUC={role_auc:.2f} -> 사실상 노이즈 수준(|d|<0.2, AUC~0.5).",
            f"breakdown_type 일치 여부(homogeneous vs heterogeneous, affected만) d={type_d:.2f}, AUC={type_auc:.2f} "
            f"-> role보다 3~7배 크지만 여전히 small~medium. 샘플 단위 실패율(unrelated가 affected보다 delta 큼) {fail_rate*100:.0f}%.",
            "",
        ]
        if lay is not None:
            attn_d = lay["last_layer_attention_map"]["homogeneous_vs_heterogeneous"]["cohens_d"]
            attn_auc = lay["last_layer_attention_map"]["homogeneous_vs_heterogeneous"]["auc"]
            attn_role_d = lay["last_layer_attention_map"]["cascade_vs_unrelated"]["cohens_d"]
            lines += [
                f"[레이어 비교] mid-layer CLS는 final CLS와 비슷한 수준(role/type-match 모두 개선 없음). 반면 "
                f"last-layer attention map은 role 신호는 여전히 0에 가깝지만(d={attn_role_d:.2f}), breakdown_type "
                f"일치 여부에서는 d={attn_d:.2f}, AUC={attn_auc:.2f}로 large effect (Cohen 기준 |d|>0.8).",
                "",
                "-> 결론 수정: DINO 거리를 'root와의 relation 강도'(role) 신호로 쓰는 것은 근거가 약함(노이즈 수준, "
                "샘플 단위 실패율 40% 안팎). 그러나 last-layer attention map은 '두 채널이 같은 종류의 이상 패턴을 "
                "보이는가'(breakdown_type match)를 매우 강하게 포착함 -> RQ1에서는 DINO 거리를 root-cause 판별용이 "
                "아니라, 채널을 '같은 이상 시그니처끼리' 클러스터링하는 용도로 재정의해서 시도해볼 가치가 있음. "
                "다만 이는 학습 시점에만 알 수 있는 GT(homogeneous)로 검증한 것이라, 실제 추론 시점에 쓸 수 있는 "
                "형태(비지도 클러스터링 등)로 재설계가 필요함.",
            ]
        else:
            lines += [
                "-> role 기반 DINO 거리는 근거가 약함. breakdown_type-match가 상대적으로 나은 신호이지만 "
                "확정하려면 레이어/attention 비교(추가 실험)가 필요.",
            ]
    _text_page(pdf, "종합 판단", lines)


def main() -> None:
    oracle_run = _latest_oracle_run()
    print(f"Using Oracle-9 run: {oracle_run}")

    with PdfPages(OUT_PDF) as pdf:
        _text_page(pdf, "Stage2 사전 진단: Oracle-Clean & DINO Relation Mini Sanity Check", [
            "목적: Stage1(DINOv2 이미지 인코딩) + Stage2(LLM) 2단계 다변량 시계열 이상탐지 파이프라인에서,",
            "Stage2가 다변량 채널 간 root cause/propagation을 얼마나 잘 추론하는지 사전 검증.",
            "",
            "실험 1 (Oracle-Clean): Stage1 이미지를 거치지 않고 GT를 텍스트로 직접 LLM에 제공했을 때 root",
            "cause 추론 성능이 기존 overlay 이미지 방식보다 높은지 확인 -> LLM 추론력의 상한선을 측정.",
            "",
            "실험 2 (DINO Relation Mini Sanity Check): DINO embedding 간 거리가 채널 간 '관계 강도'를",
            "반영하는 신호로 쓸 수 있는지, LLM 호출 없이 로컬 계산만으로 사전 검증.",
        ])
        build_experiment1_pages(pdf, oracle_run)
        build_experiment2_pages(pdf)
        build_verdict_page(pdf, oracle_run)

    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
