"""
Unified benchmark comparison: LLM-TSAD (논문1) vs VLM4TS (논문2)

공통 데이터셋: VLM4TS의 Orion 데이터셋 (SMAP, MSL 등) — 이미 로컬에 있음
공통 평가 지표: Point-wise Precision / Recall / F1

사용법:
    python compare_benchmark.py --dataset SMAP --model gpt-4o --api_key sk-...
    python compare_benchmark.py --dataset MSL --vit_only  # API 없이 ViT4TS만 비교

환경 변수로 API 키 설정도 가능:
    set OPENAI_API_KEY=sk-...
    python compare_benchmark.py --dataset SMAP
"""

import os
import sys
import ast
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
VLM4TS_DIR = ROOT / "VLM4TS"
LLM_TSAD_DIR = ROOT / "LLM-TSAD" / "src"

sys.path.insert(0, str(VLM4TS_DIR / "src"))
sys.path.insert(0, str(LLM_TSAD_DIR))

# LLM-TSAD의 openai_api.py가 open("credentials.yml")를 import 시점에 실행하므로
# cwd를 LLM-TSAD/src 로 바꾼 후 import 해야 함
_original_cwd = os.getcwd()
os.chdir(str(LLM_TSAD_DIR))
try:
    from neurips_our.AnoAgent import AnoAgent as _AnoAgent
    # openai_api.py가 credentials.yml에서 키를 못 찾을 경우 env var로 fallback하도록 패치
    import openai_api as _openai_api
    _orig_openai_client = _openai_api.openai_client
    def _patched_openai_client(model, api_key=None, base_url="https://api.openai.com/v1"):
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url)
    _openai_api.openai_client = _patched_openai_client
    _LLMTSAD_AVAILABLE = True
except Exception as _e:
    _AnoAgent = None
    _LLMTSAD_AVAILABLE = False
    print(f"[경고] LLM-TSAD import 실패: {_e}")
finally:
    os.chdir(_original_cwd)

DATA_DIR = VLM4TS_DIR / "data"
RESULTS_DIR = ROOT / "comparison_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── 1. 데이터 로드 ─────────────────────────────────────────────────────────────

def load_anomalies():
    """VLM4TS anomalies.csv → {signal_name: [[ts_start, ts_end], ...]}"""
    anomalies_path = DATA_DIR / "anomalies.csv"
    anomalies_dict = {}
    with open(anomalies_path, "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        signal = parts[0]
        try:
            events = ast.literal_eval(parts[1].strip('"'))
            anomalies_dict[signal] = events
        except Exception:
            pass
    return anomalies_dict


def load_signal(dataset_name: str, signal_name: str):
    """
    Returns:
        df : DataFrame with columns [timestamp, value], sorted by timestamp
    """
    path = DATA_DIR / dataset_name / f"{signal_name}.csv"
    df = pd.read_csv(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def timestamps_to_binary(timestamps: np.ndarray, gt_intervals: list) -> np.ndarray:
    """
    타임스탬프 기반 GT 구간 → 인덱스 기반 binary 벡터 변환.

    gt_intervals: [[ts_start, ts_end], ...]  (타임스탬프, 포함 구간)
    timestamps  : 각 row의 Unix timestamp 배열
    반환        : 0/1 numpy array, shape (len(timestamps),)
    """
    gt_binary = np.zeros(len(timestamps), dtype=int)
    for ts_start, ts_end in gt_intervals:
        mask = (timestamps >= ts_start) & (timestamps <= ts_end)
        gt_binary[mask] = 1
    return gt_binary


def intervals_to_binary(df: pd.DataFrame, intervals_df: pd.DataFrame) -> np.ndarray:
    """
    VLM4TS 출력 intervals (start, end 타임스탬프) → binary 벡터.
    intervals_df: DataFrame with columns [start, end]
    """
    timestamps = df["timestamp"].values
    pred_binary = np.zeros(len(timestamps), dtype=int)
    for _, row in intervals_df.iterrows():
        mask = (timestamps >= row["start"]) & (timestamps <= row["end"])
        pred_binary[mask] = 1
    return pred_binary


# ── 2. VLM4TS (ViT4TS stage-1, API 불필요) 실행 ───────────────────────────────

_vit4ts_cache: dict = {}

def get_vit4ts_detector(alpha: float = 0.01):
    """ViT4TS 모델을 한 번만 로드하고 캐시해서 재사용."""
    from models.vit4ts import ViT4TS
    key = alpha
    if key not in _vit4ts_cache:
        _vit4ts_cache[key] = ViT4TS(
            window_size=240,
            window_step_ratio=4.0,
            model_name="ViT-B-16",
            image_size=(224, 224),
            alpha=alpha,
            verbose=False,
        )
    return _vit4ts_cache[key]


def run_vit4ts(df: pd.DataFrame, alpha: float = 0.01):
    """
    Returns:
        pred_binary : 0/1 array aligned with df rows
        intervals_df: DataFrame [start, end, severity]
    """
    detector = get_vit4ts_detector(alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, timestamps = detector.predict_scores(df)
    intervals_df = detector.get_intervals(scores, timestamps, alpha=alpha)
    pred_binary = intervals_to_binary(df, intervals_df)
    return pred_binary, intervals_df


def run_vlm4ts_full(df: pd.DataFrame, alpha: float = 0.01, vlm_model: str = "gpt-4o"):
    """
    VLM4TS 풀 파이프라인: ViT4TS 스크리닝 + VLM 검증
    Returns:
        pred_binary : 0/1 array aligned with df rows
        intervals_df: DataFrame [start, end, severity]
    """
    from models.vlm4ts import VLM4TS

    detector = VLM4TS(
        vit4ts_params={
            "window_size": 240,
            "window_step_ratio": 4.0,
            "model_name": "ViT-B-16",
            "image_size": (224, 224),
            "alpha": alpha,
            "verbose": False,
        },
        alpha=alpha,
        vlm_model=vlm_model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        intervals_df = detector.detect(df)
    pred_binary = intervals_to_binary(df, intervals_df)
    return pred_binary, intervals_df


# ── 3. LLM-TSAD (AnoAgent) 실행 ───────────────────────────────────────────────

def run_llm_tsad(df: pd.DataFrame, model_name: str = "gpt-4o",
                 anomaly_ratio: float = 0.05,
                 use_image: bool = True,
                 use_deseasonal: bool = True):
    """
    Returns:
        pred_binary : 0/1 array aligned with df rows (index 기준)

    LLM-TSAD의 AnoAgent는 인덱스(0-based 정수) 기준으로 예측을 반환함.
    VLM4TS 데이터의 길이가 max_ts_len(2000)을 넘으면 슬라이딩 윈도우 처리됨.
    """
    import torch

    if not _LLMTSAD_AVAILABLE:
        raise RuntimeError("LLM-TSAD import 실패. 위의 경고 메시지를 확인하세요.")

    AnoAgent = _AnoAgent
    values = torch.tensor(df["value"].values.astype(float), dtype=torch.float32)

    agent = AnoAgent(
        data_name="orion",          # 범용 태그
        llm_model=model_name,
        max_ts_len=2000,
        index_type="number",        # 인덱스 기반 프롬프트 (타임스탬프 불필요)
        min_acf_period=24,
        value_scale=10,
    )

    pred_vector = agent.inference(
        values,
        anomaly_ratio=anomaly_ratio,
        use_deseasonal=use_deseasonal,
        use_image=use_image,
    )
    # pred_vector: numpy array of 0/1, length = len(df)
    pred_binary = (np.array(pred_vector).flatten() > 0).astype(int)
    return pred_binary


# ── 4. 공통 평가 ───────────────────────────────────────────────────────────────

from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Point-wise F1 + Affiliation F1."""
    empty = {"precision": 0.0, "recall": 0.0, "f1": 0.0,
             "affi_precision": 0.0, "affi_recall": 0.0, "affi_f1": 0.0}
    perfect = {"precision": 1.0, "recall": 1.0, "f1": 1.0,
               "affi_precision": 1.0, "affi_recall": 1.0, "affi_f1": 1.0}

    if gt.sum() == 0 and pred.sum() == 0:
        return perfect
    if gt.sum() == 0 or pred.sum() == 0:
        return empty

    # Point-wise
    p = precision_score(gt, pred, zero_division=0)
    r = recall_score(gt, pred, zero_division=0)
    f = f1_score(gt, pred, zero_division=0)

    # Affiliation
    try:
        events_pred = convert_vector_to_events(pred.tolist())
        events_gt   = convert_vector_to_events(gt.tolist())
        Trange = (0, len(pred))
        aff = pr_from_events(events_pred, events_gt, Trange)
        affi_p = aff["precision"]
        affi_r = aff["recall"]
        affi_f = 2 * affi_p * affi_r / (affi_p + affi_r) if (affi_p + affi_r) > 0 else 0.0
    except Exception:
        affi_p = affi_r = affi_f = 0.0

    return {
        "precision":      round(p,      4),
        "recall":         round(r,      4),
        "f1":             round(f,      4),
        "affi_precision": round(affi_p, 4),
        "affi_recall":    round(affi_r, 4),
        "affi_f1":        round(affi_f, 4),
    }


# ── 5. 메인 비교 루프 ──────────────────────────────────────────────────────────

def run_comparison(dataset_name: str, model_name: str, alpha: float,
                   vit_only: bool, max_signals: int | None,
                   vlm_model: str = "gpt-4o", run_vlm_full: bool = False):

    print(f"\n{'='*60}")
    print(f"Dataset   : {dataset_name}")
    print(f"Alpha     : {alpha}  (VLM4TS threshold)")
    print(f"LLM-TSAD  : {model_name}  |  vit_only={vit_only}")
    print(f"VLM4TS 풀 : {run_vlm_full}  (vlm_model={vlm_model})")
    print(f"{'='*60}\n")

    anomalies = load_anomalies()
    dataset_dir = DATA_DIR / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"데이터셋 '{dataset_name}' 없음: {dataset_dir}\n"
            "VLM4TS 디렉터리에서 먼저 다운로드하세요:\n"
            "  python VLM4TS/src/preprocessing/download_data.py " + dataset_name
        )

    signals = [f.stem for f in sorted(dataset_dir.glob("*.csv"))]
    if max_signals:
        signals = signals[:max_signals]

    # ── 중간 저장 파일 경로 (재시작 시 이어서 가능) ──
    suffix = "vit_only" if vit_only else model_name.replace("/", "-")
    if run_vlm_full:
        suffix += f"_vlm4ts-{vlm_model}"
    out_path = RESULTS_DIR / f"{dataset_name}_{suffix}_alpha{alpha}.csv"

    # 이미 저장된 결과 로드 (이어서 실행)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done_signals = set(existing["signal"].tolist())
        records = existing.to_dict("records")
        print(f"[재시작] 기존 결과 {len(done_signals)}개 로드: {sorted(done_signals)}")
    else:
        done_signals = set()
        records = []

    for sig in signals:
        if sig in done_signals:
            print(f"[SKIP] {sig} — 이미 완료")
            continue
        if sig not in anomalies:
            print(f"[SKIP] {sig} — ground truth 없음")
            continue

        print(f"[{sig}] 로드 중...", end=" ", flush=True)
        df = load_signal(dataset_name, sig)
        gt_intervals = anomalies[sig]
        gt_binary = timestamps_to_binary(df["timestamp"].values, gt_intervals)

        if gt_binary.sum() == 0:
            print("GT 이상 없음, 스킵")
            continue

        def null_metrics():
            return {k: None for k in
                    ["precision","recall","f1","affi_precision","affi_recall","affi_f1"]}

        # ── ViT4TS (논문2 stage-1) ──
        try:
            pred_vit, _ = run_vit4ts(df, alpha=alpha)
            metrics_vit = compute_metrics(gt_binary, pred_vit)
            print(f"ViT4TS F1={metrics_vit['f1']:.4f}(affi={metrics_vit['affi_f1']:.4f})",
                  end="  |  ")
        except Exception as e:
            print(f"ViT4TS 실패: {e}", end="  |  ")
            metrics_vit = null_metrics()

        # ── VLM4TS 풀 (논문2 stage-1+2, gpt-4o vision) ──
        metrics_vlm = null_metrics()
        if run_vlm_full:
            try:
                pred_vlm, _ = run_vlm4ts_full(df, alpha=alpha, vlm_model=vlm_model)
                metrics_vlm = compute_metrics(gt_binary, pred_vlm)
                print(f"VLM4TS F1={metrics_vlm['f1']:.4f}(affi={metrics_vlm['affi_f1']:.4f})",
                      end="  |  ")
            except Exception as e:
                print(f"VLM4TS 실패: {e}", end="  |  ")

        # ── LLM-TSAD (논문1) ──
        metrics_llm = null_metrics()
        if not vit_only:
            try:
                pred_llm = run_llm_tsad(df, model_name=model_name)
                metrics_llm = compute_metrics(gt_binary, pred_llm)
                print(f"LLM-TSAD F1={metrics_llm['f1']:.4f}(affi={metrics_llm['affi_f1']:.4f})")
            except Exception as e:
                print(f"LLM-TSAD 실패: {e}")
        else:
            print()

        records.append({
            "dataset": dataset_name,
            "signal": sig,
            "n_points": len(df),
            "anomaly_ratio": round(gt_binary.mean(), 4),
            # ViT4TS (논문2 stage-1)
            "vit4ts_f1":             metrics_vit["f1"],
            "vit4ts_affi_f1":        metrics_vit["affi_f1"],
            "vit4ts_precision":      metrics_vit["precision"],
            "vit4ts_recall":         metrics_vit["recall"],
            "vit4ts_affi_precision": metrics_vit["affi_precision"],
            "vit4ts_affi_recall":    metrics_vit["affi_recall"],
            # VLM4TS 풀 (논문2 stage-1+2)
            "vlm4ts_f1":             metrics_vlm["f1"],
            "vlm4ts_affi_f1":        metrics_vlm["affi_f1"],
            "vlm4ts_precision":      metrics_vlm["precision"],
            "vlm4ts_recall":         metrics_vlm["recall"],
            "vlm4ts_affi_precision": metrics_vlm["affi_precision"],
            "vlm4ts_affi_recall":    metrics_vlm["affi_recall"],
            # LLM-TSAD (논문1)
            "llmtsad_f1":             metrics_llm["f1"],
            "llmtsad_affi_f1":        metrics_llm["affi_f1"],
            "llmtsad_precision":      metrics_llm["precision"],
            "llmtsad_recall":         metrics_llm["recall"],
            "llmtsad_affi_precision": metrics_llm["affi_precision"],
            "llmtsad_affi_recall":    metrics_llm["affi_recall"],
        })

        # ── 신호 하나 끝날 때마다 즉시 저장 ──
        pd.DataFrame(records).to_csv(out_path, index=False)
        print(f"  → 저장 완료 ({len(records)}개 누적): {out_path.name}")

    if not records:
        print("결과 없음.")
        return

    df_results = pd.DataFrame(records)

    # ── 요약 출력 ──
    print(f"\n{'='*70}")
    print("평균 성능")
    print(f"{'='*70}")
    print(f"{'모델':<22} {'Point-F1':>10} {'Affi-F1':>10} {'P':>8} {'R':>8} {'Affi-P':>8} {'Affi-R':>8}")
    print(f"{'-'*70}")

    def mean_or_none(col):
        v = df_results[col].dropna()
        return v.mean() if len(v) > 0 else None

    def fmt(v):
        return f"{v:.4f}" if v is not None else "  N/A"

    rows = [
        ("ViT4TS (논문2 1단계)",
         "vit4ts_f1", "vit4ts_affi_f1",
         "vit4ts_precision", "vit4ts_recall",
         "vit4ts_affi_precision", "vit4ts_affi_recall",
         True),
        ("VLM4TS (논문2 풀)",
         "vlm4ts_f1", "vlm4ts_affi_f1",
         "vlm4ts_precision", "vlm4ts_recall",
         "vlm4ts_affi_precision", "vlm4ts_affi_recall",
         run_vlm_full),
        ("LLM-TSAD (논문1)",
         "llmtsad_f1", "llmtsad_affi_f1",
         "llmtsad_precision", "llmtsad_recall",
         "llmtsad_affi_precision", "llmtsad_affi_recall",
         not vit_only),
    ]

    scores = {}
    for name, f1c, affi_f1c, pc, rc, affi_pc, affi_rc, show in rows:
        if not show:
            continue
        f1_v     = mean_or_none(f1c)
        affi_f1v = mean_or_none(affi_f1c)
        p_v      = mean_or_none(pc)
        r_v      = mean_or_none(rc)
        affi_pv  = mean_or_none(affi_pc)
        affi_rv  = mean_or_none(affi_rc)
        print(f"  {name:<20} {fmt(f1_v):>10} {fmt(affi_f1v):>10} "
              f"{fmt(p_v):>8} {fmt(r_v):>8} {fmt(affi_pv):>8} {fmt(affi_rv):>8}")
        scores[name] = (f1_v, affi_f1v)

    # 승자 결정 (affiliation F1 기준)
    if run_vlm_full and not vit_only:
        vlm_affi = scores.get("VLM4TS (논문2 풀)", (None, None))[1]
        llm_affi = scores.get("LLM-TSAD (논문1)", (None, None))[1]
        if vlm_affi is not None and llm_affi is not None:
            winner = "LLM-TSAD (논문1)" if llm_affi > vlm_affi else "VLM4TS (논문2 풀)"
            print(f"\n  → 승자 (Affiliation F1 기준): {winner}")
    elif not vit_only:
        vit_affi = scores.get("ViT4TS (논문2 1단계)", (None, None))[1]
        llm_affi = scores.get("LLM-TSAD (논문1)", (None, None))[1]
        if vit_affi is not None and llm_affi is not None:
            winner = "LLM-TSAD (논문1)" if llm_affi > vit_affi else "ViT4TS (논문2 1단계)"
            print(f"\n  → 승자 (Affiliation F1 기준): {winner}")

    # ── 최종 저장 (이미 루프에서 저장됐지만 한 번 더 확인) ──
    df_results.to_csv(out_path, index=False)
    print(f"\n최종 결과 저장: {out_path}")
    show_cols = ["signal", "vit4ts_f1", "vit4ts_affi_f1"]
    if run_vlm_full:
        show_cols += ["vlm4ts_f1", "vlm4ts_affi_f1"]
    if not vit_only:
        show_cols += ["llmtsad_f1", "llmtsad_affi_f1"]
    print(df_results[show_cols].to_string(index=False))


# ── 6. CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-TSAD vs VLM4TS 통합 벤치마크")
    parser.add_argument("--dataset", default="SMAP",
                        help="데이터셋 이름 (예: SMAP, MSL, realTraffic)")
    parser.add_argument("--model", default="gpt-4o",
                        help="LLM-TSAD에 쓸 LLM 모델 (gpt-4o, gemini-1.5-flash 등)")
    parser.add_argument("--api_key", default=None,
                        help="OpenAI API 키 (없으면 OPENAI_API_KEY 환경 변수 사용)")
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="VLM4TS 이상 탐지 임계값 (기본: 0.01)")
    parser.add_argument("--vit_only", action="store_true",
                        help="API 없이 ViT4TS만 실행 (LLM-TSAD 건너뜀)")
    parser.add_argument("--vlm_full", action="store_true",
                        help="VLM4TS 풀 파이프라인도 실행 (gpt-4o vision 필요)")
    parser.add_argument("--vlm_model", default="gpt-4o",
                        help="VLM4TS에 쓸 vision 모델 (기본: gpt-4o)")
    parser.add_argument("--max_signals", type=int, default=None,
                        help="처리할 최대 신호 수 (테스트용)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # API 키 설정
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if not args.vit_only and not os.environ.get("OPENAI_API_KEY"):
        print("경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("  --api_key 옵션을 사용하거나 --vit_only 플래그로 ViT4TS만 실행하세요.")
        sys.exit(1)

    run_comparison(
        dataset_name=args.dataset,
        vlm_model=args.vlm_model,
        run_vlm_full=args.vlm_full,
        model_name=args.model,
        alpha=args.alpha,
        vit_only=args.vit_only,
        max_signals=args.max_signals,
    )
