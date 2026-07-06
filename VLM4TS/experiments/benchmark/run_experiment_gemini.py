"""
LLM-TSAD (논문1) vs VLM4TS (논문2) 통합 비교 실험
- 공통 모델: Gemini 2.5 Flash (무료 티어)
- 공통 데이터셋: SMAP, MSL
- 평가 지표: F1-max (alpha=0.1, 0.01, 0.001 중 최대)
- 체크포인트 시스템: 중단 후 이어서 실행 가능

사용법:
    set GEMINI_API_KEY=여기에_Gemini_키
    python run_experiment.py
    python run_experiment.py --dataset SMAP --max_signals 5  # 테스트
"""

import os
import sys
import ast
import json
import time
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
VLM4TS_DIR  = ROOT / "VLM4TS"
LLMTSAD_DIR = ROOT / "LLM-TSAD" / "src"
DATA_DIR    = VLM4TS_DIR / "data"
RESULTS_DIR = ROOT / "comparison_results"
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE  = RESULTS_DIR / "checkpoint.json"
PROGRESS_CSV     = RESULTS_DIR / "results_progress.csv"
SUMMARY_CSV      = RESULTS_DIR / "results_summary.csv"
SCREEN_CACHE_DIR = RESULTS_DIR / "screening_cache"
SCREEN_CACHE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(VLM4TS_DIR / "src"))
sys.path.insert(0, str(LLMTSAD_DIR))  # neurips_our 패키지 경로 추가

# LLM-TSAD는 credentials.yml 위치 때문에 cwd 변경 후 import
_orig_cwd = os.getcwd()
os.chdir(str(LLMTSAD_DIR))
try:
    from neurips_our.AnoAgent import AnoAgent as _AnoAgent
    import openai_api as _openai_api
    # openai_client 패치: env var 우선 사용
    def _patched_openai_client(model, api_key=None, base_url="https://api.openai.com/v1"):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY")
        return OpenAI(api_key=key, base_url=base_url)
    _openai_api.openai_client = _patched_openai_client
    _LLMTSAD_OK = True
except Exception as e:
    print(f"[경고] LLM-TSAD import 실패: {e}")
    _LLMTSAD_OK = False
finally:
    os.chdir(_orig_cwd)

# ── Gemini 설정 ────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-flash-latest"
ALPHAS       = [0.1, 0.01, 0.001]
SLEEP_SEC    = 7  # 무료 10 RPM → 6초, 여유 두어 7초


# ══════════════════════════════════════════════════════════════════════════════
# 1. 체크포인트
# ══════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> set:
    if CHECKPOINT_FILE.exists():
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return set(tuple(x) for x in data.get("done", []))
    return set()

def save_checkpoint(done: set):
    CHECKPOINT_FILE.write_text(
        json.dumps({"done": [list(x) for x in done]}, indent=2),
        encoding="utf-8"
    )

def mark_done(done: set, dataset: str, signal: str, model: str):
    done.add((dataset, signal, model))
    save_checkpoint(done)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 결과 저장
# ══════════════════════════════════════════════════════════════════════════════

_CSV_COLS = [
    "dataset", "signal_id", "model",
    "f1_max", "f1_alpha",
    "f1_01", "f1_001", "f1_0001",
    "precision", "recall",
    "affi_f1", "affi_precision", "affi_recall",
    "status", "timestamp"
]

def append_result(row: dict):
    df = pd.DataFrame([row])
    if PROGRESS_CSV.exists():
        df.to_csv(PROGRESS_CSV, mode="a", header=False, index=False)
    else:
        df.to_csv(PROGRESS_CSV, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 데이터 로드
# ══════════════════════════════════════════════════════════════════════════════

def load_anomalies() -> dict:
    path = DATA_DIR / "anomalies.csv"
    result = {}
    with open(path) as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(",", 1)
        if len(parts) == 2:
            try:
                result[parts[0]] = ast.literal_eval(parts[1].strip('"'))
            except Exception:
                pass
    return result

def load_signal(dataset: str, signal: str) -> pd.DataFrame:
    path = DATA_DIR / dataset / f"{signal}.csv"
    df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
    return df

def timestamps_to_binary(timestamps: np.ndarray, gt_intervals: list) -> np.ndarray:
    gt = np.zeros(len(timestamps), dtype=int)
    for ts_start, ts_end in gt_intervals:
        gt[(timestamps >= ts_start) & (timestamps <= ts_end)] = 1
    return gt

def intervals_to_binary(df: pd.DataFrame, intervals_df: pd.DataFrame) -> np.ndarray:
    ts = df["timestamp"].values
    pred = np.zeros(len(ts), dtype=int)
    for _, row in intervals_df.iterrows():
        pred[(ts >= row["start"]) & (ts <= row["end"])] = 1
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# 4. 평가 지표
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    from affiliation.generics import convert_vector_to_events
    from affiliation.metrics import pr_from_events

    if gt.sum() == 0 and pred.sum() == 0:
        return dict(precision=1.0, recall=1.0, f1=1.0,
                    affi_precision=1.0, affi_recall=1.0, affi_f1=1.0)
    if gt.sum() == 0 or pred.sum() == 0:
        return dict(precision=0.0, recall=0.0, f1=0.0,
                    affi_precision=0.0, affi_recall=0.0, affi_f1=0.0)

    p = precision_score(gt, pred, zero_division=0)
    r = recall_score(gt, pred, zero_division=0)
    f = f1_score(gt, pred, zero_division=0)

    try:
        ep = convert_vector_to_events(pred.tolist())
        eg = convert_vector_to_events(gt.tolist())
        aff = pr_from_events(ep, eg, (0, len(pred)))
        ap, ar = aff["precision"], aff["recall"]
        af = 2*ap*ar/(ap+ar) if (ap+ar) > 0 else 0.0
    except Exception:
        ap = ar = af = 0.0

    return dict(precision=round(p,4), recall=round(r,4), f1=round(f,4),
                affi_precision=round(ap,4), affi_recall=round(ar,4),
                affi_f1=round(af,4))


# ══════════════════════════════════════════════════════════════════════════════
# 5. ViT4TS 실행
# ══════════════════════════════════════════════════════════════════════════════

_vit_cache = {}

def run_vit4ts(df: pd.DataFrame, alpha: float) -> np.ndarray:
    from models.vit4ts import ViT4TS
    if alpha not in _vit_cache:
        _vit_cache[alpha] = ViT4TS(
            window_size=240, window_step_ratio=4.0,
            model_name="ViT-B-16", image_size=(224, 224),
            alpha=alpha, verbose=False
        )
    detector = _vit_cache[alpha]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, timestamps = detector.predict_scores(df)
    intervals = detector.get_intervals(scores, timestamps, alpha=alpha)
    return intervals_to_binary(df, intervals)


# ══════════════════════════════════════════════════════════════════════════════
# 6. VLM4TS 풀 실행 (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

def run_vlm4ts(df: pd.DataFrame, alpha: float, gemini_key: str, cache_key: str = None) -> np.ndarray:
    from models.vlm4ts import VLM4TS
    detector = VLM4TS(
        vit4ts_params=dict(window_size=240, window_step_ratio=4.0,
                           model_name="ViT-B-16", image_size=(224,224),
                           alpha=alpha, verbose=False),
        alpha=alpha,
        vlm_model=GEMINI_MODEL,
        api_key=gemini_key,
        verbose=False,
    )

    # 스크리닝 캐시 경로
    cache_file = None
    if cache_key:
        alpha_str = str(alpha).replace(".", "p")
        cache_file = SCREEN_CACHE_DIR / f"{cache_key}_{alpha_str}.json"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cache_file and cache_file.exists():
            # 캐시된 ViT 스크리닝 결과 불러오기
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            vit_intervals = pd.DataFrame(cached) if cached else pd.DataFrame(columns=["start", "end", "severity"])
            intervals = detector.verify_intervals(df, vit_intervals)
        else:
            # ViT 스크리닝 실행 후 캐시 저장
            vit4ts = detector.vit4ts
            vit4ts.alpha = alpha
            vit_intervals = vit4ts.detect(df)
            if cache_file:
                cache_file.write_text(
                    json.dumps(vit_intervals.to_dict("records") if len(vit_intervals) > 0 else []),
                    encoding="utf-8"
                )
            intervals = detector.verify_intervals(df, vit_intervals)

    return intervals_to_binary(df, intervals)


# ══════════════════════════════════════════════════════════════════════════════
# 7. LLM-TSAD 실행 (Gemini)
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_tsad(df: pd.DataFrame, gemini_key: str) -> np.ndarray:
    import torch

    if not _LLMTSAD_OK:
        raise RuntimeError("LLM-TSAD import 실패")

    # Gemini API 키를 환경변수에 임시 설정
    os.environ["GEMINI_API_KEY"] = gemini_key

    values = torch.tensor(df["value"].values.astype(float), dtype=torch.float32)
    agent = _AnoAgent(
        data_name="orion",
        llm_model=GEMINI_MODEL,
        max_ts_len=2000,
        index_type="number",
        min_acf_period=24,
        value_scale=10,
    )
    # Rate limit: Gemini 호출마다 sleep은 gemini_api.py에서 처리
    pred = agent.inference(
        values,
        anomaly_ratio=0.05,
        use_deseasonal=True,
        use_image=True,
    )
    time.sleep(SLEEP_SEC)  # 슬라이딩 윈도우 마지막 청크 후 대기
    return (np.array(pred).flatten() > 0).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 8. F1-max 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_f1_max(gt: np.ndarray, run_fn, alphas=ALPHAS):
    """
    alphas별로 run_fn(alpha) → pred_binary 실행 후 F1-max 반환.
    run_fn이 None이면 단일 pred를 사용 (LLM-TSAD처럼 alpha 무관한 모델).
    """
    best = {"f1": -1}
    f1_by_alpha = {}
    for a in alphas:
        pred = run_fn(a)
        m = compute_metrics(gt, pred)
        f1_by_alpha[a] = round(m["f1"], 4)
        if m["f1"] > best["f1"]:
            best = {**m, "alpha": a}
    return best, f1_by_alpha


# ══════════════════════════════════════════════════════════════════════════════
# 9. 시그널 하나 처리
# ══════════════════════════════════════════════════════════════════════════════

def process_signal(dataset, signal, gt_intervals, model_name, gemini_key, done):
    """하나의 (dataset, signal, model) 조합 처리."""
    key = (dataset, signal, model_name)
    if key in done:
        print(f"  [SKIP] {signal} / {model_name} — 이미 완료")
        return

    print(f"  [{signal}] {model_name} 실행 중...", end=" ", flush=True)
    df = load_signal(dataset, signal)
    gt = timestamps_to_binary(df["timestamp"].values, gt_intervals)

    if gt.sum() == 0:
        print("GT 없음, 스킵")
        return

    try:
        if model_name == "ViT4TS":
            # alpha 3개 중 F1-max
            best, f1s = compute_f1_max(
                gt,
                lambda a: run_vit4ts(df, a),
            )
        elif model_name == "VLM4TS":
            ck = f"{dataset}_{signal}"
            best, f1s = compute_f1_max(
                gt,
                lambda a: run_vlm4ts(df, a, gemini_key, cache_key=ck),
            )
            time.sleep(SLEEP_SEC)
        elif model_name == "LLM-TSAD":
            # alpha 무관 — 한 번만 실행 후 동일 pred로 3 alpha 계산
            pred = run_llm_tsad(df, gemini_key)
            m = compute_metrics(gt, pred)
            best = {**m, "alpha": "fixed"}
            f1s = {0.1: m["f1"], 0.01: m["f1"], 0.001: m["f1"]}
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print(f"F1-max={best['f1']:.4f}  Affi-F1={best['affi_f1']:.4f}")
        row = {
            "dataset": dataset, "signal_id": signal, "model": model_name,
            "f1_max":   best["f1"],
            "f1_alpha": best.get("alpha"),
            "f1_01":    f1s.get(0.1),
            "f1_001":   f1s.get(0.01),
            "f1_0001":  f1s.get(0.001),
            "precision": best["precision"],
            "recall":    best["recall"],
            "affi_f1":   best["affi_f1"],
            "affi_precision": best["affi_precision"],
            "affi_recall":    best["affi_recall"],
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        print(f"실패: {e}")
        row = {
            "dataset": dataset, "signal_id": signal, "model": model_name,
            "f1_max": None, "f1_alpha": None,
            "f1_01": None, "f1_001": None, "f1_0001": None,
            "precision": None, "recall": None,
            "affi_f1": None, "affi_precision": None, "affi_recall": None,
            "status": f"error: {e}",
            "timestamp": datetime.now().isoformat(),
        }

    append_result(row)
    if row["status"] == "ok":
        mark_done(done, dataset, signal, model_name)


# ══════════════════════════════════════════════════════════════════════════════
# 10. 요약 테이블 생성
# ══════════════════════════════════════════════════════════════════════════════

def generate_summary():
    if not PROGRESS_CSV.exists():
        print("결과 파일 없음")
        return
    df = pd.read_csv(PROGRESS_CSV)
    df = df[df["status"] == "ok"]
    summary = (
        df.groupby(["dataset", "model"])
        .agg(
            avg_f1_max=("f1_max", "mean"),
            avg_precision=("precision", "mean"),
            avg_recall=("recall", "mean"),
            avg_affi_f1=("affi_f1", "mean"),
            n_signals=("signal_id", "count"),
        )
        .reset_index()
        .round(4)
    )
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"\n{'='*70}")
    print("최종 결과 요약")
    print(f"{'='*70}")
    print(f"{'Dataset':<8} {'Model':<12} {'F1-max':>8} {'Affi-F1':>8} {'P':>8} {'R':>8} {'N':>5}")
    print(f"{'-'*70}")
    for _, row in summary.iterrows():
        print(f"{row['dataset']:<8} {row['model']:<12} "
              f"{row['avg_f1_max']:>8.4f} {row['avg_affi_f1']:>8.4f} "
              f"{row['avg_precision']:>8.4f} {row['avg_recall']:>8.4f} "
              f"{int(row['n_signals']):>5}")
    print(f"\n요약 저장: {SUMMARY_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["SMAP", "MSL"])
    parser.add_argument("--models",   nargs="+", default=["ViT4TS", "VLM4TS", "LLM-TSAD"])
    parser.add_argument("--gemini_key", default=None,
                        help="Gemini API 키 (없으면 GEMINI_API_KEY 환경변수 사용)")
    parser.add_argument("--max_signals", type=int, default=None)
    parser.add_argument("--summary_only", action="store_true",
                        help="실험 없이 저장된 결과로 요약만 생성")
    args = parser.parse_args()

    if args.summary_only:
        generate_summary()
        return

    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        print("오류: GEMINI_API_KEY 환경변수 또는 --gemini_key 옵션 필요")
        sys.exit(1)

    done = load_checkpoint()
    print(f"체크포인트 로드: {len(done)}개 이미 완료\n")

    anomalies = load_anomalies()

    for dataset in args.datasets:
        dataset_dir = DATA_DIR / dataset
        if not dataset_dir.exists():
            print(f"[오류] {dataset} 데이터셋 없음: {dataset_dir}")
            continue

        signals = sorted([f.stem for f in dataset_dir.glob("*.csv")])
        if args.max_signals:
            signals = signals[:args.max_signals]

        print(f"\n{'='*60}")
        print(f"데이터셋: {dataset}  ({len(signals)}개 신호)")
        print(f"{'='*60}")

        for sig in signals:
            if sig not in anomalies:
                continue
            print(f"\n[{sig}]")
            for model in args.models:
                process_signal(dataset, sig, anomalies[sig], model, gemini_key, done)

    print("\n\n실험 완료!")
    generate_summary()


if __name__ == "__main__":
    main()
