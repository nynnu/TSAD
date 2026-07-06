"""
LLM-TSAD (논문1) vs VLM4TS (논문2) 통합 비교 실험 — GPT 버전
- 공통 모델: GPT-4o
- 공통 데이터셋: SMAP, MSL
- 평가 지표: F1-max (alpha=0.1, 0.01, 0.001 중 최대)
- 체크포인트 시스템: 중단 후 이어서 실행 가능

사용법:
    python run_experiment_gpt.py --dataset SMAP --openai_key sk-...
    python run_experiment_gpt.py --dataset SMAP --max_signals 3 --openai_key sk-...
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
RESULTS_DIR = ROOT / "comparison_results_gpt"
RESULTS_DIR.mkdir(exist_ok=True)

CHECKPOINT_FILE  = RESULTS_DIR / "checkpoint.json"
PROGRESS_CSV     = RESULTS_DIR / "results_progress.csv"
SUMMARY_CSV      = RESULTS_DIR / "results_summary.csv"
SCREEN_CACHE_DIR = RESULTS_DIR / "screening_cache"
SCREEN_CACHE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(VLM4TS_DIR / "src"))
sys.path.insert(0, str(LLMTSAD_DIR))

# LLM-TSAD import
_orig_cwd = os.getcwd()
os.chdir(str(LLMTSAD_DIR))
try:
    from neurips_our.AnoAgent import AnoAgent as _AnoAgent
    import openai_api as _openai_api
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

# ── GPT 설정 ───────────────────────────────────────────────────────────────────
GPT_MODEL = "gpt-4o"
ALPHAS    = [0.1, 0.01, 0.001]
SLEEP_SEC = 5  # GPT TPM 제한 대응


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
# 6. VLM4TS 풀 실행 (GPT-4o)
# ══════════════════════════════════════════════════════════════════════════════

def run_vlm4ts(df: pd.DataFrame, alpha: float, openai_key: str, cache_key: str = None) -> np.ndarray:
    from models.vlm4ts_backup import VLM4TS

    detector = VLM4TS(
        vit4ts_params=dict(window_size=240, window_step_ratio=4.0,
                           model_name="ViT-B-16", image_size=(224,224),
                           alpha=alpha, verbose=False),
        alpha=alpha,
        vlm_model=GPT_MODEL,
        api_key=openai_key,
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
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            vit_intervals = pd.DataFrame(cached) if cached else pd.DataFrame(columns=["start", "end", "severity"])
            intervals = detector.verify_intervals(df, vit_intervals)
        else:
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
# 7. LLM-TSAD 실행 (GPT-4o)
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_tsad(df: pd.DataFrame, openai_key: str) -> np.ndarray:
    import torch

    if not _LLMTSAD_OK:
        raise RuntimeError("LLM-TSAD import 실패")

    os.environ["OPENAI_API_KEY"] = openai_key

    values = torch.tensor(df["value"].values.astype(float), dtype=torch.float32)
    agent = _AnoAgent(
        data_name="orion",
        llm_model=GPT_MODEL,
        max_ts_len=2000,
        index_type="number",
        min_acf_period=24,
        value_scale=10,
    )
    pred = agent.inference(
        values,
        anomaly_ratio=0.05,
        use_deseasonal=True,
        use_image=True,
    )
    time.sleep(SLEEP_SEC)
    return (np.array(pred).flatten() > 0).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# 8. F1-max 계산
# ══════════════════════════════════════════════════════════════════════════════

def compute_f1_max(gt: np.ndarray, run_fn, alphas=ALPHAS):
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

def process_signal(dataset, signal, gt_intervals, model_name, openai_key, done):
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
            best, f1s = compute_f1_max(gt, lambda a: run_vit4ts(df, a))
        elif model_name == "VLM4TS":
            ck = f"{dataset}_{signal}"
            best, f1s = compute_f1_max(
                gt, lambda a: run_vlm4ts(df, a, openai_key, cache_key=ck)
            )
            time.sleep(SLEEP_SEC)
        elif model_name == "LLM-TSAD":
            pred = run_llm_tsad(df, openai_key)
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
        print("No results file found.")
        return
    df = pd.read_csv(PROGRESS_CSV)

    df_ok = df[df["status"] == "ok"].copy()
    df_ok = df_ok.sort_values("timestamp").groupby(
        ["dataset", "signal_id", "model"], as_index=False
    ).last()

    summary = (
        df_ok.groupby(["dataset", "model"])
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

    MODEL_ORDER = ["ViT4TS", "VLM4TS", "LLM-TSAD"]
    W = 74

    print(f"\n{'='*W}")
    print(f"  Anomaly Detection Benchmark Results  (LLM: GPT-4o)")
    print(f"{'='*W}")
    print(f"  {'Dataset':<8} {'Model':<12} {'F1-max':>8} {'Prec':>8} {'Rec':>8} {'Affi-F1':>9} {'N':>4}")
    print(f"  {'-'*70}")

    for dataset in sorted(df_ok["dataset"].unique()):
        sub = summary[summary["dataset"] == dataset]
        for model in MODEL_ORDER:
            row = sub[sub["model"] == model]
            if row.empty:
                continue
            r = row.iloc[0]
            print(f"  {r['dataset']:<8} {r['model']:<12} "
                  f"{r['avg_f1_max']:>8.4f} {r['avg_precision']:>8.4f} "
                  f"{r['avg_recall']:>8.4f} {r['avg_affi_f1']:>9.4f} {int(r['n_signals']):>4}")
        print(f"  {'-'*70}")

    print(f"{'='*W}")
    print(f"  Saved: {SUMMARY_CSV}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. 메인
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["SMAP", "MSL"])
    parser.add_argument("--dataset",  type=str, default=None,
                        help="단일 데이터셋 (예: SMAP)")
    parser.add_argument("--models",   nargs="+", default=["ViT4TS", "VLM4TS", "LLM-TSAD"])
    parser.add_argument("--openai_key", default=None,
                        help="OpenAI API 키 (없으면 OPENAI_API_KEY 환경변수 사용)")
    parser.add_argument("--max_signals", type=int, default=None)
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        generate_summary()
        return

    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("오류: OPENAI_API_KEY 환경변수 또는 --openai_key 옵션 필요")
        sys.exit(1)

    datasets = [args.dataset] if args.dataset else args.datasets

    done = load_checkpoint()
    print(f"체크포인트 로드: {len(done)}개 이미 완료\n")

    anomalies = load_anomalies()

    for dataset in datasets:
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
                process_signal(dataset, sig, anomalies[sig], model, openai_key, done)

    print("\n\n실험 완료!")
    generate_summary()


if __name__ == "__main__":
    main()
