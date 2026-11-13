"""
CATCH (ICLR 2025, 공식 코드 github.com/decisionintelligence/CATCH) — 학습형 다변량 이상탐지
  서버(entity)마다 train으로 학습 → test에 틱별 점수. 하이퍼파라미터는 공식 스크립트 그대로
  (ASD: scripts/multivariate_detection/detect_score/ASD_dataset_{i}_script/CATCH.sh, SMD: SMD_script/CATCH.sh)
  공식 코드는 겹치지 않는 seq_len 창으로 점수를 내서 마지막 꼬리(< seq_len)는 점수가 없음 → 마지막 점수로 채움
사용: python run_catch.py ASD [omi-1 ...]   /   python run_catch.py SMD [machine-1-1 ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

np.Inf = np.inf  # 공식 코드가 NumPy 1.x의 np.Inf를 씀 (NumPy 2에서 삭제됨) — 코드는 안 고치고 이름만 맞춤

CATCH_DIR = Path("/private/tmp/claude-501/-Users-na-yeonkim-Desktop----------------/0d00be24-9568-4712-be05-36ae303a5dc2/scratchpad/CATCH")
sys.path.insert(0, str(CATCH_DIR))
import os  # noqa: E402
if os.environ.get("CATCH_DEVICE", "mps") == "mps":
    from ts_benchmark.baselines.catch.CATCH_mps import CATCH  # noqa: E402  (공식 CATCH.py에서 장치만 cuda→mps로 바꾼 복사본)
else:
    from ts_benchmark.baselines.catch.CATCH import CATCH  # noqa: E402

HERE = Path(__file__).resolve().parent
MV = HERE.parents[2] / "mv_data"
ASD_HP = {i: None for i in range(1, 13)}


def hparams(ds, e):
    if ds == "ASD":
        sh = CATCH_DIR / "scripts/multivariate_detection/detect_score" / f"ASD_dataset_{e.split('-')[1]}_script" / "CATCH.sh"
    else:
        sh = CATCH_DIR / "scripts/multivariate_detection/detect_score/SMD_script/CATCH.sh"
    txt = sh.read_text()
    return json.loads(txt.split("--model-hyper-params '")[1].split("'")[0])


def run(ds, e, out):
    torch.manual_seed(0); np.random.seed(0)
    tr = np.loadtxt(MV / ds / "train" / f"{e}.txt", delimiter=",")
    te = np.loadtxt(MV / ds / "test" / f"{e}.txt", delimiter=",")
    m = CATCH(**hparams(ds, e))
    m.detect_fit(pd.DataFrame(tr), pd.DataFrame(te))
    s, _ = m.detect_score(pd.DataFrame(te))
    s = np.asarray(s).reshape(-1)
    full = np.full(len(te), s[-1] if len(s) else 0.0, dtype=float)
    full[:len(s)] = s[:len(te)]
    np.savez(out / f"{e}.npz", scores=full, n_scored=len(s))


def main():
    ds = sys.argv[1]
    ents = sys.argv[2:] or ([f"omi-{i}" for i in range(1, 13)] if ds == "ASD"
                            else sorted(p.stem for p in (MV / "SMD" / "test").glob("*.txt")))
    out = HERE / ("catch_scores" if ds == "ASD" else "catch_scores_smd")
    out.mkdir(exist_ok=True)
    for e in ents:
        if (out / f"{e}.npz").exists():
            continue
        t = time.time(); run(ds, e, out)
        print(f"{ds} {e} 완료 {time.time() - t:.0f}초", flush=True)


if __name__ == "__main__":
    main()
