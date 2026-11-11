"""
Stage1 개선 시도 A: 주파수(FFT) 기반 스펙트럼 거리를 병렬 신호로 써서, DINOv2가
놓친 GT 채널을 구제하는지 확인 (VLM 콜 없음, DINOv2도 안 씀 -- 순수 FFT라 매우 빠름).

DINOv2는 "모양(shape)"을 보므로 주기성이 깨지는 이상(같은 진폭 범위 안에서
주파수만 바뀌는 경우)을 놓칠 수 있다는 가설. 채널의 test 윈도우 FFT 크기 스펙트럼을
train bank 윈도우들의 평균 스펙트럼과 코사인 거리로 비교, calib로 z-표준화.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "experiments" / "stage1" / "active"))
sys.path.insert(0, str(BASE / "experiments" / "stage2" / "active"))
import colab_multivariate_v2 as cm
from step1v3_dino_graph_smd import load_smd, _centered_window, WIN
from experiment_stage2_k4_adaptive import GT_SEGMENTS_PATH, _bank_and_calib_idx


def spectrum(window_1d):
    w = window_1d - window_1d.mean()
    mag = np.abs(np.fft.rfft(w))
    s = mag.sum()
    return mag / s if s > 1e-9 else mag


def spectral_score(train_ch, window_ch):
    windows = cm.get_windows(train_ch)  # 기본 WINDOW_SIZE/STEP
    bank_idx, calib_idx = _bank_and_calib_idx(len(windows))
    bank_specs = np.array([spectrum(windows[i]) for i in bank_idx])
    bank_mean = bank_specs.mean(axis=0)
    bank_mean_n = bank_mean / (np.linalg.norm(bank_mean) + 1e-9)

    def dist_to_bank(spec):
        s_n = spec / (np.linalg.norm(spec) + 1e-9)
        return 1 - float(np.dot(s_n, bank_mean_n))

    calib_dists = np.array([dist_to_bank(spectrum(windows[i])) for i in calib_idx])
    mu, sigma = calib_dists.mean(), calib_dists.std() + 1e-8

    test_dist = dist_to_bank(spectrum(window_ch))
    return (test_dist - mu) / sigma


def main():
    zs_cache = json.loads((BASE / "experiments" / "results_stage2_k4_adaptive" / "zs_cache.json").read_text(encoding="utf-8"))
    z_loose = norm.ppf(1 - 0.1)
    missed = []
    for sid, d in zs_cache.items():
        entity = d["entity"]
        zs = {int(c): (v if v is not None else -1e9) for c, v in d["zs"].items()}
        gt = set(d["gt"])
        for c in gt:
            if -1e8 < zs[c] <= z_loose:
                missed.append((entity, c, sid, zs[c]))
    targets = missed[:15]
    print(f"파일럿 대상 {len(targets)}개 (앞선 시도들과 동일한 15개)")

    entity_data = {}
    segments_by_id = {}
    for seg in json.loads(GT_SEGMENTS_PATH.read_text(encoding="utf-8")):
        segments_by_id[f"{seg['entity']}_{seg['start']}_{seg['end']}"] = seg

    rescued = 0
    for ti, (entity, c, sid, z_orig) in enumerate(targets):
        seg = segments_by_id[sid]
        cs, ce = seg["start"], seg["end"]
        if entity not in entity_data:
            entity_data[entity] = load_smd(entity)
        train, test = entity_data[entity]
        center = (cs + ce) // 2
        s_, e_ = _centered_window(len(test), center, WIN)
        window = test[s_:e_]

        z_freq = spectral_score(train[:, c], window[:, c])
        rescued_flag = z_freq > z_loose
        if rescued_flag:
            rescued += 1
        print(f"[{ti+1}/{len(targets)}] {entity} ch{c} {sid}: z_원본={z_orig:.3f} "
              f"z_freq={z_freq:.3f} {'구제!' if rescued_flag else ''}", flush=True)

    print(f"\n=== 15개 놓친 채널 중 주파수 신호가 구제한 개수: {rescued}/{len(targets)} "
          f"(z>loose={z_loose:.2f}) ===")


if __name__ == "__main__":
    main()
