"""
Stage1 최종 모델 (나연, 2026-10-21 저장 설정) — Stage2에 넘길 이상 구간 + 채널 후보 + 채널별 확대 위치를 machine별 JSON으로 저장

  이상 구간 = (spike 점수 + 높이 점수) 상위 0.2%  OR  탈주기 점수(채널 최댓값) 상위 1%
  채널 후보 = 그 구간 안에서 (spike 점수 + 높이 점수) > machine 상위 10%  OR  채널 탈주기 점수 > machine 상위 10%
  정답 라벨은 쓰지 않음.

  실행:  python stage1.py --data ../../mv_data/SMD --out output
         python stage1.py --data ../../mv_data/SMD --out output --machines machine-1-1 machine-3-11
  결과:  output/{machine}.json  (형식은 README.md)
  캐시:  cache/{machine}_dino.npz  — DINOv2 점수(spike, height). 가장 오래 걸리는 부분이라 한 번 계산하면 다시 씀
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
WIN, GRID = 224, 16            # 그림 한 장 = 224틱, DINOv2 ViT-B/14 → 16×16 patch
P = WIN // GRID                # 열 하나 = 14틱
ROLL_WIN = 60                  # 탈주기 이동평균 길이
INTERVAL_SPIKE, INTERVAL_DESEASON, CHANNEL_PCT = 0.2, 1.0, 10.0   # 상위 %
ZOOM_HALF = 30                 # 확대 위치 ±30틱 (Stage2 확대 그림 60틱)
MERGE_GAP = 60                 # 이 틱 이하로 떨어진 이상 구간은 하나로 합침 (= 탈주기 이동평균 길이, 확대 그림 폭)


# ── 그림 만들기 ─────────────────────────────────────────────────────
def ts_to_image(x, g_min, g_max, size=224):
    """세로 눈금 g_min~g_max로 선 그래프 그림 (흰 바탕, 검은 선 두께 2)"""
    normed = np.clip((x - g_min) / (g_max - g_min + 1e-8), 0.0, 1.0)
    xs = (np.arange(len(x)) * (size - 1) / (len(x) - 1)).astype(int)
    ys = np.clip(size - 1 - (normed * (size - 5) + 2).astype(int), 0, size - 1)
    img = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(img).line(list(zip(xs.tolist(), ys.tolist())), fill=(0, 0, 0), width=2)
    return img


# ── DINOv2: spike 점수 + 높이 점수 ──────────────────────────────────
def load_dino():
    import torch
    from torchvision import transforms
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False).to(dev).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return m, tf, dev


def patches(dino, imgs):
    import torch
    m, tf, dev = dino
    with torch.no_grad():
        return m.forward_features(torch.stack([tf(im) for im in imgs]).to(dev))["x_norm_patchtokens"].cpu().numpy()   # (N, 256, 768)


def dino_scores(dino, tr, te):
    """채널 × 틱 spike 점수, 높이 점수 (열 14틱마다 같은 값). 그림을 못 그리는 채널은 NaN"""
    sc0, clf = pickle.load(open(HERE / "models" / "spike_direction.pkl", "rb"))
    sc1, reg = pickle.load(open(HERE / "models" / "height_v3.pkl", "rb"))
    T, C = te.shape; nwin = T // WIN
    spike = np.full((C, T), np.nan, np.float32); height = np.full((C, T), np.nan, np.float32)
    for c in range(C):
        g0, g1 = float(tr[:, c].min()), float(tr[:, c].max())
        if g1 - g0 < 1e-6:                     # train 상수 채널
            if np.ptp(te[:, c]) < 1e-6:
                continue                       # test도 상수면 건너뜀
            g0, g1 = 0.0, 1.0                  # test에서 움직이면 0~1 눈금
        imgs = [ts_to_image(te[w * WIN:(w + 1) * WIN, c], min(g0, te[w * WIN:(w + 1) * WIN, c].min()),
                            max(g1, te[w * WIN:(w + 1) * WIN, c].max())) for w in range(nwin)]   # train 범위 넘으면 눈금 넓힘
        for s0 in range(0, nwin, 64):
            for j, V in enumerate(patches(dino, imgs[s0:s0 + 64])):
                w = s0 + j; sl = slice(w * WIN, (w + 1) * WIN)
                spike[c, sl] = np.repeat(clf.decision_function(sc0.transform(V)).reshape(GRID, GRID).max(0), P)   # 열마다 16행 최댓값
                v = V.reshape(GRID, GRID, -1); F = np.concatenate([v.max(0), v.mean(0)], 1)                       # 열 특징 [최댓값, 평균]
                height[c, sl] = np.repeat(reg.predict(sc1.transform(F)), P)
    return spike, height


# ── 보정 ───────────────────────────────────────────────────────────
def drop_flat(raw, te, tr):
    """움직임이 train 범위의 1% 미만인 그림은 점수 없음 (평평한 채널 부풀림 방지)"""
    r = raw.astype(float).copy(); rg = np.ptp(tr, 0); rg = np.where(rg < 1e-6, 1.0, rg)
    for w in range(raw.shape[1] // WIN):
        r[np.ptp(te[w * WIN:(w + 1) * WIN], 0) < 0.01 * rg, w * WIN:(w + 1) * WIN] = np.nan
    return r


def calib(raw):
    """채널별 보정: (점수 − 그 채널 test 중앙값) ÷ 그 채널 test MAD"""
    out = np.full_like(raw, np.nan)
    for c in range(raw.shape[0]):
        v = raw[c, ::P]; v = v[np.isfinite(v)]
        if len(v) < 10:
            continue
        med = np.median(v); out[c] = (raw[c] - med) / max(1.4826 * np.median(np.abs(v - med)), 1e-6)
    return out


# ── 탈주기 ─────────────────────────────────────────────────────────
def period(x, lo=5, hi=1000):
    xc = x - x.mean()
    if xc.std() < 1e-9:
        return None
    f = np.abs(np.fft.rfft(xc)); fr = np.fft.rfftfreq(len(xc)); m = (fr > 1 / hi) & (fr < 1 / lo)
    if not m.any() or f[m].max() < 3 * np.median(f[m]):
        return None                            # 뚜렷한 주기 없음
    return int(np.clip(round(1 / fr[m][np.argmax(f[m])]), lo, hi))


def deseason(tr, te):
    """채널 × 틱: train 주기 패턴을 뺀 잔차의 60틱 이동평균, train 기준 z. 주기 없는 채널은 0"""
    out = np.zeros((te.shape[1], len(te)))
    for c in range(te.shape[1]):
        a, b = tr[:, c], te[:, c]
        if a.std() < 1e-6 or (Pd := period(a)) is None:
            continue
        prof = np.array([a[k::Pd].mean() for k in range(Pd)])
        ra = uniform_filter1d(np.abs(a - prof[np.arange(len(a)) % Pd]), ROLL_WIN)
        rb = uniform_filter1d(np.abs(b - prof[np.arange(len(b)) % Pd]), ROLL_WIN)
        out[c] = (rb - ra.mean()) / max(ra.std(), 1e-6)
    return out


# ── Stage1 ─────────────────────────────────────────────────────────
def segs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0]); return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def merge_close(iv, gap=MERGE_GAP):
    """앞 구간 끝과 다음 구간 시작 사이가 gap틱 이하면 하나로 합침 (한 이상이 점수 1~2열 끊겨 여러 조각으로 나오는 것 방지)"""
    out = []
    for a, b in iv:
        if out and a - out[-1][1] <= gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def run_machine(e, data, cache, dino_fn):
    tr = np.loadtxt(data / "train" / f"{e}.txt", delimiter=","); te = np.loadtxt(data / "test" / f"{e}.txt", delimiter=",")
    f = cache / f"{e}_dino.npz"
    if f.exists():
        d = np.load(f); spike, height = d["spike"], d["height"]
    else:
        spike, height = dino_scores(dino_fn(), tr, te); np.savez(f, spike=spike, height=height)
    T = min(len(te), spike.shape[1]); te = te[:T]
    X = calib(drop_flat(spike[:, :T], te, tr)) + calib(drop_flat(height[:, :T], te, tr))   # spike 점수 + 높이 점수
    D = deseason(tr, te)[:, :T]; Dm = D.max(0)
    fin = np.isfinite(X); Xv = np.where(fin, X, -np.inf)
    ix, idd = np.percentile(X[fin], 100 - INTERVAL_SPIKE), np.percentile(Dm, 100 - INTERVAL_DESEASON)
    mask = (Xv > ix).any(0) | (Dm > idd)
    pool = X[:, ::P]; tx = np.percentile(pool[np.isfinite(pool)], 100 - CHANNEL_PCT); td = np.percentile(D, 100 - CHANNEL_PCT)
    # 평가용 점수 (SOTA 지표): 두 점수를 각자 threshold로 나눠 큰 값 → 1 초과 = 위 규칙과 같음
    tick_score = np.maximum(Xv.max(0) / ix, Dm / idd)                     # (T,)  1 초과 = 이상 구간
    channel_score = np.maximum(Xv / tx, D / td)                           # (C, T) 1 초과 = 채널 후보
    np.savez_compressed(cache.parent / "scores" / f"{e}.npz", tick_score=tick_score.astype(np.float32),
                        channel_score=channel_score.astype(np.float32))
    intervals = []
    for a, b in merge_close(segs(mask)):
        xs, ds = Xv[:, a:b].max(1), D[:, a:b].max(1); chans = []
        for c in np.where((xs > tx) | (ds > td))[0]:
            by_spike = bool(xs[c] > tx); by_des = bool(ds[c] > td)
            # 확대 위치: spike 쪽으로 뽑힌 채널은 점수가 가장 높은 열의 가운데, 탈주기만으로 뽑힌 채널은 탈주기 점수가 가장 높은 틱
            t = a + (int(np.argmax(Xv[c, a:b])) // P * P + P // 2 if by_spike else int(np.argmax(D[c, a:b])))
            t = min(t, b - 1)
            chans.append(dict(channel=int(c), source="both" if by_spike and by_des else "spike" if by_spike else "deseason",
                              spike_score=round(float(xs[c]), 3) if np.isfinite(xs[c]) else None, deseason_score=round(float(ds[c]), 3),
                              zoom_center=int(t), zoom=[max(0, int(t) - ZOOM_HALF), min(T, int(t) + ZOOM_HALF)]))
        chans.sort(key=lambda r: -(r["spike_score"] if r["spike_score"] is not None else -1e9))
        intervals.append(dict(start=int(a), end=int(b), channels=chans))
    return dict(machine=e, length=int(T), n_channels=int(te.shape[1]),
                config=dict(interval=f"(spike+height) top {INTERVAL_SPIKE}% OR deseason top {INTERVAL_DESEASON}%, merged if gap <= {MERGE_GAP}",
                            channel=f"(spike+height) or deseason > machine top {CHANNEL_PCT}%", zoom_half_width=ZOOM_HALF,
                            spike_threshold=round(float(tx), 4), deseason_threshold=round(float(td), 4)),
                intervals=intervals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="train/, test/ 폴더가 있는 데이터 폴더 (예: mv_data/SMD)")
    ap.add_argument("--out", type=Path, default=HERE / "output"); ap.add_argument("--cache", type=Path, default=HERE / "cache")
    ap.add_argument("--machines", nargs="*", help="생략하면 test/ 안의 전부")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True); a.cache.mkdir(parents=True, exist_ok=True)
    (a.cache.parent / "scores").mkdir(exist_ok=True)
    holder = {}
    dino_fn = lambda: holder.setdefault("m", load_dino())
    for e in a.machines or sorted(p.stem for p in (a.data / "test").glob("*.txt")):
        r = run_machine(e, a.data, a.cache, dino_fn)
        (a.out / f"{e}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        print(f"{e}: 구간 {len(r['intervals'])}개, 구간당 채널 평균 {np.mean([len(i['channels']) for i in r['intervals']] or [0]):.1f}개", flush=True)


if __name__ == "__main__":
    main()
