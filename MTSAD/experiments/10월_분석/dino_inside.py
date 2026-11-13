"""
DINOv2 속 들여다보기: 이상 구간이 들어간 그래프 이미지를 넣었을 때
  (1) 입력 이미지  (2) 마지막 레이어 CLS attention  (3) patch 낯섦(train patch와 최근접 거리)  (4) patch PCA 색칠
을 나란히 그린다. 모델은 load_model()/extract()만 바꾸면 DINOv3 등으로 교체 가능.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "24주차_원인분석"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "20주차실험"))
import colab_multivariate_v2 as cm  # noqa: E402
from exp_C_globalnorm import ts_to_image_global  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
SMD = Path(__file__).resolve().parents[2] / "mv_data" / "SMD"
OUT = Path(__file__).resolve().parent
WIN, N_BANK, GRID = 224, 30, 16          # DINOv2 ViT-B/14: 224/14 = 16x16 patch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# (entity, GT 시작, GT 끝, 채널, 설명) -- 서로 다른 모양의 이상
CASES = [
    ("machine-1-6", 22417, 22420, 12, "짧은 spike (3틱, 값은 train 범위 안)"),
    ("machine-1-6", 22252, 22260, 12, "짧은 plateau (8틱)"),
    ("machine-1-8", 15640, 15756, 1, "level shift 116틱 (DINOv2만 잡음)"),
    ("machine-3-5", 1288, 1439, 27, "긴 이상 151틱 (DINOv2만 잡음)"),
    ("machine-3-3", 19349, 19830, 24, "긴 이상 481틱 (셋 다 놓침)"),
]


# ── 모델 (DINOv3로 바꿀 때는 이 두 함수만 교체) ─────────────────────
def load_model():
    m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    return m.to(DEVICE).eval()


def extract(model, imgs):
    """return: cls (N,768), patches (N,256,768), cls_attn (N,256) -- 마지막 레이어, head 평균"""
    buf = {}
    attn = model.blocks[-1].attn
    h = attn.qkv.register_forward_hook(lambda _m, _i, o: buf.__setitem__("qkv", o))
    with torch.no_grad():
        x = torch.stack([cm.dinov2_transform(im) for im in imgs]).to(DEVICE)
        out = model.forward_features(x)
    h.remove()
    B, T, _ = buf["qkv"].shape
    qkv = buf["qkv"].reshape(B, T, 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    a = ((q @ k.transpose(-2, -1)) * attn.scale).softmax(-1)      # (B, heads, T, T)
    cls_attn = a[:, :, 0, 1:].mean(1)                             # CLS -> patch
    return (out["x_norm_clstoken"].cpu().numpy(), out["x_norm_patchtokens"].cpu().numpy(),
            cls_attn.cpu().numpy())


# ── 그리기 ─────────────────────────────────────────────────────────
def up(m):  # 16x16 -> 224x224
    return F.interpolate(torch.tensor(m)[None, None].float(), size=224, mode="nearest")[0, 0].numpy()


def main():
    model = load_model()
    cache = {}
    fig, ax = plt.subplots(len(CASES), 4, figsize=(17, 4.3 * len(CASES)))
    ax = np.atleast_2d(ax)
    for r, (e, a, b, ch, desc) in enumerate(CASES):
        if e not in cache:
            cache[e] = (np.loadtxt(SMD / "train" / f"{e}.txt", delimiter=","),
                        np.loadtxt(SMD / "test" / f"{e}.txt", delimiter=","))
        tr, te = cache[e]
        g_min, g_max = float(tr[:, ch].min()), float(tr[:, ch].max())
        L = b - a
        s = a - 56 if L > WIN else int(a + L / 2 - WIN / 2)
        s = int(np.clip(s, 0, len(te) - WIN))
        img = ts_to_image_global(te[s:s + WIN, ch], g_min, g_max)

        starts = np.arange(0, len(tr) - WIN, WIN)
        bank_idx = np.random.default_rng(0).choice(starts, min(N_BANK, len(starts)), replace=False)
        bank = [ts_to_image_global(tr[i:i + WIN, ch], g_min, g_max) for i in bank_idx]

        _, p_te, att = extract(model, [img])
        _, p_bank, _ = extract(model, bank)
        P = p_te[0] / np.linalg.norm(p_te[0], axis=1, keepdims=True)
        Bk = p_bank.reshape(-1, p_bank.shape[-1])
        Bk = Bk / np.linalg.norm(Bk, axis=1, keepdims=True)
        strange = 1 - (P @ Bk.T).max(1)                            # 최근접 train patch와의 코사인 거리

        pca = PCA(3).fit(np.concatenate([Bk, P]))
        rgb = pca.transform(P)
        rgb = (rgb - rgb.min(0)) / (rgb.max(0) - rgb.min(0) + 1e-8)

        im = np.asarray(img)
        x0, x1 = (a - s) * 223 / (WIN - 1), min((b - s) * 223 / (WIN - 1), 223)
        panels = [("입력 이미지 (빨강=GT)", None),
                  ("Attention (CLS가 보는 곳)", att[0].reshape(GRID, GRID)),
                  ("Patch 낯섦 (train 최근접 거리)", strange.reshape(GRID, GRID)),
                  ("PCA 색칠 (비슷한 patch=비슷한 색)", rgb.reshape(GRID, GRID, 3))]
        for c, (title, m) in enumerate(panels):
            A = ax[r, c]
            A.imshow(im)
            if m is not None and m.ndim == 2:
                hm = A.imshow(up(m), cmap="jet", alpha=0.5)
                plt.colorbar(hm, ax=A, fraction=0.046)
            elif m is not None:
                A.imshow(np.kron(m, np.ones((14, 14, 1))), alpha=0.6)
            A.axvline(x0, color="r", lw=1); A.axvline(max(x1, x0 + 2), color="r", lw=1)
            if c == 0:
                A.axvspan(x0, max(x1, x0 + 2), color="r", alpha=0.25)
                A.set_ylabel(f"{e} ch{ch}\n{desc}", fontsize=9)
            A.set_title(title, fontsize=10); A.set_xticks([]); A.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "dino_inside.png", dpi=90)
    print("saved", OUT / "dino_inside.png")


if __name__ == "__main__":
    main()
