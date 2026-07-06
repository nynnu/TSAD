"""Generate overlay plot figures for report."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

out_dir = Path(__file__).parent / 'overlay_figures'
out_dir.mkdir(parents=True, exist_ok=True)
smd = Path(__file__).resolve().parent.parent / 'mv_data' / 'SMD'

COLORS = ['black', 'red', 'blue', 'green']
COLORS_RGB = [(0,0,0), (220,50,50), (50,50,220), (50,180,50)]
W = 224

def get_intervals(labels):
    ivs, in_seg, start = [], False, 0
    for i in range(len(labels)):
        if labels[i] and not in_seg:
            start, in_seg = i, True
        elif not labels[i] and in_seg:
            ivs.append((start, i-1))
            in_seg = False
    if in_seg:
        ivs.append((start, len(labels)-1))
    return ivs

def get_group(train, C=38):
    corr = np.corrcoef(train.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0)
    seed = 22  # high variance channel
    return sorted(range(C), key=lambda i: -np.abs(corr[seed, i]))[:4]

def make_overlay_224(data, start, channels):
    img = Image.new('RGB', (224, 224), 'white')
    pixels = img.load()
    for ci, color in zip(channels, COLORS_RGB):
        window = data[start:start+224, ci]
        w_min, w_max = window.min(), window.max()
        normed = (window - w_min) / (w_max - w_min + 1e-8)
        n = len(normed)
        for i in range(n - 1):
            x0 = int(i * 223 / (n - 1))
            x1 = int((i + 1) * 223 / (n - 1))
            y0 = 223 - int(normed[i] * 219 + 2)
            y1 = 223 - int(normed[i + 1] * 219 + 2)
            y0 = max(0, min(223, y0))
            y1 = max(0, min(223, y1))
            steps = max(abs(x1-x0), abs(y1-y0), 1)
            for s in range(steps + 1):
                t = s / steps
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))
                if 0 <= x < 224 and 0 <= y < 224:
                    pixels[x, y] = color
    return img

for entity in ['machine-1-5', 'machine-1-1']:
    print(f"\n=== {entity} ===")
    test = np.loadtxt(smd / 'test' / f'{entity}.txt', delimiter=',')
    labels = np.loadtxt(smd / 'test_label' / f'{entity}.txt', delimiter=',').astype(int)
    train = np.loadtxt(smd / 'train' / f'{entity}.txt', delimiter=',')

    ivs = get_intervals(labels)
    group = get_group(train)
    print(f"  T={len(test)}, anom={labels.mean():.4f}, group={group}")
    print(f"  Anomaly intervals: {ivs[:5]}")

    normal_start = 1000
    anom_mid = (ivs[0][0] + ivs[0][1]) // 2 if ivs else 5000
    anom_start = max(0, anom_mid - W // 2)

    # Figure 1: matplotlib normal vs anomaly
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, start, title in [(axes[0], normal_start, 'Normal'), (axes[1], anom_start, 'Anomaly')]:
        for ci, color in zip(group, COLORS):
            window = test[start:start+W, ci]
            w_min, w_max = window.min(), window.max()
            normed = (window - w_min) / (w_max - w_min + 1e-8)
            ax.plot(normed, color=color, linewidth=1.2, alpha=0.8, label=f'ch{ci}')
        ax.set_title(f'{title} (t={start}~{start+W})', fontsize=13)
        ax.set_xlim(0, W-1)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlabel('Time step')
        if title == 'Anomaly':
            ax.axvspan(0, W, alpha=0.08, color='red')

    plt.suptitle(f'{entity}: Overlay Plot (channels {group})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / f'{entity}_overlay_compare.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {entity}_overlay_compare.png")

    # Figure 2: DINOv2 224x224 input images
    img_n = make_overlay_224(test, normal_start, group)
    img_a = make_overlay_224(test, anom_start, group)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_n); axes[0].set_title('Normal', fontsize=12); axes[0].axis('off')
    axes[1].imshow(img_a); axes[1].set_title('Anomaly', fontsize=12); axes[1].axis('off')
    plt.suptitle(f'{entity}: DINOv2 Input (224x224)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / f'{entity}_dinov2_input.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {entity}_dinov2_input.png")

    # Figure 3: timeline with anomaly highlight
    fig, ax = plt.subplots(1, 1, figsize=(14, 3.5))
    vs = max(0, anom_start - 800)
    ve = min(len(test), anom_start + W + 800)
    for ci, color in zip(group, COLORS):
        seg = test[vs:ve, ci]
        s_min, s_max = seg.min(), seg.max()
        normed = (seg - s_min) / (s_max - s_min + 1e-8)
        ax.plot(range(vs, ve), normed, color=color, linewidth=0.7, alpha=0.7, label=f'ch{ci}')
    for s, e in ivs:
        if s < ve and e > vs:
            ax.axvspan(max(s, vs), min(e, ve), alpha=0.15, color='red',
                      label='Anomaly GT' if s == ivs[0][0] else '')
    ax.set_title(f'{entity}: Timeline with Anomaly Region', fontsize=13)
    ax.set_xlabel('Time step')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / f'{entity}_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {entity}_timeline.png")

print(f"\nAll saved to: {out_dir}")
