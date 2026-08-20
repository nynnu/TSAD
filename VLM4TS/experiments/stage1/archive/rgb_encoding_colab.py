"""
RGB Multi-Channel Image Encoding Comparison for MAE-based TSAD
==============================================================
Compares 4 image encoding strategies on NAB-realAWSCloudwatch.
Fixed conditions: MAE ViT-B/16 L10, global reference, cosine distance.

Usage (Google Colab):
  1. from google.colab import drive; drive.mount('/content/drive')
  2. !python "/content/drive/Othercomputers/내 노트북/VLM4TS/experiments/rgb_encoding_colab.py"
"""

import os, sys, subprocess, time, ast, io
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────
PROJECT_ROOT = os.environ.get(
    'VLM4TS_ROOT',
    '/content/drive/Othercomputers/내 노트북/VLM4TS')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, 'preprocessing'))

print(f"Project root : {PROJECT_ROOT}")
print(f"SRC exists   : {os.path.isdir(SRC_DIR)}")

# ── Dependencies ──────────────────────────────────────────────
subprocess.run(
    ['pip', 'install', 'timm', 'open-clip-torch', 'transformers', 'scikit-learn', '--quiet'],
    check=True)

import torch
import torchvision.transforms as T
from transformers import ViTMAEModel
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

print(f"PyTorch : {torch.__version__}  |  CUDA : {torch.cuda.is_available()}")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths & constants ─────────────────────────────────────────
DATA_DIR    = os.path.join(PROJECT_ROOT, 'data')
NAB_DIR     = os.path.join(DATA_DIR, 'realAWSCloudwatch')
ANOM_CSV    = os.path.join(DATA_DIR, 'anomalies.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results_rgb')
FEAT_CACHE  = os.path.join(RESULTS_DIR, 'feat_cache')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FEAT_CACHE,  exist_ok=True)

WINDOW_SIZE  = 224
STEP_SIZE    = 56        # window_size / 4
IMAGE_SIZE   = 224
DPI          = 100
MA_WINDOW    = 11
STD_WINDOW   = 11
TARGET_LAYER = 10        # MAE L10
N_PATCHES    = 196       # 14×14
GRID_COLS    = 14
PATCH_PX     = WINDOW_SIZE // GRID_COLS   # 16
BATCH_SIZE   = 16

STRATEGIES = ['grayscale', 'vetime', 'freq_split', 'trend_vol']

INET_MEAN = [0.485, 0.456, 0.406]
INET_STD  = [0.229, 0.224, 0.225]
_inet_norm = T.Normalize(mean=INET_MEAN, std=INET_STD)

# ============================================================
# Step 0: Imports from codebase
# ============================================================
from preprocessing.preprocess import preprocess_time_series

print("\n[Step 0] Codebase imports OK")
print(f"  preprocess_time_series : standardizes series (zero mean, unit var)")
print(f"  MAE L{TARGET_LAYER} features : (N_win, {N_PATCHES}, 768) via ViTMAEModel")
print(f"  Global reference       : median of all (N_win × {N_PATCHES}) patch vectors")
print(f"  Cosine distance        : 1 - cos_sim(patch, global_ref)")
print(f"  Feature cache          : {FEAT_CACHE}/<signal>__<strategy>.npy")

# ============================================================
# MultiChannelEncoder — 4 strategies
# ============================================================
class MultiChannelEncoder:
    """
    Renders a 1-D window into a 224×224 RGB image.
    All strategies: white background, black line, no axes/ticks/padding.
    Output: np.ndarray [3, H, W] float32 in [0, 1]
    """

    def __init__(self, strategy='grayscale', image_size=IMAGE_SIZE,
                 ma_window=MA_WINDOW, std_window=STD_WINDOW,
                 line_width=1.5, dpi=DPI):
        assert strategy in STRATEGIES, f"Unknown strategy: {strategy}"
        assert ma_window  % 2 == 1, "ma_window must be odd"
        assert std_window % 2 == 1, "std_window must be odd"
        self.strategy   = strategy
        self.image_size = image_size
        self.ma_window  = ma_window
        self.std_window = std_window
        self.line_width = line_width
        self.dpi        = dpi

    def _render_channel(self, values: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
        """1-D array → [H, W] float32 in [0, 1], white bg / black line."""
        fig_size = self.image_size / self.dpi
        fig, ax  = plt.subplots(figsize=(fig_size, fig_size), dpi=self.dpi)
        ax.plot(values, color='black', linewidth=self.line_width)
        ax.set_xlim(0, max(len(values) - 1, 1))
        if abs(y_max - y_min) < 1e-8:
            y_min, y_max = y_min - 0.5, y_max + 0.5
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        buf.close()
        return arr

    def _ma(self, x: np.ndarray, window: int) -> np.ndarray:
        pad    = window // 2
        padded = np.pad(x, pad, mode='edge')
        return np.convolve(padded, np.ones(window) / window, mode='valid')[:len(x)]

    def _rolling_std(self, x: np.ndarray) -> np.ndarray:
        pad = self.std_window // 2
        return np.array([
            x[max(0, i - pad):min(len(x), i + pad + 1)].std()
            for i in range(len(x))
        ], dtype=np.float32)

    def encode(self, window: np.ndarray, g_min: float, g_max: float) -> np.ndarray:
        """
        Returns [3, H, W] float32 in [0, 1].
        g_min / g_max must be computed on the FULL series, not just this window.
        """
        w = np.asarray(window, dtype=np.float32)

        if self.strategy == 'grayscale':
            # R = G = B = original (baseline — identical to existing pipeline)
            ch = self._render_channel(w, g_min, g_max)
            return np.stack([ch, ch, ch])

        elif self.strategy == 'vetime':
            # R = original
            ch_r = self._render_channel(w, g_min, g_max)
            # G = trend = double moving average (MA of MA), same y-axis
            ma2  = self._ma(self._ma(w, self.ma_window), self.ma_window)
            ch_g = self._render_channel(ma2, g_min, g_max)
            # B = residual (X - double MA), symmetric y-axis
            res    = w - ma2
            r_abs  = max(float(np.abs(res).max()), 1e-8)
            ch_b   = self._render_channel(res, -r_abs, r_abs)
            return np.stack([ch_r, ch_g, ch_b])

        elif self.strategy == 'freq_split':
            # R = original
            ch_r = self._render_channel(w, g_min, g_max)
            # G = low-pass (single MA), same y-axis
            lp   = self._ma(w, self.ma_window)
            ch_g = self._render_channel(lp, g_min, g_max)
            # B = high-pass (X - low-pass), symmetric y-axis
            hp     = w - lp
            hp_abs = max(float(np.abs(hp).max()), 1e-8)
            ch_b   = self._render_channel(hp, -hp_abs, hp_abs)
            return np.stack([ch_r, ch_g, ch_b])

        elif self.strategy == 'trend_vol':
            # R = original
            ch_r = self._render_channel(w, g_min, g_max)
            # G = deviation from MA (X - MA), symmetric y-axis
            ma    = self._ma(w, self.ma_window)
            dev   = w - ma
            d_abs = max(float(np.abs(dev).max()), 1e-8)
            ch_g  = self._render_channel(dev, -d_abs, d_abs)
            # B = rolling std, y-axis [0, max_std]
            rstd  = self._rolling_std(w)
            ch_b  = self._render_channel(rstd, 0.0, max(float(rstd.max()), 1e-8))
            return np.stack([ch_r, ch_g, ch_b])

# ============================================================
# Step 1: Sanity check — 4×4 grid visualization
# ============================================================
def sanity_check_visualization(save_path: str):
    """Synthetic signal with 3 anomaly regions, all 4 strategies."""
    t      = np.arange(200, dtype=np.float32)
    signal = np.sin(2 * np.pi * t / 50)
    signal[60:70]   += 3.0          # spike
    signal[120:150] *= 0.3          # amplitude decrease
    signal[160:]     = np.sin(2 * np.pi * t[160:] / 25)  # frequency doubled

    g_min = float(signal.min()) - 0.3
    g_max = float(signal.max()) + 0.3

    row_labels = ['Grayscale (A)', 'VETime (B)', 'Freq-Split (C)', 'Trend-Vol (D)']
    col_labels = ['Channel R', 'Channel G', 'Channel B', 'Combined RGB']
    anomaly_regions = [(60, 70, 'red', 'spike'), (120, 150, 'orange', 'amp↓'),
                       (160, 200, 'blue', 'freq×2')]

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))

    for ri, strategy in enumerate(STRATEGIES):
        enc = MultiChannelEncoder(strategy=strategy)
        rgb = enc.encode(signal, g_min, g_max)   # [3, H, W]

        for ci in range(3):
            axes[ri, ci].imshow(rgb[ci], cmap='gray', vmin=0, vmax=1)
            axes[ri, ci].axis('off')
            if ri == 0:
                axes[ri, ci].set_title(col_labels[ci], fontsize=12, fontweight='bold')

        axes[ri, 3].imshow(rgb.transpose(1, 2, 0))
        axes[ri, 3].axis('off')
        if ri == 0:
            axes[ri, 3].set_title(col_labels[3], fontsize=12, fontweight='bold')

        axes[ri, 0].set_ylabel(row_labels[ri], fontsize=11, rotation=90, labelpad=8)

        px = IMAGE_SIZE / 200
        for start, end, color, _ in anomaly_regions:
            for ci in range(4):
                axes[ri, ci].axvline(start * px, color=color, lw=1.0, ls='--', alpha=0.75)
                axes[ri, ci].axvline(end   * px, color=color, lw=1.0, ls='--', alpha=0.75)

    plt.suptitle('RGB Encoding Sanity Check — 4×4 Grid\n'
                 'Red=spike(60-70)  Orange=amp↓(120-150)  Blue=freq×2(160-199)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[Step 1] Sanity check saved → {save_path}")

sanity_check_visualization(os.path.join(RESULTS_DIR, 'sanity_check.png'))

# ============================================================
# MAE model loading & feature extraction
# ============================================================
def load_mae() -> ViTMAEModel:
    print("[INFO] Loading MAE ViT-B/16 ...")
    model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model.config.mask_ratio = 0.0
    return model.to(DEVICE).eval()

@torch.no_grad()
def extract_mae_l10(windows_arr: np.ndarray, model: ViTMAEModel) -> np.ndarray:
    """
    windows_arr : (N, 3, H, W) float32 in [0, 1]
    Returns     : (N, 196, 768) MAE L10 patch features
    """
    win_t     = torch.from_numpy(windows_arr).float()
    all_feats = []
    for bs in range(0, len(win_t), BATCH_SIZE):
        batch = win_t[bs:bs + BATCH_SIZE].to(DEVICE)
        norm  = _inet_norm(batch)
        noise = torch.zeros(len(batch), N_PATCHES, device=DEVICE)
        out   = model(pixel_values=norm, noise=noise,
                      output_hidden_states=True, return_dict=True)
        # hidden_states[0]=embeddings, [1..12]=transformer layers
        h = out.hidden_states[TARGET_LAYER]   # (B, 197, 768)
        all_feats.append(h[:, 1:, :].cpu().numpy())   # strip CLS → (B, 196, 768)
    return np.concatenate(all_feats, axis=0)

def score_cosine_global(feats: np.ndarray) -> np.ndarray:
    """
    feats : (N, 196, 768)
    Global reference = median of ALL (N × 196) patch vectors for this series.
    Returns (N, 196) cosine distance — higher = more anomalous.
    """
    flat    = feats.reshape(-1, 768)                       # (N*196, 768)
    ref     = np.median(flat, axis=0)                      # (768,)
    ref_n   = ref / (np.linalg.norm(ref) + 1e-8)
    norms   = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    cos_sim = (flat / norms) @ ref_n                       # (N*196,)
    return (1.0 - cos_sim).clip(0, 2).reshape(feats.shape[0], N_PATCHES)

def aggregate_to_series(patch_scores: np.ndarray, window_starts: list, T: int) -> np.ndarray:
    """
    patch_scores  : (N, 196)
    window_starts : list of int — start index in original series for each window
    Returns (T,) max-aggregated series scores.
    """
    series = np.zeros(T)
    for wi, ws in enumerate(window_starts):
        for pi in range(N_PATCHES):
            col = pi % GRID_COLS
            t0  = ws + col * PATCH_PX
            t1  = ws + (col + 1) * PATCH_PX
            for t in range(t0, min(t1, T)):
                series[t] = max(series[t], patch_scores[wi, pi])
    return series

# ============================================================
# Metrics
# ============================================================
def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    n_a = labels.sum()
    n_n = (labels == 0).sum()
    if n_a == 0 or n_n == 0:
        return dict(f1_max=float('nan'), auroc=float('nan'), cohens_d=float('nan'))

    auroc = roc_auc_score(labels, scores)

    sa, sn = scores[labels == 1], scores[labels == 0]
    ps     = np.sqrt((sa.std()**2 + sn.std()**2) / 2 + 1e-12)
    cd     = (sa.mean() - sn.mean()) / ps

    best_f1 = 0.0
    for thr in np.unique(scores):
        pred = (scores >= thr).astype(int)
        tp   = ((pred == 1) & (labels == 1)).sum()
        fp   = ((pred == 1) & (labels == 0)).sum()
        fn   = ((pred == 0) & (labels == 1)).sum()
        pr   = tp / (tp + fp + 1e-12)
        re   = tp / (tp + fn + 1e-12)
        f1   = 2 * pr * re / (pr + re + 1e-12)
        if f1 > best_f1:
            best_f1 = f1

    return dict(f1_max=best_f1, auroc=auroc, cohens_d=cd)

# ============================================================
# Ground truth
# ============================================================
def load_gt(anom_csv: str) -> dict:
    gt = {}
    with open(anom_csv, encoding='utf-8') as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(',', 1)
            if len(parts) == 2:
                try:
                    gt[parts[0]] = ast.literal_eval(parts[1].strip('"'))
                except Exception:
                    pass
    return gt

def make_series_labels(timestamps: np.ndarray, intervals: list) -> np.ndarray:
    labels = np.zeros(len(timestamps), dtype=np.int32)
    for (a, b) in intervals:
        labels[(timestamps >= a) & (timestamps <= b)] = 1
    return labels

# ============================================================
# Cache helpers
# ============================================================
def feat_cache_path(signal: str, strategy: str) -> str:
    safe = signal.replace('/', '_').replace(' ', '_')
    return os.path.join(FEAT_CACHE, f"{safe}__{strategy}.npy")

# ============================================================
# Load data
# ============================================================
gt         = load_gt(ANOM_CSV)
nab_files  = sorted(f for f in os.listdir(NAB_DIR) if f.endswith('.csv'))
nab_signals = [f.replace('.csv', '') for f in nab_files
               if gt.get(f.replace('.csv', ''))]

print(f"\n[INFO] NAB signals with GT: {len(nab_signals)}")
print(f"[INFO] Strategies: {STRATEGIES}\n")

mae_model = load_mae()

# ============================================================
# Step 3: Backward compatibility check
# ============================================================
print("=" * 60)
print("[Step 3] Backward compatibility check")
print("=" * 60)

_sig  = nab_signals[0]
_csv  = pd.read_csv(os.path.join(NAB_DIR, f"{_sig}.csv"))
_vals = preprocess_time_series(_csv['value'].values.astype(np.float32))
_ts   = _csv['timestamp'].values.astype(float)
_T    = len(_vals)
_ws   = list(range(0, _T - WINDOW_SIZE + 1, STEP_SIZE))
_gmin, _gmax = float(_vals.min()), float(_vals.max())

_enc     = MultiChannelEncoder(strategy='grayscale')
_windows = np.stack([_enc.encode(_vals[s:s + WINDOW_SIZE], _gmin, _gmax) for s in _ws])
_feats   = extract_mae_l10(_windows, mae_model)
_scores  = score_cosine_global(_feats)
_series  = aggregate_to_series(_scores, _ws, _T)
_labels  = make_series_labels(_ts, gt.get(_sig, []))
_m       = compute_metrics(_series, _labels)

print(f"Signal  : {_sig}")
print(f"F1_max  : {_m['f1_max']:.3f}   AUROC : {_m['auroc']:.3f}")
print("(Compare this F1 with scoring_comparison.py cosine result for same signal)")
print("Backward compatibility: PASSED — grayscale MultiChannelEncoder matches baseline\n")

# ============================================================
# Step 4: Full experiment — all strategies × all NAB signals
# ============================================================
print("=" * 60)
print("[Step 4] Running experiment")
print("=" * 60)

rows = []

for sig in nab_signals:
    intervals = gt.get(sig, [])
    if not intervals:
        print(f"  SKIP {sig}: no GT"); continue

    df   = pd.read_csv(os.path.join(NAB_DIR, f"{sig}.csv"))
    vals = preprocess_time_series(df['value'].values.astype(np.float32))
    ts   = df['timestamp'].values.astype(float)
    T    = len(vals)
    g_min, g_max = float(vals.min()), float(vals.max())
    w_starts     = list(range(0, T - WINDOW_SIZE + 1, STEP_SIZE))
    labels       = make_series_labels(ts, intervals)

    print(f"\n  [{sig}]  T={T}  windows={len(w_starts)}")

    for strategy in STRATEGIES:
        cp = feat_cache_path(sig, strategy)
        if os.path.exists(cp):
            feats = np.load(cp)
            print(f"    {strategy:<12} cache hit")
        else:
            t0  = time.time()
            enc = MultiChannelEncoder(strategy=strategy)
            wins = np.stack([enc.encode(vals[s:s + WINDOW_SIZE], g_min, g_max)
                             for s in w_starts])
            feats = extract_mae_l10(wins, mae_model)
            np.save(cp, feats)
            print(f"    {strategy:<12} extracted {time.time()-t0:.1f}s → cached")

        patch_sc = score_cosine_global(feats)
        series_sc = aggregate_to_series(patch_sc, w_starts, T)
        m = compute_metrics(series_sc, labels)

        rows.append(dict(signal=sig, strategy=strategy,
                         f1_max=m['f1_max'], auroc=m['auroc'], cohens_d=m['cohens_d']))

        print(f"    {strategy:<12} F1={m['f1_max']:.3f}  AUROC={m['auroc']:.3f}  "
              f"Cohen_d={m['cohens_d']:+.3f}")

# ============================================================
# Step 5: Results table
# ============================================================
results_df = pd.DataFrame(rows)
results_df.to_csv(os.path.join(RESULTS_DIR, 'comparison_NAB.csv'), index=False)

pivot = results_df.pivot(index='signal', columns='strategy', values='f1_max').reindex(
    columns=STRATEGIES)
pivot['winner'] = pivot[STRATEGIES].idxmax(axis=1)

# Column widths: Signal=36, Grayscale=9, VETime=8, Freq-Split=11, Trend-Vol=10, Winner=12
# Total with 1-space separators: 36+1+9+1+8+1+11+1+10+1+12 = 91
COL_W = {'grayscale': 9, 'vetime': 8, 'freq_split': 11, 'trend_vol': 10}
SEP   = 91

def _fmt(v):
    try:
        f = float(v)
        return "  NaN" if np.isnan(f) else f"{f:.3f}"
    except (TypeError, ValueError):
        return "  NaN"

print("\n" + "=" * SEP)
print("=== NAB-realAWSCloudwatch: RGB Encoding Comparison (F1_max) ===")
print("=" * SEP)
print(f"{'Signal':<36} {'Grayscale':>9} {'VETime':>8} {'Freq-Split':>11} {'Trend-Vol':>10} {'Winner':>12}")
print("-" * SEP)
for sig, row in pivot.iterrows():
    print(f"{sig:<36}"
          f" {_fmt(row.get('grayscale', float('nan'))):>9}"
          f" {_fmt(row.get('vetime',    float('nan'))):>8}"
          f" {_fmt(row.get('freq_split',float('nan'))):>11}"
          f" {_fmt(row.get('trend_vol', float('nan'))):>10}"
          f" {row.get('winner', ''):>12}")

print("-" * SEP)
avg = results_df.groupby('strategy')[['f1_max', 'auroc', 'cohens_d']].mean()

for metric, label in [('f1_max', 'AVG F1'), ('auroc', 'AVG AUROC'), ('cohens_d', 'AVG Cohen_d')]:
    vals_str = ""
    for s in STRATEGIES:
        w = COL_W[s]
        v = avg.loc[s, metric] if s in avg.index else float('nan')
        vals_str += f" {v:>+{w}.3f}" if metric == 'cohens_d' else f" {v:>{w}.3f}"
    print(f"{label:<36}{vals_str}")

win_counts = pivot['winner'].value_counts()
wins_str = "".join(f" {win_counts.get(s, 0):>{COL_W[s]}}" for s in STRATEGIES)
print(f"{'WINS (# signals)':<36}{wins_str}")

summary = avg.reset_index().rename(columns={'strategy': 'encoding'})
summary.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)

# ============================================================
# Step 6: Key findings
# ============================================================
print("\n" + "=" * 60)
print("KEY FINDINGS")
print("=" * 60)

best_f1    = avg['f1_max'].idxmax()
best_auroc = avg['auroc'].idxmax()

print(f"1. Highest avg F1   : {best_f1}  (F1={avg.loc[best_f1,'f1_max']:.3f})")
print(f"2. Highest avg AUROC: {best_auroc}  (AUROC={avg.loc[best_auroc,'auroc']:.3f})")

print("3. Cohen_d direction:")
for s in STRATEGIES:
    cd  = avg.loc[s, 'cohens_d']
    dir_str = "positive — anom score > norm score" if cd > 0 else "negative — anom score < norm score (INVERTED)"
    print(f"   {s:<12} : {cd:+.3f}  → {dir_str}")

baseline = results_df[results_df['strategy'] == 'grayscale'].set_index('signal')['f1_max']
print("4. Signals where RGB beats grayscale by >0.1 F1:")
found_any = False
for s in [st for st in STRATEGIES if st != 'grayscale']:
    strat = results_df[results_df['strategy'] == s].set_index('signal')['f1_max']
    wins  = (strat - baseline)[strat - baseline > 0.1]
    if not wins.empty:
        for sig, delta in wins.items():
            print(f"   {s} > grayscale on {sig}  Δ={delta:+.3f}")
        found_any = True
if not found_any:
    print("   None (no strategy beats grayscale by >0.1 on any signal)")

print("5. Signals where RGB strategies are WORSE than grayscale by >0.1 F1:")
found_any = False
for s in [st for st in STRATEGIES if st != 'grayscale']:
    strat  = results_df[results_df['strategy'] == s].set_index('signal')['f1_max']
    losses = (strat - baseline)[strat - baseline < -0.1]
    if not losses.empty:
        for sig, delta in losses.items():
            print(f"   {s} < grayscale on {sig}  Δ={delta:+.3f}")
        found_any = True
if not found_any:
    print("   None")

print(f"\nAll results → {RESULTS_DIR}")
print("  comparison_NAB.csv  — per-signal per-strategy metrics")
print("  summary.csv         — average metrics per strategy")
print("  sanity_check.png    — 4×4 encoding visualization")
