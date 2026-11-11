"""
Association Discrepancy (Anomaly Transformer, ICLR 2022) vs CUSUM for
boundary refinement (design doc §7's unexecuted item, audited 2026-08-05).

Per SCHEMA.md's paper-adoption rule: the full paper was read cover-to-cover
(tsad/raw-text/Anomaly Transformer (raw-text).md, 3832 lines, ingested prior
to this session -- see the wiki page's detailed Table 2/4-11/13-14 + Appendix
D-L ablation coverage) and the official GitHub code
(github.com/thuml/Anomaly-Transformer) was cloned and read (model/
AnomalyTransformer.py, model/attn.py, solver.py's train()/test() loss and
scoring logic) before writing this script. Two hardcoded `.cuda()` calls in
model/attn.py (AnomalyAttention.__init__ and .forward) were patched to be
device-agnostic (`.to(sigma.device)`) purely for CPU compatibility -- the
minimax loss, KL-discrepancy formula, and combined test-time energy score
(softmax(-series_loss-prior_loss) * reconstruction_loss, temperature=50) are
otherwise ported EXACTLY as in solver.py's train()/test().

Scope note: the official repo trains ONE model on ALL 28 SMD machines
concatenated (10 epochs, d_model=512, batch=256) -- infeasible on local CPU
within this session's budget. This script trains a reduced-scale model
PER ENTITY (d_model=64, e_layers=2, 3 epochs) on that entity's own train
split only. This is a genuine methodology change (smaller model, fewer
machines, fewer epochs) and is reported as such -- not a full reproduction.

Question: for boundary refinement (Step5), does the Association Discrepancy
signal (jointly modeling all 38 channels, following the paper's own
multivariate design) localize the true anomaly interval better than a
classical two-sided CUSUM statistic (computed per-channel, max-aggregated
over the GT-implicated channels for a comparable multivariate signal)?
Same 9 real SMD segments, same IoU-against-window-relative-true-boundary
metric as §27 (index-aware boundary experiment), win_size=100 (AT
architecture's window size, not 224 -- prior-association distance matrix is
baked to this size at construction).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

AT_REPO = Path(r"C:\Users\김나영\Desktop\TSAD\Anomaly-Transformer")
sys.path.insert(0, str(AT_REPO))
from model.AnomalyTransformer import AnomalyTransformer  # noqa: E402

BASE = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
SMD_DIR = BASE / "mv_data" / "SMD"
RESULTS_DIR = BASE / "experiments" / "results_at_discrepancy_vs_cusum"
CHECKPOINT = RESULTS_DIR / "checkpoint.json"

WIN = 100  # AT's win_size, architecture-constrained
N_CHANNELS = 38
D_MODEL, N_HEADS, E_LAYERS, D_FF = 64, 4, 2, 64
K_WEIGHT = 3
LR = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 32
TEMPERATURE = 50

SEGMENTS = [
    ("machine-1-1", (15849, 16368), [1, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (18071, 18528), [1, 2, 9, 10, 12, 13, 14, 15]),
    ("machine-1-1", (24679, 24682), [9, 13, 14, 15]),
    ("machine-1-2", (4629, 4688), [9, 10, 11, 13, 15, 18]),
    ("machine-1-2", (15925, 15973), [6, 7, 10, 11, 13, 14, 20, 30]),
    ("machine-1-2", (22264, 22336), [1, 2, 3, 4]),
    ("machine-1-5", (10620, 10637), [1, 2, 3, 4, 7, 24, 26, 32]),
    ("machine-1-5", (14068, 14072), [19, 20, 21, 22, 28, 31]),
    ("machine-1-5", (21287, 21298), [1, 2, 3, 4, 6, 7, 24, 26]),
]


def load_smd(entity):
    train = np.loadtxt(SMD_DIR / "train" / f"{entity}.txt", delimiter=",")
    test = np.loadtxt(SMD_DIR / "test" / f"{entity}.txt", delimiter=",")
    return train, test


def _centered_window(arr_len, center, win):
    s = max(0, min(arr_len - win, center - win // 2))
    return s, s + win


def my_kl_loss(p, q):
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def train_at(train_scaled: np.ndarray) -> nn.Module:
    """Faithful port of solver.py's train() minimax loop, reduced scale, CPU."""
    model = AnomalyTransformer(win_size=WIN, enc_in=N_CHANNELS, c_out=N_CHANNELS,
                                d_model=D_MODEL, n_heads=N_HEADS, e_layers=E_LAYERS, d_ff=D_FF,
                                dropout=0.0, activation="gelu", output_attention=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    n_windows = (len(train_scaled) - WIN) // WIN  # non-overlapping, step=win_size (official default)
    windows = np.stack([train_scaled[i * WIN:(i + 1) * WIN] for i in range(n_windows)]).astype(np.float32)

    for epoch in range(NUM_EPOCHS):
        perm = np.random.permutation(n_windows)
        losses = []
        for bstart in range(0, n_windows, BATCH_SIZE):
            idx = perm[bstart:bstart + BATCH_SIZE]
            batch = torch.from_numpy(windows[idx])
            optimizer.zero_grad()
            output, series, prior, _ = model(batch)

            series_loss, prior_loss = 0.0, 0.0
            for u in range(len(prior)):
                norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, WIN)
                series_loss += (torch.mean(my_kl_loss(series[u], norm_prior.detach())) +
                                 torch.mean(my_kl_loss(norm_prior.detach(), series[u])))
                prior_loss += (torch.mean(my_kl_loss(norm_prior, series[u].detach())) +
                                torch.mean(my_kl_loss(series[u].detach(), norm_prior)))
            series_loss /= len(prior)
            prior_loss /= len(prior)

            rec_loss = criterion(output, batch)
            loss1 = rec_loss - K_WEIGHT * series_loss
            loss2 = rec_loss + K_WEIGHT * prior_loss
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            losses.append(rec_loss.item())
        print(f"    epoch {epoch+1}/{NUM_EPOCHS} mean rec_loss={np.mean(losses):.4f}", flush=True)

    model.eval()
    return model


def at_energy(model: nn.Module, window_scaled: np.ndarray) -> np.ndarray:
    """Per-timestep combined energy for one WIN-length window, exactly solver.py's test() formula."""
    criterion = nn.MSELoss(reduce=False)
    x = torch.from_numpy(window_scaled.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        output, series, prior, _ = model(x)
        loss = torch.mean(criterion(x, output), dim=-1)
        series_loss, prior_loss = 0.0, 0.0
        for u in range(len(prior)):
            norm_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, WIN)
            series_loss += my_kl_loss(series[u], norm_prior.detach()) * TEMPERATURE
            prior_loss += my_kl_loss(norm_prior, series[u].detach()) * TEMPERATURE
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = (metric * loss).squeeze(0).numpy()
    return cri


def cusum_stat(channel_series: np.ndarray, mu: float, sigma: float, drift_frac: float = 0.5) -> np.ndarray:
    """Standard two-sided CUSUM, combined |S+|,|S-| per timestep."""
    if sigma < 1e-9:
        return np.zeros(len(channel_series))
    h = drift_frac * sigma
    s_pos, s_neg = 0.0, 0.0
    out = np.zeros(len(channel_series))
    for t, x in enumerate(channel_series):
        s_pos = max(0.0, s_pos + (x - mu) - h)
        s_neg = min(0.0, s_neg + (x - mu) + h)
        out[t] = max(s_pos, -s_neg)
    return out


def detect_interval(score: np.ndarray) -> tuple[int, int]:
    thresh = score.mean() + 2 * score.std()
    above = np.where(score > thresh)[0]
    if len(above) == 0:
        peak = int(np.argmax(score))
        return peak, peak
    return int(above.min()), int(above.max())


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return inter / union if union > 0 else 0.0


def load_checkpoint():
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    entity_models = {}
    entity_scalers = {}
    entity_test = {}

    for entity, (cs, ce), gt_dims in SEGMENTS:
        seg_id = f"{entity}_{cs}_{ce}"
        if seg_id in checkpoint:
            print(f"[SKIP] {seg_id}", flush=True)
            continue

        if entity not in entity_models:
            print(f"[{entity}] loading + training AT (reduced scale, {NUM_EPOCHS} epochs) ...", flush=True)
            train, test = load_smd(entity)
            scaler = StandardScaler().fit(train)
            train_scaled = scaler.transform(train)
            entity_test[entity] = (test, scaler)
            entity_models[entity] = train_at(train_scaled)
            entity_scalers[entity] = scaler
        model = entity_models[entity]
        test, scaler = entity_test[entity]
        test_scaled = scaler.transform(test)

        gt_channels = [d - 1 for d in gt_dims]
        center = (cs + ce) // 2
        s, e = _centered_window(len(test), center, WIN)
        true_start = max(0, cs - s)
        true_end = min(WIN - 1, ce - s)
        if true_start > true_end:
            true_start, true_end = 0, WIN - 1

        # AT: multivariate energy over all 38 channels
        energy = at_energy(model, test_scaled[s:e])
        at_pred = detect_interval(energy)
        at_iou = iou(at_pred, (true_start, true_end))

        # CUSUM: per-channel on GT-implicated channels (raw scale, train stats), max-aggregated
        train_raw2, _ = load_smd(entity)
        cusum_scores = []
        for c in gt_channels:
            mu, sigma = float(train_raw2[:, c].mean()), float(train_raw2[:, c].std())
            cusum_scores.append(cusum_stat(test[s:e, c], mu, sigma))
        cusum_agg = np.max(np.stack(cusum_scores), axis=0)
        cusum_pred = detect_interval(cusum_agg)
        cusum_iou = iou(cusum_pred, (true_start, true_end))

        checkpoint[seg_id] = {
            "entity": entity, "cs": cs, "ce": ce, "true_start": true_start, "true_end": true_end,
            "at_pred": list(at_pred), "at_iou": at_iou,
            "cusum_pred": list(cusum_pred), "cusum_iou": cusum_iou,
        }
        save_checkpoint(checkpoint)
        print(f"[OK] {seg_id}: true=({true_start},{true_end}) AT={at_pred} iou={at_iou:.3f}  "
              f"CUSUM={cusum_pred} iou={cusum_iou:.3f}", flush=True)

    rows = list(checkpoint.values())
    if not rows:
        print("No results.")
        return
    summary = {
        "n": len(rows),
        "at": {
            "mean_iou": float(np.mean([r["at_iou"] for r in rows])),
            "nonzero_iou_rate": float(np.mean([1.0 if r["at_iou"] > 0 else 0.0 for r in rows])),
            "exact_match_rate": float(np.mean([1.0 if r["at_iou"] == 1.0 else 0.0 for r in rows])),
        },
        "cusum": {
            "mean_iou": float(np.mean([r["cusum_iou"] for r in rows])),
            "nonzero_iou_rate": float(np.mean([1.0 if r["cusum_iou"] > 0 else 0.0 for r in rows])),
            "exact_match_rate": float(np.mean([1.0 if r["cusum_iou"] == 1.0 else 0.0 for r in rows])),
        },
    }
    print(f"\n{'='*60}\nASSOCIATION DISCREPANCY vs CUSUM SUMMARY (n={len(rows)})\n{'='*60}")
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    run()
