"""Worker: runs ONE signal and prints JSON result to stdout. Called by orchestrator."""
import ast, base64, io, json, os, re, sys, time, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE         = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS")
OFFICIAL_SRC = Path(r"C:\Users\김나영\Desktop\TSAD\VLM4TS_official\src")
ANOMS_CSV    = BASE / "data/anomalies.csv"
sys.path.insert(0, str(OFFICIAL_SRC))

from openai import OpenAI
API_KEY = os.environ.get("OPENAI_API_KEY", "")
_client = OpenAI(api_key=API_KEY)

def _sig_path(ds, sig):
    if ds == "NAB":
        return BASE / "data/realAWSCloudwatch" / f"{sig}.csv"
    return BASE / "data" / ds / f"{sig}.csv"

def load_signal(ds, sig):
    df = pd.read_csv(_sig_path(ds, sig))
    return df["timestamp"].values.astype(float), df["value"].values.astype(float)

def load_gt_intervals(sig, timestamps):
    anoms = pd.read_csv(ANOMS_CSV)
    row   = anoms[anoms["signal"] == sig]
    if row.empty:
        return []
    events = ast.literal_eval(row.iloc[0]["events"])
    ivs = []
    for ts_s, ts_e in events:
        i_s = int(np.searchsorted(timestamps, ts_s, side="left"))
        i_e = int(np.searchsorted(timestamps, ts_e, side="right") - 1)
        i_s = max(0, min(i_s, len(timestamps) - 1))
        i_e = max(0, min(i_e, len(timestamps) - 1))
        if i_s <= i_e:
            ivs.append((i_s, i_e))
    return ivs

def _ov(a, b):
    return not (a[1] < b[0] or b[1] < a[0])

def f1_official(gt, pred):
    if not gt: return 0., 0., 0.
    TP = sum(sum(1 for g in gt if _ov(d, g)) for d in pred)
    FP = sum(1 for d in pred if not any(_ov(d, g) for g in gt))
    FN = sum(1 for g in gt  if not any(_ov(g, d) for d in pred))
    p = TP/(TP+FP) if (TP+FP) else 0.
    r = TP/(TP+FN) if (TP+FN) else 0.
    return (2*p*r/(p+r) if p+r else 0.), p, r

def f1_fixed(gt, pred):
    if not gt: return 0., 0., 0.
    TP_p = sum(1 for d in pred if any(_ov(d, g) for g in gt))
    TP_g = sum(1 for g in gt  if any(_ov(g, d) for d in pred))
    FP   = sum(1 for d in pred if not any(_ov(d, g) for g in gt))
    FN   = sum(1 for g in gt  if not any(_ov(g, d) for d in pred))
    p = TP_p/(TP_p+FP) if (TP_p+FP) else 0.
    r = TP_g/(TP_g+FN) if (TP_g+FN) else 0.
    return (2*p*r/(p+r) if p+r else 0.), p, r

VLM_PROMPT = """You are an expert in time-series anomaly detection. You will see:
1. A full time-series plot (X=index, Y=value)
2. Preliminary anomaly intervals from a vision model

Your task: refine these proposals — remove false positives, add missed anomalies.

Reply ONLY with JSON:
{"interval_index": [[start, end], ...], "confidence": [1|2|3, ...], "abnormal_description": "..."}
If no anomalies: {"interval_index": [], "confidence": [], "abnormal_description": "none"}
"""

def make_plot(vals):
    fig, ax = plt.subplots(figsize=(12, 3.5), dpi=100)
    ax.plot(np.arange(len(vals)), vals, color="black", lw=0.6)
    ax.set_xlabel("Time"); ax.set_ylabel("Value"); ax.set_title("Time Series")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def stage2(vals, s1_ivs, sleep=4.0, tries=5):
    img = make_plot(vals)
    prompt = VLM_PROMPT + f"\nVision model detected: {s1_ivs}"
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", "detail": "high"}},
    ]
    for attempt in range(tries):
        try:
            time.sleep(sleep)
            resp = _client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.1, max_tokens=400,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
            try: return json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try: return json.loads(m.group(0))
                    except: pass
            return {"interval_index": s1_ivs, "confidence": []}
        except Exception as exc:
            err = str(exc).lower()
            if "rate_limit" in err or "429" in err:
                time.sleep((attempt+1)*30)
            elif "quota" in err:
                return None
            else:
                time.sleep(5)
    return None

if __name__ == "__main__":
    ds, sig = sys.argv[1], sys.argv[2]
    timestamps, vals = load_signal(ds, sig)
    gt_ivs = load_gt_intervals(sig, timestamps)

    # Stage1
    from models.vit4ts import ViT4TS
    df = pd.DataFrame({"timestamp": timestamps, "value": vals})
    det = ViT4TS(alpha=0.01, verbose=False)
    result = det.detect(df)
    s1_ivs = []
    if not result.empty:
        for _, row in result.iterrows():
            s = int(np.searchsorted(timestamps, row["start"], side="left"))
            e = int(np.searchsorted(timestamps, row["end"],   side="right") - 1)
            s = max(0, min(s, len(timestamps)-1))
            e = max(0, min(e, len(timestamps)-1))
            if s <= e: s1_ivs.append([s, e])

    # Stage2
    res = stage2(vals, s1_ivs)
    confirmed = []
    if res:
        for iv in res.get("interval_index", []):
            if isinstance(iv, (list, tuple)) and len(iv)==2:
                s, e = int(iv[0]), int(iv[1])
                s = max(0, min(s, len(timestamps)-1))
                e = max(0, min(e, len(timestamps)-1))
                if s <= e: confirmed.append([s, e])

    f1o, po, ro = f1_official(gt_ivs, confirmed)
    f1f, pf, rf = f1_fixed(gt_ivs, confirmed)
    s1o, *_ = f1_official(gt_ivs, s1_ivs)
    s1f, *_ = f1_fixed(gt_ivs, s1_ivs)

    out = {
        "ds": ds, "sig": sig, "T": len(timestamps), "n_gt": len(gt_ivs),
        "stage1_f1_official": s1o, "stage1_f1_fixed": s1f,
        "stage2_f1_official": f1o, "stage2_f1_fixed": f1f,
        "stage2_p_fixed": pf, "stage2_r_fixed": rf,
        "n_stage1": len(s1_ivs), "n_stage2": len(confirmed),
        "s1_ivs": s1_ivs, "s2_ivs": confirmed,
    }
    print("RESULT:" + json.dumps(out))
