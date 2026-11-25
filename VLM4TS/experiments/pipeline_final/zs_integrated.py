"""ZERO-SHOT DINOv2 Stage1 + ONE integrated index-aware VLM call per page (VLM4TS style), SMD. Fixed before any call; run once on dev, once on held.
Zero-shot: nothing from the train split; every statistic is from the test series of the machine.
Stage1 (chosen on dev, analysis/dino_joint_usual.py): zero-shot DINOv2 scores (test-range plots, synthetic-spike heads) -> per-channel usual value (phase median over
  cycles of the test) -> R0 recipe on the DINO residual (Mahalanobis + 3rd channel, 'UZ') -> machine-level decision layer (paper_stage2.decide_incidents).
Stage2: the test timeline cut into 3000-tick pages (every page, so the VLM can add anywhere, as VLM4TS sees the whole series). Per page:
  image  - 8 metrics that deviate most on the page (test-only residual R0, so channels moved by a missed event are on screen), raw values scaled to the test usual
           range (grey band), 300 index points (max-deviation sample of 10 ticks), the SYSTEM row = test-only joint score Z0; Stage1 intervals shaded blue.
  text   - index-aware: Stage1 intervals (index), the 30 most unusual SYSTEM points (index, value), per metric its 6 most deviating points (index, deviation in
           usual-noise units).
  answer - the final intervals of the page (keep / shorten / lengthen / drop / add, all in one list). Unparsable -> the page keeps Stage1.
Edits are counted apart (dropped Stage1 intervals, removed ticks, extension ticks touching a Stage1 interval, new intervals), each with its in-label share.
Usage: python zs_integrated.py --split dev            (images + prompts only, no call)
       python zs_integrated.py --split dev --execute-vlm
       python zs_integrated.py --split dev --evaluate"""
import sys, argparse, base64, hashlib, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; EXP = HERE.parent; sys.path[:0] = [str(EXP/"analysis"), str(EXP/"stage2"), str(EXP/"stage1_final"), str(HERE)]
import torch  # noqa: F401  (torch DLLs before scipy on Windows)
from sklearn.covariance import LedoitWolf
from stage1 import drop_flat, calib, period
from paper_stage2 import decide_incidents
from testonly_stage1 import testonly_score
OUT = HERE/"results/zs_integrated"; DATA = EXP.parent/"mv_data/SMD"; C0 = EXP/"analysis/results/testonly_20261124/cache"
PAGE, NPTS, K = 3000, 300, 8
zs = lambda v: (v-v.mean())/max(v.std(), 1e-9)
PROMPT = """You will see one page of a server machine's monitoring data: a plot and numbers with the same time index (0 to {n}).
Rows: the {k} metrics that deviate most on this page, each scaled to its usual range (grey band = usual range of this machine), and SYSTEM = how unusual the machine is as a whole (about 2.3 or more is rare).
Blue shading = preliminary anomalies from a vision detector. They can be false alarms, too short, too long, or miss an anomaly.

Preliminary intervals (index): {pre}
SYSTEM, most unusual points (index, value): {sysd}
Per metric, most deviating points (index, deviation in units of its usual noise):
{per}

Give the final anomaly intervals on this page: keep, shorten, lengthen or drop the blue ones, and add anomalies they missed.
Reply only with JSON: {{"intervals": [[start, end], ...]}}   (empty list if the page is normal)"""


def maha(U, clip):
    live = U.std(0) > 1e-9; Uc = np.clip(U[:, live], -clip, clip); lw = LedoitWolf().fit(Uc); d = Uc-lw.location_; return np.einsum("ti,ij,tj->t", d, lw.precision_, d)


def stage1(m, te):
    """Zero-shot DINOv2 'UZ' + decision layer (same computation as analysis/dino_joint_usual.py)."""
    T = len(te); q = np.load(C0/f"SMD_{m}_dino_zs.npz"); tr = np.vstack([np.quantile(te, .005, 0), np.quantile(te, .995, 0)])
    X = (calib(drop_flat(q["spike"][:, :T], te, tr))+calib(drop_flat(q["height"][:, :T], te, tr))).T
    med = np.nanmedian(X, 0); med = np.where(np.isfinite(med), med, 0.); X = np.where(np.isfinite(X), X, med); U = np.zeros_like(X)
    for c in range(X.shape[1]):
        x = X[:, c]
        if x.std() < 1e-9: continue
        P = period(te[:, c]) if te[:, c].std() > 1e-9 else None
        u = np.array([np.median(x[k::P]) for k in range(P)])[np.arange(T) % P] if P else np.full(T, np.median(x))
        e = x-u; sc = 1.4826*np.median(np.abs(e-np.median(e))); U[:, c] = e/(sc if sc > 1e-9 else max(e.std(), 1e-6))
    U3 = -np.sort(-np.maximum(U, 0), 1)[:, 2]; Z = (zs(np.log1p(maha(U, 5)))+zs(np.log1p(np.minimum(U3, 20))))/np.sqrt(2)
    return [(int(e["start"]), int(e["end"])) for e in decide_incidents(Z)]


def machines(sp):
    ms = sorted(p.stem for p in (DATA/"test").glob("*.txt") if (p.stem.startswith("machine-1-")) == (sp == "dev"))
    M = {}
    for m in ms:
        y = np.loadtxt(DATA/"test_label"/f"{m}.txt", delimiter=",").astype(bool).ravel(); te = np.loadtxt(DATA/"test"/f"{m}.txt", delimiter=",", ndmin=2)
        R0, Z0 = testonly_score("SMD", m); T = min(len(y), len(te), len(R0)); te, y, R0, Z0 = te[:T], y[:T], R0[:T], Z0[:T]
        lo, hi = np.quantile(te, .005, 0), np.quantile(te, .995, 0); rng = hi-lo; rng[rng < 1e-9] = np.inf
        M[m] = dict(te=te, y=y, R0=np.clip(R0, -20, 20), Z0=Z0, lo=lo, rng=rng, T=T, S1=stage1(m, te))
    return M


def page(mm, s):
    T = mm["T"]; e = min(T, s+PAGE); step = int(np.ceil((e-s)/NPTS)); idx = np.arange(s, e, step); n = len(idx)
    A = np.abs(mm["R0"][s:e]); ch = list(np.argsort(-A.max(0))[:K])
    def samp(v):   # per index point the value of largest magnitude in its step-tick bin (keeps spikes)
        return np.array([v[i:i+step][np.argmax(np.abs(v[i:i+step]))] for i in range(0, e-s, step)])
    rows = [samp(np.clip((mm["te"][s:e, c]-mm["lo"][c])/mm["rng"][c], -1, 3) if np.isfinite(mm["rng"][c]) else np.zeros(e-s)) for c in ch]
    dev = [samp(mm["R0"][s:e, c]) for c in ch]; sysd = np.array([mm["Z0"][s:e][i:i+step].max() for i in range(0, e-s, step)])
    pre = [(max(0, (a-s)//step), min(n-1, (b-1-s)//step)) for a, b in mm["S1"] if a < e and b > s]
    return dict(s=s, e=e, step=step, n=n, ch=ch, rows=rows, dev=dev, sysd=sysd, pre=pre)


def draw(m, pg, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    f, ax = plt.subplots(K+1, 1, figsize=(14, 1.0*(K+1)+1), sharex=True); x = np.arange(pg["n"])
    for i, (c, r) in enumerate(zip(pg["ch"], pg["rows"])):
        ax[i].axhspan(0, 1, color="0.88"); ax[i].plot(x, r, lw=.8, color="k"); ax[i].set_ylim(-1.05, 3.05); ax[i].set_yticks([]); ax[i].set_ylabel(f"metric {c}", rotation=0, ha="right", fontsize=8)
    ax[K].plot(x, pg["sysd"], lw=.9, color="tab:red"); ax[K].axhline(2.33, ls="--", lw=.6, color="0.4"); ax[K].set_ylabel("SYSTEM", rotation=0, ha="right", fontsize=8)
    for a in ax:
        for p, q in pg["pre"]: a.axvspan(p, q+1, color="tab:blue", alpha=.18)
    ax[K].set_xticks(np.arange(0, pg["n"]+1, 25)); ax[K].set_xlim(0, pg["n"]); ax[K].grid(axis="x", lw=.3); f.tight_layout(); f.savefig(path, dpi=90); plt.close(f)
    top = np.sort(np.argsort(-pg["sysd"])[:30])
    per = "\n".join(f"metric {c}: " + ", ".join(f"({i}, {d[i]:.1f})" for i in np.sort(np.argsort(-np.abs(d))[:6])) for c, d in zip(pg["ch"], pg["dev"]))
    return PROMPT.format(n=pg["n"]-1, k=K, pre=json.dumps([list(map(int, p)) for p in pg["pre"]]), sysd=", ".join(f"({i}, {pg['sysd'][i]:.1f})" for i in top), per=per)


def key(m, s): return f"{m}_{s}"


def run(a):
    call = None
    if a.execute_vlm: from run_stage2_review_hfz_v2 import call
    M = machines(a.split); (OUT/"images").mkdir(parents=True, exist_ok=True); (OUT/"answers").mkdir(parents=True, exist_ok=True); jobs = []
    for m, mm in M.items():
        for s in range(0, mm["T"], PAGE):
            img = OUT/"images"/f"{key(m, s)}.png"; pr = draw(m, page(mm, s), img); (OUT/"images"/f"{key(m, s)}.txt").write_text(pr, encoding="utf8"); jobs.append((img, pr, OUT/"answers"/f"{key(m, s)}.json"))
    def ask(j):
        img, pr, dest = j; fp = hashlib.sha256(img.read_bytes()+pr.encode()).hexdigest()
        if dest.exists() and json.loads(dest.read_text(encoding="utf8")).get("fp") == fp: return
        if call is None: return
        try: raw = call(pr, base64.b64encode(img.read_bytes()).decode(), max_tokens=300)
        except Exception as ex: print("fail", dest.name, str(ex)[:120]); return
        dest.write_text(json.dumps(dict(raw=raw, fp=fp), ensure_ascii=False), encoding="utf8")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as ex: list(ex.map(ask, jobs))
    print(a.split, len(M), "machines", len(jobs), "pages", sum(j[2].exists() for j in jobs), "answered")


def evaluate_all(a):
    from iterate_common import mask, evaluate, G
    from evaluate_v2b_asd import events
    M = machines(a.split); R = {"Stage1 zero-shot DINOv2 (UZ + decision layer)": [], "Stage1 + integrated VLM": []}
    st = dict(pages=0, unparsed=0, dropped=[0, 0], removed=[0, 0], extended=[0, 0], new=[0, 0], new_intervals=[0, 0])
    for m, mm in M.items():
        T, y = mm["T"], mm["y"]; k1 = mask(mm["S1"], T); kf = k1.copy()
        for s in range(0, T, PAGE):
            pg = page(mm, s); st["pages"] += 1; f = OUT/"answers"/f"{key(m, s)}.json"
            try: L = json.loads(re.search(r"\{.*\}", json.loads(f.read_text(encoding="utf8"))["raw"], re.S).group(0))["intervals"]
            except Exception: st["unparsed"] += 1; continue
            kp = np.zeros(pg["e"]-s, bool)
            for iv in L:
                try: i0, i1 = sorted((int(iv[0]), int(iv[1])))
                except Exception: continue
                kp[max(0, i0*pg["step"]):min(pg["e"]-s, (i1+1)*pg["step"])] = True
            kf[s:pg["e"]] = kp
        for a_, b_ in mm["S1"]:
            if not kf[a_:b_].any(): st["dropped"][0] += 1; st["dropped"][1] += int(y[a_:b_].any())
        rm = k1 & ~kf; st["removed"][0] += int(rm.sum()); st["removed"][1] += int((rm & y).sum())
        for p, q in events(kf & ~k1):
            touch = (p > 0 and k1[p-1]) or (q < T and k1[q]); g = "extended" if touch else "new"; st[g][0] += q-p; st[g][1] += int(y[p:q].sum())
            if not touch: st["new_intervals"][0] += 1; st["new_intervals"][1] += int(y[p:q].any())
        for n, kk in zip(R, (k1, kf)):
            e = evaluate(kk, y); e["aff"] = float(G.metric_Affiliation(y.astype(int), mm["Z0"], preds=kk.astype(int))); e["m"] = m; R[n].append(e)
    base = {x["m"]: x["tick_f1"] for x in R["Stage1 zero-shot DINOv2 (UZ + decision layer)"]}
    pan = {n: {f: round(float(np.mean([x[f] for x in v])), 3) for f in ("tick_f1", "int_f1", "int_p", "pa_f1", "aff", "ev_r", "alarm")} | dict(worst=round(min(x["tick_f1"] for x in v), 3),
           WL=[sum(x["tick_f1"] > base[x["m"]]+1e-9 for x in v), sum(x["tick_f1"] < base[x["m"]]-1e-9 for x in v)]) for n, v in R.items()}
    edits = {k: (v if not isinstance(v, list) else dict(n=v[0], in_label=v[1], share=round(v[1]/max(v[0], 1), 3))) for k, v in st.items()}
    out = dict(split=a.split, panel=pan, edits=edits, per_machine={n: {x["m"]: x["tick_f1"] for x in v} for n, v in R.items()}); print(json.dumps({k: out[k] for k in ("panel", "edits")}, indent=1, default=int))
    (OUT/f"evaluation_{a.split}.json").write_text(json.dumps(out, indent=1, default=int))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="dev"); ap.add_argument("--execute-vlm", action="store_true"); ap.add_argument("--evaluate", action="store_true"); a = ap.parse_args()
    evaluate_all(a) if a.evaluate else run(a)
