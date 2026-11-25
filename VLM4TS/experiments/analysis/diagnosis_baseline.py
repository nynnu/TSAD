"""Channel diagnosis baseline on the GT events (interpretation_label: root-cause channels per segment, 1-indexed, segment ends inclusive). No VLM. Labels only evaluate.
Rankers (score per channel over the event, larger = more likely root cause), all label-free:
  resid_max / resid_mean : |phase-aligned residual r| (Stage1 residual, per channel, independent)
  raw_z                  : |test - train mean| / train std, max over the event (no daily profile)
  maha_contrib           : per-channel contribution d_i (P d)_i to the Ledoit-Wolf Mahalanobis distance of r, mean over the event (uses the static correlation)
  cond_resid             : channel predicted from all other channels by ridge on train residuals, |error| / train error std, max over the event (relation break)
  cond_x_resid           : rank mean of resid_max and cond_resid
  random                 : expected value of a random ranking
Metrics (OmniAnomaly / InterFusion protocol): HitRate@100% and @150% (top floor(P|G|) channels, share of G found), NDCG@100%/@150%; per anomaly type for ASD (metric / temporal / metric-temporal).
Also: for GT channels missed by resid_max at 100%, their resid_max and how cond_resid ranks them; GT set size distribution (random baseline grows with |G|/C).
Usage: python diagnosis_baseline.py -> results/diagnosis_baseline_20261122/"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/"pipeline_final"), str(ROOT/"stage1_final")]
from residual_stage2 import usual_value
OUT = ROOT/"analysis/results/diagnosis_baseline_20261122"; OUT.mkdir(parents=True, exist_ok=True); DATA = ROOT.parent/"mv_data"; CL = 20


def segments_gt(path):
    out = []
    for line in path.read_text().split():
        rng, ch = line.split(":"); a, b = map(int, rng.split("-")); out.append((a, b+1, [int(c)-1 for c in ch.split(",") if c]))
    return out


def hit_ndcg(score, G, P):
    k = int(np.floor(P*len(G))); top = list(np.argsort(-score, kind="stable")[:k]); Gs = set(G)
    hit = len(Gs & set(top))/len(G); dcg = sum(1/np.log2(i+2) for i, c in enumerate(top) if c in Gs); idcg = sum(1/np.log2(i+2) for i in range(min(k, len(G)))); return hit, dcg/idcg


def random_expect(C, G, P, n=300, rng=np.random.default_rng(0)):
    return np.mean([hit_ndcg(rng.random(C), G, P) for _ in range(n)], 0)


def machine(ds, m):
    tr = np.loadtxt(DATA/ds/"train"/f"{m}.txt", delimiter=",", ndmin=2); te = np.loadtxt(DATA/ds/"test"/f"{m}.txt", delimiter=",", ndmin=2); C = te.shape[1]
    ex, sig, ex_tr = usual_value(tr, len(te), with_train=True); R = np.clip((te-ex)/sig, -CL, CL); Rt = np.clip((tr-ex_tr)/sig, -CL, CL)
    live = tr.std(0) > 1e-6; mu, sd = tr.mean(0), np.maximum(tr.std(0), 1e-6); RZ = np.clip(np.abs(te-mu)/sd, 0, CL)
    Pm = np.zeros((C, C)); Pm[np.ix_(live, live)] = LedoitWolf().fit(Rt[:, live]).precision_
    pred = np.zeros_like(R); err_sd = np.ones(C)
    for c in np.flatnonzero(live):
        o = [j for j in np.flatnonzero(live) if j != c]
        if not o: continue
        rg = Ridge(alpha=10.).fit(Rt[:, o], Rt[:, c]); err_sd[c] = max((Rt[:, c]-rg.predict(Rt[:, o])).std(), 1e-3); pred[:, c] = rg.predict(R[:, o])
    CR = np.abs(R-pred)/err_sd; CR[:, ~live] = 0; out = []
    for a, b, G in segments_gt(DATA/ds/"interpretation_label"/f"{m}.txt"):
        b = min(b, len(R)); r = R[a:b]; d = r @ Pm
        s = dict(resid_max=np.abs(r).max(0), resid_mean=np.abs(r).mean(0), raw_z=RZ[a:b].max(0), maha_contrib=(r*d).mean(0), cond_resid=CR[a:b].max(0))
        s["cond_x_resid"] = rankdata(s["resid_max"])+rankdata(s["cond_resid"])
        for k in s: s[k] = np.where(live, s[k], -1e9) if k != "cond_x_resid" else np.where(live, s[k], -1e9)
        out.append(dict(m=m, a=a, b=b, G=G, C=C, n_live=int(live.sum()), scores={k: v.tolist() for k, v in s.items()}))
    return out


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    summary = {}; RANK = ["resid_max", "resid_mean", "raw_z", "maha_contrib", "cond_resid", "cond_x_resid"]
    for ds in ("ASD", "SMD"):
        cache = OUT/f"events_{ds}.json"
        if cache.exists(): EV = json.loads(cache.read_text())
        else:
            EV = []
            for f in sorted((DATA/ds/"interpretation_label").glob("*.txt")):
                if f.stem == "anomaly_type": continue
                EV += machine(ds, f.stem); print(ds, f.stem, flush=True)
            cache.write_text(json.dumps(EV))
        types = {}
        if ds == "ASD":
            for line in (DATA/ds/"interpretation_label/anomaly_type.txt").read_text().split()[1:]:
                m, seg, t = line.split(","); a = int(seg.split("-")[0]); types[(m, a)] = t
        res = {}; groups = {"all": EV}
        if types:
            for t in sorted(set(types.values())): groups[t] = [e for e in EV if types.get((e["m"], e["a"])) == t]
        for g, evs in groups.items():
            r = {}
            for P in (1.0, 1.5):
                for k in RANK: v = np.array([hit_ndcg(np.array(e["scores"][k]), e["G"], P) for e in evs]); r[f"{k}@{int(P*100)}"] = [round(float(x), 3) for x in v.mean(0)]
                rv = np.array([random_expect(e["C"], e["G"], P) for e in evs]); r[f"random@{int(P*100)}"] = [round(float(x), 3) for x in rv.mean(0)]
            r["n_events"] = len(evs); r["mean_G_share"] = round(float(np.mean([len(e["G"])/e["C"] for e in evs])), 3); res[g] = r
            print(f"\n=== {ds} {g}: {len(evs)} events, mean |G|/C {r['mean_G_share']}")
            for P in (100, 150): print(f"  @{P}%  " + "  ".join(f"{k} {r[f'{k}@{P}'][0]:.3f}/{r[f'{k}@{P}'][1]:.3f}" for k in RANK+["random"]))
        # missed channels
        miss_r, miss_rank_cond, hit_r = [], [], []
        for e in EV:
            s = np.array(e["scores"]["resid_max"]); k = len(e["G"]); top = set(np.argsort(-s, kind="stable")[:k]); cr = np.array(e["scores"]["cond_resid"]); crank = rankdata(-cr, method="min")
            for c in e["G"]:
                (hit_r if c in top else miss_r).append(s[c])
                if c not in top: miss_rank_cond.append(crank[c] <= k)
        res["missed_gt_channels"] = dict(n=len(miss_r), share_of_gt=round(len(miss_r)/(len(miss_r)+len(hit_r)), 3), resid_max_below_3=round(float(np.mean(np.array(miss_r) < 3)), 3), median_resid_missed=round(float(np.median(miss_r)), 2), median_resid_hit=round(float(np.median(hit_r)), 2), recovered_by_cond_resid_in_top_k=round(float(np.mean(miss_rank_cond)), 3))
        print("  missed GT channels (resid_max @100%):", res["missed_gt_channels"])
        # false positives of resid_max: top-k non-GT channels
        fp_r = [np.array(e["scores"]["resid_max"])[c] for e in EV for c in np.argsort(-np.array(e["scores"]["resid_max"]), kind="stable")[:len(e["G"])] if c not in e["G"]]
        res["top_k_non_gt"] = dict(n=len(fp_r), median_resid=round(float(np.median(fp_r)), 2) if fp_r else None)
        summary[ds] = res
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist([np.log1p(hit_r), np.log1p(miss_r)], bins=30, label=["GT channel found (top |G|)", "GT channel missed"], color=["tab:green", "tab:red"]); ax[0].axvline(np.log1p(3), color="k", ls=":"); ax[0].set(xlabel="log1p max |residual| in the event", title=f"{ds}: GT root-cause channels by residual size"); ax[0].legend(fontsize=8)
        ks = ["all"]+[g for g in groups if g != "all"]; xs = np.arange(len(ks)); w = .8/(len(RANK)+1)
        for i, k in enumerate(RANK+["random"]): ax[1].bar(xs+i*w, [res[g][f"{k}@100"][0] for g in ks], w, label=k)
        ax[1].set_xticks(xs+.4); ax[1].set_xticklabels([f"{g}\n(n={res[g]['n_events']})" for g in ks], fontsize=8); ax[1].set(ylabel="HitRate@100%", title=f"{ds}: diagnosis HitRate@100%"); ax[1].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(OUT/f"diagnosis_{ds}.png", dpi=110); plt.close(fig)
    (OUT/"summary.json").write_text(json.dumps(summary, indent=1), encoding="utf8"); print("saved", OUT)
