"""Our zero-shot scores on EXACTLY the TSB-AD-M SMD files (22, Eva 20), with TSB-AD's Standard-F1 protocol (point F1 at the oracle threshold per file, then
averaged). Mapping found by matching the data values: 19 files are whole SMD test series; id_8 = machine-1-3[2500:], id_11 = machine-3-2[1000:], id_21 =
machine-1-4[2500:]. Labels are read from the TSB-AD csv. Our scores are computed on the full SMD test series and sliced (zero-shot statistics therefore see up to
2500 extra leading ticks on 3 files). Usage: python tsbad_smd_eval.py"""
import sys, json, glob, inspect, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"pipeline_final")); sys.argv = sys.argv[:1]
import zs_integrated as Z
from sklearn.metrics import precision_recall_curve
from testonly_stage1 import testonly_score, baseline_testonly
src = inspect.getsource(Z.stage1).replace('return [(int(e["start"]), int(e["end"])) for e in decide_incidents(Z)]', "return Z").replace("def stage1", "def stage1_score")
ns = dict(Z.__dict__); exec(src, ns)
TS = Path(r"C:/Users/김나영/Desktop/TSAD/TSB-AD-data/TSB-AD-M"); D = Z.DATA
MAP = {1: ("machine-1-2", 0), 2: ("machine-3-6", 0), 3: ("machine-2-3", 0), 4: ("machine-3-7", 0), 5: ("machine-3-8", 0), 6: ("machine-2-9", 0), 7: ("machine-2-1", 0),
       8: ("machine-1-3", 2500), 9: ("machine-1-7", 0), 10: ("machine-3-4", 0), 11: ("machine-3-2", 1000), 12: ("machine-3-9", 0), 13: ("machine-2-2", 0),
       14: ("machine-2-6", 0), 15: ("machine-3-1", 0), 16: ("machine-1-1", 0), 17: ("machine-1-5", 0), 18: ("machine-3-11", 0), 19: ("machine-2-7", 0),
       20: ("machine-2-8", 0), 21: ("machine-1-4", 2500), 22: ("machine-3-10", 0)}
def f1max(s, y):
    p, r, _ = precision_recall_curve(y, s); return float(np.max(2*p*r/(p+r+1e-5)))
def _main():
  R = {}; chk = []
  for i, (m, off) in MAP.items():
      f = glob.glob(str(TS/f"*_SMD_id_{i}_*.csv"))[0]; y = pd.read_csv(f).iloc[:, -1].to_numpy().astype(bool); n = len(y)
      ours = np.loadtxt(D/"test_label"/f"{m}.txt", delimiter=",").ravel().astype(bool)[off:off+n]; chk.append(int((ours != y).sum()))
      te = np.loadtxt(D/"test"/f"{m}.txt", delimiter=",", ndmin=2); T = len(te)
      S = {"R0 test-only residual": testonly_score("SMD", m)[1], "zero-shot DINO UZ": ns["stage1_score"](m, te)} | {f"{b} test-only": baseline_testonly("SMD", m, b) for b in ("KNN", "PCA", "IForest")}
      for k, s in S.items(): R.setdefault(k, {})[i] = f1max(np.asarray(s)[off:off+n], y)
      print(i, m, off, n, "label mismatches", chk[-1], {k: round(v[i], 3) for k, v in R.items()}, flush=True)
  eva = [i for i in MAP if i not in (6, 16)]
  out = {k: dict(all22=round(float(np.mean(list(v.values()))), 3), eva20=round(float(np.mean([v[i] for i in eva])), 3)) for k, v in R.items()}
  print(json.dumps(out, indent=1)); json.dump(dict(summary=out, per_file=R, label_mismatch=chk), open(Path(__file__).resolve().parent/"results/resid_dino_20261125/tsbad_smd.json", "w"), indent=1)


if __name__ == "__main__": _main()
