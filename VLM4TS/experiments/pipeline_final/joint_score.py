"""Joint multivariate evidence (added 2026-11-17): how the channels move TOGETHER, which Stage1 (one channel at a time) does not see. Label-free, fitted on the TRAIN residuals only.
Residual r_t = (value - usual value at the same phase) / train residual std, all channels (residual_stage2.usual_value).
  score   : Mahalanobis^2 of r_t under the train residual covariance (Ledoit-Wolf), clipped at +-20 per channel, then a 14-tick maximum filter (one DINO column).
            Large when a COMBINATION of channels is unusual even if every channel is inside its own range.
  pairs   : for a span, the channel pairs whose usual relation is broken most. For a pair (i, j) with train |rho| >= .5: z_t = (r_j - rho r_i) / sqrt(1 - rho^2), judged against the train spread of z at the SAME amplitude
            (6 amplitude bins; the noise of a tight pair grows with the size of the move), strength = max_t |z_t| / spread. Flagged 'unseen' when the span is larger than anything in training. Greedy, a channel in at most two pairs.
  add     : the machine's top ticks of the score, as many as Stage1 flags, become extra candidates, merged with Stage1 episodes by the same rule (neighbours joined while the union stays <= 164 ticks).
Evidence so far (analysis/results/joint_curve_20261117, joint_score_20261117): at equal kept time the score alone beat the Stage1-based final ranking on ASD (tick F1 0.370 vs 0.274 at 4% of the timeline) and tied it on SMD (0.337 vs 0.341) with
a much higher event recall (0.78 vs 0.65); AUROC of Stage1 episodes within machine 0.867 (ASD) / 0.707 (SMD). The merged pool and the assembled structure have NOT been evaluated end to end."""
import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.covariance import LedoitWolf

COLUMN = 14       # one DINO column (stage1.P)
MERGE = 164       # same union limit as contracts.group_episodes
CLIP = 20.
MIN_RHO = .5      # a pair needs a real usual relation (|train correlation| >= this) to be able to break it
NB = 6            # amplitude bins for the pair spread
SCALE_FLOOR = .3


class JointModel:
    def __init__(self, train_resid, pairs=True):
        """train_resid: (ticks, channels) train residuals. Channels without variance are ignored. pairs=False skips the pair statistics (slow for many channels; broken_pairs is then unavailable)."""
        self.C = train_resid.shape[1]; self.keep = np.flatnonzero(train_resid.std(0) > 1e-6)
        X = np.clip(train_resid[:, self.keep], -CLIP, CLIP); lw = LedoitWolf().fit(X); self.loc = lw.location_; self.prec = lw.precision_
        sd = np.sqrt(np.diag(lw.covariance_)); self.rho = lw.covariance_/np.outer(sd, sd); np.fill_diagonal(self.rho, 0.)
        self.q99 = float(np.quantile(maximum_filter1d(self._maha(X), COLUMN), .99))   # level exceeded by 1% of normal TRAIN ticks, same 14-tick filter as score() (a test period is usually higher: the profile is estimated, not exact)
        self.pairs = {}
        if pairs: self._fit_pairs(X-self.loc)

    def _fit_pairs(self, X):
        """Per pair with |rho| >= MIN_RHO: the spread of z = (x_b - rho x_a)/sqrt(1-rho^2) on normal train data in NB amplitude bins (u = larger |x| of the two), so a deviation is judged against the spread at the SAME size
        (the pair relation is noisier when both channels are far from usual), and the largest amplitude seen in training."""
        self.pairs = {}
        for a in range(X.shape[1]):
            for b in range(a+1, X.shape[1]):
                rho = self.rho[a, b]
                if abs(rho) < MIN_RHO: continue
                z = (X[:, b]-rho*X[:, a])/np.sqrt(max(1-rho*rho, 1e-6)); u = np.maximum(np.abs(X[:, a]), np.abs(X[:, b])); edges = np.unique(np.quantile(u, np.linspace(0, 1, NB+1)[1:-1])); k = np.searchsorted(edges, u, side="right")
                sc = np.array([max(1.4826*np.median(np.abs(z[k == i]-np.median(z[k == i]))), SCALE_FLOOR) if (k == i).sum() >= 20 else 1. for i in range(len(edges)+1)]); self.pairs[(a, b)] = (rho, edges, sc, float(u.max()))

    def _maha(self, X):
        X = X-self.loc; return np.einsum("ti,ij,tj->t", X, self.prec, X)

    def score(self, R):
        """Per-tick joint score (Mahalanobis^2) of residuals R (ticks, channels), 14-tick maximum filter applied."""
        return maximum_filter1d(self._maha(np.clip(R[:, self.keep], -CLIP, CLIP)), COLUMN)

    def contributions(self, r):
        """Per-channel share of the Mahalanobis^2 of one residual vector (sums to the score; can be negative where a channel offsets another)."""
        x = np.clip(r[self.keep], -CLIP, CLIP)-self.loc; out = np.zeros(self.C); out[self.keep] = x*(self.prec@x); return out

    def broken_pairs(self, R_span, k=3):
        """[(channel i, channel j, train rho, strength, unseen)] for the k pairs whose usual relation is broken most inside the span (R_span: ticks, channels). strength = max over the span of |z| divided by the train spread of z at the same
        amplitude (units: 'times the usual spread'); unseen = the span reaches an amplitude larger than anything in training (the spread is then extrapolated in proportion). Only pairs with |train rho| >= MIN_RHO; a channel in at most two pairs."""
        X = np.clip(R_span[:, self.keep], -CLIP, CLIP)-self.loc; cand = []
        for (a, b), (rho, edges, sc, umax) in self.pairs.items():
            z = np.abs((X[:, b]-rho*X[:, a])/np.sqrt(max(1-rho*rho, 1e-6))); u = np.maximum(np.abs(X[:, a]), np.abs(X[:, b])); s = np.where(u > umax, sc[-1]*u/umax, sc[np.searchsorted(edges, u, side="right")]); cand.append((float((z/s).max()), a, b, rho, bool((u > umax).any())))
        cand.sort(reverse=True); used = {}; out = []
        for s, a, b, rho, un in cand:
            if used.get(a, 0) >= 2 or used.get(b, 0) >= 2: continue
            used[a] = used.get(a, 0)+1; used[b] = used.get(b, 0)+1; out.append((int(self.keep[a]), int(self.keep[b]), float(rho), s, un))
            if len(out) == k: break
        return out


def segments(mask):
    d = np.diff(np.r_[0, np.asarray(mask).astype(int), 0]); return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def joint_intervals(score, n_ticks):
    """Intervals of the top n_ticks ticks of the joint score (n_ticks = what Stage1 flags, so the extra pool is as large as Stage1's own)."""
    if n_ticks <= 0: return []
    mask = np.zeros(len(score), bool); mask[np.argsort(-score)[:int(n_ticks)]] = True; return segments(mask)


def merge_candidates(stage1_episodes, joint_ivs, merge=MERGE):
    """Stage1 episodes (dicts with start, end, channels) and joint intervals (start, end) into one list sorted by start. Neighbouring items are joined while the union stays <= merge ticks.
    Every result has origin 'stage1', 'joint' or 'both' and keeps the Stage1 channel entries of its Stage1 parts (empty for a joint-only candidate)."""
    items = [dict(start=e["start"], end=e["end"], channels=list(e["channels"]), origin="stage1") for e in stage1_episodes]+[dict(start=int(s), end=int(t), channels=[], origin="joint") for s, t in joint_ivs]
    items.sort(key=lambda e: (e["start"], e["end"])); out = []
    for it in items:
        if out and it["end"]-out[-1]["start"] <= merge:
            cur = out[-1]; cur["end"] = max(cur["end"], it["end"]); cur["channels"] += it["channels"]; cur["origin"] = cur["origin"] if cur["origin"] == it["origin"] else "both"
        else: out.append(dict(it))
    for e in out:   # one entry per channel: keep the strongest Stage1 evidence
        best = {}
        for c in e["channels"]:
            if c["channel"] not in best or (c.get("spike_score") or -1e9) > (best[c["channel"]].get("spike_score") or -1e9): best[c["channel"]] = c
        e["channels"] = sorted(best.values(), key=lambda c: -(c.get("spike_score") if c.get("spike_score") is not None else -1e9))
    return out
