"""Evaluation-only annotation contract. Missing interpretation labels are unknown."""
import numpy as np


def read_annotation_contract(data, machine):
    labels = np.loadtxt(data/"test_label"/f"{machine}.txt",delimiter=",").astype(bool).ravel()
    change = np.diff(np.r_[False,labels,False].astype(int))
    events = list(zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)))
    interpretations = []
    path = data/"interpretation_label"/f"{machine}.txt"
    lines = path.read_text(encoding="utf8").splitlines() if path.exists() else []
    for line in lines:
        span, text = line.split(":")
        a,b = map(int,span.split("-"))
        interpretations.append((a,b,{int(x)-1 for x in text.split(",")}))
    result=[]
    for a,b in events:
        overlapping=[r for r in interpretations if max(a,r[0])<min(b,r[1])]
        exact=[r for r in overlapping if (r[0],r[1])==(a,b)]
        status="exact" if len(exact)==1 and len(overlapping)==1 else "mismatched" if overlapping else "missing"
        result.append(dict(start=int(a),end=int(b),annotation_status=status,
                           channels=sorted(exact[0][2]) if status=="exact" else None))
    return result


def channel_truth(events, start, end):
    hit=[e for e in events if max(start,e["start"])<min(end,e["end"])]
    if not hit:
        return set(), "negative_event"
    if any(e["annotation_status"]!="exact" for e in hit):
        return None, "unknown_channel_truth"
    return set().union(*(set(e["channels"]) for e in hit)), "annotated_event"
