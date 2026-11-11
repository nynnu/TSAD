"""Explicit temporal evidence output, matched index/no-index HFZ images."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_bank_vlm_pilot import ROOT, call, parse_response, save

source=ROOT/"analysis/results/index_aligned_pilot_20261110"
out=ROOT/"analysis/results/index_localization_pilot_20261110"
out.mkdir(parents=True,exist_ok=True)
rows=json.loads((source/"summary.json").read_text(encoding="utf8"))

def run(row):
    page=dict(channels=[dict(channel=x["channel"]) for x in row["index_evidence"]["channels"]],n_channels=38)
    prompt=row["prompt"]+(
        '\nAlso return "evidence_intervals":[{"channel":integer,"start":integer,"end":integer}]. '
        'Give the visually supported time span of each selected anomalous channel, using the 224-tick '
        'tile-relative axis, zero based, end excluded. Use 0 <= start < end <= 224. '
        'Describe visible evidence rather than copying the full candidate or zoom boundaries. '
        'Each selected channel needs at least one evidence interval. Empty selection requires an empty list. '
        'These are evidence spans, not causal claims. Keep the other JSON fields.')
    path=Path(row["image"])
    fingerprint=hashlib.sha256(path.read_bytes()+prompt.encode()).hexdigest()
    name=f"{row['case']['machine']}_{row['case']['core'][0]}_{row['variant']}"
    destination=out/f"{name}.json"
    if destination.exists():
        saved=json.loads(destination.read_text(encoding="utf8"))
        if saved["fingerprint"]!=fingerprint: raise ValueError("Changed inputs")
        return saved
    try:
        raw=call(prompt,base64.b64encode(path.read_bytes()).decode(),max_tokens=1100)
        parsed=parse_response(raw,page)
        text=raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        evidence=json.loads(text)["evidence_intervals"]
        if parsed is None or not isinstance(evidence,list): raise ValueError("Invalid response")
        for span in evidence:
            if any(type(span.get(k)) is not int for k in ("channel","start","end")): raise ValueError("Invalid index type")
            if not 0<=span["start"]<span["end"]<=224 or span["channel"] not in parsed["anomalous_channels"]: raise ValueError("Invalid evidence bounds")
        if {s["channel"] for s in evidence}!=set(parsed["anomalous_channels"]): raise ValueError("Incomplete evidence")
        error=None
    except Exception as exc:
        error=type(exc).__name__
        parsed=None
        evidence=[]
    result=dict(case=row["case"],variant=row["variant"],fingerprint=fingerprint,
                prompt=prompt,raw=raw if 'raw' in locals() else None,parsed=parsed,
                evidence=evidence,error=error,index_evidence=row["index_evidence"])
    save(destination,result)
    print(name,error or parsed["event_validity"],evidence,flush=True)
    return result

with ThreadPoolExecutor(max_workers=4) as pool:
    save(out/"summary.json",list(pool.map(run,rows)))
