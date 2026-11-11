"""Same-candidate channel expansion pilot; GT is used only in case selection/evaluation."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import numpy as np
from paired_bank_vlm_pilot import ROOT, build_plan, call, parse_response, save, summarize_core, render, ReferenceQuality, estimate_reference_period
from contracts import make_pages, display_candidate, final_prompt
from index_aligned_pilot import index_evidence
from annotation_contract import read_annotation_contract, channel_truth

source=ROOT/"analysis/results/index_aligned_pilot_20261110"
out=ROOT/"analysis/results/channel_expansion_matched_20261110"
out.mkdir(parents=True,exist_ok=True)
prior=[r for r in json.loads((source/"summary.json").read_text(encoding="utf8")) if r["variant"]=="index"]
selected=[]
for positive in (False,True):
    selected += [r for r in prior if (r["case"]["category"]=="gt_overlap")==positive][:2]
save(out/"selection.json",[r["case"] for r in selected])
cache={}
reports=[]

for ref in selected:
    case=ref["case"]; machine=case["machine"];a,b=case["core"]
    if machine not in cache:
        data=ROOT.parent/"mv_data/SMD"
        train=np.loadtxt(data/"train"/f"{machine}.txt",delimiter=",")
        test=np.loadtxt(data/"test"/f"{machine}.txt",delimiter=",")
        doc=json.loads((ROOT/"pipeline_final/results/SMD_final_v1/stage1_snapshots"/f"{machine}.json").read_text(encoding="utf8"))
        cache[machine]=(train,test,build_plan(doc,test),ReferenceQuality(train,estimate_reference_period))
    train,test,plan,provider=cache[machine]
    page,=[p for p in plan["pages"] if [p["event_core_start"],p["event_core_end"]]==[a,b] and p["review_round"]==0]
    core=next(c for e in plan["events"] for c in e["cores"] if c["core_id"]==page["core_id"])
    reviewed={r["channel"] for r in page["channels"]}
    rows=[display_candidate(test,c,a,b,page["start"]) for c in range(test.shape[1]) if c not in reviewed]
    extra=make_pages(core,rows,1)
    metadata=json.JSONDecoder().raw_decode(ref["prompt"].split("Reference quality: ",1)[1])[0]
    artifacts={page["page_id"]:{"reference_quality":metadata}}
    records={page["page_id"]:{"parsed":ref["parsed"]}}
    assert summarize_core(core,[page],records,artifacts)["status"]==ref["decision"]["status"]
    jobs=[]
    for added in extra:
        image=out/f"{added['page_id']}.png"; item=dict(added)
        render(train,test,item,image,reference_provider=provider)
        artifacts[added["page_id"]]={"reference_quality":item["reference_quality"]}
        prompt=final_prompt(item)+"\nIndex alignment guide: "+json.dumps(index_evidence(added,test),separators=(",",":"))
        prompt+="\nThese indices locate plots only; a peak location does not establish anomaly."
        jobs.append((added,image,prompt))
        print(added["page_id"],"rendered",flush=True)
    def review(job):
        added,image,prompt=job
        dest=out/f"{added['page_id']}.json"
        fingerprint=hashlib.sha256(image.read_bytes()+prompt.encode()).hexdigest()
        if dest.exists():
            result=json.loads(dest.read_text(encoding="utf8"))
            if result["fingerprint"]!=fingerprint:raise ValueError("Changed inputs")
            return added,result
        try:
            raw=call(prompt,base64.b64encode(image.read_bytes()).decode(),max_tokens=700)
            parsed=parse_response(raw,added)
            error=None if parsed else "invalid_schema"
        except Exception as exc:
            raw=parsed=None;error=type(exc).__name__
        result=dict(fingerprint=fingerprint,prompt=prompt,raw=raw,parsed=parsed,error=error)
        save(dest,result)
        print(added["page_id"],error or parsed["event_validity"],flush=True)
        return added,result
    with ThreadPoolExecutor(max_workers=4) as pool:
        for added,result in pool.map(review,jobs):
            records[added["page_id"]]=result
    decision=summarize_core(core,[page]+extra,records,artifacts)
    gt,status=channel_truth(read_annotation_contract(ROOT.parent/"mv_data/SMD",machine),a,b)
    raw_before=set(ref["parsed"]["anomalous_channels"])
    raw_after=set().union(*(set(r["parsed"]["anomalous_channels"]) for r in records.values() if r.get("parsed")))
    reports.append(dict(case=case,gt_channels=sorted(gt) if gt is not None else None,
                        annotation_status=status,before=ref["decision"],after=decision,
                        raw_before=sorted(raw_before),raw_after=sorted(raw_after),
                        shown_before=sorted(reviewed),extra_calls=len(extra),
                        errors=[r["error"] for r in records.values() if r.get("error")]))
    save(out/"summary.json",reports)
    print(machine,a,"before",ref["decision"]["status"],"after",decision["status"],flush=True)
