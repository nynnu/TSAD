"""Stage-2 v2: review-HFZ for the train-calibrated Stage-1 JSON contract.

This runner never re-selects a Stage-1 channel.  It renders the exact 224-tick
tile and evidence-ranked candidates supplied by Stage 1, then asks a VLM for two
separate decisions: event validity and within-tile channel ranking.

Live code used by pipeline_final: render, estimate_reference_period, scale, call.
LEGACY, kept only because earlier analysis scripts import them (audit_reference_v3, scope_attribution_pilot,
yellow_only_demo, test_evidence_contract, test_routing_repairs): evidence_pages, prompt, parse, main.
The production prompt and parser are pipeline_final/contracts.final_prompt / parse_response; do not extend these.
"""
import argparse, base64, hashlib, json, os, re, sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

WIN, ZOOM, PAGE = 224, 60, 8


def evidence_pages(item):
    """Partition candidates without omission; no cross-page decision implied."""
    candidates=item['channels']
    if not candidates: raise ValueError('Empty candidate list')
    if len({c['channel'] for c in candidates})!=len(candidates):raise ValueError('Duplicate candidate channel')
    count=(len(candidates)+PAGE-1)//PAGE
    for i in range(count):
        yield dict(item,channels=candidates[i*PAGE:(i+1)*PAGE],page_index=i+1,page_count=count)


def focus_bounds(evidence,tile_start):
    if 'focus_interval' in evidence:
        left,right=[int(t)-tile_start for t in evidence['focus_interval']]
        if right-left!=ZOOM or not 0<=left<right<=WIN:
            raise ValueError('Explicit Stage1 focus must be 60 ticks inside the evidence tile')
        return left,right
    peak=int(np.clip(evidence['zoom_center']-tile_start,0,WIN-1))
    left=max(0,min(WIN-ZOOM,peak-ZOOM//2))
    return left,left+ZOOM


def scale(train, values):
    lo, hi = np.quantile(train, [.01, .99], axis=0)
    hi = np.where(hi - lo < 1e-8, lo + 1., hi)
    # Keep the unbounded normalised trace for full/zoom panels.  Clipping here
    # made every strong departure look like the same flat line at 1.35.
    return (values - lo) / (hi - lo), lo, hi


def estimate_reference_period(values):
    """Train-only autocorrelation with a held-out-half consistency check."""
    values=np.asarray(values,float)
    def find(x):
        x=x-np.median(x)
        if np.std(x)<1e-8: return None,0.
        # Clip rare extremes so one spike cannot establish a period.
        bound=np.quantile(np.abs(x),.99); x=np.clip(x,-bound,bound)
        n=len(x); size=1 << (2*n-1).bit_length()
        ac=np.fft.irfft(np.abs(np.fft.rfft(x,size))**2,size)[:n]
        ac=ac/np.maximum(1,np.arange(n,0,-1)); ac=ac/max(ac[0],1e-12)
        from scipy.signal import find_peaks
        peaks,_=find_peaks(ac[4:min(n//4,1025)],height=.4,prominence=.1)
        if not len(peaks): return None,0.
        peaks=peaks+4; best=peaks[np.argmax(ac[peaks])]
        # Prefer the first strong fundamental over a near-equal harmonic.
        best=int(next((p for p in peaks if ac[p]>=ac[best]-.05),best))
        return best,float(ac[best])
    p,s=find(values); p1,_=find(values[:len(values)//2]); p2,_=find(values[len(values)//2:])
    stable=p is not None and p1 is not None and p2 is not None and all(abs(h-p)<=max(2,.1*p) for h in (p1,p2))
    return (p,s) if stable else (None,s)


def train_reference_band(train, tile, channel, focus, lo, hi, test=None, tile_start=0,
                         return_metadata=False):
    """Retrieve normal windows using only candidate context before the focus."""
    fs, fe = focus
    norm_train = (train[:, channel] - lo[channel]) / (hi[channel] - lo[channel])
    period,strength=estimate_reference_period(norm_train)
    absolute_focus=tile_start+fs
    context=min(max(48,2*period if period else 48),512,
                absolute_focus if test is not None else fs)
    horizon=fe-fs
    starts=np.arange(context,len(norm_train)-horizon+1,max(1,min(14,period//12 if period else 14)))
    if not len(starts): raise ValueError('Training series too short for reference retrieval')
    if test is not None:
        target=(test[absolute_focus-context:absolute_focus,channel]-lo[channel])/(hi[channel]-lo[channel])
    else: target=tile[fs-context:fs,channel]
    bank=np.stack([norm_train[s-context:s] for s in starts])
    if context:
        # Absolute context levels preserve amplitude; first differences match
        # local direction/shape. Two cycles disambiguate phase when available.
        level=np.mean((bank-target)**2,axis=1)
        change=np.mean((np.diff(bank,axis=1)-np.diff(target))**2,axis=1) if context>1 else level
        distance=level+.5*change
    else: distance=np.zeros(len(starts))
    chosen=[]
    for idx in np.argsort(distance,kind='stable'):
        if all(abs(int(starts[idx])-int(starts[j]))>=max(horizon,period or 0) for j in chosen):
            chosen.append(int(idx))
        if len(chosen)==8: break
    refs=np.stack([norm_train[starts[j]:starts[j]+horizon] for j in chosen])
    # This is a descriptive retrieval envelope, never a calibrated prediction interval.
    meta=dict(period=period,period_strength=strength,context=context,n_refs=len(chosen),
              context_rmse=float(np.sqrt(distance[chosen[0]])),
              reference_starts=[int(starts[j]) for j in chosen],
              limited=context<48 or (period is not None and context<2*period) or len(chosen)<4)
    result=np.quantile(refs,[.10,.50,.90],axis=0)
    return (result,meta) if return_metadata else result


def robust_ylim(values, floor=-.35):
    """Show ordinary dynamics while explicitly marking extreme departures."""
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v): return floor, 1.35
    # A 98th-percentile display scale resists one isolated huge spike.  The
    # spike is retained as an annotated marker, never silently clipped.
    top = max(1.35, float(np.quantile(v, .98)) * 1.15)
    return floor, top


def plot_adaptive(ax, x, y, top, **kwargs):
    y = np.asarray(y, float); visible = y.copy(); high = visible > top
    # Keep the temporal polyline continuous.  NaN masking made the evidence
    # look like missing observations; cap only the display coordinate and make
    # every capped value explicit with a triangle and its true maximum.
    visible[high] = top * .97
    line = ax.plot(x, visible, **kwargs)[0]
    # The white halo keeps the black candidate legible over a blue normal band.
    if kwargs.get("color") == "#111827":
        line.set_path_effects([pe.Stroke(linewidth=kwargs.get("lw", 1.0) + 2.0, foreground="white"), pe.Normal()])
    if high.any():
        ax.scatter(np.asarray(x)[high], np.full(high.sum(), top * .97), marker="^", s=16,
                   color=kwargs.get("color", "#111827"), zorder=4, clip_on=False)
        ax.text(.99, .91, f"▲ max {np.nanmax(y):.1f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5, color=kwargs.get("color", "#111827"))


def render(train, test, item, image_path, reference_provider=None, display_reference_quality=True, display_v2=False):
    """display_v2 (opt-in, so earlier experiments render unchanged): grey train min-max band on every
    full trace, and every heatmap row numbered. The red 0/1 lines are the 1-99% range, so routine
    spikes of sparse channels sit far above 1 even though normal train reached that level."""
    start, end = item["start"], item["end"]
    tile_raw = test[start:end]
    if len(tile_raw) != WIN:
        raise ValueError(f"Tile must be {WIN} ticks, got {len(tile_raw)} at {start}:{end}")
    tile, lo, hi = scale(train, tile_raw)
    candidates = item["channels"][:PAGE]
    if not candidates or len(item['channels'])>PAGE:
        raise ValueError('Expected 1..8 explicitly screened candidates; silent truncation is forbidden')
    channels = [x["channel"] for x in candidates]
    height=3.1+(1.6 if reference_provider is not None else 1.45)*len(channels)+(2.1 if display_v2 else 0)
    fig = plt.figure(figsize=(13,height), dpi=140)
    # A direct row grid avoids a large nested-grid gap and gives every trace
    # useful vertical space even on eight-channel pages.
    gs = fig.add_gridspec(len(channels)+1,2,height_ratios=[3.6 if display_v2 else 1.55]+[1.25]*len(channels),
                         width_ratios=[1.35,1],hspace=.78,wspace=.25,
                         top=1-1.3/height,bottom=.045,left=.065,right=.985)
    ax = fig.add_subplot(gs[0,:]); ax.imshow(np.clip(tile.T, -.35, 1.35), aspect="auto", cmap="viridis", vmin=-.35, vmax=1.35)
    ticks=list(range(0,tile.shape[1],1 if display_v2 else 5))
    ax.set(yticks=ticks, yticklabels=[f'ch {i}' for i in ticks],
           xlabel="time within 224-tick evidence tile", title=f"All channels; red markers = page candidates {channels}")
    if display_v2:ax.tick_params(axis='y',labelsize=5.5)
    ax.scatter(np.full(len(channels),-2),channels,marker='>',color='crimson',s=18,clip_on=False)
    if 'event_core_start' in item and item.get('show_event_core',True):
        for boundary in (item['event_core_start']-start,item['event_core_end']-start):ax.axvline(boundary,color='#7c3aed',lw=1.1)
        ax.set_xlabel('time within evidence tile; purple bounds = proposed event core')
    for label in ax.get_yticklabels():
        if label.get_text() and int(label.get_text().split()[1]) in channels:
            label.set_color("crimson"); label.set_fontweight("bold")
    reference_metadata=[]
    for row, evidence in enumerate(candidates):
        c = evidence["channel"];fs,fe=focus_bounds(evidence,start)
        # Use the same monotonic display transform as the zoom. The previous
        # fixed lower limit silently hid negative excursions.
        display=np.arcsinh(tile[:,c]); low=float(np.nanmin(display)); high=float(np.nanmax(display))
        margin=max(.08,(high-low)*.08)
        full = fig.add_subplot(gs[row+1, 0]); full.plot(np.arange(WIN),display,color='#111827',lw=1.35)
        full.axvspan(fs, fe - 1, color="#f59e0b", alpha=.12)
        full.axvline(fs, color="#d97706", lw=.8, ls="--"); full.axvline(fe - 1, color="#d97706", lw=.8, ls="--")
        if 'event_core_start' in item and item.get('show_event_core',True):
            for boundary in (item['event_core_start']-start,item['event_core_end']-start):full.axvline(boundary,color='#7c3aed',lw=.85)
        full.axhline(0, color="#dc2626", ls="--", lw=.65); full.axhline(np.arcsinh(1), color="#dc2626", ls="--", lw=.65)
        if display_v2:
            edge=(np.array([train[:,c].min(),train[:,c].max()])-lo[c])/(hi[c]-lo[c])
            full.axhspan(*np.arcsinh(edge),color='#9ca3af',alpha=.22,lw=0,zorder=0)   # clipped to the axes when train reached further
        full.set(xlim=(0, WIN - 1), ylim=(min(-.1,low-margin),max(1.,high+margin)), ylabel=f"ch {c}"); full.tick_params(labelsize=6)
        if item.get('evidence_role'):
            full.set_title(f"{evidence.get('source','evidence')} · {evidence['dominant_component']}",fontsize=8)
        zoom = fig.add_subplot(gs[row+1, 1])
        retrieve=reference_provider.band if reference_provider is not None else train_reference_band
        (q10, med, q90), reference = retrieve(train, tile, c, (fs, fe), lo, hi,
                                                       test=test,tile_start=start,return_metadata=True)
        reference_metadata.append(dict(channel=c,focus_start=start+fs,focus_end=start+fe,**reference))
        x = np.arange(fs, fe)
        # Apply the same monotonic compression to candidate and retrieved
        # normal traces.  Unlike residualisation, this preserves the actual
        # normal median *shape* as a blue overlay while preventing one huge
        # candidate spike from flattening the whole panel.
        candidate_zoom = np.arcsinh(tile[fs:fe, c])
        q10_t, med_t, q90_t = np.arcsinh(q10), np.arcsinh(med), np.arcsinh(q90)
        # Visual contract for the VLM: the filled band is the primary
        # retrieved range; the blue line is only a representative train
        # temporal shape.  A page-level reading rule below avoids a tiny local
        # legend obscuring the evidence in every small multiple.
        weak=display_reference_quality and reference.get('reference_status') in ('out_of_support','insufficient_support','validation_unstable')
        zoom.fill_between(x, q10_t, q90_t, color="#9ca3af" if weak else "#93c5fd", alpha=.30,
                          label="normal range (10–90%)")
        zoom.plot(x, candidate_zoom, color="#111827", lw=1.7,
                  path_effects=[pe.Stroke(linewidth=3.7, foreground="white"), pe.Normal()],
                  label="candidate")
        # Plot the normal median *after* the black trace. Otherwise the black
        # trace and its white halo completely conceal near-identical reference
        # spikes. Dashes leave portions of both overlapping lines visible.
        zoom.plot(x, med_t, color="#2563eb", lw=1.55, ls=(0,(3,2)), zorder=5,
                  label="normal shape (median)")
        lower=float(np.nanmin(np.r_[candidate_zoom,q10_t]));upper=float(np.nanmax(np.r_[candidate_zoom,q90_t]))
        padding=max(.08,(upper-lower)*.08)
        lo_zoom = min(-.35, lower-padding)
        hi_zoom = max(1.35, upper+padding)
        if 'event_core_start' in item and item.get('show_event_core',True):
            for boundary in (item['event_core_start']-start,item['event_core_end']-start):
                if fs<=boundary<=fe-1:zoom.axvline(boundary,color='#7c3aed',lw=.85)
        zoom.set(xlim=(fs, fe - 1), ylim=(lo_zoom, hi_zoom),
                 title=f"{'period '+str(reference['period']) if reference['period'] else 'no stable period'} · context {reference['context']} · refs {reference['n_refs']}"
                       + (' · LIMITED' if reference['limited'] else '')
                       + (' · LOW SUPPORT' if weak else '') )
        zoom.tick_params(labelsize=6)
        if reference_provider is not None and display_reference_quality:
            zoom.set_title(zoom.get_title()+f"\ncontext support p={reference['context_support_p']:.3f} · normal-train band hit {1-reference['validation_outside_fraction']:.0%}",fontsize=8)
    audit_warning=' · TRAIN AUDIT WARNING' if item.get('calibration_context',{}).get('transfer_warning') else ''
    if reference_provider is not None and display_reference_quality:
        item['reference_quality']=[dict(channel=r['channel'],status=r['reference_status'],
            context_support_p=r['context_support_p'],validation_outside_fraction=r['validation_outside_fraction'],
            validation_focus_rmse_q90=r['validation_focus_rmse_q90'],
            support_audit_warning=r['support_audit_warning'],support_audit_failure_fraction=r['support_audit_failure_fraction']) for r in reference_metadata]
    heading=item.get('view_title','Stage-2 review-HFZ v2')
    fig.suptitle(f"{heading} · {item['band']} band · {item['machine']} [{start}, {end}) · page {item.get('page_index',1)}/{item.get('page_count',1)}{audit_warning}", fontsize=9.5,y=1-.16/height)
    fig.text(.5, 1-.55/height,
             "RIGHT ZOOM — range: blue = context supported, grey = low support · blue dashed: train median · black: candidate"
             if reference_provider is not None and display_reference_quality else "RIGHT ZOOM — light blue: retrieved range (not a normality boundary) · blue dashed: median · black: candidate",
             ha="center", va="center", fontsize=7.4,
             bbox={"boxstyle":"round,pad=.27", "fc":"#f8fafc", "ec":"#94a3b8", "lw":.7})
    fig.text(.5,1-.84/height,('Trace axes: asinh(train-normalized value) · red dashed: normalized 0 / 1 (train 1-99%), not anomaly thresholds · grey band: min-max seen in normal train'
             if display_v2 else 'Trace axes: asinh(train-normalized value) · red dashed: normalized 0 / 1, not anomaly thresholds'),
             ha='center',va='center',fontsize=7)
    image_path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(image_path, bbox_inches="tight"); plt.close(fig)
    image_path.with_suffix('.reference.json').write_text(json.dumps(reference_metadata,indent=2),encoding='utf-8')
    return base64.b64encode(image_path.read_bytes()).decode()


def prompt(item):
    """LEGACY prompt for the older Stage-1 band contract; the pipeline uses contracts.final_prompt."""
    page = [x["channel"] for x in item["channels"][:PAGE]]
    role=item.get('evidence_role','legacy')
    context=item.get('calibration_context',{})
    quality=json.dumps(item.get('reference_quality',[]),separators=(',',':'))
    warning=('Temporal normal-train audit showed poor threshold transfer. Score bands are candidate-routing signals, not reliable anomaly confidence.'
             if context.get('transfer_warning') else '')
    return f'''You are reviewing page {item.get('page_index',1)} of {item.get('page_count',1)} for one Stage-1 proposal in a multivariate time series. This is a {item["band"]}-band proposal, not ground truth. Judge only displayed channels. A normal_repeat judgment on this page does not establish normality of other pages or the whole event.

Evidence role: {role}. State evidence means departure from historical absolute levels; dynamic evidence means local variation/change. A persistent level departure can be normal or anomalous, and a quiet local trace does not rule out a sustained anomaly. {warning}

Reference support metadata: {quality}. If supplied, context_supported describes only pre-focus context similarity, not focus normality. Grey/LOW SUPPORT indicates an unsupported or unstable reference; do not interpret its mismatch as anomaly proof. validation_unstable/support_audit_warning means frozen support calibration failed to transfer to later normal train. Historical validation_outside_fraction describes normal train behavior outside the descriptive band and is not a test anomaly probability. Large historical forecast error weakens confidence even when the context is supported. Prefer uncertain when the reference cannot support a decision.

The top heatmap shows all channels on one train-normalized 224-tick axis. Each row below shows a Stage-1 candidate: full tile at left and local peak zoom at right. Blue dashed is the retrieved train-reference median, light blue its descriptive 10–90% range, and black the candidate. All three use the same monotonic asinh display transform. Compare timing, shape and amplitude with the retrieval limitations below.

The light-blue band is only a descriptive range of retrieved training examples, NOT a calibrated normality boundary. Retrieval uses only pre-focus observations, matches levels and first differences, and uses up to two train-estimated cycles when a stable period exists. References are separated in training time. The title reports the estimated period, context length and reference count; LIMITED means insufficient context or references. Even an unflagged reference may be a poor match. Median aggregation can erase normal spikes whose timing differs between references. A candidate outside this band is therefore not sufficient evidence of anomaly. Compare recurrence, amplitude, width, before/after behavior and coordinated channel evidence. Short anomalies are possible: persistence is supporting evidence, not a requirement. Prefer uncertain when reference mismatch prevents a reliable decision. First decide event validity, then rank only candidates {page}. Do not infer causality.

Respond ONLY JSON:
{{"event_validity":"normal_repeat"|"uncertain"|"anomalous","anomalous_channels":[subset of {page}],"channel_ranking":[ordered subset of {page}],"confidence":"low"|"medium"|"high"}}'''


def parse(raw, allowed):
    """LEGACY parser. Differs from contracts.parse_response: it accepts event_validity=anomalous with no
    anomalous_channels and ignores additional_channels_to_review. Not used by the pipeline."""
    if not raw: return None
    m = re.search(r"\{.*\}", raw, re.S)
    try: d = json.loads(m.group(0) if m else raw)
    except Exception: return None
    if not isinstance(d,dict): return None
    valid = {"normal_repeat", "uncertain", "anomalous"}
    if d.get("event_validity") not in valid: return None
    if d.get('confidence') not in {'low','medium','high'}: return None
    for field in ('anomalous_channels','channel_ranking'):
        xs=d.get(field)
        if not isinstance(xs,list) or any(type(x) is not int or x not in allowed for x in xs): return None
        if len(xs)!=len(set(xs)): return None
    if not set(d['anomalous_channels']).issubset(d['channel_ranking']): return None
    if d['event_validity']=='normal_repeat' and d['anomalous_channels']: return None
    return d


_CLIENT = None


def _client():
    """One client for the whole run. The SDK retries connection errors, 408/409/429 and 5xx with backoff;
    the default of 2 retries let 20 of 48 calls fail with APIConnectionError in one run."""
    global _CLIENT
    if _CLIENT is None:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)   # .env wins over the environment
        _CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=6, timeout=120)
    return _CLIENT


def call(prompt_text, image, max_tokens=200):
    client = _client()
    r = client.chat.completions.create(model="gpt-4o", temperature=0, max_tokens=max_tokens, messages=[{"role":"user","content":[{"type":"text","text":prompt_text},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+image,"detail":"high"}}]}])
    return r.choices[0].message.content


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", type=Path, required=True); ap.add_argument("--stage1", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", action="store_true"); ap.add_argument("--limit", type=int); ap.add_argument("--review-only", action="store_true")
    ap.add_argument('--reference-quality',action='store_true',help='Use disjoint fit-bank reference with train-validated context support')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True); images = args.out / "images"; checkpoint = args.out / "checkpoint.json"
    done = json.loads(checkpoint.read_text()) if checkpoint.exists() else {}; rows=[]
    code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.reference_quality:
        from reference_quality_v3 import ReferenceQuality
        code_hash+=hashlib.sha256(Path(__file__).with_name('reference_quality_v3.py').read_bytes()).hexdigest()
    for path in sorted(args.stage1.glob("*.json")):
        d=json.loads(path.read_text()); entity=d["machine"]; train=np.loadtxt(args.data/"train"/(entity+".txt"),delimiter=","); test=np.loadtxt(args.data/"test"/(entity+".txt"),delimiter=",")
        data_hash=hashlib.sha256(train.tobytes()+test.tobytes()).hexdigest()
        provider=ReferenceQuality(train,estimate_reference_period) if args.reference_quality else None
        for i, item in enumerate(page for tile in d['intervals'] for page in evidence_pages(tile)):
            if args.review_only and item["band"] != "review": continue
            item=dict(item, machine=entity); key=f"{entity}:{item['start']}:{item['end']}:{item['band']}:{item.get('parent_start',item['start'])}:p{item['page_index']}"; image_path=images/(key.replace(":","_")+".png")
            image=render(train,test,item,image_path,reference_provider=provider)
            fingerprint=hashlib.sha256((code_hash+data_hash+json.dumps(item,sort_keys=True)+prompt(item)).encode()).hexdigest()
            cache_key=key+':'+fingerprint
            result=done.get(cache_key)
            if args.run and result is None:
                result=parse(call(prompt(item),image),{x["channel"] for x in item["channels"][:PAGE]}); done[cache_key]=result; checkpoint.write_text(json.dumps(done,indent=2))
            rows.append(dict(key=key,machine=entity,band=item["band"],start=item["start"],end=item["end"],
                             parent_start=item.get('parent_start'),parent_end=item.get('parent_end'),
                             page_index=item['page_index'],page_count=item['page_count'],
                             candidates=[x["channel"] for x in item["channels"]],reference_quality=item.get('reference_quality'),vlm=result,image=str(image_path)))
            print(key, "rendered" if not args.run else result, flush=True)
            if args.limit and len(rows)>=args.limit: break
        if args.limit and len(rows)>=args.limit: break
    (args.out/"results.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)); print(json.dumps({"n":len(rows),"out":str(args.out)},ensure_ascii=False))

if __name__ == "__main__": main()
