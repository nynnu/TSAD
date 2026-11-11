"""Retrospective Stage1 timing and selected-channel zoom coverage audit.

GT is used only for post-hoc evaluation. No threshold or candidate is changed.
Interpretation labels use 1-based channel IDs, converted to zero-based here.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate_tiled_events import segments


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def interval_gap(a,b,u,v):
    return max(u-b,a-v,0)


def main(data,stage1,selection,out):
    out.mkdir(parents=True,exist_ok=True);entities=[]
    for machine in ('machine-1-1','machine-1-2','machine-1-6'):
        d=read(stage1/(machine+'.json'));y=np.loadtxt(data/'test_label'/(machine+'.txt'),delimiter=',').astype(bool).ravel();gt=segments(y)
        anns=[]
        for line in (data/'interpretation_label'/(machine+'.txt')).read_text(encoding='utf-8').splitlines():
            span,ids=line.split(':');u,v=map(int,span.split('-'));anns.append((u,v,{int(k)-1 for k in ids.split(',')}))
        events=d['detected_events']+d['review_events'];parents=[(int(e['start']),int(e['end']),e['band']) for e in events]
        unique={(a,b,band) for a,b,band in parents};parents=sorted(unique)
        event_rows=[]
        for u,v in gt:
            overlap=[(a,b,band) for a,b,band in parents if a<v and u<b]
            mask=np.zeros(v-u,bool)
            for a,b,_ in overlap:mask[max(a,u)-u:min(b,v)-u]=True
            best=max(overlap,key=lambda x:(max(0,min(x[1],v)-max(x[0],u))/max(x[1]-x[0]+v-u-max(0,min(x[1],v)-max(x[0],u)),1),-x[0])) if overlap else None
            event_rows.append(dict(gt=[int(u),int(v)],any_parent_overlap=bool(overlap),gt_coverage=float(mask.mean()),
                best_parent=list(best) if best else None,best_start_offset=int(best[0]-u) if best else None,
                best_end_offset=int(best[1]-v) if best else None,
                best_parent_gt_events=sum(best[0]<y and x<best[1] for x,y in gt) if best else None))
        misses=[]
        for a,b,band in parents:
            if any(a<v and u<b for u,v in gt):continue
            if not gt:continue
            u,v=min(gt,key=lambda z:(interval_gap(a,b,*z),z[0]));gap=interval_gap(a,b,u,v)
            relation='before' if b<=u else 'after' if a>=v else 'overlap'
            misses.append(dict(parent=[a,b],band=band,nearest_gt=[int(u),int(v)],gap_ticks=int(gap),relation=relation))
        matched=[r for r in event_rows if r['best_parent']]
        entities.append(dict(machine=machine,gt_events=len(gt),parents=len(parents),gt_with_overlap=len(matched),
            full_gt_time_covered=sum(r['gt_coverage']==1 for r in event_rows),
            mean_gt_coverage=float(np.mean([r['gt_coverage'] for r in event_rows])),
            best_start_before=sum(r['best_start_offset']<0 for r in matched),
            best_start_equal=sum(r['best_start_offset']==0 for r in matched),
            best_start_after=sum(r['best_start_offset']>0 for r in matched),
            matched_parent_spans_multiple_gt=sum(r['best_parent_gt_events']>1 for r in matched),
            nonoverlap_parents=len(misses),nonoverlap_before_within5=sum(r['relation']=='before' and r['gap_ticks']<=5 for r in misses),
            nonoverlap_after_within5=sum(r['relation']=='after' and r['gap_ticks']<=5 for r in misses),
            event_rows=event_rows,nonoverlap_rows=misses))
    # Zoom audit is conditional on the preselected 24 difficult GT events,
    # and only on channels that the saved Stage1 tile actually selected.
    saved=read(selection);doc=read(stage1/'machine-1-6.json');zoom_rows=[]
    for s in saved:
        if s['category']!='lost_high_gt':continue
        pa,pb=s['parent'];start,end=s['tile'];u,v=s['gt_event'];gtc=set(s['gt_channels'])
        matches=[t for t in doc['intervals'] if t['parent_start']==pa and t['parent_end']==pb and t['start']==start and t['end']==end and [r['channel'] for r in t['channels']]==s['selected_channels']]
        if len(matches)!=1:raise ValueError('Ambiguous saved tile')
        t=matches[0];core_a=max(pa,start);core_b=min(pb,end)
        relevant=[c for c in t['channels'] if c['channel'] in gtc]
        parent_cover=max(0,min(core_b,v)-max(core_a,u))/(v-u)
        zoom_mask=np.zeros(v-u,bool)
        for c in relevant:
            a,b=c['focus_interval'];lo=max(u,a);hi=min(v,b)
            if lo<hi:zoom_mask[lo-u:hi-u]=True
        zoom_rows.append(dict(case=s['case'],gt=[u,v],core=[core_a,core_b],gt_channels=len(gtc),gt_channels_displayed=len(relevant),
            core_gt_coverage=parent_cover,zoom_union_gt_coverage=float(zoom_mask.mean()),
            gt_displayed_channel_zoom_overlap=sum(c['focus_interval'][0]<v and u<c['focus_interval'][1] for c in relevant),
            gt_displayed_channel_zoom_fully_covers=sum(c['focus_interval'][0]<=u and v<=c['focus_interval'][1] for c in relevant),
            gt_displayed_channel_score_peak_in_event=sum(u<=c['score_peak']<v for c in relevant)))
    payload=dict(entities=entities,selected_zoom_cases=zoom_rows,
        limitations='Exploratory GT-based diagnostic; no independent evaluation, no VLM outcome or causal attribution. Zoom union is conditional on saved candidate channels and selected tiles, not Stage1 detection accuracy.')
    (out/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    for row in entities:
        print(json.dumps({k:v for k,v in row.items() if k not in ('event_rows','nonoverlap_rows')},ensure_ascii=False))
    print('zoom_summary',json.dumps(dict(n=len(zoom_rows),core_full=sum(r['core_gt_coverage']==1 for r in zoom_rows),
        zoom_union_full=sum(r['zoom_union_gt_coverage']==1 for r in zoom_rows),
        zoom_gt_channel_cases=sum(r['gt_displayed_channel_zoom_overlap']>0 for r in zoom_rows),
        at_least_one_gt_channel_zoom_fully_covers=sum(r['gt_displayed_channel_zoom_fully_covers']>0 for r in zoom_rows),
        gt_displayed_channel_rows=sum(r['gt_channels_displayed'] for r in zoom_rows),
        score_peaks_in_event=sum(r['gt_displayed_channel_score_peak_in_event'] for r in zoom_rows),
        zoom_wider_than_core=sum(r['zoom_union_gt_coverage']>r['core_gt_coverage'] for r in zoom_rows)),ensure_ascii=False))
    # Absolute ticks make the distinction between event core and zoom explicit.
    selected_cases=[next(x for x in saved if x['case']==i) for i in (1,43)]
    fig,axes=plt.subplots(2,1,figsize=(12,4.5),layout='constrained')
    for ax,s in zip(axes,selected_cases):
        tile=s['tile'];parent=s['parent'];core=[max(tile[0],parent[0]),min(tile[1],parent[1])]
        t=next(t for t in doc['intervals'] if [t['parent_start'],t['parent_end']]==parent and [t['start'],t['end']]==tile and [c['channel'] for c in t['channels']]==s['selected_channels'])
        shown=([next(c for c in t['channels'] if c['channel']==ch) for ch in (17,0,14,15)] if s['case']==43 else t['channels'][:4])
        for i,c in enumerate(shown):ax.broken_barh([(c['focus_interval'][0],60)],(i-.3,.6),facecolors='#fbbf24',alpha=.75)
        ax.axvspan(core[0],core[1],color='#8b5cf6',alpha=.35,label='Stage1 candidate core')
        gt_window=[653,658]
        ax.axvspan(gt_window[0],gt_window[1],color='#ef4444',alpha=.6,label='GT event [653,658)')
        ax.set(xlim=(min(tile[0],gt_window[0]),max(tile[1],gt_window[1])),ylim=(-.7,3.7),
            yticks=range(4),yticklabels=['ch '+str(c['channel']) for c in shown],xlabel='absolute tick',
            title=f"case {s['case']}: core {core}; orange = each channel zoom")
        ax.legend(loc='upper left',fontsize=8)
    fig.savefig(out/'timing_zoom_examples.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--stage1',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.data,a.stage1,a.selection,a.out)
