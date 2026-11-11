"""Identify Stage1 proposals also visible in a page's per-channel zooms.

This is diagnostic metadata only: overlapping views do not prove an error,
identify a cause, relocate events, or authorize deletion of a proposal.
"""

def overlapping_candidates(target, all_tiles):
    current=(int(target['parent_start']),int(target['parent_end']))
    focus=[(int(c['channel']),*map(int,c['focus_interval'])) for c in target['channels'] if 'focus_interval' in c]
    others=sorted({(int(t['parent_start']),int(t['parent_end'])) for t in all_tiles}-{current})
    hits=[]
    for a,b in others:
        channels=[c for c,l,r in focus if max(a,l)<min(b,r)]
        if channels:hits.append({'parent_start':a,'parent_end':b,'zoom_channels':channels})
    return {'target_parent':list(current),'other_candidate_overlaps':hits,
            'scope_ambiguous':bool(hits),'automatic_event_decision':False}
