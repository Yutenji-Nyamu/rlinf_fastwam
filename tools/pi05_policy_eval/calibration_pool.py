"""Outcome-blind compatible historical U collection and repeated-state dedup."""
import json
import math
import socket
from pathlib import Path


def collect(roots, chunk, max_queries=None):
    host=socket.gethostname();rows=[];sources=[]
    for root in map(Path,roots):
        if not root.exists():continue
        for done in sorted((root/'batches').glob('*/done.json')):
            b=done.parent
            try:
                c=json.loads((b/'config.json').read_text())
                decisions=json.loads((b/'decisions.json').read_text())
            except (OSError,ValueError):continue
            if (c.get('num_steps')!=10 or c.get('execute_chunk')!=chunk or
                c.get('takeover_mode','weights')!='weights' or
                c.get('env',{}).get('task_config',{}).get('task_name')!='move_pillbottle_pad' or
                c.get('model',{}).get('openpi',{}).get('action_chunk')!=50):continue
            weak=c['model']['model_path'];strong=Path(c.get('strong_weights','unused')).parent.name
            histories={}
            for d in decisions:
                slot=d['slot'];prefix=histories.setdefault(slot,[])
                u=d.get('u')
                if u is not None and math.isfinite(u) and (max_queries is None or d['query']<max_queries):
                    # A shared policy prefix/seed is a state identifier only within
                    # its host. Retain distinct observed values if runtime drift
                    # makes the otherwise identical key disagree.
                    state=[host,weak,d['actual_seed'],slot,d.get('noise_seed'),d['query'],d['start'],prefix.copy()]
                    rows.append({'u':u,'query':d['query'],'start':d['start'],
                                 'state':state,'source':str(b),'chunk':chunk})
                prefix.append([strong if d['strong'] else 'weak',d['end']-d['start']])
            sources.append(str(b))
    return {'host':host,'chunk':chunk,'max_queries':max_queries,'rows':rows,'source_batches':sources}


def deduplicate(pools):
    unique={};raw=0
    for pool in pools:
        for row in pool['rows']:
            raw+=1
            key=json.dumps(row['state'],separators=(',',':'))+'|'+float(row['u']).hex()
            unique.setdefault(key,row)
    return {'raw_decisions':raw,'unique_decisions':len(unique),'rows':list(unique.values()),
            'method':'host + weak model + seed/slot/noise/query/start + policy prefix + observed U; no success labels',
            'sources':[{'host':p['host'],'batches':len(p['source_batches'])}for p in pools]}
