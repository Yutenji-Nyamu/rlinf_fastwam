"""Summarize allocated-byte metadata. Candidates are discussion, never a delete list."""
import json
from pathlib import Path, PurePosixPath
root=Path(__file__).resolve().parents[1]
ev=root/'docs/rlinf-robotwin-pi0-online-bc/evidence'
r=json.loads((ev/'BC_PARAMETERS_STORAGE_LIVE_20260905.json').read_text(encoding='utf-8'))
G=1024**3
sizes={p['path']:p['allocated_bytes']/G for p in r['storage']['du']}
candidates={k:[] for k in ('new_bc_step1','old_pi0_bc_v7_step2','dsrl_formal_older','dsrl_component_smokes','rlt_stage1_smoke','grpo_deferred')}
for run in r['storage']['checkpoint_runs']:
    for gen in run['generations']:
        p=gen['path'];step=gen['step'];key=None
        if '/online-bc/' in p and step==1 and ('eval8x4-gpu6-20260905-v8/' in p or 'pi05-pillbottle-smoke32x1-' in p):key='new_bc_step1'
        elif '/online-bc/' in p and 'eval16x2-gpu6-20260905-v7/' in p and step==2:key='old_pi0_bc_v7_step2'
        elif '/rlinf-current-dsrl/formal-' in p and step in (65,130,195):key='dsrl_formal_older'
        elif '/rlinf-current-dsrl/smoke-' in p:key='dsrl_component_smokes'
        elif '/rlinf-rlt/smoke-stage1-current-ar-2step-' in p:key='rlt_stage1_smoke'
        if key:
            files=[{'path':str(PurePosixPath(p)/f['file']),**f} for f in gen['files'] if f['bytes']>=G]
            candidates[key].append({'generation':p,'files':files,'large_gib':sum(f['allocated_bytes'] for f in files)/G})
summary={'snapshot':r['time'],'finished':r['storage']['finished'],'du_rc':r['storage']['du_rc'],
 'top':{k:v for k,v in sizes.items() if str(PurePosixPath(k).parent)=='/data/chenyiteng'},
 'total_gib':sizes['/data/chenyiteng'],'results_subgroups':{k:v for k,v in sizes.items() if str(PurePosixPath(k).parent) in ('/data/chenyiteng/results','/data/chenyiteng/results/rlinf-shenzhen')},
 'candidate_summary':{k:{'files':sum(len(x['files']) for x in a),'gib':sum(x['large_gib'] for x in a)} for k,a in candidates.items()},
 'candidates':candidates,'scope':'Discussion only: metadata presence is not restart/dependency/keep-point validation. No deletion authorization in this turn.'}
(ev/'BC_STORAGE_DISCUSSION_SUMMARY_20260905.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='candidates'},ensure_ascii=False,indent=2))
