"""Metadata-only size analysis and non-executable checkpoint candidate manifest."""
import json,re
from pathlib import Path, PurePosixPath
source=Path('docs/server-admin/SZ_CHEN_STORAGE_STRUCTURE_20260905.json')
x=json.loads(source.read_text(encoding='utf-8'))
gib=1024**3
runs=[];candidates=[]
for run in x['checkpoint_runs']:
    gens=[]
    for gen in run['generations']:
        match=re.fullmatch(r'global_step_(\d+)',gen['generation'])
        large=[f for f in gen['big_files'] if f['bytes']>=gib]
        gens.append({'name':gen['generation'],'step':int(match[1]) if match else None,'path':gen['path'],'allocated_gib':gen['allocated_bytes']/gib,'large':large,'small_count':len(gen['small_files']),'big_count':len(gen['big_files'])})
    steps=[g['step'] for g in gens if g['step'] is not None]
    latest=max(steps) if steps else None
    kept={latest} if latest is not None else set()
    is_sidney='/pi05-sidney/' in run['root'] and 'formal100' in run['root']
    if is_sidney:kept|={70,100} # known best saved checkpoint and current resume origin
    for gen in gens:
        if gen['step'] is not None and gen['step'] not in kept:
            for f in gen['large']:
                candidates.append({'path':str(PurePosixPath(gen['path'])/f['relative']),'bytes':f['bytes'],'allocated_bytes':f['allocated_bytes'],'run_root':run['root'],'generation':gen['name'],'basis':'discussion candidate only; keep highest step number, additionally Sidney70/100; source/resume/completeness must be rechecked before any deletion'})
    runs.append({'root':run['root'],'total_gib':sum(g['allocated_gib'] for g in gens),'latest_number_only':latest,'kept_numbers':sorted(kept),'generations':gens})
result={'snapshot_started':x['time_started'],'snapshot_finished':x['time_finished'],'status':'READ_ONLY_DISCUSSION_NOT_DELETION_AUTHORIZATION','checkpoint_allocated_gib':sum(r['total_gib'] for r in runs),'checkpoint_roots':len(runs),'candidates_count':len(candidates),'candidate_allocated_gib':sum(c['allocated_bytes'] for c in candidates)/gib,'candidate_files':candidates,'runs':runs}
out=Path('docs/server-admin/SZ_CHECKPOINT_CLEANUP_DISCUSSION_MANIFEST_20260905.json')
out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('candidate_files','runs')},ensure_ascii=False))
for r in sorted(runs,key=lambda r:r['total_gib'],reverse=True)[:20]:
    print(json.dumps({'root':r['root'],'gib':round(r['total_gib'],2),'steps':[(g['name'],round(g['allocated_gib'],2),len(g['large']),g['small_count']) for g in r['generations']],'keep_numbers':r['kept_numbers']},ensure_ascii=False))
print('CATEGORY_CANDIDATES')
for key in ('/grpo/','/ppo/','/pi05-sidney/','/pi05/','/fastwam-grpo/','/online-bc/','/rlinf-current-dsrl/','/rlinf-rlt/'):
    v=[c for c in candidates if key in c['path']]
    print(key,len(v),round(sum(c['allocated_bytes'] for c in v)/gib,2))
