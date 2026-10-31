import copy, json, math, os, shutil, subprocess, time
from pathlib import Path
D=Path('/home/chenyiteng/deployment-20261101/gad-k4-v1')
old=Path('/home/chenyiteng/deployment-20261031/pi05-nstep-sz1-g67-v1/source')
D.mkdir(parents=True,exist_ok=True)
if not (D/'source').exists():
    shutil.copytree(old,D/'source',ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
ref=Path('/home/chenyiteng/deployment-20261029/single-extend400-sz1-g67-retention')
cp=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67')
gad=dict(version=1,samples=4,weight_range=[.5,1.5],uniform_mix=.5,priority_alpha=.6,importance_beta=1.,auxiliary_weighting=True)
for method in ['endpoint','cp_v2','onedp','mfd']:
    c=read((cp if method=='cp_v2' else ref)/f'{method}.json')
    c.update(gad=gad,experiment_name='gad-'+method,epochs=400,checkpoint_keep_last=1,
             checkpoint_keep_steps=[],retention_mode='latest_complete_only',
             cuda_allocator_conf='expandable_segments:True',smoke_inference_batch=32)
    per=math.ceil(4738/c['batch_size'])
    fractions=(.0125,.025,.05,.1,.15,.2,.3,.4,.5,.65,.8,1.)
    points=sorted({max(1,round(100*per*f)) for f in fractions}|{per*e for e in range(120,401,20)})
    c.update(evaluation_steps=points,checkpoint_steps=400*per+1)
    if method in ['endpoint','cp_v2']:
        c['cosine_budget_boundaries']=[per*e for e in ([100,200,400] if method=='endpoint' else [200,400])]
    save(D/'formal-configs'/f'{method}.json',c)
    smoke=copy.deepcopy(c);smoke.update(log_every=1)
    save(D/f'{method}.json',smoke)
O=Path('/home/chenyiteng/results/pi05-distill/gad-smoke-20261101')
O.mkdir(parents=True,exist_ok=True)
src=Path(read(ref/'prepared.json')['output'])
for gpu in [6,7]:
    save(O/f'environment-g{gpu}.json',read(src/f'environment-g{gpu}.json'))
save(D/'prepared.json',dict(output=str(O),physical_gpus=[6,7],poll_seconds=.25))
save(D/'formal-plan.json',dict(methods=['endpoint','cp_v2','onedp','mfd']))
save(D/'preparation.json',dict(time=time.time(),source=str(old),purpose='four isolated GAD smokes then original collection and formal queues',control=str(ref)))
print(json.dumps(dict(deployment=str(D),output=str(O),source=str(D/'source'))))
