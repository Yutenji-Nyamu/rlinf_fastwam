import json,os,sys,tarfile,shutil,subprocess,time
from pathlib import Path
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'))
from common import read,save,same
D=Path('/data/chenyiteng/deployment-20261020/c20-sz2-g45-v1')
assert os.getuid()==20001 and not D.exists();D.mkdir(parents=True)
with tarfile.open('/data/chenyiteng/c20-migration-20261020.tar.gz')as t:t.extractall(D/'imported')
R=D/'source';R.mkdir()
with tarfile.open('/data/chenyiteng/c20-source-20261020.tar.gz')as t:t.extractall(R)
old=read(D/'imported/transaction/original-prepared.json')
rank=Path('/data/chenyiteng/results/pi05-policy-eval/pillbottle-c20-rank-sz2-20261020')
sweep=rank.with_name('pillbottle-c20-sweep-sz2-20261020')
assert not rank.exists()
shutil.copytree(D/'imported/rank',rank)
for n in ['batches','summaries','thresholds']:(rank/n).mkdir(exist_ok=True)
def replace(v):
 if isinstance(v,dict):return{k:replace(x)for k,x in v.items()}
 if isinstance(v,list):return[replace(x)for x in v]
 if isinstance(v,str):return v.replace(old['roots']['rank'],str(rank)).replace(old['source'],str(R))
 return v
for f in(rank/'configs').glob('*.json'):save(f,replace(read(f)))
for g in(4,5):
 env=read(Path('/data/chenyiteng/results/pi05-u-video/20261017-sz2-v2')/f'environment-g{g}.json')
 oldrepo=env['REPO_PATH']
 env={k:v.replace(oldrepo,str(R))if isinstance(v,str)else v for k,v in env.items()}
 env.update(REPO_PATH=str(R),TMPDIR=str(D/f'tmp-g{g}'),CUDA_VISIBLE_DEVICES=str(g),DOJO_GPU_SCOPE=str(g))
 env.pop('RAY_ADDRESS',None);Path(env['TMPDIR']).mkdir()
 save(rank/f'environment-g{g}.json',env)
 save(D/'c20-rank'/f'lane-g{g}-plan.json',replace(read(D/'imported/transaction/c20-rank'/f'lane-g{g}-plan.json')))
T=D/'templates';(T/'configs').mkdir(parents=True)
for b in range(4):
 shutil.copy2(rank/'configs'/f'c20_grpo_u_tau1_cp180_b{b}.json',T/'configs'/f'grpo_u_tau1_cp180_b{b}.json')
for g in(4,5):shutil.copy2(rank/f'environment-g{g}.json',T/f'environment-g{g+2}.json')
plan=read(S/'plan.json');state=read(S/'status.json');rows=[]
for g in('4','5'):
 r=state['slots'][g];assert r['phase']=='RLT_RUNNING'and same(r['identity'])
 rows.append(dict(gpu=g,identity=r['identity'],request_path=r['request'],request=read(r['request'])))
save(D/'rlt-before.json',rows)
save(D/'prepared.json',dict(time=time.time(),source=str(R),roots={'rank':str(rank),'c20':str(sweep)},
 scheduler_plan_before=plan,source_host='sz1',gpus=[4,5],method_unchanged=True))
shutil.copy2('/data/chenyiteng/c20-prior-pool-20261020.json',D/'prior-c20-pool.json')
save(D/'repair-context.json',{'authorization':'User moved queued C20 rank and sweep to SZ2 4/5 before RLT','no_new_budget':True})
print(json.dumps({'prepared':str(D),'source':str(R),'configs':len(list((rank/'configs').glob('c20_*_b[0-3].json'))),'missing_weights':[c['strong_weights']for c in [read(p)for p in(rank/'configs').glob('c20_*_b0.json')]if c.get('strong_weights')and not Path(c['strong_weights']).exists()]}))
