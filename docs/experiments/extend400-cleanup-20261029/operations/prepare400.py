import copy,json,math,os,shutil,sys,time
from pathlib import Path
import torch
assert os.getuid()==1003
base=Path('/home/chenyiteng/deployment-20261028')
B=base/'physical-v2-formal-sz1-g67'
D=Path('/home/chenyiteng/deployment-20261029/single-extend400-sz1-g67')
O=Path('/home/chenyiteng/results/pi05-distill/single-extend400-sz1-g67-20261029')
def read(p):return json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
assert not (D/'identity.json').exists()
assert shutil.disk_usage('/home').free>450*2**30
D.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
if not (D/'source').exists():shutil.copytree(B/'source',D/'source',ignore=shutil.ignore_patterns('__pycache__','.git'))
sys.path.insert(0,str(D/'source'))
from rlinf.algorithms.pi05_distillation.resume_extension import validate_resume
save(D/'prepared.json',dict(output=str(O),physical_gpus=[6,7],poll_seconds=.25))
save(D/'eval-template.json',read(B/'eval-template.json'))
if not (D/'seeds').exists():shutil.copytree(B/'seeds',D/'seeds')
BO=Path(read(B/'prepared.json')['output'])
for g in [6,7]:
 env=read(BO/f'environment-g{g}.json')
 env={k:v.replace(str(B/'source'),str(D/'source')) if isinstance(v,str) else v for k,v in env.items()}
 env.update(TMPDIR=str(D/f'tmp-g{g}'));Path(env['TMPDIR']).mkdir(exist_ok=True)
 save(O/f'environment-g{g}.json',env)
save(O/'baseline-reference.json',read(BO/'baseline-reference.json'))
plan=copy.deepcopy(read(B/'formal-plan.json'))
plan.update(methods=['endpoint','cp','onedp','mfd'],budget_epochs=400,predecessor_identity=read(B/'identity.json'),predecessor_deployment=str(B),initial_receipts={},extension_audits={},storage='Home; retain all new evaluation points; existing 200-epoch checkpoints unchanged',evaluation_budget_epochs=list(range(220,401,20)))
for method,previous in [('endpoint','single-extend200-sz1-g67'),('cp','single-extend200-sz1-g67'),('onedp','dmd-extend200-sz1-g67'),('mfd','mfd-extend200-sz1-g67')]:
 p=base/previous;old=read(p/f'{method}.json');oldout=Path(read(p/'prepared.json')['output'])
 receipt=read(p/'queue-complete.json')['outcomes'][method]['result']['final']
 ck=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False)
 assert ck['config']==old and old['epochs']==200
 assert ck['optimizer']['state'] and len(ck['rng'])==2 and Path(receipt['policy']).is_file()
 if method in ['onedp','mfd']:assert ck['auxiliary'] and ck['aux_optimizer']['state']
 if method=='cp':assert ck['auxiliary'] and ck['aux_optimizer'] is None
 per_epoch=math.ceil(ck['windows']/old['batch_size'])
 c=copy.deepcopy(old);c.update(epochs=400,checkpoint_steps=400*per_epoch+1,checkpoint_keep_steps=[n*per_epoch for n in range(220,401,20)])
 audit=validate_resume(ck,c,ck['windows'],2,extend_completed=True)
 save(D/f'{method}.json',c)
 plan['initial_receipts'][method]=receipt;plan['extension_audits'][method]=audit
 del ck
save(D/'formal-plan.json',plan)
save(D/'resume-state-check.json',plan['extension_audits'])
print(json.dumps({'deployment':str(D),'output':str(O),'audits':plan['extension_audits'],'eval_epochs':plan['evaluation_budget_epochs']}))
