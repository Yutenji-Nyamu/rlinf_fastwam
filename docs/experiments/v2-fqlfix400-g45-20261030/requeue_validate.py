import json,os,subprocess,sys,time,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
D=Path('/home/chenyiteng/deployment-20261030/v2-fqlfix400-sz1-g45');R=D/'source';P=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67')
sys.path[:0]=[str(R),str(R/'tools/pi05_distillation')]
from rlinf.algorithms.pi05_distillation.resume_extension import validate_resume
from single_formal_plan import validate_post_eval_retention,validate_reused_fql_replay
from run_queue import read,save
plan=read(D/'formal-plan.json');O=Path(read(D/'prepared.json')['output']);audits={}
import torch
for m in ['dmd_v2','mfd_v2']:
 ck=torch.load(plan['initial_receipts'][m]['path'],map_location='cpu',weights_only=False,mmap=True)
 c=read(D/(m+'.json'));audits[m]=validate_resume(ck,c,4738,2,extend_completed=True)
 assert all(k in ck for k in ['student','auxiliary','optimizer','aux_optimizer','rng']) and len(ck['rng'])==2
 validate_post_eval_retention(m,c,plan,O/m);del ck
f=read(D/'fql_fixed_teacher.json');counts=validate_reused_fql_replay(read(f['transition_manifest']),plan['heldout_seeds'])
assert math.ceil(counts['transitions']/f['batch_size'])*f['epochs']==4400
validate_post_eval_retention('fql_fixed_teacher',f,plan,O/'fql_fixed_teacher')
assert f['evaluation_steps'][-1]==4400 and f['alpha']==30 and f['batch_size']==448
plan['extension_audits']=audits;plan['fql_budget_receipt']={'budget_epochs':400,'updates':4400,'evaluation_points':27,'data':counts};save(D/'formal-plan.json',plan)
cmd=[sys.executable,'-B','-m','pytest','-q','tools/pi05_distillation/test_resume_extension.py','tools/pi05_distillation/test_single_formal.py','tools/pi05_distillation/test_queue_storage.py']
p=subprocess.run(cmd,cwd=R,env=dict(os.environ,PYTHONPATH=str(R)+':'+str(R/'tools/pi05_distillation')),capture_output=True,text=True)
(D/'tests.log').write_text(p.stdout+p.stderr);print(p.stdout[-1500:]+p.stderr[-500:]);assert p.returncode==0
save(D/'validation.json',dict(time=time.time(),passed=True,extension_audits=audits,fql=plan['fql_budget_receipt'],gpu_used=False,tests=p.stdout.strip()))
print(json.dumps({'validation':'PASS','dmd_mfd_steps':[2600,5200],'fql_updates':4400,'fql_eval_points':27,'CP':'pending user clarification'}))
