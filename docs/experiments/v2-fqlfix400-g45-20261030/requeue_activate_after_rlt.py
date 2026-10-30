import json,os,shutil,subprocess,sys,time
from pathlib import Path
assert os.getuid()==1003 and os.uname().nodename=='admin'
D=Path('/home/chenyiteng/deployment-20261030/v2-fqlfix400-sz1-g45');P=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67');S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'));from common import read,save,same
assert not (D/'activation.json').exists() and read(D/'validation.json')['passed']
plan=read(D/'formal-plan.json');prior=plan['predecessor_identity'];old67=read(P/'identity.json');rlt=read(S/'owner-identity.json');status=read(P/'training-queue-status.json')
assert not same(prior) and same(old67) and same(rlt) and status['method']!='fql_fixed_teacher'
assert not list(Path(read(P/'prepared.json')['output']).glob('summary-fql_fixed_teacher*'))
plan.pop('pending_cp_decision',None);plan['cp_decision']='User cancelled added original CP400; existing g67 CPv2 remains'
# Home must also hold the existing g67 extensions; use Data for this new pair.
prepared=read(D/'prepared.json');oldout=Path(prepared['output']);O=Path('/data/chenyiteng/results/pi05-distill/v2-fqlfix400-sz1-g45-20261030');assert not O.exists()
O.mkdir(parents=True)
for g in [4,5]:shutil.copy2(oldout/f'environment-g{g}.json',O/f'environment-g{g}.json')
prepared['output']=str(O);save(D/'prepared.json',prepared)
plan['storage']='Data for new g45 stages; existing g67 remains Home. New points latest2 plus stage best; prior full200 untouched.';save(D/'formal-plan.json',plan)
rp=read(S/'plan.json');assert all(rp['slots'][str(g)]['external_identity']==prior for g in [4,5])
assert all(read(S/'status.json')['slots'][str(g)]['phase']=='RLT_RUNNING' for g in [4,5])
save(D/'rlt-plan-before.json',rp);save(D/'g67-plan-before.json',read(P/'formal-plan.json'));save(D/'g67-fql-before.json',read(P/'fql_fixed_teacher.json'))
with (D/'queue.log').open('xb') as f:
 child=subprocess.Popen(['/usr/bin/python3','-u','-B',str(D/'source/tools/pi05_distillation/single_formal_queue.py'),str(D)],cwd=D/'source',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
deadline=time.monotonic()+10
while time.monotonic()<deadline:
 if (D/'identity.json').exists():
  me=read(D/'identity.json')
  if me['pid']==child.pid and same(me):break
 assert child.poll() is None;time.sleep(.1)
assert me['pid']==child.pid and same(me) and not same(prior)
# Retire the not-yet-started old entry. The already-loaded legacy owner will
# read an empty evaluation schedule at the method boundary, doing no training
# or evaluation. Its final=None/0-evals is routing completion, not a result.
retire=read(P/'fql_fixed_teacher.json');retire['evaluation_steps']=[];retire['queue_redirect']={'deployment':str(D),'budget_epochs':400,'reason':'User moved FQL-fix to g45; no execution at old slot'};save(P/'fql_fixed_teacher.json',retire)
p67=read(P/'formal-plan.json');p67['methods'].remove('fql_fixed_teacher');p67.get('method_budget_epochs',{}).pop('fql_fixed_teacher',None);p67['redirected_methods']={'fql_fixed_teacher':retire['queue_redirect']};save(P/'formal-plan.json',p67)
save(P/'moved-fql_fixed_teacher.json',dict(time=time.time(),status='MOVED_NOT_RUN',destination=str(D),budget_epochs=400,legacy_entry_zero_updates=True))
# Reuse the accepted exact-driver / namespace-cleanup handoff, with fresh
# RLT runtimes and latest complete restore points. Healthy g67 stays untouched.
sys.path.insert(0,str(D));from priority_pair_handoff import handoff
handoff_receipt=handoff(D,[4,5],me)
# Preserve root-volume emergency permits after the accepted registration.
rp=read(S/'plan.json');E=Path('/tmp/chenyiteng-rlt-handoff-20261030')
for g in [4,5]:
 slot=rp['slots'][str(g)];assert slot['external_identity']==me
 permit=E/f'advance-after-owner-exit-g{g}-v2fix400.json';save(permit,dict(time=time.time(),identity=me,armed=True,requires='exact queue owner exited AND GPU clear; not a completion receipt'));slot['external_release']=str(permit);slot['external_blockers']=[prior]
save(S/'plan.json',rp)
assert same(old67) and same(me) and same(read(S/'owner-identity.json'))
receipt=dict(time=time.time(),new_owner=me,unchanged_g45_predecessor=prior,unchanged_g67_owner=old67,rlt_handoff=handoff_receipt,g45=plan['methods'],g67=p67['methods'],budget_epochs=400,output=str(O),retention='latest2 plus stage best, preserve previous200',cp_added=False,free_gib={p:shutil.disk_usage(p).free/2**30 for p in ['/home','/data']},g67_training_unchanged=True,only_low_priority_g45_rlt_preempted=True)
save(D/'activation.json',receipt);print(json.dumps(receipt))
