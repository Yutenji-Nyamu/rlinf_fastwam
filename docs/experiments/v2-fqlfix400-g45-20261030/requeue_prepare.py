import json,os,shutil,sys,time,copy
from pathlib import Path
assert os.getuid()==1003 and os.uname().nodename=='admin'
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1');sys.path.insert(0,str(S/'ops'));from common import read,save,same
D=Path('/home/chenyiteng/deployment-20261030/v2-fqlfix400-sz1-g45');O=Path('/home/chenyiteng/results/pi05-distill/v2-fqlfix400-sz1-g45-20261030')
assert not D.exists() and not O.exists()
D.mkdir(parents=True);O.mkdir(parents=True)
P=Path('/home/chenyiteng/deployment-20261029/single-extend400-fqlfix-sz1-g67');V=Path('/home/chenyiteng/deployment-20261028/physical-v2-formal-sz1-g67')
R=D/'source';shutil.copytree(P/'source',R,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
for n in ['seeds'] :shutil.copytree(P/n,D/n)
shutil.copy2(P/'eval-template.json',D/'eval-template.json')
plan=read(P/'formal-plan.json');priorD=Path('/data/chenyiteng/deployment-20261025/endpoint-multitask-c10-sz1-g45');prior=read(priorD/'identity.json')
plan.update(methods=['dmd_v2','mfd_v2','fql_fixed_teacher'],predecessor_identity=prior,predecessor_deployment=str(priorD),initial_receipts={},extension_audits={},method_budget_epochs={'dmd_v2':400,'mfd_v2':400,'fql_fixed_teacher':400},checkpoint_reserve_gib={'dmd_v2':14,'mfd_v2':14,'fql_fixed_teacher':8},pending_cp_decision='Original CP full checkpoints were deleted; user asked fresh CP vs moving CP v2')
for method in ['dmd_v2','mfd_v2']:
 old=read(V/(method+'.json'));c=copy.deepcopy(old);c.update(epochs=400,checkpoint_steps=5201,evaluation_steps=list(range(2860,5201,260)))
 assert c['batch_size']==384 and c['microbatch_per_rank']==192 and c['loss_coordinate_scope']=='physical'
 ck=read(Path('/data/chenyiteng/results/pi05-distill/physical-v2-formal-sz1-g67-20261028')/method/'checkpoint.json')
 assert ck['step']==2600 and all(Path(ck[k]).is_file() and Path(ck[k]).stat().st_size>64*1024**2 for k in ['path','policy'])
 plan['initial_receipts'][method]=ck;save(D/(method+'.json'),c)
f=read(P/'fql_fixed_teacher.json');assert f['epochs']==2200 and f['evaluation_steps'][-1]==2200
f.update(epochs=4400,checkpoint_steps=4401,evaluation_steps=f['evaluation_steps']+list(range(2420,4401,220)))
assert len(f['evaluation_steps'])==27 and f['alpha']==30 and f['q_diagnostics'];save(D/'fql_fixed_teacher.json',f)
for g,oldg in [(4,6),(5,7)]:
 env=read(Path(read(P/'prepared.json')['output'])/f'environment-g{oldg}.json')
 for k in ['REPO_PATH','RLINF_CODE_WORKING_DIR']:env[k]=str(R)
 env['EMBODIED_PATH']=str(R/'examples/embodiment');env['PYTHONPATH']=str(R)+':'+env['ROBOTWIN_PATH'];env['TMPDIR']=str(D/f'tmp-g{g}')
 env.update(CUDA_VISIBLE_DEVICES=str(g),DOJO_GPU_SCOPE=str(g));Path(env['TMPDIR']).mkdir();save(O/f'environment-g{g}.json',env)
save(D/'prepared.json',dict(output=str(O),physical_gpus=[4,5],poll_seconds=.25));save(D/'formal-plan.json',plan)
save(D/'preparation-receipt.json',dict(time=time.time(),prepared=True,source_reused=str(P/'source'),gpu_started=False,prior_alive=same(prior),queued_methods=plan['methods']))
print(json.dumps({'deployment':str(D),'output':str(O),'prior_alive':same(prior),'source_files':sum(1 for p in R.rglob('*') if p.is_file()),'source_mib':sum(p.stat().st_size for p in R.rglob('*') if p.is_file())/2**20}))
