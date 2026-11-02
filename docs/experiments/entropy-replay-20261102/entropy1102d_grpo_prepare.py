import json,os,shutil,sys,time
from pathlib import Path
import yaml
D=Path('/home/chenyiteng/deployment-20261102/grpo-clean-entropy-g45')
old=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-clean-half128-seed42-formal200-phys45-20260911-v1')
run=old.with_name('grpo-clean-entropy-g45-20261102')
O=Path('/data/chenyiteng/results/pi05-entropy/20261102-clean-sixpoint')
assert os.getuid()==1003 and not D.exists() and not run.exists()
D.mkdir();(D/'tools').mkdir();(D/'config').mkdir();(run/'runtime').mkdir(parents=True);O.mkdir(parents=True)
for n in ['grpo_entropy_queue.py','grpo_entropy_driver.py']:shutil.copy2(D.parent/n,D/'tools'/n)
cfg=yaml.safe_load((old/'runtime/resolved.yaml').read_text())
name=run.name
def replace(x):
 if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
 if isinstance(x,list):return [replace(v) for v in x]
 if isinstance(x,str):return x.replace(str(old),str(run))
 return x
new=replace(cfg)
new['runner']['logger']['experiment_name']=name
new['runner']['save_interval']=40
for key in ['actor','rollout','env']:new[key]['group_name']=key+'_entropy_clean_g45_1102'
(D/'config/grpo_clean.yaml').write_text(yaml.safe_dump(new,sort_keys=False))
(D/'baseline.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(run/'runtime/resolved.yaml').write_text(yaml.safe_dump(new,sort_keys=False))
env=replace(json.loads((old/'runtime/environment.json').read_text()))
for key in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY']:env[key]='http://127.0.0.1:17890'
(D/'training-environment.json').write_text(json.dumps(env))
p={'run':str(run),'output':str(O),'training_source':'/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-clean-half-seed42-20260911','legacy_ops':'/data/chenyiteng/results/server-maintenance-20260911/grpo-seed42-half/ops.py','python':'/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python','namespace':'RLinf_grpo_clean_entropy_g45_20261102','checkpoint_root':str(run/name/'checkpoints'),'predecessor_identity':json.loads((D.parent/'entropy-replay-g45-v3/identity.json').read_text()),'gpus':[4,5],'total_rounds':200,'milestones':[0,40,80,120,160,200],'pool_episodes':32,'pool_num_envs_per_card':16,'base_model':new['actor']['model']['model_path'],'inference_source':'/home/chenyiteng/deployment-20261029/fqlfix-shadow-sz1/source','retention':'retain five complete milestone checkpoints plus existing base; exempt from distillation rolling cleanup','expected_checkpoint_gib':134.3,'failure':'exact namespace cleanup; preserve partial outputs; advance to single-task FQL then RLT'}
(D/'plan.json').write_text(json.dumps(p,indent=2))
# Only paths, run/group names and the user-requested five save intervals differ.
def flat(x,prefix=''):
 if isinstance(x,dict):return {k:v for n,a in x.items() for k,v in flat(a,prefix+'.'+n).items()}
 return {prefix:x}
a,b=flat(cfg),flat(new);diff={k:[a.get(k),b.get(k)]for k in a.keys()|b.keys()if a.get(k)!=b.get(k)}
(D/'config-diff.json').write_text(json.dumps(diff,indent=2))
assert new['runner']['max_steps']==200 and new['env']['train']['task_config']['task_name']=='move_pillbottle_pad'
assert new['actor']['global_batch_size']==512 and new['algorithm']['dvac_gradient_weighting']['mode']=='off'
assert Path(p['legacy_ops']).is_file() and Path(p['training_source']).is_dir()
print(json.dumps({'deployment':str(D),'plan':p,'changed_keys':list(diff),'ready_training':True}))
