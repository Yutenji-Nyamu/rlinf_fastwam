import os,json,sys,subprocess,tarfile,time
from pathlib import Path
assert os.getuid()==1003
T=Path('/data/chenyiteng/deployment-20261017/pi05-policy-budget-eval-v1')
O=Path('/data/chenyiteng/results/pi05-policy-eval/pillbottle-20261017-v1')
BASE=Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-signals-sz1-20261003')
R=BASE.parent/'pi05-policy-budget-eval-20261017'
S=Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
assert not R.exists() and not T.exists() and not O.exists()
subprocess.run(['git','-C',str(BASE),'worktree','add','-b','codex/pi05-policy-budget-eval-20261017',str(R),'3319b5ed4fb4e8ad406cc583f2eaa33b687f9103'],check=True)
with tarfile.open('/data/chenyiteng/pi05-policy-eval-source1017.tar.gz') as t:
 assert all(m.isfile() and m.name.startswith('tools/pi05_policy_eval/') and '..' not in Path(m.name).parts for m in t.getmembers())
 t.extractall(R,filter='data')
T.mkdir(parents=True);(O/'configs').mkdir(parents=True);(O/'batches').mkdir()
sys.path.insert(0,str(R));sys.path.insert(0,str(S/'ops'));from common import read,save,same,gpus
import torch
old=Path('/data/chenyiteng/results/pi05-signals/20261003-v1')
base=read(old/'batches/move_pillbottle_pad_b0/config.json');task=base['task']
evals=read(R/'rlinf/envs/robotwin/seeds/eval_seeds.json')[task]['success_seeds']
train=read(R/'rlinf/envs/robotwin/seeds/train_seeds.json')[task]['success_seeds']
current=Path('/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/grpo-u-tau1-notrick-pillbottle256-sz1-g67-200-1014')
resolved=read(current/'runtime/resolved.yaml')
old_eval_cfg=resolved['env']['eval'];seed=old_eval_cfg['seed']
from rlinf.envs.robotwin.seed_utils import partition_success_seeds
prior=[]
for offset in (0,1):prior.extend(partition_success_seeds(torch.tensor(evals),base_seed=seed,seed_offset=offset,total_num_processes=2,num_group=16)[:16].tolist())
excluded=set(train)|set(evals[:32])|set(prior)
available=[int(s) for s in evals if s not in excluded]
assert len(available)>=192 and len(set(evals))==len(evals)
evaluation=available[:128];calibration=available[128:192]
save(O/'seeds.json',dict(task=task,eval_seeds=evaluation,calibration_seeds=calibration,excluded_prior_eval=prior,excluded_historical_signals=evals[:32],train_overlap=sorted(set(evaluation+calibration)&set(train)),selection='official order; excluded training and prior fixed eval/first32 historical signals; no outcome filtering'))
envbase=read(old/'environment.json')
for g in (6,7):
 e={k:v.replace(str(BASE),str(R)) if isinstance(v,str) else v for k,v in envbase.items()}
 e.update(CUDA_VISIBLE_DEVICES=str(g),DOJO_GPU_SCOPE=str(g),TMPDIR=str(T/f'tmp-g{g}'))
 Path(e['TMPDIR']).mkdir();save(O/f'environment-g{g}.json',e)
strong='/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-clean-half128-seed42-formal200-phys45-20260911-v1/pi05_grpo_clean_half128_seed42_formal200_phys45_20260911_v1/checkpoints/global_step_140/actor/model_state_dict/full_weights.pt'
assert Path(strong).is_file()
conditions=[('weak_n10_c50','weak',10,50),('strong_cp140','strong',10,50)]+[(f'weak_n{n}_c50','weak',n,50) for n in (5,15,20,30)]+[(f'weak_n10_c{c}','weak',10,c) for c in (10,20,30)]+[('calibration','calibration',10,50),('high_u','high',10,50),('low_u','low',10,50),('random','random',10,50)]
plans={g:dict(gpu=g,stages=[]) for g in (6,7)}
configs=[]
for name,policy,n,c in conditions:
 for g in (6,7):
  stage=dict(condition=name,configs=[])
  for b in ([g-6] if policy=='calibration' else [g-6,g-4]):
   selected=(calibration if policy=='calibration' else evaluation)[b*32:(b+1)*32];assert len(selected)==32
   seeds=O/'configs'/f'{name}_b{b}_seeds.json';save(seeds,{task:{'success_seeds':selected}})
   cfg=json.loads(json.dumps(base));out=O/'batches'/f'{name}_b{b}'
   cfg.update(condition=name,policy=policy,num_envs=32,gpu=g,batch=b,step_limit=200,num_steps=n,execute_chunk=c,noise_seed=421017+b*100003,gate_seed=170017+b*200003,source_commit='3319b5ed4fb4e8ad406cc583f2eaa33b687f9103',output=str(out),strong_weights=strong,threshold_path=str(O/'threshold.json'))
   ec=cfg['env'];ec.update(total_num_envs=32,max_steps_per_rollout_epoch=200,max_episode_steps=200,seeds_path=str(seeds));ec['task_config'].update(step_lim=200,save_path=str(out/'env'));ec['video_cfg']['video_base_dir']=str(out/'unused')
   path=O/'configs'/f'{name}_b{b}.json';save(path,cfg);stage['configs'].append(str(path));configs.append(str(path))
  plans[g]['stages'].append(stage)
for g,p in plans.items():save(T/f'lane-g{g}-plan.json',p)
plan=read(S/'plan.json');state=read(S/'status.json');owner=read(S/'owner-identity.json')
assert same(owner) and all(state['slots'][g]['phase']=='WAIT_EXTERNAL' for g in ('4','5','6','7'))
assert plan['slots']['6']['external_identity']==plan['slots']['7']['external_identity']
save(T/'prepared.json',dict(time=time.time(),source=str(R),output=str(O),source_parent='3319b5ed4fb4e8ad406cc583f2eaa33b687f9103',plan_before=plan,owner_before=owner,state_before=state,old_run=str(current),old_identity=plan['slots']['6']['external_identity'],old_release=plan['slots']['6']['external_release'],configs=configs,conditions=conditions,episodes=1600,physical_gpus=[6,7],per_card_parallelism=32))
print(json.dumps(dict(prepared=True,source=str(R),conditions=len(conditions),batches=len(configs),episodes=1600,unique_test_seeds=len(set(evaluation)),calibration_seeds=len(calibration),old_eval_seed=seed,owner=owner)))
