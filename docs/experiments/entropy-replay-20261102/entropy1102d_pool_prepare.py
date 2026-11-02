import copy,json,os,shutil,time
from pathlib import Path
import yaml
D=Path('/home/chenyiteng/deployment-20261102/grpo-clean-entropy-g45')
p=json.loads((D/'plan.json').read_text());O=Path(p['output']);cfg=yaml.safe_load((D/'config/grpo_clean.yaml').read_text())
old=Path('/data/chenyiteng/results/pi05-signals/20261003-v1')
manifest=json.loads((old/'manifest.json').read_text())
base=None
for f in manifest['configs']:
 c=json.loads(Path(f).read_text())
 if c['task']=='move_pillbottle_pad':base=c;break
assert base is not None
allseeds=json.loads(Path(cfg['env']['eval']['seeds_path']).read_text())['move_pillbottle_pad']['success_seeds'][:32]
assert len(allseeds)==32 and len(set(allseeds))==32
(O/'pool').mkdir(exist_ok=True)
oldenv=json.loads(Path('/data/chenyiteng/results/pi05-noise-video/20261019-sz1-v2/environment-g4.json').read_text())
for i,gpu in enumerate([4,5]):
 c=copy.deepcopy(base);dest=O/'pool'/f'g{gpu}'
 seedfile=D/f'pool-seeds-g{gpu}.json';seedfile.write_text(json.dumps({'move_pillbottle_pad':{'success_seeds':allseeds[i*16:(i+1)*16]}}))
 c.update(task='move_pillbottle_pad',gpu=gpu,batch=i,num_envs=16,output=str(dest),step_limit=200,model=copy.deepcopy(cfg['actor']['model']),policy_weights=str(Path(p['checkpoint_root'])/'global_step_200/actor/model_state_dict/full_weights.pt'),noise_samples=4,noise_seed=42+i,acceptance=True,nstep_signals=False,save_replay_observations=True)
 ec=copy.deepcopy(cfg['env']['eval']);ec.update(total_num_envs=16,seeds_path=str(seedfile),auto_reset=False,ignore_terminations=True)
 ec['video_cfg'].update(save_video=False,video_base_dir=str(dest/'unused'))
 ec['task_config'].update(save_path=str(dest/'env'),eval_video_log=False,step_lim=200)
 c['env']=ec
 (D/f'pool-g{gpu}.json').write_text(json.dumps(c,indent=2))
 env={k:v.replace(oldenv['REPO_PATH'],p['inference_source']) if isinstance(v,str) else v for k,v in oldenv.items()}
 env.update(CUDA_VISIBLE_DEVICES=str(gpu),DOJO_GPU_SCOPE=str(gpu),TMPDIR=str(D/f'tmp-g{gpu}'))
 Path(env['TMPDIR']).mkdir(exist_ok=True);env.pop('RAY_ADDRESS',None)
 (D/f'inference-environment-g{gpu}.json').write_text(json.dumps(env))
source=Path(p['inference_source'])
(D/'tools/pi05_u_video').mkdir(exist_ok=True)
for name in ['recording.py','noise_samples.py']:shutil.copy2(source/'tools/pi05_u_video'/name,D/'tools/pi05_u_video'/name)
print(json.dumps({'pool':32,'per_gpu':16,'source':'accepted noise4 + full video collector','weight':'CP200','step_limit':200,'horizon':50,'execute_chunk':50,'new_fields':['strict full-state checkpoint overlay','saved instructions for replay'],'seeds':allseeds}))
