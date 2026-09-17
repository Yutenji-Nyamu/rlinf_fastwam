"""Fresh 800-round RLT tau2.5/tau3 preparation and explicit single-card launch."""
import copy,importlib.util,json,os,subprocess,sys
from pathlib import Path
ST=Path('/data/chenyiteng/results/server-maintenance-20260917/rlt-t25-t3-half4-800')
sp=importlib.util.spec_from_file_location('proven_ops','/data/chenyiteng/results/server-maintenance-20260913/method-cutover/ops.py')
b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
assert b.sha(sp.origin)=='0ea45901647183cd5036694056433192f797ed5f4eb055b3ffe91339066d1876'
b.ST=ST;b.OPS=ST/'ops.py';b.SPECS=b.read(ST/'specs.json')

def environment(key,run):
 s=b.SPECS[key];env=b.read(Path(s['control_run'])/'runtime/environment.json')
 return {k:v.replace(s['control_root'],s['root']).replace(s['control_run'],str(run)) for k,v in env.items()}
b.environment=environment

def prepare(key):
 from omegaconf import OmegaConf
 s=b.SPECS[key];receipt=b.source_check(key);control=Path(s['control_run'])
 baseline=OmegaConf.to_container(OmegaConf.load(control/'tensorboard/config.yaml'),resolve=True)
 assert not b.diff(baseline,OmegaConf.to_container(OmegaConf.load(control/'runtime/resolved.yaml'),resolve=True))
 clean_path=Path(s['clean_run'])/'tensorboard/config.yaml';clean=OmegaConf.to_container(OmegaConf.load(clean_path),resolve=True)
 def repl(v):
  if isinstance(v,dict):return {k:repl(x) for k,x in v.items()}
  if isinstance(v,list):return [repl(x) for x in v]
  return v.replace(s['control_root'],s['root']).replace(s['control_run'],s['run']) if isinstance(v,str) else v
 cfg=repl(copy.deepcopy(baseline));cfg['runner']['logger']['experiment_name']=Path(s['run']).name
 cfg['runner']['resume_dir']=None;cfg['runner']['ckpt_path']=None;cfg['runner']['max_steps']=800
 method=cfg['algorithm']['rlt_dvac'];method.update(mode='apply',factor_mapping='exp_mean',temperature_local=s['temperature'],temperature_chunk=s['temperature'],alpha_local=1.,alpha_chunk=1.,success_scale=1.)
 for name in ('success_scale_schedule','direction_schedule'):method.pop(name,None)
 cfg['rollout']['rlt_feature_model']['openpi']['rlt_dvac_mode']='apply'
 cfg['cluster']['component_placement']['actor,env,rollout']=s['gpus'][0]
 identity={'rollout.rlt_feature_model.openpi.rlt_dvac_mode','runner.logger.experiment_name','runner.logger.log_path','runner.resume_dir','runner.ckpt_path','runner.max_steps','cluster.component_placement.actor,env,rollout'}|{f'env.{p}.{n}' for p in ('train','eval') for n in ('seeds_path','task_config.save_path','video_cfg.video_base_dir')}
 delta=b.diff(baseline,cfg);assert all(k in identity or k in ('algorithm.rlt_dvac.temperature_local','algorithm.rlt_dvac.temperature_chunk') for k in delta),delta
 clean_diff=b.diff(clean,cfg);assert all(k in identity or k.startswith('algorithm.rlt_dvac.') for k in clean_diff),clean_diff
 assert cfg['runner']['max_steps']==800 and cfg['runner']['val_check_interval']==cfg['runner']['save_interval']==25
 assert cfg['actor']['global_batch_size']==512 and cfg['actor']['micro_batch_size']==256 and cfg['algorithm']['update_epoch']==5
 assert cfg['env']['train']['total_num_envs']==4 and cfg['env']['train']['rollout_epoch']==1 and cfg['algorithm']['replay_buffer']['cache_size']==80000
 assert cfg['actor']['optim']['lr']==cfg['actor']['critic_optim']['lr']==1e-4 and cfg['algorithm']['critic_actor_ratio']==2
 sched=cfg['algorithm']['rlt_schedule'];weights=cfg['algorithm']['actor_weight_schedule']
 assert [sched[k] for k in ('warmup_min_size','warmup_post_collect_updates','max_updates_per_train_step')]==[10000,15000,800]
 assert [weights[k] for k in ('warmup_updates','ramp_updates')]==[10000,25000]
 assert method['mapping']=='two_level_batch' and method['success_scale']==1 and method['temperature_local']==method['temperature_chunk']==s['temperature']
 assert not any(k.endswith('schedule') for k in method)
 assert cfg['env']['eval']['total_num_envs']*cfg['env']['eval']['rollout_epoch']==20
 assert cfg['rollout']['rlt_feature_model']['model_path']==clean['rollout']['rlt_feature_model']['model_path'] and Path(cfg['rollout']['rlt_feature_model']['model_path']).exists()
 assert '2000' in cfg['rollout']['rlt_feature_model']['model_path']
 for p in ('train','eval'):assert b.sha(cfg['env'][p]['seeds_path'])==b.sha(clean['env'][p]['seeds_path'])
 pre=ST/key/'prepared-formal';assert not pre.exists() and not Path(s['run']).exists();pre.mkdir()
 OmegaConf.save(OmegaConf.create(cfg),pre/'resolved.yaml',resolve=True)
 b.save(pre/'environment.json',environment(key,s['run']));b.save(pre/'baseline-diff.json',delta);b.save(pre/'clean4-diff.json',clean_diff)
 argv=[b.PY,'-u','-B',str(ST/'ops.py'),'driver',key,'formal','--config-path',str(Path(s['run'])/'runtime'),'--config-name','resolved','hydra.run.dir=.','hydra.output_subdir=null','hydra.job.chdir=false','hydra/job_logging=stdout']
 b.save(pre/'argv.json',argv);(pre/'command.txt').write_text(b.shlex.join(argv)+'\n');(pre/'command.sh').write_text('#!/usr/bin/env bash\nexec '+b.shlex.join(argv)+'\n')
 (pre/'wrapper.sh').write_text('#!/usr/bin/env bash\nset +e\nruntime=$1\nulimit -n 4096 || exit 91\ndate --iso-8601=seconds > "$runtime/started_at.txt"\nbash "$runtime/command.sh" > "$runtime/driver.log" 2>&1\nrc=$?\nprintf "%s\\n" "$rc" > "$runtime/exit_code.txt"\ndate --iso-8601=seconds > "$runtime/finished_at.txt"\nexit "$rc"\n')
 for n in ('command.sh','wrapper.sh'):subprocess.run(['bash','-n',str(pre/n)],check=True)
 env=b.read(pre/'environment.json');assert all(Path(p).exists() for p in env['PYTHONPATH'].split(':') if p)
 b.save(pre/'contract.json',{'time':b.now(),'key':key,'mode':'formal','source_head':receipt['head'],'source_hashes':receipt['source_hashes'],'baseline_path':str(control/'tensorboard/config.yaml'),'baseline_sha256':b.sha(control/'tensorboard/config.yaml'),'clean4_baseline_path':str(clean_path),'clean4_baseline_sha256':b.sha(clean_path),'ops_sha256':b.sha(ST/'ops.py'),'root':s['root'],'run':s['run'],'namespace':s['namespace'],'gpus':s['gpus'],'max_steps':800,'fresh':True,'resume_checkpoint':None,'resume_round':0,'baseline_diff':delta,'clean4_diff':clean_diff,'stop':'round800, explicit user stop, or unrecoverable runtime error; no performance threshold','files':{n:b.sha(pre/n) for n in ('resolved.yaml','environment.json','argv.json','command.sh','wrapper.sh')}})
 return {'key':key,'prepared':str(pre),'passed':True,'run':s['run'],'namespace':s['namespace'],'gpus':s['gpus'],'tau':s['temperature'],'source_head':receipt['head'],'baseline_diff':delta,'clean4_diff':clean_diff,'command':b.shlex.join(argv),'launch_command':b.shlex.join([b.PY,'-u','-B',str(ST/'ops.py'),'launch',key]),'training_started':False}

def launch(key):
 assert key in ('t25','t3');p=b.launch(key,'formal');print(json.dumps({'key':key,'launched':True,'wrapper':b.proc(p.pid),'receipt':str(ST/key/'formal-launch.json')}),flush=True)

if __name__=='__main__':
 assert os.getuid()==1003
 if sys.argv[1]=='driver':b.driver(sys.argv[2],sys.argv[3])
 elif sys.argv[1]=='launch':launch(sys.argv[2])
 else:raise ValueError('Use launch <key> only after the root controller confirms/stops the intended old GPU job.')
