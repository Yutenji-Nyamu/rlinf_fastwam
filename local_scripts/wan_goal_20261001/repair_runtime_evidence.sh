set -eu
/data/chenyiteng/projects/wan-goal-sz3/envs/pi05-wan/bin/python -B - <<'PY'
import json,math,re,subprocess,sys,time
from pathlib import Path
sys.modules.setdefault('tensorflow',None)
R=Path('/data/chenyiteng/projects/wan-goal-sz3');P=Path('/data/chenyiteng/projects/robodojo-openwam-sz3')
D=P/'runs/sz3_pi05_official_6300_n4_dual_20260929_r2'
W=R/'runs/wan-goal-sz3-20261001-r6';F=W/'pi05-formal'
sys.path.insert(0,str(R/'scripts/resource_switch'))
from common import account,alive,read,sha
account();active=read(D/'active-continuation.json');ready=read(D/'prepare-continuation-20261001-wan-goal-v6/ready.json')
assert active['attempt_dir']==str(D/'continuation-20261001-wan-goal-v6') and alive(active,command=True)
assert ready['restore_rlt_after_dojo'] and not ready['reuse_borrowed_cycle'] and ready['benchmark_unchanged'] and ready['return_rlt_direct']
placement=read(F/'verified-placement.json')
from omegaconf import OmegaConf
cfgfile=F/'tensorboard/config.yaml';cfg=OmegaConf.load(cfgfile)
assert cfg.runner.max_epochs==1000 and cfg.runner.resume_dir is None
assert cfg.env.train.total_num_envs==64 and cfg.algorithm.group_size==8 and cfg.env.train.rollout_epoch==8
assert cfg.actor.global_batch_size==2048 and cfg.actor.micro_batch_size==128
assert cfg.actor.model.openpi_data.wrist_mode=='disabled' and cfg.algorithm.filter_rewards is True
assert cfg.env.train.max_episode_steps==320 and cfg.actor.model.num_action_chunks==8
assert cfg.actor.model.openpi.action_horizon==10 and cfg.actor.model.openpi.num_steps==5
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tb=next((p for p in (F/'tensorboard/all',F/'tensorboard') if list(p.glob('events.out.tfevents.*'))))
acc=EventAccumulator(str(tb),size_guidance={'scalars':0});acc.Reload();tags=acc.Tags().get('scalars',[])
metrics={}
for tag in tags:
    if any(t in tag for t in ('grad_norm','loss_mask_fraction','advantages_min','advantages_max','total_loss','success_once','time/step')):
        metrics[tag]=[{'step':p.step,'value':p.value if math.isfinite(p.value) else str(p.value),'wall_time':p.wall_time} for p in acc.Scalars(tag)]
protocol={
 'max_epochs':cfg.runner.max_epochs,'resume_dir':cfg.runner.resume_dir,'save_interval':cfg.runner.save_interval,
 'N':cfg.env.train.total_num_envs,'G':cfg.algorithm.group_size,'R':cfg.env.train.rollout_epoch,
 'episode_steps':cfg.env.train.max_episode_steps,'chunk':cfg.actor.model.num_action_chunks,
 'action_horizon':cfg.actor.model.openpi.action_horizon,'num_steps':cfg.actor.model.openpi.num_steps,
 'global_batch':cfg.actor.global_batch_size,'micro_batch':cfg.actor.micro_batch_size,
 'wrist_mode':cfg.actor.model.openpi_data.wrist_mode,'filter_rewards':cfg.algorithm.filter_rewards,
 'learning_rate':cfg.actor.optim.lr,'model_path':cfg.actor.model.model_path,
 'component_placement':OmegaConf.to_container(cfg.cluster.component_placement),
 'real_libero_evaluation_interval':cfg.runner.val_check_interval}
log=F/'command.log';text=log.read_bytes()[-20000:].decode(errors='replace')
report={'time':time.time(),'run_dir':str(F),'owner_alive':True,'pipeline_phase':read(D/'pipeline-current.json')['phase'],
 'sequence':read(W/'sequence-current.json'),'protocol':protocol,'placement':placement,
 'resolved_config_sha256':sha(cfgfile),'source_yaml_sha256':sha(R/'RLinf-pi05/examples/embodiment/config/wan_goal_pi05_headonly_formal_sz3.yaml'),
 'restoration':{'unique_continuation':active['attempt_dir'],'cycle':active['cycle_dir'],
   'restore_rlt_after_dojo':True,'after_wm':'RLT_DIRECT','resume_dojo':False,'preserved_episodes_at_prepare':ready['preserved_episodes'],
   'ready_sha256':sha(D/'prepare-continuation-20261001-wan-goal-v6/ready.json')},
 'metrics':metrics,'normal_exit':read(F/'wm-exit.json') if (F/'wm-exit.json').is_file() else None,
 'recent_primary_error_lines':[s[-400:] for s in text.splitlines() if 'Traceback' in s or 'OutOfMemoryError' in s or 'InductorError' in s],
 'limitation':'WM reward-model success is not real LIBERO success; formal checkpoint first saved at epoch40.'}
progress=re.findall(r'Generating Rollout Epochs:\s+\d+%.*?\|\s*(\d+)/(\d+)\s+\[([^\]]+)\]',text)
report['recent_rollout_progress']=[{'completed':int(a),'total':int(b),'timing':c} for a,b,c in progress[-2:]]
report['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
report['memory']={k:int(v.split()[0])*1024 for k,v in (s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','MemTotal')}
report['completed_runner_epochs']=max((p['step']+1 for p in metrics.get('train/actor/grad_norm',[])),default=0)
report['exceeded_previous_failure_point']=report['completed_runner_epochs']>4
mask={p['step']:p['value'] for p in metrics.get('rollout/loss_mask_fraction',[])}
report['effective_update_steps']=[p['step'] for p in metrics.get('train/actor/grad_norm',[])
    if isinstance(p['value'],(int,float)) and p['value']>0 and isinstance(mask.get(p['step']), (int,float)) and mask[p['step']]>0]
report['checkpoint_directories']=[str(p) for p in (F/cfg.runner.logger.experiment_name/'checkpoints').glob('global_step_*') if p.is_dir()]
report['monitor_diagnostics']={p.name:read(p) for p in F.glob('wm-*-error.json')}
report['route']='WM_RLT_DIRECT in tested unique owner; no separate guard'
print(json.dumps(report,allow_nan=False))
PY
