"""Finite native -> WM -> RM -> ten RL rounds sequence, GPU4/5 only.

Reuses existing scoped process cleanup.
No fallback scheduler or GPU lease. Phase changes are direct child completion events.
"""
import argparse
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid


def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
    plan=read(a.plan); root=Path(plan['root']);code=root/'code';out=root/'run'
    spec=importlib.util.spec_from_file_location('existing_owner',plan['existing_owner'])
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    origin=read(plan['origin_plan']);old.load_lifecycle(origin);H=old.H
    assert os.getuid()==20001
    old.pidfd_probe()
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=uuid.uuid4().hex;catalog=old.Catalog(out/'processes.json',token)
    save(out/'owner.json',dict(H.proc(os.getpid()),token=token))
    raw_env=read(origin['environment_file']);scope=read(root/'scope.json')
    env=dict(raw_env,**scope); env['PYTHONPATH']=str(code)+':'+plan['repo']+':'+env.get('PYTHONPATH','')
    for k in old.MASKS:env.pop(k,None)
    gpu_env=dict(raw_env,CUDA_VISIBLE_DEVICES='4,5',CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONPATH=str(code)+':'+plan['opendw'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for k in ['LD_PRELOAD','RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST']:gpu_env.pop(k,None)
    native_base=read(root/'native_base.json');rl_base=read(root/'rl_base.json')
    seeds=read(root/'seed_plan.json')
    def stopped(sig,frame):raise RuntimeError('Owner received signal '+str(sig))
    signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
    namespaces=['initial']+[f'native{i:03d}' for i in range(10,200,10)]+[f'rl{i:03d}' for i in range(10,201,10)]
    allow=dict(origin,owner_dir=str(out),management_namespace='wmcycle1009_ops',trials=[dict(namespace='wmcycle1009_'+n) for n in namespaces])
    old.add_allowlist(allow)
    def launch(argv,phase,child_env,cwd=None):
        target=out/phase;target.mkdir(parents=True,exist_ok=True)
        save(out/'status.json',dict(time=H.now(),phase=phase,argv=argv))
        with (target/'process.log').open('a') as log:
            child=subprocess.Popen(argv,cwd=cwd or str(root),env=dict(child_env,**{old.TOKEN:token,old.PHASE:phase}),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        catalog.add(H.proc(child.pid),phase,'exact Popen child')
        return child
    def run(argv,phase,child_env,cwd=None,optional_oom=False):
        started=time.monotonic();child=launch(argv,phase,child_env,cwd)
        while child.poll() is None:time.sleep(5)
        old.cleanup(origin,catalog)
        result=dict(exit_code=child.returncode,seconds=time.monotonic()-started)
        save(out/phase/'result.json',result)
        if child.returncode:
            log=(out/phase/'process.log').read_text(errors='replace')
            if optional_oom and ('out of memory' in log.lower() or 'outofmemoryerror' in log.lower()):return None
            raise RuntimeError(phase+' failed: '+str(child.returncode))
        return result
    def native(phase,n,ids,checkpoint=None,smoke=False):
        target=out/phase;target.mkdir(parents=True,exist_ok=True)
        cfg=copy.deepcopy(native_base);e=cfg['env']['eval'];r=cfg['runner']
        seed_file=target/'seeds.json';entry=copy.deepcopy(seeds['source_entry']);entry['success_seeds']=ids
        save(seed_file,{'lift_pot':entry})
        e.update(total_num_envs=n,rollout_epoch=len(ids)//n,seeds_path=str(seed_file),
            use_fixed_reset_state_ids=False,max_steps_per_rollout_epoch=64 if smoke else 384)
        e['task_config']['save_path']=str(target/'robotwin_data')
        cfg['cluster']['component_placement']={'env':'4-5' if n==64 else '5','rollout':'4'}
        cfg['env']['group_name']='WMCycleEnv_'+phase;cfg['rollout']['group_name']='WMCycleRollout_'+phase
        r.update(ckpt_path=checkpoint,resume_dir=None,per_worker_log_path=str(target/'worker_metrics'))
        r['logger'].update(log_path=str(target),experiment_name=phase)
        config=target/'config.json';save(config,cfg)
        argv=[plan['rl_python'],'-u','-B',str(code/'native_driver.py'),'--config',str(config),
            '--receipt-dir',str(target),'--private-repo',plan['repo'],'--environment-fragment',str(root/'scope.json'),
            '--capture-dir',str(target/'capture'),'--namespace','wmcycle1009_'+phase]
        result=run(argv,phase,env,plan['repo'],optional_oom=smoke)
        if result:
            eps=list((target/'capture').glob('*/episode.json'));records=[read(p) for p in eps]
            result.update(episodes=len(records),complete=sum(int(x['complete']) for x in records),
                chunks=sum(len(x['chunks']) for x in records),successes=sum(int(x['success']) for x in records))
            save(target/'result.json',result)
            if smoke:assert result['chunks']>=n, 'No complete native action blocks'
            else:assert result['complete']==len(ids), 'Incomplete native collection'
        return result
    def pack(phase,roots,partial=False):
        argv=[plan['wm_python'],'-u',str(code/'cycle_data.py'),'--roots',*map(str,roots),
            '--output',str(out/phase/'data'),'--seed-plan',str(root/'seed_plan.json')]
        if partial:argv.append('--allow-partial')
        run(argv,phase+'_pack',dict(gpu_env,CUDA_VISIBLE_DEVICES=''))
        return out/phase/'data'
    def wm(phase,dataset,checkpoint,batch,steps=None,optional=False):
        argv=[plan['wm_python'],'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(code/'wm_train.py'),
            '--bundle',plan['bundle'],'--checkpoint',checkpoint,'--dataset',str(dataset),
            '--output',str(out/phase/'train'),'--batch',str(batch),'--epochs','5']
        if steps:argv.extend(['--steps',str(steps)])
        result=run(argv,phase,gpu_env,optional_oom=optional)
        return read(out/phase/'train/result.json') if result else None
    def policy(end,checkpoint,wm_checkpoint,rm_checkpoint,threshold):
        phase=f'rl{end:03d}';target=out/phase;target.mkdir(parents=True,exist_ok=True)
        bundle=target/'bundle';bundle.mkdir()
        for asset in Path(plan['bundle']).iterdir():
            (bundle/asset.name).symlink_to(wm_checkpoint if asset.name=='model.pt' else asset,target_is_directory=asset.is_dir())
        service=copy.deepcopy(origin['services'][0]);argv=service['argv'];argv[argv.index('--bundle')+1]=str(bundle)
        argv[argv.index('--reward-checkpoint')+1]=rm_checkpoint;argv[argv.index('--output-dir')+1]=str(target/'wm_records')
        service_env=dict(raw_env,**service['environment'],CUDA_VISIBLE_DEVICES='5')
        server=launch(argv,phase+'_service',service_env,service['cwd'])
        deadline=time.monotonic()+1200
        while True:
            if server.poll() is not None:raise RuntimeError('WM service exited loading '+phase)
            try:health=old.http(service['url'])
            except (OSError,ValueError):health=None
            if health and health.get('ok') and health.get('pid')==server.pid:break
            if time.monotonic()>deadline:raise TimeoutError('WM service startup')
            time.sleep(5)
        cfg=copy.deepcopy(rl_base);r=cfg['runner']
        r.update(max_epochs=200,max_steps=end,resume_dir=checkpoint,save_interval=10,val_check_interval=10,
            per_worker_log_path=str(target/'worker_metrics'))
        r['logger'].update(log_path=str(target),experiment_name=phase)
        for component in ['env','actor','rollout']:cfg[component]['group_name']='WMCycle_'+phase+'_'+component
        cfg['env']['train']['success_reward_threshold']=threshold
        save(target/'config.yaml',cfg)
        driver_plan=dict(origin,owner_dir=str(out),repo=plan['repo'],token=token,graphics_fragment=str(root/'scope.json'),
            trials=[dict(key=phase,config=str(target/'config.yaml'),namespace='wmcycle1009_'+phase)])
        save(target/'driver_plan.json',driver_plan)
        driver=launch([plan['rl_python'],'-u','-B',plan['existing_owner'],'--plan',str(target/'driver_plan.json'),
            'driver','--key',phase],phase,env,plan['repo'])
        while driver.poll() is None:
            if server.poll() is not None:raise RuntimeError('WM exited during '+phase)
            time.sleep(5)
        assert driver.returncode==0,phase+' driver failed'
        old.cleanup(origin,catalog)
        cp=target/phase/'checkpoints'/f'global_step_{end}'
        assert (cp/'actor/model_state_dict/full_weights.pt').is_file(), 'Missing completed policy checkpoint'
        return str(cp)
    error=None
    try:
        # Reuse already captured complete native blocks for the WM update
        # smoke. Initial N64 collection is the formal collection itself.
        n=64
        data=pack('wm_probe_data',[Path(plan['probe_capture'])],True)
        trained=wm('wm_probe4',data,plan['wm_checkpoint'],4,steps=2,optional=True)
        if trained is None:
            trained=wm('wm_probe2',data,plan['wm_checkpoint'],2,steps=2)
        batch=int(trained['microbatch'])
        initial=192;additional=128 if n==64 else 96
        save(out/'selected.json',dict(native_n=n,initial_total=initial,initial_train=160,heldout=32,
            initial_r=initial//n,additional=additional,additional_r=additional//n,wm_microbatch=batch,
            wm_train_gpus=[4,5],rl_n=64,rl_r=8,rl_g=8,wm_inference_batch=16))
        initial_ids=seeds['train_pool'][:160]+seeds['heldout']
        native('initial',n,initial_ids)
        dataset=pack('initial',[out/'initial/capture'])
        policy_cp=None;wm_cp=plan['wm_checkpoint'];rm_cp=plan['rm_checkpoint'];threshold=plan['threshold']
        previous_roots=[out/'initial/capture']
        for start in range(0,200,10):
            if start:
                phase=f'native{start:03d}'
                pool=seeds['train_pool'];offset=(160+(start//10-1)*additional)%len(pool)
                ids=(pool+pool)[offset:offset+additional]
                weights=str(Path(policy_cp)/'actor/model_state_dict/full_weights.pt')
                native(phase,n,ids,weights)
                # Keep the initial real data/holdout plus newest on-policy trajectories.
                dataset=pack(phase,[out/'initial/capture',out/phase/'capture'])
            phase=f'wm{start:03d}'
            trained=wm(phase,dataset,wm_cp,batch);wm_cp=trained['checkpoint']
            rm_phase=f'rm{start:03d}'
            argv=[plan['rl_python'],'-u','-B',str(code/'rm_train.py'),'--dataset-dir',str(dataset/'rm'),
                '--output-dir',str(out/rm_phase/'train'),'--pretrained-path',plan['rm_pretrained'],
                '--resume-checkpoint',rm_cp,'--global-batch','64','--micro-batch','64']
            run(argv,rm_phase,dict(gpu_env,CUDA_VISIBLE_DEVICES='4',PYTHONPATH=str(code)))
            report=read(out/rm_phase/'train/report.json')
            if report['validation']['recall']>0 and 0<report['validation']['threshold']<=1:
                rm_cp=report['checkpoint'];threshold=report['validation']['threshold']
            save(out/f'offline_{start:03d}.json',dict(wm_checkpoint=wm_cp,rm_checkpoint=rm_cp,threshold=threshold))
            policy_cp=policy(start+10,policy_cp,wm_cp,rm_cp,threshold)
        save(out/'complete.json',dict(time=H.now(),checkpoint=policy_cp))
    except BaseException as exc:
        error=dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc());save(out/'error.json',error)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        old.cleanup(origin,catalog)
        save(out/'final.json',dict(error=error,time=H.now()))
    if error:raise SystemExit(1)

if __name__=='__main__':main()
