"""Prepare accepted Clean seeds/model/limits with B32 and opt-in observations."""
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,'/data/chenyiteng/deployment-20261008/bc-signal-tau-v1/ops')
from common import read, save


def prepare(D, O, revision):
    old=Path('/data/chenyiteng/results/pi05-signals/20261003-v1')
    envsource=Path('/data/chenyiteng/results/pi05-noise-video/20261019-sz1-v2/environment-g4.json')
    original=read(old/'manifest.json');configs=[read(p)for p in original['configs']]
    groups={}
    for c in configs:groups.setdefault(c['task'],[]).append(c)
    assert len(configs)==100 and len(groups)==50 and not O.exists()
    assert os.getuid()==1003 and D==ROOT.parent
    (O/'configs').mkdir(parents=True);(O/'batches').mkdir()
    tasks=sorted(groups,key=lambda t:(t not in ['adjust_bottle','beat_block_hammer'],t))
    jobs={6:[],7:[]};now=time.time()
    for i,task in enumerate(tasks):
        bases=sorted(groups[task],key=lambda c:c['batch']);assert len(bases)==2
        assert all(c['num_envs']==16 and c['model']==bases[0]['model'] for c in bases)
        seeds=sum([read(c['env']['seeds_path'])[task]['success_seeds']for c in bases],[])
        assert len(seeds)==len(set(seeds))==32
        gpu=6+i%2;c=copy.deepcopy(bases[0]);name=task+'_b32';dest=O/'batches'/name
        sf=O/'configs'/(name+'_seeds.json');save(sf,{task:{'success_seeds':seeds}})
        c.update(gpu=gpu,batch=0,num_envs=32,output=str(dest),source_commit=revision,
                 nstep_signals=True,noise_samples=4,noise_extra_batch=96,
                 benchmark_noise_batches=[96,48],acceptance=False,video_fps=25,
                 environment_step_timeout_seconds=300,
                 config_sources=[str(old/'configs'/f"{task}_b{b['batch']}.json")for b in bases],
                 signal_schema='nstep-v1',executed_steps=10,solver_budgets=[1,5,10],
                 batch_protocol='merge two accepted B16 seed lists into B32; same seeds, changed batch RNG allocation')
        c['env'].update(total_num_envs=32,seeds_path=str(sf))
        c['env']['task_config']['save_path']=str(dest/'env')
        c['env']['video_cfg'].update(save_video=False,video_base_dir=str(dest/'unused'))
        assert c['model']==bases[0]['model'] and c['step_limit']==bases[0]['step_limit']
        cp=O/'configs'/(name+'.json');save(cp,c);jobs[gpu].append(dict(task=task,config=str(cp)))
    oldenv=read(envsource)
    for gpu in [6,7]:
        env={k:v.replace(oldenv['REPO_PATH'],str(ROOT)) for k,v in oldenv.items()}
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),DOJO_GPU_SCOPE=str(gpu),TMPDIR=str(D/f'tmp-g{gpu}'))
        Path(env['TMPDIR']).mkdir();env.pop('RAY_ADDRESS',None)
        save(O/f'environment-g{gpu}.json',env)
        save(D/f'lane-g{gpu}-plan.json',dict(gpu=gpu,output=str(O),jobs=jobs[gpu],prepared_time=now))
    manifest=dict(time=now,source=str(ROOT),source_commit=revision,output=str(O),tasks=50,
                  episodes=1600,episodes_per_task=32,per_card_parallelism=32,
                  physical_gpus=[6,7],solver_budgets=[1,5,10],executed_steps=10,
                  noise_samples=4,original=str(old),environment_source=str(envsource),
                  horizon=50,execute_chunk=50,recording='accepted head video plus 3-camera query observations and physical mapping',
                  stop_conditions='task 3600s/stale900s -> skip; B32 OOM -> two B16 once; invalid signal -> release RLT',
                  priority='Nstep signal -> independent per-card RLT -> existing no-save RLT fallback',
                  configs=[j['config'] for g in jobs.values() for j in g])
    save(O/'manifest.json',manifest);save(D/'prepared.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items()if k!='configs'}))


if __name__=='__main__':
    prepare(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
