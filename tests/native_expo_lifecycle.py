"""Native N4 evaluation teardown -> N1 rollout lifecycle; no learning/budget use."""
import argparse,copy,faulthandler,gc,hashlib,json,os,resource,sys,time
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]
ROOT=SOURCE.parent
sys.path.insert(0,str(SOURCE))
faulthandler.enable(all_threads=True)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['baseline','fixed'],required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    assert 1<=args.repeats<=3
    logpath=ROOT/('native-'+args.mode+'-events.jsonl')
    assert not logpath.exists(),'Use a new diagnostic attempt; never overwrite evidence'
    def event(name,**kw):
        row={'time':time.time(),'event':name,**kw}
        with logpath.open('a') as f:
            f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
        (ROOT/'driver-heartbeat').touch()
        print(json.dumps(row),flush=True)
    event('imports_started')
    import torch
    from omegaconf import OmegaConf
    from rlinf.algorithms.expo_ft.backend import create_robotwin_env
    from rlinf.algorithms.expo_ft.lifecycle import close_robotwin_env
    inputs=json.loads((ROOT/'inputs.json').read_text())
    eval_cfg=copy.deepcopy(inputs['evaluation']['config'])
    eval_cfg.pop('_expo_seed_sha256')
    seeds=json.loads(Path(inputs['evaluation']['seed_path']).read_text())['turn_switch']['success_seeds']
    assert len(seeds)==20 and torch.cuda.device_count()==4
    for i in range(4):
        with torch.cuda.device(i):torch.cuda.manual_seed(42+1009*i)
    def close(env):
        if args.mode=='fixed':close_robotwin_env(env)
        else:env.offload()
    def cycle(number,count):
        cfg=copy.deepcopy(eval_cfg if count==4 else inputs['env'])
        cfg['task_config']['save_path']=str(ROOT/'native-env-data'/args.mode/str(number))
        cfg['video_cfg']['video_base_dir']=str(ROOT/'native-video'/args.mode/str(number))
        event('create_before',cycle=number,count=count)
        env=create_robotwin_env(OmegaConf.create(cfg),num_envs=count,seed_offset=0)
        event('create_after',cycle=number,count=count)
        for group in range(2):
            event('reset_before',cycle=number,group=group)
            obs,_=env.reset(env_seeds=seeds[group*count:(group+1)*count])
            event('reset_after',cycle=number,group=group)
            action=obs['states'].to(torch.float32)[:,None,:].clone()
            for step in range(2):obs,*_=env.step(action,auto_reset=False)
            event('step_after',cycle=number,group=group)
        saved=torch.cuda.get_rng_state_all()
        event('close_before',cycle=number)
        close(env)
        event('close_after',cycle=number)
        for i in range(4):torch.cuda.synchronize(i)
        event('sync_after_close',cycle=number)
        # Return releases the function's final vector reference, as evaluate does.
        event('function_return_before',cycle=number)
        return saved
    for i in range(args.repeats):
        for count in [4,1]:
            number=i*2+(count==1)
            saved=cycle(number,count)
            event('function_return_after',cycle=number)
            gc.collect();event('gc_after',cycle=number)
            torch.cuda.set_rng_state_all(saved)
            for device in range(4):torch.cuda.synchronize(device)
            event('rng_restore_after',cycle=number)
    event('pass',repeats=args.repeats)
    if args.mode=='fixed':
        receipt={'ok':True,'mode':'fixed','repeats':args.repeats,
                 'inputs_sha256':hashlib.sha256((ROOT/'inputs.json').read_bytes()).hexdigest(),
                 'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'lifecycle_sha256':hashlib.sha256((SOURCE/'rlinf/algorithms/expo_ft/lifecycle.py').read_bytes()).hexdigest(),
                 'real_evaluation_score':False,'training_actions':0,'time':time.time()}
        (ROOT/'native-check.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
