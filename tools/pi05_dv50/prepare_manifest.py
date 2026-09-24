"""Freeze fifty RoboTwin tasks, source identities and two 16-env batches each."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess


def prepare(base_config, repo, output, environment, gpus=(6, 7)):
    import yaml
    repo, output = Path(repo), Path(output)
    original = json.loads(Path(base_config).read_text())
    asset = Path(original['env']['eval']['assets_path'])
    limits = yaml.safe_load((asset/'task_config/_eval_step_limit.yml').read_text())
    tasks = sorted(p.stem for p in (asset/'envs').glob('*.py')
                   if p.stem not in {'__init__', '_base_task', 'reward', '_GLOBAL_CONFIGS'})
    assert len(tasks) == 50 and set(tasks) <= limits.keys()
    seeds = json.loads((repo/'rlinf/envs/robotwin/seeds/eval_seeds.json').read_text())
    git = lambda path: subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    head = git(repo)
    metadata = {'source_commit': head, 'robotwin_commit': git(asset),
        'model_path': original['rollout']['model']['model_path'], 'tasks': 50,
        'episodes_per_task': 32, 'episodes': 1600, 'gpus': list(gpus),
        'selected_l': 3, 'sampler': 'native OpenPI eval flow ODE; 10 denoising steps',
        'dv_definition': 'sum over 14 action dimensions of population variance over final 3 endpoint predictions',
        'video': '4 fps query-aligned head-camera preview', 'configs': []}
    output.mkdir(parents=True, exist_ok=False)
    (output/'configs').mkdir()
    env = json.loads(Path(environment).read_text())
    old = env['REPO_PATH']
    env = {k: v.replace(old, str(repo)) for k,v in env.items()}
    env.pop('RAY_ADDRESS', None)
    (output/'environment.json').write_text(json.dumps(env,indent=2))
    # First batches are smoke AND reusable formal data. Remaining work is queue-claimed.
    ordered = ['turn_switch','adjust_bottle'] + [t for t in tasks if t not in {'turn_switch','adjust_bottle'}]
    for ti, task in enumerate(ordered):
        values = seeds.get(task,{}).get('success_seeds',[])
        source = 'official eval_seeds.json' if len(values)>=32 else 'fixed generated; not expert-filtered'
        values = values[:32] if len(values)>=32 else [100000+tasks.index(task)*10000+i*100 for i in range(32)]
        assert len(values)==len(set(values))==32
        for batch in range(2):
            name = f'{task}_b{batch}'
            dest = output/'batches'/name
            seed_path = output/'configs'/f'{name}_seeds.json'
            seed_path.write_text(json.dumps({task:{'success_seeds':values[batch*16:(batch+1)*16]}}))
            cfg = {'task':task,'batch':batch,'num_envs':16,'gpu':None,'selected_l':3,
                'step_limit':int(limits[task]),'seed_source':source,'noise_seed':42+tasks.index(task)*2+batch,
                'source_commit':head,'robotwin_commit':metadata['robotwin_commit'],
                'output':str(dest),'model':copy.deepcopy(original['rollout']['model']),
                'env':copy.deepcopy(original['env']['eval'])}
            ec = cfg['env']
            ec.update(total_num_envs=16,auto_reset=False,ignore_terminations=True,
                      max_steps_per_rollout_epoch=cfg['step_limit'],max_episode_steps=cfg['step_limit'],
                      seeds_path=str(seed_path),use_fixed_reset_state_ids=True)
            ec['video_cfg'].update(save_video=False,video_base_dir=str(dest/'unused'))
            ec['task_config'].update(task_name=task,step_lim=cfg['step_limit'],save_path=str(dest/'env'),
                                     collect_data=False,eval_video_log=False)
            path = output/'configs'/f'{name}.json'
            path.write_text(json.dumps(cfg,indent=2))
            metadata['configs'].append(str(path))
    metadata['smoke'] = [str(output/'configs'/f'{t}_b0.json') for t in ordered[:2]]
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('base_config');p.add_argument('repo');p.add_argument('output');p.add_argument('environment')
    a=p.parse_args(); m=prepare(a.base_config,a.repo,a.output,a.environment)
    print(json.dumps({k:v for k,v in m.items() if k!='configs'},indent=2))
