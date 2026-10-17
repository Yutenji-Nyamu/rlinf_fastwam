"""Reuse all 100 accepted DV50/U configs and exact seed files; no reselection."""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
OLD = Path('/data/chenyiteng/results/pi05-signals/20261003-v1')
O = Path('/data/chenyiteng/results/pi05-u-video/20261017-v1')
D = Path('/data/chenyiteng/deployment-20261017/pi05-u-video-v1')
S = Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
sys.path.insert(0, str(S / 'ops'))
from common import read, save, same


def prepare():
    assert os.getuid() == 1003 and not O.exists() and not D.exists()
    original = read(OLD / 'manifest.json')
    paths = [Path(p) for p in original['configs']]
    assert len(paths) == 100
    configs = [read(p) for p in paths]
    assert len({c['task'] for c in configs}) == 50 and all(c['num_envs'] == 16 for c in configs)
    plan = read(S / 'plan.json')
    assert all(same(plan['slots'][g]['external_identity']) for g in ['6', '7'])
    assert all('pi05-policy-budget-eval-v1' in plan['slots'][g]['external_release'] for g in ['6', '7'])
    (O / 'configs').mkdir(parents=True)
    (O / 'batches').mkdir()
    D.mkdir(parents=True)
    oldenv = read(OLD / 'environment.json')
    jobs = {6: [], 7: []}
    revision = subprocess.check_output(['git', '-C', str(R), 'rev-parse', 'HEAD'], text=True).strip()
    for c in configs:
        g = 6 + c['batch']
        assert g in jobs
        name = f"{c['task']}_b{c['batch']}"
        dest = O / 'batches' / name
        seedfile = O / 'configs' / (name + '_seeds.json')
        shutil.copyfile(c['env']['seeds_path'], seedfile)
        old = json.loads(json.dumps(c))
        c.update(output=str(dest), gpu=g, source_commit=revision, acceptance=False,
                 video_fps=25, signal='U10/5 only; accepted per-action formula',
                 config_source=str(OLD / 'configs' / (name + '.json')))
        ec = c['env']
        ec['seeds_path'] = str(seedfile)
        ec['task_config']['save_path'] = str(dest / 'env')
        ec['video_cfg'].update(save_video=False, video_base_dir=str(dest / 'unused'))
        ec['task_config']['eval_video_log'] = False
        assert c['step_limit'] == old['step_limit'] and c['model'] == old['model']
        path = O / 'configs' / (name + '.json')
        save(path, c)
        jobs[g].append(dict(config=str(path), task=c['task'], batch=c['batch']))
    for g in [6, 7]:
        first = 'turn_switch' if g == 6 else 'adjust_bottle'
        jobs[g].sort(key=lambda r: (r['task'] != first, r['task']))
        firstcfg = read(jobs[g][0]['config'])
        firstcfg['acceptance'] = True
        save(jobs[g][0]['config'], firstcfg)
        env = {k: v.replace(oldenv['REPO_PATH'], str(R)) if isinstance(v, str) else v for k, v in oldenv.items()}
        env.update(CUDA_VISIBLE_DEVICES=str(g), DOJO_GPU_SCOPE=str(g), TMPDIR=str(D / f'tmp-g{g}'))
        Path(env['TMPDIR']).mkdir()
        env.pop('RAY_ADDRESS', None)
        save(O / f'environment-g{g}.json', env)
        save(D / f'lane-g{g}-plan.json', dict(gpu=g, jobs=jobs[g], predecessor=plan['slots'][str(g)]))
    manifest = dict(time=time.time(), source=str(R), output=str(O), source_commit=revision,
                    tasks=50, episodes_per_task=32, episodes=1600, batches=100,
                    per_card_parallelism=16, physical_gpus=[6, 7], original=str(OLD),
                    plan_before=plan, policy='original pi05 M10 H50; side M5 same full noise',
                    recording='25Hz simulated timeline, native head only; every physics step mapped',
                    acceptance='First two real B16 batches must fully decode and validate before remaining jobs',
                    order='current policy-budget evaluation -> U/video -> original per-card RLT')
    save(O / 'manifest.json', manifest)
    save(D / 'prepared.json', manifest)
    print(json.dumps(manifest))


if __name__ == '__main__':
    prepare()
