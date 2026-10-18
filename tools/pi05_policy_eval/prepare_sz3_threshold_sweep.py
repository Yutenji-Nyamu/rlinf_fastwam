"""Prepare one authorized SZ3 6/7 sweep without stopping running RLT."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

S = Path('/data/chenyiteng/deployment-20261008/bc-signal-tau-v1')
T = Path('/data/chenyiteng/deployment-20261018/cp180-threshold-sweep-sz3-v1')
O = Path('/data/chenyiteng/results/pi05-policy-eval/pillbottle-cp180-threshold-sweep-20261018-v1')
OLD = O.parent / 'pillbottle-cp180-takeover-20261018-v1'
OLD_T = T.parent / 'cp180-takeover-sz3-v1'
R = Path(__file__).resolve().parents[2]
OLD_R = R.parent / 'pi05-cp180-takeover-20261018'
sys.path.insert(0, str(S / 'ops'))
from common import gpus, read, same, save  # noqa: E402
from threshold_sweep import prepare  # noqa: E402


def main(commit):
    assert os.getuid() == 20001 and not T.exists() and not O.exists()
    T.mkdir(parents=True)
    shutil.copytree(OLD_T / 'bootstrap', T / 'bootstrap')
    manifest = prepare(OLD, O, T, '/data/chenyiteng/sweep-calibration1018.json', commit)
    for gpu in (6, 7):
        env = {key: value.replace(str(OLD_R), str(R)).replace(str(OLD_T), str(T))
               for key, value in read(OLD / f'environment-g{gpu}.json').items()}
        env['TMPDIR'] = str(T / f'tmp-g{gpu}')
        Path(env['TMPDIR']).mkdir()
        assert env['CUDA_VISIBLE_DEVICES'] == env['DOJO_GPU_SCOPE'] == str(gpu)
        assert str(T / 'bootstrap') in env['PYTHONPATH']
        save(O / f'environment-g{gpu}.json', env)
    profile = Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d/00-cp180-20261018.json')
    assert profile.exists()
    assert Path(read(manifest['configs'][0])['strong_weights']).stat().st_size == 8526574644
    # Same native protocol tests; new tests exercise only frozen gate selection.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               ROBOTWIN_PATH='/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support',
               PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(T))
    python = '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python'
    result = subprocess.run([python, '-B', '-m', 'unittest', '-v', 'test_protocol', 'test_threshold_sweep'],
                            cwd=R / 'tools/pi05_policy_eval', env=env, text=True, capture_output=True)
    print(result.stdout + result.stderr, flush=True)
    assert result.returncode == 0
    # Verify paired workload/config deltas once before touching the running jobs.
    permitted = {'condition', 'policy', 'output', 'source_commit', 'threshold_path', 'sweep_target_fraction', 'env'}
    for path in manifest['configs']:
        config = read(path)
        base = read(OLD / 'configs' / f"cp180_high_u_b{config['batch']}.json")
        assert all(value == base[key] for key, value in config.items() if key not in permitted)
        assert read(config['env']['seeds_path']) == read(base['env']['seeds_path'])
        altered_env = dict(config['env'])
        altered_env['seeds_path'] = base['env']['seeds_path']
        altered_env['task_config'] = dict(altered_env['task_config'], save_path=base['env']['task_config']['save_path'])
        altered_env['video_cfg'] = dict(altered_env['video_cfg'], video_base_dir=base['env']['video_cfg']['video_base_dir'])
        assert altered_env == base['env']
    plan, state, owner = read(S / 'plan.json'), read(S / 'status.json'), read(S / 'owner-identity.json')
    assert same(owner)
    rlt = []
    for gpu in ('6', '7'):
        slot = state['slots'][gpu]
        assert slot['phase'] == 'RLT_RUNNING' and same(slot['identity'])
        assert slot['request'] == plan['slots'][gpu]['fallback']
        request = read(slot['request'])
        rlt.append(dict(gpu=gpu, identity=slot['identity'], request_path=slot['request'], request=request))
    save(T / 'prepared.json', dict(time=time.time(), source=str(R), output=str(O),
                                  plan_before=plan, owner_before=owner, rlt_before=rlt, **manifest))
    save(T / 'preflight.json', dict(time=time.time(), passed=True, tests=6,
                                   physical_gpus=[6, 7], graphics_profile=str(profile), configs=len(manifest['configs'])))
    print('PREPARED: 30 conditions, 3840 episodes, B32 per GPU; no jobs stopped', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
