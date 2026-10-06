"""Prepare CPU-only DSRL scopes and RLT fallback; does not launch/stop workloads.

Input JSON: control_dir, stop_attempt, release_receipt, dsrl_repo, dsrl_python.
stop_attempt.capture is the unique checkpoint source; never select a newer CP.
"""
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import shutil
import time
import yaml
from lease_common import read, sha, save, load, MASKS


def replace_outputs(value, old, new):
    if isinstance(value, dict):
        return {k: replace_outputs(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_outputs(v, old, new) for v in value]
    if isinstance(value, str):
        return new.name if value == old.name else value.replace(str(old), str(new))
    return value


def scope(control, gpu, uuid, base_env, repo, old_repo):
    directory = control / ('scope-g' + str(gpu))
    bootstrap = directory / 'bootstrap'
    bootstrap.mkdir(parents=True)
    (directory / 'receipts').mkdir()
    marker = directory / ('libdsrl_u_g' + str(gpu) + '_20261006.so')
    (directory / 'marker.c').write_text('int dsrl_u_scope_marker(void) { return 1; }\n')
    subprocess.run(['cc', '-shared', '-fPIC', '-nostdlib', '-Wl,-soname,' + marker.name,
                    '-o', str(marker), str(directory / 'marker.c')], check=True, capture_output=True)
    marker.chmod(0o500)
    (bootstrap / 'gpu_scope_runtime.py').write_bytes(Path(__file__).with_name('gpu_scope_runtime.py').read_bytes())
    (bootstrap / 'sitecustomize.py').write_text(
        'import os\np=os.environ.get("RLINF_OPENDW_GPU_SCOPE_MANIFEST")\nif p:\n'
        ' try:\n  from gpu_scope_runtime import install\n  install(p)\n'
        ' except BaseException as e:\n  os.write(2,("DSRL GPU SCOPE FAILED: "+str(e)+"\\n").encode());os._exit(78)\n')
    profile = Path('/home/chenyiteng/.nv/nvidia-application-profiles-rc.d') / ('00-dsrl-u-g' + str(gpu) + '-20261006.json')
    assert not profile.exists(), 'Scope profile already exists; inspect, never overwrite'
    save(profile, {'rules': [{'pattern': {'feature': 'dso', 'matches': marker.name},
                              'profile': ['EGLVisibleDGPUDevices', 1 << gpu]}]}, True)
    manifest = {'schema': 1, 'uid': 1003, 'hostname': 'admin', 'token': control.name + '-g' + str(gpu),
                'physical_gpu': gpu, 'gpu_uuid': uuid, 'minor': gpu, 'mask': 1 << gpu,
                'receipts_dir': str(directory / 'receipts')}
    for key, path in {'marker': marker, 'profile': profile, 'bootstrap': bootstrap / 'sitecustomize.py',
                      'runtime': bootstrap / 'gpu_scope_runtime.py'}.items():
        manifest[key + '_path'], manifest[key + '_sha256'] = str(path), sha(path)
    save(directory / 'scope.json', manifest, True)
    env = {k: v for k, v in base_env.items() if not k.startswith('RLT_') and k not in MASKS}
    inherited = [part for part in env.get('PYTHONPATH', '').split(':') if part]
    # Keep assets/runtime dependencies, replace the former source and private RLT injection.
    inherited = [part for part in inherited if not (
        part == old_repo or part.startswith(old_repo + '/') or 'native_overlay' in part
        or ('scope-' in part and part.endswith('/bootstrap')))]
    env.update(HOME='/home/chenyiteng', PYTHONDONTWRITEBYTECODE='1',
               PATH=str(Path(sys.executable).parent) + ':' + os.environ.get('PATH', '/usr/bin:/bin'),
               RLINF_CODE_WORKING_DIR=str(repo), REPO_PATH=str(repo),
               EMBODIED_PATH=str(repo / 'examples/embodiment'),
               ROBOTWIN_PI0_BASE_PATH='/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab',
               ROBOTWIN_PI0_NORM_STATS_PATH='/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab/physical-intelligence/robotwin/norm_stats.json',
               RLINF_OPENDW_GPU_SCOPE_MANIFEST=str(directory / 'scope.json'),
               LD_PRELOAD=str(marker), __GL_APPLICATION_PROFILE='1',
               PYTHONPATH=':'.join([str(bootstrap), str(repo), *inherited]))
    return env, {str(path): sha(path) for path in [directory / 'scope.json', marker, profile,
                                                 bootstrap / 'sitecustomize.py', bootstrap / 'gpu_scope_runtime.py']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    args = parser.parse_args()
    data, code = read(args.inputs), Path(__file__).parent.resolve()
    assert shutil.which('cc') and shutil.which('gcc'), 'Missing proven runtime compiler tools'
    control = Path(data['control_dir'])
    assert control.parent == Path('/data/chenyiteng/deployment-20261006') and control.name.startswith('dsrl-')
    assert not control.exists() and os.getuid() == 1003
    stop_attempt = read(data['stop_attempt'])
    capture = stop_attempt['capture']
    release = read(data['release_receipt'])
    assert release['all_released'] is True and release['owner'] == capture['owner']
    assert Path(data['release_receipt']).parent == Path(data['stop_attempt']).parent
    old_plan = read(Path(capture['stage']) / 'plan.json')
    percard = load(old_plan['owner_script'], 'prepare_existing_owner')
    b = percard.b
    assert capture['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    common_path = str(Path(b.__file__).resolve())
    q = read(capture['targets']['6']['task_plan']['path'])
    for gpu in ('6', '7'):
        record = capture['targets'][gpu]['task_plan']
        assert sha(record['path']) == record['sha256'], 'Original RLT task plan changed'
    assert q['task'] == 'click_bell' and q['ops'] == read(capture['targets']['7']['task_plan']['path'])['ops']
    os.umask(0o077)
    control.mkdir()
    for name in ('requests', 'receipts', 'rlt-fallback'):
        (control / name).mkdir()
    pins = {str(path): sha(path) for path in code.glob('*.py')}
    pins.update({common_path: sha(common_path), q['ops']: sha(q['ops']),
                 data['stop_attempt']: sha(data['stop_attempt']), data['release_receipt']: sha(data['release_receipt'])})
    resumed = copy.deepcopy(q)
    slots = {}
    for gpu in (6, 7):
        old = capture['targets'][str(gpu)]
        run = Path(old['run'])
        runtime = run / 'runtime'
        old_cfg = yaml.safe_load((runtime / 'resolved.yaml').read_text())
        assert old_cfg['runner']['max_steps'] == old_cfg['runner']['max_epochs'] == 3000
        assert old_cfg['env']['train']['total_num_envs'] == 8
        newrun = run.with_name('rlt-after-' + control.name + '-g' + str(gpu))
        newrt = newrun / 'runtime'
        assert not newrun.exists()
        newrt.mkdir(parents=True)
        cfg = replace_outputs(old_cfg, run, newrun)
        cfg['runner']['resume_dir'] = old['checkpoint']['path']
        env = replace_outputs(read(runtime / 'environment.json'), run, newrun)
        original_scope = Path(env['RLINF_OPENDW_GPU_SCOPE_MANIFEST'])
        original_manifest = read(original_scope)
        assert original_manifest['physical_gpu'] == gpu and original_manifest['gpu_uuid'] == old['gpu_uuid']
        pins[str(original_scope)] = sha(original_scope)
        for key in ('marker', 'profile', 'bootstrap', 'runtime'):
            path = original_manifest[key + '_path']
            assert sha(path) == original_manifest[key + '_sha256']
            pins[path] = sha(path)
        (newrt / 'resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        save(newrt / 'environment.json', env, True)
        ns = 'rlt-after-' + control.name + '-g' + str(gpu)
        resumed['runs'][old['role']].update(run=str(newrun), namespace=ns)
        scope_env, scope_pins = scope(control, gpu, old['gpu_uuid'], env,
                                      Path(data['dsrl_repo']), q['repo'])
        # Never carry RLT's renamed log root into DSRL.
        scope_env.pop('RLT_LOG_ROOT', None)
        pins.update(scope_pins)
        for path in (newrt / 'resolved.yaml', newrt / 'environment.json'):
            pins[str(path)] = sha(path)
        slots[str(gpu)] = {'role': 'clean' if gpu == 6 else 'u', 'gpu_uuid': old['gpu_uuid'],
                           'original': old, 'fallback': resumed['runs'][old['role']],
                           'fallback_role': old['role'], 'scope_env': scope_env}
        save(control / ('gpu' + str(gpu) + '-scope-env.json'), scope_env, True)
    fallback_plan = control / 'rlt-fallback/plan.json'
    save(fallback_plan, resumed, True)
    # Reuse the actual driver's pure preflight on the copied plan. Only output
    # paths, namespace and resume_dir changed; all resource/algorithm fields stay.
    fallback_ops = load(q['ops'], 'prepared_fallback_check')
    fallback_ops.ST = fallback_plan.parent
    assert fallback_ops.checked() == resumed
    pins[str(fallback_plan)] = sha(fallback_plan)
    plan = {'schema': 1, 'prepared_at': time.time(), 'uid': 1003, 'hostname': 'admin',
            'boot_id': capture['boot_id'], 'original_stage': capture['stage'],
            'original_owner': capture['owner'], 'protected': capture['protected'],
            'release_receipt': data['release_receipt'], 'stop_attempt': data['stop_attempt'],
            'common_source': common_path, 'rlt_ops': q['ops'], 'fallback_plan': str(fallback_plan),
            'dsrl_repo': data['dsrl_repo'], 'dsrl_python': data['dsrl_python'],
            'ray_dashboard_url': q['ray_dashboard_url'], 'ray_address': q['ray_address'],
            'watch': old_plan['watch'], 'slots': slots, 'pins': pins,
            'code': str(code), 'states': 'DEV_HOLD,SMOKE,FORMAL,RELEASE_CHECK,RLT_STARTING,RLT_RESTORED'}
    save(control / 'plan.json', plan, True)
    save(control / 'prepared.json', {'time': time.time(), 'plan_sha256': sha(control / 'plan.json'),
                                     'inputs_sha256': sha(args.inputs)}, True)
    print(__import__('json').dumps({'prepared': str(control), 'plan_sha256': sha(control / 'plan.json'),
                                  'workloads_started': 0, 'workloads_stopped': 0}))


if __name__ == '__main__':
    main()
