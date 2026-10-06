"""One policy GPU4 + one B32 WM GPU5, inheriting exact cleanup/RLT return."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

M = None
THIS = Path(__file__).absolute()
OWNER_ENV = None
SCOPE_KEYS = {'RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST', 'PYTHONPATH', 'LD_PRELOAD',
              '__GL_APPLICATION_PROFILE', '__GL_APPLICATION_PROFILE_LOG', 'HOME', 'USER', 'LOGNAME'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(cfg):
    value = copy.deepcopy(cfg)
    for path in [('env','group_name'),('actor','group_name'),('rollout','group_name'),
                 ('runner','logger','log_path'),('runner','logger','experiment_name'),
                 ('runner','per_worker_log_path'),('env','train','video_cfg','video_base_dir'),
                 ('env','eval','video_cfg','video_base_dir'),('env','eval','task_config','save_path'),
                 ('algorithm','dvac_gradient_weighting','output_dir')]:
        parent = value
        for key in path[:-1]:
            parent = parent[key]
        parent.pop(path[-1], None)
    return value


def validate(plan, frozen=False):
    global OWNER_ENV
    assert os.getuid() == 20001 and M.socket.gethostname() == 'h100-gpu01'
    M.pidfd_probe()
    owner = M.owned_path(plan['owner_dir'], exists=frozen)
    cycle = M.owned_path(plan['lifecycle_path'])
    cp = M.C.load_plan(cycle)
    assert cp['physical_gpus'] == plan['physical_gpus'] == [4,5]
    assert plan['python'] == cp['python']
    repo = M.owned_path(plan['repo'])
    assert M.subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip() == plan['repo_head']
    assert plan['mode'] == 'two_gpu_b32_formal'
    for path, digest in plan['source_sha256'].items():
        assert sha(M.owned_path(path)) == digest, 'Frozen source changed: '+path
    assert [r['key'] for r in plan['trials']] == ['startup_smoke','formal']
    donor = read(plan['reference_config'])
    formal = read(plan['trials'][1]['config'])
    expected = copy.deepcopy(donor)
    expected['cluster']['component_placement'] = {'actor':'4','env':'5','rollout':'4'}
    expected['env']['train']['service_urls'] = [plan['services'][0]['url']]
    expected['actor']['micro_batch_size'] = 16
    expected['actor']['fsdp_config']['resume_source_world_size'] = 2
    expected['runner']['resume_dir'] = plan['resume_dir']
    assert normalized(formal) == normalized(expected), 'Unexpected formal protocol change'
    assert formal['env']['train']['total_num_envs'] == 64 and formal['env']['train']['rollout_epoch'] == 8
    assert formal['algorithm']['group_size'] == 8 and formal['actor']['global_batch_size'] == 2048
    assert formal['runner']['max_steps'] == 200
    assert formal['runner']['save_interval'] == formal['runner']['val_check_interval'] == 10
    smoke = read(plan['trials'][0]['config'])
    expected = copy.deepcopy(formal)
    expected['runner'].update(max_steps=plan['resume_step']+1, save_interval=1, val_check_interval=1)
    expected['env']['train'].update(rollout_epoch=1, max_episode_steps=32, max_steps_per_rollout_epoch=32)
    expected['env']['eval'].update(max_episode_steps=32, max_steps_per_rollout_epoch=32)
    expected['env']['eval']['task_config']['step_lim'] = 32
    expected['actor']['global_batch_size'] = 64
    expected['rollout']['batch_probe_sizes'] = [32,64,128]
    assert normalized(smoke) == normalized(expected), 'Unexpected smoke change'
    for row in plan['trials']:
        assert row['namespace'].startswith('opendw_') and re.fullmatch('[A-Za-z0-9_-]+',row['namespace'])
        assert row['num_envs'] == 64 and 0 < row['timeout_seconds'] <= 60*86400
        assert Path(read(row['config'])['runner']['logger']['log_path']).is_relative_to(owner/row['key'])
        if frozen:
            assert row['config_sha256'] == sha(row['config'])
    M.validate_services(plan['services'], owner)
    environment = read(plan['environment_file'])
    assert not any(key in environment for key in M.MASKS)
    fragment = read(plan['graphics_fragment'])
    assert set(fragment) == SCOPE_KEYS
    if frozen:
        assert read(owner/'scope-activated.json')['physical_gpus'] == [4,5]
        assert plan['owner_script_sha256'] == sha(THIS)
        assert plan['environment_sha256'] == sha(plan['environment_file'])
        assert plan['lifecycle_plan_sha256'] == sha(cycle/'plan.json')
        environment.update(fragment)
    else:
        OWNER_ENV = environment
    return owner, cycle, repo, environment


def smoke_gate(plan):
    root = Path(plan['owner_dir'])
    logs = list((root/'startup_smoke').rglob('*.log')) + list((root/'startup_smoke').rglob('*.out'))
    joined = '\n'.join(p.read_text(errors='replace') for p in logs if p.stat().st_size < 8000000)
    assert 'WMRL_TWO_TO_ONE_RESTORE ' in joined, 'No optimizer-preserving restore evidence'
    assert 'WMRL_POLICY_BATCH_PROBE ' in joined, 'No actual policy batch evidence'
    assert 'WMRL_ACTOR_MEMORY ' in joined, 'No actor update memory evidence'
    # Successful smoke includes finite updates, actual B64 prediction and native
    # evaluation/save. It is a resource smoke; zero rewards in one C32 are allowed.
    events = root/'services/wm5/records/service-events.jsonl'
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    completed = [r for r in rows if r.get('event')=='batch_completed' and r.get('actual_wm_batch')==32]
    assert len(completed) >= 2, 'Two real B32 forwards must process N64'
    result = {'passed':True, 'actual_wm_batch':32, 'actual_policy_batch':64,
              'wm_full_batches':len(completed), 'wm_seconds':[r['seconds'] for r in completed],
              'optimizer_preserved':True, 'resource_smoke_only':True}
    M.record(root/'startup-smoke-accepted.json',result)


def install(plan):
    global M, THIS
    THIS = next(Path(p) for p,d in plan['source_sha256'].items() if Path(p).resolve()==THIS.resolve() and d==sha(THIS))
    source = Path(plan['base_owner_module'])
    assert sha(source) == plan['source_sha256'][str(source)]
    spec = importlib.util.spec_from_file_location('two_gpu_reused_owner', source)
    M = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = M
    spec.loader.exec_module(M)
    M.__file__ = str(THIS)
    M.validate = validate
    M.load_lifecycle(plan)
    original_stop = M.C.stop
    def stop_and_scope(stage):
        result = original_stop(stage)
        assert not M.H.gpu_processes([4,5])
        fragment = read(plan['graphics_fragment'])
        manifest = M.owned_path(fragment['RLINF_OPENDW_FORMAL_GRAPHICS_MANIFEST'])
        assert sha(manifest) == plan['scope_manifest_sha256']
        assert OWNER_ENV is not None
        OWNER_ENV.update(fragment)
        M.record(Path(plan['owner_dir'])/'scope-activated.json',
                 {'physical_gpus':[4,5], 'reused_existing_profile':True, 'manifest_sha256':sha(manifest)})
        return result
    M.C.stop = stop_and_scope
    original_record = M.record
    def recorded(path, value):
        original_record(path,value)
        if Path(path)==Path(plan['owner_dir'])/'startup_smoke/result.json' and value['exit_code']==0:
            smoke_gate(plan)
    M.record = recorded
    original_driver = M.run_driver
    def scoped_driver(frozen, key):
        import ray
        original = ray.init
        def inject(*args, **kwargs):
            env = kwargs.setdefault('runtime_env',{}).setdefault('env_vars',{})
            for name in SCOPE_KEYS:
                assert os.environ.get(name)
                env[name] = os.environ[name]
            return original(*args,**kwargs)
        ray.init = inject
        try:
            return original_driver(frozen,key)
        finally:
            ray.init = original
    M.run_driver = scoped_driver
    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('action',choices=('owner','driver'))
    parser.add_argument('--key')
    args = parser.parse_args()
    plan = read(args.plan)
    module = install(plan)
    if args.action == 'owner':
        module.owner_main(plan)
    else:
        module.run_driver(plan,args.key)
