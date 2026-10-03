"""One fixed 500-episode LIBERO evaluation on a new private Ray instance, GPUs 2-3.

The outer evaluation owner must catalog and precisely clean this token's process
tree on success/failure. This driver never discovers or stops any Ray instance.
--preflight composes/verifies the exact inference configuration without Ray/CUDA.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import runpy
import socket
import subprocess
import sys
import time


DECLARED_ROOT = Path('/data/chenyiteng/projects/wan-goal-sz3')
ROOT = DECLARED_ROOT.resolve(strict=True)
FORMAL = (ROOT / 'runs/wan-goal-sz3-20261001-r6/pi05-formal').resolve(strict=True)
EXPERIMENT = 'wan_goal_pi05_headonly_formal_sz3'


def emit_file(path, data):
    payload = json.dumps(data, indent=2, allow_nan=False) + '\n'
    with path.open('x') as stream:
        stream.write(payload)


def file_signature(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--formal-config', required=True)
    parser.add_argument('--config-dir', required=True)
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--kind', choices=['original', '40', '80'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    assert os.getuid() == 20001 and pwd.getpwuid(os.getuid()).pw_name == 'chenyiteng'
    repo = Path(args.repo).resolve(strict=True)
    assert repo == (ROOT / 'RLinf-pi05').resolve(strict=True)
    formal_config = Path(args.formal_config).resolve(strict=True)
    assert formal_config.is_relative_to(FORMAL)
    source_dir = Path(args.config_dir).resolve(strict=True)
    log_dir = Path(args.log_dir).resolve()
    assert log_dir.is_relative_to(ROOT / 'evaluations') and log_dir != FORMAL
    assert 'wm-official-20261002-' in str(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ.update(EMBODIED_PATH=str(repo / 'examples/embodiment'),
                      WM_EVAL_RUN_DIR=str(log_dir),
                      WAN_GOAL_PI05_PATH=str(DECLARED_ROOT / 'models/pi05-libero'))
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict
    with initialize_config_dir(version_base='1.1', config_dir=str(source_dir)):
        cfg = compose(config_name='libero_goal_pi05_official_reference_23')
    model = OmegaConf.to_container(cfg.rollout.model, resolve=True)
    assert model['num_action_chunks'] == 5 and model['num_steps'] == 5
    assert model['openpi']['task'] == 'eval'
    assert model['openpi']['action_horizon'] == 10
    assert model['openpi_data']['wrist_mode'] == 'required'
    assert model['openpi_data']['extra_delta_transform'] is False
    assert model['openpi']['discrete_state_input'] is False
    assert cfg.env.eval.task_suite_name == 'libero_goal'
    assert cfg.env.eval.total_num_envs == 20 and cfg.env.eval.rollout_epoch == 25
    assert cfg.env.eval.max_episode_steps == cfg.env.eval.max_steps_per_rollout_epoch == 320
    assert cfg.env.eval.group_size == 1 and cfg.env.eval.seed == 0
    assert cfg.env.eval.use_fixed_reset_state_ids and cfg.env.eval.is_eval
    assert cfg.runner.only_eval and cfg.runner.resume_dir is None
    assert OmegaConf.to_container(cfg.cluster.component_placement) == {'env,rollout': '2-3'}
    ckpt = None
    before = None
    if args.kind != 'original':
        ckpt = (FORMAL / EXPERIMENT / 'checkpoints' / f'global_step_{args.kind}' /
                'actor/model_state_dict/full_weights.pt').resolve(strict=True)
        assert ckpt.is_relative_to(FORMAL)
        before = file_signature(ckpt)
        cfg.runner.ckpt_path = str(ckpt)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(resolved)
    launch_cfg = log_dir / 'launch-config.yaml'
    cfg_text = OmegaConf.to_yaml(cfg, resolve=True)
    if launch_cfg.exists():
        assert launch_cfg.read_text() == cfg_text, 'Previously prepared config changed'
    else:
        with launch_cfg.open('x') as stream:
            stream.write(cfg_text)
    vendor = Path(os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']).resolve(strict=True)
    assert vendor.is_relative_to(source_dir) and json.loads(vendor.read_text()) == {
        'file_format_version': '1.0.0', 'ICD': {'library_path': 'libEGL_mesa.so.0'}}
    assert os.environ.get('WM_EVAL_RENDER_BACKEND') == 'mesa_llvmpipe'
    assert os.environ.get('WM_EVAL_MESA_DEVICE_ID') == '8'
    assert os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1'
    assert os.environ.get('GALLIUM_DRIVER') == 'llvmpipe' and os.environ.get('LP_NUM_THREADS') == '2'
    contract = {'time': time.time(), 'kind': args.kind, 'episodes': 500,
                'tasks': 10, 'actions_per_episode': 320, 'physical_gpus': [2, 3],
                'concurrent_environments': 20, 'evaluation_epochs': 25,
                'render_backend': 'mesa_llvmpipe', 'software_device': 8, 'rendering_uses_cuda': False,
                'egl_vendor_file': str(vendor), 'egl_vendor_sha256': hashlib.sha256(vendor.read_bytes()).hexdigest(),
                'software_render_threads_per_environment': 2,
                'policy_seeds_by_rank': [42, 43], 'environment_seed_base': 0,
                'model_contract': model, 'checkpoint': str(ckpt) if ckpt else None,
                'formal_config': str(formal_config),
                'formal_config_sha256': hashlib.sha256(formal_config.read_bytes()).hexdigest(),
                'launch_config_sha256': hashlib.sha256(launch_cfg.read_bytes()).hexdigest(),
                'checkpoint_signature': before, 'real_environment': True,
                'training_actions': 0, 'world_model_used': False, 'protocol': 'native-Pi0Eval-dual-C5-M5-Goal320', 'source_reference': 'd34d4c320d08cb982de034aa9a011f08dc0fa217', 'infrastructure_adapters': ['private Ray isolation', 'Mesa simulator child renderer', 'fixed policy RNG after loading', 'trial receipts'], 'resource_schedule': '20 environments x25 epochs instead of500x1'}
    print(json.dumps(contract, allow_nan=False), flush=True)
    if args.preflight:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        return
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2,3'
    token = os.environ['WM_OWNER_TOKEN']
    assert len(token) >= 16
    namespace = os.environ['CLUSTER_NAMESPACE']
    assert namespace.startswith('wan_goal_eval_')
    ray_tmp = Path(os.environ['WAN_GOAL_RAY_TMPDIR'])
    assert str(ray_tmp).startswith('/data/chenyiteng/we/') and len(str(ray_tmp)) < 45
    assert not ray_tmp.exists()
    ray_tmp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert 1024 < args.port < 65535
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    address = f'127.0.0.1:{args.port}'
    assert os.environ['RAY_ADDRESS'] == address
    os.environ.update(RAY_USAGE_STATS_ENABLED='0', RAY_DEDUP_LOGS='0',
                      MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', ROBOT_PLATFORM='LIBERO',
                      LIBERO_TYPE='standard', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    os.environ['PYTHONPATH'] = str(source_dir) + os.pathsep + str(repo) + os.pathsep + os.environ.get('PYTHONPATH', '')
    sys.path[:0] = [str(source_dir), str(repo)]
    ray_cli = Path(sys.executable).parent / 'ray'
    cmd = [str(ray_cli), 'start', '--head', '--node-ip-address=127.0.0.1',
           f'--port={args.port}', '--num-gpus=2', '--num-cpus=64',
           '--include-dashboard=false', '--disable-usage-stats',
           '--object-store-memory=8589934592', f'--temp-dir={ray_tmp}']
    emit_file(log_dir / 'launch.json', dict(contract, owner_token=token,
              driver_pid=os.getpid(), ray_address=address, namespace=namespace,
              ray_temp_dir=str(ray_tmp), ray_command=cmd, egl_vendor_file=str(vendor),
              egl_vendor_sha256=hashlib.sha256(vendor.read_bytes()).hexdigest()))
    subprocess.run(cmd, check=True, cwd=repo)
    import ray
    from rlinf.scheduler import Cluster
    Cluster.NAMESPACE = namespace
    original_init = ray.init

    def scoped_init(*init_args, **kwargs):
        requested = kwargs.get('address', init_args[0] if init_args else None)
        if requested in (None, 'auto'):
            kwargs['address'] = address
        elif requested != address:
            raise RuntimeError(f'Unexpected Ray destination: {requested}')
        kwargs['namespace'] = namespace
        runtime = kwargs.setdefault('runtime_env', {})
        runtime.setdefault('env_vars', {}).update(WM_OWNER_TOKEN=token,
            CLUSTER_NAMESPACE=namespace, RAY_ADDRESS=address, PYTHONPATH=os.environ['PYTHONPATH'],
            __EGL_VENDOR_LIBRARY_FILENAMES=str(vendor), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
            WM_EVAL_RENDER_BACKEND='mesa_llvmpipe', WM_EVAL_MESA_DEVICE_ID='8',
            LIBGL_ALWAYS_SOFTWARE='1', GALLIUM_DRIVER='llvmpipe', LP_NUM_THREADS='2')
        return original_init(*init_args, **kwargs)
    ray.init = scoped_init
    from rlinf.utils.placement import HybridComponentPlacement
    original_placement = HybridComponentPlacement.__init__

    def checked_placement(self, config, cluster):
        original_placement(self, config, cluster)
        records = {}
        for name in ('env', 'rollout'):
            rows = self.get_strategy(name).get_placement(cluster, True)
            assert self.get_world_size(name) == 2
            assert [row.visible_accelerators for row in rows] == [['2'], ['3']]
            assert [row.cluster_node_rank for row in rows] == [0, 0]
            records[name] = [asdict(row) for row in rows]
        path = log_dir / 'verified-placement.json'
        if not path.exists():
            emit_file(path, records)
        print('VERIFIED_PHYSICAL_PLACEMENT_23', flush=True)
    HybridComponentPlacement.__init__ = checked_placement
    from wm_eval_workers import FixedSeedEvalRolloutWorker, RecordedEvalEnvWorker
    import rlinf.workers.rollout.hf.huggingface_worker as rollout_module
    import rlinf.workers.env.env_worker as env_module
    rollout_module.MultiStepRolloutWorker = FixedSeedEvalRolloutWorker
    env_module.EnvWorker = RecordedEvalEnvWorker
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    original_evaluate = EmbodiedEvalRunner.evaluate

    def sequential_component_init(self):
        # Initialize each entire group collectively: EnvWorker's original
        # init_worker contains an all-rank barrier and cannot run rank by rank.
        self.rollout.init_worker().wait()
        print('PRIVATE_EVAL_ALL_POLICY_MODELS_READY', flush=True)
        self.env.init_worker().wait()
        print('PRIVATE_EVAL_ALL_GRAPHICS_CONSTRUCTORS_READY', flush=True)

    EmbodiedEvalRunner.init_workers = sequential_component_init

    def recorded_evaluate(self):
        metrics = original_evaluate(self)
        values = {key: value.item() if hasattr(value, 'item') else value
                  for key, value in metrics.items()}
        assert values['num_trajectories'] == 500, values
        score = float(values['success_once'])
        assert math.isfinite(score) and 0 <= score <= 1
        successes = int(round(score * 500))
        assert abs(score - successes / 500) < 1e-6
        tasks = {}
        trials = []
        for rank in range(2):
            receipt = json.loads((log_dir / f'env-rank-{rank}.json').read_text())
            for stage in receipt['stages']:
                trials.extend(map(tuple, stage['seen_task_trial_ids']))
                for key, row in stage['task_stats'].items():
                    total = tasks.setdefault(key, {'success': 0, 'total': 0})
                    total['success'] += row['success']
                    total['total'] += row['total']
        assert len(trials) == len(set(trials)) == 500
        assert len(tasks) == 10 and all(row['total'] == 50 for row in tasks.values())
        assert sum(row['success'] for row in tasks.values()) == successes
        if ckpt:
            assert file_signature(ckpt) == before, 'Checkpoint changed during evaluation'
        emit_file(log_dir / 'evaluation-result.json', dict(contract, finished=time.time(),
            ok=True, metrics=values, successes=successes, task_stats=tasks,
            unique_task_trials=500))
        return metrics
    EmbodiedEvalRunner.evaluate = recorded_evaluate
    sys.argv = [str(repo / 'evaluations/eval_embodied_agent.py'), '--config-path',
                str(log_dir), '--config-name', 'launch-config']
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
