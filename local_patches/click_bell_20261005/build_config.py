"""Derive a fresh click_bell run from the resolved working B16 formal config."""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlparse


def flat(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix: value}
    return {key: child for name, item in value.items()
            for key, child in flat(item, prefix + '.' + name if prefix else name).items()}


def remote(value):
    path = PurePosixPath(value)
    if not path.is_relative_to('/data/chenyiteng') or '..' in path.parts:
        raise ValueError('Require explicit absolute /data/chenyiteng path')
    return str(path)


def build(base, *, name, owner_dir, initial_state, native_seeds, sft_path, service_urls):
    if not re.fullmatch('[A-Za-z0-9_-]+', name):
        raise ValueError('Require unique simple run name')
    if len(service_urls) != 2 or len(set(service_urls)) != 2:
        raise ValueError('Require two independent loopback services')
    for endpoint in service_urls:
        parsed = urlparse(endpoint)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError('Services must be bare loopback HTTP host:port')
    cfg = copy.deepcopy(base)
    train, runner, actor, native = cfg['env']['train'], cfg['runner'], cfg['actor'], cfg['env']['eval']
    assert cfg['cluster']['component_placement'] == {'actor': '4,5', 'env': '6,7', 'rollout': '4,5'}
    assert (train['total_num_envs'], train['rollout_epoch'], train['group_size'], train['chunk']) == (64, 8, 8, 32)
    assert train['max_episode_steps'] == train['max_steps_per_rollout_epoch'] == 384
    assert train['task_name'] == native['task_config']['task_name'] == 'adjust_bottle'
    assert train.get('reward_source', 'worldarena') == 'worldarena', 'Use the working pre-Rynn formal config'
    assert (actor['global_batch_size'], actor['micro_batch_size'], cfg['algorithm']['update_epoch']) == (2048, 8, 2)
    assert (runner['max_epochs'], runner['max_steps'], runner['save_interval'], runner['val_check_interval']) == (1000, 200, 10, 10)
    assert actor['model'] == cfg['rollout']['model']
    assert actor['model']['model_path'] == sft_path, 'SFT must be the original base model, not a task RL checkpoint'
    assert not re.search(r'global_step_|/checkpoint[s]?/', sft_path)
    assert native['total_num_envs'] == 32 and native['rollout_epoch'] == native['group_size'] == 1
    assert train['auto_reset'] is False and train['ignore_terminations'] is False
    owner = PurePosixPath(remote(owner_dir))
    run = owner / 'formal'
    train.update(task_name='click_bell', initial_state_path=remote(initial_state),
                 reward_source='worldarena_t5_classifier', reward_mode='first_success_binary',
                 use_rel_reward=False, reward_coef=1.0, success_reward_threshold=0.9,
                 service_urls=service_urls)
    train['video_cfg']['video_base_dir'] = str(run / 'video/train')
    native['seeds_path'] = remote(native_seeds)
    native['task_config'].update(task_name='click_bell', save_path=str(run / 'robotwin_data/eval'))
    native['video_cfg']['video_base_dir'] = str(run / 'video/eval')
    runner.update(resume_dir=None, ckpt_path=None, per_worker_log_path=str(run / 'worker-metrics'))
    runner['logger'].update(log_path=str(run), experiment_name=name + '-formal')
    cfg['env']['group_name'] = 'OpenDWEnv_' + name
    cfg['actor']['group_name'] = 'OpenDWActor_' + name
    cfg['rollout']['group_name'] = 'OpenDWRollout_' + name
    cfg['algorithm']['dvac_gradient_weighting']['output_dir'] = str(run / 'unused-dv')
    smoke = copy.deepcopy(cfg)
    sr = smoke['runner']
    sr.update(max_epochs=1, max_steps=1, save_interval=1, val_check_interval=1)
    smoke['env']['train'].update(rollout_epoch=1, max_episode_steps=32, max_steps_per_rollout_epoch=32)
    smoke['env']['eval'].update(max_episode_steps=32, max_steps_per_rollout_epoch=32)
    smoke['env']['eval']['task_config']['step_lim'] = 32
    smoke['actor']['global_batch_size'] = 64
    smoke = json.loads(json.dumps(smoke).replace(str(run), str(owner / 'startup_smoke')).replace(name + '-formal', name + '-startup'))
    before, after = flat(base), flat(cfg)
    manifest = {'task': 'click_bell', 'initialization': 'original_SFT; smoke_checkpoint_not_used_for_formal',
                'reward': 'max of 8 predicted main-frame classifier scores >= 0.9: give 1 once at C32 boundary and terminate; otherwise 0',
                'parallel': {'N': 64, 'G': 8, 'wm_batch': 16, 'actor_ranks': 2, 'env_ranks': 2, 'actor_microbatch': 8},
                'formal': {'R': 8, 'C': 32, 'max_actions': 384, 'trajectories_per_round': 512, 'global_batch': 2048, 'update_epochs': 2, 'rounds': 200, 'save_eval_every': 10},
                'smoke': {'R': 1, 'C': 32, 'max_actions': 32, 'rounds': 1, 'global_batch': 64, 'native_eval_N': 32, 'tests_learning_quality': False},
                'diff': [{'key': key, 'before': before.get(key), 'after': after.get(key)} for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)],
                'launched': False}
    return cfg, smoke, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config', type=Path, required=True)
    p.add_argument('--base-config-sha256', required=True)
    for arg in ('name', 'owner-dir', 'initial-state', 'native-seeds', 'sft-path'):
        p.add_argument('--' + arg, required=True)
    p.add_argument('--service-urls', nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    digest = hashlib.sha256(a.base_config.read_bytes()).hexdigest()
    if digest != a.base_config_sha256:
        raise ValueError('Base config differs from the verified running config')
    reset_path = Path(a.initial_state)
    reset_receipt = json.loads(reset_path.with_suffix('.json').read_text())
    if reset_receipt['task'] != 'click_bell' or reset_receipt['count'] != 50 or reset_receipt['sha256'] != hashlib.sha256(reset_path.read_bytes()).hexdigest():
        raise ValueError('Require verified click_bell clean50 reset bundle')
    seeds = json.loads(Path(a.native_seeds).read_text()).get('click_bell', {}).get('success_seeds')
    if not isinstance(seeds, list) or len(seeds) < 32 or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('Require existing click_bell task seed list, no seed generation/filtering')
    cfg, smoke, manifest = build(json.loads(a.base_config.read_text()), name=a.name, owner_dir=a.owner_dir,
        initial_state=a.initial_state, native_seeds=a.native_seeds, sft_path=a.sft_path, service_urls=a.service_urls)
    manifest.update(base_config=str(a.base_config), base_sha256=digest)
    a.output.mkdir(parents=True, exist_ok=False)
    for filename, value in [('formal.yaml', cfg), ('startup_smoke.yaml', smoke), ('config-contract.json', manifest)]:
        (a.output / filename).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
