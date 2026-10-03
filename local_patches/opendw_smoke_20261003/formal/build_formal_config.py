"""Prepare a fresh formal config from the frozen long smoke, without launching.

Keep its complete WM/policy/GRPO contract. Recover the Control's actual runner
budget (max_epochs=1000, max_steps=200), native eval environment and save cadence.
The eval task/seed/assets/output paths are explicit new-run inputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


CONTROL_REF = '2151a08ee1bd75df1bef0d8190e594bd5c7f7977'
PLACEMENT = {'actor': '4,5', 'env': '6,7', 'rollout': '4,5'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat(value, prefix=''):
    if not isinstance(value, dict):
        return {prefix: value}
    return {key: item for name, child in value.items()
            for key, item in flat(child, prefix + '.' + name if prefix else name).items()}


def remote(value):
    path = PurePosixPath(value)
    if not path.is_relative_to('/data/chenyiteng') or '..' in path.parts or path == PurePosixPath('/data/chenyiteng'):
        raise ValueError('Use an absolute task path strictly under /data/chenyiteng')
    return str(path)


def build(smoke, control, *, name, run_dir, services, native_assets, native_eval_seeds):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('A unique simple experiment name is required')
    if len(services) != 2 or len(set(services)) != 2:
        raise ValueError('Two independent WM service endpoints are required')
    from urllib.parse import urlparse
    for url in services:
        parsed = urlparse(url)
        if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or parsed.port is None
                or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError('WM URLs must be distinct 127.0.0.1 HTTP ports')
    if len({urlparse(url).port for url in services}) != 2:
        raise ValueError('WM service ports must differ')
    cfg = copy.deepcopy(smoke)
    e, a, r = cfg['env']['train'], cfg['actor'], cfg['runner']
    assert cfg['cluster']['component_placement'] == PLACEMENT
    assert (e['total_num_envs'], e['rollout_epoch'], e['group_size'], e['chunk']) == (64, 8, 8, 32)
    assert e['env_type'] == 'opendw_robotwin' and e['task_name'] == 'adjust_bottle'
    assert e['max_episode_steps'] == e['max_steps_per_rollout_epoch'] == 384
    assert a['global_batch_size'] == 2048 and a['micro_batch_size'] == 8 and cfg['algorithm']['update_epoch'] == 2
    assert r['max_epochs'] == r['max_steps'] == 1 and r.get('resume_dir') is None and r.get('ckpt_path') is None
    assert (control['runner']['max_epochs'], control['runner']['max_steps']) == (1000, 200)
    assert control['runner']['save_interval'] == 10
    run_dir = PurePosixPath(remote(run_dir))
    r.update(max_epochs=control['runner']['max_epochs'], max_steps=control['runner']['max_steps'],
             save_interval=control['runner']['save_interval'], val_check_interval=10,
             resume_dir=None, ckpt_path=None, per_worker_log_path=str(run_dir / 'worker-metrics'))
    r['logger'].update(log_path=str(run_dir), experiment_name=name)
    cfg['env']['group_name'] = 'OpenDWEnv_' + name
    e['service_urls'] = [url.rstrip('/') for url in services]
    e['video_cfg']['video_base_dir'] = str(run_dir / 'video/train')
    a['group_name'] = 'OpenDWActor_' + name
    cfg['rollout']['group_name'] = 'OpenDWRollout_' + name
    cfg['algorithm']['dvac_gradient_weighting']['output_dir'] = str(run_dir / 'unused-dv')
    # Start from the actual clean, three-camera Aloha Control, not the public
    # task default (Piper/randomized/head-only), and never reuse its move_can_pot
    # seed file under a renamed adjust_bottle task.
    native = copy.deepcopy(control['env']['eval'])
    assert native['env_type'] == 'robotwin' and native['total_num_envs'] == 32
    assert native['rollout_epoch'] == native['group_size'] == 1
    assert native['task_config']['embodiment'] == ['aloha-agilex']
    assert native['task_config']['camera']['collect_head_camera'] is True
    assert native['task_config']['camera']['collect_wrist_camera'] is True
    native.update(max_episode_steps=384, max_steps_per_rollout_epoch=384, enable_offload=True,
                  assets_path=remote(native_assets), seeds_path=remote(native_eval_seeds))
    native['task_config'].update(task_name='adjust_bottle', step_lim=384,
                                 save_path=str(run_dir / 'robotwin_data/eval'))
    native['video_cfg']['video_base_dir'] = str(run_dir / 'video/eval')
    assert native['is_eval'] is True and native['use_fixed_reset_state_ids'] is True
    cfg['env']['eval'] = native
    before, after = flat(smoke), flat(cfg)
    return cfg, dict(source_control_ref=CONTROL_REF, writes_configuration_only=True, launched=False,
                     start='fresh_same_SFT_as_smoke; direct formal start authorized 20261004; no CP continuation',
                     budget=dict(max_epochs=1000, max_steps=200, effective_runner_iterations=200,
                                 trajectories_per_iteration=512, chunk_slots_per_iteration=6144,
                                 update_epochs=2, scheduled_optimizer_steps_per_iteration=6,
                                 gradient_accumulation_per_actor=128,
                                 save_interval=10, native_eval_interval=10,
                                 native_eval_N=32, native_eval_R=1, native_eval_G=1,
                                 native_eval_task='adjust_bottle', native_eval_C=32, native_eval_L=384),
                     eval_note='Built-in real RoboTwin evaluation. WM env is train only. Activate the audited GPU4-7 graphics scope before launch; direct-start uses formal native initialization as the first native test.',
                     checkpoint_note='Existing runner evaluates before saving at matching intervals; an eval failure can prevent that checkpoint. No runner order change here.',
                     config_diff=[dict(key=k, before=before.get(k), after=after.get(k))
                                  for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-config', required=True, type=Path)
    parser.add_argument('--control-config', required=True, type=Path)
    parser.add_argument('--control-config-sha256', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--service-urls', nargs=2, required=True)
    parser.add_argument('--native-assets', required=True)
    parser.add_argument('--native-eval-seeds', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sha(args.control_config) != args.control_config_sha256:
        raise ValueError('Pinned Control JSON bytes differ from the reviewed source export')
    cfg, manifest = build(json.loads(args.smoke_config.read_text()), json.loads(args.control_config.read_text()),
                          name=args.name, run_dir=args.run_dir, services=args.service_urls,
                          native_assets=args.native_assets, native_eval_seeds=args.native_eval_seeds)
    # Generation runs on the server; ensure native reset selection will not
    # silently fall back to random seeds due to a missing/wrong task key.
    seeds = json.loads(Path(args.native_eval_seeds).read_text())
    values = seeds.get('adjust_bottle', {}).get('success_seeds')
    if (not isinstance(values, list) or len(values) < 32 or any(type(v) is not int for v in values)
            or len(set(values)) != len(values)):
        raise ValueError('Need the existing adjust_bottle native seed list with at least 32 unique integer seeds; do not auto-generate/filter seeds')
    if not Path(args.native_assets).is_dir():
        raise ValueError('Native RoboTwin assets checkout is missing')
    manifest.update(smoke_config=str(args.smoke_config), smoke_config_sha256=sha(args.smoke_config),
                    control_config=str(args.control_config), control_config_sha256=sha(args.control_config),
                    native_eval_seeds=str(args.native_eval_seeds), native_eval_seeds_sha256=sha(args.native_eval_seeds))
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / (args.name + '.yaml')
    target.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    manifest['config_sha256'] = sha(target)
    (args.output / 'formal-contract.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(config=str(target), contract=manifest['budget'], launched=False)))


if __name__ == '__main__':
    main()
