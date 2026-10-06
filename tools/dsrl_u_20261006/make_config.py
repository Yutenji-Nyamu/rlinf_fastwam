"""Derive the authorized pair from the completed DSRL resolved configuration.

Run on SZ1. Never starts a worker; writes one new run's frozen input files.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.algorithms.dsrl_ugrow import U_SPEC


def flatten(value, prefix=''):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            out.update(flatten(child, f'{prefix}.{key}' if prefix else key))
        return out
    return {prefix: value}


def build(role, gpu, run, repo, phase='formal', micro_batch=64, resume=None,
          worker_env=None):
    assert (role, gpu) in [('clean', 6), ('u', 7)]
    assert micro_batch in (64, 128, 256)
    template = Path(__file__).parent / 'provenance/dsrl_pi0_formal_200.yaml'
    original = OmegaConf.to_container(OmegaConf.load(template), resolve=True)
    cfg = copy.deepcopy(original)
    model_path = '/data/chenyiteng/models/rlinf-native/sidney-pi05-robotwin-e49e2ab'
    model = cfg['actor']['model']
    model.update(model_path=model_path, num_action_chunks=10, num_steps=10)
    model['openpi'].update(config_name='pi05_sidney_robotwin', action_chunk=10,
                           num_steps=10, dsrl_u_enabled=role == 'u',
                           dsrl_u_spec=copy.deepcopy(U_SPEC) if role == 'u' else None)
    model['openpi_data'] = {
        'repo_id': 'SidneyXie/pi05_robotwin',
        'norm_stats_path': model_path + '/physical-intelligence/robotwin/norm_stats.json',
        'default_prompt': 'adjust the bottle',
    }
    cfg['rollout']['model']['model_path'] = model_path
    cfg['actor']['micro_batch_size'] = micro_batch
    cfg['cluster']['component_placement']['actor, env, rollout'] = str(gpu)
    if worker_env is not None:
        assert not any(k in worker_env for k in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'))
        assert worker_env['RLINF_CODE_WORKING_DIR'] == str(repo)
        assert worker_env['RLINF_OPENDW_GPU_SCOPE_MANIFEST']
        label = f'dsrl_gpu{gpu}'
        cfg['cluster']['node_groups'] = [{
            'label': label, 'node_ranks': '0',
            'env_configs': [{'node_ranks': '0',
                            'python_interpreter_path': '/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python',
                            'env_vars': [
                {key: str(value)} for key, value in sorted(worker_env.items())
            ]}],
        }]
        cfg['cluster']['component_placement'] = {
            'actor,env,rollout': {'node_group': label, 'placement': str(gpu)}
        }
    cfg['algorithm']['dsrl_u'] = {
        'enabled': role == 'u', 'spec': copy.deepcopy(U_SPEC) if role == 'u' else None,
        'temperature': 2.5, 'log_eps': 1e-12, 'minmax_eps': 1e-6,
    }
    cfg['algorithm']['replay_buffer']['schema_version'] = 2 if role == 'u' else 1
    for split in ('train', 'eval'):
        env = cfg['env'][split]
        env['seeds_path'] = str(Path(repo) / f'rlinf/envs/robotwin/seeds/{split}_seeds.json')
        env['video_cfg']['video_base_dir'] = str(run / 'video' / split)
        env['task_config']['save_path'] = str(run / 'robotwin_data' / split)
    cfg['runner']['logger'].update(log_path=str(run), experiment_name=run.name)
    cfg['runner']['resume_dir'] = str(resume) if resume else None
    if phase in ('smoke', 'resume-smoke'):
        # Exercise Gaussian collection, learned latent, eval, save and fresh-process
        # restore. Formal parameters are never inferred from this reduced budget.
        cfg['runner'].update(max_steps=2 if phase == 'smoke' else 3,
                             val_check_interval=2, save_interval=2)
        cfg['algorithm']['replay_buffer']['warmup_size'] = 1
        for split in ('train', 'eval'):
            cfg['env'][split]['max_steps_per_rollout_epoch'] = 20
        cfg['env']['eval']['rollout_epoch'] = 1
        assert (resume is None) == (phase == 'smoke')
    else:
        assert phase == 'formal' and resume is None
    for split in ('train', 'eval'):
        assert cfg['env'][split]['total_num_envs'] == 4
    assert cfg['actor']['global_batch_size'] == 256
    assert cfg['algorithm']['utd_ratio'] == 20
    before, after = flatten(original), flatten(cfg)
    diff = {k: {'old': before.get(k), 'new': after.get(k)}
            for k in sorted(before.keys() | after.keys()) if before.get(k) != after.get(k)}
    return cfg, {'role': role, 'gpu': gpu, 'phase': phase, 'changes': diff,
                 'base_sha256': hashlib.sha256(template.read_bytes()).hexdigest(),
                 'meaning': 'Historical collection U weights the whole actor SAC loss; TD/alpha/sampling unchanged.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['clean', 'u'], required=True)
    p.add_argument('--gpu', type=int, choices=[6, 7], required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--phase', choices=['formal', 'smoke', 'resume-smoke'], default='formal')
    p.add_argument('--micro-batch', type=int, default=64)
    p.add_argument('--resume', type=Path)
    p.add_argument('--environment', type=Path, required=True,
                   help='Task-private runtime environment from the lease preparation.')
    a = p.parse_args()
    assert a.run.is_absolute() and a.repo.is_absolute()
    worker_env = json.loads(a.environment.read_text())
    cfg, report = build(a.role, a.gpu, a.run, a.repo, a.phase, a.micro_batch, a.resume, worker_env)
    runtime = a.run / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    output = runtime / 'resolved.yaml'
    assert not output.exists() and not (runtime / 'config-diff.json').exists()
    output.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)), encoding='utf-8')
    report['resolved_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    (runtime / 'config-diff.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'config': str(output), 'sha256': report['resolved_sha256'],
                      'changed_fields': len(report['changes']), 'phase': a.phase}))


if __name__ == '__main__':
    main()
