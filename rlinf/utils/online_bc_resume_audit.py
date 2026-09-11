"""Read-only verification immediately after a successful online-BC load."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rlinf.utils.utils import get_rng_state


def _digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((tuple(v.shape), v.dtype)).encode())
            h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((v.shape, v.dtype)).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            for key in sorted(v): h.update(str(key).encode()); visit(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v: visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def _optimizer_signature(state):
    rows = []
    for key, item in sorted(state['state'].items()):
        rows.append({'id': key, 'step': float(item['step']),
                     'exp_avg_shape': list(item['exp_avg'].shape),
                     'exp_avg_sq_shape': list(item['exp_avg_sq'].shape),
                     'sample': _digest([item['exp_avg'].flatten()[:32], item['exp_avg_sq'].flatten()[:32]])})
    return {'states': rows, 'param_groups': state['param_groups']}


def audit_online_bc_resume(worker, load_base_path):
    base = Path(load_base_path)
    state = torch.load(base/'local_shard_checkpoint'/f'checkpoint_rank_{worker._rank}.pt',
                       map_location='cpu', mmap=True, weights_only=False)
    target = base/'online_bc'/f'rank_{worker._rank}'
    pool = torch.load(target/'success_replay.pt', map_location='cpu', mmap=True, weights_only=True)
    learner = torch.load(target/'learner.pt', map_location='cpu', weights_only=True)
    expected_optim = state['optimizers'][0]
    optim_equal = _optimizer_signature(worker.optimizer.state_dict()) == _optimizer_signature(expected_optim)
    scheduler_equal = worker.lr_scheduler.state_dict() == state['lr_schedulers'][0]
    actor_rng_equal = _digest(get_rng_state()) == _digest(state['rng'])
    pool_rng_equal = torch.equal(worker.replay_buffer.rng.get_state(), pool['rng'])
    pool_equal = (len(worker.replay_buffer.records) == len(pool['records']) and
                  worker.replay_buffer.episodes == pool['episodes'] and
                  worker.replay_buffer.archive_id == pool['archive_id'])
    result = {'checkpoint': str(base), 'model_load_completed': True,
              'optimizer_steps': sorted({float(v['step']) for v in worker.optimizer.state.values()}),
              'optimizer_states': len(worker.optimizer.state), 'optimizer_signature_matches': optim_equal,
              'optimizer_sample': 'step, groups, both moment shapes and first32 values of every parameter state',
              'scheduler_matches': scheduler_equal, 'scheduler_last_epoch': worker.lr_scheduler.last_epoch,
              'actor_rng_matches': actor_rng_equal, 'pool_rng_matches': pool_rng_equal, 'pool_counts_match': pool_equal,
              'learner_update_step': worker.update_step, 'learner_matches': worker.update_step == learner['update_step'],
              'success_episodes': worker.replay_buffer.episodes, 'query_records': len(worker.replay_buffer.records),
              'filtered_success_episodes': worker.replay_buffer.filtered_success_episodes,
              'method_state': worker.dvac_new_state() if getattr(worker, 'dvac_normalization', None) == 'two_level_batch' else None,
              'rollout_rng_and_environment': 'Not present in the historical checkpoint; original configured restart behavior retained.'}
    result['passed'] = all((optim_equal, scheduler_equal, actor_rng_equal, pool_rng_equal, pool_equal, result['learner_matches']))
    output = Path(worker.cfg.runner.logger.log_path)/'runtime'/f'restore-actor-rank{worker._rank}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    assert result['passed'], result
    worker.log_info('BC_RESUME_VERIFIED '+json.dumps(result, sort_keys=True))
