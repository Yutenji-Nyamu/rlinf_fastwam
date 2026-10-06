"""Restore an owned two-rank FSDP1 local checkpoint into one identical rank.

The model dictionaries are full tensors in this deployment. Adam moments are
local flat-parameter shards. Preserve both moments, steps, scheduler and rank-0
RNG, checking the destination flat sizes rather than resetting the optimizer.
"""
import copy
import json
from pathlib import Path
import torch


def merge_optimizer_shards(left, right, optimizer):
    assert left['param_groups'] == right['param_groups'], 'Source optimizer groups differ'
    destination = optimizer.state_dict()
    assert len(destination['param_groups']) == len(left['param_groups'])
    sizes = {}
    for saved, current, live in zip(left['param_groups'], destination['param_groups'], optimizer.param_groups):
        assert saved['params'] == current['params'], 'Flat parameter ordering changed'
        assert len(saved['params']) == len(live['params'])
        for index, parameter in zip(saved['params'], live['params']):
            assert parameter.ndim == 1, 'This conversion requires FSDP1 flat parameters'
            sizes[index] = parameter.numel()
    assert left['state'].keys() == right['state'].keys()
    state, proof = {}, []
    for index, source in left['state'].items():
        other = right['state'][index]
        assert source.keys() == other.keys()
        target, count = {}, sizes[index]
        for key, value in source.items():
            peer = other[key]
            if not torch.is_tensor(value):
                assert value == peer
                target[key] = copy.deepcopy(value)
            elif value.ndim == 0:
                assert torch.equal(value, peer), 'Optimizer scalar differs across ranks'
                target[key] = value.clone()
            else:
                assert value.ndim == peer.ndim == 1 and value.dtype == peer.dtype
                assert value.shape == peer.shape
                if value.numel() == (count + 1) // 2:
                    joined = torch.cat((value, peer))
                    assert 0 <= joined.numel() - count <= 1
                    target[key] = joined[:count].clone()
                elif value.numel() == count:
                    assert torch.equal(value, peer), 'Replicated optimizer state differs'
                    target[key] = value.clone()
                else:
                    raise ValueError(f'Unexpected shard size for parameter {index}: {value.numel()} -> {count}')
                assert target[key].shape == (count,) and torch.isfinite(target[key]).all()
        state[index] = target
        proof.append({'parameter': index, 'numel': count,
                      'step': float(target['step']) if 'step' in target else None})
    return {'state': state, 'param_groups': copy.deepcopy(left['param_groups'])}, proof


def load_two_rank(model, optimizer, scheduler, load_path):
    from rlinf.utils.utils import set_rng_state
    assert torch.distributed.get_world_size() == 1
    root = Path(load_path) / 'local_shard_checkpoint'
    paths = [root / f'checkpoint_rank_{rank}.pt' for rank in (0, 1)]
    assert all(path.is_file() and not path.is_symlink() for path in paths)
    checkpoints = [torch.load(p, map_location='cpu', weights_only=False, mmap=True) for p in paths]
    a, b = checkpoints
    assert a['fsdp_version'] == b['fsdp_version'] == 'fsdp'
    assert a['model'].keys() == b['model'].keys()
    assert all(type(v) is torch.Tensor for v in a['model'].values())
    for name, value in a['model'].items():
        assert value.shape == b['model'][name].shape and value.dtype == b['model'][name].dtype
        assert torch.equal(value, b['model'][name]), f'Full model differs between ranks: {name}'
    assert a['lr_schedulers'] == b['lr_schedulers']
    merged, proof = merge_optimizer_shards(a['optimizers'], b['optimizers'], optimizer)
    model.load_state_dict(a['model'], strict=True)
    optimizer.load_state_dict(merged)
    schedulers = list(scheduler) if isinstance(scheduler, (list, tuple)) else [scheduler]
    assert len(schedulers) == len(a['lr_schedulers'])
    for live, saved in zip(schedulers, a['lr_schedulers']):
        live.load_state_dict(saved)
    set_rng_state(a['rng'])
    print('WMRL_TWO_TO_ONE_RESTORE ' + json.dumps({
        'source': str(root), 'source_world_size': 2, 'target_world_size': 1,
        'full_model_keys': len(a['model']), 'optimizer_parameters': proof,
        'model_strict_loaded': True, 'optimizer_preserved': True,
        'scheduler_preserved': True, 'rng_source_rank': 0,
    }), flush=True)
