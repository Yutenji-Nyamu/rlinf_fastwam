"""One bounded policy batch comparison, enabled only in the startup smoke."""
import gc
import json
import time
import numpy as np
import torch


def resized(value, old, new):
    if isinstance(value, dict):
        return {key: resized(item, old, new) for key, item in value.items()}
    if torch.is_tensor(value) and value.ndim and value.shape[0] == old:
        return value.index_select(0, torch.arange(new, device=value.device) % old)
    if isinstance(value, np.ndarray) and value.ndim and value.shape[0] == old:
        return value[np.arange(new) % old]
    if isinstance(value, list) and len(value) == old:
        return [value[index % old] for index in range(new)]
    return value


def run_policy_probe(worker, env_obs):
    config = worker.cfg.rollout.get('batch_probe_sizes', None)
    if not config or getattr(worker, '_wmrl_batch_probe_complete', False):
        return
    worker._wmrl_batch_probe_complete = True
    assert list(config) == [32, 64, 128], 'Bounded batch sweep only'
    batch = int(env_obs['states'].shape[0])
    assert batch == 64 and worker._world_size == 1
    outputs = []
    for size in config:
        observation = resized(env_obs, batch, size)
        result = None
        try:
            with torch.no_grad(), torch.random.fork_rng():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for iteration in range(3):
                    started = time.perf_counter()
                    result = worker.hf_model.predict_action_batch(env_obs=observation, mode='train')
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - started)
                    actions = result[0]
                    assert actions.shape[0] == size
                    assert np.isfinite(actions).all() if isinstance(actions, np.ndarray) else torch.isfinite(actions).all()
                    del result
                    result = None
                outputs.append(dict(batch=size, warmup_seconds=samples[0], measured_seconds=samples[1:],
                    allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_peak_bytes=torch.cuda.max_memory_reserved(), status='ok'))
        except torch.cuda.OutOfMemoryError as exc:
            outputs.append(dict(batch=size, status='oom', error=str(exc)[:500]))
            if size <= 64:
                raise
        finally:
            del result, observation
            gc.collect()
            torch.cuda.empty_cache()
        if outputs[-1]['status'] != 'ok':
            break
    print('WMRL_POLICY_BATCH_PROBE ' + json.dumps(outputs), flush=True)
