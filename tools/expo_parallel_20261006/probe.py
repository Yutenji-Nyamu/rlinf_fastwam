"""Isolated restored-state N1/N4 simulation and complete unchanged B64 learner call."""
import argparse
import copy
import os
import random
import sys
import time
import traceback
from common import *

def main():
    p = argparse.ArgumentParser(); p.add_argument('--devices', type=int, choices=(1, 2), required=True)
    a = p.parse_args(); output = STAGE / f'trial-{a.devices}'
    assert os.getuid() == 20001 and os.environ['CUDA_VISIBLE_DEVICES'] == ','.join(UUIDS[:a.devices])
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    sys.path.insert(0, str(SOURCE / 'examples/embodiment'))
    from train_expo_formal import finite, restore_rng, rng_state
    from train_expo_ft import digest, frozen_sample
    from rlinf.algorithms.expo_ft.backend import Pi05Backend, create_robotwin_env
    from rlinf.algorithms.expo_ft.core import ExpoConfig, ExpoLearner
    from rlinf.algorithms.expo_ft.formal_replay import FormalReplay
    from rlinf.algorithms.expo_ft.lifecycle import close_robotwin_env
    heartbeat = output / 'heartbeat'
    def state(phase, **extra):
        heartbeat.touch(); atomic(output / 'status.json', dict(time=time.time(), phase=phase, **extra))
    state('initializing'); torch.set_num_threads(4)
    assert torch.cuda.device_count() == a.devices
    inputs = read(STAGE / 'baseline/inputs.json'); cfg = inputs['formal']; seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    before_index = sha(TRAIN / 'replay/index.json')
    saved = torch.load(STAGE / 'baseline/checkpoint-latest.pt', map_location='cpu', weights_only=False)
    assert {k: digest(saved[k]) for k in PAYLOAD} == saved['hashes']; finite(saved)
    change_devices(saved, a.devices)
    expected = {k: digest(saved[k]) for k in PAYLOAD}
    backend = Pi05Backend(OmegaConf.create(inputs['model']), device='cuda:0', lr=cfg['base_lr'],
        candidate_microbatch=cfg['candidate_microbatch'], observation_microbatch=cfg['observation_microbatch'],
        fm_microbatch=cfg['fm_microbatch'], parallel_devices=a.devices, image_augmentation=False,
        source_head=inputs['source_head'])
    core = dict(inputs['core'], parallel_devices=a.devices)
    learner = ExpoLearner(ExpoConfig(**core), device='cuda:0', seed=seed)
    replay = FormalReplay(root=TRAIN / 'replay', demo_path=inputs['demo_path'], seed=seed)
    generator = torch.Generator(device='cuda:0').manual_seed(seed)
    backend.load_state_dict(saved['base']); learner.load_state_dict(saved['core']); replay.load_state_dict(saved['replay'])
    restore_rng(saved['rng'], generator)
    restored = dict(base=backend.state_dict(), core=learner.state_dict(), replay=replay.state_dict(),
                    cadence=saved['cadence'], rng=rng_state(generator), progress=saved['progress'])
    assert {k: digest(restored[k]) for k in PAYLOAD} == expected
    del restored
    atomic(output / 'restore.json', dict(ok=True, hashes=expected, counters=saved['cadence']['counters']))
    frozen = frozen_sample(backend); timings = {}; env = None
    seeds = read(Path(inputs['evaluation']['seed_path']))['turn_switch']['success_seeds']
    try:
        for n in (1, 4):
            state(f'native-{n}-env'); t = time.monotonic()
            config = copy.deepcopy(inputs['env'] if n == 1 else inputs['evaluation']['config'])
            config['task_config']['save_path'] = str(output / f'native-{n}')
            config.setdefault('video_cfg', {})['save_video'] = False
            env = create_robotwin_env(OmegaConf.create(config), num_envs=n, seed_offset=0)
            obs, _ = env.reset(env_seeds=seeds[:n])
            start = time.monotonic()
            base = backend.sample_normalized(obs, num_candidates=8, generator=generator)[:, :, :10, :14]
            selected = learner.select_actions(backend.critic_observation(obs), base)['actions']
            actions = backend.decode(obs, selected); finite(actions)
            torch.cuda.synchronize(); inference_seconds = time.monotonic() - start
            for step in range(10):
                obs, *_ = env.step(actions[:, step:step+1, :], auto_reset=False); heartbeat.touch()
            atomic(output / f'native-{n}.json', dict(ok=True, envs=n, actions_per_env=10,
                   candidates_shape=list(base.shape), renderer=env.expo_renderer_binding,
                   inference_seconds=inference_seconds, training_actions_committed=0))
            close_robotwin_env(env); env = None
            timings[f'native-{n}'] = time.monotonic() - t
            del obs, base, selected, actions
        # Independent trial reuses exactly the checkpoint sampling/RNG state.
        replay.load_state_dict(saved['replay']); restore_rng(saved['rng'], generator)
        learner.generator.set_state(saved['core']['_extra_state']['rng_state'].cpu())
        start_calls = learner.update_calls; del saved
        for device in range(a.devices): torch.cuda.reset_peak_memory_stats(device)
        state('learner-Q20-B64'); t = time.monotonic()
        def sample():
            heartbeat.touch(); return replay.sample(64, backend, 'cuda:0')
        def candidates(obs):
            heartbeat.touch(); return backend.sample_normalized(obs['env_obs'], num_candidates=8, generator=generator)[:, :, :10, :14]
        def fm():
            state('learner-FM-B64'); obs, actions, _ = replay.sample_fm(64)
            return backend.fm_update(obs, actions)
        metrics = learner.update_call(sample, candidates, fm); finite(metrics)
        torch.cuda.synchronize(); timings['learner_seconds'] = time.monotonic() - t
        assert learner.update_calls == start_calls + 1 and backend.base_updates == start_calls + 1
        assert learner.critic_steps == (start_calls + 1) * 20 and frozen_sample(backend) == frozen
        assert metrics['base/fm_callback_called'] == 1 and metrics['base/base_global_batch'] == 64
        assert sha(TRAIN / 'replay/index.json') == before_index
        atomic(output / 'result.json', dict(ok=True, time=time.time(), devices=a.devices, timings=timings,
               metrics=metrics, peak_allocated=[torch.cuda.max_memory_allocated(d) for d in range(a.devices)],
               peak_reserved=[torch.cuda.max_memory_reserved(d) for d in range(a.devices)],
               production_unchanged=True, trial_updates_discarded=True, native_envs=[1, 4], batch_size=64))
        state('passed')
    finally:
        if env is not None: close_robotwin_env(env)

if __name__ == '__main__':
    try: main()
    except BaseException as error:
        d = int(sys.argv[sys.argv.index('--devices') + 1]); output = STAGE / f'trial-{d}'
        atomic(output / 'result.json', dict(ok=False, time=time.time(), devices=d, type=type(error).__name__,
               error=str(error), traceback=traceback.format_exc(), production_unchanged=True))
        raise
