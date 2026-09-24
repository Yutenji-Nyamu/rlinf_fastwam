"""Independent, inference-only RoboTwin batches; reuse RLinf model/env APIs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time
import traceback


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def run(config_path):
    import cv2
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from rlinf.models.embodiment.openpi import get_model
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    from rlinf.algorithms.dvac_train_weighting import compute_endpoint_variance

    cfg = json.loads(Path(config_path).read_text())
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'done.json').exists(), 'Completed batch must not be repeated'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(cfg['gpu'])
    assert cfg['num_envs'] == 16 and cfg['selected_l'] == 3
    started = time.time()
    save(out / 'started.json', {'time': started, 'pid': os.getpid(), 'gpu': cfg['gpu']})
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    model = get_model(OmegaConf.create(cfg['model'])).cuda().eval()
    env_cfg = OmegaConf.create(cfg['env'])
    # No autoreset: successes remain terminal and cannot silently become new episodes.
    assert not env_cfg.auto_reset and env_cfg.ignore_terminations
    env = RoboTwinEnv(env_cfg, 16, 0, 1, None)
    obs, _ = env.reset()
    requested = env.reset_state_ids.cpu().tolist()
    # RoboTwin records the successful setup's seed as ep_num, including native retries.
    actual = [int(sub.task.ep_num) for sub in env.venv.envs]
    seeds = [{'slot': i, 'requested': requested[i], 'actual': actual[i],
              'native_retry': requested[i] != actual[i]} for i in range(16)]
    save(out / 'seeds.json', seeds)
    # Reset noise RNG after simulator setup, which also changes the global torch RNG.
    random.seed(cfg['noise_seed']); np.random.seed(cfg['noise_seed'])
    torch.manual_seed(cfg['noise_seed']); torch.cuda.manual_seed_all(cfg['noise_seed'])
    heads = obs['main_images'].cpu().numpy()
    h, w = heads.shape[1:3]
    videos = []
    frames = [[] for _ in range(16)]
    for i in range(16):
        path = out / f'episode_{i:02d}.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 4, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot open video writer {path}')
        videos.append(writer)
    success = np.zeros(16, dtype=bool)
    counts = np.zeros(16, dtype=int)
    query_rows = []
    def snapshot(images):
        for i, im in enumerate(images):
            videos[i].write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            frames[i].append(time.time())
    snapshot(heads)
    try:
        for q in range((cfg['step_limit'] + 49) // 50):
            active = ~success & (counts < cfg['step_limit'])
            if not active.any():
                break
            before = counts.copy()
            with torch.inference_mode():
                actions, result = model.predict_action_batch(
                    obs, mode='eval', compute_values=False, return_dvac_telemetry=True)
                trace = result['dvac_telemetry']
                raw = compute_endpoint_variance(trace['z_endpoint'], 3).cpu().numpy()
            if raw.shape != (16, 50) or not np.isfinite(raw).all():
                raise RuntimeError('Unexpected DV shape or nonfinite DV')
            z = trace['z_endpoint'].float().cpu().numpy()
            act = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
            assert act.shape == (16, 50, 14) and np.isfinite(act).all()
            # Native wrapper applies a whole 50-action chunk through TOPP interpolation.
            # Keep that behavior; its physical interpolation samples are not action slots.
            obss, _, _, _, _ = env.chunk_step(act)
            obs = obss[-1]
            counts = np.array([int(s.task.take_action_cnt) for s in env.venv.envs])
            after_success = np.array([bool(s.task.eval_success) for s in env.venv.envs])
            submitted = active[:, None] & (np.arange(50)[None, :] < (counts-before)[:, None])
            # On success TOPP can stop partway through interpolation; the exact original
            # waypoint prefix is unavailable. Exclude it from executed-only aggregates.
            exact = submitted & ~after_success[:, None]
            snapshot(obs['main_images'].cpu().numpy())
            np.savez_compressed(out / f'query_{q:03d}.npz', dv=raw, z_endpoint=z,
                                env_action=act, submitted_mask=submitted,
                                executed_mask=exact, active=active, success_after=after_success,
                                action_slot_start=before, action_slot_end=counts)
            for i in range(16):
                if not active[i]:
                    continue
                values = raw[i, submitted[i]]
                query_rows.append({'slot': i, 'query': q, 'action_slot_start': int(before[i]),
                    'action_slot_end': int(counts[i]), 'dv_mean': float(values.mean()),
                    'submitted': int(submitted[i].sum()), 'executed_prefix_known': not bool(after_success[i]),
                    'success_after': bool(after_success[i]), 'pre_frame': q, 'post_frame': q+1,
                    'wall_time': time.time()})
            success |= after_success
            save(out / 'progress.json', {'time': time.time(), 'query': q+1,
                 'queries_max': (cfg['step_limit']+49)//50, 'successes': int(success.sum())})
            del result, trace, actions, z
        save(out / 'queries.json', query_rows)
        episodes = [{'slot': i, 'task': cfg['task'], 'batch': cfg['batch'], **seeds[i],
            'seed_source': cfg['seed_source'], 'noise_seed': cfg['noise_seed'],
            'success': bool(success[i]), 'action_slots_submitted': int(counts[i]),
            'termination': 'success' if success[i] else 'step_limit',
            'video': f'episode_{i:02d}.mp4', 'frames': len(frames[i]),
            'video_sampling': 'head camera before/after each policy query; 4 fps preview',
            'physical_action_prefix': 'unknown for the terminal success chunk (TOPP interpolation)'}
            for i in range(16)]
        save(out / 'episodes.json', episodes)
    finally:
        for writer in videos:
            writer.release()
        env.offload(clear_cache=True)
        env.venv.env_thread_pool.shutdown(wait=True)
    save(out / 'done.json', {'time': time.time(), 'elapsed_s': time.time()-started,
        'episodes': 16, 'successes': int(success.sum()), 'queries': q+1,
        'source_commit': cfg['source_commit']})
    print('DV50_DONE ' + json.dumps(json.loads((out/'done.json').read_text())), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    try:
        run(args.config)
    except BaseException:
        cfg = json.loads(Path(args.config).read_text())
        out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
        save(out/'error.json', {'time': time.time(), 'error': traceback.format_exc()})
        raise
