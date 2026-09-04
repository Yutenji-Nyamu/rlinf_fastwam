"""Bounded real RoboTwin multicamera regression, without policy or training."""
import argparse
import ctypes
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

SYMBOL = ('_ZN8svulkan28renderer10RTRenderer6renderERNS_5scene6CameraE'
          'RKN2vk23ArrayProxyNoTemporariesIKNS5_9SemaphoreEEERKNS6_IKNS5_5Flags'
          'INS5_21PipelineStageFlagBitsEEEEERKNS6_IKmEESB_SM_')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--prepare', action='store_true')
    args = p.parse_args()
    out = Path(args.output)
    if args.prepare:
        out.mkdir(parents=True, exist_ok=False)
        base = OmegaConf.load(args.source_config)
        env = OmegaConf.create(OmegaConf.to_container(base.env.eval, resolve=True))
        env.total_num_envs = env.group_size = 1
        env.auto_reset = env.enable_offload = False
        env.video_cfg.save_video = False
        env.video_cfg.video_base_dir = str(out / 'video')
        env.task_config.save_path = str(out / 'robotwin_data')
        assert env.task_config.ray_tracing_denoiser == 'none'
        seeds = json.loads(Path(env.seeds_path).read_text())[env.task_config.task_name]['success_seeds'][:2]
        cfg = OmegaConf.create({'env': env, 'seeds': seeds,
            'frames_per_scene': 64, 'physics_steps_per_frame': 1,
            'physical_gpu': 6, 'policy_queries': 0, 'optimizer_updates': 0,
            'wall_timeout_seconds': 300, 'render': {'shader': 'rt', 'spp': 32, 'path_depth': 8}})
        OmegaConf.save(cfg, out / 'resolved.yaml', resolve=True)
        print(OmegaConf.to_yaml(cfg), flush=True)
        return
    cfg = OmegaConf.load(out / 'resolved.yaml')
    expected = Path(os.environ['RLINF_SCENE_FENCE_LIBRARY']).resolve()
    # Check the process-global symbol, not merely the fact that a .so was mapped.
    class DlInfo(ctypes.Structure):
        _fields_ = [('name', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]
    lib = ctypes.CDLL(None)
    fn = getattr(lib, SYMBOL)
    info = DlInfo()
    lib.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    assert lib.dladdr(ctypes.cast(fn, ctypes.c_void_p), ctypes.byref(info))
    assert Path(info.name.decode()).resolve() == expected, info.name
    print('NATIVE_BINDING', info.name.decode(), hex(info.address), flush=True)
    import numpy as np
    import torch
    import sapien
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
    torch.set_num_threads(4)
    calls = []
    original = sapien.render.set_ray_tracing_denoiser
    def check_denoiser(value):
        assert value == 'none', value
        calls.append(value)
        return original(value)
    sapien.render.set_ray_tracing_denoiser = check_denoiser
    start = time.monotonic()
    env = RoboTwinEnv(cfg.env, 1, 0, 1, None)
    records = []
    try:
        for seed in cfg.seeds:
            env.reset(env_seeds=[int(seed)])
            task = env.venv.envs[0].task
            for frame in range(cfg.frames_per_scene):
                task.scene.step()
                obs = task.get_obs()
                images = {key: obs['observation'][key]['rgb'] for key in
                          ['left_camera', 'right_camera', 'head_camera']}
                for key, value in images.items():
                    assert value.ndim == 3 and value.shape[-1] == 3, (key, value.shape)
                    assert np.isfinite(value).all()
                if frame in (0, cfg.frames_per_scene - 1):
                    row = {'seed': int(seed), 'frame': frame,
                           'shapes': {k: list(v.shape) for k, v in images.items()},
                           'means': {k: float(v.mean()) for k, v in images.items()}}
                    records.append(row)
                    print('FRAME', json.dumps(row), flush=True)
    finally:
        env.offload()
        env.venv.env_thread_pool.shutdown(wait=True)
    result = {'status': 'passed', 'scenes': len(cfg.seeds),
              'frames': len(cfg.seeds) * cfg.frames_per_scene,
              'camera_images': len(cfg.seeds) * cfg.frames_per_scene * 3,
              'elapsed_seconds': time.monotonic() - start,
              'denoiser_calls': calls, 'native_library': str(expected), 'records': records}
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('SMOKE_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
