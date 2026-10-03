"""Private evaluation adapters: fixed RNG, serialized graphics, and receipts.

The imported RLinf policy/environment methods retain their original behavior.
This module is loaded only by the new private evaluator, never by WM training.
"""
import json
import os
from pathlib import Path
import random
import time
import uuid
from contextlib import contextmanager

import numpy as np
import torch
from rlinf.envs.utils import get_env_attr
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


GRAPHICS_WAIT_SECONDS = 900


def _graphics_event(directory, rank, slot, token, event, **details):
    record = dict(time=time.time(), pid=os.getpid(), uid=os.getuid(), token=token,
                  rank=rank, slot=slot, event=event, **details)
    with (Path(directory) / f'rank-{rank}-slot-{slot}.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')


@contextmanager
def _graphics_lock(directory, rank, slot, token, operation):
    # Each spawned child opens its own descriptor; no GPU or lock state is
    # inherited from a parent. The lock belongs only to this private phase.
    assert os.environ.get('WM_OWNER_TOKEN') == token
    if os.environ.get('WM_EVAL_RENDER_BACKEND') == 'mesa_llvmpipe':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        assert os.environ.get('MUJOCO_EGL_DEVICE_ID') == os.environ.get('WM_EVAL_MESA_DEVICE_ID') == '8'
        assert os.environ.get('LIBGL_ALWAYS_SOFTWARE') == '1'
        assert os.environ.get('GALLIUM_DRIVER') == 'llvmpipe'
        assert os.environ.get('LP_NUM_THREADS') == '2'
        _graphics_event(directory, rank, slot, token, 'software_operation_start', operation=operation)
        yield  # Software contexts can run in parallel; no NVIDIA graphics lock.
        _graphics_event(directory, rank, slot, token, 'software_operation_complete', operation=operation)
        return
    import fcntl
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(rank)
    lock_path = Path(directory) / 'graphics.lock'
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    acquired = False
    started = time.monotonic()
    _graphics_event(directory, rank, slot, token, 'lock_wait', operation=operation)
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() - started >= GRAPHICS_WAIT_SECONDS:
                    raise TimeoutError(f'Private graphics lock wait exceeded {GRAPHICS_WAIT_SECONDS}s')
                time.sleep(0.1)
        _graphics_event(directory, rank, slot, token, 'lock_acquired', operation=operation)
        yield
        _graphics_event(directory, rank, slot, token, 'operation_complete', operation=operation)
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


class _SerializedReset:
    def __init__(self, reset, directory, token, rank, slot):
        self.reset = reset
        self.directory, self.token, self.rank, self.slot = directory, token, rank, slot
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        with _graphics_lock(self.directory, self.rank, self.slot, self.token,
                            f'reset-{self.count}'):
            # Preserve the original hard_reset and randomization behavior,
            # with exactly the original number of reset calls and actions.
            return self.reset(*args, **kwargs)


class _SerializedEnvFactory:
    def __init__(self, factory, directory, token, rank, slot):
        self.factory = factory
        self.directory, self.token, self.rank, self.slot = directory, token, rank, slot

    def __call__(self):
        assigned_gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
        assert assigned_gpu == str(self.rank + 2)
        assert os.environ.get('WM_EVAL_RENDER_BACKEND') == 'mesa_llvmpipe'
        assert os.environ.get('WM_EVAL_MESA_DEVICE_ID') == '8'
        # Change only the spawned simulator process. Parent EnvWorker and
        # policy workers retain the scheduler's CUDA assignment for transport.
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['WM_EVAL_MESA_DEVICE_ID']
        with _graphics_lock(self.directory, self.rank, self.slot, self.token, 'constructor'):
            env = self.factory()  # Includes the original env.seed call.
            from OpenGL import GL
            def gl_text(value):
                return value.decode() if isinstance(value, bytes) else str(value)
            graphics = dict(gl_vendor=gl_text(GL.glGetString(GL.GL_VENDOR)),
                            gl_renderer=gl_text(GL.glGetString(GL.GL_RENDERER)),
                            gl_version=gl_text(GL.glGetString(GL.GL_VERSION)),
                            nvidia_egl_loaded='libEGL_nvidia' in Path('/proc/self/maps').read_text())
            assert graphics['gl_vendor'] == 'Mesa'
            assert 'llvmpipe' in graphics['gl_renderer'].lower()
            assert not graphics['nvidia_egl_loaded']
        env.reset = _SerializedReset(env.reset, self.directory, self.token, self.rank, self.slot)
        ready = dict(time=time.time(), pid=os.getpid(), uid=os.getuid(), token=self.token,
                     rank=self.rank, slot=self.slot,
                     assigned_gpu=assigned_gpu, render_backend='mesa_llvmpipe', software_device=8,
                     cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                     lp_num_threads=os.environ.get('LP_NUM_THREADS'), **graphics)
        path = Path(self.directory) / f'rank-{self.rank}-slot-{self.slot}-ready.json'
        temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
        with temporary.open('x') as stream:
            json.dump(ready, stream, indent=2)
        # Publish only complete JSON and refuse to replace an existing receipt.
        os.link(temporary, path)
        temporary.unlink()
        return env


class FixedSeedEvalRolloutWorker(MultiStepRolloutWorker):
    def init_worker(self):
        assert self.only_eval and not self.enable_train
        assert self.cfg.env.eval.total_num_envs == 20 and self.cfg.env.eval.rollout_epoch == 25
        seed = 42 + self._rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        super().init_worker()
        assert type(self.hf_model).__name__ == "Pi0Eval"
        # Model construction can consume RNG; inference starts from the same
        # rank seed for original/CP40/CP80, after all weight loading completes.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        record = {"time": time.time(), "rank": self._rank,
                  "policy_seed": seed, "mode": "eval",
                  "model_class": type(self.hf_model).__name__,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        path = Path(self.cfg.runner.logger.log_path) / f"policy-rank-{self._rank}.json"
        with path.open("x") as stream:
            json.dump(record, stream, indent=2)


class RecordedEvalEnvWorker(EnvWorker):
    def init_worker(self):
        assert self.cfg.runner.only_eval and not self.enable_train
        assert self.cfg.env.eval.total_num_envs == 20 and self.cfg.env.eval.rollout_epoch == 25
        assert self.stage_num == 1 and self._world_size == 2
        assert self.cfg.env.eval.task_suite_name == 'libero_goal'
        assert self.cfg.env.eval.is_eval and self.cfg.env.eval.auto_reset
        assert self.cfg.env.eval.ignore_terminations
        assert self.cfg.env.eval.max_episode_steps == self.cfg.env.eval.max_steps_per_rollout_epoch == 320
        record = {'rank': self._rank, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'egl_vendor': os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES')}
        assert record['cuda_visible_devices'] == str(self._rank + 2)
        assert Path(record['egl_vendor']).is_file()
        assert os.environ.get('WM_EVAL_RENDER_BACKEND') == 'mesa_llvmpipe'
        assert os.environ.get('WM_EVAL_MESA_DEVICE_ID') == '8'
        path = Path(self.cfg.runner.logger.log_path) / f'graphics-rank-{self._rank}.json'
        directory = Path(self.cfg.runner.logger.log_path).resolve(strict=True) / 'graphics-init'
        directory.mkdir(mode=0o700, exist_ok=True)
        token = os.environ['WM_OWNER_TOKEN']
        assert str(uuid.UUID(token)) == token
        record.update(serial_graphics=False, render_backend='mesa_llvmpipe', software_device=8,
                      lp_num_threads=os.environ.get('LP_NUM_THREADS'),
                      graphics_wait_seconds=GRAPHICS_WAIT_SECONDS)
        with path.open('x') as stream:
            json.dump(record, stream, indent=2)
        from rlinf.envs.sim.libero.libero_env import LiberoEnv
        original_get_env_fns = LiberoEnv.get_env_fns

        def serialized_get_env_fns(env):
            assert env.is_eval and env.num_envs == 10
            assert env.total_num_processes == 2 and env.seed_offset == self._rank
            factories = original_get_env_fns(env)
            assert len(factories) == 10
            return [_SerializedEnvFactory(factory, str(directory), token, self._rank, slot)
                    for slot, factory in enumerate(factories)]

        LiberoEnv.get_env_fns = serialized_get_env_fns
        started = time.monotonic()
        try:
            result = super().init_worker()  # Preserve the all-rank import barrier.
            # ReconfigureSubprocEnv starts children without waiting for their
            # constructors. Wait here so no reset overlaps unfinished contexts.
            paths = [directory / f'rank-{self._rank}-slot-{slot}-ready.json'
                     for slot in range(10)]
            while not all(path.exists() for path in paths):
                if time.monotonic() - started >= GRAPHICS_WAIT_SECONDS:
                    raise TimeoutError(f'Private graphics constructors exceeded {GRAPHICS_WAIT_SECONDS}s')
                time.sleep(0.1)
            for slot, path in enumerate(paths):
                ready = json.loads(path.read_text())
                assert ready['token'] == token and ready['rank'] == self._rank
                assert ready['slot'] == slot and ready['uid'] == os.getuid()
                assert ready['assigned_gpu'] == str(self._rank + 2) and ready['cuda_visible_devices'] == ''
                assert ready['render_backend'] == 'mesa_llvmpipe' and ready['software_device'] == 8
                assert ready['gl_vendor'] == 'Mesa' and 'llvmpipe' in ready['gl_renderer'].lower()
                assert not ready['nvidia_egl_loaded'] and ready['lp_num_threads'] == '2'
            return result
        finally:
            LiberoEnv.get_env_fns = original_get_env_fns

    def evaluate(self, input_channel, rollout_channel):
        assert self.cfg.runner.only_eval
        metrics = super().evaluate(input_channel, rollout_channel)
        stages = []
        for env in self.eval_env_list:
            stats = get_env_attr(env, "_task_success_stats")
            seen = get_env_attr(env, "_eval_seen_trials")
            stages.append({"task_stats": stats,
                           "seen_task_trial_ids": sorted([list(item) for item in seen])})
        path = Path(self.cfg.runner.logger.log_path) / f"env-rank-{self._rank}.json"
        with path.open("x") as stream:
            json.dump({"time": time.time(), "rank": self._rank, "stages": stages},
                      stream, indent=2)
        return metrics
