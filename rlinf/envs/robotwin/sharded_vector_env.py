"""Parallelize independent RoboTwin shards using the existing disposable-process RPC."""

import logging
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .process_vector_env import ProcessVectorEnv

logger = logging.getLogger(__name__)


class SimulatorTransportError(RuntimeError):
    pass


class _Shard(ProcessVectorEnv):
    def _start(self):
        if self._process is not None:
            return super()._start()
        requested = self.operation
        self.operation = "initialize"
        try:
            return super()._start()
        finally:
            self.operation = requested

    def _call(self, method, *args, **kwargs):
        self.operation = method
        started = time.monotonic()
        try:
            return super()._call(method, *args, **kwargs)
        except (TimeoutError, OSError, EOFError) as exc:
            raise SimulatorTransportError(f"RoboTwin shard {self._kwargs['first_index']} "
                                          f"pid={self.worker_pid} operation={method}: {exc}") from exc
        except RuntimeError as exc:
            if str(exc).startswith(("RoboTwin simulator process exited", "RoboTwin simulator exited before")):
                raise SimulatorTransportError(f"RoboTwin shard {self._kwargs['first_index']} "
                                              f"operation={method}: {exc}") from exc
            raise
        finally:
            elapsed = time.monotonic() - started
            if elapsed >= 30 and method == "reset":
                logger.info("RoboTwin reset: first_env=%s count=%s pid=%s seconds=%.1f",
                            self._kwargs["first_index"], self._kwargs["n_envs"], self.worker_pid, elapsed)

    def _receive(self):
        try:
            return super()._receive()
        except TimeoutError:
            child = self._process
            logger.error("RoboTwin stalled: pid=%s first_env=%s count=%s operation=%s",
                         self.worker_pid, self._kwargs["first_index"], self._kwargs["n_envs"], self.operation)
            if child is not None and child.poll() is None:
                try:
                    status = Path(f"/proc/{child.pid}/status").read_text().splitlines()
                    caught = next(line.split()[1] for line in status if line.startswith("SigCgt:"))
                    if int(caught, 16) & (1 << (signal.SIGUSR1 - 1)):
                        child.send_signal(signal.SIGUSR1)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            raise


class ShardedVectorEnv:
    def __init__(self, *, n_envs, env_seeds, shards, torch_threads=1, **kwargs):
        self.n_envs = n_envs
        count = min(shards, n_envs)
        if count < 1:
            raise ValueError("At least one simulator shard is required")
        self.parts = []
        self.pool = None
        seeds = list(range(n_envs)) if env_seeds is None else list(env_seeds)
        if len(seeds) != n_envs:
            raise ValueError("Expected one initial seed per environment")
        for i in range(count):
            start, end = i * n_envs // count, (i + 1) * n_envs // count
            worker = _Shard(factory="serial_robotwin_worker:make_vector", torch_threads=torch_threads,
                            first_index=start, n_envs=end-start, env_seeds=seeds[start:end], **kwargs)
            self.parts.append((start, end, worker))

    @property
    def env_thread_pool(self):
        """The EXPO lifecycle joins outstanding work before offloading this vector."""
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=len(self.parts), thread_name_prefix="robotwin-ipc")
        return self.pool

    def _map(self, requests):
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=len(self.parts), thread_name_prefix="robotwin-ipc")
        futures = [self.pool.submit(fn, *args) for fn, args in requests]
        return [f.result() for f in futures]

    def reset(self, env_idx=None, env_seeds=None):
        if env_idx is None:
            indices = list(range(self.n_envs))
        elif isinstance(env_idx, int):
            indices = [env_idx]
        else:
            indices = list(map(int, env_idx))
        if len(set(indices)) != len(indices) or any(i < 0 or i >= self.n_envs for i in indices):
            raise ValueError("Invalid RoboTwin reset indices")
        if env_seeds is None:
            by_index = {}
        elif len(env_seeds) == len(indices):
            by_index = dict(zip(indices, env_seeds))
        elif len(env_seeds) == self.n_envs:
            # The pinned vector keeps each scene's current seed when a partial
            # reset is paired with a full-width seed array.
            by_index = {}
        else:
            raise ValueError("Expected full-vector or indexed reset seeds")
        requests = []
        for start, end, worker in self.parts:
            selected = [i for i in indices if start <= i < end]
            if selected:
                seeds = [int(by_index[i]) for i in selected] if by_index else None
                requests.append((worker.reset, ([i-start for i in selected], seeds)))
        try:
            self._map(requests)
        except SimulatorTransportError:
            if len(indices) != self.n_envs or not by_index:
                raise
            # Every lane is being reset with an explicit seed. No partial rollout
            # survives, so the complete reset can be repeated once.
            logger.exception("Recreating simulator shards for the same full reset")
            self.close()
            self._map(requests)

    def get_obs(self):
        return [obs for part in self._map([(w.get_obs, ()) for _, _, w in self.parts]) for obs in part]

    def step(self, actions):
        if len(actions) != self.n_envs:
            raise ValueError("Expected one action chunk per environment")
        result = self._map([(w.step, (actions[a:b],)) for a, b, w in self.parts])
        return tuple([item for part in field for item in part] for field in zip(*result))

    def check_seeds(self, seeds):
        if len(seeds) != self.n_envs:
            raise ValueError("Expected one seed per environment")
        return [item for part in self._map([(w.check_seeds, (seeds[a:b],)) for a, b, w in self.parts]) for item in part]

    def close(self, clear_cache=True):
        if self.pool is None:
            return
        try:
            if self.pool._shutdown:
                # A caller already joined the executor. Native RPC workers still need to close.
                for _, _, worker in self.parts:
                    worker.close(clear_cache)
            else:
                self._map([(w.close, (clear_cache,)) for _, _, w in self.parts])
        finally:
            if clear_cache:
                self.pool.shutdown(wait=True)
                self.pool = None
