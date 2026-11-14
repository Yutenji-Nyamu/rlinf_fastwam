# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Release RoboTwin native renderer resources at the process boundary.

The RLinf wrapper retains seed scheduling, rewards and metrics. This proxy only
forwards the existing VectorEnv API and discards scenes at the same offload
boundary as VectorEnv.close(clear_cache=True).
"""

import copy
import ctypes
import importlib
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProcessVectorEnv:
    """Run a vector simulator in one disposable child process per active phase."""

    def __init__(
        self,
        *,
        rpc_timeout_s: float = 900,
        factory: str = "robotwin.envs.vector_env:VectorEnv",
        torch_threads: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._kwargs = copy.deepcopy(kwargs)
        self._factory = factory
        self._torch_threads = torch_threads
        self._timeout = rpc_timeout_s
        self._process: subprocess.Popen | None = None
        self._pipe = None
        self._lock = threading.RLock()

    @property
    def worker_pid(self) -> int | None:
        """Return the current simulator PID, or None while offloaded."""
        return self._process.pid if self._process is not None else None

    def _receive(self) -> Any:
        if not self._pipe.poll(self._timeout):
            raise TimeoutError(f"RoboTwin simulator RPC exceeded {self._timeout}s")
        try:
            ok, result = self._pipe.recv()
        except EOFError as exc:
            raise RuntimeError("RoboTwin simulator process exited during RPC") from exc
        if not ok:
            raise RuntimeError(f"RoboTwin simulator failed:\n{result}")
        return result

    def _start(self) -> None:
        if self._process is not None:
            if self._process.poll() is not None:
                raise RuntimeError("RoboTwin simulator exited before offload")
            return
        self._pipe, child_pipe = multiprocessing.Pipe()
        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(child_pipe.fileno()),
                    str(os.getpid()),
                ],
                pass_fds=(child_pipe.fileno(),),
            )
        finally:
            child_pipe.close()
        self._pipe.send((self._factory, self._kwargs, self._torch_threads))
        try:
            self._receive()
        except BaseException:
            self.close()
            raise
        logger.info("RoboTwin simulator started: pid=%s", self.worker_pid)

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            self._start()
            self._pipe.send((method, args, kwargs))
            return self._receive()

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the existing full or partial reset without changing seeds."""
        return self._call("reset", *args, **kwargs)

    def get_obs(self) -> Any:
        """Return the original NumPy image, state and instruction observations."""
        return self._call("get_obs")

    def step(self, actions: Any) -> Any:
        """Forward an unchanged action chunk and return its original results."""
        return self._call("step", actions)

    def check_seeds(self, seeds: Any) -> Any:
        """Run the simulator's seed checks in the current child process."""
        return self._call("check_seeds", seeds)

    def close(self, clear_cache: bool = True) -> None:
        """Discard the child and all its native resources at full offload."""
        with self._lock:
            if self._process is None:
                return
            if not clear_cache:
                self._call("close", clear_cache=False)
                return
            child, pipe = self._process, self._pipe
            try:
                if child.poll() is None:
                    try:
                        pipe.send(("_shutdown", (), {}))
                        child.wait(timeout=15)
                    except (BrokenPipeError, EOFError, subprocess.TimeoutExpired):
                        child.terminate()
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=15)
            finally:
                pipe.close()
                self._pipe = None
                self._process = None
            logger.info(
                "RoboTwin simulator released: pid=%s exit=%s",
                child.pid,
                child.returncode,
            )


def _worker(fd: int, parent_pid: int) -> None:
    """Serve local IPC; process exit releases Vulkan/OIDN/CUDA allocations."""
    from multiprocessing.connection import Connection

    if sys.platform == "linux":
        # Kill the renderer even if a Ray worker is terminated during rendering.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != parent_pid:
            os._exit(1)
    pipe = Connection(fd)
    try:
        factory, kwargs, torch_threads = pipe.recv()
        module, name = factory.split(":")
        cls = getattr(importlib.import_module(module), name)
        if torch_threads is not None:
            import torch

            torch.set_num_threads(torch_threads)
        env = cls(**kwargs)
        pipe.send((True, None))
        while True:
            method, args, kwargs = pipe.recv()
            if method == "_shutdown":
                # Avoid native interpreter-teardown ordering bugs. This boundary
                # already discards all scenes in the in-process implementation.
                break
            try:
                pipe.send((True, getattr(env, method)(*args, **kwargs)))
            except Exception:  # noqa: BLE001 - forward remote exceptions
                pipe.send((False, traceback.format_exc()))
    except EOFError:
        pass
    except Exception:  # noqa: BLE001 - report worker initialization failures
        pipe.send((False, traceback.format_exc()))
    finally:
        pipe.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Expected --worker PIPE_FD PARENT_PID")
    _worker(int(sys.argv[2]), int(sys.argv[3]))
