"""Bounded wait on an already submitted RoboTwin step; never replay an action."""
import faulthandler
import logging
import sys
from concurrent.futures import TimeoutError


def wait_for_step(future, env_id, warn_seconds=120, hard_seconds=600):
    try:
        return future.result(timeout=warn_seconds)
    except TimeoutError:
        # A TimeoutError raised by the task itself is a real task failure.
        if future.done():
            return future.result()
        logging.warning(
            "RoboTwin SubEnv %s step still pending after %ss; waiting on the "
            "same action up to %ss total", env_id, warn_seconds, hard_seconds
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        try:
            return future.result(timeout=hard_seconds - warn_seconds)
        except TimeoutError as error:
            if future.done():
                return future.result()
            raise TimeoutError(
                f"SubEnv {env_id} step exceeded {hard_seconds}s; action was not replayed"
            ) from error


def step_with_bounded_wait(self, actions):
    if len(self.envs) == 0:
        self._init_envs()
    futures = {
        i: self.env_thread_pool.submit(self.envs[i].step, actions[i])
        for i in range(self.n_envs)
    }
    results = []
    for i in range(self.n_envs):
        try:
            results.append(wait_for_step(futures[i], i))
        except Exception as error:
            raise RuntimeError(
                f"SubEnv {i} step error: {type(error).__name__}: {error!r}"
            ) from error
    return self.transform(results)
