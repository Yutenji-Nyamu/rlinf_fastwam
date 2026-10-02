"""Close an EXPO-owned RoboTwin vector only after its worker threads have left."""
from concurrent.futures import ThreadPoolExecutor


def close_robotwin_env(env):
    if env is None or getattr(env, '_expo_lifecycle_closed', False):
        return
    vector = env.venv
    pool = vector.env_thread_pool
    if not isinstance(pool, ThreadPoolExecutor):
        raise TypeError('EXPO lifecycle fix requires the inspected thread-based VectorEnv')
    # Future.result() returns before the worker loop drops its last work item.
    # Join first, while all scene and planner objects are still valid.
    pool.shutdown(wait=True, cancel_futures=True)
    if any(thread.is_alive() for thread in pool._threads):
        raise RuntimeError('RoboTwin worker survived shutdown; refuse native teardown')
    env.offload()
    env._expo_lifecycle_closed = True
