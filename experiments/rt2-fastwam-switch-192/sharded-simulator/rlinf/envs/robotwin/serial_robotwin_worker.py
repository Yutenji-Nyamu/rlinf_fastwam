"""Construct the installed RoboTwin vector on the simulator process's main thread."""

import faulthandler
import os
import signal
import threading
from concurrent.futures import Future


class MainThreadExecutor:
    def submit(self, function, *args, **kwargs):
        assert threading.current_thread() is threading.main_thread()
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future

    def shutdown(self, wait=True):
        pass


def make_vector(*, first_index, **kwargs):
    assert threading.current_thread() is threading.main_thread()
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    if os.environ.get("RLINF_SCENE_FENCE_LIBRARY"):
        from scene_fence import load_scene_fence
        load_scene_fence()
    from robotwin.envs.vector_env import VectorEnv
    import sapien

    # Both pinned RoboTwin tasks set these limits before their first renderer.
    # SAPIEN caches the engine via a weak pointer. Retain one reference across
    # scene teardown, including the throwaway scenes used to validate seeds.
    sapien.render.set_global_config(max_num_materials=50000, max_num_textures=50000)
    renderer = sapien.SapienRenderer()

    # Both pinned VectorEnv implementations construct SubEnvs synchronously.
    # Keep their task-specific setup, reset retries, action and observation code.
    env = VectorEnv(**kwargs)
    env._renderer_lifetime = renderer
    env.env_thread_pool.shutdown(wait=True)
    env.env_thread_pool = MainThreadExecutor()
    for index, scene in enumerate(env.envs, start=first_index):
        scene.env_id = index
    return env
