"""Server-only ownership/ordering regression checks; no torch or GPU import."""
import importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from types import SimpleNamespace

path=Path(__file__).resolve().parents[1]/'rlinf/algorithms/expo_ft/lifecycle.py'
spec=importlib.util.spec_from_file_location('expo_lifecycle_under_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class LifecycleChecks(unittest.TestCase):
    def test_worker_native_analogue_destroyed_before_scene_close(self):
        events=[];pool=ThreadPoolExecutor(max_workers=1)
        local=threading.local()
        class NativeResource:
            def __del__(self):events.append('worker_resource_destroyed')
        def work():local.resource=NativeResource()
        pool.submit(work).result()
        def offload():
            self.assertEqual(events,['worker_resource_destroyed'])
            self.assertTrue(all(not t.is_alive() for t in pool._threads))
            events.append('scene_closed')
        env=SimpleNamespace(venv=SimpleNamespace(env_thread_pool=pool),offload=offload)
        module.close_robotwin_env(env)
        self.assertEqual(events,['worker_resource_destroyed','scene_closed'])

    def test_close_waits_for_pending_work_and_is_idempotent(self):
        events=[];pool=ThreadPoolExecutor(max_workers=1)
        started=threading.Event();release=threading.Event()
        def work():started.set();release.wait();events.append('work_done')
        pool.submit(work);self.assertTrue(started.wait(1))
        timer=threading.Timer(.1,release.set);timer.start()
        def offload():
            self.assertEqual(events,['work_done']);events.append('closed')
        env=SimpleNamespace(venv=SimpleNamespace(env_thread_pool=pool),offload=offload)
        module.close_robotwin_env(env);module.close_robotwin_env(env);timer.join()
        self.assertEqual(events,['work_done','closed'])

    def test_unknown_vector_cannot_silently_bypass_lifecycle(self):
        env=SimpleNamespace(venv=SimpleNamespace(env_thread_pool=object()))
        with self.assertRaises(TypeError):module.close_robotwin_env(env)


if __name__=='__main__':unittest.main()
