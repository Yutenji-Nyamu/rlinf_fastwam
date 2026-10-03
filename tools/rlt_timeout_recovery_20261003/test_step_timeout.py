import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from concurrent.futures import Future, TimeoutError

p = Path(__file__).parents[2] / 'rlinf/envs/robotwin/step_timeout.py'
spec = importlib.util.spec_from_file_location('step_timeout', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Pending:
    def __init__(self, finish): self.finish, self.calls = finish, []
    def done(self): return False
    def result(self, timeout=None):
        self.calls.append(timeout)
        if len(self.calls) == 1 or not self.finish: raise TimeoutError()
        return 'same-action-result'


class Tests(unittest.TestCase):
    def test_delayed_step_waits_same_future(self):
        f = Pending(True)
        with patch.object(m.faulthandler, 'dump_traceback') as dump:
            self.assertEqual(m.wait_for_step(f, 3), 'same-action-result')
        self.assertEqual(f.calls, [120, 480])
        dump.assert_called_once()

    def test_hang_is_bounded(self):
        f = Pending(False)
        with patch.object(m.faulthandler, 'dump_traceback'):
            with self.assertRaisesRegex(TimeoutError, 'exceeded 600s'):
                m.wait_for_step(f, 1)
        self.assertEqual(f.calls, [120, 480])

    def test_task_exception_preserved(self):
        f = Future(); error = TimeoutError('inside task'); f.set_exception(error)
        with self.assertRaises(TimeoutError) as caught: m.wait_for_step(f, 0)
        self.assertIs(caught.exception, error)

    def test_ready_result_unchanged(self):
        f = Future(); value = object(); f.set_result(value)
        self.assertIs(m.wait_for_step(f, 0), value)


if __name__ == '__main__': unittest.main()
