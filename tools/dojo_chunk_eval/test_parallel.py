"""Focused regression checks for inter-task isolation and unchanged episode budgets."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from parallel_queue import next_distinct, pending_jobs
from run_task import unowned_gpu_processes
from shared_policy import clone_model


class ParallelTests(unittest.TestCase):
    def test_gpu_snapshot_exit_does_not_abort_siblings(self):
        resident = {'pid': 1, 'uid': 1003, 'start': 10, 'state': 'R'}
        other = {'pid': 3, 'uid': 2000, 'start': 20, 'state': 'R'}
        o = SimpleNamespace(proc=lambda pid: {1: resident, 3: other}.get(pid),
                            same=lambda identity: identity['pid'] in (1, 3))
        with patch('run_task.shared_process', return_value=False):
            self.assertEqual(unowned_gpu_processes(o, [{'pid': 1}, {'pid': 2}], resident, '/private'), [])
            self.assertEqual(unowned_gpu_processes(o, [{'pid': 3}], resident, '/private'), [other])

    def test_private_policy_state_shared_weights(self):
        weight, executable = object(), object()
        base = SimpleNamespace(policy=SimpleNamespace(
            _model=weight, _sample_actions=executable,
            _sample_kwargs={'num_steps': 10}, _rng=object()),
            observation_window={'old': True}, _latest_env_idx_list=[9])
        a, b = clone_model(base), clone_model(base)
        a.policy._sample_kwargs['num_steps'] = 1
        a.policy._rng = object()
        a.observation_window = {'task': 'A'}
        a._latest_env_idx_list.append(3)
        self.assertEqual(b.policy._sample_kwargs['num_steps'], 10)
        self.assertIsNone(b.observation_window)
        self.assertEqual(b._latest_env_idx_list, [0])
        self.assertIsNot(a.policy._rng, b.policy._rng)
        self.assertIs(a.policy._model, b.policy._model)
        self.assertIs(a.policy._sample_actions, b.policy._sample_actions)

    def test_distinct_tasks_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = out / 'tasks/a-c10-n1/result.json'
            result.parent.mkdir(parents=True)
            result.write_text('{}')
            jobs = pending_jobs(['a', 'b', 'c'], out)
            self.assertEqual(len(jobs), 11)
            i = next_distinct(jobs, {0: {'task': 'a'}, 1: {'task': 'b'}})
            self.assertEqual(jobs[i][0], 'c')
            self.assertIsNone(next_distinct(jobs, {i:{'task':t} for i,t in enumerate(['a','b','c'])}))


if __name__ == '__main__':
    unittest.main()
