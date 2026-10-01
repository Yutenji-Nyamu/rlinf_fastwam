"""Server-only stdlib signal/commit boundary checks; no CUDA/model import.

Run on the approved server: python -B tests/test_expo_formal_driver.py.
The AST extraction exercises the actual driver boundary helpers without
constructing native models, environments or changing any resource owner.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest


path = Path(__file__).resolve().parents[1] / 'examples/embodiment/train_expo_formal.py'
tree = ast.parse(path.read_text())
names = {'StopRequested', 'StopControl', 'atomic_json', 'native_boundary', 'periodic_evaluation_due'}
selected = ast.Module(body=[node for node in tree.body
                           if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names],
                      type_ignores=[])
namespace = dict(contextmanager=contextmanager, json=json, os=os)
exec(compile(selected, str(path), 'exec'), namespace)
StopRequested = namespace['StopRequested']
StopControl = namespace['StopControl']
atomic_json = namespace['atomic_json']
native_boundary = namespace['native_boundary']
periodic_evaluation_due = namespace['periodic_evaluation_due']


class DriverBoundaryChecks(unittest.TestCase):
    def test_success_on_native_horizon_keeps_terminal_without_timeout(self):
        terminal, timeout, precedence = native_boundary(True, True)
        self.assertTrue(terminal)
        self.assertFalse(timeout)
        self.assertTrue(precedence)

    def test_native_boundary_retains_success_timeout_and_running_cases(self):
        for flags, expected in (((True, False), (True, False, False)),
                                ((False, True), (False, True, False)),
                                ((False, False), (False, False, False))):
            with self.subTest(flags=flags):
                self.assertEqual(native_boundary(*flags), expected)

    def test_budget_complete_skips_periodic_to_run_only_final(self):
        evaluation = {'every_episodes': 25}; progress = {'evaluation_periodic': []}
        for episodes in (25, 50, 100):
            with self.subTest(episodes=episodes):
                self.assertFalse(periodic_evaluation_due(
                    SimpleNamespace(remaining_actions=0, episodes_completed=episodes), evaluation, progress))
                self.assertTrue(periodic_evaluation_due(
                    SimpleNamespace(remaining_actions=1, episodes_completed=episodes), evaluation, progress))

    def test_periodic_requires_real_due_uncommitted_boundary(self):
        evaluation = {'every_episodes': 25}
        for episodes, completed in ((0, []), (24, []), (25, [25])):
            with self.subTest(episodes=episodes, completed=completed):
                self.assertFalse(periodic_evaluation_due(
                    SimpleNamespace(remaining_actions=1, episodes_completed=episodes), evaluation,
                    {'evaluation_periodic': completed}))

    def test_signal_in_commit_is_deferred_until_publication(self):
        stop = StopControl(); events = []
        with self.assertRaises(StopRequested):
            with stop.transaction():
                stop.handler(signal.SIGTERM, None)
                events.append('replay-published')
                events.append('checkpoint-published')
        self.assertEqual(events, ['replay-published', 'checkpoint-published'])
        self.assertEqual(stop.depth, 0)

    def test_collection_records_signal_until_real_boundary_is_saved(self):
        stop = StopControl(); stop.collecting = True
        stop.handler(signal.SIGTERM, None)
        self.assertTrue(stop.requested)
        stop.check()
        with stop.transaction():
            stop.check()
        stop.collecting = False
        with self.assertRaises(StopRequested):
            stop.check()

    def test_signal_inside_learner_unwinds_without_commit(self):
        stop = StopControl()
        with self.assertRaises(StopRequested):
            stop.handler(signal.SIGTERM, None)
        self.assertEqual(stop.depth, 0)

    def test_failed_commit_preserves_the_actual_error(self):
        stop = StopControl()
        with self.assertRaisesRegex(ValueError, 'disk failure'):
            with stop.transaction():
                stop.handler(signal.SIGTERM, None)
                raise ValueError('disk failure')
        self.assertEqual(stop.depth, 0)

    def test_nested_commit_does_not_unwind_halfway(self):
        stop = StopControl(); committed = False
        with self.assertRaises(StopRequested):
            with stop.transaction():
                with stop.transaction():
                    stop.handler(signal.SIGTERM, None)
                committed = True
        self.assertTrue(committed)

    def test_json_publication_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'checkpoint.json'
            atomic_json(target, {'calls': 0, 'pending': 5})
            atomic_json(target, {'calls': 1, 'pending': 4})
            self.assertEqual(json.loads(target.read_text()), {'calls': 1, 'pending': 4})
            self.assertFalse(target.with_name('checkpoint.json.partial').exists())


if __name__ == '__main__':
    unittest.main()
