"""Server-only stdlib checks: python -B tests/test_expo_formal_cadence.py."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


path = Path(__file__).resolve().parents[1] / 'rlinf/algorithms/expo_ft/formal_cadence.py'
spec = importlib.util.spec_from_file_location('_expo_formal_cadence_cpu', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
FormalCadence = module.FormalCadence


class FormalCadenceChecks(unittest.TestCase):
    def warmed(self, extra=1000):
        cadence = FormalCadence(max_physical_actions=2000 + extra)
        for _ in range(10):
            self.assertEqual(cadence.finish_episode(200), 0)
        self.assertEqual(cadence.warmup_actions, 2000)
        self.assertEqual(cadence.post_warmup_actions, 0)
        self.assertEqual(cadence.completed_calls, 0)
        return cadence

    def test_warmup_never_creates_call_debt(self):
        cadence = self.warmed()
        self.assertFalse(cadence.in_warmup)
        self.assertEqual(cadence.finish_episode(200), 5)
        for _ in range(5):
            cadence.complete_call()
        self.assertEqual(cadence.completed_calls, 5)
        self.assertEqual(cadence.carry_actions, 0)

    def test_variable_episode_lengths_carry_real_remainder(self):
        cadence = self.warmed()
        self.assertEqual(cadence.finish_episode(57), 1)
        self.assertEqual(cadence.carry_actions, 17)
        cadence.complete_call()
        self.assertEqual(cadence.finish_episode(23), 1)
        self.assertEqual(cadence.carry_actions, 0)
        cadence.complete_call()
        self.assertEqual(cadence.completed_calls, 2)

    def test_budget_inside_warmup_has_no_fake_update(self):
        cadence = FormalCadence(max_physical_actions=137)
        self.assertEqual(cadence.finish_episode(137, budget_truncated=True), 0)
        self.assertEqual(cadence.remaining_actions, 0)
        self.assertEqual(cadence.budget_truncated_episodes, 1)
        with self.assertRaises(ValueError):
            cadence.complete_call()

    def test_final_budget_carry_is_not_rounded_up(self):
        cadence = self.warmed(extra=99)
        self.assertEqual(cadence.finish_episode(99, budget_truncated=True), 2)
        cadence.complete_call(); cadence.complete_call()
        self.assertEqual(cadence.remaining_actions, 0)
        self.assertEqual(cadence.carry_actions, 19)
        self.assertEqual(cadence.completed_calls, 2)
        with self.assertRaises(ValueError):
            cadence.complete_call()

    def test_boundary_restore_preserves_pending_calls(self):
        cadence = self.warmed()
        cadence.finish_episode(200)
        saved = copy.deepcopy(cadence.state_dict())
        cadence.complete_call(); cadence.complete_call()
        restored = FormalCadence(max_physical_actions=3000)
        restored.load_state_dict(saved)
        self.assertEqual(restored.pending_calls, 5)
        self.assertEqual(restored.completed_calls, 0)
        self.assertEqual(restored.state_dict(), saved)
        with self.assertRaises(ValueError):
            restored.finish_episode(1)
        for _ in range(5):
            restored.complete_call()
        self.assertEqual(restored.finish_episode(1), 0)

    def test_pending_below_minimum_allows_collection_then_drains(self):
        cadence = FormalCadence(max_physical_actions=20000)
        for _ in range(10):
            cadence.finish_episode(1)
        self.assertFalse(cadence.in_warmup)
        self.assertFalse(cadence.can_learn)
        self.assertEqual(cadence.finish_episode(40), 1)
        self.assertEqual(cadence.physical_actions, 50)
        with self.assertRaises(ValueError):
            cadence.complete_call()
        saved = copy.deepcopy(cadence.state_dict())
        restored = FormalCadence(max_physical_actions=20000)
        restored.load_state_dict(saved)
        self.assertEqual(restored.finish_episode(14), 1)
        self.assertTrue(restored.can_learn)
        self.assertEqual(restored.carry_actions, 14)
        with self.assertRaises(ValueError):
            restored.finish_episode(1)
        restored.complete_call()
        self.assertEqual(restored.completed_calls, 1)
        self.assertEqual(restored.finish_episode(26), 1)
        self.assertEqual(restored.carry_actions, 0)

    def test_pending_calls_accumulate_before_minimum(self):
        cadence = FormalCadence(max_physical_actions=20000, minimum_online_actions=128)
        for _ in range(10):
            cadence.finish_episode(1)
        self.assertEqual(cadence.finish_episode(40), 1)
        self.assertEqual(cadence.finish_episode(40), 2)
        self.assertEqual(cadence.finish_episode(38), 2)
        self.assertTrue(cadence.can_learn)
        self.assertEqual(cadence.carry_actions, 38)
        cadence.complete_call(); cadence.complete_call()

    def test_budget_below_minimum_keeps_pending_without_training(self):
        cadence = FormalCadence(max_physical_actions=60)
        for _ in range(10):
            cadence.finish_episode(1)
        self.assertEqual(cadence.finish_episode(50, budget_truncated=True), 1)
        self.assertEqual(cadence.remaining_actions, 0)
        self.assertFalse(cadence.can_learn)
        with self.assertRaises(ValueError):
            cadence.complete_call()
        restored = FormalCadence(max_physical_actions=60)
        restored.load_state_dict(copy.deepcopy(cadence.state_dict()))
        self.assertEqual(restored.pending_calls, 1)

    def test_minimum_and_schema_mismatch_refuse_restore(self):
        saved = self.warmed().state_dict()
        with self.assertRaises(ValueError):
            FormalCadence(max_physical_actions=3000, minimum_online_actions=65).load_state_dict(saved)
        legacy = copy.deepcopy(saved); legacy['version'] = 1
        with self.assertRaises(ValueError):
            FormalCadence(max_physical_actions=3000).load_state_dict(legacy)

    def test_mismatch_and_counter_corruption_refuse_restore(self):
        saved = self.warmed().state_dict()
        with self.assertRaises(ValueError):
            FormalCadence(max_physical_actions=3001).load_state_dict(saved)
        for name in ('physical_actions', 'carry_actions', 'pending_calls', 'episodes_completed'):
            corrupt = copy.deepcopy(saved)
            corrupt['counters'][name] += 1
            with self.subTest(counter=name), self.assertRaises(ValueError):
                FormalCadence(max_physical_actions=3000).load_state_dict(corrupt)

    def test_invalid_actions_budget_and_boundary_sequence(self):
        cadence = FormalCadence(max_physical_actions=20)
        for actions in (0, -1, True, 21, 201):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                cadence.finish_episode(actions)
        with self.assertRaises(ValueError):
            cadence.finish_episode(19, budget_truncated=True)
        self.assertEqual(cadence.state_dict()['counters']['physical_actions'], 0)


if __name__ == '__main__':
    unittest.main()
