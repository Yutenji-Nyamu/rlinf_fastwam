"""Focused CPU checks for the budget boundary and forbidden recipe changes."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from migrate import extend_metadata, check_inputs
from common import CHANGED_SOURCE

path = Path(__file__).resolve().parents[2] / 'rlinf/algorithms/expo_ft/formal_cadence.py'
spec = importlib.util.spec_from_file_location('cadence_test', path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
Cadence = module.FormalCadence

class ExtensionTests(unittest.TestCase):
    def state(self):
        cadence = Cadence(20000, episodes_completed=128, physical_actions=20000, warmup_actions=1827,
                         post_warmup_actions=18173, carry_actions=13, completed_calls=454, budget_truncated_episodes=1)
        return {'cadence': cadence.state_dict(), 'progress': {'evaluation_final': True, 'evaluation_initial': True, 'online_success': 56}}

    def test_old_20k_format_unchanged(self):
        state = self.state()['cadence']
        self.assertNotIn('prior_budget_truncations', state['contract'])
        Cadence(20000).load_state_dict(state)

    def test_continue_without_new_warmup_or_lost_carry(self):
        saved = self.state(); old = copy.deepcopy(saved['cadence']['counters']); extend_metadata(saved)
        cadence = Cadence(60000, prior_budget_truncations=1); cadence.load_state_dict(saved['cadence'])
        self.assertFalse(cadence.in_warmup); self.assertEqual(saved['cadence']['counters'], old)
        self.assertEqual(cadence.finish_episode(27), 1); cadence.complete_call()
        self.assertEqual(cadence.completed_calls, 455); self.assertEqual(cadence.carry_actions, 0)

    def test_final_second_budget_cut_and_no_overrun(self):
        saved = self.state(); extend_metadata(saved)
        cadence = Cadence(60000, prior_budget_truncations=1); cadence.load_state_dict(saved['cadence'])
        while cadence.remaining_actions:
            amount = min(137, cadence.remaining_actions)
            cadence.finish_episode(amount, budget_truncated=amount == cadence.remaining_actions)
            while cadence.pending_calls: cadence.complete_call()
        self.assertEqual(cadence.physical_actions, 60000); self.assertEqual(cadence.completed_calls, 1454)
        self.assertEqual(cadence.budget_truncated_episodes, 2); self.assertEqual(cadence.carry_actions, 13)
        with self.assertRaises(ValueError): cadence.finish_episode(1)

    def test_wrong_original_boundary_rejected(self):
        saved = self.state(); saved['cadence']['counters']['physical_actions'] -= 1
        with self.assertRaises(AssertionError): extend_metadata(saved)

    def test_only_authorized_recipe_difference(self):
        old = {'formal': {'max_physical_actions': 20000, 'batch_size': 64}, 'seed': 42,
               'port_source_manifest': {k: 'old' for k in CHANGED_SOURCE} | {'protected': 'same'}}
        new = copy.deepcopy(old); new['formal']['max_physical_actions'] = 60000
        for k in CHANGED_SOURCE: new['port_source_manifest'][k] = 'new'
        check_inputs(old, new)
        new['formal']['batch_size'] = 128
        with self.assertRaises(AssertionError): check_inputs(old, new)

    def test_failure_and_success_return_contract(self):
        import ast
        tree = ast.parse((Path(__file__).parent / 'owner.py').read_text())
        node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'release_kind']
        ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), 'owner', 'exec'), ns)
        self.assertEqual(ns['release_kind'](False, True), 'failed')
        self.assertEqual(ns['release_kind'](True, True), 'completed')
        self.assertEqual(ns['release_kind'](False, False), 'not_started')

if __name__ == '__main__': unittest.main()
