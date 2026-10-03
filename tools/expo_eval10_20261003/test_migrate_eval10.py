"""CPU-only contract tests; run on the server with the staged migration module."""
import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from migrate_eval10 import check_input_diff, DRIVER


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.old = {'evaluation': {'every_episodes': 25, 'episodes': 20},
                    'formal': {'num_envs': 1, 'max_physical_actions': 20000},
                    'port_source_manifest': {DRIVER: 'old', 'protected.py': 'fixed'}}
        self.new = copy.deepcopy(self.old)
        self.new['evaluation']['every_episodes'] = 10
        self.new['port_source_manifest'].update({DRIVER: 'new'})

    def test_only_authorized_change_allowed(self):
        check_input_diff(self.old, self.new)

    def test_eval_sample_change_rejected(self):
        self.new['evaluation']['episodes'] = 10
        with self.assertRaises(ValueError): check_input_diff(self.old, self.new)

    def test_training_budget_change_rejected(self):
        self.new['formal']['max_physical_actions'] = 40000
        with self.assertRaises(ValueError): check_input_diff(self.old, self.new)

    def test_collector_parallelism_change_rejected(self):
        self.new['formal']['num_envs'] = 4
        with self.assertRaises(ValueError): check_input_diff(self.old, self.new)

    def test_protected_source_change_rejected(self):
        self.new['port_source_manifest']['protected.py'] = 'changed'
        with self.assertRaises(ValueError): check_input_diff(self.old, self.new)

    def test_unreviewed_source_addition_rejected(self):
        self.new['port_source_manifest']['extra.py'] = 'new'
        with self.assertRaises(ValueError): check_input_diff(self.old, self.new)

    def test_evaluation_schedule_preserves_history_and_final(self):
        path = Path(__file__).parent / 'train_expo_formal.py'
        tree = ast.parse(path.read_text())
        node, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'periodic_evaluation_due']
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        due = namespace['periodic_evaluation_due']
        progress = {'evaluation_periodic': [25, 50, 75]}
        for episode, remaining, expected in [(76, 100, False), (80, 100, True), (90, 100, True),
                                             (75, 100, False), (80, 0, False), (0, 100, False)]:
            self.assertEqual(due(SimpleNamespace(episodes_completed=episode, remaining_actions=remaining),
                                 {'every_episodes': 10}, progress), expected)
        self.assertFalse(due(SimpleNamespace(episodes_completed=80, remaining_actions=100),
                             {'every_episodes': 10}, {'evaluation_periodic': [25, 50, 75, 80]}))


if __name__ == '__main__':
    unittest.main()
