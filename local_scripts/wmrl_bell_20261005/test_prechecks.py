"""Server CPU tests: finite prechecks and exact cleanup coverage, no GPUs/Ray."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

SPEC = importlib.util.spec_from_file_location('finite_prechecks_tested', Path(__file__).with_name('prechecks.py'))
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def reward_report(tp=5, fn=1, fp=1, tn=20):
    return dict(strict_load=True, rows=[{'native_label': 1, 'score': .95}], thresholds={'0.9': dict(
        true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn)})


class PrecheckTests(unittest.TestCase):
    def test_bell_labels_distinguished_at_fixed_threshold(self):
        self.assertTrue(P.bell_decision(reward_report())['passed'])

    def test_constant_yes_no_and_reversed_rejected(self):
        for counts in ((5, 0, 20, 0), (0, 5, 0, 20), (0, 5, 20, 0)):
            self.assertFalse(P.bell_decision(reward_report(*counts))['passed'])

    def test_missing_native_class_not_reported_as_accuracy(self):
        result = P.bell_decision(reward_report(0, 0, 0, 32))
        self.assertFalse(result['passed'])
        self.assertIn('lacks', result['reason'])

    def test_native_namespaces_added_without_mutating_formal_trials(self):
        original = [{'key': 'formal', 'namespace': 'opendw_formal'}]
        rows = [{'key': 'native_rynn32', 'kind': 'native', 'namespace': 'opendw_native'},
                {'key': 'rynn32', 'kind': 'rynn_binary'}]
        plan = dict(trials=original, prechecks=rows)
        result = P.scoped_plan(plan)
        self.assertEqual([r['key'] for r in result['trials']], ['formal', 'native_rynn32'])
        self.assertEqual(plan['trials'], original)
        self.assertEqual(len(original), 1)

    def test_hook_inserts_once_and_cleanup_sees_native_namespace(self):
        observations = []
        module = types.ModuleType('fake_original_owner')
        module.cleanup = lambda plan, catalog, phase=None: observations.append(plan['trials'])
        module.add_allowlist = lambda plan: observations.append(plan['trials'])
        module.validate = lambda plan, frozen=False: None
        module.record = lambda path, value: None
        source = ("def owner_main(input_plan):\n"
                  "    plan = input_plan\n"
                  "    try:\n"
                  "        borrowed = True\n"
                  "        for row in plan['trials']:\n"
                  "            env = dict(base_env)\n"
                  "    finally:\n"
                  "        unchanged_finally()\n")
        plan = dict(trials=[{'key': 'formal', 'namespace': 'opendw_formal'}],
                    prechecks=[{'key': 'native_rynn32', 'kind': 'native', 'namespace': 'opendw_native'}])
        P.install(module, plan, source)
        module.cleanup(plan, None)
        module.add_allowlist(plan)
        self.assertEqual([r['namespace'] for r in observations[0]], ['opendw_formal', 'opendw_native'])
        self.assertEqual(observations[0], observations[1])
        self.assertIn('run_finite_prechecks', module.owner_main.__code__.co_names)
        self.assertIn('unchanged_finally', module.owner_main.__code__.co_names)
        with self.assertRaises(AssertionError):
            P.install(module, plan, source)

    def test_failed_child_is_cleaned_before_error_propagates_to_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            owner = Path(temp)
            environment = owner / 'environment.json'
            environment.write_text('{}')
            cleaned = []
            def save(path, value):
                Path(path).write_text(json.dumps(value))
            M = types.SimpleNamespace(read=lambda path: json.loads(Path(path).read_text()),
                MASKS=['CUDA_VISIBLE_DEVICES'], cleanup=lambda plan, catalog, phase: cleaned.append(phase) or {'all_stopped': True},
                record=save)
            row = dict(key='rynn32', kind='rynn_binary', timeout_seconds=60, commands=[
                dict(argv=['python', '-u', '-B', 'fixture.py'], cwd=temp, mode='gpu', physical_gpu=4)])
            plan = dict(owner_dir=temp, environment_file=str(environment), prechecks=[row])
            child = types.SimpleNamespace(poll=lambda: 7, returncode=7)
            with self.assertRaisesRegex(RuntimeError, 'precheck process failed'):
                P.run_prechecks(plan, None, lambda *a: child, M)
            self.assertEqual(cleaned, ['rynn32'])
            self.assertEqual(json.loads((owner / 'rynn32/command-0.json').read_text())['exit_code'], 7)


if __name__ == '__main__':
    unittest.main()
