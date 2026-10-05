"""Pure owner-contract checks; run on the server before any transition."""
import ast
import copy
import json
import io
from pathlib import Path
import tempfile
import unittest

import rynn_formal_owner as owner
import rynn_handoff as handoff


def service(key, kind, gpu, port):
    return dict(key=key, kind=kind, physical_gpu=gpu, url=f'http://127.0.0.1:{port}')


class OwnerContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plan = dict(owner_dir=str(self.root), services=[
            service('wm6', 'wm', 6, 18946), service('wm7', 'wm', 7, 18947),
            service('rm4', 'rynn_success', 4, 18954), service('rm5', 'rynn_success', 5, 18955)],
            rm_gate=dict(output=str(self.root / 'rm_gate/result.json')))

    def tearDown(self):
        self.tmp.cleanup()

    def test_typed_services_have_exact_four_cards(self):
        owner.validate_service_layout(self.plan)
        bad = copy.deepcopy(self.plan)
        bad['services'][-1]['physical_gpu'] = 0
        with self.assertRaises(AssertionError):
            owner.validate_service_layout(bad)
        bad = copy.deepcopy(self.plan)
        bad['services'][-1]['url'] = bad['services'][0]['url']
        with self.assertRaises(AssertionError):
            owner.validate_service_layout(bad)

    def test_gate_rejects_unknown_batch_or_failure_and_freezes_result(self):
        output = Path(self.plan['rm_gate']['output'])
        output.parent.mkdir()
        for value in (dict(passed=False, selected_rm_batch=8), dict(passed=True, selected_rm_batch=32),
                      dict(passed=True, selected_rm_batch=True)):
            output.write_text(json.dumps(value))
            with self.assertRaises(AssertionError):
                owner.validate_gate_result(self.plan)
        output.write_text(json.dumps(dict(passed=True, selected_rm_batch=8)))
        receipt = dict(path=str(output), sha256=owner.sha(output), selected_rm_batch=8)
        (output.parent / 'frozen-result.json').write_text(json.dumps(receipt))
        owner.verify_saved_gate(self.plan)
        output.write_text(json.dumps(dict(passed=True, selected_rm_batch=16)))
        with self.assertRaises(AssertionError):
            owner.verify_saved_gate(self.plan)

    def test_reward_route_and_semantics(self):
        train = dict(reward_source='rynn_success', rynn_invalid_reward_sentinel=-1.0,
            rynn_run_id='run/smoke', rynn_batch_size_file=self.plan['rm_gate']['output'],
            rynn_service_urls=[s['url'] for s in self.plan['services'][2:]],
            service_urls=[s['url'] for s in self.plan['services'][:2]],
            auto_reset=False, ignore_terminations=False, use_rel_reward=False,
            reward_coef=1.0, success_reward_threshold=.9)
        cfg = dict(env=dict(train=train), runner=dict(resume_dir='global_step_70'))
        owner.validate_reward_config(cfg, self.plan)
        shadow = owner.legacy_shadow(cfg)
        self.assertTrue(shadow['env']['train']['use_rel_reward'])
        self.assertNotIn('rynn_run_id', shadow['env']['train'])
        self.assertEqual(cfg['runner']['resume_dir'], 'global_step_70')
        train['rynn_service_urls'].reverse()
        with self.assertRaises(AssertionError):
            owner.validate_reward_config(cfg, self.plan)

    def test_owner_extension_keeps_catalog_launch_and_adoption(self):
        source = """def owner_main(input_plan):
    assert not (cycle/'rlt-stopped.json').exists(), 'Owner must witness its own borrowing'
    borrowed = False
    try:
        for row in plan['trials']:
            env = dict(base_env)
    finally:
        cleanup(plan, catalog)
"""
        changed = owner.owner_source_with_gate(source, True)
        ast.parse(changed)
        self.assertIn('borrowed = adopted_borrow', changed)
        self.assertEqual(changed.count('run_reward_gate('), 1)
        self.assertLess(changed.index('run_reward_gate('), changed.index("for row in plan['trials']"))
        self.assertIn('cleanup(plan, catalog)', changed)
        with self.assertRaises(AssertionError):
            owner.owner_source_with_gate(changed, True)

    def test_handoff_extension_only_changes_known_source_fragments(self):
        source = """def main():
    assert Path(__file__).resolve() == (CODE / 'batch16_handoff.py').resolve()
    state = D / 'prepared'
    support = OLD_WRAPPER.parent
    wrapper = CODE / 'batch16_formal_owner.py'
    launch = CODE / 'batch16_formal_owner.py'
    signal_exact(old_identity)
"""
        changed = handoff.adapted_main_source(source)
        ast.parse(changed)
        self.assertIn("D / 'handoff'", changed)
        self.assertIn('signal_exact(old_identity)', changed)
        self.assertNotIn('batch16_formal_owner.py', changed)
        with self.assertRaises(AssertionError):
            handoff.adapted_main_source(changed)

    def test_rynn_control_posts_empty_body_and_wm_keeps_original(self):
        called = []
        def original(url, **kwargs):
            called.append((url, kwargs))
            return {'original': True}
        def opener(request, timeout):
            self.assertEqual(request.data, b'')
            self.assertEqual(request.full_url, 'http://127.0.0.1:18954/offload')
            return io.BytesIO(b'{"ok":true,"is_offloaded":true}')
        http = owner.service_http(original, ['http://127.0.0.1:18954'], opener)
        self.assertTrue(http('http://127.0.0.1:18954', '/offload', post=True)['is_offloaded'])
        self.assertEqual(called, [])
        self.assertEqual(http('http://127.0.0.1:18946', '/offload', post=True), {'original': True})
        self.assertEqual(len(called), 1)

    def test_learning_gate_rejects_zero_gradient_or_zero_groups(self):
        rows = [dict(tag='train/actor/grad_norm', step=0, value=.04),
                dict(tag='train/actor/rynn_effective_group_fraction', step=0, value=.25)]
        self.assertTrue(owner.validate_learning_records(rows)['passed'])
        for index, value in ((0, 0), (0, float('nan')), (1, 0), (1, -1)):
            bad = copy.deepcopy(rows)
            bad[index]['value'] = value
            with self.assertRaises(AssertionError):
                owner.validate_learning_records(bad)


if __name__ == '__main__':
    unittest.main()
