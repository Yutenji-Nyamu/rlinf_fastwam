"""CPU-only refusal and continuation checks. Run on the server; no Ray/CUDA."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import queue_continuation as q


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.owner = {'pid': 123, 'uid': 20001, 'start': 456}
        self.cycle = {'cycle_id': q.CYCLE_ID, 'runs': {}}
        self.previous = {'task': 'place_object_stand', 'uid': 20001, 'head': 'same',
                         'runs': {'stage1-full': {'gpus': [2]}, 'clean': {'gpus': [4]}, 'combo': {'gpus': [5]}},
                         'gates': {}, 'old_runs': []}
        self.plan = copy.deepcopy(self.previous)
        for role, gpu in [('clean', 4), ('combo', 5)]:
            row = {'new_run': '/return/' + role, 'namespace': 'return-' + role}
            self.cycle['runs']['gpu' + str(gpu)] = row
            self.plan['old_runs'].append({'run': row['new_run'], 'namespace': row['namespace'], 'gpus': [gpu], 'identity': None})
            self.plan['gates'][role] = {'kind': 'global', 'gpu': gpu, 'owner': self.owner,
                'cycle_plan': str(q.CTRL / q.CYCLE_ID / 'plan.json'), 'cycle_sha256': 'sha',
                'final': str(q.CTRL / 'final.json'), 'receipts': [str(q.CTRL / 'rlt-status.json')]}

    def validate(self):
        q.validate_plan_transition(self.previous, self.plan, self.cycle, 'sha', self.owner)

    def test_only_rebind_is_accepted(self):
        self.validate()

    def test_source_head_change_rejected(self):
        self.plan['head'] = 'changed'
        with self.assertRaisesRegex(AssertionError, 'Only gates'):
            self.validate()

    def test_formal_configuration_change_rejected(self):
        self.plan['runs']['clean']['config'] = '/new-config'
        with self.assertRaisesRegex(AssertionError, 'Only gates'):
            self.validate()

    def test_other_cycle_rejected(self):
        self.cycle['cycle_id'] = 'previous-cycle'
        with self.assertRaisesRegex(AssertionError, 'Wrong continuation cycle'):
            self.validate()

    def test_wrong_return_run_rejected(self):
        self.plan['old_runs'][0]['run'] = '/unrelated'
        with self.assertRaisesRegex(AssertionError, 'Wrong returned run'):
            self.validate()

    def test_stale_prebound_return_identity_rejected(self):
        self.plan['old_runs'][0]['identity'] = self.owner
        with self.assertRaisesRegex(AssertionError, 'must be read'):
            self.validate()

    def test_mismatched_borrow_owner_rejected(self):
        self.plan['gates']['combo']['owner'] = dict(self.owner, start=457)
        with self.assertRaisesRegex(AssertionError, 'Borrow owner differs'):
            self.validate()


class StageReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name) / 'stage'
        self.stage.mkdir()
        self.old = {'pid': 9, 'uid': os.getuid(), 'start': 8, 'boot_id': 'boot'}
        self.plan = {'task': 'place_object_stand', 'uid': os.getuid(), 'boot_id': 'boot', 'runs': {}}
        for role in ('stage1-full', 'clean', 'combo'):
            run = Path(self.temp.name) / role
            (run / 'runtime').mkdir(parents=True)
            self.plan['runs'][role] = {'run': str(run)}
        weights = Path(self.temp.name) / 'full_weights.pt'
        weights.write_bytes(b'complete-fixture')
        self.plan['stage1_full_weights'] = str(weights)
        self.save(self.stage / 'owner-identity.json', self.old)
        self.save(self.stage / 'queue-status.json', {'task': 'place_object_stand', 'stage1': 'COMPLETE',
                 'roles': {'clean': 'WAITING_BORROW_RETURN', 'combo': 'WAITING_BORROW_RETURN'}})
        self.save(self.stage / 'stage1-complete.json', {'weights': str(weights), 'bytes': weights.stat().st_size})
        runtime = Path(self.plan['runs']['stage1-full']['run']) / 'runtime'
        self.save(runtime / 'finished.json', {'exit_code': 0})
        (runtime / 'exit_code.txt').write_text('0')
        self.save(runtime / 'driver-identity.json', {'pid': 77, 'uid': os.getuid(), 'start': 66})
        self.ops = SimpleNamespace(same=Mock(return_value=False))

    def save(self, path, value):
        path.write_text(json.dumps(value))

    def validate(self):
        with patch.object(q, 'MINIMUM_STAGE1_BYTES', 1):
            return q.validate_completed_stage(self.stage, self.plan, self.old, self.ops)

    def test_complete_stage_is_reused_without_weight_rewrite(self):
        before = (self.stage / 'owner-identity.json').read_bytes()
        proof = self.validate()
        self.assertEqual(proof['bytes'], len(b'complete-fixture'))
        self.assertEqual((self.stage / 'owner-identity.json').read_bytes(), before)

    def test_live_old_owner_blocks_registration(self):
        self.ops.same.return_value = True
        with self.assertRaisesRegex(AssertionError, 'Original queue owner is still alive'):
            self.validate()

    def test_any_formal_cutover_receipt_blocks_replay(self):
        (self.stage / 'clean-cutover-bound.json').write_text('{}')
        with self.assertRaisesRegex(AssertionError, 'side effects'):
            self.validate()

    def test_formal_launch_blocks_replay(self):
        (Path(self.plan['runs']['combo']['run']) / 'runtime/launch.json').write_text('{}')
        with self.assertRaisesRegex(AssertionError, 'Formal driver already touched'):
            self.validate()

    def test_weight_size_change_blocks_reuse(self):
        Path(self.plan['stage1_full_weights']).write_bytes(b'changed')
        with self.assertRaisesRegex(AssertionError, 'size changed'):
            self.validate()

    def test_failed_stage1_cannot_be_called_complete(self):
        runtime = Path(self.plan['runs']['stage1-full']['run']) / 'runtime'
        self.save(runtime / 'finished.json', {'exit_code': 1})
        with self.assertRaisesRegex(AssertionError, 'Stage1 did not exit'):
            self.validate()

    def test_continuation_identity_prevents_second_owner(self):
        (self.stage / 'owner-identity-continuation.json').write_text('{}')
        with self.assertRaisesRegex(AssertionError, 'Never replay'):
            self.validate()


class LoopTests(unittest.TestCase):
    def test_waiting_gate_cannot_launch_stage1_or_cutover(self):
        state = {'stage1': 'COMPLETE', 'roles': {'clean': 'WAITING_BORROW_RETURN', 'combo': 'WAITING_BORROW_RETURN'}}
        ops = SimpleNamespace(launch=Mock(side_effect=AssertionError('must not launch')),
                              stop_old=Mock(side_effect=AssertionError('must not stop')))
        original = SimpleNamespace(gate=Mock(return_value=None))
        q.role_iteration({}, state, {}, set(), {}, ops, original)
        ops.launch.assert_not_called()
        ops.stop_old.assert_not_called()
        self.assertEqual(state['stage1'], 'COMPLETE')

    def test_foreign_gate_error_never_stops_target(self):
        state = {'stage1': 'COMPLETE', 'roles': {'clean': 'WAITING_BORROW_RETURN', 'combo': 'WAITING_BORROW_RETURN'}}
        ops = SimpleNamespace(ST=Path('/tmp/unused'), now=lambda: 'now', save=Mock(), stop_old=Mock(), launch=Mock())
        original = SimpleNamespace(gate=Mock(side_effect=AssertionError('foreign cycle')))
        failures = {}
        q.role_iteration({}, state, {}, set(), failures, ops, original)
        ops.stop_old.assert_not_called()
        ops.launch.assert_not_called()
        self.assertEqual(set(failures), {'clean', 'combo'})
        self.assertEqual(state['roles'], {'clean': 'NEEDS_ATTENTION', 'combo': 'NEEDS_ATTENTION'})


if __name__ == '__main__':
    unittest.main()
