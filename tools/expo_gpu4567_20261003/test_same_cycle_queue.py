"""CPU-only test of the unchanged deployed queue gate for same-cycle adoption.

Run on SZ2 with --queue-source <path from the existing queue plan>. No processes
are started or signalled. Temporary files model only gate receipts and routes.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


QUEUE = None


class SameCycleQueueGate(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='expo-queue-gate-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        self.borrow_owner = dict(pid=111, uid=20001, start=222)
        self.replacement_owner = dict(pid=333, uid=20001, start=444)
        self.returned_driver = dict(pid=555, uid=20001, start=666, namespace='same-return-ns')
        self.owner_live = False
        self.queue = QUEUE
        self.old_ops = self.queue.ops
        self.old_watch = self.queue.WATCH
        self.addCleanup(setattr, self.queue, 'ops', self.old_ops)
        self.addCleanup(setattr, self.queue, 'WATCH', self.old_watch)
        self.queue.ops = types.SimpleNamespace(
            read=lambda path: json.loads(Path(path).read_text()),
            same=lambda row: self.owner_live and row == self.borrow_owner,
            normalized=lambda row: dict(row),
        )
        self.queue.WATCH = self.root / 'watch.json'
        self.cycle_path = self.root / 'same-cycle' / 'plan.json'
        self.returned_run = self.root / 'original-returned-run'
        self.target = dict(run=str(self.returned_run), namespace='same-return-ns', gpus=[4], identity=None)
        self.cycle = dict(cycle_id='same-cycle', runs={'gpu4': dict(new_run=str(self.returned_run), namespace='same-return-ns')})
        self.write(self.cycle_path, self.cycle)
        self.final = self.root / 'final.json'
        self.status = self.root / 'rlt-status.json'
        self.plan = dict(
            boot_id=self.boot, uid=20001,
            gates={'clean': dict(kind='global', final=str(self.final), owner=self.borrow_owner,
                                 receipts=[str(self.status)], cycle_plan=str(self.cycle_path),
                                 cycle_sha256=hashlib.sha256(self.cycle_path.read_bytes()).hexdigest(),
                                 gpu=4, anchors={})},
            old_runs=[self.target],
        )
        self.receipt = dict(cycle_id='same-cycle', all_first_rounds_verified=True,
                            runs={'gpu4': dict(first_round_verified=True, resume_identity=self.returned_driver)})
        self.write(self.returned_run / 'runtime' / 'driver-identity.json', self.returned_driver)
        self.write(self.queue.WATCH, dict(runs={'gpu4': self.target}))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def gate(self):
        return self.queue.gate(self.plan, 'clean')

    def successful_return(self):
        # A replacement owner publishes the original cycle's terminal proof.
        self.write(self.final, dict(owner=self.replacement_owner, gpu_released=True,
                                    rlt_first_rounds_verified=True))
        self.write(self.status, self.receipt)

    def test_retired_original_owner_without_final_keeps_queue_waiting(self):
        self.assertIsNone(self.gate())

    def test_final_without_rlt_first_round_proof_keeps_queue_waiting(self):
        self.write(self.final, dict(owner=self.replacement_owner))
        self.assertIsNone(self.gate())
        self.write(self.status, dict(all_first_rounds_verified=False))
        self.assertIsNone(self.gate())

    def test_same_cycle_replacement_owner_return_opens_unchanged_gate(self):
        self.successful_return()
        target = self.gate()
        self.assertEqual(target['identity'], self.returned_driver)
        self.assertEqual(target['run'], self.target['run'])

    def test_live_original_owner_blocks_even_with_terminal_receipts(self):
        self.successful_return()
        self.owner_live = True
        self.assertIsNone(self.gate())

    def test_foreign_cycle_receipt_is_rejected(self):
        self.successful_return()
        altered = copy.deepcopy(self.receipt)
        altered['cycle_id'] = 'different-cycle'
        self.write(self.status, altered)
        with self.assertRaisesRegex(AssertionError, 'another cycle'):
            self.gate()

    def test_reused_returned_pid_is_rejected(self):
        self.successful_return()
        self.write(self.returned_run / 'runtime' / 'driver-identity.json',
                   dict(self.returned_driver, start=667))
        with self.assertRaisesRegex(AssertionError, 'identity changed'):
            self.gate()

    def test_changed_return_namespace_is_rejected(self):
        self.successful_return()
        self.write(self.returned_run / 'runtime' / 'driver-identity.json',
                   dict(self.returned_driver, namespace='different-namespace'))
        with self.assertRaises(AssertionError):
            self.gate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-source', type=Path, required=True)
    args, rest = parser.parse_known_args()
    path = args.queue_source.resolve(strict=True)
    placeholder = types.ModuleType('ops')
    previous_ops = sys.modules.get('ops')
    sys.modules['ops'] = placeholder
    spec = importlib.util.spec_from_file_location('_same_cycle_original_queue', path)
    QUEUE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(QUEUE)
    if previous_ops is None:
        del sys.modules['ops']
    else:
        sys.modules['ops'] = previous_ops
    print('Testing actual queue source SHA256:', hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)
