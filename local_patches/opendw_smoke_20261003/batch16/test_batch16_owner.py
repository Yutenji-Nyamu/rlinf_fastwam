"""CPU-only acceptance boundaries; no imports of Ray, torch or GPU libraries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import batch16_formal_owner as owner


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def file_row(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'checkpoint fixture ' + path.name.encode())
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


class Batch16OwnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old = self.root / 'formal-v2'
        self.old.mkdir()
        self.patches = [patch.object(owner, 'ROOT', self.root), patch.object(owner, 'SOURCE_OWNER', self.old)]
        for item in self.patches:
            item.start()
        write(self.old / 'owner-plan.json', {'owner': 'frozen test'})
        checkpoint = self.old / 'saved/global_step_2'
        ranks = [dict(rank=i, complete=True, files=[file_row(checkpoint / ('actor/local_shard_checkpoint/checkpoint_rank_' + str(i) + '.pt'))])
                 for i in range(2)]
        self.receipt = dict(schema=1, kind='opendw-formal-resume-checkpoint', source_owner=str(self.old),
            source_owner_plan_sha256=owner.sha(self.old / 'owner-plan.json'), checkpoint_path=str(checkpoint),
            completed_step=2, complete=True, actor_world_size=2,
            runner_state=dict(global_step=2, optimizer_state_saved=True, scheduler_state_saved=True), ranks=ranks,
            full_weights=file_row(checkpoint / 'actor/model_state_dict/full_weights.pt'))
        self.receipt_path = self.root / 'resume.json'
        write(self.receipt_path, self.receipt)
        self.new = self.root / 'formal-b16'
        write(self.new / 'startup_smoke/driver-finished.json', {'exit_code': 0})
        write(self.new / 'startup_smoke/verified-placement.json', {})
        self.event = dict(event='batch_completed', configured_wm_batch=16, actual_wm_batch=16,
            execution_mode='batched', outputs_finite=True, rows_completed=16,
            kernel_proof=dict(kernel='infer_joint_batch', batch_size=16, denoiser_batch_sizes=[16] * 10,
                decoded_video_shape=[16, 3, 9, 224, 224], action_shape=[16, 32, 14]))
        services = []
        for gpu in (6, 7):
            directory = self.new / 'services' / ('wm' + str(gpu)) / 'records'
            write(directory / 'service-events.jsonl', self.event)
            services.append(dict(key='wm' + str(gpu), physical_gpu=gpu, argv=['service.py', '--output-dir', str(directory)]))
        self.plan = dict(owner_dir=str(self.new), services=services,
            resume_checkpoint=dict(receipt=str(self.receipt_path), sha256=owner.sha(self.receipt_path)))

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_complete_resume_contract(self):
        self.assertEqual(owner.verify_resume(self.receipt_path)['completed_step'], 2)

    def test_partial_rank_or_missing_optimizer_rejected(self):
        for field in ('rank', 'optimizer'):
            value = copy.deepcopy(self.receipt)
            if field == 'rank':
                value['ranks'][1]['complete'] = False
            else:
                value['runner_state']['optimizer_state_saved'] = False
            write(self.receipt_path, value)
            with self.assertRaises(AssertionError):
                owner.verify_resume(self.receipt_path)

    def test_checkpoint_changed_after_receipt_rejected(self):
        Path(self.receipt['ranks'][0]['files'][0]['path']).write_bytes(b'changed')
        with self.assertRaises(AssertionError):
            owner.verify_resume(self.receipt_path)

    def test_checkpoint_step_mismatch_rejected(self):
        self.receipt['completed_step'] = 3
        write(self.receipt_path, self.receipt)
        with self.assertRaises(AssertionError):
            owner.verify_resume(self.receipt_path)

    def test_true_b16_on_both_services_passes(self):
        gate = owner.smoke_batch_gate(self.plan)
        self.assertEqual([row['physical_gpu'] for row in gate['services']], [6, 7])
        self.assertFalse(gate['startup_checkpoint_used_for_formal'])

    def test_http_batch_without_actual_kernel_batch_rejected(self):
        event = copy.deepcopy(self.event)
        event['actual_wm_batch'] = 1
        write(self.new / 'services/wm7/records/service-events.jsonl', event)
        with self.assertRaises(AssertionError):
            owner.smoke_batch_gate(self.plan)

    def test_hidden_denoiser_downgrade_or_nonfinite_output_rejected(self):
        for field in ('denoiser', 'finite'):
            event = copy.deepcopy(self.event)
            if field == 'denoiser':
                event['kernel_proof']['denoiser_batch_sizes'][-1] = 1
            else:
                event['outputs_finite'] = False
            write(self.new / 'services/wm7/records/service-events.jsonl', event)
            with self.assertRaises(AssertionError):
                owner.smoke_batch_gate(self.plan)

    def test_gate_prefix_remains_valid_after_formal_appends_logs(self):
        gate = owner.smoke_batch_gate(self.plan)
        write(self.new / 'batch16-smoke-gate.json', gate)
        for row in gate['services']:
            with Path(row['path']).open('a') as stream:
                stream.write('\n' + json.dumps(self.event))
        self.assertEqual(owner.verify_saved_gate(self.plan)['status'], 'passed')
        Path(gate['services'][0]['path']).write_bytes(b'overwritten')
        with self.assertRaises(AssertionError):
            owner.verify_saved_gate(self.plan)


if __name__ == '__main__':
    unittest.main()
