"""CPU-only returned-driver lease checks; execute on server, not on Windows."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('returned_lease_under_test', Path(__file__).with_name('returned_rlt_lease.py'))
returned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(returned)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ReturnedLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = returned._base()
        self.fallback = self.make_checkpoint(self.root / 'old', 100)
        self.current = self.root / 'returned'

    def make_checkpoint(self, run, step, *, complete=True):
        cp = run / run.name / 'checkpoints' / f'global_step_{step}'
        write(cp / 'actor/sac_components/rlt_trainer_state/complete.json',
              dict(complete=complete, saved_runner_step=step, actor_world_size=1, update_step=0))
        if not complete:
            return cp
        write(cp / 'actor/sac_components/replay_buffer/rank_0/metadata.json', dict(total_samples=3))
        write(cp / 'actor/sac_components/replay_buffer/rank_0/trajectory_index.json', dict(trajectory_id_list=[1, 2, 3]))
        write(cp / 'actor/dcp_checkpoint/.metadata', {'fixture': True})
        write(cp / 'actor/sac_components/target_model/checkpoint_rank_0.pt', {'fixture': True})
        return self.base._checkpoint(run)

    def test_no_new_complete_save_reuses_proven_checkpoint(self):
        self.make_checkpoint(self.current, 125, complete=False)
        cp = returned._select_checkpoint(self.base, self.current, self.fallback)
        self.assertEqual(cp, self.fallback)
        self.assertIsNot(cp, self.fallback)

    def test_new_complete_save_supersedes_fallback(self):
        new = self.make_checkpoint(self.current, 125)
        self.assertEqual(returned._select_checkpoint(self.base, self.current, self.fallback), new)

    def test_corrupt_complete_save_never_falls_back(self):
        new = self.make_checkpoint(self.current, 125)
        write(Path(new['path']) / 'actor/sac_components/replay_buffer/rank_0/metadata.json', dict(total_samples=99))
        with self.assertRaises(AssertionError):
            returned._select_checkpoint(self.base, self.current, self.fallback)

    def test_checkpoint_fallback_is_bound_to_one_run(self):
        stage = self.root / 'lease'
        write(stage / 'lease.json', dict(lease_kind=returned.KIND, row={'run': str(self.current)},
                                       returned_from={'checkpoint': self.fallback, 'job_id': 'expected'}))
        with patch.object(returned, '_base', return_value=self.base):
            bound = returned._bound_base(stage)
        self.assertEqual(bound._checkpoint(self.current), self.fallback)
        with self.assertRaisesRegex(RuntimeError, 'No complete RLT checkpoint'):
            bound._checkpoint(self.root / 'another-run')

    def test_unauthorized_card_is_rejected_before_source_or_files(self):
        with patch.object(returned, '_base', side_effect=AssertionError('must not load source')):
            for gpu in (0, 6, 7):
                with self.subTest(gpu=gpu), self.assertRaises(AssertionError) as caught:
                    returned.prepare_from_return(self.root / 'previous', self.root / 'new',
                                                 gpu=gpu, operation_id='v2', new_run='/unused', new_namespace='unused')
                self.assertNotEqual(str(caught.exception), 'must not load source')
        self.assertFalse((self.root / 'new').exists())

    def release_fixture(self):
        done = self.root / 'original-finished.json'
        write(done, dict(time=10, exit_code=143, actors=[], gpu_remaining=[]))
        row = dict(run=str(self.current), namespace='returned-namespace')
        lease = dict(row=row, returned_from={'original_role_terminal': str(done)},
                     pins={str(done): self.base._sha(done)})
        stopped = dict(time=100, actors=[dict(job_id='return-job', ray_namespace=row['namespace'])])
        return lease, stopped, done

    def test_old_owner_terminal_is_not_a_new_driver_ack(self):
        lease, stopped, done = self.release_fixture()
        before = done.read_bytes()
        self.assertIsNone(returned._release_evidence(self.base, lease, stopped))
        cleanup = self.current / 'runtime/cleanup-targets.json'
        write(cleanup, dict(time=101, job_id='return-job', actors=stopped['actors']))
        evidence = returned._release_evidence(self.base, lease, stopped)
        self.assertEqual(evidence['kind'], 'returned_driver_cleanup_and_original_role_terminal')
        self.assertEqual(done.read_bytes(), before)
        self.assertFalse((self.current / 'runtime/finished.json').exists())

    def test_cleanup_rejects_wrong_job_and_stale_receipt(self):
        lease, stopped, _ = self.release_fixture()
        cleanup = self.current / 'runtime/cleanup-targets.json'
        write(cleanup, dict(time=99, job_id='return-job', actors=stopped['actors']))
        self.assertIsNone(returned._release_evidence(self.base, lease, stopped))
        write(cleanup, dict(time=101, job_id='another-job', actors=[]))
        with self.assertRaisesRegex(AssertionError, 'Cleanup job'):
            returned._release_evidence(self.base, lease, stopped)
        write(cleanup, dict(time=101, job_id='return-job',
                           actors=[dict(job_id='return-job', ray_namespace='another-namespace')]))
        with self.assertRaisesRegex(AssertionError, 'another namespace'):
            returned._release_evidence(self.base, lease, stopped)


if __name__ == '__main__':
    unittest.main()
