"""Run these stdlib-only checks on the authorized server; no CUDA/workloads."""
import copy
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
m = load('reborrow_owner'); r = load('reborrow_resources')


class ReborrowChecks(unittest.TestCase):
    @contextmanager
    def checkpoint_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); previous = base / 'previous-cycle'; previous.mkdir()
            run_root = base / 'runs'; run_root.mkdir(); rows = {}; identities = {}
            for gpu in (4, 5, 6, 7):
                key = 'gpu' + str(gpu); run = run_root / ('run-' + key); runtime = run / 'runtime'
                runtime.mkdir(parents=True)
                row = dict(pid=100 + gpu, uid=20001, start_ticks=1000 + gpu, boot_id='a', state='S')
                identities[row['pid']] = row
                rows[key] = dict(new_run=str(run), namespace='namespace-' + key, gpus=[gpu])
                (runtime / 'driver-identity.json').write_text(json.dumps(dict(row, namespace='namespace-' + key)))
                (previous / (key + '-launched.json')).write_text(json.dumps(dict(
                    identity=row, run=str(run), namespace='namespace-' + key)))
            plan = dict(cycle_id=previous.name, host='h100-gpu02', runs=rows)
            (previous / 'plan.json').write_text(json.dumps(plan))
            (previous / 'resumed-dispatched.json').write_text(json.dumps(dict(
                cycle_id=previous.name, runs={key: row['new_run'] for key, row in rows.items()})))
            def complete(key, *, nested=False, value=True):
                run = Path(rows[key]['new_run']); root = run / run.name if nested else run
                state = root / 'checkpoints/global_step_25/actor/sac_components/rlt_trainer_state'
                state.mkdir(parents=True, exist_ok=True)
                (state / 'complete.json').write_text(json.dumps(dict(complete=value)))
                return state
            with patch.object(m, 'PREVIOUS', previous), patch.object(m, 'RLT_RUNS', run_root):
                yield rows, identities, complete, previous

    def test_checkpoint_wait_requires_fresh_complete_on_all_four(self):
        with self.checkpoint_fixture() as (rows, identities, complete, previous):
            clock = [0]; reports = []
            def sleep(seconds):
                self.assertEqual(seconds, 30); clock[0] += seconds
                for key in rows:
                    complete(key, nested=key == 'gpu5', value=(key != 'gpu7' or clock[0] >= 60))
            result = m.wait_rlt_checkpoints(identities.__getitem__, sleep=sleep,
                monotonic=lambda: clock[0], report=lambda value: reports.append(value))
            self.assertEqual(clock[0], 60)
            self.assertTrue(all(row['checkpoint_root'] is None for row in reports[0].values()))
            self.assertIsNone(reports[1]['gpu7']['checkpoint'])
            self.assertTrue(all(row['checkpoint'] for row in result.values()))
            self.assertEqual(set(previous.iterdir()), {previous / 'plan.json', previous / 'resumed-dispatched.json'}
                | {previous / (key + '-launched.json') for key in rows})

    def test_checkpoint_wait_rejects_ambiguous_or_redirected_roots(self):
        for failure in ('ambiguous', 'symlink'):
            with self.subTest(failure=failure), self.checkpoint_fixture() as (rows, identities, complete, previous):
                complete('gpu4')
                run = Path(rows['gpu4']['new_run'])
                if failure == 'ambiguous':
                    complete('gpu4', nested=True)
                else:
                    (run / run.name).symlink_to(Path(rows['gpu5']['new_run']), target_is_directory=True)
                with self.assertRaises(ValueError):
                    m.wait_rlt_checkpoints(identities.__getitem__, sleep=lambda _: self.fail('Unexpected wait'))

    def test_checkpoint_wait_stops_on_driver_failure_or_changed_receipt(self):
        for failure in ('pid_reuse', 'finished', 'receipt_changed'):
            with self.subTest(failure=failure), self.checkpoint_fixture() as (rows, identities, complete, previous):
                clock = [0]
                def sleep(seconds):
                    clock[0] += seconds
                    if failure == 'pid_reuse':
                        identities[104] = dict(identities[104], start_ticks=9999)
                    elif failure == 'finished':
                        (Path(rows['gpu4']['new_run']) / 'runtime/finished.json').write_text('{"exit_code": 1}')
                    else:
                        with (previous / 'plan.json').open('a') as stream: stream.write(' ')
                with self.assertRaises(ValueError):
                    m.wait_rlt_checkpoints(identities.__getitem__, sleep=sleep, monotonic=lambda: clock[0])
                self.assertEqual(clock[0], 30)

    def test_checkpoint_wait_times_out_read_only_at_two_hours(self):
        with self.checkpoint_fixture() as (rows, identities, complete, previous):
            clock = [0]
            with self.assertRaises(TimeoutError):
                m.wait_rlt_checkpoints(identities.__getitem__, monotonic=lambda: clock[0],
                    sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
            self.assertEqual(clock[0], 7200)
            self.assertTrue(all(not list(Path(row['new_run']).glob('**/checkpoints')) for row in rows.values()))

    def test_new_control_cycle_keeps_training_paths(self):
        self.assertNotEqual(m.ROOT, m.TRAIN_ROOT)
        self.assertEqual(m.PREVIOUS, m.TRAIN_ROOT / 'rlt-cycle-expo-turn-switch-repair-20261002-v1')
        self.assertEqual(m.CYCLE, m.ROOT / 'rlt-cycle-expo-eval10-20261003-v1')
        command = m.driver_command()
        self.assertEqual(command[command.index('--run') + 1], str(m.TRAIN_ROOT / 'run'))
        self.assertEqual(command[command.index('--resume') + 1], str(m.TRAIN_ROOT / 'run/checkpoint-latest.pt'))
        self.assertEqual(command[command.index('--max-physical-actions') + 1], '20000')

    def test_previous_return_requires_all_four_proofs(self):
        current = dict(status='RLT_RESTORED')
        final = dict(gpu_released=True, rlt_first_rounds_verified=True)
        release = dict(cycle_id=r.PREVIOUS.name, all_workers_stopped=True)
        status = dict(cycle_id=r.PREVIOUS.name, all_first_rounds_verified=True)
        r.validate_previous(current, final, release, status)
        for index in range(4):
            values = copy.deepcopy([current, final, release, status]); values[index] = {}
            with self.assertRaises(ValueError): r.validate_previous(*values)

    def test_gpu_release_tolerates_lag_but_requires_settling(self):
        clock = [0.0]; probes = []
        def probe():
            probes.append(clock[0]); return [dict(gpu=m.UUIDS[0])] if clock[0] < 7 else []
        m.wait_gpu_release(probe, sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds), monotonic=lambda: clock[0])
        self.assertEqual(clock[0], 9.0)
        self.assertGreaterEqual(len(probes), 10)

    def test_gpu_release_reappearing_allocation_resets_settle(self):
        clock = [0.0]
        def probe(): return [dict(gpu=m.UUIDS[1])] if clock[0] == 1 else []
        m.wait_gpu_release(probe, sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds), monotonic=lambda: clock[0])
        self.assertEqual(clock[0], 4.0)

    def test_gpu_release_fails_closed_after_bound(self):
        clock = [0.0]
        with self.assertRaises(TimeoutError):
            m.wait_gpu_release(lambda: [dict(gpu=m.UUIDS[2])], timeout=60,
                sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds), monotonic=lambda: clock[0])
        self.assertEqual(clock[0], 60.0)
        with self.assertRaises(ValueError): m.wait_gpu_release(lambda: [], timeout=59)

    def test_other_gpu_does_not_block_release(self):
        clock = [0.0]
        m.wait_gpu_release(lambda: [dict(gpu='other')], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
                           monotonic=lambda: clock[0])
        self.assertEqual(clock[0], 2.0)

    def test_identity_formats_and_pid_reuse(self):
        short = dict(pid=10, uid=20001, start=100)
        full = dict(pid=10, uid=20001, start_ticks=100, boot_id='a', state='S')
        self.assertEqual(m.anchor(short), m.anchor(full))
        self.assertTrue(m.exact_live(short, lambda pid: full))
        self.assertFalse(m.exact_live(dict(short, start=101), lambda pid: full))
        self.assertFalse(m.exact_live(dict(short, boot_id='b'), lambda pid: full))
        self.assertFalse(m.exact_live(short, lambda pid: dict(full, state='Z')))

    def test_rebind_requires_exact_two_queues_and_new_owner(self):
        owner = dict(pid=10, uid=20001, start_ticks=100)
        ready = dict(version=1, ok=True, cycle_id=m.CYCLE.name, owner=owner,
                     queues=[dict(stage=str(path)) for path in m.QUEUES])
        plan = dict(cycle_id=m.CYCLE.name, predecessor_cycle=str(m.PREVIOUS))
        m.validate_rebind(ready, plan, owner)
        for change in ('owner', 'cycle', 'queue'):
            bad = copy.deepcopy(ready)
            if change == 'owner': bad['owner']['start_ticks'] += 1
            elif change == 'cycle': bad['cycle_id'] = 'old'
            else: bad['queues'][0]['stage'] = '/data/chenyiteng/foreign'
            with self.assertRaises(ValueError): m.validate_rebind(bad, plan, owner)

    def test_migration_cannot_change_payload_or_save_policy(self):
        hashes = {key: 'a' * 64 for key in m.PAYLOAD}
        good = dict(ok=True, old_inputs_sha256='b' * 64, old_payload_hashes=hashes, new_payload_hashes=copy.deepcopy(hashes),
                    save_policy_changed=False, replay_path_identity_changed=False)
        m.validate_migration(good, 'b' * 64)
        for key in m.PAYLOAD:
            bad = copy.deepcopy(good); bad['new_payload_hashes'][key] = 'c' * 64
            with self.assertRaises(ValueError): m.validate_migration(bad, 'b' * 64)
        for key in ('save_policy_changed', 'replay_path_identity_changed'):
            bad = copy.deepcopy(good); bad[key] = True
            with self.assertRaises(ValueError): m.validate_migration(bad, 'b' * 64)

    def test_release_before_driver_uses_not_started_without_masking_started_failure(self):
        self.assertEqual(m.release_terminal_status(False, [], False), 'not_started')
        self.assertEqual(m.release_terminal_status(False, [dict(pid=10)], True), 'failed')
        self.assertEqual(m.release_terminal_status(False, [], True), 'failed')
        self.assertEqual(m.release_terminal_status(True, [dict(pid=10)], True), 'completed')


if __name__ == '__main__':
    unittest.main()
