"""CPU-only fixtures: run on the server, no Ray/Torch/TensorBoard required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import analyze_multigpu_smoke as A


def config(length=32):
    return dict(env=dict(train=dict(total_num_envs=64, rollout_epoch=8, chunk=32, frame_stride=4,
                 max_episode_steps=length, max_steps_per_rollout_epoch=length, env_type='opendw_robotwin',
                 group_size=8, auto_reset=False, ignore_terminations=False, use_rel_reward=True,
                 reward_coef=1., success_reward_threshold=.9, seed=0)),
                algorithm=dict(group_size=8, filter_rewards=True, rewards_lower_bound=.1,
                 rewards_upper_bound=.9, adv_type='grpo', reward_type='chunk_level', update_epoch=2),
                cluster=dict(component_placement=A.PLACEMENT.copy()),
                rollout=dict(pipeline_stage_num=1),
                runner=dict(max_steps=1, max_epochs=1, resume_dir=None, ckpt_path=None))


def requests(length=32, early=False):
    result = {0: [], 1: []}
    for rank in (0, 1):
        for epoch in range(8):
            for chunk in range(1 if early else length // 32):
                call = len(result[rank])
                rows = [dict(row=i, env_index=i, global_env_index=rank * 32 + i,
                             env_process_index=rank, env_process_count=2, reset_id=42,
                             score_last=.2 + i % 8 * .01, score_max=.95 if early else .4)
                        for i in range(32)]
                result[rank].append(dict(request_id=f'seed{rank}-call{call}', completed=True,
                                         batch=32, rows=rows, row_starts=[]))
    return result


class MultiGpuAnalysisTests(unittest.TestCase):
    def test_repeated_reset_and_local_ids_do_not_merge_rank_or_epoch(self):
        report = A.reconstruct(config(), requests())
        self.assertEqual(report['status'], 'reconstructed')
        self.assertEqual(report['trajectories'], 512)
        self.assertEqual(len(report['groups']), 64)
        self.assertEqual(report['retained_groups'], 64)
        self.assertEqual(report['valid_chunks_after_filter'], 512)
        self.assertEqual(report['nonzero_advantage_groups'], 64)
        self.assertEqual(len({(g['env_rank'], g['rollout_epoch'], g['local_group']) for g in report['groups']}), 64)

    def test_full_horizon_and_early_success_remain_separate_epochs(self):
        full = A.reconstruct(config(384), requests(384))
        early = A.reconstruct(config(384), requests(384, early=True))
        self.assertEqual(full['valid_chunks_after_filter'], 6144)
        self.assertEqual(early['valid_chunks_after_filter'], 512)
        self.assertEqual(early['reward_model_success_trajectories'], 512)
        self.assertLess(early['terminal_score']['max'], .9)  # max-hit and final reward differ

    def test_incomplete_or_crossed_rank_or_reset_is_unknown(self):
        for mutation in ('missing_tail', 'wrong_global', 'reset_change'):
            with self.subTest(mutation=mutation):
                value = requests(384)
                if mutation == 'missing_tail':
                    value[1].pop()
                elif mutation == 'wrong_global':
                    value[1][0]['rows'][0]['global_env_index'] = 0
                else:
                    value[0][1]['rows'][0]['reset_id'] = 11
                self.assertEqual(A.reconstruct(config(384), value)['status'], 'unknown')

    def test_zero_and_equal_rewards_do_not_prove_learning(self):
        value = requests()
        for by_rank in value.values():
            for req in by_rank:
                for row in req['rows']:
                    row.update(score_last=.00001, score_max=.00001)
        rec = A.reconstruct(config(), value)
        cp = dict(finite_nonzero_update_state=True, cuda_initialized=False)
        tb = dict(finite=True, stable=True, nonzero_grad=True, nonzero_advantage=True)
        self.assertEqual(A.decide(True, rec, tb, cp), 'no_valid_group')
        valid = A.reconstruct(config(), requests())
        self.assertEqual(A.decide(True, valid, {}, cp), 'unknown')
        self.assertEqual(A.decide(True, valid, tb, {}), 'nonzero_training_signal_observed_checkpoint_unverified')
        self.assertEqual(A.decide(True, valid, tb, cp), 'effective_update_verified')
        self.assertEqual(A.decide(False, valid, tb, cp), 'unknown')

    def test_compute_per_gpu_and_graphics_outside_are_distinct(self):
        resources = [dict(processes=[dict(pid=101, VmRSS_kib=10), dict(pid=202, VmRSS_kib=20)],
                          gpu_processes=[dict(pid=101, gpu=4, uuid='GPU-a', type='C'),
                                         dict(pid=202, gpu=6, uuid='GPU-b', type='C'),
                                         dict(pid=101, gpu=0, uuid='GPU-zero', type='G')],
                          compute_memory_csv='101, GPU-a, 4096 MiB\n202, GPU-b, 8192 MiB\n303, GPU-b, 9999 MiB')]
        report = A.resources_summary(resources, {})
        self.assertEqual(report['managed_compute_peak_bytes_by_gpu']['4'], 4096 * 1024**2)
        self.assertEqual(report['managed_compute_peak_bytes_by_gpu']['6'], 8192 * 1024**2)
        self.assertIsNone(report['managed_compute_peak_bytes_by_gpu']['5'])
        self.assertEqual(report['owned_outside_gpu4_7_contexts'], [dict(gpu=0, pid=101, type='G')])

    def test_repeated_request_envelopes_and_naive_times(self):
        events = []
        for start in (10, 50):
            events.extend([dict(event='request_started', request_id='seed0-call0', pid=1, timestamp_utc=start, batch=1),
                           dict(event='row_completed', request_id='seed0-call0', pid=1, timestamp_utc=start+1, row=0),
                           dict(event='request_completed', request_id='seed0-call0', pid=1, timestamp_utc=start+2)])
        result = A.request_envelopes(events)
        self.assertEqual(len(result), 2)
        self.assertEqual([r['start'] for r in result], [10., 50.])
        self.assertIsNone(A.timestamp('2026-10-03 23:00:00'))
        self.assertEqual(A.timestamp('2026-10-03 23:00:00', '+08:00'), A.timestamp('2026-10-03T15:00:00Z'))

    def test_report_json_markdown_svg_with_unknowns(self):
        trial = dict(key='n64_short', trial_seconds=12, learning_status='unknown', binding_verified=False,
                     services=[dict(key='wm6', physical_gpu=6, row_count=0,
                                    wm_plus_reward_row_seconds=A.unknown('missing'),
                                    score_last_per_chunk=A.unknown('missing'),
                                    actions=dict(actions=A.unknown('missing')))],
                     reconstruction=A.unknown('Incomplete rollout'), resources=A.resources_summary([], {}))
        report = dict(trials=[trial], unassigned_requests=[], issues=[])
        self.assertIn('unknown', A.markdown(report))
        self.assertEqual(ET.fromstring(A.svg(report)).tag, '{http://www.w3.org/2000/svg}svg')
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
