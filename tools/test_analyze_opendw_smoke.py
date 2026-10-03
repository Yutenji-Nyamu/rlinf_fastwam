"""Server CPU fixtures; no GPU, network or TensorBoard dependency required."""

import json
from pathlib import Path
import tempfile
import unittest

from analyze_opendw_smoke import analyze, markdown, timestamp


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def jsonl(self, relative, values):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(value) for value in values) + "\n", encoding="utf-8")

    def setup_trials(self):
        trials, resources = [], []
        for key, n, start, end in [("n8", 8, 100, 200), ("n16", 16, 300, 400)]:
            config = {"env": {"train": {"max_episode_steps": 32, "max_steps_per_rollout_epoch": 32,
                      "rollout_epoch": 1, "use_rel_reward": True, "reward_coef": 1.0}},
                      "algorithm": {"group_size": 8, "filter_rewards": True,
                                    "rewards_lower_bound": 0.1, "rewards_upper_bound": 0.9}}
            config_path = self.write(key + "/config.json", config)
            trials.append({"key": key, "num_envs": n, "config": str(config_path)})
            self.write(key + "/result.json", {"seconds": end - start, "exit_code": 0})
            self.write(key + "/driver-finished.json", {"time": end, "exit_code": 0})
            resources.extend([{"time": start, "phase": key, "processes": [{"pid": 1, "VmRSS_kib": 2048}],
                               "compute_memory_csv": "1, GPU-example, 4096 MiB"},
                              {"time": end + 1, "phase": key + "_complete"}])
        self.write("owner-plan.json", {"trials": trials})
        self.jsonl("resources.jsonl", resources)

    @staticmethod
    def request(start, n, scores, complete=True):
        base = {"pid": 1, "request_id": "seed1-call0"}  # deliberately repeats
        events = [{**base, "timestamp_utc": start, "event": "request_started", "batch": n}]
        for row, score in enumerate(scores):
            events.append({**base, "timestamp_utc": start + row + 1, "event": "row_completed",
                           "row": row, "seconds": 2.0, "score_last": score, "score_max": score,
                           "VmRSS_bytes": 1024**3, "Pss_bytes": 512 * 1024**2,
                           "cuda_allocated_bytes": 2048 * 1024**2})
        if complete:
            events.append({**base, "timestamp_utc": start + n + 2, "event": "request_completed", "batch": n})
        return events

    def test_repeated_request_ids_are_separate_and_g8_filter_reconstructed(self):
        self.setup_trials()
        self.jsonl("service/service-events.jsonl", self.request(110, 8, [0.2] * 8) +
                   self.request(310, 16, [0.95] * 8 + [0.5] * 8))
        self.jsonl("n8/metrics.jsonl", [{"metrics": {"actor/grad_norm": 1.5, "actor/loss": 0.4}}])
        (self.root / "n8/experiment/checkpoints/global_step_1/actor").mkdir(parents=True)
        report = analyze(self.root)
        a, b = report["trials"]
        self.assertEqual([a["request_count"], b["request_count"]], [1, 1])
        self.assertNotEqual(a["requests"][0]["index"], b["requests"][0]["index"])
        self.assertEqual(a["reward_filter"]["retained_groups"], 1)
        self.assertEqual(b["reward_filter"]["retained_groups"], 1)
        self.assertEqual(len(b["reward_filter"]["groups"]), 2)
        self.assertEqual(a["pure_wm_row_seconds"]["status"], "unknown")
        self.assertTrue(a["training"]["nonzero_grad_norm_reported"])
        self.assertEqual(a["training"]["checkpoints"][0]["status"], "unknown_no_explicit_completion_marker")
        self.assertIn("not established", a["training"]["effective_learning"])
        self.assertEqual(a["memory"]["managed_compute_memory_sample_peak_bytes"], 4096 * 1024**2)
        self.assertIn("n16", markdown(report))
        json.dumps(report, allow_nan=False)

    def test_incomplete_request_stays_unassigned_and_loss_is_not_gradient_evidence(self):
        self.setup_trials()
        self.jsonl("service-events.jsonl", self.request(110, 8, [0.3] * 7, complete=False))
        self.jsonl("n8/metrics.jsonl", [{"actor/loss": 0.4}])
        report = analyze(self.root)
        self.assertEqual(len(report["unassigned_requests"]), 1)
        self.assertEqual(report["trials"][0]["reward_filter"]["status"], "unknown")
        self.assertIsNone(report["trials"][0]["training"]["nonzero_grad_norm_reported"])

    def test_naive_time_requires_explicit_offset(self):
        self.assertIsNone(timestamp("2026-10-03 12:00:00"))
        self.assertEqual(timestamp("2026-10-03 12:00:00", "+08:00"),
                         timestamp("2026-10-03T04:00:00Z"))


if __name__ == "__main__":
    unittest.main()
