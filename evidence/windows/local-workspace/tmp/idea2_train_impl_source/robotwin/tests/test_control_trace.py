import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "envs" / "control_trace.py"
SPEC = importlib.util.spec_from_file_location("control_trace", MODULE_PATH)
control_trace = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = control_trace
SPEC.loader.exec_module(control_trace)


class ControlTracePureTest(unittest.TestCase):
    def test_deterministic_selection(self):
        config = {
            "enabled": True,
            "worker_indices": [0],
            "env_slots": [0],
            "max_episodes_per_slot": 1,
            "worker_index": 0,
            "env_slot": 0,
            "episode_index_within_slot": 0,
        }
        self.assertTrue(control_trace.is_trace_selected(config))
        self.assertFalse(
            control_trace.is_trace_selected({**config, "episode_index_within_slot": 1})
        )
        self.assertFalse(control_trace.is_trace_selected({**config, "env_slot": 1}))

    def test_progress_mapping_and_query_end_budget(self):
        mapping = control_trace.progress_to_action_mapping(0.5, 50)
        self.assertEqual(mapping["h_lo"], 24)
        self.assertEqual(mapping["h_hi"], 25)
        self.assertAlmostEqual(mapping["h_fraction"], 0.5)

        sampler = control_trace.ProgressBinSampler(chunk_len=50, max_frames=50)
        decisions = []
        for step in range(500):
            progress = (step + 1) / 500
            decisions.append(
                sampler.select(progress, query_end=(step == 499), success=False)
            )
        decisions = [decision for decision in decisions if decision is not None]
        self.assertLessEqual(len(decisions), 50)
        self.assertEqual(decisions[-1].reason, "query_end")
        self.assertEqual(decisions[-1].mapping["h_lo"], 49)

    def test_success_frame_is_terminal_sample(self):
        sampler = control_trace.ProgressBinSampler(chunk_len=50, max_frames=50)
        self.assertIsNotNone(sampler.select(0.01))
        decision = sampler.select(0.2, success=True, query_end=True)
        self.assertEqual(decision.reason, "success")

    def test_recording_budget_survives_environment_recreation(self):
        with tempfile.TemporaryDirectory() as output_dir:
            config = {
                "enabled": True,
                "output_dir": output_dir,
                "worker_indices": [0],
                "env_slots": [0],
                "max_episodes_per_slot": 1,
                "worker_index": 0,
                "env_slot": 0,
                "episode_index_within_slot": 0,
            }
            first = control_trace.ControlTraceRecorder.from_config(
                config,
                reset_id=7,
                task_name="adjust_bottle",
            )
            self.assertIsNotNone(first)
            first.finish("test")

            # A freshly constructed SubEnv presents episode index zero again,
            # but the run-level claim prevents a second recording.
            second = control_trace.ControlTraceRecorder.from_config(
                config,
                reset_id=7,
                task_name="adjust_bottle",
            )
            self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
