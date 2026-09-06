from __future__ import annotations

import ast
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.robotwin.fastwam_policy.dvac_telemetry import (
    FastWAMDvacTelemetryWriter,
    compute_z_endpoint,
)


class FastWAMDvacTelemetryTest(unittest.TestCase):
    def test_default_is_off_in_config_and_model_signature(self) -> None:
        config = (ROOT / "configs" / "sim_robotwin.yaml").read_text(encoding="utf-8")
        self.assertRegex(config, r"dvac_telemetry:\s*\n\s+enabled: false")

        tree = ast.parse(
            (ROOT / "src" / "fastwam" / "models" / "wan22" / "fastwam.py").read_text(
                encoding="utf-8"
            )
        )
        infer_action = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "infer_action"
        )
        names_with_defaults = [arg.arg for arg in infer_action.args.args][
            -len(infer_action.args.defaults) :
        ]
        default_by_name = dict(zip(names_with_defaults, infer_action.args.defaults))
        self.assertIsInstance(default_by_name["return_action_denoising_trace"], ast.Constant)
        self.assertIs(default_by_name["return_action_denoising_trace"].value, False)

    def test_capture_math_does_not_draw_rng_or_change_euler_result(self) -> None:
        def run(enabled: bool):
            rng = np.random.default_rng(7)
            x = rng.standard_normal((32, 14), dtype=np.float32)
            state_after_draw = json.dumps(rng.bit_generator.state, sort_keys=True)
            xs = [x.copy()]
            vs = []
            x_nexts = []
            timesteps = np.asarray([1000.0, 700.0, 300.0], dtype=np.float32)
            deltas = np.asarray([-0.3, -0.4, -0.3], dtype=np.float32)
            for timestep, delta in zip(timesteps, deltas):
                velocity = x * np.float32(0.125) + timestep / np.float32(10000.0)
                x = x + delta * velocity
                if enabled:
                    vs.append(velocity.copy())
                    x_nexts.append(x.copy())
                    xs.append(x.copy())
            state_after_capture = json.dumps(rng.bit_generator.state, sort_keys=True)
            trace = None
            if enabled:
                trace = {
                    "x_chain": np.stack(xs),
                    "v_chain": np.stack(vs),
                    "timesteps": timesteps,
                    "deltas": deltas,
                    "x_next": np.stack(x_nexts),
                }
            return x, state_after_draw, state_after_capture, trace

        action_off, draw_off, state_off, _ = run(False)
        action_on, draw_on, state_on, trace = run(True)
        np.testing.assert_array_equal(action_off, action_on)
        self.assertEqual(draw_off, draw_on)
        self.assertEqual(state_off, state_on)
        assert trace is not None
        np.testing.assert_array_equal(trace["x_chain"][1:], trace["x_next"])

    def test_endpoint_formula_and_writer_round_trip(self) -> None:
        x0 = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
        v = np.ones((1, 3, 2), dtype=np.float32) * 2
        timesteps = np.asarray([500], dtype=np.float32)
        z = compute_z_endpoint(x0, v, timesteps)
        np.testing.assert_array_equal(z, x0[:-1] - 1.0)

        observation = {
            "observation": {
                "head_camera": {"rgb": np.zeros((4, 5, 3), dtype=np.uint8)},
                "left_camera": {"rgb": np.ones((3, 2, 3), dtype=np.uint8)},
                "right_camera": {"rgb": np.full((3, 2, 3), 2, dtype=np.uint8)},
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            official_video_dir = Path(tmp) / "official-videos"
            writer = FastWAMDvacTelemetryWriter(
                output,
                manifest={
                    "run_id": "unit",
                    "policy": "fastwam",
                    "task_config": "demo_clean",
                    "official_video_dir": str(official_video_dir),
                    "H": 3,
                    "C": 2,
                    "M": 1,
                    "D": 2,
                },
                project_root=ROOT,
                skip_get_obs_within_replan=False,
            )
            writer.begin_episode(1)
            writer.record_query(
                task="adjust_bottle",
                instruction="adjust the bottle",
                reset_id=9,
                query_start_action_slot=0,
                planned_exec_length=2,
                observation=observation,
                robot_state=np.zeros(2, dtype=np.float32),
                final_model_action=x0[-1],
                env_action=x0[-1],
                trace={
                    "x_chain": x0,
                    "v_chain": v,
                    "timesteps": timesteps,
                    "deltas": np.asarray([-0.5], dtype=np.float32),
                    "x_next": x0[1:],
                },
            )
            writer.record_action(success_after=False)
            writer.record_action(success_after=True)
            writer.close()

            with np.load(output / "traces" / "episode0001_query0000.npz") as payload:
                np.testing.assert_array_equal(payload["x_chain"], x0)
                np.testing.assert_array_equal(payload["x_next"], x0[1:])
            with (output / "queries.csv").open(newline="", encoding="utf-8") as handle:
                query = next(csv.DictReader(handle))
            self.assertEqual(query["executed_length"], "2")
            self.assertEqual(query["success_after"], "True")
            with (output / "episodes.csv").open(newline="", encoding="utf-8") as handle:
                episode = next(csv.DictReader(handle))
            self.assertEqual(episode["success"], "True")
            self.assertEqual(episode["first_success_action_slot"], "1")
            self.assertEqual(
                episode["video_path"],
                str(official_video_dir / "episode9_randomized-false_success-true.mp4"),
            )
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["raw_shapes_per_query"]["x_chain"], [2, 3, 2])
            with self.assertRaises(FileExistsError):
                FastWAMDvacTelemetryWriter(
                    output,
                    manifest={"run_id": "duplicate"},
                    project_root=ROOT,
                    skip_get_obs_within_replan=False,
                )

    def test_failure_is_finalized_once_at_next_reset(self) -> None:
        observation = {
            "observation": {
                camera: {"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
                for camera in ("head_camera", "left_camera", "right_camera")
            }
        }
        x_chain = np.zeros((2, 1, 1), dtype=np.float32)
        trace = {
            "x_chain": x_chain,
            "v_chain": np.zeros((1, 1, 1), dtype=np.float32),
            "timesteps": np.asarray([1000], dtype=np.float32),
            "deltas": np.asarray([-1], dtype=np.float32),
            "x_next": x_chain[1:],
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            writer = FastWAMDvacTelemetryWriter(
                output,
                manifest={"run_id": "failure", "policy": "fastwam"},
                project_root=ROOT,
                skip_get_obs_within_replan=True,
            )
            writer.begin_episode(1)
            writer.record_query(
                task="adjust_bottle",
                instruction="adjust the bottle",
                reset_id=3,
                query_start_action_slot=0,
                planned_exec_length=1,
                observation=observation,
                robot_state=np.zeros(1, dtype=np.float32),
                final_model_action=np.zeros((1, 1), dtype=np.float32),
                env_action=np.zeros((1, 1), dtype=np.float32),
                trace=trace,
            )
            writer.record_action(success_after=False)
            writer.begin_episode(2)
            writer.close()

            with (output / "episodes.csv").open(newline="", encoding="utf-8") as handle:
                episodes = list(csv.DictReader(handle))
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0]["episode_id"], "1")
            self.assertEqual(episodes[0]["success"], "False")


if __name__ == "__main__":
    unittest.main()
