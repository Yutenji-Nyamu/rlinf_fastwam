"""CPU fixtures for B16 HTTP row ordering, tail batches and resource logging.

Run on the server beside wm_batch.py and the frozen reward/telemetry modules.
No checkpoint, CUDA initialization, HTTP listener, or GPU process is needed.
"""
import io
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent))
import opendw_service_batched as S


class FakeLog:
    def __init__(self):
        self.phase = "ready_loaded"
        self.rows = []

    def event(self, event, **fields):
        self.rows.append(dict(event=event, **fields))


def data_for(size):
    images = np.stack([np.full((384, 320, 3), i, np.uint8) for i in range(size)])
    actions = np.stack([np.full((32, 14), i / 100, np.float32) for i in range(size)])
    states = np.stack([np.full(14, i, np.float32) for i in range(size)])
    data = dict(images=images, actions=actions, states=states,
                seeds=np.arange(100, 100 + size, dtype=np.int64),
                instructions=np.asarray(["short" if i % 2 == 0 else "long text" for i in range(size)]),
                global_env_indices=np.arange(200, 200 + size, dtype=np.int64))
    stream = io.BytesIO()
    np.savez(stream, **data)
    return S.decode_request(stream.getvalue())


def backend_for(mode="batched"):
    b = object.__new__(S.Backend)
    b.args = types.SimpleNamespace(execution_mode=mode, wm_batch_size=16 if mode == "batched" else 1,
                                   evidence_samples=0)
    b.log, b.lock, b.torch, b.device = FakeLog(), threading.RLock(), torch, torch.device("cpu")
    b.requests_completed = b.rows_completed = b.samples_saved = b.batch_calls_completed = b.max_actual_wm_batch = 0
    b.onload = Mock(return_value={"ok": True})
    b.pil_to_model_tensor = lambda image, device, dtype: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
    b.policy = types.SimpleNamespace(
        config=types.SimpleNamespace(sigma_shift=5.0),
        normalize_action_condition=lambda action, state: torch.from_numpy(action.copy()),
        normalize_state=lambda state: torch.from_numpy(state.copy()).unsqueeze(0),
        format_prompt=lambda value: "prompt " + value,
        norm_stats={"action": {"mean": np.zeros(14), "std": np.ones(14)},
                    "state": {"mean": np.zeros(14), "std": np.ones(14)}},
        normalization_mode="zscore_14d", action_condition_mode="absolute")
    b.policy.model = types.SimpleNamespace(torch_dtype=torch.float32)
    b.policy.model.encode_prompt = Mock(side_effect=lambda prompt: (
        torch.full((1, 2 if prompt.endswith("short") else 3, 4), 1.0 if prompt.endswith("short") else 2.0),
        torch.ones((1, 2 if prompt.endswith("short") else 3), dtype=torch.bool)))

    def reward(images, labels):
        expected = ["short" if int(image[0, 0, 0]) % 2 == 0 else "long text" for image in images]
        if expected != labels:
            raise AssertionError("Reward text/image ordering changed")
        return images[:, 0, 0, 0].float() / 100

    b.reward = types.SimpleNamespace(compute_reward=Mock(side_effect=reward))
    return b


def fake_prediction(model, **kwargs):
    size = len(kwargs["seeds"])
    assert tuple(kwargs["input_image"].shape) == (size, 3, 384, 320)
    assert tuple(kwargs["action"].shape) == (size, 32, 14)
    assert tuple(kwargs["proprio"].shape) == (size, 14)
    assert tuple(kwargs["context"].shape) == (size, 3, 4)
    videos = []
    for row, seed in enumerate(kwargs["seeds"]):
        index = seed - 100
        assert int(kwargs["input_image"][row, 0, 0, 0]) == index
        assert int(kwargs["proprio"][row, 0]) == index
        assert abs(float(kwargs["action"][row, 0, 0]) - index / 100) < 1e-5
        assert kwargs["context_mask"][row].tolist() == ([True, True, False] if index % 2 == 0 else [True] * 3)
        videos.append([Image.new("RGB", (320, 384), color=(index, index, index)) for _ in range(9)])
    return dict(video=videos, batch_size=size, denoiser_batch_sizes=[size] * 10)


class BatchServiceTests(unittest.TestCase):
    def test_ranges_preserve_all_rows_and_tail_without_padding(self):
        for size, expected in ((1, [(0, 1)]), (16, [(0, 16)]),
                               (19, [(0, 16), (16, 19)]), (32, [(0, 16), (16, 32)])):
            with self.subTest(size=size):
                self.assertEqual(S.batch_ranges(size, 16), expected)
        with self.assertRaises(ValueError):
            S.batch_ranges(0, 16)

    def test_true_batches_preserve_seed_state_action_reward_and_output_order(self):
        b, data = backend_for(), data_for(19)
        original = {key: value.copy() for key, value in data.items()}
        with patch.object(S, "infer_joint_batch", side_effect=fake_prediction) as kernel:
            encoded = b.infer(data)
        self.assertEqual([len(call.kwargs["seeds"]) for call in kernel.call_args_list], [16, 3])
        self.assertEqual([call.args[0].shape[0] for call in b.reward.compute_reward.call_args_list], [128, 24])
        with np.load(io.BytesIO(encoded), allow_pickle=False) as response:
            np.testing.assert_array_equal(response["next_images"][:, 0, 0, 0], np.arange(19))
            np.testing.assert_allclose(response["scores"], np.repeat(np.arange(19)[:, None] / 100, 8, axis=1))
            self.assertEqual(response["timing_s"].shape, (19,))
        for key, value in original.items():
            np.testing.assert_array_equal(data[key], value)
        batches = [row for row in b.log.rows if row["event"] == "batch_completed"]
        rows = [row for row in b.log.rows if row["event"] == "row_completed"]
        self.assertEqual([row["actual_wm_batch"] for row in batches], [16, 3])
        self.assertTrue(all(row["outputs_finite"] for row in batches))
        self.assertEqual([row["global_env_index"] for row in rows], list(range(200, 219)))
        self.assertEqual(b.max_actual_wm_batch, 16)
        self.assertEqual(b.batch_calls_completed, 2)
        # Each of two unique prompts is encoded once per batch, not per row.
        self.assertEqual(b.policy.model.encode_prompt.call_count, 4)

    def test_kernel_error_does_not_fallback_to_b1(self):
        b = backend_for()
        b.policy.model.infer_joint = Mock(side_effect=AssertionError("B1 must not run"))
        with patch.object(S, "infer_joint_batch", side_effect=RuntimeError("CUDA OOM")), \
             self.assertRaisesRegex(RuntimeError, "CUDA OOM"):
            b.infer(data_for(16))
        b.policy.model.infer_joint.assert_not_called()
        self.assertEqual(b.requests_completed, 0)
        self.assertFalse(any(row["event"] == "batch_completed" for row in b.log.rows))

    def test_kernel_must_report_the_full_batch_on_every_denoiser_step(self):
        b = backend_for()

        def wrong(model, **kwargs):
            result = fake_prediction(model, **kwargs)
            result["denoiser_batch_sizes"][-1] = 1
            return result

        with patch.object(S, "infer_joint_batch", side_effect=wrong), \
             self.assertRaisesRegex(RuntimeError, "true batch"):
            b.infer(data_for(16))

    def test_b1_reference_is_explicit_and_never_uses_batch_kernel(self):
        b = backend_for("b1_reference")
        b.policy.model.infer_joint = Mock(side_effect=lambda **kwargs: {
            "video": [Image.new("RGB", (320, 384), color=(kwargs["seed"] - 100,) * 3) for _ in range(9)]})
        with patch.object(S, "infer_joint_batch") as kernel:
            b.infer(data_for(2))
        kernel.assert_not_called()
        self.assertEqual(b.policy.model.infer_joint.call_count, 2)
        self.assertEqual(b.max_actual_wm_batch, 1)

    def test_invalid_video_or_reward_shape_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "videos"):
            S.validate_videos([], 1)
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            S.validate_videos([[Image.new("RGB", (100, 100))] * 9], 1)
        b = backend_for()
        b.reward.compute_reward = Mock(return_value=torch.tensor([float("nan")] * 8))
        with self.assertRaisesRegex(RuntimeError, "probabilities"):
            b._score_videos([[Image.new("RGB", (320, 384))] * 9], ["short"])

    def test_resource_rows_use_cached_expensive_sample_until_monitor_refresh(self):
        log = object.__new__(S.ResourceLog)
        log._resource_lock = threading.Lock()
        log._resource_cache = None
        log.phase = "world_model_inference"
        log.torch = None
        log._sample_resources = Mock(return_value={"resource_timestamp_utc": "fixture", "Pss_bytes": 123})
        for _ in range(32):
            self.assertEqual(log.snapshot()["Pss_bytes"], 123)
        self.assertEqual(log._sample_resources.call_count, 1)
        log.snapshot(refresh_resources=True)
        self.assertEqual(log._sample_resources.call_count, 2)


if __name__ == "__main__":
    unittest.main()
