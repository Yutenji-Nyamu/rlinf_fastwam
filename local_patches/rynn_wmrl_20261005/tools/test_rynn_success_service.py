"""CPU contract tests; run on the server without loading weights or CUDA."""

import io
import threading
import unittest
from types import SimpleNamespace

import numpy as np

import rynn_success_service as service


def request(rows=2):
    return {
        "frames": np.zeros((rows, 8, 256, 320, 3), dtype=np.uint8),
        "instructions": np.asarray(["Adjust the bottle"] * rows),
        "episode_uids": np.asarray([f"episode-{i}" for i in range(rows)]),
        "end_action_indices": np.full(rows, 32, dtype=np.int64),
        "frame_action_indices": np.tile(np.arange(4, 33, 4), (rows, 1)),
        "request_id": np.asarray("round-1-block-1"),
    }


class ParserTests(unittest.TestCase):
    def test_yes_no_are_distinct_from_description_and_match(self):
        result = service.parse_analysis("- Video Description: Yes, the robot moved.\n- Match: Yes\n- Success: No")
        self.assertIs(result["success"], False)
        self.assertIs(result["match"], True)
        self.assertEqual(result["parse_status"], "ok")
        self.assertIsNone(service.parse_analysis("Video Description: Success: Yes")["success"])

    def test_missing_and_truncated_are_unknown(self):
        self.assertEqual(service.parse_analysis("- Match: Yes")["parse_status"], "missing_success")
        self.assertEqual(service.parse_analysis("- Match: Yes", truncated=True)["parse_status"], "truncated")
        self.assertIsNone(service.parse_analysis("- Success: Yes/No")["success"])

    def test_conflicting_success_is_not_accepted(self):
        for text in ("- Success: Yes\n- Success: No", "- Match: No\n- Success: Yes"):
            result = service.parse_analysis(text)
            self.assertIsNone(result["success"])
            self.assertTrue(result["parse_status"].startswith("conflicting"))

    def test_valid_answer_can_have_case_whitespace_punctuation(self):
        self.assertIs(service.parse_analysis("  - success: YES.\n- Match: yes")["success"], True)


class RequestTests(unittest.TestCase):
    def test_npz_roundtrip_without_optional_override(self):
        payload = io.BytesIO()
        np.savez(payload, **request(1))
        decoded = service.decode_request(payload.getvalue())
        self.assertEqual(decoded["episode_uids"], ["episode-0"])
        self.assertNotIn("rm_batch_size", decoded)

    def test_npz_roundtrip_preserves_identity_timing_and_batch_override(self):
        data = request()
        data["rm_batch_size"] = np.asarray(4, dtype=np.int64)
        payload = io.BytesIO()
        np.savez(payload, **data)
        decoded = service.decode_request(payload.getvalue())
        self.assertEqual(decoded["episode_uids"], ["episode-0", "episode-1"])
        self.assertEqual(decoded["request_id"], "round-1-block-1")
        self.assertEqual(decoded["rm_batch_size"], 4)
        np.testing.assert_array_equal(decoded["frame_action_indices"], data["frame_action_indices"])

    def test_wrong_endpoint_frame_and_repeated_identity_rejected(self):
        data = request()
        data["frame_action_indices"][0, -1] = 31
        with self.assertRaises(ValueError):
            service.validate_request(data)
        data = request()
        data["episode_uids"][1] = data["episode_uids"][0]
        with self.assertRaises(ValueError):
            service.validate_request(data)

    def test_nonchronological_history_and_object_npz_rejected(self):
        data = request()
        data["frame_action_indices"][0, 1] = 0
        with self.assertRaises(ValueError):
            service.validate_request(data)
        data = request()
        data["instructions"] = data["instructions"].astype(object)
        payload = io.BytesIO()
        np.savez(payload, **data)
        with self.assertRaises(ValueError):
            service.decode_request(payload.getvalue())

    def test_future_frame_and_zero_batch_override_rejected(self):
        data = request()
        data["frame_action_indices"][0, 3] = 33
        with self.assertRaises(ValueError):
            service.validate_request(data)
        data = request()
        data["rm_batch_size"] = np.asarray(0)
        with self.assertRaises(ValueError):
            service.validate_request(data)


class BatchTests(unittest.TestCase):
    def test_explicit_bf16_cast_handles_custom_fp32_value_head(self):
        import torch
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16),
            torch.nn.Linear(4, 2, bias=False, dtype=torch.float32))
        hidden = torch.ones((2, 4), dtype=torch.bfloat16)
        original_backbone = model[0].weight.detach().clone()
        expected_head = model[1].weight.detach().to(torch.bfloat16).clone()
        with self.assertRaises(RuntimeError):
            model(hidden)
        converted = service.prepare_bf16_cpu_model(model, torch)
        output = converted(hidden)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.equal(model[0].weight, original_backbone))
        self.assertTrue(torch.equal(model[1].weight, expected_head))
        self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.bfloat16
                            for p in converted.parameters()))

    def test_bucketing_and_tail_restore_original_order(self):
        samples = [{"input_ids": np.zeros((1, n)), "image_grid_thw": np.ones((1, 8, 3))}
                   for n in (9, 12, 9, 9, 12, 9, 9)]
        buckets = service.bucket_indices(samples, 2)
        self.assertEqual(buckets, [[0, 2], [3, 5], [6], [1, 4]])
        output = service.restore_order(len(samples), [(rows, [f"row-{i}" for i in rows]) for rows in buckets])
        self.assertEqual(output, [f"row-{i}" for i in range(len(samples))])
        with self.assertRaises(RuntimeError):
            service.restore_order(2, [([0, 0], ["x", "y"])])
        with self.assertRaises(RuntimeError):
            service.restore_order(2, [([1], ["x"])])

    def test_tensor_collate_keeps_image_and_text_sample_order(self):
        import torch
        samples = []
        for row in range(3):
            samples.append({"input_ids": torch.full((1, 5), row, dtype=torch.long),
                "attention_mask": torch.ones((1, 5), dtype=torch.long),
                "pixel_values": torch.full((1, 8, 4), float(row)),
                "image_grid_thw": torch.full((1, 8, 3), row + 1, dtype=torch.long)})
        result = service.collate_prepared(samples, torch, "cpu")
        self.assertEqual(tuple(result["input_ids"].shape), (3, 5))
        self.assertEqual(tuple(result["pixel_values"].shape), (24, 4))
        self.assertEqual(result["pixel_values"][8:16, 0].tolist(), [1.] * 8)
        self.assertEqual(result["image_grid_thw"][16:, 0].tolist(), [3] * 8)
        samples[1]["input_ids"] = torch.zeros((1, 6))
        with self.assertRaises(ValueError):
            service.collate_prepared(samples, torch, "cpu")


class FakeTensor:
    def __init__(self, device="cuda"):
        self.device = SimpleNamespace(type=device)


def fake_runtime(fail_release=False):
    events = []
    class CUDA:
        def is_initialized(self): return True
        def synchronize(self, device): events.append("synchronize")
        def empty_cache(self): events.append("empty_cache")
        def memory_allocated(self, device): return 0
        def memory_reserved(self, device): return 0
        def max_memory_allocated(self, device): return 200
        def max_memory_reserved(self, device): return 300
    class Model:
        def __init__(self):
            self.param = FakeTensor()
            self.rope_deltas = FakeTensor()
        def parameters(self): return [self.param]
        def buffers(self): return []
        def modules(self): return [self]
        def to(self, device):
            events.append("model_to_" + device)
            if fail_release:
                raise RuntimeError("release failed")
            self.param.device.type = device
            return self
    scorer = service.RynnSuccessService.__new__(service.RynnSuccessService)
    scorer.torch = SimpleNamespace(cuda=CUDA(), Tensor=FakeTensor)
    scorer.device, scorer.model = "cuda:0", Model()
    scorer.lock = threading.Lock()
    scorer.healthy, scorer.is_offloaded = True, False
    scorer.completed_requests, scorer.last_error = 0, None
    scorer.last_memory, scorer.fingerprint = {}, {"sha256": "test-only"}
    scorer.gpu_identity, scorer.batch_size = {"physical_gpu": 4}, 16
    return scorer, events


class LifecycleTests(unittest.TestCase):
    def test_offload_receipt_follows_sync_clear_and_cpu_assertion(self):
        scorer, events = fake_runtime()
        result = scorer.offload()
        self.assertEqual(events, ["synchronize", "model_to_cpu", "synchronize", "empty_cache", "synchronize"])
        self.assertTrue(result["ok"])
        self.assertTrue(result["is_offloaded"])
        self.assertFalse(result["busy"])
        self.assertIsNone(scorer.model.rope_deltas)

    def test_release_failure_and_busy_do_not_report_success(self):
        scorer, _ = fake_runtime(fail_release=True)
        with self.assertRaises(RuntimeError):
            scorer.offload()
        self.assertFalse(scorer.healthy)
        self.assertFalse(scorer.is_offloaded)
        scorer, _ = fake_runtime()
        scorer.lock.acquire()
        try:
            with self.assertRaises(service.BusyError):
                scorer.offload()
        finally:
            scorer.lock.release()
        self.assertTrue(scorer.healthy)

    def test_request_batch_above_cap_rejected_before_onload(self):
        scorer, events = fake_runtime()
        data = request(1)
        data["rm_batch_size"] = np.asarray(32)
        with self.assertRaises(ValueError):
            scorer.infer(data)
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
