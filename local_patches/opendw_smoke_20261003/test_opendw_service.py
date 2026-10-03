"""CPU protocol tests; no model weights, CUDA initialization, or remote service."""

import io
import json
from pathlib import Path
import sys
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
from opendw_service import (
    ThreadingHTTPServer, clear_dw_runtime_caches, decode_request,
    encode_response, make_handler, set_dw_device,
)
from opendw_action_telemetry import action_telemetry


def request_payload(batch=2):
    buffer = io.BytesIO()
    np.savez(buffer, images=np.zeros((batch, 384, 320, 3), dtype=np.uint8),
             actions=np.zeros((batch, 32, 14), dtype=np.float32),
             states=np.zeros((batch, 14), dtype=np.float32), seeds=np.arange(batch),
             instructions=np.asarray(["adjust the bottle"] * batch))
    return buffer.getvalue()


class FakeLog:
    def event(self, *args, **kwargs):
        pass


class FakeBackend:
    def __init__(self):
        self.is_offloaded = True
        self.fatal_error = None
        self.log = FakeLog()

    def health(self):
        return {"ok": True, "is_offloaded": self.is_offloaded}

    def onload(self):
        self.is_offloaded = False
        return self.health()

    def offload(self):
        self.is_offloaded = True
        return self.health()

    def infer(self, data):
        self.onload()
        return encode_response(data["images"], np.full((len(data["images"]), 8), 0.2, dtype=np.float32))


class ServiceTests(unittest.TestCase):
    def test_telemetry_uses_separate_parsed_stats_and_does_not_modify_inputs(self):
        actions = np.zeros((32, 14), dtype=np.float32)
        actions[0, 0] = 6.0
        actions[1, 0] = -6.0
        state = np.full(14, 10.0, dtype=np.float32)
        original_actions, original_state = actions.copy(), state.copy()
        stats = {"action": {"mean": np.zeros(14), "std": np.ones(14)},
                 "state": {"mean": np.full(14, 10.0), "std": np.ones(14)}}
        result = action_telemetry(actions, state, norm_stats=stats,
                                  normalization_mode="zscore_14d", action_condition_mode="absolute")
        measured = result["wm_normalization"]["actions"]
        self.assertEqual(measured["abs_z_gt5_count"], 2)
        self.assertEqual(measured["abs_z_gt5_count_by_dim"], [2] + [0] * 13)
        self.assertAlmostEqual(measured["abs_z_gt5_fraction"], 2 / (32 * 14))
        self.assertAlmostEqual(measured["raw_clip_difference_max_by_dim"][0], 1.0)
        self.assertEqual(result["wm_normalization"]["state"]["abs_z_gt5_count"], 0)
        self.assertEqual(result["adjacent_command_abs_delta"]["max_by_dim"][0], 12.0)
        np.testing.assert_array_equal(actions, original_actions)
        np.testing.assert_array_equal(state, original_state)
        json.dumps(result, allow_nan=False)

    def test_telemetry_does_not_guess_unparsed_stat_schema(self):
        result = action_telemetry(np.zeros((32, 14)), np.zeros(14),
                                  norm_stats={"action": {"default": {"global_mean": [0] * 14}}},
                                  normalization_mode="zscore_14d", action_condition_mode="absolute")
        self.assertFalse(result["wm_normalization"]["actions"]["available"])
        self.assertFalse(result["wm_normalization"]["state"]["available"])
        self.assertEqual(result["raw_actions"]["max"], 0.0)

    def test_npz_roundtrip_shapes_and_probabilities(self):
        data = decode_request(request_payload())
        encoded = encode_response(data["images"], np.full((2, 8), 0.4), [1, 2])
        with np.load(io.BytesIO(encoded), allow_pickle=False) as result:
            self.assertEqual(result["next_images"].shape, (2, 384, 320, 3))
            self.assertEqual(result["scores"].dtype, np.float32)
            self.assertEqual(result["timing_s"].tolist(), [1, 2])

    def test_bad_raw_action_is_rejected_before_backend(self):
        data = decode_request(request_payload())
        data["actions"][0, 0, 6] = 1.1
        buffer = io.BytesIO()
        np.savez(buffer, **data)
        with self.assertRaisesRegex(ValueError, "clip gripper"):
            decode_request(buffer.getvalue())

    def test_device_cache_is_updated_and_vae_cache_cleared(self):
        class Model:
            def __init__(self):
                self.device = "cpu"
                self.vae = type("VAE", (), {})()
                self.vae.model = type("VAEInner", (), {"clear_cache": lambda s: setattr(s, "cleared", True)})()
            def to(self, device):
                self.moved_to = device
        model = Model()
        set_dw_device(model, "cuda:0")
        self.assertEqual(model.device, "cuda:0")
        self.assertEqual(model.moved_to, "cuda:0")
        clear_dw_runtime_caches(model)
        self.assertTrue(model.vae.model.cleared)

    def test_loopback_http_health_infer_offload_and_bad_request(self):
        backend = FakeBackend()
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(backend))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        address = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(address + "/health") as response:
                self.assertTrue(json.load(response)["is_offloaded"])
            infer = urllib.request.Request(address + "/infer", data=request_payload(), method="POST")
            with urllib.request.urlopen(infer) as response:
                with np.load(io.BytesIO(response.read()), allow_pickle=False) as result:
                    self.assertEqual(result["scores"].shape, (2, 8))
            control = urllib.request.Request(address + "/offload", data=b"{}", method="POST")
            with urllib.request.urlopen(control) as response:
                self.assertTrue(json.load(response)["is_offloaded"])
            bad = urllib.request.Request(address + "/onload", data=b'{"unexpected":1}', method="POST")
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad)
            self.assertEqual(error.exception.code, 400)
            self.assertIsNone(backend.fatal_error)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
