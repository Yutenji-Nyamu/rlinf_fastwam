"""Local HTTP contract tests: no real model, GPU, Ray or remote connection."""

import base64
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import threading
import urllib.error

import pytest
import torch
from PIL import Image


SOURCE = Path(__file__).resolve().parents[1] / "rlinf/utils/rynnvalue_client.py"
SPEC = importlib.util.spec_from_file_location("standalone_rynnvalue_client", SOURCE)
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)
REVISION = "8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e"
FINGERPRINT = {"model_id": "Alibaba-DAMO-Academy/RynnValue-8B", "revision": REVISION,
               "fingerprint_sha256": "fake-test-only"}


@pytest.fixture
def fake_service():
    state = {"score_calls": 0, "health_ok": True, "status": 200,
             "mutate": lambda result: result, "bodies": []}

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            assert self.path == "/health"
            self.reply(200, {"ok": state["health_ok"], "fingerprint": copy.deepcopy(FINGERPRINT)})

        def do_POST(self):
            assert self.path == "/score"
            state["score_calls"] += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["bodies"].append(body)
            result = {"ok": True, "episode_id": body["episode_id"],
                      "remaining_seconds": [9.0, 7.0, 8.0], "delta_seconds": [2.0, -1.0],
                      "fingerprint": copy.deepcopy(FINGERPRINT)}
            self.reply(state["status"], state["mutate"](result))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["endpoint"] = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def client(state, path, **kwargs):
    return CLIENT.RynnValueClient(
        state["endpoint"], path, kwargs.get("revision", REVISION),
        kwargs.get("robot", "dual arm robot"), "main RGB camera", timeout_seconds=5,
    )


def episode():
    task = b"place the object"
    text = torch.zeros(4096, dtype=torch.uint8)
    text[:len(task)] = torch.tensor(list(task), dtype=torch.uint8)
    images = [torch.full((2, 3, 3), value, dtype=torch.uint8) for value in (10, 20, 30)]
    return [{"rabc_episode_key": torch.arange(16, dtype=torch.uint8),
             "rabc_task_utf8": text.clone(), "rabc_task_length": torch.tensor(len(task)),
             "query_idx": torch.tensor(index), "action": torch.full((14,), float(index)),
             "rabc_pre_image": images[index].clone(), "rabc_post_image": images[index + 1].clone()}
            for index in range(2)]


def overwrite(**changes):
    def mutate(result):
        result.update(changes)
        return result
    return mutate


def test_remaining_seconds_sign_final_boundary_and_persistent_cache(fake_service, tmp_path):
    model = client(fake_service, tmp_path)
    source = episode()
    scored, deltas, key = model.score_episode(source)
    assert deltas.tolist() == [2.0, -1.0]
    assert [row["rabc_value_before"].item() for row in scored] == [9.0, 7.0]
    assert [row["rabc_value_after"].item() for row in scored] == [7.0, 8.0]
    assert key == bytes(range(16)).hex()
    assert "rabc_pre_image" not in scored[0] and "rabc_post_image" not in scored[1]
    assert "rabc_pre_image" in source[0] and "rabc_post_image" in source[1]
    assert bytes(scored[0]["rabc_identity"].tolist()).hex() == model.identity_hash
    encoded = fake_service["bodies"][0]["frames_png_b64"]
    assert len(encoded) == 3
    for frame, expected in zip(encoded, (10, 20, 30)):
        with Image.open(io.BytesIO(base64.b64decode(frame))) as image:
            assert image.mode == "RGB" and image.getpixel((0, 0)) == (expected,) * 3
    # A new client process identity can reuse the exact immutable admitted result.
    restored = client(fake_service, tmp_path)
    restored_rows, restored_deltas, restored_key = restored.score_episode(source)
    assert restored_key == key and torch.equal(deltas, restored_deltas)
    assert torch.equal(restored_rows[0]["rabc_identity"], scored[0]["rabc_identity"])
    assert fake_service["score_calls"] == 1


@pytest.mark.parametrize("changes", [
    {"remaining_seconds": [9.0, 7.0]},
    {"delta_seconds": [-2.0, 1.0]},
    {"remaining_seconds": [9.0, float("nan"), 8.0]},
    {"remaining_seconds": [513.0, 7.0, 8.0], "delta_seconds": [506.0, -1.0]},
    {"fingerprint": {**FINGERPRINT, "revision": "different-checkpoint"}},
    {"episode_id": "different-episode"},
    {"ok": False},
])
def test_invalid_score_never_becomes_cached_training_data(fake_service, tmp_path, changes):
    model = client(fake_service, tmp_path)
    fake_service["mutate"] = overwrite(**changes)
    with pytest.raises((ValueError, RuntimeError)):
        model.score_episode(episode())
    assert not list(tmp_path.glob("*.json"))


def test_http_failure_does_not_substitute_scores(fake_service, tmp_path):
    model = client(fake_service, tmp_path)
    fake_service["status"] = 500
    with pytest.raises(urllib.error.HTTPError):
        model.score_episode(episode())
    assert not list(tmp_path.glob("*.json"))


def test_unhealthy_or_wrong_revision_fails_before_collection(fake_service, tmp_path):
    with pytest.raises(ValueError):
        client(fake_service, tmp_path, revision="wrong-revision")
    fake_service["health_ok"] = False
    with pytest.raises((ValueError, RuntimeError)):
        client(fake_service, tmp_path)
    assert fake_service["score_calls"] == 0


@pytest.mark.parametrize("change", ["boundary", "query_order", "task", "episode_key"])
def test_inconsistent_episode_rejected_before_http(fake_service, tmp_path, change):
    model = client(fake_service, tmp_path)
    rows = episode()
    if change == "boundary":
        rows[1]["rabc_pre_image"][0, 0, 0] += 1
    elif change == "query_order":
        rows[1]["query_idx"] = torch.tensor(3)
    elif change == "task":
        rows[1]["rabc_task_utf8"][0] = ord("x")
    else:
        rows[1]["rabc_episode_key"][0] += 1
    with pytest.raises(ValueError):
        model.score_episode(rows)
    assert fake_service["score_calls"] == 0


@pytest.mark.parametrize("changed", ["request", "identity", "cached_delta"])
def test_cache_mismatch_fails_without_rescoring(fake_service, tmp_path, changed):
    model = client(fake_service, tmp_path)
    rows = episode()
    _, _, key = model.score_episode(rows)
    if changed == "request":
        rows[-1]["rabc_post_image"][0, 0, 0] += 1
    else:
        path = tmp_path / f"{key}.json"
        cached = json.loads(path.read_text())
        if changed == "identity":
            cached["identity"]["robot_description"] = "another embodiment"
        else:
            cached["result"]["delta_seconds"][0] *= -1
        path.write_text(json.dumps(cached))
    with pytest.raises(ValueError):
        model.score_episode(rows)
    assert fake_service["score_calls"] == 1
