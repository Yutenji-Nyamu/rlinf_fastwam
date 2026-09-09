# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU HTTP client and immutable admission cache for the frozen value scorer."""

import base64
import hashlib
import io
import json
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import torch
from PIL import Image


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class RynnValueClient:
    def __init__(self, endpoint, cache_path, expected_revision, robot_description,
                 camera_description, timeout_seconds=900):
        parsed = urlsplit(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("The frozen scorer must be a local HTTP service.")
        self.endpoint = endpoint.rstrip("/")
        self.cache_path = Path(cache_path)
        self.timeout = float(timeout_seconds)
        self.robot_description = str(robot_description)
        self.camera_description = str(camera_description)
        # Explicitly bypass any system proxy for the local service.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        health = self.request("/health")
        if health.get("ok") is not True:
            raise ValueError("Frozen scorer is not healthy.")
        self.fingerprint = health["fingerprint"]
        if self.fingerprint["revision"] != expected_revision:
            raise ValueError("Scorer checkpoint revision mismatch.")
        self.identity = {"version": 1, "fingerprint": self.fingerprint,
                         "robot_description": self.robot_description,
                         "camera_description": self.camera_description,
                         "signal": "remaining_seconds_before_minus_after"}
        self.identity_hash = canonical_hash(self.identity)

    def request(self, route, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.endpoint + route, data=data,
                                         headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _png(image):
        image = torch.as_tensor(image).detach().cpu()
        if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("RynnValue requires raw uint8 HWC RGB images.")
        stream = io.BytesIO()
        Image.fromarray(image.numpy(), mode="RGB").save(stream, format="PNG")
        return base64.b64encode(stream.getvalue()).decode("ascii")

    def score_episode(self, episode):
        if not episode:
            raise ValueError("Cannot score an empty episode.")
        keys = [bytes(row["rabc_episode_key"].tolist()).hex() for row in episode]
        if len(set(keys)) != 1:
            raise ValueError("Mixed episode keys in scoring request.")
        instructions = [bytes(row["rabc_task_utf8"][:int(row["rabc_task_length"])].tolist()).decode("utf8")
                        for row in episode]
        if len(set(instructions)) != 1 or not instructions[0].strip():
            raise ValueError("Missing or changing task instruction within episode.")
        if [int(row["query_idx"]) for row in episode] != list(range(len(episode))):
            raise ValueError("Scoring requires consecutive query boundaries starting at zero.")
        for previous, current in zip(episode, episode[1:]):
            if not torch.equal(previous["rabc_post_image"], current["rabc_pre_image"]):
                raise ValueError("Raw pre/post query image boundaries do not align.")
        frames = [episode[0]["rabc_pre_image"]] + [row["rabc_post_image"] for row in episode]
        body = {"episode_id": keys[0], "instruction": instructions[0],
                "robot_description": self.robot_description,
                "camera_description": self.camera_description,
                "frames_png_b64": [self._png(frame) for frame in frames]}
        request_hash = canonical_hash(body)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        path = self.cache_path / (keys[0] + ".json")
        if path.exists():
            cached = json.loads(path.read_text())
            if cached["identity"] != self.identity or cached["request_hash"] != request_hash:
                raise ValueError("Frozen scorer cache identity/content mismatch.")
            result = cached["result"]
        else:
            result = self.request("/score", body)
        if result["fingerprint"] != self.fingerprint:
            raise ValueError("Scorer identity changed during admission.")
        if result.get("ok") is not True or result.get("episode_id") != keys[0]:
            raise ValueError("Scorer result failed or episode identity mismatch.")
        values = torch.tensor(result["remaining_seconds"], dtype=torch.float64)
        deltas = torch.tensor(result["delta_seconds"], dtype=torch.float64)
        if values.shape != (len(episode) + 1,) or deltas.shape != (len(episode),):
            raise ValueError("Scorer returned misaligned query values.")
        if not torch.isfinite(values).all() or not torch.isfinite(deltas).all():
            raise ValueError("Scorer returned nonfinite values.")
        if (values < 0).any() or (values > 512).any():
            raise ValueError("Raw remaining seconds outside official decoded range.")
        if not torch.allclose(values[:-1] - values[1:], deltas, atol=1e-6, rtol=1e-6):
            raise ValueError("Scorer delta sign or alignment mismatch.")
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            with temporary.open("x") as stream:
                json.dump({"identity": self.identity, "request_hash": request_hash,
                           "instruction": instructions[0], "result": result}, stream, indent=2)
            temporary.replace(path)
        scored = []
        identity_bytes = torch.tensor(list(bytes.fromhex(self.identity_hash)), dtype=torch.uint8)
        for i, row in enumerate(episode):
            record = {key: value for key, value in row.items()
                      if key not in ("rabc_pre_image", "rabc_post_image")}
            record.update(rabc_delta_seconds=deltas[i].clone(),
                          rabc_value_before=values[i].clone(), rabc_value_after=values[i + 1].clone(),
                          rabc_identity=identity_bytes.clone())
            scored.append(record)
        return scored, deltas, keys[0]
