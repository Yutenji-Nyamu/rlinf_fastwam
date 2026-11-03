# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Ensure planned simulator replacement preserves reset request and is bounded."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinf.envs.robodojo import robodojo_env as adapter


class Connection:
    def __init__(self, result, sent):
        self.result, self.sent = result, sent

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def send(self, request):
        self.sent.append(request)

    def poll(self, _):
        return True

    def recv(self):
        return self.result


class RecycleTests(unittest.TestCase):
    def env(self):
        env = adapter.RoboDojoEnv.__new__(adapter.RoboDojoEnv)
        env.cfg, env.gpu, env.mode = {"reset_timeout": 10}, 6, "train"
        env.socket, env._key = "test-socket", b"test-only"
        return env

    def test_reset_survives_clean_process_replacement(self):
        sent = []
        responses = [
            Connection({"restart_native": True}, sent),
            FileNotFoundError(),
            Connection({"ok": True, "value": "same-request-result"}, sent),
        ]
        with (
            patch.object(adapter, "Client", side_effect=responses),
            patch.object(adapter.time, "sleep"),
        ):
            result = self.env()._rpc("reset", layout_ids=[34] * 8, group_size=8)
        self.assertEqual(result, "same-request-result")
        self.assertEqual(sent[0], sent[1])
        self.assertEqual(sent[1]["layout_ids"], [34] * 8)

    def test_repeated_reset_restart_is_bounded(self):
        with (
            patch.object(
                adapter,
                "Client",
                side_effect=lambda *_a, **_k: Connection({"restart_native": True}, []),
            ),
            patch.object(adapter.time, "sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "restart limit"):
                self.env()._rpc("reset", layout_ids=[34] * 8, group_size=8)


if __name__ == "__main__":
    unittest.main()
