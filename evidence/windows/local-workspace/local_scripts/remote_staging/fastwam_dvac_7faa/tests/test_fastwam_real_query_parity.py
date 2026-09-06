from __future__ import annotations

import ast
import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "experiments" / "robotwin" / "fastwam_real_query_parity.py"
MODEL = ROOT / "src" / "fastwam" / "models" / "wan22" / "fastwam.py"


class FastWAMRealQueryParityTest(unittest.TestCase):
    def test_generator_hook_is_opt_in(self) -> None:
        tree = ast.parse(MODEL.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "infer_action"
        )
        defaults = dict(zip([arg.arg for arg in method.args.args[-len(method.args.defaults) :]], method.args.defaults))
        self.assertIsInstance(defaults["action_generator"], ast.Constant)
        self.assertIsNone(defaults["action_generator"].value)

    def test_global_rng_snapshot_round_trip(self) -> None:
        spec = importlib.util.spec_from_file_location("fastwam_real_query_parity", HARNESS)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        initial = module._rng_snapshot()
        random.random()
        np.random.random()
        torch.rand(3)
        module._rng_restore(initial)
        restored = module._rng_snapshot()
        self.assertTrue(module._rng_equal(initial, restored))
        self.assertEqual(module._rng_digest(initial), module._rng_digest(restored))


if __name__ == "__main__":
    unittest.main()
