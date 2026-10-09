"""Focused CPU checks for EXPO historical U/Norm integration (server only)."""

import copy
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as arrow
import pyarrow.parquet as pq
import torch
from PIL import Image
from test_expo_core import batch, learner

from rlinf.algorithms.expo_ft.formal_replay import IMAGE_COLUMNS, FormalReplay
from rlinf.algorithms.expo_ft.signals import SignalWeighting, validate_trace
from rlinf.algorithms.norm_signal import capture_expert_norm
from rlinf.algorithms.ugrow_signal import compute_ugrow_signal


def trace(n=60):
    return dict(
        raw=torch.arange(1, n + 1).float(),
        query=torch.arange(n) // 10,
        position=torch.arange(n) % 10,
        parent=(torch.arange(n) // 10) % 8,
        edited=torch.zeros(n, dtype=torch.bool),
        base_version=torch.zeros(n, dtype=torch.long),
    )


def weighting(kind="ugrow_10_5", target="both", **kwargs):
    return SignalWeighting(
        dict(kind=kind, target=target, dropout=False, anneal=False, **kwargs)
    )


class SignalChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_two_levels_and_neutral_demo_global_domain(self):
        raw = torch.exp(torch.linspace(-4, 4, 200).reshape(4, 50))
        valid = torch.ones_like(raw, dtype=torch.bool)
        valid[0] = False
        raw[0] = float("nan")
        w, m = weighting().weights(
            dict(signal_raw=raw, signal_valid=valid),
            "fm",
            completed_episodes=1,
            update_step=0,
        )
        torch.testing.assert_close(w[0], torch.ones(50))
        self.assertFalse(w.requires_grad)
        self.assertGreater(float(w[1:].std()), 0)
        self.assertGreater(float(w[1].std()), 0)
        self.assertGreater(float(w[1:].mean(-1).std()), 0)
        torch.testing.assert_close(w[1:].mean(), torch.tensor(1.0))
        only, _ = weighting().weights(
            dict(signal_raw=raw[1:], signal_valid=valid[1:]),
            "fm",
            completed_episodes=1,
            update_step=0,
        )
        torch.testing.assert_close(only, w[1:])
        self.assertEqual(m["eligible_windows"], 3)

    def test_editor_chunk_only_and_all_demos_neutral(self):
        raw = torch.arange(1, 41).float().reshape(4, 10)
        valid = torch.ones_like(raw, dtype=torch.bool)
        s = weighting()
        w, m = s.weights(
            dict(signal_raw=raw, signal_valid=valid),
            "editor",
            completed_episodes=1,
            update_step=0,
        )
        self.assertEqual(tuple(w.shape), (4,))
        self.assertGreater(float(w.std()), 0)
        self.assertEqual(m["alpha_local"], 0)
        w, _ = s.weights(
            dict(signal_raw=raw, signal_valid=~valid),
            "editor",
            completed_episodes=1,
            update_step=0,
        )
        torch.testing.assert_close(w, torch.ones(4))

    def test_r200_dropout_private_rng_and_resume_counter(self):
        cfg = dict(
            kind="norm_residual_t5_l3",
            target="both",
            dropout=True,
            dropout_probability=0.2,
            anneal=True,
        )
        s = SignalWeighting(cfg)
        raw = torch.arange(1, 3201).float().reshape(64, 50)
        b = dict(signal_raw=raw, signal_valid=torch.ones_like(raw, dtype=torch.bool))
        tr = torch.get_rng_state().clone()
        py = random.getstate()
        a, ma = s.weights(b, "fm", completed_episodes=30, update_step=17)
        resumed, mr = SignalWeighting(cfg).weights(
            b, "fm", completed_episodes=30, update_step=17
        )
        torch.testing.assert_close(a, resumed)
        self.assertEqual(ma, mr)
        self.assertGreater(ma["dropout_windows"], 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), tr))
        self.assertEqual(random.getstate(), py)
        z, m = s.weights(b, "fm", completed_episodes=200, update_step=18)
        torch.testing.assert_close(z, torch.ones_like(z))
        self.assertEqual(m["alpha_local"], 0)
        self.assertEqual(m["alpha_chunk"], 0)
        cfg["dropout_probability"] = 1
        z, _ = SignalWeighting(cfg).weights(
            b, "fm", completed_episodes=1, update_step=18
        )
        torch.testing.assert_close(z, torch.ones_like(z))

    def test_independent_targets_and_invalid_contracts(self):
        self.assertEqual(
            SignalWeighting().weights(None, "fm", completed_episodes=0, update_step=0),
            (None, {}),
        )
        self.assertFalse(weighting(target="fm").active("editor"))
        self.assertFalse(weighting(target="editor").active("fm"))
        for cfg in (
            dict(kind="x", target="fm"),
            dict(kind="ugrow_10_5", target="fm", temperature_chunk=0),
            dict(kind="ugrow_10_5", target="fm", oops=1),
        ):
            with self.assertRaises(ValueError):
                SignalWeighting(cfg)
        t = trace()
        validate_trace(t, 60)
        t["parent"][3] = 2
        with self.assertRaises(ValueError):
            validate_trace(t, 60)

    def test_fm_global_microbatch_gradient_and_unit_equivalence(self):
        s = weighting()
        raw = torch.arange(1, 201).float().reshape(4, 50)
        w, _ = s.weights(
            dict(signal_raw=raw, signal_valid=torch.ones_like(raw, dtype=torch.bool)),
            "fm",
            completed_episodes=1,
            update_step=0,
        )
        x = torch.randn(4, 50, 32)
        a = torch.nn.Parameter(torch.ones(32))
        b = torch.nn.Parameter(a.detach().clone())
        ((x * a).square() * w[..., None]).mean().backward()
        for start, end in ((0, 1), (1, 3), (3, 4)):
            (
                ((x[start:end] * b).square() * w[start:end, :, None]).mean()
                * ((end - start) / 4)
            ).backward()
        torch.testing.assert_close(a.grad, b.grad)
        # Actual adapter reduction, replacing only the native FM producer.
        from types import SimpleNamespace
        from unittest.mock import patch

        from openpi.models import model as om
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

        from rlinf.algorithms.expo_ft.backend import _NativeBatchAdapter

        dummy = torch.nn.Linear(1, 1)
        dummy.config = SimpleNamespace(use_rlt=False)
        errors = torch.rand(4, 50, 32, requires_grad=True)
        with (
            patch.object(om.Observation, "from_dict", return_value=None),
            patch.object(PI0Pytorch, "forward", return_value=errors),
        ):
            adapter = _NativeBatchAdapter(dummy)
            actual = adapter({}, torch.zeros_like(errors), "fm", w)
            torch.testing.assert_close(
                actual, (errors * w[..., None]).mean().reshape(1)
            )
            torch.testing.assert_close(
                adapter({}, torch.zeros_like(errors), "fm", torch.ones_like(w)),
                adapter({}, torch.zeros_like(errors), "fm"),
            )

    def test_full_editor_objective_gradient_alpha_and_critic_unchanged(self):
        a = learner()
        b = copy.deepcopy(a)
        clean = copy.deepcopy(a)
        data = batch(a.config)
        weights = torch.tensor([0.6, 0.8, 1.1, 1.5])
        before = [p.detach().clone() for p in a.critic.parameters()]
        a._editor_temperature_update(data, weights)
        clean._editor_temperature_update(data)
        obs = b._observation(data["obs"])
        ref = data["actions"]
        b.critic.requires_grad_(False)
        with torch.no_grad():
            features = b._features(obs)
        delta, lp = b._sample_editor(features, obs["proprio"], ref)
        q = b._forward(b.critic, features, obs["proprio"], ref + delta).mean(0)
        (
            (b.config.entropy_scale * b.temperature.detach() * lp - q) * weights
        ).mean().backward()
        for pa, pb in zip(a.editor.parameters(), b.editor.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad)
        for old, p in zip(before, a.critic.parameters()):
            torch.testing.assert_close(old, p)
        torch.testing.assert_close(a.log_temperature, clean.log_temperature)

    def test_editor_microbatch_weights_and_unit_compatibility(self):
        a = learner()
        b = copy.deepcopy(a)
        data = batch(a.config)
        b.config = copy.copy(b.config)
        object.__setattr__(b.config, "editor_microbatch_size", 2)
        # randn changes its consumption for different shapes; pin the same
        # per-row epsilon to test reduction/gradient equivalence, not RNG layout.
        epsilon = torch.randn(
            4, a.config.flat_action_dim, generator=torch.Generator().manual_seed(9)
        )

        def fixed_sampler(agent):
            cursor = 0

            def sample(features, proprio, reference):
                nonlocal cursor
                count = reference.shape[0]
                eps = epsilon[cursor : cursor + count]
                cursor += count
                return agent.editor.sample(
                    features, proprio, reference, None, epsilon=eps
                )

            return sample

        a._sample_editor = fixed_sampler(a)
        b._sample_editor = fixed_sampler(b)
        weights = torch.tensor([0.6, 0.8, 1.1, 1.5])
        a._editor_temperature_update(data, weights)
        b._editor_temperature_update(data, weights)
        for pa, pb in zip(a.editor.parameters(), b.editor.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-5, atol=2e-6)
        a = learner()
        b = copy.deepcopy(a)
        a._editor_temperature_update(data)
        b._editor_temperature_update(data, torch.ones(4))
        for pa, pb in zip(a.editor.parameters(), b.editor.parameters()):
            torch.testing.assert_close(pa, pb)

    def test_norm_replica_hooks_are_isolated_and_removed(self):
        from types import SimpleNamespace

        base = [torch.nn.Identity() for _ in range(3)]
        replica = [layer._replicate_for_data_parallel() for layer in base]
        wrap = lambda layers: SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(
                gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=layers))
            )
        )
        x = torch.ones(2, 50, 4)
        with capture_expert_norm(wrap(base), 50) as a:
            with capture_expert_norm(wrap(replica), 50) as b:
                for layer in base:
                    layer(x)
                for layer in replica:
                    layer(2 * x)
        self.assertEqual(len(a), 3)
        self.assertEqual(len(b), 3)
        torch.testing.assert_close(b[0], 2 * a[0])
        self.assertTrue(all(not l._forward_hooks for l in base + replica))
        with self.assertRaises(RuntimeError):
            with capture_expert_norm(wrap(base), 50):
                raise RuntimeError("stop")
        self.assertTrue(all(not l._forward_hooks for l in base))

    def test_ugrow_only_physical_dimensions(self):
        a = torch.ones(2, 50, 32)
        b = -a
        b[..., 14:] = 100
        u = compute_ugrow_signal(a, b)
        torch.testing.assert_close(u, torch.ones(2, 50))
        self.assertFalse(u.requires_grad)


class ReplaySignalChecks(unittest.TestCase):
    def test_real_index_alignment_sampling_cache_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            demo = root / "demo"
            (demo / "data/chunk-000").mkdir(parents=True)
            (demo / "meta").mkdir()
            image = io.BytesIO()
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
                image, format="PNG"
            )
            data = {key: [{"bytes": image.getvalue()}] * 60 for key in IMAGE_COLUMNS}
            data.update(
                {
                    "action": [[0.0] * 14] * 60,
                    "observation.state": [[0.0] * 14] * 60,
                    "task_index": [0] * 60,
                }
            )
            pq.write_table(
                arrow.table(data), demo / "data/chunk-000/episode_000000.parquet"
            )
            (demo / "prepared.json").write_text(
                json.dumps(
                    dict(
                        complete=True, dataset_revision="fixture", episodes=1, frames=60
                    )
                )
            )
            (demo / "meta/tasks.jsonl").write_text(
                json.dumps(dict(task_index=0, task="fixture"))
            )
            c = weighting().contract
            pool = FormalReplay(root / "pool", demo, signal_contract=c)
            obs = dict(
                main_images=torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
                wrist_images=torch.zeros(1, 2, 8, 8, 3, dtype=torch.uint8),
                states=torch.zeros(1, 14),
                task_descriptions=["fixture"],
            )
            pool.append_episode(
                [obs] * 60,
                torch.zeros(60, 14),
                [0.0] * 59 + [1.0],
                True,
                False,
                True,
                "online-000000",
                obs,
                signal_trace=trace(),
            )
            for horizon, start in ((10, 7), (50, 3)):
                wins = pool._windows([(1, start), (0, start)], horizon)
                torch.testing.assert_close(
                    wins[0]["signal_raw"],
                    torch.arange(start + 1, start + horizon + 1).float(),
                )
                self.assertTrue(wins[0]["signal_valid"].all())
                self.assertFalse(wins[1]["signal_valid"].any())
            # Cache hit must not re-validate/hash the whole episode.
            from unittest.mock import patch

            with patch(
                "rlinf.algorithms.expo_ft.signals.validate_trace",
                side_effect=AssertionError("revalidated"),
            ):
                pool._windows([(1, 1)], 50)
            saved = pool.state_dict()
            refs = pool._draw(16)
            restored = FormalReplay(root / "pool", demo, signal_contract=c)
            restored.load_state_dict(saved)
            self.assertEqual(refs, restored._draw(16))
            restored.sample_fm(64)
            self.assertEqual(restored.last_fm_signal["signal_raw"].shape, (64, 50))
            with self.assertRaises(ValueError):
                FormalReplay(root / "pool", demo)
            with self.assertRaises(ValueError):
                FormalReplay(
                    root / "pool",
                    demo,
                    signal_contract=weighting(kind="norm_residual_t5_l3").contract,
                )


if __name__ == "__main__":
    unittest.main()
