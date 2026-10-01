"""Server-only high-information checks; no VLA weights / downloads required.

Run from deployed source: python -m unittest discover -s tests -p test_expo_core.py -v
The real ResNet / real VLA / real RoboTwin rollout belong to the GPU smoke, not
this small deterministic logic suite. Windows does not execute project tests.
"""

import copy
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn


CORE_PATH = Path(__file__).resolve().parents[1] / "rlinf/algorithms/expo_ft/core.py"
SPEC = importlib.util.spec_from_file_location("expo_core_under_test", CORE_PATH)
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


class TinyVision(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.project = nn.Linear(3, output_dim)

    def forward(self, images):
        pixels = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
        return torch.tanh(self.project(pixels.mean(dim=(1, 3, 4))))


def config(**overrides):
    values = dict(chunk_length=2, action_dim=3, proprio_dim=3, num_views=2,
                  image_size=8, image_latent_dim=12, proprio_latent_dim=4,
                  hidden_dims=(16, 16, 16), critic_updates=20)
    values.update(overrides)
    return core.ExpoConfig(**values)


def learner(cfg=None, seed=17):
    cfg = config() if cfg is None else cfg
    return core.ExpoLearner(cfg, seed=seed, vision_encoder=TinyVision(cfg.image_latent_dim))


def observation(cfg, batch_size=4):
    generator = torch.Generator().manual_seed(913)
    return {"images": torch.randint(0, 255, (batch_size, cfg.num_views, 3, 8, 8),
                                    generator=generator, dtype=torch.uint8),
            "proprio": torch.randn(batch_size, cfg.proprio_dim, generator=generator)}


def batch(cfg):
    obs = observation(cfg)
    generator = torch.Generator().manual_seed(417)
    return {"obs": obs, "next_obs": observation(cfg),
            "actions": torch.randn(4, cfg.chunk_length, cfg.action_dim, generator=generator) * .2,
            "rewards": torch.tensor([1., 0., .5, -.1]),
            "continuations": torch.tensor([0., 1., 0., 1.]),
            "executed_steps": torch.tensor([1, 2, 2, 1]), "valids": torch.ones(4)}


def base_sampler(cfg):
    def sample(obs):
        b = obs["proprio"].shape[0]
        values = torch.linspace(-.3, .4, cfg.n_base).view(1, -1, 1, 1)
        return values.expand(b, -1, cfg.chunk_length, cfg.action_dim).clone()
    return sample


def snapshots(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def changed(before, module):
    return any(not torch.equal(before[name], value) for name, value in module.named_parameters())


@unittest.skipIf(os.name == "nt", "Project tests execute only on an authorized server")
class ExpoCoreChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_defaults_and_action_dimensions(self):
        cfg = core.ExpoConfig()
        self.assertEqual((cfg.n_base, cfg.n_edit, cfg.num_qs, cfg.num_min_qs), (8, 8, 10, 2))
        self.assertEqual((cfg.chunk_length, cfg.action_dim, cfg.flat_action_dim), (10, 14, 140))
        self.assertEqual(cfg.resolved_target_entropy, -70)
        self.assertEqual(cfg.critic_updates, 20)

    def test_candidates_bounded_delta_and_random_target_pair(self):
        agent = learner()
        cfg = agent.config
        obs = observation(cfg)
        base = base_sampler(cfg)(obs) + 1.7  # Combined actions must remain unsquashed.
        selection = agent.select_actions(obs, base)
        self.assertEqual(selection["candidate_actions"].shape, (4, 16, 2, 3))
        torch.testing.assert_close(selection["candidate_actions"][:, :8], base)
        delta = selection["candidate_actions"][:, 8:] - base
        self.assertLessEqual(float(delta.abs().max()), cfg.edit_scale + 1e-6)
        self.assertGreater(float(selection["candidate_actions"].min()), 1.0)
        pair = selection["selection_q_indices"]
        self.assertEqual(len(pair.unique()), 2)
        with torch.no_grad():
            features = agent._features(agent._observation(obs))
            expected = agent.target_critic(features, obs["proprio"], selection["candidate_actions"], pair).min(0).values
        torch.testing.assert_close(selection["scores"], expected)
        torch.testing.assert_close(selection["index"], expected.argmax(1))
        rows = torch.arange(4)
        torch.testing.assert_close(selection["actions"], selection["candidate_actions"][rows, selection["index"]])

    def test_editor_scaled_tanh_log_probability(self):
        agent = learner()
        cfg = agent.config
        obs = agent._observation(observation(cfg))
        with torch.no_grad():
            for parameter in agent.editor.parameters():
                parameter.zero_()
            reference = torch.zeros(4, 2, 3)
            delta, log_prob = agent.editor.sample(agent._features(obs), obs["proprio"], reference,
                                                 agent.generator, deterministic=True)
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        expected = cfg.flat_action_dim * (-.5 * torch.log(torch.tensor(2 * torch.pi))
                                         - torch.log(torch.tensor(cfg.edit_scale)))
        torch.testing.assert_close(log_prob, expected.expand(4))

    def test_physical_k_terminal_and_timeout_targets(self):
        rewards, steps = core.discounted_chunk_return(
            torch.tensor([[1., 2., 500.], [3., 4., 5.]]),
            torch.tensor([[1., 1., 0.], [1., 1., 1.]]), .9)
        torch.testing.assert_close(rewards, torch.tensor([2.8, 10.65]))
        self.assertEqual(steps.tolist(), [2, 3])
        result = core.chunk_td_target(rewards, torch.tensor([0., 1.]), steps, torch.tensor([100., 10.]), .9)
        torch.testing.assert_close(result, torch.tensor([2.8, 17.94]))
        with self.assertRaises(ValueError):
            core.discounted_chunk_return(torch.ones(1, 3), torch.tensor([[1, 0, 1]]), .99)

    def test_critic_then_base_then_editor_temperature_and_gradient_routes(self):
        agent = learner()
        data = batch(agent.config)
        before_vision, before_q = snapshots(agent.vision_encoder), snapshots(agent.critic)
        before_editor = snapshots(agent.editor)
        before_alpha = agent.log_temperature.detach().clone()
        callbacks = []

        def fm_callback():
            callbacks.append((agent.critic_steps, agent.editor_steps, agent.temperature_steps))
            return {"loss": .3, "weight_delta": .1}

        metrics = agent.update_call(lambda: data, base_sampler(agent.config), fm_callback)
        self.assertEqual(callbacks, [(20, 0, 0)])
        self.assertEqual((agent.update_calls, agent.critic_steps, agent.editor_steps, agent.temperature_steps), (1, 20, 1, 1))
        self.assertTrue(changed(before_vision, agent.vision_encoder))
        self.assertTrue(changed(before_q, agent.critic))
        self.assertTrue(changed(before_editor, agent.editor))
        self.assertFalse(torch.equal(before_alpha, agent.log_temperature))
        self.assertGreater(metrics["critic_grad_norm"], 0)
        self.assertGreater(metrics["editor_grad_norm"], 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertEqual(len(agent.last_bootstrap_q_indices), 2)

        # A standalone actor step changes neither shared vision nor critic params.
        after_vision, after_q = snapshots(agent.vision_encoder), snapshots(agent.critic)
        agent._editor_temperature_update(data)
        self.assertFalse(changed(after_vision, agent.vision_encoder))
        self.assertFalse(changed(after_q, agent.critic))
        self.assertTrue(all(parameter.grad is None for parameter in agent.vision_encoder.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in agent.critic.parameters()))

    def test_full_state_dict_roundtrip_reproduces_next_stochastic_selection(self):
        first = learner()
        data = batch(first.config)
        first.update_call(lambda: data, base_sampler(first.config), lambda: {"weight_delta": .01})
        checkpoint = io.BytesIO()
        torch.save(first.state_dict(), checkpoint)
        checkpoint.seek(0)
        second = learner(first.config, seed=994)
        second.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
        for name, value in first.named_parameters():
            torch.testing.assert_close(value, dict(second.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(first.get_extra_state()["critic_steps"], second.get_extra_state()["critic_steps"])
        self.assertTrue(second.critic_optimizer.state_dict()["state"])
        self.assertTrue(second.editor_optimizer.state_dict()["state"])
        self.assertTrue(second.temperature_optimizer.state_dict()["state"])
        a = first.select_actions(data["obs"], base_sampler(first.config)(data["obs"]))
        b = second.select_actions(data["obs"], base_sampler(second.config)(data["obs"]))
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        # One further full update must reproduce optimizer/target/RNG evolution.
        first.update_call(lambda: data, base_sampler(first.config), lambda: {})
        second.update_call(lambda: data, base_sampler(second.config), lambda: {})
        for name, value in first.named_parameters():
            torch.testing.assert_close(value, dict(second.named_parameters())[name], rtol=0, atol=0)

    def test_strict_restore_refuses_method_budget_drift(self):
        first = learner()
        second = learner(config(edit_scale=.1))
        with self.assertRaises(ValueError):
            second.load_state_dict(copy.deepcopy(first.state_dict()), strict=True)


if __name__ == "__main__":
    unittest.main()
