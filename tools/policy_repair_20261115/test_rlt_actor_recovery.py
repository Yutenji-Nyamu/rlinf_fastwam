"""CPU tests for actor recovery isolation, one-time scope, and the BC/Q schedule."""

import ast
import copy
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch


def load_methods(source):
    tree = ast.parse(Path(source).read_text())
    names = {"_recover_actor_from_policy", "_actor_objective_weights"}
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in methods} != names:
        raise AssertionError("Expected production methods were not found")
    module = ast.Module(body=methods, type_ignores=[])
    scope = {"torch": torch, "os": os}
    exec(compile(module, str(source), "exec"), scope)
    return scope


class Strategy:
    def get_model_state_dict(self, model, **kwargs):
        return {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

    def load_model_with_state_dict(self, model, state, **kwargs):
        model.load_state_dict(state, strict=True)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.actor_mean = torch.nn.Linear(2, 1)
        self.actor_logstd = torch.nn.Linear(2, 1)
        self.q_head = torch.nn.Linear(3, 1)
        self.register_buffer("action_scale", torch.tensor(1.0))


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.environ.get(
            "RLT_WORKER_SOURCE",
            str(
                Path(__file__).resolve().parents[2]
                / "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py"
            ),
        )
        methods = load_methods(source)
        cls.recover = staticmethod(methods["_recover_actor_from_policy"])
        cls.weights = staticmethod(methods["_actor_objective_weights"])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        root = Path(self.folder.name)
        self.origin = root / "global_step_46"
        self.policy_path = root / "bc.pt"
        model = Policy()
        old = Policy()
        with torch.no_grad():
            for value in old.parameters():
                value.fill_(0.125)
        torch.save(old.state_dict(), self.policy_path)
        actor_params = [
            p for name, p in model.named_parameters() if not name.startswith("q_head.")
        ]
        actor_opt = torch.optim.Adam(actor_params, lr=1e-4)
        critic_opt = torch.optim.Adam(model.q_head.parameters(), lr=1e-4)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        actor_opt.step()
        critic_opt.step()
        actor_opt.zero_grad()
        critic_opt.zero_grad()
        recovery = {
            "enable": True,
            "resume_checkpoint": str(self.origin),
            "policy_path": str(self.policy_path),
        }
        self.worker = SimpleNamespace(
            cfg=SimpleNamespace(
                algorithm={"rlt_actor_recovery": recovery},
                actor=SimpleNamespace(
                    model=SimpleNamespace(model_type="rlt_mlp_policy"),
                    optim=SimpleNamespace(lr=5e-5),
                ),
            ),
            model=model,
            target_model=copy.deepcopy(model),
            optimizer=actor_opt,
            qf_optimizer=critic_opt,
            lr_scheduler=SimpleNamespace(base_lrs=[1e-4]),
            _strategy=Strategy(),
            logger=logging.getLogger("recovery_test"),
            update_step=417570,
            replay_buffer=object(),
            demo_buffer=object(),
        )

    def test_only_actor_changes_and_only_actor_adam_clears(self):
        worker = self.worker
        before = copy.deepcopy(worker.model.state_dict())
        target_before = copy.deepcopy(worker.target_model.state_dict())
        critic_opt_before = copy.deepcopy(worker.qf_optimizer.state_dict())
        replay, reference = worker.replay_buffer, worker.demo_buffer
        self.assertTrue(worker.optimizer.state)
        self.recover(worker, self.origin / "actor")
        recovered = torch.load(self.policy_path, weights_only=True)
        for key, value in worker.model.state_dict().items():
            expected = (
                recovered[key]
                if key.startswith(("backbone.", "actor_mean.", "actor_logstd."))
                else before[key]
            )
            self.assertTrue(torch.equal(value, expected), key)
        self.assertEqual(len(worker.optimizer.state), 0)
        self.assertEqual(worker.optimizer.param_groups[0]["lr"], 5e-5)
        self.assertEqual(worker.lr_scheduler.base_lrs, [5e-5])
        for key, value in worker.target_model.state_dict().items():
            self.assertTrue(torch.equal(value, target_before[key]), key)
        for parameter, state in worker.qf_optimizer.state_dict()["state"].items():
            for key, value in state.items():
                self.assertTrue(
                    torch.equal(value, critic_opt_before["state"][parameter][key])
                )
        self.assertEqual(worker.update_step, 417570)
        self.assertIs(worker.replay_buffer, replay)
        self.assertIs(worker.demo_buffer, reference)

    def test_later_checkpoint_does_not_reapply_recovery(self):
        worker = self.worker
        before = copy.deepcopy(worker.model.state_dict())
        self.policy_path.unlink()  # A later resume does not even read the old policy.
        self.recover(worker, self.origin.with_name("global_step_48") / "actor")
        for key, value in worker.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(worker.optimizer.state)

    def test_bad_actor_shape_does_not_modify_model(self):
        before = copy.deepcopy(self.worker.model.state_dict())
        bad = torch.load(self.policy_path, weights_only=True)
        bad["actor_mean.weight"] = torch.zeros(5, 7)
        torch.save(bad, self.policy_path)
        with self.assertRaises(ValueError):
            self.recover(self.worker, self.origin / "actor")
        for key, value in self.worker.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        self.assertTrue(self.worker.optimizer.state)

    def test_two_bc_rounds_then_ten_round_ramp(self):
        anchor = self.worker.update_step
        updates_per_round = 11200
        warmup = anchor + 2 * updates_per_round
        ramp = 10 * updates_per_round
        self.worker.cfg.algorithm["actor_weight_schedule"] = {
            "enable": True,
            "warmup_updates": warmup,
            "ramp_updates": ramp,
            "warmup_bc_weight": 1.0,
            "warmup_q_weight": 0.0,
            "online_bc_weight": 1.0,
            "online_q_weight": 0.005,
        }
        for step in [anchor, warmup - 1]:
            self.worker.update_step = step
            bc, q, _ = self.weights(self.worker)
            self.assertEqual((bc, q), (1.0, 0.0))
        self.worker.update_step = warmup + ramp // 2 - 1
        self.assertAlmostEqual(self.weights(self.worker)[1], 0.0025)
        self.worker.update_step = warmup + ramp - 1
        self.assertAlmostEqual(self.weights(self.worker)[1], 0.005)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
