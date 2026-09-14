"""CPU-only checks of actual worker methods, without Ray/device construction."""

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.dvac_two_level import build_two_level_success_weights
from rlinf.algorithms.rlt.dvac_weighting import build_rlt_bc_targets_and_weights


class _Base:
    def __init__(self, cfg):
        self.cfg, self._rank, self._world_size, self.update_step = cfg, 0, 1, 0


def _worker_class():
    # Compile the real method bodies; replace only the heavy worker constructor.
    path = Path(__file__).resolve().parents[2] / "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py"
    tree = ast.parse(path.read_text())
    selected = {
        "RLTACLossMixin": {
            "_flatten_chunk", "_chunk_shape", "_bc_metrics",
            "_rlt_dvac_selected_variances", "_prepare_global_batch",
            "_rlt_dvac_success_scale_schedule", "_effective_rlt_dvac_success_scale",
            "_actor_objective_weights",
        },
        "RLTACFSDPPolicy": {
            "__init__", "_rlt_contract", "_rlt_state_payload",
            "_validate_rlt_state", "_restore_rlt_state",
        },
    }
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in selected:
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in selected[node.name]]
            if node.name == "RLTACFSDPPolicy":
                node.bases = [ast.Name("RLTACLossMixin", ast.Load()), ast.Name("_Base", ast.Load())]
            classes.append(node)
    namespace = dict(torch=torch, OmegaConf=OmegaConf, math=math, json=json,
                     hashlib=hashlib, _Base=_Base,
                     build_two_level_success_weights=build_two_level_success_weights,
                     build_rlt_bc_targets_and_weights=build_rlt_bc_targets_and_weights)
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), str(path), "exec"), namespace)
    worker = namespace["RLTACFSDPPolicy"]
    worker._RLT_STATE_SCHEMA_VERSION = 1
    return worker


def _make(*, schedule=True, course_override=None, scale_override=None):
    course = dict(enable=True, warmup_updates=20000, ramp_updates=50000,
                  warmup_bc_weight=5., warmup_q_weight=0.,
                  online_bc_weight=2.5, online_q_weight=.45)
    course.update(course_override or {})
    dvac = dict(mode="apply", mapping="two_level_batch", application="success_episode_bc",
                success_target="reference", outer_scope="successful_batch", applied_horizon=10,
                alpha_local=1., alpha_chunk=1., success_scale=4.)
    if schedule:
        dvac["success_scale_schedule"] = dict(enable=True, anchor="actor_weight_schedule_end", final_scale=2.)
        dvac["success_scale_schedule"].update(scale_override or {})
    cfg = OmegaConf.create(dict(actor=dict(global_batch_size=4, model=dict(num_action_chunks=10, action_dim=2)),
        env=dict(train=dict(auto_reset=False)), algorithm=dict(rlt_dvac=dvac, actor_weight_schedule=course,
        bc_weight=2.5, q_weight=.45, rlt_resume=dict(enable=True, contract=dict(baseline="clean8")))))
    return _worker_class()(cfg)


def _batch():
    profile = torch.tensor([-1., 1.] * 5)
    v = torch.ones(4, 3, 50)
    v[:, 1, :10] = torch.stack((profile, profile - 8, profile + 2, profile + 8)).exp()
    return dict(curr_obs=dict(teacher_dvac_v=v, episode_success=torch.tensor([True, False, True, False])),
                actions=torch.full((4, 20), -9.))


@pytest.mark.parametrize(("step", "scale"), [(0, 4.), (69998, 4.), (69999, 2.), (70000, 2.)])
def test_scale_uses_actual_actor_curriculum_boundary(step, scale):
    worker = _make(); worker.update_step = step
    before = copy.deepcopy(worker.rlt_dvac_cfg)
    effective, metrics = worker._effective_rlt_dvac_success_scale()
    assert effective == scale
    assert metrics["rlt_dvac/success_scale_switched"] == float(scale == 2.)
    assert worker._rlt_dvac_success_scale_schedule()["end_update_step"] == 69999
    assert worker.rlt_dvac_success_scale == 4. and worker.rlt_dvac_cfg == before


def test_missing_or_disabled_schedule_preserves_old_weights_and_contract():
    legacy = _make(schedule=False)
    expected_contract = {"baseline": "clean8", "rlt_dvac": dict(legacy.rlt_dvac_cfg)}
    assert json.loads(legacy._rlt_contract()[0]) == expected_contract
    for step in (0, 70000, 200000):
        legacy.update_step = step
        assert legacy._effective_rlt_dvac_success_scale() == (4., {})
        prepared, _ = legacy._prepare_global_batch(_batch(), train_actor=True)
        torch.testing.assert_close(prepared["rlt_dvac_new_weights"][[0, 2]].mean(), torch.tensor(4.))
    disabled = _make(scale_override={"enable": False, "anchor": "unused"}, course_override={"enable": False})
    assert disabled._effective_rlt_dvac_success_scale() == (4., {})
    assert "rlt_dvac_success_scale_anchor" not in json.loads(disabled._rlt_contract()[0])


@pytest.mark.parametrize(("scale", "course"), [({"anchor": "round"}, {}), ({"final_scale": 0.}, {}), ({}, {"enable": False})])
def test_invalid_opt_in_schedule_rejected_at_init(scale, course):
    with pytest.raises(ValueError):
        _make(scale_override=scale, course_override=course)


def test_same_batch_preserves_shape_direction_failures_and_halves_only_success_bc_gradient():
    worker = _make(); batch = _batch(); success = batch["curr_obs"]["episode_success"]
    rng = torch.get_rng_state().clone(); raw_v = batch["curr_obs"]["teacher_dvac_v"].clone()
    weights, gradients = [], []
    for step in (69998, 70000):
        worker.update_step = step
        prepared, metrics = worker._prepare_global_batch(batch, train_actor=True)
        w = prepared["rlt_dvac_new_weights"]; weights.append(w)
        pi = torch.linspace(.1, 1., 80).reshape(4, 20).requires_grad_()
        ref = torch.linspace(-.8, .9, 80).reshape(4, 10, 2)
        loss, _ = worker._bc_metrics(pi, batch["actions"], ref, None,
            episode_success=success, success_weights=w, apply_success_weights=True)
        gradients.append(torch.autograd.grad(loss, pi)[0])
        assert not w.requires_grad and w.shape == (4, 10)
        assert metrics["rlt_dvac/success_scale_effective"] in (4., 2.)
    torch.testing.assert_close(weights[1][success], weights[0][success] * .5, rtol=0, atol=0)
    torch.testing.assert_close(weights[1][~success], torch.ones_like(weights[1][~success]), rtol=0, atol=0)
    torch.testing.assert_close(gradients[1][success], gradients[0][success] * .5)
    torch.testing.assert_close(gradients[1][~success], gradients[0][~success])
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(raw_v, batch["curr_obs"]["teacher_dvac_v"])
    assert "rlt_dvac_new_weights" not in batch
    untouched, metrics = worker._prepare_global_batch(batch, train_actor=False)
    assert untouched is batch and metrics == {}


@pytest.mark.parametrize("step", [69998, 70000])
def test_resume_recovers_phase_and_rejects_changed_course_anchor(step):
    worker = _make(); worker.update_step = step
    payload = worker._rlt_state_payload(runner_step=193)
    restored = _make(); restored._restore_rlt_state(payload)
    assert restored.update_step == step
    assert restored._effective_rlt_dvac_success_scale() == worker._effective_rlt_dvac_success_scale()
    for field, value in [("warmup_updates", 10000), ("ramp_updates", 25000), ("online_bc_weight", 3.)]:
        changed = _make(course_override={field: value})
        with pytest.raises(ValueError, match="rlt_resume_contract"):
            changed._validate_rlt_state(payload)
