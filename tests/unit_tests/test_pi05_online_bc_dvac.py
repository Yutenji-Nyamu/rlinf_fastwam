# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.online_bc_dvac import OnlineBCDvac, log_moments


def test_pi05_dvac_composition_changes_only_method_and_placement(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[2]
    for key, value in {
        "REPO_PATH": root,
        "EMBODIED_PATH": root / "examples/embodiment",
        "ASSETS_PATH": tmp_path,
        "PI05_MODEL_PATH": tmp_path / "pi05",
        "ONLINE_BC_RUN_DIR": tmp_path / "run",
    }.items():
        monkeypatch.setenv(key, str(value))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/embodiment/config")
    ):
        base = compose(
            config_name="robotwin_adjust_bottle_online_bc_openpi",
            overrides=["+online_bc_model=pi05_sidney"],
        )
        method = compose(
            config_name="robotwin_adjust_bottle_online_bc_openpi",
            overrides=["+online_bc_model=pi05_sidney", "+bc_dvac=bounded_half"],
        )
    a, b = [OmegaConf.to_container(c, resolve=True) for c in (base, method)]
    assert b["algorithm"]["online_bc"].pop("dvac") == {
        "enabled": True,
        "tail_steps": 3,
        "window": 5,
        "alpha": 0.125,
        "z_clip": 2.0,
        "log_eps": 1e-12,
        "std_floor": 1e-6,
    }
    assert b["cluster"]["component_placement"]["actor,env,rollout"] == "7"
    b["cluster"] = a["cluster"]
    b["runner"]["logger"]["experiment_name"] = a["runner"]["logger"]["experiment_name"]
    assert a == b
    assert method.actor.model.openpi.config_name == "pi05_sidney_robotwin"
    assert method.actor.model.num_steps == 10
    assert method.env.eval.total_num_envs == 8 and method.env.eval.rollout_epoch == 4


def test_half_range_extreme_values_and_partial_masks():
    calibrator = OnlineBCDvac(alpha=0.125)
    calibrator.annotate([], log_moments(torch.tensor([0.1, 1.0, 10.0]), 1e-12))
    v = torch.logspace(-12, 12, 50)
    mask = torch.ones(50, 14, dtype=torch.bool)
    mask[:4] = False
    mask[4:8, :7] = False
    records = [
        {"dvac_v": values, "action_valid_mask": mask}
        for values in (v, v.flip(0), torch.ones(50), v.roll(20))
    ]
    calibrator.annotate([records], torch.zeros(3))
    for row in records:
        w = row["action_weights"]
        q = mask.sum(-1)
        assert w.shape == (50,) and not w.requires_grad
        assert w.min() >= 0.5 and w.max() <= 1.5
        torch.testing.assert_close((q * w).sum() / q.sum(), torch.tensor(1.0))
    assert records[0]["action_weights"].max() > records[0]["action_weights"].min()
    assert records[2]["action_weights"].eq(1).all()
