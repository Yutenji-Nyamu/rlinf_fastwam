# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""CPU checks for actual recipes, shape propagation, and update-budget capacity."""

import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from omegaconf import OmegaConf
from prepare import BASE, read, sha


def main():
    rows = {}
    for g, e in read(BASE / "prepared-dsrl.json")["entries"].items():
        d = Path(e["deployment"])
        c = OmegaConf.to_container(
            OmegaConf.load(d / "config/formal.yaml"), resolve=True
        )
        p = read(d / "plan.json")
        assert c["runner"]["resume_dir"] is None
        assert c["env"]["train"]["total_num_envs"] == 16
        assert c["env"]["train"]["max_episode_steps"] == (
            700 if e["bench"] == "dojo" else 200
        )
        assert p["gpu"] == int(g) and p["training_source"] == e["source"]
        assert not c["algorithm"]["backup_entropy"]
        assert c["algorithm"]["critic_actor_ratio"] == 2
        for phase in p["phases"]:
            assert Path(phase["command"][3]).is_file(), phase["command"]
        if e["method"] == "rlt":
            a = c["algorithm"]
            s = a["rlt_schedule"]
            h = c["env"]["train"]["max_steps_per_rollout_epoch"]
            assert (s["warmup_min_size"], s["warmup_post_collect_updates"]) == (
                20000,
                30000,
            )
            assert (
                c["actor"]["global_batch_size"],
                c["actor"]["micro_batch_size"],
            ) == (512, 256)
            assert a["actor_weight_schedule"]["warmup_updates"] == 20000
            assert a["actor_weight_schedule"]["ramp_updates"] == 50000
            assert a["update_epoch"] == 5
            for side in ["actor", "rollout"]:
                assert (
                    c[side]["model"]["num_action_chunks"]
                    == c[side]["model"]["ref_num_action_chunks"]
                    == 10
                )
            f = c["rollout"]["rlt_feature_model"]
            assert f["num_action_chunks"] == f["openpi"]["action_chunk"] == 10
            assert f["openpi"]["action_horizon"] == 50
            assert a["rlt_resume"]["contract"]["action_chunk"] == 10
            assert a["gamma"] == (0.99 if e["bench"] == "rt2" else 0.999)
            assert (
                sha(f["openpi"]["rlt_token_path"])
                == a["rlt_resume"]["contract"]["stage1_token_sha256"]
            )
            new_budget = (16 * h // 10) * 5
            cap = s["max_updates_per_train_step"]
            assert cap == 2 * new_budget
            # Even a worst-case no-success collection repays the warmup debt.
            updates = 0
            for iteration in range(32):
                desired = 30000 + iteration * new_budget
                updates += min(desired - updates, cap)
            assert updates == desired
            rows[g] = {
                "batch": 512,
                "micro": 256,
                "chunk": 10,
                "max_new_updates": new_budget,
                "cap": cap,
                "warmup_debt_paid": True,
                "token_verified": True,
            }
        else:
            assert c["algorithm"]["utd_ratio"] == 5
            assert c["algorithm"]["dsrl_critic_warmup_updates"] == 2000
            assert c["algorithm"]["replay_buffer"]["warmup_size"] == 2000
            assert (
                c["actor"]["global_batch_size"] == c["actor"]["micro_batch_size"] == 256
            )
            for side in ["actor", "rollout"]:
                o = c[side]["model"]["openpi"]
                assert o["dsrl_fp32_small"] and o["dsrl_noise_groups"] == 50
                assert o["dsrl_action_noise_dim"] == 1600 and o["dsrl_noise_bound"] == 2
            assert c["actor"]["model"] == c["rollout"]["model"]
            rows[g] = {
                "utd": 5,
                "q_warmup": 2000,
                "batch": 256,
                "noise_dim": 1600,
                "fp32_small": True,
            }
        rows[g]["episodes"] = c["runner"]["max_steps"] * 16
    print(json.dumps({"recipes": rows, "result": "pass"}))


if __name__ == "__main__":
    main()
