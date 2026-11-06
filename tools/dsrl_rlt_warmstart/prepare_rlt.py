# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Inherit the accepted two benchmark environments and the clean RLT route."""

import copy
import hashlib
import json
import os
import time
from pathlib import Path

from omegaconf import OmegaConf

B = Path("/data/chenyiteng/deployment-20261106/dsrl-rlt-warmstart")
R = Path(__file__).resolve().parents[2]
ACTIVE = Path("/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json")
S = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")


def read(p):
    return json.loads(Path(p).read_text())


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, indent=2))


def main():
    active = read(ACTIVE)
    entries = {}
    base = OmegaConf.to_container(
        OmegaConf.load(
            R / "examples/embodiment/config/pi05_rlt_turn_clean_20260923.yaml"
        ),
        resolve=True,
    )
    for gpu, bench in [(6, "dojo"), (7, "rt2")]:
        old = Path(active["slots"][str(gpu)]["deployment"])
        oldowner = Path(active["slots"][str(gpu)]["owner"])
        op = read(old / "plan.json")
        oc = OmegaConf.to_container(
            OmegaConf.load(old / "config/formal.yaml"), resolve=True
        )
        d = B / f"{bench}-rlt-clean-g{gpu}"
        d.mkdir(exist_ok=False)
        run = Path("/data/chenyiteng/results/dsrl-rlt-warmstart-20261106") / d.name
        run.mkdir(parents=True, exist_ok=False)

        def relocate(x):
            return json.loads(
                json.dumps(x)
                .replace(op["training_source"], str(R))
                .replace(str(old), str(d))
                .replace(op["run"], str(run))
            )

        p = relocate(op)
        predecessor = read(oldowner / "identity.json")
        p.update(
            training_source=str(R),
            run=str(run),
            gpu=gpu,
            gpus=[gpu],
            bench=bench,
            method="rlt",
            namespace=f"RLinf_warmstart_rlt_g{gpu}_20261106",
            pre_registered=True,
            predecessors=[predecessor],
            from_rlt=False,
            model_contract_required=False,
            acceleration_trial=False,
            protected_output=f"/data/chenyiteng/models/protected-eval/dsrl-rlt-warmstart-20261106/{d.name}",
            rlt_source_sha="3f35e89b9a9aa61d66aba2dfead41aebc2ee19cc",
        )
        p["stage1_data"] = (
            "/data/chenyiteng/datasets/robodojo-rlt-dustbin-20261106"
            if bench == "dojo"
            else "/data/chenyiteng/datasets/robotwin2/canonical/turn-switch-aloha-clean50-20260919"
        )
        p["gpu_admission"] = read(S / "plan.json")["slots"][str(gpu)]
        p.pop("defer_formal_to", None)
        stage = {
            "name": "stage1",
            "command": [
                p["python"],
                "-u",
                "-B",
                str(R / "tools/dsrl_rlt_warmstart/stage1.py"),
                str(d),
            ],
            "timeout": 18000,
        }
        p["pre_native_phases"] = [stage] if bench == "dojo" else []
        p["phases"] = [] if bench == "dojo" else [stage]
        p["phases"].append(
            {
                "name": "reference_eval",
                "command": [
                    p["python"],
                    "-u",
                    "-B",
                    str(R / "tools/dsrl_rlt_warmstart/reference_eval.py"),
                    str(d),
                ],
                "timeout": 2400,
            }
        )
        for phase in ("smoke", "smoke_resume", "benchmark", "formal"):
            cmd = (
                [
                    p["python"],
                    "-u",
                    "-B",
                    str(R / "tools/dsrl_rlt_warmstart/rlt_bench_driver.py"),
                    str(d),
                ]
                if phase == "benchmark"
                else [
                    p["python"],
                    "-u",
                    "-B",
                    str(R / "tools/dsrl_rlt_warmstart/rlt_training_entry.py"),
                    "--config-path",
                    str(d / "config"),
                    "--config-name",
                    phase,
                ]
            )
            p["phases"].append(
                {
                    "name": phase,
                    "command": cmd,
                    "timeout": None if phase == "formal" else 2400,
                }
            )
        c = copy.deepcopy(base)
        chunk = 50 if bench == "dojo" else 20
        c["env"] = relocate(oc["env"])
        c["cluster"] = relocate(oc["cluster"])
        c["actor"]["global_batch_size"] = c["actor"]["micro_batch_size"] = 256
        c["actor"]["model"].update(
            num_action_chunks=chunk,
            ref_num_action_chunks=chunk,
            output_activation="identity",
        )
        c["actor"].update(defer_metrics=True, foreach_target_update=True)
        c["actor"]["optim"]["fused"] = c["actor"]["critic_optim"]["fused"] = True
        c["rollout"]["model"] = copy.deepcopy(c["actor"]["model"])
        c["rollout"]["model"]["model_path"] = None
        feature = relocate(oc["actor"]["model"])
        o = feature["openpi"]
        for key in list(o):
            if key.startswith("dsrl_"):
                o.pop(key)
        o.update(
            use_dsrl=False,
            use_rlt=True,
            train_expert_only=False,
            rlt_train_vla=False,
            rlt_alpha=0.0,
            rlt_input_dim=2048,
            rlt_embed_dim=2048,
            rlt_prefix_seq_len=768,
            rlt_num_layers=2,
            rlt_num_heads=8,
            rlt_mlp_ratio=4.0,
            rlt_image_only=True,
            rlt_use_mask=True,
            rlt_dvac_mode="off",
            rlt_action_adapter="robodojo_aloha_delta_v1"
            if bench == "dojo"
            else "robotwin_aloha_canonical_v1",
            rlt_token_path=str(run / "stage1/token-final.pt"),
        )
        c["rollout"]["rlt_feature_model"] = feature
        c["runner"].update(
            max_steps=400, val_check_interval=20, save_interval=20, resume_dir=None
        )
        c["runner"]["logger"].update(
            log_path=str(run / "formal"), experiment_name="formal"
        )
        a = c["algorithm"]
        a.update(gamma=0.999, critic_actor_ratio=2, dv_observe=False)
        a["rlt_dvac"] = {"mode": "off"}
        a["rlt_schedule"].update(
            warmup_min_size=2000,
            warmup_post_collect_updates=5000,
            max_updates_per_train_step=1600,
        )
        a["actor_weight_schedule"].update(warmup_updates=5000, ramp_updates=10000)
        # Quantile-normalized Dojo arms are deltas; keep state recovery in the
        # OpenPI output transform exactly once, then the native joint reorder.
        a["rlt_resume"]["contract"] = {
            "base_source": p["rlt_source_sha"],
            "model_identity": feature["model_path"],
            "config_name": o["config_name"],
            "reconstruction": "current_causal_ar",
            "stage1_token_path": o["rlt_token_path"],
            "stage1_token_sha256": "pending_stage1",
            "norm_stats_sha256": hashlib.sha256(
                Path(feature["openpi_data"]["norm_stats_path"]).read_bytes()
            ).hexdigest(),
            "train_vla": False,
            "prefix_seq_len": 768,
            "image_only": True,
            "use_mask": True,
            "z_dim": 2048,
            "action_horizon": 50,
            "action_chunk": chunk,
            "action_dim": 14,
            "canonical_adapter_version": o["rlt_action_adapter"],
            "route": "full_task",
            "replay": "compact_transition",
            "bootstrap": "pure_truncation_only",
            "gamma_per_physics_step": 0.999,
            "warmup_transitions": 2000,
            "warmup_updates": 5000,
        }
        (d / "config").mkdir()
        OmegaConf.save(OmegaConf.create(c), d / "config/formal.yaml")
        sf = copy.deepcopy(feature)
        sf["openpi"]["rlt_token_path"] = None
        OmegaConf.save(OmegaConf.create(sf), d / "config/stage1.yaml")
        for phase, steps in [("smoke", 1), ("smoke_resume", 2)]:
            sc = copy.deepcopy(c)
            sc["runner"].update(
                max_steps=steps,
                val_check_interval=0,
                save_interval=1,
                resume_dir=None
                if phase == "smoke"
                else str(run / "smoke/smoke/checkpoints/global_step_1"),
            )
            sc["runner"]["logger"].update(
                log_path=str(run / phase), experiment_name=phase
            )
            sc["algorithm"]["rlt_schedule"].update(
                warmup_min_size=8,
                warmup_post_collect_updates=16,
                max_updates_per_train_step=16,
            )
            sc["algorithm"]["actor_weight_schedule"].update(
                warmup_updates=16, ramp_updates=32
            )
            sc["env"]["train"].update(
                max_steps_per_rollout_epoch=100 if bench == "dojo" else 40,
                max_episode_steps=100 if bench == "dojo" else 40,
            )
            # Smoke and resume must share the same schedule contract.
            sc["algorithm"]["rlt_resume"]["contract"].update(
                warmup_transitions=8, warmup_updates=16
            )
            OmegaConf.save(OmegaConf.create(sc), d / f"config/{phase}.yaml")
        save(
            d / "training-environment.json",
            relocate(read(old / "training-environment.json")),
        )
        if bench == "dojo":
            save(
                d / "sim-environment.json", relocate(read(old / "sim-environment.json"))
            )
            save(d / f"native-g{gpu}.json", relocate(read(old / f"native-g{gpu}.json")))
            (d / "rpc").mkdir(mode=0o700)
            (d / "rpc/authkey").write_bytes(os.urandom(32))
            (d / "rpc/authkey").chmod(0o600)
        save(d / "plan.json", p)
        entries[str(gpu)] = {
            "deployment": str(d),
            "owner": str(d),
            "old_owner": str(oldowner),
            "predecessor": predecessor,
            "run": str(run),
            "bench": bench,
            "source": str(R),
        }
    prepared = read(B / "prepared-dsrl.json")
    prepared["entries"].update(entries)
    save(B / "prepared-dsrl.json", prepared)
    save(B / "prepared-rlt.json", {"time": time.time(), "entries": entries})
    print(json.dumps(entries))


if __name__ == "__main__":
    main()
