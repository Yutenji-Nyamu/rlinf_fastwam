# Copyright 2026 The RLinf Authors and task-reward contributors.
# Licensed under the Apache License, Version 2.0.
"""Derive the SZ2 terminal-score run from the saved, accepted C32/B16 config."""

import copy
import json
import os
import uuid
from pathlib import Path


def main():
    root = Path(
        "/data/chenyiteng/deployment-20261021/wm-topreward-continuous-sz2-g45-v1"
    )
    repo = root / "source"
    assets = Path("/data/chenyiteng/projects/opendw-robotwin-smoke-20261003")
    queue = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")
    previous = root.with_name("wm-topreward-sz2-g45-v1")
    assert os.getuid() == 20001 and not (root / "run-plan.json").exists()
    base = json.loads(
        (
            repo / "docs/world-model/topreward_20261021/reference/rl_base.json"
        ).read_text()
    )
    bundle = root / "wm-bundle"
    bundle.mkdir()
    checkpoint = (
        assets
        / "wm-cycle-20261009-v2/run/wm000/train/checkpoints/weights/step_000835.pt"
    )
    (bundle / "model.pt").symlink_to(checkpoint)
    for name in ("norm_stats.json", "vae", "tokenizer", "text_encoder"):
        (bundle / name).symlink_to(assets / "models/dw05-robotwin" / name)
    run = root / "run"
    run.mkdir()
    configs = root / "config"
    configs.mkdir()

    def save(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n")

    trials = []
    for key in ("smoke", "formal"):
        cfg = copy.deepcopy(base)
        out = run / key
        cfg["runner"]["logger"].update(log_path=str(out), experiment_name=key)
        cfg["runner"]["per_worker_log_path"] = str(out / "worker-metrics")
        for component in ("actor", "env", "rollout"):
            cfg[component]["group_name"] = "TOPRewardTerminal_" + key + "_" + component
        for mode in ("train", "eval"):
            cfg["env"][mode]["video_cfg"]["video_base_dir"] = str(out / "video" / mode)
        cfg["env"]["eval"]["task_config"]["save_path"] = str(out / "robotwin_data/eval")
        cfg["env"]["train"].update(
            reward_backend="topreward",
            reward_source="qwen3_vl_raw_true_probability",
            reward_mode="terminal_score",
            use_rel_reward=False,
            reward_coef=1.0,
            ignore_terminations=False,
            service_urls=["http://127.0.0.1:18986"],
        )
        cfg["algorithm"]["filter_rewards"] = False
        cfg["algorithm"]["dvac_gradient_weighting"]["output_dir"] = str(out / "dvac")
        if key == "smoke":
            cfg["runner"].update(max_steps=1, save_interval=1, val_check_interval=1)
            cfg["env"]["train"]["total_num_envs"] = 16
            cfg["env"]["train"]["rollout_epoch"] = 1
            cfg["algorithm"]["update_epoch"] = 1
            cfg["actor"]["global_batch_size"] = 192  # 16 trajectories * 12 chunks
        save(configs / (key + ".yaml"), cfg)
        trials.append(
            {
                "key": key,
                "config": str(configs / (key + ".yaml")),
                "namespace": "topreward_terminal_sz2_1021_" + key,
                "episode_steps": 384,
                "num_envs": cfg["env"]["train"]["total_num_envs"],
                "timeout_seconds": 7200 if key == "smoke" else 5184000,
            }
        )
    # Reuse the deployed SZ2 GPU5 renderer binding for the native evaluation
    # workers. Actor/rollout retain their explicit physical CUDA4 placement.
    q = json.loads(
        Path(
            "/data/chenyiteng/deployment-20261020/c20-sz2-g45-v1/fallback-g5.json"
        ).read_text()
    )
    inherited = json.loads((Path(q["runtime"]) / "environment.json").read_text())
    keys = (
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_HOME",
        "MUJOCO_GL",
        "VK_ICD_FILENAMES",
        "XDG_RUNTIME_DIR",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    )
    env = {k: inherited[k] for k in keys if k in inherited}
    env.update(
        HOME="/home/chenyiteng",
        USER="chenyiteng",
        LOGNAME="chenyiteng",
        RAY_ADDRESS="127.0.0.1:26379",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
    )
    scope = {
        k: inherited[k]
        for k in (
            "RLINF_OPENDW_GPU_SCOPE_MANIFEST",
            "LD_PRELOAD",
            "__GL_APPLICATION_PROFILE",
        )
    }
    scope["PYTHONPATH"] = ":".join(
        (
            str(repo),
            str(queue / "scope-g5/bootstrap"),
            "/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support",
        )
    )
    save(root / "environment.json", env)
    save(root / "graphics-fragment.json", scope)
    code = repo / "local_scripts/wovr_cycle_20261008/code"
    reward = repo / "local_scripts/topreward_20261021"
    argv = [
        str(assets / "venv/bin/python"),
        "-u",
        "-B",
        str(code / "opendw_service_batched.py"),
        "--bundle",
        str(bundle),
        "--reward-backend",
        "topreward",
        "--reward-checkpoint",
        "/data/chenyiteng/models/Qwen3-VL-8B-Instruct-0c351dd0",
        "--qwen-python",
        str(previous / "qwen-venv/bin/python"),
        "--physical-gpu",
        "5",
        "--port",
        "18986",
        "--output-dir",
        str(run / "services/wm5/records"),
        "--evidence-samples",
        "640",
        "--execution-mode",
        "batched",
        "--wm-batch-size",
        "16",
    ]
    plan = {
        "owner_dir": str(run),
        "repo": str(repo),
        "physical_gpus": [4, 5],
        "token": uuid.uuid4().hex,
        "python": "/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python",
        "queue_ops": str(queue / "ops"),
        "existing_owner": str(
            assets / "lift-two-gpu-b16-lean-20261007-v1/code/owner.py"
        ),
        "environment_file": str(root / "environment.json"),
        "graphics_fragment": str(root / "graphics-fragment.json"),
        "ray_address": "127.0.0.1:26379",
        "ray_dashboard_url": "http://127.0.0.1:28266",
        "management_namespace": "topreward_terminal_sz2_1021_ops",
        "trials": trials,
        "borrow_receipt": str(root / "borrowed.json"),
        "release_receipt": str(root / "released.json"),
        "services": [
            {
                "key": "wm5",
                "physical_gpu": 5,
                "argv": argv,
                "cwd": str(code),
                "url": "http://127.0.0.1:18986",
                "startup_seconds": 1200,
                "environment": {
                    "PYTHONPATH": ":".join(
                        (
                            str(code),
                            str(reward),
                            str(assets / "opendw"),
                            str(assets / "payload_v2/tools"),
                        )
                    ),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                },
            }
        ],
    }
    save(root / "run-plan.json", plan)
    print(
        json.dumps(
            {
                "plan": str(root / "run-plan.json"),
                "configs": [t["config"] for t in trials],
                "policy": base["actor"]["model"]["model_path"],
                "wm": str(checkpoint),
            }
        )
    )


if __name__ == "__main__":
    main()
