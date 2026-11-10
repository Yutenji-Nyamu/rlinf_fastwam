# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Prepare a narrowly scoped stability retry from the accepted live exports; never stop jobs."""

import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from omegaconf import OmegaConf

BASE = Path("/data/chenyiteng/deployment-20261110/stability-retry")
EXPORT = Path(
    "/data/chenyiteng/projects/rlinf-shenzhen/exports/stability-retry-20261110"
)
RESULTS = Path("/home/chenyiteng/results/stability-retry-20261110")
ACTIVE = Path("/data/chenyiteng/deployment-20261105/dsrl-fp32-accel/active-runs.json")
SCHEDULER = Path("/data/chenyiteng/deployment-20261008/bc-signal-tau-v1")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {
            k: v
            for key, item in value.items()
            for k, v in flatten(item, f"{prefix}.{key}" if prefix else key).items()
        }
    return {prefix: value}


def config_changes(before, after):
    a, b = flatten(before), flatten(after)
    return {
        k: {"before": a.get(k), "after": b.get(k)}
        for k in sorted(a.keys() | b.keys())
        if a.get(k) != b.get(k)
    }


def retry_recipe(config, bench, method, successful):
    """Only learning-recipe and explicit evaluation/sample budget changes."""
    c = copy.deepcopy(config)
    c["runner"]["resume_dir"] = None
    c["env"]["train"]["total_num_envs"] = 16
    if bench == "dojo":
        c["env"]["eval"].update(total_num_envs=16, rollout_epoch=2)
    c["runner"].update(
        max_steps=313 if method == "rlt" else 200,
        val_check_interval=10 if method == "rlt" else (7 if bench == "rt2" else 6),
        save_interval=10 if method == "rlt" else (35 if bench == "rt2" else 2),
    )
    if method == "dsrl":
        c["algorithm"]["utd_ratio"] = 5
        c["runner"]["max_steps"] = 400 if bench == "rt2" else 200
        c["algorithm"]["noise_prior_kl_weight"] = 0.5
        c["algorithm"]["entropy_tuning"].update(
            alpha_type="fixed_alpha", initial_alpha=0.0
        )
        return c
    old = successful
    c["actor"]["global_batch_size"] = old["actor"]["global_batch_size"]
    c["actor"]["micro_batch_size"] = old["actor"]["micro_batch_size"]
    c["algorithm"]["actor_weight_schedule"] = copy.deepcopy(
        old["algorithm"]["actor_weight_schedule"]
    )
    c["algorithm"]["rlt_schedule"] = copy.deepcopy(old["algorithm"]["rlt_schedule"])
    c["algorithm"]["update_epoch"] = old["algorithm"]["update_epoch"]
    c["algorithm"]["critic_actor_ratio"] = old["algorithm"]["critic_actor_ratio"]
    c["algorithm"]["gamma"] = (
        old["algorithm"]["gamma"] if bench == "rt2" else config["algorithm"]["gamma"]
    )
    for key in ["actor", "rollout"]:
        c[key]["model"].update(num_action_chunks=10, ref_num_action_chunks=10)
    f = c["rollout"]["rlt_feature_model"]
    f["num_action_chunks"] = 10
    f["openpi"]["action_chunk"] = 10
    # The old 8-env/C10/200-step cap (1600) is twice its maximum new-data
    # update budget (8*200/10*5). Preserve that headroom under new parallelism
    # and Dojo's 700-step episodes, so warmup debt can also be paid down.
    horizon = c["env"]["train"]["max_steps_per_rollout_epoch"]
    max_new = 16 * horizon // 10
    c["algorithm"]["rlt_schedule"]["max_updates_per_train_step"] = (
        2 * max_new * c["algorithm"]["update_epoch"]
    )
    contract = c["algorithm"]["rlt_resume"]["contract"]
    contract.update(
        action_chunk=10,
        gamma_per_physics_step=c["algorithm"]["gamma"],
        warmup_transitions=20000,
        warmup_updates=30000,
    )
    # First experiment is genuine BC throughout; opening Q requires its result.
    c["algorithm"]["actor_weight_schedule"]["enable"] = False
    c["algorithm"].update(
        q_weight=0.0,
        bc_weight=1.0,
        reference_dropout_prob=0.0,
        rlt_reference_bootstrap=True,
        rlt_reference_replay_size=20000,
    )
    c["algorithm"]["demo_buffer"] = {
        "enable_cache": True,
        "cache_size": 20000,
        "sample_window_size": 20000,
        "min_buffer_size": 1,
        "auto_save": False,
    }
    return c


def main():
    assert os.getuid() == 1003
    assert not (BASE / "prepared-dsrl.json").exists(), "Do not prepare twice"
    active = read(ACTIVE)
    sp, ss = read(SCHEDULER / "plan.json"), read(SCHEDULER / "status.json")
    old_success = read(sp["slots"]["6"]["fallback"])
    success_cfg = OmegaConf.to_container(
        OmegaConf.load(Path(old_success["runtime"]) / "resolved.yaml"), resolve=True
    )
    assert success_cfg["env"]["train"]["task_config"]["task_name"] == "click_bell"
    BASE.mkdir(parents=True, exist_ok=True)
    save(BASE / "previous-active.json", active)
    save(BASE / "successful-rlt-config.json", success_cfg)
    source_receipts = {}
    for method, g in [("dsrl", "4"), ("rlt", "6")]:
        src, target = Path(active["slots"][g]["source"]), EXPORT / method
        assert not target.exists()
        shutil.copytree(
            src,
            target,
            symlinks=True,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git", ".pytest_cache"
            ),
        )
        shutil.copytree(
            Path(__file__).parent, target / "tools/stability_retry_20261110"
        )
        files = [
            "rlinf/workers/actor/fsdp_sac_policy_worker.py",
            "rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py",
            "tools/dsrl_rlt_warmstart/rlt_training_entry.py",
            "tools/dsrl_rlt_warmstart/rt2_owner.py",
        ]
        hashes = {name: sha(src / name) for name in files if (src / name).is_file()}
        assert all(sha(target / name) == value for name, value in hashes.items())
        source_receipts[method] = {
            "inherited": str(src),
            "export": str(target),
            "unchanged_sha256": hashes,
        }
    from apply_changes import apply

    apply(EXPORT)
    entries, changes = {}, {}
    for g, bench, method in [
        ("4", "rt2", "dsrl"),
        ("5", "dojo", "dsrl"),
        ("6", "dojo", "rlt"),
    ]:
        old = Path(active["slots"][g]["deployment"])
        oldowner = Path(active["slots"][g]["owner"])
        op = read(old / "plan.json")
        oldc = OmegaConf.to_container(
            OmegaConf.load(old / "config/formal.yaml"), resolve=True
        )
        root, d = EXPORT / method, BASE / f"{bench}-{method}-clean-g{g}"
        run = RESULTS / d.name
        d.mkdir(exist_ok=False)
        run.mkdir(parents=True, exist_ok=False)
        (d / "config").mkdir()

        def reloc(obj, op=op, root=root, old=old, d=d, run=run):
            text = (
                json.dumps(obj)
                .replace(op["training_source"], str(root))
                .replace(str(old), str(d))
                .replace(op["run"], str(run))
            )
            return json.loads(text)

        p = reloc(op)
        predecessor = (
            ss["slots"]["4"]["identity"]
            if g == "4"
            else read(oldowner / "identity.json")
        )
        assert predecessor["uid"] == 1003
        p.update(
            training_source=str(root),
            run=str(run),
            gpu=int(g),
            gpus=[int(g)],
            method=method,
            bench=bench,
            from_rlt=g == "4",
            pre_registered=g != "4",
            predecessors=[predecessor],
            namespace=f"RLinf_stability_{method}_{bench}_g{g}_20261110",
            num_envs=16,
            native_envs_per_card=16,
            acceleration_trial=False,
            model_contract_required=False,
            protected_output=f"/data/chenyiteng/models/protected-eval/stability-retry-20261110/{d.name}",
        )
        p.pop("defer_formal_to", None)
        p["gpu_admission"] = copy.deepcopy(sp["slots"][g])
        p["pre_native_phases"] = []
        c = retry_recipe(oldc, bench, method, success_cfg)
        c = reloc(c)
        c["runner"]["logger"].update(
            log_path=str(run / "formal"), experiment_name="formal"
        )
        if method == "rlt":
            token = oldc["rollout"]["rlt_feature_model"]["openpi"]["rlt_token_path"]
            manifest = read(Path(op["run"]) / "stage1/complete.json")
            assert sha(token) == manifest["token_sha256"]
            c["rollout"]["rlt_feature_model"]["openpi"]["rlt_token_path"] = token
            c["algorithm"]["rlt_resume"]["contract"].update(
                stage1_token_path=token, stage1_token_sha256=manifest["token_sha256"]
            )
            manifest.update(
                reused_from=str(Path(op["run"]) / "stage1"), token_path=token
            )
            save(run / "stage1/complete.json", manifest)
            p["stage1_reuse"] = {
                "token": token,
                "sha256": manifest["token_sha256"],
                "source_run": op["run"],
            }
        OmegaConf.save(OmegaConf.create(c), d / "config/formal.yaml")
        entry = (
            "tools/dsrl_rlt_warmstart/rlt_training_entry.py"
            if method == "rlt"
            else "tools/dojo_offpolicy/training_entry.py"
        )
        p["phases"] = []
        if method == "rlt":
            # Same accepted reference adapter check, now with the C10 interface.
            p["phases"].append(
                {
                    "name": "reference_eval",
                    "command": [
                        p["python"],
                        "-u",
                        "-B",
                        str(root / "tools/dsrl_rlt_warmstart/reference_eval.py"),
                        str(d),
                    ],
                    "timeout": 2400,
                }
            )
            smoke = copy.deepcopy(c)
            smoke["runner"].update(max_steps=2, val_check_interval=0, save_interval=1)
            smoke["runner"]["logger"].update(
                log_path=str(run / "smoke"), experiment_name="smoke"
            )
            smoke["algorithm"]["rlt_schedule"].update(
                warmup_min_size=8,
                warmup_post_collect_updates=16,
                max_updates_per_train_step=32,
            )
            smoke["env"]["train"].update(
                max_steps_per_rollout_epoch=50 if bench == "dojo" else 40,
                max_episode_steps=50 if bench == "dojo" else 40,
            )
            OmegaConf.save(OmegaConf.create(smoke), d / "config/smoke.yaml")
            p["phases"].append(
                {
                    "name": "smoke",
                    "command": [
                        p["python"],
                        "-u",
                        "-B",
                        str(root / entry),
                        "--config-path",
                        str(d / "config"),
                        "--config-name",
                        "smoke",
                    ],
                    "timeout": 2400,
                }
            )
        # add_smoke.py inserts a short DSRL update/checkpoint check before formal.
        # Formal starts fresh with the full warmup budget.
        p["phases"].append(
            {
                "name": "formal",
                "command": [
                    p["python"],
                    "-u",
                    "-B",
                    str(root / entry),
                    "--config-path",
                    str(d / "config"),
                    "--config-name",
                    "formal",
                ],
                "timeout": None,
            }
        )
        env_source = old
        if not (env_source / "training-environment.json").exists():
            env_source = (
                Path("/data/chenyiteng/deployment-20261108/dsrl-rlt-retry") / d.name
            )
        for name in [
            "training-environment.json",
            "sim-environment.json",
            f"native-g{g}.json",
        ]:
            if (env_source / name).exists():
                obj = reloc(read(env_source / name))
                obj = json.loads(json.dumps(obj).replace(str(env_source), str(d)))
                if name == "training-environment.json":
                    obj.update(DOJO_GRPO_DEPLOYMENT=str(d), PYTHONDONTWRITEBYTECODE="1")
                save(d / name, obj)
        if bench == "dojo":
            (d / "rpc").mkdir(mode=0o700)
            (d / "rpc/authkey").write_bytes(os.urandom(32))
            (d / "rpc/authkey").chmod(0o600)
        save(d / "plan.json", p)
        changes[g] = config_changes(oldc, c)
        entries[g] = {
            "deployment": str(d),
            "owner": str(d),
            "old_owner": str(oldowner),
            "predecessor": predecessor,
            "source": str(root),
            "run": str(run),
            "bench": bench,
            "method": method,
            "episodes": c["runner"]["max_steps"] * 16,
        }
    save(BASE / "prepared-dsrl.json", {"time": time.time(), "entries": entries})
    save(BASE / "config-diff.json", changes)
    save(BASE / "source-reuse.json", source_receipts)
    print(
        json.dumps(
            {
                "entries": entries,
                "source": source_receipts,
                "diff_fields": {g: len(v) for g, v in changes.items()},
            }
        )
    )


if __name__ == "__main__":
    main()
