# Copyright 2026 The RLinf Authors. Licensed under the Apache License, Version 2.0.
"""Prepare only: reuse accepted C10 config, official limits and native seed loader.

Creates no processes or GPU contexts. Collection batches can later be run by
pi05_policy_eval/run_batch.py under the already accepted per-card GPU scope.
"""

import argparse
import copy
import json
from pathlib import Path


def prepare(reference, destination, gpu, tasks=None, batches=12):
    reference, destination = Path(reference), Path(destination)
    spec = json.loads((reference / "task-list.json").read_text())
    all_tasks = spec["tasks"] if isinstance(spec, dict) else spec
    template = json.loads((reference / "eval-template.json").read_text())
    train = json.loads((reference / "endpoint.json").read_text())
    destination.mkdir(parents=True, exist_ok=False)
    configs = []
    order = {}
    assert 1 <= batches <= 12
    output_root = Path(json.loads((reference / "prepared.json").read_text())["output"])
    # A disjoint, task-specific seed range; actual retries are checked too.
    for task_id, task in enumerate(all_tasks):
        name = task["task"]
        if tasks and name not in tasks:
            continue
        limit = int(task["step_limit"])
        forbidden = set()
        for path in reference.glob(f"eval-seeds*/{name}.json"):
            value = json.loads(path.read_text())
            forbidden.update(int(v) for v in value[name]["success_seeds"])
        for path in output_root.glob(f"batches/*/{name}/seeds.json"):
            forbidden.update(int(v["actual"]) for v in json.loads(path.read_text()))
        for batch in range(batches):
            seed_path = destination / f"{name}-b{batch}-seeds.json"
            # Native reset may advance a rejected seed. Separate slots so those
            # retries cannot immediately collide with the next environment.
            seeds = [
                10000000 + task_id * 100000 + batch * 1000 + slot * 100
                for slot in range(8)
            ]
            seed_path.write_text(
                json.dumps(
                    {
                        name: {
                            "success_seeds": seeds,
                            "note": "legacy loader field; unfiltered collection seeds, not success-selected",
                        }
                    }
                )
            )
            c = copy.deepcopy(template)
            c.update(
                suite="pi05_distillation_multitask",
                task=name,
                global_task_id=task_id,
                num_envs=8,
                gpu=gpu,
                step_limit=limit,
                official_step_limit=limit,
                num_steps=10,
                execute_chunk=10,
                policy="weak",
                condition=f"fql-{name}-b{batch}",
                batch=batch,
                output=str(destination / "batches" / f"{name}-b{batch}"),
                collect_transitions=True,
                collection_failure_log=str(
                    destination.parent / "eval-fql-collection/progress.json"
                ),
                recording_mode="none",
                record_full_video=False,
                record_decisions=False,
                collect_weak_u=False,
                record_signals=False,
                noise_seed=821026 + 1000003 * batch + 1009 * task_id,
                forbidden_actual_seeds=sorted(forbidden),
                warmup_prompt=name.replace("_", " "),
            )
            for key in ("distill_weights", "strong_weights", "weak_weights"):
                c.pop(key, None)
            c["model"] = copy.deepcopy(train["model"])
            c["env"].update(
                total_num_envs=8,
                seeds_path=str(seed_path),
                max_steps_per_rollout_epoch=limit,
                max_episode_steps=limit,
                auto_reset=False,
                ignore_terminations=True,
            )
            c["env"]["task_config"].update(
                task_name=name, step_lim=limit, save_path=str(Path(c["output"]) / "env")
            )
            path = destination / f"{name}-b{batch}.json"
            path.write_text(json.dumps(c, indent=2))
            configs.append(str(path))
            order[str(path)] = (batch, task_id)
    configs.sort(key=order.__getitem__)
    train.update(
        method="fql_fixed_teacher",
        horizon=50,
        execute_chunk=10,
        teacher_steps=10,
        transition_manifest=str(destination / "transition-manifest.json"),
        epochs=10,
        q_width=512,
        q_depth=4,
        q_lr=3e-4,
        discount=0.99,
        target_tau=0.005,
        alpha=10.0,
        checkpoint_keep_steps=[],
        checkpoint_keep_last=2,
        log_every=10,
        budget_resolved=False,
        checkpoint_steps=0,
    )
    train.pop("data_manifest", None)
    (destination / "fql.json").write_text(json.dumps(train, indent=2))
    (destination / "collection-plan.json").write_text(
        json.dumps(
            {
                "configs": configs,
                "launch": False,
                "recovery": "use accepted finite batch timeout/owned-process cleanup; no automatic launch here",
                "note": "Training/evaluation schedule must be resolved from completed real data, not HF window count",
            },
            indent=2,
        )
    )
    return configs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--gpu", required=True, type=int)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--batches", type=int, default=12)
    a = p.parse_args()
    print(json.dumps(prepare(a.reference, a.destination, a.gpu, a.tasks, a.batches)))


if __name__ == "__main__":
    main()
