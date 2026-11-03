# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Accepted SAC runner, namespace cleanup and rolling full checkpoints."""

import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def main():
    d = Path(os.environ["DOJO_GRPO_DEPLOYMENT"])
    plan = json.loads((d / "plan.json").read_text())
    phase = sys.argv[sys.argv.index("--config-name") + 1]
    if phase == "formal" and plan.get("defer_formal_to"):
        (d / "formal-deferred.json").write_text(
            json.dumps({"time": time.time(), "next": plan["defer_formal_to"]})
        )
        return
    if phase == "smoke" and not (d / "model-contract.json").exists():
        subprocess.run(
            [
                plan["python"],
                "-u",
                "-B",
                str(
                    Path(plan["training_source"])
                    / "tools/dojo_offpolicy/model_contract.py"
                ),
            ],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(plan["gpu"])),
            check=True,
            timeout=600,
        )
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    sys.path.insert(0, plan["training_source"])
    from rlinf.runners.embodied_runner import EmbodiedRunner

    original_save = EmbodiedRunner._save_checkpoint
    original_evaluate = EmbodiedRunner.evaluate

    def save_latest(self):
        original_save(self)
        root = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "checkpoints"
        ).resolve()
        current = root / f"global_step_{self.global_step}"
        assert current.is_dir()
        (current / "dojo-complete.json").write_text(
            json.dumps({"step": self.global_step, "time": time.time()})
        )
        removed = []
        for old in root.glob("global_step_*"):
            if old == current or not (old / "dojo-complete.json").is_file():
                continue
            assert old.resolve().is_relative_to(root)
            for f in old.rglob("*"):
                if (
                    f.is_file()
                    and not f.is_symlink()
                    and f.stat().st_size >= 1024 * 1024
                ):
                    removed.append({"path": str(f), "bytes": f.stat().st_size})
                    f.unlink()
        with (d / "retention.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {"time": time.time(), "kept": str(current), "removed": removed}
                )
                + "\n"
            )

    def evaluate(self):
        metrics = original_evaluate(self)
        with (d / "evaluations.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "phase": phase,
                        "round": self.global_step,
                        "time": time.time(),
                        "metrics": metrics,
                    },
                    default=float,
                )
                + "\n"
            )
        return metrics

    EmbodiedRunner._save_checkpoint = save_latest
    EmbodiedRunner.evaluate = evaluate
    original_run = EmbodiedRunner.run

    def run(self):
        if phase == "formal" and self.global_step == 0:
            self.update_rollout_weights()
            metrics = self.evaluate()
            self.metric_logger.log(
                data={f"eval/{k}": v for k, v in metrics.items()}, step=0
            )
        return original_run(self)

    def progress(self, step):
        metrics = {}
        if phase == "formal" and (
            self.global_step % self.cfg.runner.val_check_interval == 0
            or self.global_step == self.max_steps
        ):
            self.update_rollout_weights()
            metrics = {f"eval/{k}": v for k, v in self.evaluate().items()}
            self.metric_logger.log(data=metrics, step=step)
        if (
            metrics
            or self.global_step % self.cfg.runner.save_interval == 0
            or self.global_step == self.max_steps
        ):
            self._save_checkpoint()
        if metrics:
            scores = [
                float(v) for k, v in metrics.items() if k.endswith("success_once")
            ]
            best_path = Path(plan["run"]) / "best-evaluation.json"
            best = (
                json.loads(best_path.read_text()).get("success", -1)
                if best_path.exists()
                else -1
            )
            if scores and scores[0] > best:
                # The VLA is immutable. Protect just the inference noise policy
                # and phase, with its exact base model/config; no extra VLA copy.
                import torch

                current = (
                    Path(self.cfg.runner.logger.log_path)
                    / self.cfg.runner.logger.experiment_name
                    / "checkpoints"
                    / f"global_step_{self.global_step}"
                )
                state = torch.load(
                    current / "actor/local_shard_checkpoint/checkpoint_rank_0.pt",
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )["model"]
                prefixes = (
                    "dsrl_action_noise_net.",
                    "actor_image_encoder.",
                    "actor_state_encoder.",
                )
                adapter = {
                    k: v.detach().clone()
                    for k, v in state.items()
                    if k.startswith(prefixes) or k == "dsrl_policy_phase"
                }
                assert "dsrl_policy_phase" in adapter and len(adapter) > 1
                output = (
                    Path(plan["protected_output"]) / f"global_step_{self.global_step}"
                )
                output.mkdir(parents=True, exist_ok=True)
                torch.save(adapter, output / "noise-policy.pt")
                receipt = {
                    "round": self.global_step,
                    "success": scores[0],
                    "base_model": plan["model"],
                    "protected": str(output),
                    "task": "put_bottles_into_dustbin",
                    "usage": "get_model(formal.actor.model), then load_state_dict(noise_policy, strict=False); frozen base and its assets required",
                }
                (output / "manifest.json").write_text(json.dumps(receipt, indent=2))
                best_path.write_text(json.dumps(receipt, indent=2))
        return metrics

    EmbodiedRunner.run = run
    EmbodiedRunner._maybe_eval_and_checkpoint = progress
    spec = importlib.util.spec_from_file_location(
        "accepted_clean_ops", plan["legacy_ops"]
    )
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    ops.RUN = Path(plan["run"]) / f"driver-{phase}-{os.getpid()}"
    (ops.RUN / "runtime").mkdir(parents=True, exist_ok=True)
    ops.ROOT = Path(plan["training_source"])
    ops.NAMESPACE = plan["namespace"] + "_" + phase + "_" + str(os.getpid())
    sys.argv = [sys.argv[0], "formal_driver", *sys.argv[1:]]
    ops.formal_driver()
    (d / f"{phase}-complete.json").write_text(
        json.dumps({"time": time.time(), "phase": phase})
    )


if __name__ == "__main__":
    main()
