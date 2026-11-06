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
    from omegaconf import OmegaConf

    config_path = d / "config" / (phase + ".yaml")
    cfg = OmegaConf.load(config_path)
    token_manifest = json.loads(
        (Path(plan["run"]) / "stage1/complete.json").read_text()
    )
    cfg.algorithm.rlt_resume.contract.stage1_token_sha256 = token_manifest[
        "token_sha256"
    ]
    OmegaConf.save(cfg, config_path)
    benchmark_complete = False
    if phase == "formal" and plan.get("acceleration_trial"):
        try:
            measured = json.loads((d / "sac-timings.json").read_text())
            applied = json.loads((d / "selected-acceleration.json").read_text())
            benchmark_complete = (
                "selected" in measured
                and measured["selected"] == applied.get("selected")
                and applied.get("scope", "").startswith("production SAC update")
            )
        except (OSError, ValueError):
            pass
    if phase == "formal" and plan.get("acceleration_trial") and not benchmark_complete:
        # A separate driver keeps the benchmark's Cluster singleton and namespace
        # out of the subsequent formal driver; terminate only this exact child.
        import signal

        def stop_benchmark(signum, frame):
            raise SystemExit(128 + signum)

        previous_handler = signal.signal(signal.SIGTERM, stop_benchmark)
        child = subprocess.Popen(
            [
                plan["python"],
                "-u",
                "-B",
                str(
                    Path(plan["training_source"])
                    / "tools/dsrl_fp32_accel/bench_driver.py"
                ),
                str(d),
            ]
        )
        try:
            assert child.wait(timeout=1800) == 0, "SAC benchmark failed"
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    child.kill()
            signal.signal(signal.SIGTERM, previous_handler)
    if phase == "formal" and plan.get("defer_formal_to"):
        (d / "formal-deferred.json").write_text(
            json.dumps({"time": time.time(), "next": plan["defer_formal_to"]})
        )
        return
    if (
        phase == "smoke"
        and plan.get("model_contract_required", True)
        and not (d / "model-contract.json").exists()
    ):
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
        if False:  # reference_eval covers the frozen VLA baseline
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
                output = (
                    Path(plan["protected_output"]) / f"global_step_{self.global_step}"
                )
                output.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {k: v.detach().clone() for k, v in state.items()},
                    output / "rlt-policy.pt",
                )
                receipt = {
                    "round": self.global_step,
                    "success": scores[0],
                    "protected": str(output),
                    "task": plan["task"],
                    "feature": self.cfg.rollout.rlt_feature_model.model_path,
                    "token": self.cfg.rollout.rlt_feature_model.openpi.rlt_token_path,
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
    cleanup = ops.cleanup_namespace

    def cleanup_with_retry(*args, **kwargs):
        # The local Ray dashboard briefly returned HTTP502 after a completed
        # CP2. Retry only its state-query exception, never a training failure.
        from ray.util.state.exception import RayStateApiException

        for attempt in range(3):
            try:
                return cleanup(*args, **kwargs)
            except RayStateApiException:
                if attempt == 2:
                    raise
                time.sleep(1)

    ops.cleanup_namespace = cleanup_with_retry
    sys.argv = [sys.argv[0], "formal_driver", *sys.argv[1:]]
    ops.formal_driver()
    (d / f"{phase}-complete.json").write_text(
        json.dumps({"time": time.time(), "phase": phase})
    )


if __name__ == "__main__":
    main()
