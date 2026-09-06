set -eu

RLT_ROOT=/root/autodl-tmp/RLinf_rlt_pi0_robotwin
PYTHON_BIN=/root/autodl-tmp/RLinf/.venv/bin/python
EVIDENCE_ROOT=/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729

cd "$RLT_ROOT"
export PYTHONPATH="$RLT_ROOT:/root/autodl-tmp/RoboTwin_RLinf"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" -B - "$EVIDENCE_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import ray
from omegaconf import OmegaConf

import rlinf.config as config_module
import rlinf.models  # noqa: F401 - preserve production model-registration side effects


class NoRayCluster:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class StaticPlacement:
    available_gpus = 2

    def __init__(self, cfg, cluster):
        self.cfg = cfg
        self.cluster = cluster

    @classmethod
    def _count_spec(cls, spec) -> int:
        if isinstance(spec, int):
            return 1
        if isinstance(spec, (list, tuple)):
            return len(spec)
        text = str(spec).strip().lower()
        if text == "all":
            return cls.available_gpus
        if "-" in text:
            start, end = (int(value) for value in text.split("-", maxsplit=1))
            return end - start + 1
        if "," in text:
            return len([value for value in text.split(",") if value.strip()])
        int(text)
        return 1

    def get_world_size(self, component: str) -> int:
        placements = self.cfg.cluster.component_placement
        for raw_components, spec in placements.items():
            components = {
                item.strip() for item in str(raw_components).split(",")
            }
            if component in components:
                return self._count_spec(spec)
        raise KeyError(f"component {component!r} is absent from placement config")


evidence_root = Path(sys.argv[1])
resolved_files = sorted(evidence_root.glob("*_resolved.yaml"))
if not resolved_files:
    raise SystemExit(f"no resolved configs found under {evidence_root}")
if ray.is_initialized():
    raise RuntimeError("Ray was initialized before no-Ray validation")

summaries = []
with (
    patch.object(config_module, "Cluster", NoRayCluster),
    patch.object(
        config_module,
        "HybridComponentPlacement",
        StaticPlacement,
    ),
):
    for path in resolved_files:
        cfg = OmegaConf.load(path)
        validated = config_module.validate_cfg(cfg)
        if ray.is_initialized():
            raise RuntimeError(f"Ray initialized while validating {path.name}")
        placement = StaticPlacement(validated, NoRayCluster())
        summary = {
            "file": path.name,
            "task_type": str(validated.runner.task_type),
            "actor_world_size": placement.get_world_size("actor"),
            "global_batch_size": int(validated.actor.global_batch_size),
            "micro_batch_size": int(validated.actor.micro_batch_size),
        }
        if validated.runner.task_type == "embodied":
            summary.update(
                {
                    "loss_type": str(validated.algorithm.loss_type),
                    "env_world_size": placement.get_world_size("env"),
                    "train_total_num_envs": int(
                        validated.env.train.total_num_envs
                    ),
                    "num_action_chunks": int(
                        validated.actor.model.num_action_chunks
                    ),
                }
            )
        summaries.append(summary)

if ray.is_initialized():
    raise RuntimeError("Ray was initialized after no-Ray validation")
print(json.dumps({"ray_initialized": False, "validated": summaries}, indent=2))
PY
