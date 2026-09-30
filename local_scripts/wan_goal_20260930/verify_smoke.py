"""Read-only, CPU-only acceptance for pinned RLinf Wan/OFT or Pi05 smoke.

Usage: python verify_smoke.py --run-dir /data/.../oft-smoke [--output RECEIPT]
Reads training artifacts only; prints JSON and optionally writes a new receipt.
Starts no Ray/model and never initializes CUDA.
Exit 0: all requested evidence passes. Exit 2: incomplete/failed/unverified.

Source: RLinf d34d4c320d08cb982de034aa9a011f08dc0fa217:
  utils/metric_logger.py; runners/embodied_runner.py;
  hybrid_engines/fsdp/strategy/base.py; utils/metric_utils.py.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

# This is an offline reader. Do this before importing torch/TensorBoard.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

TAGS = {
    "grad_norm": "train/actor/grad_norm",
    "total_loss": "train/actor/total_loss",
    "loss_mask_fraction": "rollout/loss_mask_fraction",
    "advantages_min": "rollout/advantages_min",
    "advantages_max": "rollout/advantages_max",
}
EXPECTED_STEPS = (0, 1)  # Runner logs before increment, CP names after increment.


def inside(root, path):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Evidence path leaves the requested run directory: {path}")
    return resolved


def read_metrics(run):
    # Aggregate logger is tensorboard/all with per_worker_log, otherwise root.
    choices = [run / "tensorboard" / "all", run / "tensorboard"]
    logdir = next((p for p in choices if list(p.glob("events.out.tfevents.*"))), None)
    if logdir is None:
        raise FileNotFoundError("No aggregate TensorBoard event files found")
    logdir = inside(run, logdir)
    # Avoid loading TensorFlow's CUDA stack through TensorBoard's lazy shim.
    sys.modules.setdefault("tensorflow", None)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    values, errors = {}, []
    for label, tag in TAGS.items():
        events = accumulator.Scalars(tag) if tag in available else []
        # If a writer reopened, latest wall_time for the same step is retained.
        by_step = {}
        for event in sorted(events, key=lambda item: item.wall_time):
            if event.step in EXPECTED_STEPS:
                by_step[event.step] = float(event.value)
        values[label] = by_step
        for step in EXPECTED_STEPS:
            value = by_step.get(step)
            if value is None:
                errors.append(f"Missing {tag} at step {step}")
            elif not math.isfinite(value):
                errors.append(f"Non-finite {tag} at step {step}")

    evidence = []
    for step in EXPECTED_STEPS:
        row = {label: by_step.get(step) for label, by_step in values.items()}
        complete = all(value is not None and math.isfinite(value) for value in row.values())
        valid = complete and (
            row["grad_norm"] > 0
            and 0 < row["loss_mask_fraction"] <= 1
            and max(abs(row["advantages_min"]), abs(row["advantages_max"])) > 0
        )
        if complete and not valid:
            errors.append(f"No confirmed nonzero RL gradient/effective samples at step {step}")
        evidence.append({"tensorboard_step": step, "checkpoint_step": step + 1,
                         "valid_rl_signal": valid, **row})

    config_path = inside(run, logdir / "config.yaml")
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    context = {
        "experiment_name": OmegaConf.select(cfg, "runner.logger.experiment_name"),
        "model_type": OmegaConf.select(cfg, "actor.model.model_type"),
        "train_expert_only": OmegaConf.select(cfg, "actor.model.openpi.train_expert_only", default=False),
        "adv_type": OmegaConf.select(cfg, "algorithm.adv_type"),
        "entropy_bonus": OmegaConf.select(cfg, "algorithm.entropy_bonus", default=0),
        "enable_sft_co_train": OmegaConf.select(cfg, "actor.enable_sft_co_train", default=False),
        "runner_max_epochs": OmegaConf.select(cfg, "runner.max_epochs"),
        "resume_dir": OmegaConf.select(cfg, "runner.resume_dir"),
        "save_full_model_weights": OmegaConf.select(cfg, "actor.fsdp_config.save_full_model_weights", default=True),
        "component_placement": OmegaConf.to_container(cfg.cluster.component_placement),
    }
    if context["adv_type"] != "grpo":
        errors.append("This acceptance script requires the GRPO recipe")
    if context["entropy_bonus"] != 0 or context["enable_sft_co_train"]:
        errors.append("Additional gradient objectives prevent attributing nonzero gradient to GRPO alone")
    if context["resume_dir"]:
        errors.append("This two-epoch fresh-smoke reader does not accept a resumed run")
    return {
        "ok": not errors, "errors": errors, "log_dir": str(logdir),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "context": context, "epochs": evidence,
        "event_files": [str(p) for p in sorted(logdir.glob("events.out.tfevents.*"))],
        "note": "Finite loss may equal zero after advantage centering; a nonzero loss scalar is not required.",
    }


def checkpoint_structure(run, experiment, step):
    actor = inside(run, run / experiment / "checkpoints" / f"global_step_{step}" / "actor")
    dcp_dir = inside(run, actor / "dcp_checkpoint")
    metadata_file = inside(run, dcp_dir / ".metadata")
    from torch.distributed.checkpoint import FileSystemReader

    # Only metadata from this explicitly selected, locally generated run is read.
    metadata = FileSystemReader(str(dcp_dir)).read_metadata()
    keys = list(metadata.state_dict_metadata)
    required_prefixes = ["fsdp_checkpoint.model", "fsdp_checkpoint.optimizers",
                         "fsdp_checkpoint.lr_schedulers", "fsdp_checkpoint.rng"]
    missing = [prefix for prefix in required_prefixes
               if not any(k == prefix or k.startswith(prefix + ".") for k in keys)]
    if missing:
        raise ValueError(f"Checkpoint metadata missing training state: {missing}")
    if not metadata.storage_data:
        raise ValueError("DCP metadata has no referenced storage")
    files = {}
    for storage in metadata.storage_data.values():
        file_path = inside(dcp_dir, dcp_dir / storage.relative_path)
        size = file_path.stat().st_size
        if storage.offset < 0 or storage.length <= 0 or storage.offset + storage.length > size:
            raise ValueError(f"Missing/truncated DCP storage extent: {file_path}")
        files[str(file_path)] = size
    full = inside(run, actor / "model_state_dict" / "full_weights.pt")
    if full.stat().st_size <= 0:
        raise ValueError("Empty full_weights.pt")
    return {
        "step": step, "ok": True, "actor_dir": str(actor),
        "format": "torch.distributed.checkpoint + full torch state_dict",
        "metadata_bytes": metadata_file.stat().st_size,
        "storage_files": files, "state_metadata_entries": len(keys),
        "full_weights": str(full), "full_weights_bytes": full.stat().st_size,
        "scope": "Metadata, every referenced shard byte extent, and readable full state_dict; no full-payload checksum or optimizer restore.",
    }


def policy_tensor_priority(name):
    """Prefer the d34 Pi05 trainable projections/expert-1, then OFT heads.

    Saved state_dict tensors do not retain requires_grad. Match the actual d34
    freeze_vlm module names; generic 'projector' also matches frozen vision.
    """
    lower = name.lower()
    if re.search(r"(?:^|\.)(?:action_(?:in|out)_proj|(?:action_)?time_mlp_(?:in|out))\.", lower):
        return (0, name)
    if (re.search(r"(?:^|\.)llm\.layers\.\d+\.(?:pre_attention_norms|pre_ffw_norms|mlps)\.1\.", lower)
            or re.search(r"(?:^|\.)llm\.layers\.\d+\.attn\.[qkvo]_proj\.1\.", lower)
            or re.search(r"(?:^|\.)llm\.final_norms\.1\.", lower)):
        return (1, name)
    if "action_head" in lower or "lm_head" in lower:
        return (2, name)
    if "projector" in lower:
        return (3, name)
    return (4 if "language_model" in lower else 5, name)


def compare_weights(first, second, max_tensors, max_elements, train_expert_only=False):
    import torch

    # mmap keeps the full model off RAM; map_location and hidden CUDA force CPU.
    before = torch.load(first, weights_only=True, mmap=True, map_location="cpu")
    after = torch.load(second, weights_only=True, mmap=True, map_location="cpu")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("Expected native full_weights.pt to be a state_dict")
    if set(before) != set(after):
        raise ValueError("Checkpoint state_dict key sets differ")
    candidates = []
    for name, left in before.items():
        right = after[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            continue
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Checkpoint tensor layout differs: {name}")
        if left.is_floating_point() and left.numel() and left.is_contiguous() and right.is_contiguous():
            candidates.append(name)

    # For this Pi05 recipe, frozen img/embedder/expert-0 must not consume the
    # bounded sample budget. Projection modules and expert-1 stay trainable.
    if train_expert_only:
        candidates = [name for name in candidates if policy_tensor_priority(name)[0] <= 1]
        if not candidates:
            raise ValueError("No d34 Pi05 trainable projection/expert-1 names found; inspect checkpoint naming")
    candidates.sort(key=policy_tensor_priority)
    rows = []
    for name in candidates[:max_tensors]:
        left, right = before[name].view(-1), after[name].view(-1)
        count = min(left.numel(), max_elements)
        indices = torch.linspace(0, left.numel() - 1, steps=count, dtype=torch.int64)
        x, y = left[indices].float(), right[indices].float()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError(f"Non-finite checkpoint parameter samples: {name}")
        delta = (y - x).abs()
        rows.append({"name": name, "shape": list(before[name].shape),
                     "dtype": str(before[name].dtype), "elements_sampled": count,
                     "changed_samples": int(torch.count_nonzero(delta)),
                     "max_abs_change": float(delta.max())})
    changed = any(row["changed_samples"] > 0 for row in rows)
    return {
        "ok": changed, "comparison": "global_step_1 -> global_step_2",
        "method": "CPU mmap; deterministic evenly spaced samples in matching floating tensors",
        "candidate_scope": "d34 Pi05 trainable projections and expert-1" if train_expert_only else "policy projections/heads preferred, then remaining floating tensors",
        "eligible_tensors": len(candidates),
        "sampled_tensors": rows,
        "error": None if changed else "No sampled change found; this does not prove all parameters unchanged.",
        "limitation": "A sampled delta alone can be weight decay. Acceptance also requires finite positive gradient, positive loss-mask fraction and nonzero valid advantage at both logged epochs. No baseline->CP1 comparison is claimed.",
    }


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment", "--experiment-name", dest="experiment_name", default=None)
    parser.add_argument("--output", help="Write a new JSON receipt; an existing file is never overwritten")
    parser.add_argument("--exit-receipt", help="Optional owned JSON with exit_code and outcome, e.g. wm-exit.json")
    parser.add_argument("--max-tensors", type=int, default=16)
    parser.add_argument("--max-elements", type=int, default=8192)
    args = parser.parse_args()
    if not 1 <= args.max_tensors <= 64 or not 1 <= args.max_elements <= 65536:
        parser.error("Bounded sampling requires max-tensors 1..64 and max-elements 1..65536")
    run = Path(args.run_dir).resolve(strict=True)
    report = {"run_dir": str(run), "audited_upstream_commit": "d34d4c320d08cb982de034aa9a011f08dc0fa217",
              "read_only": True, "cuda_visible_devices": "", "errors": []}
    try:
        metrics = read_metrics(run)
        report["metrics"] = metrics
        experiment = args.experiment_name or metrics["context"]["experiment_name"]
        if not isinstance(experiment, str) or Path(experiment).name != experiment:
            raise ValueError("experiment_name must be one path component")
        checkpoints = [checkpoint_structure(run, experiment, step) for step in (1, 2)]
        report["checkpoints"] = checkpoints
        report["weight_change"] = compare_weights(
            checkpoints[0]["full_weights"], checkpoints[1]["full_weights"],
            args.max_tensors, args.max_elements,
            train_expert_only=bool(metrics["context"]["train_expert_only"]))
        report["process_exit_confirmed"] = None
        exit_receipt = Path(args.exit_receipt) if args.exit_receipt else run / "wm-exit.json"
        if args.exit_receipt or exit_receipt.is_file():
            receipt = json.loads(exit_receipt.resolve(strict=True).read_text())
            report["exit_receipt"] = str(exit_receipt.resolve())
            report["process_exit_confirmed"] = (receipt.get("exit_code") == 0
                                                 and receipt.get("outcome") == "completed")
            if not report["process_exit_confirmed"]:
                report["errors"].append("Provided process receipt does not confirm normal completion")
        report["ok"] = bool(metrics["ok"] and report["weight_change"]["ok"] and not report["errors"])
    except Exception as exc:
        report["ok"] = False
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "VALID_GRPO_SMOKE_EVIDENCE" if report["ok"] else "INCOMPLETE_OR_FAILED_EVIDENCE"
    report["scope"] = "Two completed logged training epochs and saved CP1/CP2; not LIBERO success rate, whole-checkpoint checksum, resource release, or process exit without a receipt."
    serialized = json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        try:
            with Path(args.output).open("x", encoding="utf-8") as output:
                output.write(serialized + "\n")
        except Exception as exc:
            report["ok"] = False
            report["status"] = "INCOMPLETE_OR_FAILED_EVIDENCE"
            report["errors"].append(f"Cannot write new receipt: {type(exc).__name__}: {exc}")
            serialized = json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
