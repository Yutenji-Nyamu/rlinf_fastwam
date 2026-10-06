"""Read-only CPU audit of a single-rank DSRL smoke/formal checkpoint.

Usage: python audit_checkpoint.py /.../global_step_2 --role u
       python audit_checkpoint.py /.../global_step_3 --role u --previous /.../global_step_2

No model, environment, Ray worker or process group is created. This script reads
trusted checkpoints produced by this experiment; it never writes a checkpoint.
The actor shard includes Python/NumPy RNG state, so its trusted local pickle is
loaded with weights_only=False, mmap=True and an explicit CPU map_location.
"""

import argparse
from contextlib import redirect_stdout
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys

# Hide devices before importing Torch, including when run beside active workers.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlinf.algorithms.dsrl_ugrow import U_SPEC, build_dsrl_u_weights, validate_signal_spec


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_snapshot(root):
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def tensor_leaves(value, prefix=""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, f"{prefix}[{index}]")


def check_finite(value, label):
    count = 0
    for name, tensor in tensor_leaves(value):
        require(tensor.device.type == "cpu", f"{label}.{name}: tensor is not on CPU")
        if tensor.is_floating_point() or tensor.is_complex():
            # Bounded temporary masks; no whole-replay FP32 copy.
            flat = tensor.reshape(-1)
            for offset in range(0, flat.numel(), 1_000_000):
                require(
                    bool(torch.isfinite(flat[offset:offset + 1_000_000]).all()),
                    f"{label}.{name}: nonfinite tensor",
                )
        count += 1
    return count


def tensor_digest(tensor):
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def phase_from_state(state, label):
    found = [value for key, value in state.items() if key.split(".")[-1] == "dsrl_policy_phase"]
    require(len(found) == 1, f"{label}: expected one dsrl_policy_phase buffer")
    require(found[0].numel() == 1, f"{label}: phase buffer is not scalar")
    phase = int(found[0].item())
    require(phase in (0, 1), f"{label}: invalid policy phase")
    return phase


def q_parameter(name):
    return any(part in {"critic_image_encoder", "critic_state_encoder", "q_head"}
               for part in name.split("."))


def canonical_shadow(shadow):
    """Match FSDP named_parameters names to canonical saved state_dict names.

    The trainer intentionally saves its live named_parameters keys, while
    get_model_state_dict removes FSDP wrapper path components. Strip only that
    exact wrapper token, and reject collisions instead of accepting a subset.
    """
    canonical = {}
    renamed = 0
    for name, value in shadow.items():
        key = ".".join(part for part in name.split(".")
                       if part != "_fsdp_wrapped_module")
        require(key not in canonical, f"Target shadow canonical-name collision: {key}")
        canonical[key] = value
        renamed += key != name
    return canonical, renamed


def audit_alpha(alpha_dir):
    """Load only the small tensor entries of the actual DCP alpha checkpoint."""
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    dcp_dir = alpha_dir / "dcp_checkpoint"
    require((dcp_dir / ".metadata").is_file(), "Missing alpha DCP .metadata")
    require(any(dcp_dir.glob("*.distcp")), "Missing alpha DCP data shard")
    reader = FileSystemReader(str(dcp_dir))
    metadata = reader.read_metadata()
    flat = {
        name: torch.empty(tuple(entry.size), dtype=entry.properties.dtype, device="cpu")
        for name, entry in metadata.state_dict_metadata.items()
        if isinstance(entry, TensorStorageMetadata)
    }
    require(bool(flat), "Alpha checkpoint contains no tensor entries")
    # Keep stdout as one JSON object even if the installed DCP prints a notice.
    with redirect_stdout(sys.stderr):
        load(
            state_dict=flat,
            storage_reader=reader,
            planner=DefaultLoadPlanner(flatten_state_dict=False, allow_partial_load=True),
        )
    finite_count = check_finite(flat, "alpha")
    base = [value for name, value in flat.items() if name.endswith("model.base_alpha")]
    require(len(base) == 1 and base[0].numel() == 1, "Alpha base parameter missing or malformed")
    alpha = float(torch.nn.functional.softplus(base[0].float()).item())
    return {"tensor_count": finite_count, "all_tensor_entries_finite": True, "alpha": alpha}


def audit_replay(replay_path, role, runner_step, trainer_contract):
    replay = torch.load(replay_path, map_location="cpu", weights_only=True, mmap=True)
    expected_schema = 2 if role == "u" else 1
    require(replay.get("schema_version") == expected_schema, "Replay role/schema mismatch")
    require(replay.get("rank") == 0 and replay.get("world_size") == 1, "Replay is not single-rank")
    capacity = int(replay["global_capacity"])
    require(capacity == 25000 and replay["local_capacity"] == capacity, "Replay capacity changed")
    resident, total, cursor = (int(replay[key]) for key in (
        "resident_size", "total_inserted", "write_cursor"
    ))
    require(0 < resident <= capacity, "Checkpoint replay must contain smoke transitions")
    require(total >= resident and 0 <= cursor < capacity, "Invalid replay counters")
    require(resident == min(total, capacity), "Replay size does not match insertion count")
    storage = replay.get("storage")
    require(isinstance(storage, dict), "Missing replay storage")
    required = {"curr_obs", "next_obs", "actions", "rewards", "continuations",
                "terminations", "truncations", "discounts"}
    if role == "u":
        required.update({"dsrl_u", "dsrl_u_valid", "dsrl_u_policy_step", "dsrl_u_phase"})
        signal_spec = validate_signal_spec(replay.get("u_spec"))
        require(signal_spec == U_SPEC == trainer_contract["signal_spec"], "Replay U spec differs")
    else:
        require(replay.get("u_spec") is None, "Clean replay unexpectedly contains a U spec")
    require(set(storage) == required, "Replay storage keys differ from role schema")
    for name, tensor in tensor_leaves(storage):
        require(tensor.ndim > 0 and tensor.shape[0] == capacity, f"Replay {name}: capacity shape mismatch")

    for obs in ("curr_obs", "next_obs"):
        require(set(storage[obs]) == {"main_images", "states"}, f"Unexpected {obs} keys")
        require(storage[obs]["main_images"].shape == (capacity, 3, 64, 64), f"Invalid {obs} image shape")
        require(storage[obs]["states"].shape == (capacity, 14), f"Invalid {obs} state shape")
        require(storage[obs]["main_images"].dtype == torch.bfloat16, f"Invalid {obs} image dtype")
        require(storage[obs]["states"].dtype == torch.float32, f"Invalid {obs} state dtype")
    require(storage["actions"].shape == (capacity, 32), "Replay actions are not compact latent32")
    require(storage["actions"].dtype == torch.bfloat16, "Compact replay latent must be BF16")
    for field in ("rewards", "continuations", "terminations", "truncations", "discounts"):
        require(storage[field].shape == (capacity, 1), f"Invalid replay {field} shape")
    for field in ("continuations", "terminations", "truncations"):
        require(storage[field].dtype == torch.bool, f"Replay {field} must be bool")
    for field in ("rewards", "discounts"):
        require(storage[field].dtype == torch.float32, f"Replay {field} must be FP32")
    term = storage["terminations"][:resident]
    require(torch.equal(storage["continuations"][:resident], ~term), "Continuation/success mismatch")
    expected_reward = torch.where(term, 0.0, -1.0)
    require(torch.equal(storage["rewards"][:resident], expected_reward), "Macro reward changed")
    require(torch.allclose(storage["discounts"][:resident],
                           torch.full((resident, 1), 0.999 ** 10)), "C10 discount changed")
    # Validate only resident data; unused ring capacity is intentionally zero-filled.
    resident_tree = {
        key: ({sub: value[:resident] for sub, value in entry.items()}
              if isinstance(entry, dict) else entry[:resident])
        for key, entry in storage.items()
    }
    finite_count = check_finite(resident_tree, "replay")
    rng = replay["rng_state"]
    require(rng.dtype == torch.uint8 and rng.ndim == 1, "Malformed replay RNG state")
    generator = torch.Generator(device="cpu")
    generator.set_state(rng)
    restored_rng = generator.get_state()
    require(torch.equal(rng, restored_rng), "Replay RNG did not restore exactly")
    next_indices = torch.randint(0, resident, (16,), generator=generator, device="cpu")
    # Reset and verify the same next draw without mutating stored replay or global RNG.
    generator.set_state(rng)
    require(torch.equal(next_indices, torch.randint(0, resident, (16,), generator=generator)),
            "Replay RNG next draw is not reproducible")
    result = {
        "schema_version": expected_schema, "global_capacity": capacity,
        "resident_size": resident, "total_inserted": total, "write_cursor": cursor,
        "seed": int(replay["seed"]), "resident_tensor_count": finite_count,
        "all_resident_tensors_finite": True, "rng_restored_exactly": True,
        "rng_sha256": tensor_digest(rng), "next_draw_sha256": tensor_digest(next_indices),
    }
    if role == "u":
        raw, valid = storage["dsrl_u"][:resident], storage["dsrl_u_valid"][:resident]
        require(raw.shape == valid.shape == (resident, 10), "U action scores need [resident,10]")
        require(raw.dtype == torch.float32 and valid.dtype == torch.bool, "U score/mask dtype mismatch")
        # This frozen producer records the submitted prefix; no fabricated tail mask.
        require(bool(valid.all()), "Submitted-prefix U spec requires all ten recorded positions valid")
        require(bool(((raw >= 0) & (raw <= 1.000001)).all()), "U relative disagreement outside [0,1]")
        step, phase = storage["dsrl_u_policy_step"][:resident], storage["dsrl_u_phase"][:resident]
        for label, value in (("step", step), ("phase", phase)):
            require(value.shape == (resident, 1) and value.dtype == torch.int64,
                    f"U {label} provenance must be int64 [resident,1]")
        require(bool((step >= 0).all()) and int(step.max()) <= runner_step, "Invalid U collection version")
        require(bool(((phase == 0) | (phase == 1)).all()), "Invalid U collection phase")
        # Replay's mean-one mapping is batch-dependent. Diagnose exactly GB256,
        # drawn from the saved RNG, rather than treating the whole pool as a batch.
        generator.set_state(rng)
        batch_indices = torch.randint(0, resident, (256,), generator=generator)
        _, metrics = build_dsrl_u_weights(
            raw[batch_indices], valid[batch_indices],
            temperature=trainer_contract["temperature"],
            log_eps=trainer_contract["log_eps"], minmax_eps=trainer_contract["minmax_eps"],
        )
        result["u"] = {
            "signal_spec": signal_spec,
            "raw_mean": float(raw.mean()), "raw_max": float(raw.max()),
            "policy_step_min": int(step.min()), "policy_step_max": int(step.max()),
            "gaussian_rows": int((phase == 0).sum()), "learned_rows": int((phase == 1).sum()),
            "saved_rng_batch256_weights": metrics,
        }
    return result


def audit(checkpoint, role, expected_runner_step=None):
    root = checkpoint.resolve(strict=True)
    if root.name == "actor":
        root = root.parent
    match = re.fullmatch(r"global_step_(\d+)", root.name)
    require(match is not None, "Checkpoint root must be global_step_N or its actor child")
    runner_step = int(match.group(1))
    if expected_runner_step is not None:
        require(runner_step == expected_runner_step, "Unexpected checkpoint runner step")
    actor = root / "actor"
    before = file_snapshot(root)
    require(not any(name.endswith(".tmp") for name in before), "Checkpoint contains unfinished .tmp files")
    paths = {
        "online": actor / "local_shard_checkpoint/checkpoint_rank_0.pt",
        "target": actor / "sac_components/target_model/checkpoint_rank_0.pt",
        "trainer": actor / "sac_components/dsrl_trainer_state_rank_0.pt",
        "replay": actor / "sac_components/replay_buffer/rank_0/dsrl_transition_replay.pt",
    }
    for name, path in paths.items():
        require(path.is_file() and path.stat().st_size > 0, f"Missing {name} checkpoint component")
    require(not list(actor.rglob("checkpoint_rank_1.pt")), "Unexpected second-rank checkpoint")
    trainer = torch.load(paths["trainer"], map_location="cpu", weights_only=True, mmap=True)
    require(trainer.get("schema_version") == 1, "Trainer sidecar schema changed")
    require(trainer.get("rank") == 0 and trainer.get("world_size") == 1, "Trainer is not single-rank")
    require(trainer.get("flat_replay") is True, "Trainer is not compact-replay DSRL")
    update_step, phase, pending = (int(trainer[key]) for key in (
        "update_step", "policy_phase", "pending_local_new_transitions"
    ))
    require(update_step >= 0 and phase in (0, 1) and pending >= 0, "Invalid trainer counters")
    saved_runner_step = trainer.get("saved_runner_step")
    if saved_runner_step is not None:
        require(saved_runner_step == runner_step, "Trainer/path runner step mismatch")
    contract = trainer.get("dsrl_u_contract")
    if role == "u":
        require(isinstance(contract, dict), "U trainer contract missing")
        require(validate_signal_spec(contract.get("signal_spec")) == U_SPEC, "U trainer spec changed")
        for key, expected in {
            "temperature": 2.5, "log_eps": 1e-12, "minmax_eps": 1e-6,
            "scope": "all_replay_chunks_global_batch", "application": "whole_actor_sac_loss",
            "mapping": "mean_log_minmax_exp_mean", "global_batch_size": 256,
        }.items():
            require(contract.get(key) == expected, f"U trainer contract changed: {key}")
    else:
        require(contract is None, "Clean trainer unexpectedly contains a U contract")

    online = torch.load(paths["online"], map_location="cpu", weights_only=False, mmap=True)
    require({"model", "optimizers", "lr_schedulers", "rng", "fsdp_version"} <= set(online),
            "Incomplete online training shard")
    require(isinstance(online["optimizers"], list) and len(online["optimizers"]) == 2,
            "Expected actor and critic optimizer states")
    require(len(online["lr_schedulers"]) == 2 and bool(online["rng"]), "Scheduler/RNG state missing")
    online_phase = phase_from_state(online["model"], "online")
    # Check trainable SAC tensors and optimizer state. Frozen VLA tensor values
    # are outside this lightweight audit; bind the separate source manifest.
    sac_prefixes = ("dsrl_action_noise_net.", "actor_image_encoder.", "actor_state_encoder.",
                    "critic_image_encoder.", "critic_state_encoder.", "q_head.")
    sac_state = {name: value for name, value in online["model"].items()
                 if name.startswith(sac_prefixes)}
    for prefix in sac_prefixes:
        require(any(name.startswith(prefix) for name in sac_state),
                f"Online shard is missing SAC component {prefix}")
    online_finite = check_finite(sac_state, "online_sac")
    optimizer_finite = check_finite(online["optimizers"], "optimizers")
    optimizer_steps = []
    for optimizer in online["optimizers"]:
        require(bool(optimizer.get("state")), "Optimizer has no saved state")
        steps = [int(value["step"].item()) for value in optimizer["state"].values()
                 if isinstance(value, dict) and isinstance(value.get("step"), torch.Tensor)]
        optimizer_steps.append({"min": min(steps) if steps else None, "max": max(steps) if steps else None})
    target = torch.load(paths["target"], map_location="cpu", weights_only=True, mmap=True)
    require(set(target) == set(online["model"]), "Online/target model inventories differ")
    for name, value in sac_state.items():
        require(target[name].shape == value.shape and target[name].dtype == value.dtype,
                f"Online/target SAC layout differs: {name}")
    target_phase = phase_from_state(target, "target")
    require(online_phase == target_phase == phase, "Online/target/sidecar phase mismatch")
    shadow = trainer.get("target_shadow_f32")
    require(isinstance(shadow, dict) and bool(shadow), "FP32 target shadow missing")
    shadow, renamed_shadow_names = canonical_shadow(shadow)
    expected_q = {name for name in target if q_parameter(name)}
    require(set(shadow) == expected_q,
            "Target shadow names differ from target-Q after FSDP normalization: "
            f"missing={sorted(expected_q - set(shadow))[:8]}, "
            f"extra={sorted(set(shadow) - expected_q)[:8]}")
    for name, value in shadow.items():
        require(value.dtype == torch.float32 and value.shape == target[name].shape,
                f"Target shadow dtype/shape mismatch: {name}")
        require(bool(torch.isfinite(value).all()), f"Nonfinite target shadow: {name}")
        require(torch.equal(value.to(target[name].dtype), target[name]),
                f"FP32 shadow does not round to saved target: {name}")
    replay = audit_replay(paths["replay"], role, runner_step, contract)
    alpha = audit_alpha(actor / "sac_components/alpha")
    after = file_snapshot(root)
    require(before == after, "Checkpoint files changed during audit; retry after save finishes")
    native_markers = [name for name in before if Path(name).name == "complete.json"]
    result = {
        "ok": True, "checkpoint": str(root), "role": role,
        "complete": True, "complete_scope": "structural_cpu_audit_not_native_commit_marker",
        "native_complete_markers": native_markers,
        "saved_runner_step": saved_runner_step, "runner_step_from_path": runner_step,
        "update_step": update_step, "policy_phase": phase, "pending_new_transitions": pending,
        "file_count": len(before), "total_bytes": sum(stat[0] for stat in before.values()),
        "files_stable_during_audit": True,
        "online_sac_finite_tensors": online_finite, "optimizer_finite_tensors": optimizer_finite,
        "optimizer_steps": optimizer_steps, "target_shadow_tensors": len(shadow),
        "target_shadow_fsdp_names_normalized": renamed_shadow_names,
        "target_shadow_roundtrip_exact": True, "alpha": alpha, "replay": replay,
        "limits": ["No native DSRL completion marker or saved_runner_step in current source",
                   "Frozen VLA weights are not scanned for finiteness; bind the source manifest",
                   "Counter advancement alone is not proof that the driver resumed"]
    }
    del online, target, trainer
    gc.collect()
    return result


def continuation(previous, current):
    deltas = {
        "runner_steps": current["runner_step_from_path"] - previous["runner_step_from_path"],
        "optimizer_updates": current["update_step"] - previous["update_step"],
        "inserted_transitions": current["replay"]["total_inserted"] - previous["replay"]["total_inserted"],
    }
    consistent = (previous["role"] == current["role"] and deltas["runner_steps"] > 0
                  and deltas["optimizer_updates"] >= 0 and deltas["inserted_transitions"] > 0
                  and current["policy_phase"] >= previous["policy_phase"])
    if previous["policy_phase"] == 1:
        consistent = consistent and deltas["optimizer_updates"] > 0
    return {
        "previous_checkpoint": previous["checkpoint"], "deltas": deltas,
        "counters_consistent_with_continuation": consistent,
        "replay_rng_changed": previous["replay"]["rng_sha256"] != current["replay"]["rng_sha256"],
        "resume_execution_proven": False,
        "required_external_evidence": "Match driver resume_dir and startup restore counters to the previous checkpoint",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--role", choices=("clean", "u"), required=True)
    parser.add_argument("--expected-runner-step", type=int)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        result = audit(args.checkpoint, args.role, args.expected_runner_step)
        if args.previous:
            previous = audit(args.previous, args.role)
            result["continuation"] = continuation(previous, result)
            result["ok"] = result["continuation"]["counters_consistent_with_continuation"]
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["ok"] else 1
    except Exception as error:
        print(json.dumps({"ok": False, "complete": False, "checkpoint": str(args.checkpoint),
                          "role": args.role, "error_type": type(error).__name__, "error": str(error)[:1600]}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
