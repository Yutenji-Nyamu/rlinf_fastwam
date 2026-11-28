"""Resume a clean RLT task after device-memory pressure on its assigned GPU."""
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from oom_classify_1127 import driver_failure, find_gpu_oom

_ABSOLUTE_TIME = re.compile(r"(?<!\d)(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d))")


def read_attempt(path, offset=0):
    with path.open("rb") as stream:
        stream.seek(offset)
        for line in stream:
            yield line.decode("utf-8", errors="replace")


def has_latest_replay_sample(replay):
    metadata = json.loads((replay / "metadata.json").read_bytes())
    index = json.loads((replay / "trajectory_index.json").read_bytes())
    order = index["trajectory_id_list"]
    if not order or metadata["size"] <= 0:
        return False
    # Older entries can remain indexed after they have rolled out of the cached window.
    latest = str(int(order[-1]))
    weights = index["trajectory_index"][latest]["model_weights_id"]
    extension = "pt" if metadata.get("trajectory_format", "pt") == "pt" else "pkl"
    sample = replay / f"trajectory_{latest}_{weights}.{extension}"
    return sample.is_file() and sample.stat().st_size > 0


def full_step(cp, local_shard, learned_alpha):
    """Both the runner and the single-rank RLT saver must have committed this step."""
    prefix, separator, number = cp.name.rpartition("_")
    if prefix != "global_step" or not separator or not number.isdecimal():
        return None
    step = int(number)
    actor = cp / "actor"
    sac = actor / "sac_components"
    replay = sac / "replay_buffer/rank_0"
    try:
        runner = json.loads((cp / "dojo-complete.json").read_bytes())
        trainer = json.loads((sac / "rlt_trainer_state/complete.json").read_bytes())
        if (runner["step"] != step or trainer["complete"] is not True
                or trainer["saved_runner_step"] != step or trainer["actor_world_size"] != 1):
            return None
        required = [sac / "rlt_trainer_state/checkpoint_rank_0.pt",
                    sac / "target_model/checkpoint_rank_0.pt"]
        local = actor / "local_shard_checkpoint/checkpoint_rank_0.pt"
        if learned_alpha:
            alpha = sac / "alpha/dcp_checkpoint"
            required.extend([alpha / ".metadata", alpha / "__0_0.distcp"])
        if local_shard:
            required.append(local)
        else:
            required.extend([actor / "dcp_checkpoint/.metadata", actor / "dcp_checkpoint/__0_0.distcp"])
        if all(f.is_file() and f.stat().st_size for f in required) and has_latest_replay_sample(replay):
            return step
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def select_retry(config):
    prior = config.runner.resume_dir
    logger = config.runner.logger
    root = Path(logger.log_path) / logger.experiment_name / "checkpoints"
    candidates = list(root.glob("global_step_*"))
    if prior:
        candidates.insert(0, Path(prior))
    local_shard = bool(config.actor.fsdp_config.use_orig_params)
    learned_alpha = config.algorithm.entropy_tuning.alpha_type != "fixed_alpha"
    named = [cp for cp in candidates if cp.name.startswith("global_step_")
             and cp.name.removeprefix("global_step_").isdecimal()]
    for cp in sorted(named, key=lambda p: int(p.name.rsplit("_", 1)[1]), reverse=True):
        if full_step(cp, local_shard, learned_alpha) is not None:
            return True, cp
    return not bool(prior), None


def socket_listening(path):
    """Wait for the local simulator to listen without connecting to its RL protocol."""
    if not path.is_socket():
        return False
    for line in Path("/proc/net/unix").read_text().splitlines()[1:]:
        fields = line.split(maxsplit=7)
        if len(fields) == 8 and fields[7] == str(path) and int(fields[3], 16) & 0x10000:
            return True
    return False


def dojo_refused(lines):
    from collections import deque
    last = deque(maxlen=18)
    for line in lines:
        if "ConnectionRefusedError: [Errno 111] Connection refused" in line:
            if any("robodojo_env.py" in prev and "in _rpc" in prev for prev in last):
                return {"kind": "dojo_local_connection_refused", "line": line.strip()}
        last.append(line)
    return None


def prepare_oom_retry(directory, output, native=None, *, budget=None, gpu=None, phase="formal"):
    """Return None for unrelated exits, False on handoff, True when a retry is ready."""
    from omegaconf import OmegaConf
    from common import save
    step, evidence, rpc, _ = driver_failure(read_attempt(output))
    plan = directory / "plan.json"
    immediate = plan.is_file() and json.loads(plan.read_text()).get("retry_sim_after_cleanup", False)
    if immediate and native and not evidence:
        evidence = dojo_refused(read_attempt(output))
    if not evidence and not (native and rpc):
        return None
    target = directory / "config/formal.yaml"
    config = OmegaConf.load(target)
    source = output.name
    if not evidence:
        lines = read_attempt(*native)
        if step is not None:
            # The native file spans the whole phase. Order its error against a known completed step.
            logger = config.runner.logger
            marker = Path(logger.log_path) / logger.experiment_name / "checkpoints" / f"global_step_{step}" / "dojo-complete.json"
            try:
                done = json.loads(marker.read_bytes())
                if done["step"] != step or not isinstance(done["time"], (int, float)):
                    return None
                lines = native_after(lines, done["time"])
            except (OSError, KeyError, TypeError, ValueError):
                return None
        evidence = find_gpu_oom(lines)
        if not evidence:
            return None
        source = native[0].name
    allowed, checkpoint = select_retry(config) if phase == "formal" else (True, None)
    if not allowed:
        return False
    receipt = dict(time=time.time(), phase=phase, source=source,
                   checkpoint=str(checkpoint) if checkpoint else None, kind=evidence["kind"],
                   start="checkpoint" if checkpoint else "initial", error=evidence["line"][-500:])
    if immediate and budget is not None:
        receipt.update(delay_seconds=0, decision={"reason": "restart_after_cleanup"})
        budget["delay"] = 0
    elif budget is not None:
        from oom_contention import retry_delay
        key = (phase, receipt["checkpoint"])
        if budget.get("key") != key:
            budget.update(key=key, retries=0, streak=0)
        evidence = dict(evidence, source=source, phase=phase, checkpoint=receipt["checkpoint"],
                        physical_gpu=int(gpu))
        delay, decision = retry_delay(evidence, [int(gpu)], budget["retries"], budget["streak"])
        receipt.update(decision=decision, delay_seconds=delay, prior_retries=budget["retries"])
        if delay is None:
            save(directory / "formal-oom-retry.json", receipt)
            return False
        budget.update(retries=budget["retries"] + 1, delay=delay,
                      streak=budget["streak"] + 1 if decision.get("external") else 0)
    if checkpoint is not None and config.runner.resume_dir != str(checkpoint):
        config.runner.resume_dir = str(checkpoint)
        temp = target.with_suffix(".oom-retry.tmp")
        OmegaConf.save(config, temp)
        os.replace(temp, target)
    save(directory / "formal-oom-retry.json", receipt)
    return True


def attempt_name(phase, attempt):
    if attempt == 0:
        return phase
    return phase + ("-retry" if attempt == 1 else f"-retry-{attempt}")


def native_after(lines, timestamp):
    for line in lines:
        stamp = _ABSOLUTE_TIME.search(line)
        if stamp:
            try:
                created = datetime.fromisoformat(stamp[1].replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            if created > timestamp:
                yield line

def drain_formal(directory):
    """Drain only processes already identified by the existing deployment owner."""
    from tools.dojo_offpolicy.owner import cleanup_owned
    from common import same
    previous = cleanup_owned(directory)
    deadline = time.monotonic() + 30
    while any(same(identity) for identity in previous):
        if time.monotonic() >= deadline:
            raise RuntimeError("Failed formal process still live after OOM cleanup")
        time.sleep(0.5)


def clear_inactive_socket(path):
    if not path.exists():
        return
    active = any(line.split(maxsplit=7)[-1] == str(path)
                 for line in Path("/proc/net/unix").read_text().splitlines()[1:])
    if active or not path.is_socket() or path.stat().st_uid != os.getuid():
        raise RuntimeError(f"Native socket still active or unexpected: {path}")
    path.unlink()

