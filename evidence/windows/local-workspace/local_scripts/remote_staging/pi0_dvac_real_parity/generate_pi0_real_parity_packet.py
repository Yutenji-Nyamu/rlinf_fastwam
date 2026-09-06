from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


HEAD = "800baf80d6eab64169cf0e691eb04a681a093ee9"
BRANCH = "codex/sz-current-pi0-dvac-observe"
WT = Path(
    "/data/chenyiteng/projects/rlinf-shenzhen/worktrees/"
    "pi0-dvac-observe-7d07a421"
)
PYTHON = Path("/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python")
ROBOTWIN = Path(
    "/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support"
)
ROBOTWIN_HEAD = "0008ae6800df9f75fc8de7098bacb01735fd8fd2"
MODEL = Path(
    "/data/chenyiteng/models/rlinf/"
    "RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50"
)
PACKET = Path(
    "/data/chenyiteng/results/dvac-observation/packets/"
    "pi0-adjust_bottle-real-query-gate-a-800baf80-v1"
)
OUTPUT = Path(
    "/data/chenyiteng/results/dvac-observation/"
    "pi0-adjust_bottle-real-query-gate-a-800baf80-v1"
)
RESET_STATE_ID = 100100052
INFERENCE_SEED = 0
PHYSICAL_GPU = 2
TIMEOUT_SECONDS = 900
KILL_AFTER_SECONDS = 120


def run(*args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(args, text=True, env=env).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str, *, executable: bool = False) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    if executable:
        path.chmod(0o755)


def main() -> None:
    if PACKET.exists():
        raise FileExistsError(f"Refusing to overwrite packet: {PACKET}")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to target existing output: {OUTPUT}")
    if run("git", "-C", str(WT), "rev-parse", "HEAD") != HEAD:
        raise RuntimeError("RLinf HEAD drifted")
    if run("git", "-C", str(WT), "branch", "--show-current") != BRANCH:
        raise RuntimeError("RLinf branch drifted")
    if run("git", "-C", str(WT), "status", "--porcelain=v1"):
        raise RuntimeError("RLinf worktree is not clean")
    if run("git", "-C", str(WT), "rev-parse", "@{upstream}") != HEAD:
        raise RuntimeError("RLinf upstream drifted")
    remote_line = run(
        "git", "-C", str(WT), "ls-remote", "personal", f"refs/heads/{BRANCH}"
    )
    if remote_line.split()[0] != HEAD:
        raise RuntimeError("RLinf personal remote drifted")
    if run("git", "-C", str(ROBOTWIN), "rev-parse", "HEAD") != ROBOTWIN_HEAD:
        raise RuntimeError("RoboTwin compatibility source drifted")
    if not PYTHON.is_file() or not MODEL.is_dir():
        raise FileNotFoundError("Pinned runtime or model is missing")

    compose_args = [
        str(PYTHON),
        str(WT / "evaluations/eval_embodied_agent.py"),
        "--config-path",
        str(WT / "evaluations/robotwin"),
        "--config-name",
        "robotwin_adjust_bottle_openpi_dvac_eval",
        r"cluster.component_placement={env\,\ rollout:2}",
        f"runner.logger.log_path={OUTPUT}",
        f"rollout.model.model_path={MODEL}",
        f"env.eval.assets_path={ROBOTWIN}",
        "env.eval.total_num_envs=1",
        "env.eval.rollout_epoch=1",
        "env.eval.max_episode_steps=200",
        "env.eval.max_steps_per_rollout_epoch=200",
        "env.eval.use_fixed_reset_state_ids=true",
        "rollout.dvac_telemetry.enabled=false",
        "--cfg",
        "job",
        "--resolve",
    ]
    compose_env = os.environ.copy()
    compose_env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "ROBOTWIN_PATH": str(ROBOTWIN),
            "ROBOT_PLATFORM": "ALOHA",
            "REPO_PATH": str(WT),
            "EMBODIED_PATH": str(WT / "examples/embodiment"),
            "PYTHONPATH": str(WT),
            "OPENPI_DATA_HOME": "/home/chenyiteng/.cache/openpi",
            "MUJOCO_GL": "osmesa",
            "PYOPENGL_PLATFORM": "osmesa",
            "HYDRA_FULL_ERROR": "1",
        }
    )
    resolved = subprocess.check_output(compose_args, text=True, env=compose_env)
    cfg = OmegaConf.create(resolved)
    if int(cfg.env.eval.total_num_envs) != 1:
        raise RuntimeError("Resolved total_num_envs is not 1")
    if str(cfg.rollout.model.model_path) != str(MODEL):
        raise RuntimeError("Resolved model path drifted")
    if str(cfg.env.eval.assets_path) != str(ROBOTWIN):
        raise RuntimeError("Resolved RoboTwin assets path drifted")
    if not bool(cfg.env.eval.use_fixed_reset_state_ids):
        raise RuntimeError("Resolved fixed-reset contract is disabled")
    if bool(cfg.rollout.dvac_telemetry.enabled):
        raise RuntimeError("Gate A resolved config must not construct the writer")

    PACKET.parent.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir()
    resolved_path = PACKET / "resolved_config.yaml"
    compose_path = PACKET / "compose_command.sh"
    launch_path = PACKET / "launch_gate_a.sh"
    budget_path = PACKET / "budget.json"
    stop_path = PACKET / "STOP_CONDITIONS.md"
    manifest_path = PACKET / "manifest.json"

    write_text(resolved_path, resolved)
    compose_exports = [
        "export CUDA_VISIBLE_DEVICES=''",
        "export NVIDIA_VISIBLE_DEVICES=none",
        "export PYTHONNOUSERSITE=1",
        "export PYTHONDONTWRITEBYTECODE=1",
        f"export ROBOTWIN_PATH={shlex.quote(str(ROBOTWIN))}",
        "export ROBOT_PLATFORM=ALOHA",
        f"export REPO_PATH={shlex.quote(str(WT))}",
        f"export EMBODIED_PATH={shlex.quote(str(WT / 'examples/embodiment'))}",
        f"export PYTHONPATH={shlex.quote(str(WT))}",
        "export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi",
        "export MUJOCO_GL=osmesa",
        "export PYOPENGL_PLATFORM=osmesa",
        "export HYDRA_FULL_ERROR=1",
    ]
    write_text(
        compose_path,
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "\n".join(compose_exports)
        + "\n"
        + shlex.join(compose_args),
        executable=True,
    )

    launch = f"""#!/usr/bin/env bash
set -euo pipefail

WT={shlex.quote(str(WT))}
PY={shlex.quote(str(PYTHON))}
ROBOTWIN={shlex.quote(str(ROBOTWIN))}
MODEL={shlex.quote(str(MODEL))}
PACKET={shlex.quote(str(PACKET))}
OUTPUT={shlex.quote(str(OUTPUT))}
EXPECTED_HEAD={HEAD}
EXPECTED_BRANCH={BRANCH}

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git -C "$WT" branch --show-current)" = "$EXPECTED_BRANCH"
test -z "$(git -C "$WT" status --porcelain=v1)"
test "$(git -C "$WT" rev-parse '@{{upstream}}')" = "$EXPECTED_HEAD"
test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = {ROBOTWIN_HEAD}
test -x "$PY"
test -d "$MODEL"
test -f "$PACKET/resolved_config.yaml"
test ! -e "$OUTPUT"

export CUDA_VISIBLE_DEVICES={PHYSICAL_GPU}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export ROBOTWIN_PATH="$ROBOTWIN"
export ROBOT_PLATFORM=ALOHA
export REPO_PATH="$WT"
export EMBODIED_PATH="$WT/examples/embodiment"
export PYTHONPATH="$WT"
export OPENPI_DATA_HOME=/home/chenyiteng/.cache/openpi
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1

cd "$WT"
exec /usr/bin/time -v timeout --signal=INT --kill-after={KILL_AFTER_SECONDS}s {TIMEOUT_SECONDS}s \
  "$PY" toolkits/probe_pi0_dvac_real_parity.py run \
  --resolved-config "$PACKET/resolved_config.yaml" \
  --model-path "$MODEL" \
  --expected-head "$EXPECTED_HEAD" \
  --reset-state-id {RESET_STATE_ID} \
  --inference-seed {INFERENCE_SEED} \
  --output-dir "$OUTPUT"
"""
    write_text(launch_path, launch, executable=True)

    budget = {
        "gate": "pi0_real_query_off_on_parity",
        "task": "adjust_bottle",
        "fixed_reset_state_id": RESET_STATE_ID,
        "simulator_resets": 1,
        "policy_queries": 2,
        "denoising_steps_per_query": 4,
        "total_denoising_steps": 8,
        "environment_action_slots": 0,
        "train_trajectories": 0,
        "optimizer_updates": 0,
        "checkpoints_written": 0,
        "model_action_horizon": 50,
        "active_action_dim": 14,
        "physical_gpu": PHYSICAL_GPU,
        "gpu_count": 1,
        "timeout_seconds": TIMEOUT_SECONDS,
        "kill_after_seconds": KILL_AFTER_SECONDS,
    }
    write_text(budget_path, json.dumps(budget, indent=2, sort_keys=True))
    write_text(
        stop_path,
        """# Gate A stop conditions

## Before launch

- Stop if RLinf HEAD/branch/upstream is not the locked reviewed commit or the worktree is dirty.
- Stop if the RoboTwin compatibility commit, pinned runtime, model, resolved config, or GPU 2 differs.
- Stop if the exact output path already exists, GPU 2 has another compute process, or live RAM/disk is insufficient.

## During launch

- Stop this packet-owned process on traceback, CUDA OOM/illegal instruction, action/RNG parity failure, trace shape/formula failure, or the 900-second timeout.
- Preserve failure evidence; do not overwrite the output path and do not stop PPO or another user's process.

## Success

- Natural exit 0 and marker `PI0_DVAC_REAL_PARITY_OK`.
- `parity.json` and `parity.npz` exist under the new exact output path.
- All action, post-RNG, endpoint, shape, and `z=x-t*v` checks are true.
""",
    )

    artifact_hashes = {
        "resolved_config.yaml": sha256(resolved_path),
        "compose_command.sh": sha256(compose_path),
        "launch_gate_a.sh": sha256(launch_path),
        "budget.json": sha256(budget_path),
        "STOP_CONDITIONS.md": sha256(stop_path),
    }
    manifest = {
        "schema_version": 1,
        "status": "READY_FOR_REVIEW_NOT_EXECUTED",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "rlinf_commit": HEAD,
            "rlinf_branch": BRANCH,
            "rlinf_upstream_commit": HEAD,
            "rlinf_personal_remote_commit": HEAD,
            "robotwin_commit": ROBOTWIN_HEAD,
            "checkpoint_revision": "92684e50c8a3dcf13b76a06713e3152625967be1",
            "norm_stats_sha256": "649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a",
            "seed_file_sha256": "194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f",
        },
        "paths": {
            "worktree": str(WT),
            "python": str(PYTHON),
            "model": str(MODEL),
            "robotwin_assets": str(ROBOTWIN),
            "packet": str(PACKET),
            "output": str(OUTPUT),
        },
        "gate": {
            "task": "adjust_bottle",
            "reset_state_id": RESET_STATE_ID,
            "inference_seed": INFERENCE_SEED,
            "physical_gpu": PHYSICAL_GPU,
            "timeout_seconds": TIMEOUT_SECONDS,
            "kill_after_seconds": KILL_AFTER_SECONDS,
            "authorization": "NOT_EXECUTED_REQUIRES_EXPLICIT_APPROVAL",
        },
        "artifacts_sha256": artifact_hashes,
    }
    write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    hashes_path = PACKET / "SHA256SUMS"
    all_hashes = {**artifact_hashes, "manifest.json": sha256(manifest_path)}
    write_text(
        hashes_path,
        "\n".join(f"{digest}  {name}" for name, digest in sorted(all_hashes.items())),
    )

    print(f"PACKET={PACKET}")
    print(f"OUTPUT={OUTPUT}")
    print(f"RESOLVED_SHA256={artifact_hashes['resolved_config.yaml']}")
    print(f"LAUNCH_SHA256={artifact_hashes['launch_gate_a.sh']}")
    print(f"MANIFEST_SHA256={all_hashes['manifest.json']}")
    print(f"BUDGET_SHA256={artifact_hashes['budget.json']}")
    print(f"STOP_SHA256={artifact_hashes['STOP_CONDITIONS.md']}")
    print("CUDA_RAY_MODEL_SIM_USED=0")
    print("PI0_REAL_PARITY_PACKET_READY=1")


if __name__ == "__main__":
    main()
