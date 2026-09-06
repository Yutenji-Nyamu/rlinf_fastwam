from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
STAGING = WORKSPACE / "tmp" / "idea2_dvac_v1_formal_stop_g54_closeout_20260821"
OUTPUT = WORKSPACE / "exports" / "idea2_dvac_v1_formal_stop_g54_20260821.zip"
METHOD_SUMMARY = STAGING / "analysis" / "METHOD_STEP54_SUMMARY.json"
README = STAGING / "README.md"
MANIFEST = STAGING / "FILE_MANIFEST.csv"
SHA256SUMS = STAGING / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_key_values(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def read_marker(name: str) -> str:
    path = STAGING / "runtime" / name
    return path.read_text(encoding="utf-8").strip() if path.exists() else "not recorded"


def iso_duration(start: str, end: str) -> str:
    delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)
    total = int(delta.total_seconds())
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def method_summary() -> dict:
    rank_paths = sorted(
        (STAGING / "run" / "dvac_train").glob(
            "actor_rank*/rollout_step0053.npz"
        )
    )
    if len(rank_paths) != 2:
        raise RuntimeError(f"Expected two final rank shards, got {rank_paths}")

    required = {
        "v_l2",
        "v_l3",
        "v_l4",
        "weights",
        "clipped_z",
        "advantages",
        "loss_mask",
        "denoise_inds",
    }
    loaded = []
    schema: dict[str, dict[str, object]] = {}
    key_lists = []
    for path in rank_paths:
        shard = np.load(path)
        keys = list(shard.files)
        key_lists.append(keys)
        missing = required.difference(keys)
        if missing:
            raise RuntimeError(f"{path} is missing {sorted(missing)}")
        loaded.append({key: shard[key] for key in keys})
        schema[path.parent.name] = {
            key: {"shape": list(shard[key].shape), "dtype": str(shard[key].dtype)}
            for key in keys
        }
    if set(key_lists[0]) != set(key_lists[1]):
        raise RuntimeError("The two final rank shards have different key sets")

    horizon = int(loaded[0]["weights"].shape[-1])

    def query_rows(key: str) -> np.ndarray:
        return np.concatenate(
            [rank[key].reshape(-1, *rank[key].shape[2:]) for rank in loaded],
            axis=0,
        )

    weights = query_rows("weights")
    clipped_z = query_rows("clipped_z")
    v_l2 = query_rows("v_l2")
    v_l3 = query_rows("v_l3")
    v_l4 = query_rows("v_l4")
    loss_mask = np.concatenate(
        [rank["loss_mask"].reshape(-1) for rank in loaded], axis=0
    ).astype(bool)
    advantages = np.concatenate(
        [rank["advantages"].reshape(-1) for rank in loaded], axis=0
    )
    denoise_inds = np.concatenate(
        [rank["denoise_inds"].reshape(-1) for rank in loaded], axis=0
    )

    valid_w = weights[loss_mask]
    valid_z = clipped_z[loss_mask]
    valid_v2 = v_l2[loss_mask]
    valid_v3 = v_l3[loss_mask]
    valid_v4 = v_l4[loss_mask]
    valid_adv = advantages[loss_mask]
    if valid_w.size == 0:
        raise RuntimeError("Final shard has no loss-mask-valid queries")

    per_query_weight = valid_w.mean(axis=1)
    positive = valid_adv > 0
    negative = valid_adv < 0
    positive_mean = float(per_query_weight[positive].mean()) if positive.any() else None
    negative_mean = float(per_query_weight[negative].mean()) if negative.any() else None

    weight_q = np.quantile(valid_w, [0.05, 0.5, 0.95])
    z_q = np.quantile(valid_z, [0.05, 0.5, 0.95])
    per_h_weight = valid_w.mean(axis=0)
    per_h_v3_median = np.median(valid_v3, axis=0)
    front = valid_w[:, : horizon // 2].mean()
    back = valid_w[:, horizon // 2 :].mean()
    v_front = valid_v3[:, : horizon // 2].mean()
    v_back = valid_v3[:, horizon // 2 :].mean()

    summary = {
        "schema_version": 1,
        "global_step": 54,
        "runner_step_zero_based": 53,
        "source_shards": [path.relative_to(STAGING).as_posix() for path in rank_paths],
        "npz_keys": sorted(key_lists[0]),
        "npz_schema_by_rank": schema,
        "total_queries": int(weights.shape[0]),
        "valid_queries": int(loss_mask.sum()),
        "valid_weight_points": int(valid_w.size),
        "horizon": horizon,
        "weight": {
            "mean": float(valid_w.mean()),
            "std": float(valid_w.std()),
            "p05_median_p95": [float(value) for value in weight_q],
            "min_max": [float(valid_w.min()), float(valid_w.max())],
            "at_0_8_pct": float(np.isclose(valid_w, 0.8, atol=1e-6).mean() * 100),
            "at_1_2_pct": float(np.isclose(valid_w, 1.2, atol=1e-6).mean() * 100),
            "front_25_mean": float(front),
            "back_25_mean": float(back),
            "back_25_minus_front_25": float(back - front),
            "h0_h49_mean": [float(per_h_weight[0]), float(per_h_weight[-1])],
            "per_h_mean": [float(value) for value in per_h_weight],
        },
        "clipped_z": {
            "p05_median_p95": [float(value) for value in z_q],
            "at_low_clip_pct": float((valid_z <= -2.0).mean() * 100),
            "at_high_clip_pct": float((valid_z >= 2.0).mean() * 100),
        },
        "advantage_split": {
            "positive_queries": int(positive.sum()),
            "negative_queries": int(negative.sum()),
            "zero_queries": int((valid_adv == 0).sum()),
            "positive_mean_weight": positive_mean,
            "negative_mean_weight": negative_mean,
            "negative_minus_positive_mean_weight": (
                None
                if positive_mean is None or negative_mean is None
                else float(negative_mean - positive_mean)
            ),
        },
        "endpoint_variance": {
            "v_l2_median": float(np.median(valid_v2)),
            "v_l3_median": float(np.median(valid_v3)),
            "v_l4_median": float(np.median(valid_v4)),
            "v_l3_back_half_over_front_half_mean": float(v_back / v_front),
            "v_l3_per_h_median": [float(value) for value in per_h_v3_median],
        },
        "denoise_ind_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(denoise_inds, return_counts=True))
        },
    }
    METHOD_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def make_readme(method: dict) -> None:
    train = json_load(STAGING / "analysis" / "TRAIN_METRICS_SUMMARY.json")
    with (STAGING / "analysis" / "TRAIN_METRICS.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        final_train_row = list(csv.DictReader(handle))[-1]
    manifest = json_load(
        STAGING / "run" / "dvac_train" / "actor_rank00" / "run_manifest.json"
    )
    checkpoint = parse_key_values(STAGING / "CHECKPOINT_POINTER.txt")
    resources = parse_key_values(STAGING / "analysis" / "RESOURCE_SUMMARY.txt")
    start = read_marker("launch_started_at.txt")
    stop_request = read_marker("stop_requested_at.txt")
    terminate_request = read_marker("terminate_requested_at.txt")
    finish = read_marker("launch_finished_at.txt")
    driver_exit = read_marker("driver.exitcode")
    observer_exit = read_marker("observer_exit.txt").replace("\n", "; ")
    config_hash = sha256(STAGING / "config" / "resolved_config.yaml")
    source_config_hash = sha256(STAGING / "config" / "source_config.yaml")
    weight = method["weight"]
    split = method["advantage_split"]
    baseline = train["baseline_comparison_g1_to_latest"]
    cgroup_gib = int(resources["cgroup_peak_bytes"]) / (1024**3)

    text = f"""# Idea2 DVAC v1 formal closeout at Global Step 54

This is a lightweight evidence package for the user-authorized stop of the first DVAC-gradient-weighting formal run. It contains no checkpoint tensors, Ray session, TensorBoard event files, or full per-step NPZ collection.

## Run identity

- Experiment: `idea2_dvac_apply_formal_100step_2gpu16env_20260821`
- Task/model: RoboTwin `adjust_bottle`, task-matched RLinf Pi0 SFT checkpoint
- Planned/completed budget: 100 / 54 global steps; last complete zero-based runner step 53
- Source: RLinf `{manifest['rlinf_source']['commit']}` on `codex/idea2-dvac-train-weighting`; RoboTwin `{manifest['robotwin_source']['commit']}`
- Resolved-config SHA256: `{config_hash}`
- Source-config SHA256: `{source_config_hash}`
- Launch command: `runtime/launch_command.txt`
- Server run path: `/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821`

## Method contract (v1)

- Train-time signal: `V_L3(q,h)` from the existing four-step flow-SDE endpoint previews; no extra model forward.
- Reference distribution: previous five completed runner steps, pooling both actor ranks, all trajectory queries, and all future positions.
- Mapping: `z=(log(V_L3+1e-12)-mean)/max(std,1e-6)`, clipped to `[-2,2]`, then `w=1+0.1z` in `[0.8,1.2]`.
- The per-`h` straight-through hook changes backward likelihood contributions only. Chunk reward, GRPO advantage, joint chunk PPO ratio/clipping, and global `clip_grad=1` remain unchanged.
- Step 1 used `w=1` while collecting the first history step; nonuniform weighting began at Step 2.

## Realized budget and stop

- Configuration: 2 A800 GPUs, 16 train environments x 16 rollout epochs, group size 8, global/micro batch 512/32, update epoch 2, `H=C=50`, active action dimension 14.
- Completed trajectories: {train['cumulative_trajectories']:,}; successful training trajectories: {train['cumulative_successes']:,}.
- Derived completed policy queries: {train['cumulative_trajectories'] * 4:,}; derived optimizer updates: {train['latest_complete_global_step'] * 4:,}.
- Time to the last complete step: `{final_train_row['elapsed_text']}`. Full wrapper lifetime: `{iso_duration(start, finish)}` (`{start}` to `{finish}`).
- User-authorized stop requested: `{stop_request}`; termination escalation marker: `{terminate_request}`; wrapper finished: `{finish}`.
- The driver log records SIGTERM during the partially collected next step (6/16 rollout epochs). Driver exit code: `{driver_exit}`. This run was intentionally stopped, not naturally completed; partial Step 55 is excluded from all completed-step metrics.
- Observer: `{observer_exit}`.

## Training results through Step 54

- Training-rollout success: latest {train['success']['latest']:.4%}; recent-5 {train['success']['recent5_mean']:.4%}; Steps 1-54 mean {train['success']['all_steps_mean']:.4%}; best {train['success']['best']:.4%} at Step {train['success']['best_step']}.
- Historical GRPO mean over the same 54 steps: {baseline['success_once']['baseline_mean_same_steps']:.4%}; current minus baseline: {baseline['success_once']['mean_delta']:+.4%}.
- Mean KL / PPO clip fraction over Steps 1-54: {baseline['approx_kl']['current_mean_g1_to_latest']:.5f} / {baseline['clip_fraction']['current_mean_g1_to_latest']:.5f}.
- Mean pre-clip gradient norm: {baseline['grad_norm']['current_mean_g1_to_latest']:.3f} versus historical {baseline['grad_norm']['baseline_mean_same_steps']:.3f}.
- These are on-policy training rollouts. Neither run performed held-out fixed-ID evaluation during training, so this package does not establish a held-out control-performance improvement.

## Final completed method shard (Step 54)

- Valid queries / weight points: {method['valid_queries']} / {method['valid_weight_points']}.
- Weight mean and p05/median/p95: {weight['mean']:.6f}; {weight['p05_median_p95'][0]:.6f} / {weight['p05_median_p95'][1]:.6f} / {weight['p05_median_p95'][2]:.6f}.
- Weight at 0.8 / 1.2: {weight['at_0_8_pct']:.3f}% / {weight['at_1_2_pct']:.3f}%.
- Back-25 minus front-25 mean weight: {weight['back_25_minus_front_25']:+.6f}.
- Positive/negative-advantage mean weight: {split['positive_mean_weight']:.6f} / {split['negative_mean_weight']:.6f}.
- Full keys, shapes, per-`h` vectors, variance levels, and denoising-index counts are in `analysis/METHOD_STEP54_SUMMARY.json`.

## Resources and checkpoint pointer

- GPU0/GPU1 peak memory: {int(resources['gpu0_peak_mib']):,} / {int(resources['gpu1_peak_mib']):,} MiB.
- Cgroup memory peak: {cgroup_gib:.2f} GiB; memory high/max/OOM/OOM-kill events: {resources['event_high']}/{resources['event_max']}/{resources['event_oom']}/{resources['event_oom_kill']}.
- Latest complete checkpoint: `{checkpoint['checkpoint_name']}`, `{int(checkpoint['checkpoint_bytes']):,}` bytes across {checkpoint['checkpoint_files']} files, at `{checkpoint['checkpoint_path']}`.
- The checkpoint is intentionally excluded from this ZIP. DVAC recent-five statistics are not stored in DCP, so resuming that checkpoint would not exactly restore the method's rolling state.

## Package selection and boundaries

- Included NPZ pairs: Step 1 warmup (`rollout_step0000`), first apply step (`rollout_step0001`), and final complete Step 54 (`rollout_step0053`).
- Included: final metrics/logs/configs, rank manifests and compact summaries, final analysis, resource summary, and one sampled control trace.
- Excluded: checkpoint bodies, the other 51 per-step NPZ pairs, raw `resources.csv`, raw `process_rss.tsv`, Ray/session files, TensorBoard events, caches, and core dump.
- The control-trace `h` coordinate is a TOPP-progress approximation, not exact original-waypoint lineage.
- `FILE_MANIFEST.csv` lists payload files with sizes and hashes. `SHA256SUMS.txt` covers every archived file except itself.
"""
    README.write_text(text, encoding="utf-8")


def payload_files(*, include_manifest: bool) -> list[Path]:
    excluded = {SHA256SUMS}
    if not include_manifest:
        excluded.add(MANIFEST)
    result = []
    for path in STAGING.rglob("*"):
        if not path.is_file() or path in excluded:
            continue
        rel = path.relative_to(STAGING).as_posix()
        if rel.startswith("run/control_trace/.claims/") or "__pycache__" in rel:
            continue
        result.append(path)
    return sorted(result, key=lambda item: item.relative_to(STAGING).as_posix())


def make_manifest_and_sums() -> None:
    for path in (MANIFEST, SHA256SUMS):
        if path.exists():
            path.unlink()
    rows = []
    for path in payload_files(include_manifest=False):
        rows.append(
            {
                "path": path.relative_to(STAGING).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)

    sum_lines = []
    for path in payload_files(include_manifest=True):
        sum_lines.append(
            f"{sha256(path)}  {path.relative_to(STAGING).as_posix()}"
        )
    SHA256SUMS.write_text("\n".join(sum_lines) + "\n", encoding="utf-8")


def make_zip() -> dict:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    prefix = OUTPUT.stem
    files = payload_files(include_manifest=True) + [SHA256SUMS]
    files = sorted(set(files), key=lambda item: item.relative_to(STAGING).as_posix())
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in files:
            archive.write(path, f"{prefix}/{path.relative_to(STAGING).as_posix()}")
    with zipfile.ZipFile(OUTPUT, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad}")
        names = archive.namelist()
    return {
        "path": str(OUTPUT),
        "bytes": OUTPUT.stat().st_size,
        "sha256": sha256(OUTPUT),
        "entries": len(names),
        "uncompressed_bytes": sum(path.stat().st_size for path in files),
    }


def main() -> None:
    if not STAGING.is_dir():
        raise FileNotFoundError(STAGING)
    method = method_summary()
    make_readme(method)
    make_manifest_and_sums()
    result = make_zip()
    print(json.dumps({"method": method, "archive": result}, indent=2))


if __name__ == "__main__":
    main()
