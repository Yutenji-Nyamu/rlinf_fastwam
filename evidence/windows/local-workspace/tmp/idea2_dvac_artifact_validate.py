from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


root = Path(sys.argv[1])
telemetry = root / "dvac_telemetry"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


query_files = sorted(telemetry.glob("query_index_rollout_rank*.csv"))
episode_files = sorted(telemetry.glob("episode_index_env_rank*.csv"))
trace_files = sorted(telemetry.glob("trace_rollout_rank*.npz"))
assert len(query_files) == len(episode_files) == len(trace_files) == 2

queries = [row for path in query_files for row in read_csv(path)]
episodes = [row for path in episode_files for row in read_csv(path)]
assert len(queries) == 64, len(queries)
assert len(episodes) == 16, len(episodes)
assert len({row["query_uid"] for row in queries}) == 64
assert len({row["episode_uid"] for row in episodes}) == 16

episode_by_idx = {int(row["episode_idx"]): row for row in episodes}
queries_by_episode: dict[int, list[dict[str, str]]] = defaultdict(list)
for row in queries:
    queries_by_episode[int(row["episode_idx"])].append(row)
assert set(queries_by_episode) == set(episode_by_idx)
for episode_idx, rows in queries_by_episode.items():
    rows.sort(key=lambda row: int(row["query_idx"]))
    assert [int(row["query_idx"]) for row in rows] == [0, 1, 2, 3]
    assert [int(row["action_slot_start"]) for row in rows] == [0, 50, 100, 150]
    assert all(row["reset_id"] == episode_by_idx[episode_idx]["reset_id"] for row in rows)
    assert int(episode_by_idx[episode_idx]["action_slots"]) == 200
    assert int(episode_by_idx[episode_idx]["final_query_idx"]) == 3
    assert int(episode_by_idx[episode_idx]["final_action_slot"]) == 200

assert Counter(int(row["rollout_rank"]) for row in queries) == {0: 32, 1: 32}
assert Counter(int(row["source_env_rank"]) for row in queries) == {0: 32, 1: 32}
assert Counter(int(row["source_env_rank"]) for row in episodes) == {0: 8, 1: 8}
assert {int(row["video_tile_index"]) for row in queries} == set(range(8))
for video_key, rows in {
    key: [row for row in queries if (row["rollout_rank"], row["video_relpath"]) == key]
    for key in {(row["rollout_rank"], row["video_relpath"]) for row in queries}
}.items():
    assert len(rows) == 32, (video_key, len(rows))
    assert Counter(int(row["video_tile_index"]) for row in rows) == {index: 4 for index in range(8)}
assert Counter(int(row["video_pre_frame"]) for row in queries) == {0: 16, 1: 16, 2: 16, 3: 16}
assert Counter(int(row["video_post_frame"]) for row in queries) == {1: 16, 2: 16, 3: 16, 4: 16}

image_paths: list[Path] = []
for row in queries:
    for column in ("head_image_relpath", "left_wrist_image_relpath", "right_wrist_image_relpath"):
        path = telemetry / row[column]
        assert path.is_file(), path
        image_paths.append(path)
assert len(image_paths) == 192
image_specs = Counter()
for path in image_paths:
    with Image.open(path) as image:
        image_specs[(image.mode, image.size)] += 1

trace_summaries = []
variance_values: dict[int, list[np.ndarray]] = {2: [], 3: [], 4: []}
for rank, path in enumerate(trace_files):
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        shapes = {key: list(data[key].shape) for key in data.files}
        dtypes = {key: str(data[key].dtype) for key in data.files}
        expected_shapes = {
            "x_chain": (32, 5, 50, 14),
            "z_endpoint": (32, 4, 50, 14),
            "timesteps": (4,),
            "final_model_action": (32, 50, 14),
            "env_action": (32, 50, 14),
            "robot_state": (32, 14),
        }
        assert set(keys) == set(expected_shapes), keys
        for key, shape in expected_shapes.items():
            assert data[key].shape == shape, (rank, key, data[key].shape)
            assert np.isfinite(data[key]).all(), (rank, key)
        assert np.allclose(data["x_chain"][:, -1], data["final_model_action"], atol=0, rtol=0)
        assert np.all(np.diff(data["timesteps"]) < 0)
        assert np.all(data["timesteps"] > 0)
        for length in variance_values:
            per_h = np.var(data["z_endpoint"][:, -length:], axis=1, ddof=0).sum(axis=-1)
            assert per_h.shape == (32, 50)
            assert np.isfinite(per_h).all()
            assert np.all(per_h >= 0)
            variance_values[length].append(per_h.reshape(-1))
        trace_summaries.append(
            {
                "rank": rank,
                "keys": keys,
                "shapes": shapes,
                "dtypes": dtypes,
                "timesteps": data["timesteps"].tolist(),
                "z_min": float(data["z_endpoint"].min()),
                "z_max": float(data["z_endpoint"].max()),
                "x_min": float(data["x_chain"].min()),
                "x_max": float(data["x_chain"].max()),
            }
        )

manifest = json.loads((telemetry / "run_manifest.json").read_text(encoding="utf-8"))
expected_manifest = {
    "run_id": "idea2_dvac_sft_smoke_2gpu_16env_v1",
    "source_commit": "61996e15cc7f5a32bd6012b61b20893d94636c82",
    "robotwin_commit": "481380fbd97cbf9ff830aedfb2279851e1e58969",
    "checkpoint_revision": "92684e50c8a3dcf13b76a06713e3152625967be1",
    "norm_stats_sha256": "649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a",
    "seed_file_sha256": "194164f7380fd7cad2a8940ca93def01c2be865da265e4af1c463d73b2aa482f",
    "model_action_horizon": 50,
    "model_action_dim": 32,
    "active_action_dim": 14,
    "execution_chunk_length": 50,
    "denoising_steps": 4,
    "rollout_world_size": 2,
    "env_world_size": 2,
    "planner_backend": "mplib",
    "eval_sampling_method": "flow_ode",
}
for key, value in expected_manifest.items():
    assert manifest[key] == value, (key, manifest[key], value)
assert "timeout" not in manifest["launch_command"]
assert "kill" not in manifest["launch_command"]
assert len(manifest["expected_rollout_shards"]) == 2
assert len(manifest["expected_episode_shards"]) == 2

success_count = sum(row["success_at_end"].lower() == "true" for row in episodes)
termination_counts = Counter(row["termination_reason"] for row in episodes)
variance_sanity = {}
for length, shards in variance_values.items():
    values = np.concatenate(shards)
    assert values.shape == (3200,)
    variance_sanity[f"L{length}"] = {
        "count_query_h": int(values.size),
        "positive_count": int(np.count_nonzero(values > 0)),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }
summary = {
    "query_rows": len(queries),
    "episode_rows": len(episodes),
    "success_count": success_count,
    "success_rate_smoke_only": success_count / len(episodes),
    "termination_counts": dict(termination_counts),
    "unique_reset_ids": len({row["reset_id"] for row in episodes}),
    "image_count": len(image_paths),
    "image_specs": {f"{mode}_{size[0]}x{size[1]}": count for (mode, size), count in image_specs.items()},
    "trace_summaries": trace_summaries,
    "variance_sanity_population_ddof0": variance_sanity,
    "manifest_runtime": {
        "model_parameter_dtype": manifest["model_parameter_dtype"],
        "run_started_at_local": manifest["run_started_at_local"],
        "hostname": manifest["hostname"],
    },
}
print(json.dumps(summary, indent=2, sort_keys=True))
print("DVAC_ARTIFACT_CONTRACT_PASS=1")
