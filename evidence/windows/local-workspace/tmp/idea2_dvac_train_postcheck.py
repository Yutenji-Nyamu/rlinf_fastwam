from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


run_dir = Path(sys.argv[1])
assert run_dir.is_dir(), run_dir

dvac_dir = run_dir / "dvac_train"
rank_dirs = sorted(dvac_dir.glob("actor_rank*"))
assert [path.name for path in rank_dirs] == ["actor_rank00", "actor_rank01"]

step_query_counts: dict[int, int] = {0: 0, 1: 0}
rank_summaries: dict[str, list[dict[str, str]]] = {}
npz_report: list[dict[str, object]] = []
control_query_matches: list[dict[str, object]] = []

for rank_dir in rank_dirs:
    manifest = json.loads((rank_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["actor_world_size"] == 2
    assert manifest["config"]["mode"] == "apply"
    assert manifest["config"]["selected_l"] == 3

    with (rank_dir / "runner_step_metrics.csv").open(newline="", encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    assert len(summaries) == 2, (rank_dir, len(summaries))
    rank_summaries[rank_dir.name] = summaries

    for step, row in enumerate(summaries):
        assert int(row["runner_step"]) == step
        assert int(row["warmup"]) == (1 if step == 0 else 0)
        if step == 0:
            assert int(row["history_count"]) == 0
        else:
            assert int(row["history_count"]) > 0

    for step in (0, 1):
        path = rank_dir / f"rollout_step{step:04d}.npz"
        assert path.is_file(), path
        with np.load(path, allow_pickle=False) as data:
            required = {
                "v_l2",
                "v_l3",
                "v_l4",
                "weights",
                "clipped_z",
                "advantages",
                "rewards",
                "done_before",
                "done_after",
                "loss_mask",
                "old_logprob_per_h",
                "denoise_inds",
            }
            assert required.issubset(data.files), (path, required - set(data.files))
            weights = data["weights"]
            clipped_z = data["clipped_z"]
            assert weights.shape == clipped_z.shape
            assert weights.shape[-1] == 50, weights.shape
            query_count = int(np.prod(weights.shape[:-1]))
            step_query_counts[step] += query_count
            assert np.isfinite(weights).all()
            assert np.isfinite(clipped_z).all()
            assert float(weights.min()) >= 0.8 - 1e-6
            assert float(weights.max()) <= 1.2 + 1e-6
            if step == 0:
                assert np.array_equal(weights, np.ones_like(weights))
                assert np.array_equal(clipped_z, np.zeros_like(clipped_z))
            else:
                assert float(weights.min()) < 1.0 < float(weights.max())

            for name in ("v_l2", "v_l3", "v_l4"):
                value = data[name]
                assert value.shape == weights.shape, (name, value.shape, weights.shape)
                assert np.isfinite(value).all()
                assert (value >= 0).all()

            # RLinf broadcasts the query boundary mask across H before the
            # actor writer; it remains query aligned even though the shape is
            # [T,B,H] rather than [T,B].
            assert data["done_before"].shape == weights.shape
            assert data["done_after"].shape == weights.shape
            metadata_keys = [name for name in data.files if name.startswith("dvac_meta_")]
            assert metadata_keys
            for name in metadata_keys:
                assert data[name].shape == weights.shape[:-1], (name, data[name].shape)

            npz_report.append(
                {
                    "path": str(path),
                    "query_count": query_count,
                    "weights_shape": list(weights.shape),
                    "weight_min": float(weights.min()),
                    "weight_mean": float(weights.mean()),
                    "weight_max": float(weights.max()),
                    "v_l3_min": float(data["v_l3"].min()),
                    "v_l3_median": float(np.median(data["v_l3"])),
                    "v_l3_max": float(data["v_l3"].max()),
                    "metadata_keys": metadata_keys,
                    "file_bytes": path.stat().st_size,
                }
            )

            if step == 0:
                match = (
                    (data["dvac_meta_source_env_rank"] == 0)
                    & (data["dvac_meta_local_env_slot"] == 0)
                    & (data["dvac_meta_rollout_epoch"] == 0)
                    & (data["dvac_meta_reset_id"] == 57)
                    & (data["dvac_meta_query_idx"] == 2)
                )
                for time_index, batch_index in np.argwhere(match):
                    curve = data["v_l3"][time_index, batch_index].astype(np.float64)
                    control_query_matches.append(
                        {
                            "actor_rank": rank_dir.name,
                            "time_index": int(time_index),
                            "batch_index": int(batch_index),
                            "action_slot_start": int(
                                data["dvac_meta_action_slot_start"][time_index, batch_index]
                            ),
                            "success_before": int(
                                data["dvac_meta_success_before"][time_index, batch_index]
                            ),
                            "v_l3": curve.tolist(),
                            "v_l3_min_h": int(np.argmin(curve)),
                            "v_l3_max_h": int(np.argmax(curve)),
                            "v_l3_at_approx_success_h10": float(curve[10]),
                        }
                    )

assert step_query_counts == {0: 512, 1: 512}, step_query_counts
assert len(control_query_matches) == 1, control_query_matches

video_paths = sorted((run_dir / "control_trace").rglob("*.mp4"))
csv_paths = sorted((run_dir / "control_trace").rglob("frames.csv"))
metadata_paths = sorted((run_dir / "control_trace").rglob("metadata.json"))
assert len(video_paths) == len(csv_paths) == len(metadata_paths) == 1

control_metadata = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
assert control_metadata["encoder_error"] is None, control_metadata
assert 0 < int(control_metadata["frame_count"]) <= 200
assert control_metadata["output_width"] == 160
assert control_metadata["output_height"] == 120
assert control_metadata["fps"] == 10

with csv_paths[0].open(newline="", encoding="utf-8") as handle:
    frame_rows = list(csv.DictReader(handle))
assert len(frame_rows) == int(control_metadata["frame_count"])
assert all(row["query_idx"] != "" and row["h_lo"] != "" for row in frame_rows)
first_success_rows = [row for row in frame_rows if row["first_success"] == "1"]
assert len(first_success_rows) <= 1

probe = subprocess.run(
    [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames,duration",
        "-of",
        "json",
        str(video_paths[0]),
    ],
    check=True,
    capture_output=True,
    text=True,
)
video_probe = json.loads(probe.stdout)
stream = video_probe["streams"][0]
assert int(stream["width"]) == 160
assert int(stream["height"]) == 120
assert stream["r_frame_rate"] == "10/1"
assert int(stream["nb_read_frames"]) == len(frame_rows)

checkpoint_dirs = sorted(run_dir.rglob("checkpoints/global_step_2"))
assert len(checkpoint_dirs) == 1, checkpoint_dirs

report = {
    "run_dir": str(run_dir),
    "rank_summaries": rank_summaries,
    "step_query_counts": step_query_counts,
    "npz": npz_report,
    "control_query_step0_q2": control_query_matches[0],
    "control_trace": {
        "video_path": str(video_paths[0]),
        "video_bytes": video_paths[0].stat().st_size,
        "frames_csv_path": str(csv_paths[0]),
        "frames_csv_bytes": csv_paths[0].stat().st_size,
        "metadata": control_metadata,
        "first_success_rows": first_success_rows,
        "ffprobe": video_probe,
    },
    "checkpoint_dir": str(checkpoint_dirs[0]),
}

output = run_dir / "idea2_train_smoke_validation.json"
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2, sort_keys=True))
print("IDEA2_TRAIN_SMOKE_VALIDATION_PASS=1")
