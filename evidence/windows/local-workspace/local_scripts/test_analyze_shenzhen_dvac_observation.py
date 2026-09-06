"""Synthetic contract tests for analyze_shenzhen_dvac_observation.py."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).with_name("analyze_shenzhen_dvac_observation.py")
SPEC = importlib.util.spec_from_file_location("sz_dvac_analysis", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_pi0(
    root: Path,
    *,
    run_id: str = "pi0-synthetic",
    scale_offset: float = 0.0,
    include_success_before: bool = True,
) -> Path:
    root.mkdir()
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "policy": "pi0",
                "task": "adjust_bottle",
                "checkpoint_revision": "pi0-checkpoint",
            }
        ),
        encoding="utf-8",
    )
    query_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    n, m, h, d = 8, 4, 50, 2
    endpoint_axis = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
    z = np.empty((n, m, h, d), dtype=np.float32)
    for row in range(n):
        for future in range(h):
            for dim in range(d):
                scale = (
                    0.2
                    + scale_offset
                    + 0.03 * row
                    + 0.005 * future
                    + 0.01 * dim
                )
                z[row, :, future, dim] = endpoint_axis * scale
    x_chain = np.zeros((n, m + 1, h, d), dtype=np.float32)
    final = np.zeros((n, h, d), dtype=np.float32)
    for episode in range(2):
        reset_id = 1000 + episode
        success = episode == 0
        episode_rows.append(
            {
                "episode_uid": f"ep{episode:06d}_reset{reset_id}",
                "episode_idx": episode,
                "eval_epoch": 0,
                "source_env_rank": 0,
                "stage_id": 0,
                "local_env_slot": episode,
                "reset_id": reset_id,
                "success": success,
                "success_at_end": success,
                "return": float(success),
                "reward": float(success),
                "action_slots": 4 * h,
                "final_query_idx": 3,
                "final_action_slot": 4 * h,
                "first_success_action_slot": 3 * h - 1 if success else "",
                "first_success_query": 2 if success else "",
                "termination_reason": "truncation",
            }
        )
        for query_idx in range(4):
            trace_row = episode * 4 + query_idx
            query_row = {
                    "trace_row": trace_row,
                    "query_uid": f"ep{episode:06d}_q{query_idx:03d}_reset{reset_id}",
                    "episode_idx": episode,
                    "eval_epoch": 0,
                    "query_idx": query_idx,
                    "action_slot_start": query_idx * h,
                    "source_env_rank": 0,
                    "stage_id": 0,
                    "local_env_slot": episode,
                    "reset_id": reset_id,
                    "rollout_rank": 0,
                    "video_worker_seed": 0,
                    "video_index": 0,
                    "video_tile_index": episode,
                    "video_pre_frame": query_idx,
                    "video_post_frame": query_idx + 1,
                    "video_relpath": "combined.mp4",
                    "success_after": bool(success and query_idx == 2),
                    "head_image_relpath": "",
                    "left_wrist_image_relpath": "",
                    "right_wrist_image_relpath": "",
                }
            if include_success_before:
                query_row["success_before"] = bool(success and query_idx == 3)
            query_rows.append(query_row)
    np.savez_compressed(
        root / "trace_rollout_rank00.npz",
        x_chain=x_chain,
        z_endpoint=z,
        final_model_action=final,
        env_action=np.zeros((n, h, d), dtype=np.float32),
        robot_state=np.zeros((n, 14), dtype=np.float32),
        timesteps=np.array([1.0, 0.75, 0.5, 0.25], dtype=np.float32),
    )
    _write_csv(root / "query_index_rollout_rank00.csv", query_rows)
    _write_csv(root / "episode_index_env_rank00.csv", episode_rows)
    return root


def _make_fastwam(
    root: Path,
    *,
    run_id: str = "fastwam-synthetic",
    manifest_key: str = "preferred",
    include_source_seed: bool = True,
) -> Path:
    root.mkdir()
    manifest = {
                "run_id": run_id,
                "policy": "fastwam",
                "task": "move_stapler_pad",
                "checkpoint": "fastwam-checkpoint",
    }
    if manifest_key == "preferred":
        manifest["action_num_train_timesteps"] = 1000
    elif manifest_key == "legacy":
        manifest["num_train_timesteps"] = 1000
    elif manifest_key == "conflict":
        manifest["action_num_train_timesteps"] = 1000
        manifest["num_train_timesteps"] = 500
    elif manifest_key != "missing":
        raise ValueError(manifest_key)
    (root / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (root / "traces").mkdir()
    query_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    m, h, c, d = 10, 32, 24, 2
    timesteps = np.linspace(900, 100, m, dtype=np.float32)
    for episode in range(4):
        success = episode < 2
        reset_id = 2000 + episode
        episode_rows.append(
            {
                "run_id": run_id,
                "policy": "fastwam",
                "task": "move_stapler_pad",
                "episode_id": episode,
                "reset_id": reset_id,
                "success": success,
                "total_action_slots": 3 * c,
                "queries": 3,
                "first_success_action_slot": 2 * c - 1 if success else "",
                "first_success_query": 1 if success else "",
                "video_path": f"episode{episode}.mp4",
            }
        )
        if include_source_seed:
            episode_rows[-1]["source_seed"] = 4300000 + episode
        for query_idx in range(3):
            trace_name = f"traces/episode{episode:04d}_query{query_idx:04d}.npz"
            x = np.empty((m + 1, h, d), dtype=np.float32)
            for endpoint in range(m + 1):
                for future in range(h):
                    for dim in range(d):
                        x[endpoint, future, dim] = (
                            0.1 * endpoint
                            + 0.02 * future * endpoint
                            + 0.01 * episode * endpoint
                            + 0.005 * query_idx * endpoint
                            + 0.001 * dim
                        )
            v = np.zeros((m, h, d), dtype=np.float32)
            np.savez_compressed(
                root / trace_name,
                x_chain=x,
                v_chain=v,
                timesteps=timesteps,
                deltas=np.ones(m, dtype=np.float32),
                x_next=x[1:],
                final_model_action=x[-1],
                env_action=x[-1],
                robot_state=np.zeros(14, dtype=np.float32),
            )
            start = query_idx * c
            query_rows.append(
                {
                    "run_id": run_id,
                    "policy": "fastwam",
                    "task": "move_stapler_pad",
                    "instruction": "move the stapler",
                    "episode_id": episode,
                    "reset_id": reset_id,
                    "query_idx": query_idx,
                    "query_start_action_slot": start,
                    "query_end_action_slot_exclusive": start + c,
                    "planned_exec_length": c,
                    "executed_length": c,
                    "success_before": bool(success and query_idx == 2),
                    "success_after": bool(success and query_idx == 1),
                    "trace_path": trace_name,
                    "head_image_path": "",
                    "left_image_path": "",
                    "right_image_path": "",
                    "video_query_index": query_idx,
                    "video_frame_start": start,
                    "video_frame_end_exclusive": start + c,
                    "terminal_success_video_frame": start + c if success and query_idx == 1 else "",
                }
            )
    _write_csv(root / "queries.csv", query_rows)
    _write_csv(root / "episodes.csv", episode_rows)
    return root


class ShenzhenDvacAnalysisTest(unittest.TestCase):
    def test_readers_decomposition_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            pi0 = _make_pi0(base / "pi0")
            fastwam = _make_fastwam(base / "fastwam")
            output = base / "derived"
            phase_path = base / "phase.csv"
            pd.DataFrame(
                [
                    {
                        "query_key": "fastwam-synthetic::episode0000_query0000",
                        "h": h,
                        "query_state_phase": "approach-state",
                        "phase_coarse": "MOVING",
                        "phase_task": "approach",
                        "phase_source": "synthetic-independent-label",
                        "phase_confidence": "high",
                    }
                    for h in range(32)
                ]
            ).to_csv(phase_path, index=False)
            summary = analysis.analyze(
                pi0_sources=[pi0],
                fastwam_sources=[fastwam],
                output=output,
                phase_annotations=phase_path,
                skip_plots=True,
                skip_storyboards=True,
            )

            self.assertEqual(summary["counts"]["sources"], 2)
            self.assertEqual(summary["counts"]["queries"], 20)
            self.assertEqual(summary["counts"]["pre_success_queries"], 17)
            self.assertEqual(summary["counts"]["episodes"], 6)
            self.assertEqual(summary["counts"]["fastwam_executed_action_frame_rows"], 288)

            query = pd.read_csv(output / "query_metrics.csv")
            horizon = pd.read_csv(output / "query_horizon.csv")
            action = pd.read_csv(output / "fastwam_action_frame_metrics.csv")
            episode = pd.read_csv(output / "episode_metrics.csv")
            episode_all = pd.read_csv(output / "episode_metrics_all_queries.csv")
            phase_episode = pd.read_csv(output / "phase_episode_metrics.csv")

            self.assertEqual(set(horizon["L"]), {2, 3, 4, 5})
            self.assertEqual(
                set(query.groupby("source_kind")["horizon"].first().to_dict().items()),
                {("pi0", 50), ("fastwam", 32)},
            )
            self.assertTrue((action["h"] < 24).all())
            self.assertTrue((action["video_frame_index"] == action["action_slot"]).all())
            labeled = action[
                action["query_key"] == "fastwam-synthetic::episode0000_query0000"
            ]
            self.assertEqual(set(labeled["phase_coarse"]), {"MOVING"})
            self.assertEqual(set(labeled["phase_task"]), {"approach"})
            self.assertEqual(set(labeled["query_state_phase"]), {"approach-state"})
            tail = horizon[
                (horizon["query_key"] == "fastwam-synthetic::episode0000_query0000")
                & (horizon["h"] >= 24)
            ]
            self.assertEqual(set(tail["phase_coarse"]), {"UNLABELED"})
            self.assertEqual(set(tail["phase_task"]), {"UNLABELED"})
            self.assertEqual(set(tail["query_state_phase"]), {"approach-state"})
            self.assertFalse((query["source_kind"] == "pi0").empty)
            self.assertEqual(int(episode["representative"].sum()), 4)
            self.assertEqual(
                int(
                    episode.loc[
                        (episode["source_kind"] == "pi0") & episode["success"].astype(bool),
                        "queries",
                    ].iloc[0]
                ),
                3,
            )
            self.assertEqual(
                int(
                    episode_all.loc[
                        (episode_all["source_kind"] == "pi0")
                        & episode_all["success"].astype(bool),
                        "queries",
                    ].iloc[0]
                ),
                4,
            )
            terminal = action[action["terminal_success_action"].astype(bool)]
            self.assertEqual(len(terminal), 2)
            self.assertTrue((terminal["action_slot"] == 47).all())
            self.assertTrue((terminal["terminal_success_video_frame"] == 48).all())
            self.assertEqual(set(phase_episode["phase_axis"]), {"action_phase", "query_state_phase"})

            np.testing.assert_allclose(
                horizon["y_ln"], horizon["b_position"] + horizon["r_raw"], atol=1e-12
            )
            np.testing.assert_allclose(
                horizon["y_ln"],
                horizon["mu"] + horizon["P_h"] + horizon["S_raw"] + horizon["I_raw"],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                horizon["R_std"], horizon["S_std"] + horizon["I_std"], atol=1e-12
            )

            pi0_l3 = horizon[
                (horizon["source_kind"] == "pi0") & (horizon["L"] == 3)
            ]
            self.assertEqual(int(pi0_l3["baseline_eligible"].sum()), 7 * 50)
            self.assertTrue((pi0_l3.loc[~pi0_l3["baseline_eligible"], "query_idx"] == 3).all())

    def test_run_is_part_of_baseline_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            p1 = _make_pi0(base / "p1", run_id="pi0-p1-16", scale_offset=0.0)
            fixed64 = _make_pi0(
                base / "fixed64", run_id="pi0-fixed64", scale_offset=1.0
            )
            traces1, _, _ = analysis.load_pi0_source(p1)
            traces2, _, _ = analysis.load_pi0_source(fixed64)
            _, horizon = analysis.derive_metrics([*traces1, *traces2])
            self.assertEqual(horizon["group_id"].nunique(), 2)
            centers = (
                horizon[horizon["L"] == 3]
                .groupby("run_id")["b_position"]
                .mean()
            )
            self.assertNotEqual(float(centers["pi0-p1-16"]), float(centers["pi0-fixed64"]))

    def test_required_success_before_and_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            missing_success = _make_pi0(
                base / "pi0-missing-success", include_success_before=False
            )
            with self.assertRaisesRegex(ValueError, "required success_before"):
                analysis.load_pi0_source(missing_success)

            preferred = _make_fastwam(base / "preferred")
            _, _, inventory = analysis.load_fastwam_source(preferred)
            self.assertEqual(inventory["action_num_train_timesteps"], 1000)
            self.assertEqual(
                inventory["action_num_train_timesteps_source"],
                "run_manifest.action_num_train_timesteps",
            )

            legacy = _make_fastwam(base / "legacy", run_id="legacy", manifest_key="legacy")
            _, _, inventory = analysis.load_fastwam_source(legacy)
            self.assertIn("legacy compatibility", inventory["action_num_train_timesteps_source"])

            conflict = _make_fastwam(
                base / "conflict", run_id="conflict", manifest_key="conflict"
            )
            with self.assertRaisesRegex(ValueError, "conflicts"):
                analysis.load_fastwam_source(conflict)

            missing = _make_fastwam(
                base / "missing", run_id="missing", manifest_key="missing"
            )
            with self.assertRaisesRegex(ValueError, "must record action_num_train_timesteps"):
                analysis.load_fastwam_source(missing)

    def test_seed_sidecar_is_read_only_and_empty_outputs_have_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fastwam = _make_fastwam(
                base / "fastwam", include_source_seed=False
            )
            payload_before = (fastwam / "episodes.csv").read_bytes()
            seed_map = base / "official_seed_map.csv"
            pd.DataFrame(
                [
                    {
                        "run_id": "fastwam-synthetic",
                        "episode_id": episode,
                        "source_seed": 9900000 + episode,
                    }
                    for episode in range(4)
                ]
            ).to_csv(seed_map, index=False)
            output = base / "seed-derived"
            summary = analysis.analyze(
                pi0_sources=[],
                fastwam_sources=[fastwam],
                output=output,
                official_seed_maps=[seed_map],
                skip_plots=True,
                skip_storyboards=True,
            )
            query = pd.read_csv(output / "query_metrics.csv")
            self.assertEqual(set(query["source_seed"]), set(range(9900000, 9900004)))
            self.assertTrue(summary["source_inventory"][0]["official_seed_map_used"])
            self.assertEqual((fastwam / "episodes.csv").read_bytes(), payload_before)

            pi0 = _make_pi0(base / "pi0")
            empty_output = base / "pi0-only"
            analysis.analyze(
                pi0_sources=[pi0],
                fastwam_sources=[],
                output=empty_output,
                skip_plots=True,
                skip_storyboards=True,
            )
            for filename, expected in (
                ("fastwam_action_frame_metrics.csv", analysis.ACTION_COLUMNS),
                ("phase_episode_metrics.csv", analysis.PHASE_EPISODE_COLUMNS),
                ("phase_summary.csv", analysis.PHASE_SUMMARY_COLUMNS),
                ("storyboard_index.csv", analysis.STORYBOARD_INDEX_COLUMNS),
            ):
                frame = pd.read_csv(empty_output / filename)
                self.assertEqual(list(frame.columns), list(expected))
                self.assertTrue(frame.empty)

    def test_optional_dependency_preflight_precedes_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with mock.patch.object(
                analysis.importlib,
                "import_module",
                side_effect=ImportError("synthetic missing dependency"),
            ):
                with self.assertRaisesRegex(RuntimeError, "matplotlib"):
                    analysis.analyze(
                        pi0_sources=[Path(temporary) / "not-read"],
                        fastwam_sources=[],
                        output=output,
                        skip_plots=False,
                        skip_storyboards=True,
                    )
            self.assertFalse(output.exists())

    def test_fastwam_endpoint_reconstruction(self) -> None:
        x = np.arange(4 * 2 * 1, dtype=np.float32).reshape(4, 2, 1)
        v = np.ones((3, 2, 1), dtype=np.float32)
        t = np.array([1000, 500, 250], dtype=np.float32)
        actual = analysis.reconstruct_fastwam_endpoints(x, v, t)
        expected = x[:-1] - np.array([1.0, 0.5, 0.25], dtype=np.float32).reshape(3, 1, 1)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
