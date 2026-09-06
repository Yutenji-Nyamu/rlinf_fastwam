from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN = Path(
    "/root/autodl-tmp/experiments/"
    "rlt_single_gpu_success_episode_bc_dvac_matched_width_formal480_20260826_v3"
)
OUT = Path(
    "/root/autodl-tmp/experiment_exports/"
    "rlt_step475_c10_episode_probe_light_20260828_v1"
)
SCAN_ROWS = 4000
EPS = 1e-12


def load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tb(value: torch.Tensor) -> np.ndarray:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise TypeError(f"expected [1,1,...] tensor, got {type(value)}")
    if tuple(value.shape[:2]) != (1, 1):
        raise ValueError(f"expected first dimensions [1,1], got {tuple(value.shape)}")
    return value.detach().cpu()[0, 0].numpy()


def flag(value: torch.Tensor) -> bool:
    return bool(np.asarray(tb(value)).astype(bool).any())


def select_three(rows: list[dict], prefix: str) -> list[tuple[str, dict]]:
    ordered = sorted(rows, key=lambda item: item["mean_log_v10"])
    if not ordered:
        return []
    indices = np.linspace(0, len(ordered) - 1, min(3, len(ordered))).round().astype(int)
    names = ["low", "middle", "high"] if len(indices) == 3 else [f"sample{i}" for i in range(len(indices))]
    return [(f"{prefix}_{name}", ordered[int(index)]) for name, index in zip(names, indices)]


def main() -> None:
    checkpoints = list(RUN.glob("**/checkpoints/global_step_475"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one Step475 checkpoint, got {checkpoints}")
    checkpoint = checkpoints[0]
    replay = checkpoint / "actor/sac_components/replay_buffer/rank_0"
    index_data = json.loads((replay / "trajectory_index.json").read_text())
    ids = [int(item) for item in index_data["trajectory_id_list"]]
    info = {int(key): value for key, value in index_data["trajectory_index"].items()}
    if ids != sorted(ids):
        raise RuntimeError("trajectory_id_list is not ordered")
    if not all(int(info[tid]["num_samples"]) == 1 for tid in ids[-SCAN_ROWS:]):
        raise RuntimeError("expected one query row per replay trajectory")

    start_pos = max(0, len(ids) - SCAN_ROWS)
    episodes: list[dict] = []
    current: list[dict] = []
    first_boundary_seen = start_pos == 0

    for absolute_pos in range(start_pos, len(ids)):
        tid = ids[absolute_pos]
        model_id = info[tid]["model_weights_id"]
        row_path = replay / f"trajectory_{tid}_{model_id}.pt"
        data = load_pt(row_path)
        obs = data["curr_obs"]
        teacher_v = np.asarray(tb(obs["teacher_dvac_v"]), dtype=np.float32).reshape(3, 50)
        rewards = np.asarray(tb(data["rewards"]), dtype=np.float32).reshape(-1)
        actions = np.asarray(tb(data["actions"]), dtype=np.float32).reshape(10, 14)
        reference = np.asarray(tb(obs["ref_chunk"]), dtype=np.float32).reshape(10, 14)
        row = {
            "trajectory_id": tid,
            "teacher_v": teacher_v,
            "rewards": rewards,
            "actions": actions,
            "reference": reference,
            "done": flag(data["dones"]),
            "termination": flag(data["terminations"]),
            "truncation": flag(data["truncations"]),
            "success": flag(obs["episode_success"]),
            "actor_switch": flag(obs["actor_switch"]),
        }
        current.append(row)
        if row["done"]:
            if first_boundary_seen:
                success_values = {item["success"] for item in current}
                if len(success_values) != 1:
                    raise RuntimeError("episode_success changed within a reconstructed episode")
                reward_success = any((item["rewards"] > 0).any() for item in current)
                if reward_success != current[0]["success"]:
                    raise RuntimeError("episode_success disagrees with positive reward")
                log_v10 = np.concatenate(
                    [np.log10(np.maximum(item["teacher_v"][1, :10], 0) + EPS) for item in current]
                )
                episodes.append(
                    {
                        "rows": current,
                        "success": current[0]["success"],
                        "student_fraction": float(np.mean([item["actor_switch"] for item in current])),
                        "mean_log_v10": float(log_v10.mean()),
                        "max_log_v10": float(log_v10.max()),
                    }
                )
            first_boundary_seen = True
            current = []

    complete_episodes = [episode for episode in episodes if episode["student_fraction"] >= 0.999]
    successes = [episode for episode in complete_episodes if episode["success"]]
    failures = [episode for episode in complete_episodes if not episode["success"]]
    chosen = select_three(successes, "success") + select_three(failures, "failure")
    if len(chosen) < 4:
        raise RuntimeError(
            f"insufficient selected episodes: successes={len(successes)}, failures={len(failures)}"
        )

    trainer_state = load_pt(
        checkpoint / "actor/sac_components/rlt_trainer_state/checkpoint_rank_0.pt"
    )
    baseline = trainer_state["rlt_dvac_baseline"]
    baseline_mean = float(baseline["mean"])
    baseline_std = max(float(baseline["std"]), float(baseline["std_floor"]))
    log_eps = float(baseline["log_eps"])

    def weights(v10: np.ndarray, strength: float) -> np.ndarray:
        z = np.clip((np.log(np.maximum(v10, 0) + log_eps) - baseline_mean) / baseline_std, -2, 2)
        raw = np.maximum(1 + strength * (z - z.mean(axis=-1, keepdims=True)), 0)
        return raw / np.maximum(raw.mean(axis=-1, keepdims=True), 1e-12)

    counterfactual_weight_stats: dict[str, dict] = {}
    all_v10 = np.stack(
        [row["teacher_v"][1, :10] for episode in complete_episodes for row in episode["rows"]]
    )
    for strength, name in ((0.5, "s0p5"), (2.0, "s2p0")):
        mapped = weights(all_v10, strength)
        ess_per_query = mapped.sum(axis=1) ** 2 / (mapped.shape[1] * (mapped**2).sum(axis=1))
        sorted_weights = np.sort(mapped, axis=1)
        top2_mass = sorted_weights[:, -2:].sum(axis=1) / sorted_weights.sum(axis=1)
        counterfactual_weight_stats[name] = {
            "strength": strength,
            "min": float(mapped.min()),
            "p05": float(np.quantile(mapped, 0.05)),
            "median": float(np.median(mapped)),
            "mean": float(mapped.mean()),
            "p95": float(np.quantile(mapped, 0.95)),
            "max": float(mapped.max()),
            "zero_fraction": float(np.mean(mapped == 0)),
            "mean_query_ess": float(ess_per_query.mean()),
            "mean_top20_mass": float(top2_mass.mean()),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    query_rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}

    progress_rows: list[dict] = []
    all_tail_minus_c10: list[float] = []
    for episode in complete_episodes:
        denominator = max(len(episode["rows"]) - 1, 1)
        for q_idx, row in enumerate(episode["rows"]):
            v_l3 = row["teacher_v"][1]
            c10_log = float(np.log10(np.maximum(v_l3[:10].mean(), 0) + EPS))
            tail_log = float(np.log10(np.maximum(v_l3[10:].mean(), 0) + EPS))
            progress_rows.append(
                {
                    "success": int(episode["success"]),
                    "progress_bin": min(9, int(10 * q_idx / denominator)),
                    "c10_log": c10_log,
                    "tail_log": tail_log,
                }
            )
            all_tail_minus_c10.append(tail_log - c10_log)

    fig_time, axes_time = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig_c10, axes_c10 = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    fig_h50, axes_h50 = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    axes_time = axes_time.ravel()
    axes_c10 = axes_c10.ravel()
    axes_h50 = axes_h50.ravel()

    for export_idx, (label, episode) in enumerate(chosen[:6]):
        rows = episode["rows"]
        v = np.stack([row["teacher_v"] for row in rows])
        v_l3 = v[:, 1]
        w05 = weights(v_l3[:, :10], 0.5)
        w20 = weights(v_l3[:, :10], 2.0)
        action_mse = np.asarray(
            [float(np.mean((row["actions"] - row["reference"]) ** 2)) for row in rows]
        )
        reward_query = np.asarray([float(row["rewards"].sum()) for row in rows])
        positive_queries = np.flatnonzero(reward_query > 0)
        first_positive_query = int(positive_queries[0]) if positive_queries.size else -1
        x = np.arange(len(rows)) * 10
        relevant = np.log10(np.maximum(v_l3[:, :10].mean(axis=1), 0) + EPS)
        future = np.log10(np.maximum(v_l3[:, 10:].mean(axis=1), 0) + EPS)

        summaries.append(
            {
                "selection": label,
                "success": int(episode["success"]),
                "query_count": len(rows),
                "first_trajectory_id": rows[0]["trajectory_id"],
                "last_trajectory_id": rows[-1]["trajectory_id"],
                "first_positive_query": first_positive_query,
                "mean_log10_v_l3_c10": float(relevant.mean()),
                "peak_log10_v_l3_c10": float(relevant.max()),
                "peak_query": int(relevant.argmax()),
                "mean_log10_v_l3_h10_49": float(future.mean()),
                "student_fraction": episode["student_fraction"],
            }
        )
        for q_idx, row in enumerate(rows):
            query_rows.append(
                {
                    "selection": label,
                    "success": int(episode["success"]),
                    "query_idx": q_idx,
                    "action_slot_start": q_idx * 10,
                    "trajectory_id": row["trajectory_id"],
                    "reward_sum": float(row["rewards"].sum()),
                    "done": int(row["done"]),
                    "log10_v_l3_c10_mean": float(relevant[q_idx]),
                    "log10_v_l3_h10_49_mean": float(future[q_idx]),
                    "v_l3_c10_peak_h": int(v_l3[q_idx, :10].argmax()),
                    "w_s0p5_min": float(w05[q_idx].min()),
                    "w_s0p5_max": float(w05[q_idx].max()),
                    "w_s2p0_min": float(w20[q_idx].min()),
                    "w_s2p0_max": float(w20[q_idx].max()),
                    "executed_reference_mse": float(action_mse[q_idx]),
                }
            )

        arrays[f"{label}_teacher_v"] = v
        arrays[f"{label}_weights_s0p5"] = w05
        arrays[f"{label}_weights_s2p0"] = w20
        arrays[f"{label}_executed_reference_mse"] = action_mse
        arrays[f"{label}_reward_query"] = reward_query

        ax = axes_time[export_idx]
        ax.plot(x, relevant, marker="o", label="executed/trained h0-9")
        ax.plot(x, future, marker=".", alpha=0.7, label="unexecuted preview h10-49")
        if first_positive_query >= 0:
            ax.axvline(first_positive_query * 10, color="green", linestyle="--", label="first positive reward")
        ax.set_title(label)
        ax.set_xlabel("executed action-slot start")
        ax.set_ylabel("log10 mean V_L3")
        ax.grid(alpha=0.25)
        if export_idx == 0:
            ax.legend(fontsize=8)

        ax = axes_c10[export_idx]
        image = ax.imshow(np.log10(np.maximum(v_l3[:, :10], 0) + EPS), aspect="auto", cmap="viridis")
        ax.set_title(f"{label}: applied C10")
        ax.set_xlabel("future action h (0-9)")
        ax.set_ylabel("query index")
        fig_c10.colorbar(image, ax=ax, fraction=0.046)

        ax = axes_h50[export_idx]
        image = ax.imshow(np.log10(np.maximum(v_l3, 0) + EPS), aspect="auto", cmap="viridis")
        ax.axvline(9.5, color="white", linestyle="--", linewidth=1)
        ax.set_title(f"{label}: teacher H50")
        ax.set_xlabel("future action h; left of line is applied C10")
        ax.set_ylabel("query index")
        fig_h50.colorbar(image, ax=ax, fraction=0.046)

    for axes in (axes_time, axes_c10, axes_h50):
        for ax in axes[len(chosen[:6]) :]:
            ax.axis("off")

    fig_time.suptitle("RLT Step475 reconstructed episode DVAC timelines", fontsize=15)
    fig_c10.suptitle("L3 DVAC inside the executed/trained C10 prefix", fontsize=15)
    fig_h50.suptitle("Same-query teacher H50 DVAC; h10-49 are diagnostic only", fontsize=15)
    fig_time.savefig(OUT / "FIG01_EPISODE_TIMELINES.png", dpi=160)
    fig_c10.savefig(OUT / "FIG02_C10_HEATMAPS.png", dpi=160)
    fig_h50.savefig(OUT / "FIG03_H50_CONTEXT_HEATMAPS.png", dpi=160)
    plt.close(fig_time)
    plt.close(fig_c10)
    plt.close(fig_h50)

    fig_progress, ax_progress = plt.subplots(figsize=(9, 5), constrained_layout=True)
    progress_stats: dict[str, list[dict]] = {}
    for success_value, label, color in ((1, "success", "tab:blue"), (0, "failure", "tab:red")):
        group_stats = []
        for progress_bin in range(10):
            values = np.asarray(
                [
                    item["c10_log"]
                    for item in progress_rows
                    if item["success"] == success_value and item["progress_bin"] == progress_bin
                ],
                dtype=np.float64,
            )
            group_stats.append(
                {
                    "progress_bin": progress_bin,
                    "count": int(values.size),
                    "mean": float(values.mean()),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                }
            )
        progress_stats[label] = group_stats
        x_progress = np.arange(10) / 9
        means = np.asarray([item["mean"] for item in group_stats])
        p25 = np.asarray([item["p25"] for item in group_stats])
        p75 = np.asarray([item["p75"] for item in group_stats])
        ax_progress.plot(x_progress, means, marker="o", color=color, label=f"{label} mean")
        ax_progress.fill_between(x_progress, p25, p75, color=color, alpha=0.15, label=f"{label} IQR")
    ax_progress.set_title("C10 DVAC over normalized episode progress (329 late-training episodes)")
    ax_progress.set_xlabel("normalized episode progress")
    ax_progress.set_ylabel("log10 mean V_L3 over h0-9")
    ax_progress.grid(alpha=0.25)
    ax_progress.legend()
    fig_progress.savefig(OUT / "FIG04_PROGRESS_AGGREGATE.png", dpi=160)
    plt.close(fig_progress)

    for name, rows in (("episode_summary.csv", summaries), ("queries.csv", query_rows)):
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(OUT / "selected_episodes.npz", **arrays)
    summary = {
        "checkpoint": str(checkpoint),
        "total_replay_rows": len(ids),
        "scanned_rows": len(ids) - start_pos,
        "complete_student_episodes": len(complete_episodes),
        "success_episodes": len(successes),
        "failure_episodes": len(failures),
        "selection_rule": "low/median/high mean log10 L3 DVAC within success and failure groups",
        "tail_minus_c10_log10": {
            "mean": float(np.mean(all_tail_minus_c10)),
            "median": float(np.median(all_tail_minus_c10)),
            "p05": float(np.quantile(all_tail_minus_c10, 0.05)),
            "p95": float(np.quantile(all_tail_minus_c10, 0.95)),
            "fraction_tail_higher": float(np.mean(np.asarray(all_tail_minus_c10) > 0)),
        },
        "normalized_progress": progress_stats,
        "counterfactual_pure_weight_stats_on_scanned_rows": counterfactual_weight_stats,
        "selected": summaries,
        "baseline": {"mean": baseline_mean, "std": baseline_std, "log_eps": log_eps},
        "output": str(OUT),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
