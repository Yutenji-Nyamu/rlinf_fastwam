"""Offline analysis for the first Idea2 DVAC telemetry collection.

This script is deliberately read-only with respect to the source run. It writes
derived CSV/JSON/PNG artifacts into a new, caller-provided output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr


TAIL_LENGTHS = (2, 3, 4)
EPS = 1e-12
SUCCESS_COLOR = "#3569b7"
FAIL_COLOR = "#c94f4f"
L_COLORS = {2: "#4c78a8", 3: "#f28e2b", 4: "#59a14f"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_shards(source: Path):
    query_parts: list[pd.DataFrame] = []
    episode_parts: list[pd.DataFrame] = []
    arrays: dict[str, list[np.ndarray]] = {}
    trace_hashes: dict[str, str] = {}
    csv_hashes: dict[str, str] = {}
    timestep_ref: np.ndarray | None = None

    trace_paths = sorted(source.glob("trace_rollout_rank*.npz"))
    if not trace_paths:
        raise RuntimeError(f"No trace shards found under {source}")

    for trace_path in trace_paths:
        rank = int(trace_path.stem.rsplit("rank", 1)[1])
        query_path = source / f"query_index_rollout_rank{rank:02d}.csv"
        if not query_path.exists():
            raise RuntimeError(f"Missing query index for rank {rank}: {query_path}")
        query = pd.read_csv(query_path)
        with np.load(trace_path, allow_pickle=False) as trace:
            required = {
                "x_chain",
                "z_endpoint",
                "final_model_action",
                "env_action",
                "robot_state",
                "timesteps",
            }
            missing = required.difference(trace.files)
            if missing:
                raise RuntimeError(f"{trace_path} missing arrays: {sorted(missing)}")
            if len(query) != trace["z_endpoint"].shape[0]:
                raise RuntimeError(
                    f"Rank {rank} query rows {len(query)} != trace rows "
                    f"{trace['z_endpoint'].shape[0]}"
                )
            if sorted(query["trace_row"].tolist()) != list(range(len(query))):
                raise RuntimeError(f"Rank {rank} trace_row is not a full 0..N-1 index")
            query = query.sort_values("trace_row").reset_index(drop=True)
            query["trace_rank"] = rank
            query_parts.append(query)
            for key in required.difference({"timesteps"}):
                arr = trace[key].copy()
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"Non-finite values in {trace_path}:{key}")
                arrays.setdefault(key, []).append(arr)
            timesteps = trace["timesteps"].copy()
            if not np.isfinite(timesteps).all():
                raise RuntimeError(f"Non-finite timesteps in {trace_path}")
            if timestep_ref is None:
                timestep_ref = timesteps
            elif not np.array_equal(timestep_ref, timesteps):
                raise RuntimeError("Timestep array differs across rollout ranks")
        trace_hashes[trace_path.name] = sha256(trace_path)
        csv_hashes[query_path.name] = sha256(query_path)

    for episode_path in sorted(source.glob("episode_index_env_rank*.csv")):
        episode_parts.append(pd.read_csv(episode_path))
        csv_hashes[episode_path.name] = sha256(episode_path)
    if not episode_parts:
        raise RuntimeError("No episode index shards found")

    query = pd.concat(query_parts, ignore_index=True)
    episode = pd.concat(episode_parts, ignore_index=True)
    combined = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    assert timestep_ref is not None

    query["array_row"] = np.arange(len(query), dtype=np.int64)
    episode["success"] = episode["success"].astype(bool)
    merged = query.merge(
        episode[["episode_idx", "reset_id", "success", "return", "termination_reason"]],
        on=["episode_idx", "reset_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["success"].isna().any():
        raise RuntimeError("Some query rows did not join to an episode outcome")
    if merged["query_uid"].duplicated().any():
        raise RuntimeError("Duplicate query_uid values")
    if len(merged) != combined["z_endpoint"].shape[0]:
        raise RuntimeError("Merged query count no longer matches trace arrays")

    return merged, episode, combined, timestep_ref, trace_hashes, csv_hashes


def compute_metrics(query: pd.DataFrame, arrays: dict[str, np.ndarray]):
    z = arrays["z_endpoint"]
    n, m, horizon, action_dim = z.shape
    if m != 4 or horizon != 50 or action_dim != 14:
        raise RuntimeError(f"Expected [N,4,50,14] z_endpoint, got {z.shape}")

    variance_by_l: dict[int, np.ndarray] = {}
    coordinate_by_l: dict[int, np.ndarray] = {}
    query_metrics = query.copy()
    horizon_rows: list[pd.DataFrame] = []

    base_horizon = pd.DataFrame(
        {
            "array_row": np.repeat(np.arange(n), horizon),
            "h": np.tile(np.arange(horizon), n),
        }
    )
    horizon_frame = base_horizon.merge(
        query[["array_row", "query_uid", "episode_idx", "query_idx", "action_slot_start", "reset_id", "success"]],
        on="array_row",
        how="left",
        validate="many_to_one",
    )

    for length in TAIL_LENGTHS:
        coordinate_variance = np.var(z[:, -length:, :, :], axis=1, ddof=0)
        per_h = coordinate_variance.sum(axis=-1)
        coordinate_by_l[length] = coordinate_variance
        variance_by_l[length] = per_h
        query_metrics[f"V_total_L{length}"] = per_h.sum(axis=1)
        query_metrics[f"log10_V_total_L{length}"] = np.log10(
            np.maximum(per_h.sum(axis=1), EPS)
        )
        horizon_frame[f"V_L{length}"] = per_h.reshape(-1)
        horizon_frame[f"log10_V_L{length}"] = np.log10(
            np.maximum(per_h.reshape(-1), EPS)
        )

    return query_metrics, horizon_frame, variance_by_l, coordinate_by_l


def select_representatives(query_metrics: pd.DataFrame) -> dict[str, object]:
    lcol = "log10_V_total_L3"
    episode_summary = (
        query_metrics.groupby(["episode_idx", "reset_id", "success"], as_index=False)
        .agg(mean_log=(lcol, "mean"), peak_log=(lcol, "max"))
        .sort_values("episode_idx")
    )
    success = episode_summary[episode_summary["success"]]
    failures = episode_summary[~episode_summary["success"]].sort_values("episode_idx")
    success_median = float(success["mean_log"].median())
    typical_success_row = success.iloc[(success["mean_log"] - success_median).abs().argmin()]
    high_success_row = success.iloc[success["peak_log"].argmax()]

    episode_ids: list[int] = [int(typical_success_row["episode_idx"])]
    high_success_id = int(high_success_row["episode_idx"])
    if high_success_id not in episode_ids:
        episode_ids.append(high_success_id)
    episode_ids.extend(int(value) for value in failures["episode_idx"].tolist())
    episode_ids = episode_ids[:4]

    high_query_row = query_metrics.loc[query_metrics[lcol].idxmax()]
    low_query_row = query_metrics.loc[query_metrics[lcol].idxmin()]
    return {
        "episode_ids": episode_ids,
        "typical_success_episode": int(typical_success_row["episode_idx"]),
        "high_variance_success_episode": high_success_id,
        "failure_episodes": [int(value) for value in failures["episode_idx"].tolist()],
        "high_query_uid": str(high_query_row["query_uid"]),
        "low_query_uid": str(low_query_row["query_uid"]),
    }


def set_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_l_sensitivity(
    output: Path,
    query_metrics: pd.DataFrame,
    horizon_metrics: pd.DataFrame,
    coordinate_by_l: dict[int, np.ndarray],
):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), constrained_layout=True)
    rng = np.random.default_rng(20260820)
    all_logs = []
    for xpos, length in enumerate(TAIL_LENGTHS, start=1):
        values = query_metrics[f"log10_V_total_L{length}"].to_numpy()
        all_logs.append(values)
        axes[0].scatter(
            xpos + rng.normal(0, 0.035, len(values)),
            values,
            s=13,
            alpha=0.45,
            color=L_COLORS[length],
            edgecolors="none",
        )
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axes[0].plot([xpos, xpos], [q1, q3], color="black", linewidth=5, alpha=0.7)
        axes[0].scatter([xpos], [med], color="white", edgecolor="black", s=26, zorder=4)
    axes[0].set_xticks([1, 2, 3], ["L=2", "L=3", "L=4"])
    axes[0].set_ylabel(r"$\log_{10}(V_{total})$")
    axes[0].set_title("Same queries, different tail lengths")

    corr = np.empty((3, 3), dtype=float)
    for i, left in enumerate(TAIL_LENGTHS):
        for j, right in enumerate(TAIL_LENGTHS):
            corr[i, j] = spearmanr(
                query_metrics[f"V_total_L{left}"],
                query_metrics[f"V_total_L{right}"],
            ).statistic
    im = axes[1].imshow(corr, vmin=0, vmax=1, cmap="Blues")
    axes[1].set_xticks(range(3), ["L2", "L3", "L4"])
    axes[1].set_yticks(range(3), ["L2", "L3", "L4"])
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{corr[i,j]:.3f}", ha="center", va="center")
    axes[1].set_title("Query-level Spearman rank")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    contribution = coordinate_by_l[3].sum(axis=(0, 1))
    contribution = contribution / contribution.sum()
    colors = ["#777777"] * len(contribution)
    colors[6] = "#e15759"
    colors[13] = "#e15759"
    axes[2].bar(np.arange(14), contribution * 100, color=colors)
    axes[2].set_xticks(np.arange(14))
    axes[2].set_xlabel("Normalized action coordinate d")
    axes[2].set_ylabel("Share of total L3 variance (%)")
    axes[2].set_title("Which coordinates dominate? (d6/d13=grippers)")
    fig.savefig(output / "FIG01_L_SENSITIVITY_AND_DIMENSIONS.png", bbox_inches="tight")
    plt.close(fig)


def save_query_position(output: Path, query_metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8), constrained_layout=True)
    for _, group in query_metrics.sort_values("query_idx").groupby("episode_idx"):
        group = group.sort_values("query_idx")
        color = SUCCESS_COLOR if bool(group["success"].iloc[0]) else FAIL_COLOR
        axes[0].plot(
            group["action_slot_start"],
            group["log10_V_total_L3"],
            color=color,
            alpha=0.35 if bool(group["success"].iloc[0]) else 0.9,
            linewidth=1,
            marker="o",
            markersize=3,
        )
    med = query_metrics.groupby("action_slot_start")["log10_V_total_L3"].median()
    axes[0].plot(med.index, med.values, color="black", linewidth=2.5, marker="o", label="median")
    axes[0].set_xticks([0, 50, 100, 150], ["q0\n0", "q1\n50", "q2\n100", "q3\n150"])
    axes[0].set_xlabel("Policy query / action-slot start")
    axes[0].set_ylabel(r"$\log_{10}(V_{total,L3})$")
    axes[0].set_title("Within-episode query progression (16 episodes)")
    axes[0].legend(frameon=False)

    rng = np.random.default_rng(20260820)
    for query_idx in range(4):
        group = query_metrics[query_metrics["query_idx"] == query_idx]
        x = query_idx + rng.normal(0, 0.045, len(group))
        colors = [SUCCESS_COLOR if flag else FAIL_COLOR for flag in group["success"]]
        axes[1].scatter(x, group["log10_V_total_L3"], c=colors, s=24, alpha=0.75)
        q1, median, q3 = np.quantile(group["log10_V_total_L3"], [0.25, 0.5, 0.75])
        axes[1].plot([query_idx, query_idx], [q1, q3], color="black", linewidth=5, alpha=0.65)
        axes[1].scatter(query_idx, median, color="white", edgecolor="black", zorder=4)
    axes[1].set_xticks(range(4), ["q0", "q1", "q2", "q3"])
    axes[1].set_xlabel("Query index (time proxy, not phase truth)")
    axes[1].set_ylabel(r"$\log_{10}(V_{total,L3})$")
    axes[1].set_title("Raw observations; red points are 2 failed episodes")
    fig.savefig(output / "FIG02_QUERY_POSITION.png", bbox_inches="tight")
    plt.close(fig)


def save_episode_grid(output: Path, query_metrics: pd.DataFrame):
    global_min = query_metrics["log10_V_total_L3"].min()
    global_max = query_metrics["log10_V_total_L3"].max()
    fig, axes = plt.subplots(4, 4, figsize=(12.2, 8.3), sharex=True, sharey=True, constrained_layout=True)
    for ax, (episode_idx, group) in zip(axes.flat, query_metrics.groupby("episode_idx", sort=True)):
        group = group.sort_values("query_idx")
        success = bool(group["success"].iloc[0])
        color = SUCCESS_COLOR if success else FAIL_COLOR
        ax.plot(group["action_slot_start"], group["log10_V_total_L3"], marker="o", color=color)
        post_success = group[group["success_before"].astype(bool)]
        if not post_success.empty:
            ax.scatter(
                post_success["action_slot_start"],
                post_success["log10_V_total_L3"],
                facecolors="white",
                edgecolors=color,
                s=35,
                linewidth=1.2,
                zorder=4,
            )
        ax.set_ylim(global_min - 0.08, global_max + 0.08)
        ax.set_title(
            f"ep{episode_idx:02d} reset {int(group['reset_id'].iloc[0])} {'S' if success else 'F'}",
            color=color,
        )
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xticks([0, 50, 100, 150], ["q0", "q1", "q2", "q3"])
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\log_{10} V_{total,L3}$")
    fig.suptitle("Paper Figure-13 style overview, but only four C50 policy queries per episode", fontsize=12)
    fig.savefig(output / "FIG03_ALL_EPISODE_TIMELINES.png", bbox_inches="tight")
    plt.close(fig)


def open_query_image(source: Path, relpath: str) -> Image.Image:
    return Image.open(source / relpath).convert("RGB")


def save_representative_story(
    source: Path,
    output: Path,
    query_metrics: pd.DataFrame,
    representatives: dict[str, object],
):
    episode_ids = representatives["episode_ids"]
    fig = plt.figure(figsize=(14.2, 3.0 * len(episode_ids)), constrained_layout=True)
    grid = fig.add_gridspec(len(episode_ids), 5, width_ratios=[1, 1, 1, 1, 1.25])
    for row_idx, episode_idx in enumerate(episode_ids):
        group = query_metrics[query_metrics["episode_idx"] == episode_idx].sort_values("query_idx")
        success = bool(group["success"].iloc[0])
        for col_idx, (_, record) in enumerate(group.iterrows()):
            ax = fig.add_subplot(grid[row_idx, col_idx])
            ax.imshow(open_query_image(source, record["head_image_relpath"]))
            phase_suffix = " / post-success" if bool(record["success_before"]) else " / pre-success"
            ax.set_title(
                f"q{int(record['query_idx'])} / slot {int(record['action_slot_start'])}{phase_suffix}"
            )
            ax.axis("off")
        ax = fig.add_subplot(grid[row_idx, 4])
        color = SUCCESS_COLOR if success else FAIL_COLOR
        ax.plot(group["action_slot_start"], group["log10_V_total_L3"], marker="o", color=color)
        for _, record in group.iterrows():
            ax.annotate(
                f"q{int(record['query_idx'])}",
                (record["action_slot_start"], record["log10_V_total_L3"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xlabel("action-slot start")
        ax.set_ylabel(r"$\log_{10} V_{total,L3}$")
        ax.set_title(
            f"ep{episode_idx:02d} reset {int(group['reset_id'].iloc[0])} — {'success' if success else 'failure'}",
            color=color,
        )
        ax.grid(alpha=0.2)
    fig.savefig(output / "FIG04_REPRESENTATIVE_EPISODE_STORIES.png", bbox_inches="tight")
    plt.close(fig)


def save_representative_heatmaps(
    output: Path,
    query_metrics: pd.DataFrame,
    variance_by_l: dict[int, np.ndarray],
    representatives: dict[str, object],
):
    episode_ids = representatives["episode_ids"]
    values = np.log10(np.maximum(variance_by_l[3], EPS))
    finite_min, finite_max = np.quantile(values, [0.01, 0.99])
    fig, axes = plt.subplots(len(episode_ids), 1, figsize=(12.2, 2.1 * len(episode_ids)), constrained_layout=True)
    if len(episode_ids) == 1:
        axes = [axes]
    last_im = None
    for ax, episode_idx in zip(axes, episode_ids):
        group = query_metrics[query_metrics["episode_idx"] == episode_idx].sort_values("query_idx")
        rows = group["array_row"].astype(int).to_numpy()
        last_im = ax.imshow(values[rows], aspect="auto", cmap="magma", vmin=finite_min, vmax=finite_max)
        ax.set_yticks(range(4), ["q0 / slot0", "q1 / slot50", "q2 / slot100", "q3 / slot150"])
        ax.set_xlabel("Future action index h within the generated H=50 chunk")
        success = bool(group["success"].iloc[0])
        ax.set_title(
            f"ep{episode_idx:02d}, reset {int(group['reset_id'].iloc[0])}, {'success' if success else 'failure'}",
            color=SUCCESS_COLOR if success else FAIL_COLOR,
        )
    assert last_im is not None
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label(r"$\log_{10} V_{L3}(h)$")
    fig.savefig(output / "FIG05_REPRESENTATIVE_HORIZON_HEATMAPS.png", bbox_inches="tight")
    plt.close(fig)


def save_query_curves(
    source: Path,
    output: Path,
    query_metrics: pd.DataFrame,
    variance_by_l: dict[int, np.ndarray],
    representatives: dict[str, object],
):
    query_uids = [representatives["low_query_uid"], representatives["high_query_uid"]]
    fig = plt.figure(figsize=(12.8, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2.5])
    for row_idx, query_uid in enumerate(query_uids):
        record = query_metrics[query_metrics["query_uid"] == query_uid].iloc[0]
        array_row = int(record["array_row"])
        ax_image = fig.add_subplot(grid[row_idx, 0])
        ax_image.imshow(open_query_image(source, record["head_image_relpath"]))
        ax_image.axis("off")
        ax_image.set_title(
            f"{query_uid}\n{'lowest' if row_idx == 0 else 'highest'} L3 Vtotal query"
        )
        ax = fig.add_subplot(grid[row_idx, 1])
        for length in TAIL_LENGTHS:
            ax.plot(
                np.arange(50),
                variance_by_l[length][array_row],
                color=L_COLORS[length],
                label=f"L={length}",
                linewidth=1.4,
            )
        peak_h = int(np.argmax(variance_by_l[3][array_row]))
        ax.axvline(peak_h, color="black", alpha=0.35, linewidth=1)
        ax.annotate(
            f"L3 peak h={peak_h}",
            (peak_h, variance_by_l[3][array_row, peak_h]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Future action index h")
        ax.set_ylabel(r"$V_L(h)$")
        ax.grid(which="both", alpha=0.18)
        ax.legend(frameon=False)
    fig.savefig(output / "FIG06_LOW_HIGH_QUERY_HORIZON_CURVES.png", bbox_inches="tight")
    plt.close(fig)


def save_endpoint_convergence(
    output: Path,
    query_metrics: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    variance_by_l: dict[int, np.ndarray],
    timesteps: np.ndarray,
    representatives: dict[str, object],
):
    query_uids = [representatives["low_query_uid"], representatives["high_query_uid"]]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.8), constrained_layout=True)
    for row_idx, query_uid in enumerate(query_uids):
        record = query_metrics[query_metrics["query_uid"] == query_uid].iloc[0]
        array_row = int(record["array_row"])
        z = arrays["z_endpoint"][array_row]
        final = arrays["final_model_action"][array_row]
        distance = np.linalg.norm(z - final[None, :, :], axis=-1)
        im = axes[row_idx, 0].imshow(
            np.log10(np.maximum(distance, EPS)), aspect="auto", cmap="viridis"
        )
        axes[row_idx, 0].set_yticks(range(4), [f"i{i}, t={value:g}" for i, value in enumerate(timesteps)])
        axes[row_idx, 0].set_xlabel("Future action index h")
        axes[row_idx, 0].set_title(
            f"{'Low' if row_idx == 0 else 'High'}-variance query: endpoint preview distance to final action"
        )
        cbar = fig.colorbar(im, ax=axes[row_idx, 0], fraction=0.03, pad=0.02)
        cbar.set_label(r"$\log_{10}\|z_i(h)-x_M(h)\|_2$")

        peak_h = int(np.argmax(variance_by_l[3][array_row]))
        stable_h = int(np.argmin(variance_by_l[3][array_row]))
        for h, marker, label in [(stable_h, "o", f"lowest-V h={stable_h}"), (peak_h, "s", f"highest-V h={peak_h}")]:
            axes[row_idx, 1].plot(
                range(4),
                distance[:, h],
                marker=marker,
                label=label,
            )
        axes[row_idx, 1].set_xticks(range(4), [f"i{i}\nt={value:g}" for i, value in enumerate(timesteps)])
        axes[row_idx, 1].set_yscale("log")
        axes[row_idx, 1].set_ylabel(r"$\|z_i(h)-x_M(h)\|_2$")
        axes[row_idx, 1].set_title("How the model revises two future actions")
        axes[row_idx, 1].grid(which="both", alpha=0.2)
        axes[row_idx, 1].legend(frameon=False)
    fig.savefig(output / "FIG07_ENDPOINT_CONVERGENCE.png", bbox_inches="tight")
    plt.close(fig)


def save_all_query_heatmap(
    output: Path,
    query_metrics: pd.DataFrame,
    variance_by_l: dict[int, np.ndarray],
):
    ordered = query_metrics.sort_values(["episode_idx", "query_idx"]).reset_index(drop=True)
    rows = ordered["array_row"].astype(int).to_numpy()
    matrix = np.log10(np.maximum(variance_by_l[3][rows], EPS))
    vmin, vmax = np.quantile(matrix, [0.01, 0.99])
    fig, ax = plt.subplots(figsize=(12.8, 8.2), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=vmin, vmax=vmax)
    for boundary in range(4, len(ordered), 4):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.35, alpha=0.6)
    centers = np.arange(16) * 4 + 1.5
    labels = []
    for episode_idx, group in ordered.groupby("episode_idx", sort=True):
        labels.append(f"ep{episode_idx:02d} {'S' if bool(group['success'].iloc[0]) else 'F'}")
    ax.set_yticks(centers, labels)
    ax.set_xlabel("Future action index h")
    ax.set_ylabel("Episode (each block contains q0..q3 from top to bottom)")
    ax.set_title("All 64 policy queries × 50 future action positions (L=3)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(r"$\log_{10} V_{L3}(h)$")
    fig.savefig(output / "FIG08_ALL_QUERY_HORIZON_HEATMAP.png", bbox_inches="tight")
    plt.close(fig)


def save_success_boundary(output: Path, query_metrics: pd.DataFrame):
    """Show the one coarse task predicate boundary that is actually recorded."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8), constrained_layout=True)
    success_episodes = query_metrics[query_metrics["success"]].copy()
    failure_episodes = query_metrics[~query_metrics["success"]].copy()

    for episode_idx, group in success_episodes.groupby("episode_idx"):
        group = group.sort_values("query_idx")
        axes[0].plot(
            group["action_slot_start"],
            group["log10_V_total_L3"],
            color=SUCCESS_COLOR,
            alpha=0.28,
            marker="o",
            markersize=3,
        )
    for episode_idx, group in failure_episodes.groupby("episode_idx"):
        group = group.sort_values("query_idx")
        axes[0].plot(
            group["action_slot_start"],
            group["log10_V_total_L3"],
            color=FAIL_COLOR,
            alpha=0.9,
            marker="s",
            markersize=4,
        )
    axes[0].axvspan(149, 151, color="black", alpha=0.12)
    axes[0].text(
        150,
        axes[0].get_ylim()[1],
        "14/16 episodes first report success\nbetween q2 and q3",
        ha="center",
        va="top",
        fontsize=8,
    )
    axes[0].set_xticks([0, 50, 100, 150], ["q0", "q1", "q2", "q3"])
    axes[0].set_xlabel("Action-slot start")
    axes[0].set_ylabel(r"$\log_{10} V_{total,L3}$")
    axes[0].set_title("Recorded task-predicate boundary (not a contact label)")

    categories = [
        ("q0 pre", query_metrics[query_metrics.query_idx == 0]),
        ("q1 pre", query_metrics[query_metrics.query_idx == 1]),
        ("q2 pre", query_metrics[query_metrics.query_idx == 2]),
        (
            "q3 post\n(n=14)",
            query_metrics[(query_metrics.query_idx == 3) & query_metrics.success_before.astype(bool)],
        ),
        (
            "q3 still pre\n(n=2)",
            query_metrics[(query_metrics.query_idx == 3) & ~query_metrics.success_before.astype(bool)],
        ),
    ]
    rng = np.random.default_rng(20260820)
    for xpos, (label, group) in enumerate(categories):
        values = group["log10_V_total_L3"].to_numpy()
        color = FAIL_COLOR if "still" in label else SUCCESS_COLOR
        axes[1].scatter(
            xpos + rng.normal(0, 0.045, len(values)), values, color=color, alpha=0.68, s=24
        )
        median = float(np.median(values))
        axes[1].plot([xpos - 0.16, xpos + 0.16], [median, median], color="black", linewidth=2)
    axes[1].set_xticks(range(len(categories)), [label for label, _ in categories])
    axes[1].set_ylabel(r"$\log_{10} V_{total,L3}$")
    axes[1].set_title("Post-success q3 is a different state regime")
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(output / "FIG09_RECORDED_SUCCESS_BOUNDARY.png", bbox_inches="tight")
    plt.close(fig)


def save_horizon_position_effect(
    output: Path,
    query_metrics: pd.DataFrame,
    variance_by_l: dict[int, np.ndarray],
):
    """Check whether variance structure is merely a far-future position effect."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.8), constrained_layout=True)
    pre_success = query_metrics[~query_metrics["success_before"].astype(bool)].copy()
    rows = pre_success["array_row"].astype(int).to_numpy()
    matrix = variance_by_l[3][rows]
    median = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, [0.25, 0.75], axis=0)
    h = np.arange(50)
    axes[0].fill_between(h, q25, q75, color="#9ecae1", alpha=0.45, label="IQR")
    axes[0].plot(h, median, color="#1f4e79", linewidth=2.2, label="median")
    for query_idx, color in zip([0, 1, 2], ["#59a14f", "#f28e2b", "#b07aa1"]):
        group = pre_success[pre_success["query_idx"] == query_idx]
        group_rows = group["array_row"].astype(int).to_numpy()
        axes[0].plot(
            h,
            np.median(variance_by_l[3][group_rows], axis=0),
            color=color,
            linewidth=1,
            alpha=0.75,
            label=f"q{query_idx} median",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Future action index h")
    axes[0].set_ylabel(r"$V_{L3}(h)$")
    axes[0].set_title("Pre-success queries: near vs far future")
    axes[0].grid(which="both", alpha=0.18)
    axes[0].legend(frameon=False, ncol=2)

    rng = np.random.default_rng(20260820)
    for query_idx in range(4):
        group = pre_success[pre_success["query_idx"] == query_idx]
        group_rows = group["array_row"].astype(int).to_numpy()
        if len(group_rows) == 0:
            continue
        near = variance_by_l[3][group_rows, :25].mean(axis=1)
        far = variance_by_l[3][group_rows, 25:].mean(axis=1)
        ratio = far / np.maximum(near, EPS)
        colors = [SUCCESS_COLOR if flag else FAIL_COLOR for flag in group["success"]]
        axes[1].scatter(
            query_idx + rng.normal(0, 0.045, len(ratio)), ratio, c=colors, s=25, alpha=0.72
        )
        axes[1].plot(
            [query_idx - 0.16, query_idx + 0.16],
            [np.median(ratio), np.median(ratio)],
            color="black",
            linewidth=2,
        )
    axes[1].axhline(1, color="black", linewidth=1, alpha=0.45)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(4), ["q0", "q1", "q2", "q3\n(fail only)"])
    axes[1].set_ylabel("mean V(h=25..49) / mean V(h=0..24)")
    axes[1].set_title("Per-query far/near variance ratio")
    axes[1].grid(axis="y", which="both", alpha=0.2)
    fig.savefig(output / "FIG10_HORIZON_POSITION_EFFECT.png", bbox_inches="tight")
    plt.close(fig)


def build_summary(
    source: Path,
    query_metrics: pd.DataFrame,
    episode: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    timesteps: np.ndarray,
    trace_hashes: dict[str, str],
    csv_hashes: dict[str, str],
    coordinate_by_l: dict[int, np.ndarray],
    representatives: dict[str, object],
) -> dict[str, object]:
    l_corr_query: dict[str, float] = {}
    l_corr_horizon: dict[str, float] = {}
    z = arrays["z_endpoint"]
    for left in TAIL_LENGTHS:
        for right in TAIL_LENGTHS:
            if left >= right:
                continue
            l_corr_query[f"L{left}_L{right}"] = float(
                spearmanr(
                    query_metrics[f"V_total_L{left}"],
                    query_metrics[f"V_total_L{right}"],
                ).statistic
            )
            left_h = np.var(z[:, -left:, :, :], axis=1, ddof=0).sum(axis=-1).reshape(-1)
            right_h = np.var(z[:, -right:, :, :], axis=1, ddof=0).sum(axis=-1).reshape(-1)
            l_corr_horizon[f"L{left}_L{right}"] = float(spearmanr(left_h, right_h).statistic)

    by_query_idx: dict[str, object] = {}
    for query_idx, group in query_metrics.groupby("query_idx"):
        by_query_idx[str(int(query_idx))] = {
            f"L{length}_median_V_total": float(group[f"V_total_L{length}"].median())
            for length in TAIL_LENGTHS
        }
        by_query_idx[str(int(query_idx))].update(
            {
                f"L{length}_median_log10_V_total": float(
                    group[f"log10_V_total_L{length}"].median()
                )
                for length in TAIL_LENGTHS
            }
        )

    by_outcome: dict[str, object] = {}
    for success, group in query_metrics.groupby("success"):
        name = "success" if bool(success) else "failure"
        by_outcome[name] = {
            "episode_count": int(group["episode_idx"].nunique()),
            "query_count": int(len(group)),
            **{
                f"L{length}_median_log10_V_total": float(
                    group[f"log10_V_total_L{length}"].median()
                )
                for length in TAIL_LENGTHS
            },
        }

    dimension_contribution = coordinate_by_l[3].sum(axis=(0, 1))
    dimension_contribution = dimension_contribution / dimension_contribution.sum()
    final_chain_delta = float(
        np.max(np.abs(arrays["x_chain"][:, -1] - arrays["final_model_action"]))
    )

    success_before_counts = (
        query_metrics.groupby(["query_idx", "success_before"]).size().astype(int).to_dict()
    )

    pre_success_rows = query_metrics.loc[
        ~query_metrics["success_before"].astype(bool), "array_row"
    ].astype(int).to_numpy()
    pre_matrix = np.var(z[pre_success_rows, -3:, :, :], axis=1, ddof=0).sum(axis=-1)
    pre_median_h = np.median(pre_matrix, axis=0)
    pre_near = pre_matrix[:, :25].mean(axis=1)
    pre_far = pre_matrix[:, 25:].mean(axis=1)
    pre_ratio = pre_far / np.maximum(pre_near, EPS)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "source_trace_sha256": trace_hashes,
        "source_csv_sha256": csv_hashes,
        "counts": {
            "episodes": int(episode["episode_idx"].nunique()),
            "queries": int(len(query_metrics)),
            "queries_per_episode": query_metrics.groupby("episode_idx").size().astype(int).to_dict(),
            "success_episodes": int(episode["success"].astype(bool).sum()),
            "failure_episodes": int((~episode["success"].astype(bool)).sum()),
            "horizon": int(arrays["z_endpoint"].shape[2]),
            "active_action_dim": int(arrays["z_endpoint"].shape[3]),
            "denoising_steps": int(arrays["z_endpoint"].shape[1]),
        },
        "timesteps": [float(value) for value in timesteps],
        "formula": {
            "coordinate_variance": "var(z[-L:,h,d], axis=denoise, ddof=0)",
            "V_L_h": "sum_d coordinate_variance",
            "V_total": "sum_h V_L_h",
            "tail_lengths": list(TAIL_LENGTHS),
        },
        "validation": {
            "all_trace_arrays_finite": True,
            "query_episode_join_complete": True,
            "unique_query_uid": True,
            "max_abs_x_chain_final_vs_final_model_action": final_chain_delta,
        },
        "L_query_level_spearman": l_corr_query,
        "L_horizon_level_spearman_descriptive_only": l_corr_horizon,
        "by_query_idx": by_query_idx,
        "by_outcome_descriptive_only": by_outcome,
        "success_before_counts": {
            f"q{int(query_idx)}_{'post_success' if bool(success_before) else 'pre_success'}": int(count)
            for (query_idx, success_before), count in success_before_counts.items()
        },
        "coarse_success_boundary": (
            "All 14 successful episodes have success_before=False at q2 and True at q3; "
            "the task predicate first becomes true during the executed q2 C50 chunk."
        ),
        "L3_dimension_variance_share": {
            f"d{index}": float(value) for index, value in enumerate(dimension_contribution)
        },
        "L3_horizon_position_effect_pre_success": {
            "query_count": int(len(pre_matrix)),
            "spearman_h_vs_across_query_median_V": float(
                spearmanr(np.arange(50), pre_median_h).statistic
            ),
            "median_far_half_over_near_half": float(np.median(pre_ratio)),
            "fraction_queries_far_half_greater_than_near_half": float(np.mean(pre_ratio > 1)),
            "fraction_queries_peak_in_far_half": float(np.mean(np.argmax(pre_matrix, axis=1) >= 25)),
        },
        "representatives": representatives,
        "evidence_limits": [
            "Four C50 queries per episode are query-level time points, not per-control-step phase resolution.",
            "No simulator contact or externally annotated MOVING/OPERATING label is present.",
            "Only two failed episodes; success/failure summaries are descriptive, not inferential.",
            "L2/L3/L4 use different denoising tails and their absolute scales are not interchangeable.",
            "No online adaptive execution occurred; N_exec effects cannot be estimated causally from this fixed-C50 run.",
        ],
    }


def write_tables(output: Path, query_metrics: pd.DataFrame, horizon_metrics: pd.DataFrame):
    query_columns = [
        "array_row",
        "trace_rank",
        "trace_row",
        "query_uid",
        "episode_idx",
        "query_idx",
        "action_slot_start",
        "reset_id",
        "success",
        "success_before",
        "source_env_rank",
        "local_env_slot",
        "head_image_relpath",
        "left_wrist_image_relpath",
        "right_wrist_image_relpath",
    ]
    for length in TAIL_LENGTHS:
        query_columns.extend([f"V_total_L{length}", f"log10_V_total_L{length}"])
    query_metrics[query_columns].sort_values(["episode_idx", "query_idx"]).to_csv(
        output / "query_metrics.csv", index=False
    )
    horizon_metrics.sort_values(["episode_idx", "query_idx", "h"]).to_csv(
        output / "horizon_metrics.csv", index=False
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"Source telemetry directory does not exist: {source}")
    if output.exists():
        raise SystemExit(f"Refusing to reuse existing output directory: {output}")
    output.mkdir(parents=True)

    query, episode, arrays, timesteps, trace_hashes, csv_hashes = read_shards(source)
    query_metrics, horizon_metrics, variance_by_l, coordinate_by_l = compute_metrics(query, arrays)
    representatives = select_representatives(query_metrics)

    set_plot_style()
    write_tables(output, query_metrics, horizon_metrics)
    save_l_sensitivity(output, query_metrics, horizon_metrics, coordinate_by_l)
    save_query_position(output, query_metrics)
    save_episode_grid(output, query_metrics)
    save_representative_story(source, output, query_metrics, representatives)
    save_representative_heatmaps(output, query_metrics, variance_by_l, representatives)
    save_query_curves(source, output, query_metrics, variance_by_l, representatives)
    save_endpoint_convergence(
        output,
        query_metrics,
        arrays,
        variance_by_l,
        timesteps,
        representatives,
    )
    save_all_query_heatmap(output, query_metrics, variance_by_l)
    save_success_boundary(output, query_metrics)
    save_horizon_position_effect(output, query_metrics, variance_by_l)

    summary = build_summary(
        source,
        query_metrics,
        episode,
        arrays,
        timesteps,
        trace_hashes,
        csv_hashes,
        coordinate_by_l,
        representatives,
    )
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"ANALYSIS_OUTPUT={output}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("FILES")
    for path in sorted(output.iterdir()):
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
