from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SNAPSHOT_ROOT = HERE.parent / "extracted"
NPZ_FILES = sorted(SNAPSHOT_ROOT.rglob("rollout_step0022.npz"))

FONT_REGULAR = Path(r"C:\Windows\Fonts\segoeui.ttf")
FONT_SEMIBOLD = Path(r"C:\Windows\Fonts\seguisb.ttf")
INK = "#17233D"
MUTED = "#5E6C84"
GRID = "#D9E1EA"
BLUE = "#2563EB"
ORANGE = "#F97316"
GREEN = "#059669"
PURPLE = "#7C3AED"
WHITE = "#FFFFFF"


def font(size: int, semibold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(
        str(FONT_SEMIBOLD if semibold else FONT_REGULAR), size=size
    )


def concat(shards: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate(
        [shard[key].reshape(-1, *shard[key].shape[2:]) for shard in shards], axis=0
    )


if len(NPZ_FILES) != 2:
    raise RuntimeError(f"expected two rank shards, found {len(NPZ_FILES)}: {NPZ_FILES}")

shards = [dict(np.load(path, allow_pickle=False)) for path in NPZ_FILES]
weights = concat(shards, "weights")
advantages = concat(shards, "advantages")[:, 0]
valid = concat(shards, "loss_mask")[:, 0].astype(bool)
rewards = concat(shards, "rewards")
residual = concat(shards, "residual_z")
query_idx = concat(shards, "dvac_meta_query_idx")
source_rank = concat(shards, "dvac_meta_source_env_rank")
episode_idx = concat(shards, "dvac_meta_episode_index_within_step")

# Propagate the terminal success in one query back to every valid query from the
# same episode.  This is an offline label only; it is not used by training.
episode_rows: dict[tuple[int, int], list[int]] = defaultdict(list)
for index, key in enumerate(zip(source_rank.tolist(), episode_idx.tolist())):
    episode_rows[key].append(index)
episode_success = np.zeros(len(valid), dtype=bool)
for indices in episode_rows.values():
    success = bool(np.any(rewards[indices].sum(axis=1) > 0))
    episode_success[indices] = success


def weight_summary(mask: np.ndarray) -> dict[str, float]:
    values = weights[mask].reshape(-1)
    return {
        "count_queries": int(mask.sum()),
        "count_action_h": int(values.size),
        "mean_weight": float(values.mean()),
        "p05_weight": float(np.quantile(values, 0.05)),
        "median_weight": float(np.quantile(values, 0.50)),
        "p95_weight": float(np.quantile(values, 0.95)),
        "mean_residual": float(residual[mask].mean()),
    }


groups: dict[str, np.ndarray] = {
    "all_valid": valid,
    "successful_episode": valid & episode_success,
    "failed_episode": valid & ~episode_success,
}
for q in range(4):
    groups[f"query_idx_{q}"] = valid & (query_idx == q)

group_summary = {name: weight_summary(mask) for name, mask in groups.items() if mask.any()}

valid_weights = weights[valid]
weight_ess_by_query = (
    valid_weights.sum(axis=1) ** 2 / np.square(valid_weights).sum(axis=1)
) / valid_weights.shape[1]

uniform_credit = np.abs(advantages[valid, None] * np.ones_like(valid_weights)).reshape(-1)
weighted_credit = np.abs(advantages[valid, None] * valid_weights).reshape(-1)


def concentration(values: np.ndarray) -> dict[str, float]:
    ordered = np.sort(values)[::-1]
    n = len(ordered)
    return {
        "ess_ratio": float(values.sum() ** 2 / np.square(values).sum() / n),
        "top10_mass": float(ordered[: n // 10].sum() / ordered.sum()),
        "top20_mass": float(ordered[: n // 5].sum() / ordered.sum()),
    }


uniform_concentration = concentration(uniform_credit)
weighted_concentration = concentration(weighted_credit)

positive = valid & (advantages > 0)
negative = valid & (advantages < 0)


def mass_ratio(mask: np.ndarray) -> float:
    base = np.abs(advantages[mask, None] * np.ones_like(weights[mask])).sum()
    changed = np.abs(advantages[mask, None] * weights[mask]).sum()
    return float(changed / base)


credit_mass_ratio = {
    "positive_advantage": mass_ratio(positive),
    "negative_advantage": mass_ratio(negative),
    "all_absolute_credit": float(weighted_credit.sum() / uniform_credit.sum()),
}

summary = {
    "source_npz": [str(path.relative_to(HERE.parents[4])) for path in NPZ_FILES],
    "global_step": 23,
    "valid_queries": int(valid.sum()),
    "valid_action_h": int(valid.sum() * weights.shape[1]),
    "successful_episodes": int(
        sum(bool(np.any(rewards[indices].sum(axis=1) > 0)) for indices in episode_rows.values())
    ),
    "episodes": int(len(episode_rows)),
    "loss_valid_successful_episodes": int(
        sum(
            bool(np.any(rewards[indices].sum(axis=1) > 0))
            and bool(np.any(valid[indices]))
            for indices in episode_rows.values()
        )
    ),
    "loss_valid_failed_episodes": int(
        sum(
            not bool(np.any(rewards[indices].sum(axis=1) > 0))
            and bool(np.any(valid[indices]))
            for indices in episode_rows.values()
        )
    ),
    "group_summary": group_summary,
    "weight_ess_ratio_mean_by_query": float(weight_ess_by_query.mean()),
    "uniform_credit_concentration": uniform_concentration,
    "weighted_credit_concentration": weighted_concentration,
    "credit_mass_ratio_vs_uniform": credit_mass_ratio,
    "weight_residual_pearson": float(
        np.corrcoef(valid_weights.reshape(-1), residual[valid].reshape(-1))[0, 1]
    ),
}
(HERE / "CREDIT_CHAIN_G23.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)

with (HERE / "CREDIT_CHAIN_G23.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "group",
            "count_queries",
            "count_action_h",
            "mean_weight",
            "p05_weight",
            "median_weight",
            "p95_weight",
            "mean_residual",
        ],
    )
    writer.writeheader()
    for name, values in group_summary.items():
        writer.writerow({"group": name, **values})


def bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    low: float,
    high: float,
    value_format: str = ".3f",
    reference: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top), title, fill=INK, font=font(26, True))
    plot_left, plot_right = left + 95, right - 25
    plot_top, plot_bottom = top + 58, bottom - 75
    for tick in range(5):
        value = low + (high - low) * tick / 4
        y = plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        text_value = format(value, value_format)
        width = draw.textlength(text_value, font=font(16))
        draw.text((plot_left - width - 10, y - 9), text_value, fill=MUTED, font=font(16))
    if reference is not None:
        y = plot_bottom - (reference - low) / (high - low) * (plot_bottom - plot_top)
        for x in range(plot_left, plot_right, 18):
            draw.line((x, y, min(x + 9, plot_right), y), fill=MUTED, width=2)
    slot = (plot_right - plot_left) / len(values)
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        x0 = plot_left + index * slot + slot * 0.18
        x1 = plot_left + (index + 1) * slot - slot * 0.18
        y = plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)
        draw.rounded_rectangle((x0, y, x1, plot_bottom), radius=7, fill=color)
        value_text = format(value, value_format)
        value_width = draw.textlength(value_text, font=font(17, True))
        draw.text(((x0 + x1 - value_width) / 2, y - 26), value_text, fill=INK, font=font(17, True))
        label_width = draw.textlength(label, font=font(16))
        draw.text(((x0 + x1 - label_width) / 2, plot_bottom + 12), label, fill=INK, font=font(16))


image = Image.new("RGB", (1800, 1120), WHITE)
draw = ImageDraw.Draw(image)
draw.text((70, 45), "R-only DVAC: credit-allocation mechanism at Global Step 23", fill=INK, font=font(41, True))
draw.text(
    (72, 103),
    "Coefficient-space diagnostics before query-level PPO gating and global gradient clipping.",
    fill=MUTED,
    font=font(21),
)

bar_panel(
    draw,
    (55, 155, 885, 580),
    "A. Mean action-h weight by final episode outcome",
    ["success", "failure"],
    [group_summary["successful_episode"]["mean_weight"], group_summary["failed_episode"]["mean_weight"]],
    [GREEN, ORANGE],
    0.94,
    1.06,
    reference=1.0,
)
bar_panel(
    draw,
    (915, 155, 1745, 580),
    "B. Mean action-h weight by query index",
    ["q0", "q1", "q2", "q3"],
    [group_summary[f"query_idx_{q}"]["mean_weight"] for q in range(4)],
    [BLUE, PURPLE, GREEN, ORANGE],
    0.94,
    1.07,
    reference=1.0,
)
bar_panel(
    draw,
    (55, 620, 885, 1075),
    "C. Absolute-credit concentration",
    ["top10 base", "top10 w", "top20 base", "top20 w"],
    [
        uniform_concentration["top10_mass"],
        weighted_concentration["top10_mass"],
        uniform_concentration["top20_mass"],
        weighted_concentration["top20_mass"],
    ],
    ["#93C5FD", BLUE, "#C4B5FD", PURPLE],
    0.20,
    0.45,
    value_format=".3f",
)
bar_panel(
    draw,
    (915, 620, 1745, 1075),
    "D. Credit mass relative to uniform GRPO",
    ["positive A", "negative A", "all |A|"],
    [
        credit_mass_ratio["positive_advantage"],
        credit_mass_ratio["negative_advantage"],
        credit_mass_ratio["all_absolute_credit"],
    ],
    [GREEN, ORANGE, PURPLE],
    0.96,
    1.04,
    reference=1.0,
)
image.save(HERE / "DVAC_CREDIT_CHAIN_G23.png", optimize=True)

print(json.dumps(summary, indent=2, ensure_ascii=False))
