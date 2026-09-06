from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(r"C:\Users\86136\Documents\rl\.tmp\qam_status_20260801_1158")
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
KV_RE = re.compile(rf"([A-Za-z_][A-Za-z0-9_./-]*)=({NUMBER})")
STEP_RE = re.compile(r"Global Step:\s*(\d+)/(\d+)")


def parse(path: Path, low: int, high: int) -> list[dict[str, float]]:
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(STEP_RE.finditer(text))
    rows = []
    for index, match in enumerate(matches):
        cycle = int(match.group(1))
        if not low <= cycle <= high:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        row = {"cycle": float(cycle)}
        for key, value in KV_RE.findall(block):
            row["time/step" if key == "step" else key] = float(value)
        rows.append(row)
    return rows


rows = (
    parse(ROOT / "v2_driver.log", 1, 25)
    + parse(ROOT / "v3_driver.log", 26, 100)
    + parse(ROOT / "v4_driver.log", 101, 380)
)
rows.sort(key=lambda row: row["cycle"])


def vals(items, key):
    return [item[key] for item in items if key in item and math.isfinite(item[key])]


def avg(items, key):
    data = vals(items, key)
    return mean(data) if data else None


def outcome(items):
    episodes = sum(item.get("num_trajectories", 0.0) for item in items)
    successes = sum(
        item.get("success_once", 0.0) * item.get("num_trajectories", 0.0)
        for item in items
    )
    return {
        "cycles": len(items),
        "episodes": int(episodes),
        "successes": int(successes),
        "rate": successes / episodes if episodes else None,
    }


def metric_summary(items):
    result = outcome(items)
    for key in (
        "qam/critic_loss",
        "qam/critic_grad_norm",
        "qam/q_mean",
        "qam/q_std_heads",
        "qam/td_target_mean",
        "qam/am_loss",
        "qam/terminal_adjoint_norm",
        "qam/fine_grad_norm",
    ):
        result[key] = avg(items, key)
    return result


segments = {
    "collect_1_25": [row for row in rows if row["cycle"] <= 25],
    "q_only_26_51": [row for row in rows if 26 <= row["cycle"] <= 51],
    "am_52_75": [row for row in rows if 52 <= row["cycle"] <= 75],
    "am_76_100": [row for row in rows if 76 <= row["cycle"] <= 100],
    "am_101_latest": [row for row in rows if row["cycle"] >= 101],
}

blocks = {}
for start in range(1, int(rows[-1]["cycle"]) + 1, 10):
    items = [row for row in rows if start <= row["cycle"] <= start + 9]
    blocks[f"{start}_{start + 9}"] = outcome(items)

am_rows = [row for row in rows if row.get("qam/fine_updates", 0.0) > 0]
am_success = [row for row in am_rows if row.get("success_once", 0.0) > 0]
am_failure = [row for row in am_rows if row.get("success_once", 0.0) == 0]


def pearson(items, x_key, y_key):
    pairs = [
        (item[x_key], item[y_key])
        for item in items
        if x_key in item and y_key in item
        and math.isfinite(item[x_key]) and math.isfinite(item[y_key])
    ]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


latest = rows[-1]
report = {
    "latest_cycle": int(latest["cycle"]),
    "overall": outcome(rows),
    "segments": {name: metric_summary(items) for name, items in segments.items()},
    "ten_cycle_blocks": blocks,
    "am_success_cycles": metric_summary(am_success),
    "am_zero_success_cycles": metric_summary(am_failure),
    "same_cycle_correlations": {
        key: pearson(am_rows, "success_once", key)
        for key in (
            "qam/critic_loss",
            "qam/q_mean",
            "qam/q_std_heads",
            "qam/am_loss",
            "qam/terminal_adjoint_norm",
            "qam/fine_grad_norm",
        )
    },
    "scale": {
        "global_total_inserts": int(latest["qam/global_total_inserts"]),
        "critic_updates": int(latest["qam/critic_updates"]),
        "fine_updates": int(latest["qam/fine_updates"]),
        "estimated_positive_macro_fraction": (
            outcome(rows)["successes"] / latest["qam/global_total_inserts"]
        ),
        "fine_updates_per_observed_success": (
            latest["qam/fine_updates"] / outcome(rows)["successes"]
        ),
    },
    "endpoint_metrics": {
        key: latest.get(key)
        for key in (
            "qam/critic_loss",
            "qam/q_mean",
            "qam/q_std_heads",
            "qam/td_target_mean",
            "qam/am_loss",
            "qam/terminal_adjoint_norm",
            "qam/fine_grad_norm",
        )
    },
}
print(json.dumps(report, ensure_ascii=False, indent=2))
