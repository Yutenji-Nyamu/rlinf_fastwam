import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_training(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    field_map = {
        "success_once": "success",
        "reward": "reward",
        "actor/approx_kl": "approx_kl",
        "actor/clip_fraction": "clip_fraction",
        "actor/clipped_ratio": "clipped_ratio",
        "actor/grad_norm": "grad_norm",
        "actor/policy_loss": "policy_loss",
        "actor/policy_loss_abs": "policy_loss_abs",
        "actor/ratio": "ratio",
        "actor/ratio_abs": "ratio_abs",
        "actor/total_loss": "total_loss",
        "actor/lr": "lr",
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        step_match = re.search(r"Global Step:\s+(\d+)/\d+", line)
        if step_match:
            if current is not None:
                rows.append(current)
            current = {"step": float(step_match.group(1))}
            elapsed = re.search(r"Elapsed:\s+([0-9:]+)", line)
            step_time = re.search(r"Step Time:\s+(%s)s" % NUMBER, line)
            if elapsed:
                parts = [int(x) for x in elapsed.group(1).split(":")]
                if len(parts) == 3:
                    current["elapsed_s"] = float(parts[0] * 3600 + parts[1] * 60 + parts[2])
                elif len(parts) == 2:
                    current["elapsed_s"] = float(parts[0] * 60 + parts[1])
            if step_time:
                current["step_time_s"] = float(step_time.group(1))
            continue
        if current is None:
            continue
        for source, target in field_map.items():
            match = re.search(re.escape(source) + r"=(%s)" % NUMBER, line)
            if match:
                current[target] = float(match.group(1))
    if current is not None:
        rows.append(current)
    return [row for row in rows if "success" in row]


def parse_resources(path: Path) -> tuple[list[dict[str, float]], dict[str, float]]:
    minute_bins: dict[int, dict[str, float]] = defaultdict(dict)
    first_time: datetime | None = None
    peaks = {
        "ram_pct": 0.0,
        "gpu0_gib": 0.0,
        "gpu1_gib": 0.0,
        "gpu_total_gib": 0.0,
        "oom": 0.0,
        "oom_kill": 0.0,
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            if first_time is None:
                first_time = timestamp
            minute = int((timestamp - first_time).total_seconds() // 60)
            values = {
                "minutes": float(minute),
                "ram_pct": float(row["cgroup_ram_pct"]),
                "gpu0_gib": float(row["gpu0_memory_mb"]) / 1024.0,
                "gpu1_gib": float(row["gpu1_memory_mb"]) / 1024.0,
                "gpu_total_gib": float(row["gpu_total_memory_mb"]) / 1024.0,
            }
            bucket = minute_bins[minute]
            bucket["minutes"] = values["minutes"]
            for key in ("ram_pct", "gpu0_gib", "gpu1_gib", "gpu_total_gib"):
                bucket[key] = max(bucket.get(key, 0.0), values[key])
                peaks[key] = max(peaks[key], values[key])
            peaks["oom"] = max(peaks["oom"], float(row["cgroup_oom"]))
            peaks["oom_kill"] = max(peaks["oom_kill"], float(row["cgroup_oom_kill"]))
    return [minute_bins[key] for key in sorted(minute_bins)], peaks


def moving_average(values: list[float], window: int = 3) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        section = values[start : index + 1]
        result.append(sum(section) / len(section))
    return result


def style_axis(ax, x_max: float) -> None:
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, x_max)
    ax.spines[["top", "right"]].set_visible(False)


def save_success(rows: list[dict[str, float]], destination: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    success = [row["success"] * 100 for row in rows]
    ma3 = moving_average(success)
    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=160)
    ax.plot(steps, success, marker="o", linewidth=1.8, label="Per-step success")
    ax.plot(steps, ma3, marker="s", linewidth=2.2, label="3-step moving average")
    style_axis(ax, max(steps))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Global step")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Fast-WAM GRPO · move_stapler_pad · success")
    ax.legend(loc="best")
    ax.annotate(f"latest {success[-1]:.2f}%", (steps[-1], success[-1]), xytext=(-72, 12), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def save_optimizer(rows: list[dict[str, float]], destination: Path) -> None:
    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10.0), dpi=160, sharex=True)
    axes[0].plot(steps, [row.get("approx_kl", 0) for row in rows], marker="o", label="Approx KL")
    axes[0].plot(steps, [row.get("clip_fraction", 0) for row in rows], marker="s", label="Clip fraction")
    axes[0].axhline(0, linewidth=1, color="grey", alpha=0.5)
    axes[0].set_ylabel("Fraction")
    axes[0].legend(loc="best")
    style_axis(axes[0], max(steps))
    axes[1].plot(steps, [row.get("grad_norm", 0) for row in rows], marker="o", label="Grad norm")
    axes[1].set_ylabel("Gradient norm")
    axes[1].legend(loc="best")
    style_axis(axes[1], max(steps))
    axes[2].plot(steps, [row.get("ratio_abs", 0) for row in rows], marker="s", label="Ratio abs")
    axes[2].set_xlabel("Global step")
    axes[2].set_ylabel("Absolute ratio deviation")
    axes[2].legend(loc="best")
    style_axis(axes[2], max(steps))
    fig.suptitle("Fast-WAM GRPO · optimization metrics")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def save_resources(resources: list[dict[str, float]], peaks: dict[str, float], destination: Path) -> None:
    hours = [row["minutes"] / 60.0 for row in resources]
    x_max = max(hours) if hours else 1.0
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.0), dpi=160, sharex=True)
    axes[0].plot(hours, [row["gpu0_gib"] for row in resources], label="GPU 0")
    axes[0].plot(hours, [row["gpu1_gib"] for row in resources], label="GPU 1")
    axes[0].axhline(80, linewidth=1, color="grey", alpha=0.5, label="80 GiB capacity")
    axes[0].set_ylabel("GPU memory (GiB)")
    axes[0].legend(loc="best")
    style_axis(axes[0], x_max)
    axes[1].plot(hours, [row["ram_pct"] for row in resources], label="Cgroup RAM")
    axes[1].axhline(100, linewidth=1, color="grey", alpha=0.5, label="Limit")
    axes[1].set_xlabel("Hours since launch")
    axes[1].set_ylabel("Cgroup RAM (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="best")
    style_axis(axes[1], x_max)
    fig.suptitle(
        "Fast-WAM GRPO · resources\n"
        f"peaks: GPU {peaks['gpu0_gib']:.1f}/{peaks['gpu1_gib']:.1f} GiB · RAM {peaks['ram_pct']:.1f}%"
    )
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    training = parse_training(args.log)
    resources, peaks = parse_resources(args.resources)
    if not training:
        raise SystemExit("No complete training steps found")

    save_success(training, args.output_dir / "move-stapler-success.png")
    save_optimizer(training, args.output_dir / "move-stapler-optimizer.png")
    save_resources(resources, peaks, args.output_dir / "move-stapler-resources.png")
    args.summary.write_text(
        json.dumps({"training": training, "resources": resources, "peaks": peaks}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
