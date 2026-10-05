"""CPU-only compact plots of the completed Rynn numeric diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("engineering_passed") is not True:
        raise ValueError("A completed, engineering-passed numeric result is required")
    grouped = {}
    for row in result["cases"]:
        grouped.setdefault(row["episode_uid"], []).append(row)
    trends, controls, episode_ids = [], [], []
    fractions = [0, .25, .5, .75, 1]
    for uid, rows in sorted(grouped.items()):
        prefixes = {r["prefix_fraction"]: r["last_remaining_value"] for r in rows if r["kind"] == "expert_prefix"}
        by_kind = {r["kind"]: r["last_remaining_value"] for r in rows if r["kind"] != "expert_prefix"}
        if set(prefixes) != set(fractions):
            continue
        if not {"repeat_final", "reversed", "return_to_initial"} <= set(by_kind):
            raise ValueError(f"Missing temporal controls for {uid}")
        trends.append([prefixes[x] for x in fractions])
        controls.append([prefixes[1], by_kind["repeat_final"], by_kind["reversed"], by_kind["return_to_initial"]])
        episode_ids.append(uid)
    trends, controls = np.asarray(trends), np.asarray(controls)
    if trends.shape != (16, 5) or controls.shape != (16, 4) or not np.isfinite(trends).all() or not np.isfinite(controls).all():
        raise ValueError("Expected 16 complete experts with finite numeric outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.output_dir / f"rynn-numeric-probe.{ext}" for ext in ("png", "svg")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite prior figures")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    colors = plt.get_cmap("tab20")(np.linspace(0, .95, 16))
    x = np.asarray(fractions) * 100
    for row, color in zip(trends, colors):
        axes[0].plot(x, row, "o-", color=color, linewidth=.8, markersize=3, alpha=.45)
    q25, q50, q75 = np.percentile(trends, [25, 50, 75], axis=0)
    axes[0].fill_between(x, q25, q75, color="#174b79", alpha=.12, label="Middle 50%")
    axes[0].plot(x, q50, "o-", color="#174b79", linewidth=2.8, markersize=6, label="Median")
    axes[0].set(title="Expert trajectory progress", xlabel="Observed prefix (%)",
                ylabel="Predicted remaining value (raw; lower = closer)", xticks=x)
    axes[0].legend(frameon=False, loc="best")
    lower_count = int((trends[:, -1] < trends[:, 0]).sum())
    axes[0].text(.02, .97, f"Final < initial: {lower_count}/16", transform=axes[0].transAxes,
                 va="top", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    cx = np.arange(4)
    # Identical horizontal offsets preserve each episode's pairing across controls.
    offsets = np.linspace(-.12, .12, 16)
    for row, color, offset in zip(controls, colors, offsets):
        axes[1].plot(cx + offset, row, "o-", color=color, linewidth=.8, markersize=4, alpha=.5)
    medians = np.median(controls, axis=0)
    axes[1].plot(cx, medians, "D", color="#172c43", markersize=8, label="Median", zorder=5)
    axes[1].set(title="Same episodes, altered visual history", xticks=cx,
        xticklabels=["Full\nprefix", "Repeated\nfinal frame", "Reversed\nvideo", "Return to\ninitial frame"],
        ylabel="Predicted remaining value (raw; lower = closer)")
    axes[1].legend(frameon=False, loc="best")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("RynnValue numeric head | adjust_bottle | 16 expert demonstrations", fontsize=14, fontweight="bold", y=.98)
    fig.text(.5, .035, "K8, fixed 320x256 head camera. Reversal / return are synthetic controls, not real failures.\n"
             "Progress evidence only: no native success/failure accuracy or AUC is established.",
             ha="center", va="bottom", fontsize=9, color="#4b5563")
    fig.tight_layout(rect=(0, .12, 1, .92), w_pad=2.5)
    fig.savefig(targets[0], dpi=170, facecolor="white")
    fig.savefig(targets[1], facecolor="white")
    plt.close(fig)
    print(json.dumps(dict(outputs=[str(path) for path in targets], episodes=episode_ids,
        final_lower_count=lower_count, prefix_medians=q50.tolist(), control_medians=medians.tolist())))


if __name__ == "__main__":
    main()
