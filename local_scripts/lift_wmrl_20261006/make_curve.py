"""Plot the frozen lift_pot RM training history; no training or server access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=here / "reference/task-reward-v2__lift-pot__history.csv")
    parser.add_argument("--output-dir", type=Path, default=here.parents[1] / "docs/world-model/task_reward_plan_20261006")
    args = parser.parse_args()
    source = args.history.read_bytes()
    rows = list(csv.DictReader(source.decode("utf-8").splitlines()))
    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    validation = [float(row["val_loss"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)) or not all(math.isfinite(x) and x >= 0 for x in train + validation):
        raise ValueError("Invalid or non-contiguous training history")
    best_index = min(range(len(rows)), key=lambda index: validation[index])
    best_epoch = epochs[best_index]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), gridspec_kw={"width_ratios": [1, 1.15]})
    train_color, validation_color = "#e67f38", "#12666b"
    for axis in axes:
        axis.plot(epochs, train, color=train_color, lw=2, label="Training loss")
        axis.plot(epochs, validation, color=validation_color, lw=2, label="Validation loss")
        axis.axvline(best_epoch, color="#68737c", lw=1, linestyle="--")
        axis.scatter([best_epoch], [validation[best_index]], color=validation_color, s=38, zorder=5)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Binary cross-entropy")
        axis.grid(axis="y", color="#dce4e7", alpha=.8, linewidth=.7)
        axis.set_axisbelow(True)
    axes[0].set_title("All 33 epochs", loc="left")
    axes[0].set_xlim(1, epochs[-1])
    axes[0].set_ylim(0, max(train + validation) * 1.10)
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_title("After the initial drop", loc="left")
    axes[1].set_xlim(5, epochs[-1] + .7)
    axes[1].set_ylim(0, max(validation[4:]) * 1.27)
    axes[1].annotate(f"Selected epoch {best_epoch}\nval = {validation[best_index]:.3f}",
                     xy=(best_epoch, validation[best_index]), xytext=(12.8, .155),
                     arrowprops={"arrowstyle": "->", "color": "#546069"}, color="#273640")
    axes[1].annotate(f"Final val = {validation[-1]:.3f}", xy=(epochs[-1], validation[-1]),
                     xytext=(24.1, .183), arrowprops={"arrowstyle": "->", "color": "#546069"}, color="#273640")
    fig.suptitle("lift_pot reward model | keep the best validation checkpoint", x=.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(.07, .035, "Train: resampled 2:1 negative / positive frames. Validation: all held-out frames. Early stop: 15 stale epochs.",
             fontsize=8.5, color="#58646d")
    fig.subplots_adjust(left=.07, right=.985, bottom=.17, top=.81, wspace=.30)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png = args.output_dir / "lift_pot_rm_loss.png"
    csv_path = args.output_dir / "lift_pot_rm_history.csv"
    fig.savefig(png, dpi=170, facecolor="white")
    plt.close(fig)
    csv_path.write_bytes(source)
    summary = dict(source=str(args.history), source_sha256=hashlib.sha256(source).hexdigest(),
                   epochs=len(rows), best_epoch=best_epoch, best_val_loss=validation[best_index],
                   final_train_loss=train[-1], final_val_loss=validation[-1],
                   decision="Use best epoch18; continued same-data training is not supported by the validation curve",
                   png=str(png), csv=str(csv_path))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
