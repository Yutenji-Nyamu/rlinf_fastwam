"""Regenerate paired eight-episode distillation curves from the archived table."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / "distill-analysis.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, method in zip(axes, ("endpoint", "onedp")):
    points = sorted(
        (r for r in data["training_points"] if r["method"] == method),
        key=lambda r: r["step"],
    )
    ax.plot(
        [0] + [p["step"] for p in points],
        [100 * 191 / 368] + [100 * p["matched8_rate"] for p in points],
        "o-",
        label=method,
    )
    ax.axhline(100 * 191 / 368, linestyle="--", color="gray", label="Raw N1")
    ax.axhline(100 * 193 / 368, linestyle=":", color="black", label="Raw N10")
    ax.axvspan(0, 556, color="gray", alpha=0.15)
    ax.set(title=method, xlabel="Training update", ylabel="Success (%)", ylim=(35, 65))
    ax.grid(alpha=0.2)
    ax.legend()
fig.suptitle("Matched eight episodes per task; no evaluation between 0 and 556")
fig.tight_layout()
fig.savefig(root / "plots" / "matched-success-replot.svg")
