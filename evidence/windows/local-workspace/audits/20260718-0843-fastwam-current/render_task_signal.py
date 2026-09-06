from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "mobile-figures"
OUT.mkdir(parents=True, exist_ok=True)

tasks = [
    "adjust_bottle",
    "pick_diverse_bottles",
    "move_stapler_pad\n(seeds ready)",
    "place_can_basket",
    "turn_switch",
    "hanging_mug",
    "open_microwave",
]
clean = np.array([100, 80, 77, 71, 61, 58, 62])
randomized = np.array([100, 85, 64, 69, 59, 62, 45])
y = np.arange(len(tasks))

plt.rcParams.update({"font.size": 11, "figure.dpi": 160})
fig, ax = plt.subplots(figsize=(8, 5.6))
height = 0.36
ax.barh(y - height / 2, clean, height, label="Clean")
ax.barh(y + height / 2, randomized, height, label="Randomized")
ax.set_yticks(y, tasks)
ax.invert_yaxis()
ax.set_xlim(0, 105)
ax.set_xlabel("Official Fast-WAM success rate (%)")
ax.set_title("Selected RoboTwin tasks")
ax.grid(axis="x", alpha=0.25)
ax.legend(loc="lower right")
for row, value in enumerate(clean):
    ax.text(value + 0.8, row - height / 2, f"{value}", va="center", fontsize=9)
for row, value in enumerate(randomized):
    ax.text(value + 0.8, row + height / 2, f"{value}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "task-success-reference.png", bbox_inches="tight")
plt.close(fig)


def mixed(p: np.ndarray | float, group_size: int) -> np.ndarray | float:
    return 100 * (1 - np.power(p, group_size) - np.power(1 - p, group_size))


p = np.linspace(0, 1, 401)
current_p = 0.9728
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(p * 100, mixed(p, 4), label="group size 4", linewidth=2)
ax.plot(p * 100, mixed(p, 8), label="group size 8", linewidth=2)
ax.scatter([current_p * 100], [mixed(current_p, 4)], marker="D", s=60, zorder=5)
ax.annotate(
    f"current p=97.28%\n{mixed(current_p, 4):.1f}% mixed/group",
    (current_p * 100, mixed(current_p, 4)),
    xytext=(65, 25),
    arrowprops={"arrowstyle": "->"},
)
ax.scatter([50], [mixed(0.5, 4)], marker="^", s=65, zorder=5)
ax.set_xlim(0, 100)
ax.set_ylim(0, 102)
ax.set_xlabel("Policy success rate p (%)")
ax.set_ylabel("P(group has success and failure) (%)")
ax.set_title("GRPO mixed-outcome signal")
ax.grid(alpha=0.25)
ax.legend(loc="lower center")
fig.tight_layout()
fig.savefig(OUT / "grpo-mixed-signal.png", bbox_inches="tight")
plt.close(fig)
