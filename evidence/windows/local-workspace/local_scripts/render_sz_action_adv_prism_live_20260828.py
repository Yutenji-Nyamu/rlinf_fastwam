from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
SNAPSHOT = ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "action-adv-prism-live-20260828-1424"
RAW = SNAPSHOT / "raw"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("plot helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run(base, label: str) -> dict[str, object]:
    rows = base.parse_grpo(RAW / label / "runtime" / "driver.log")
    values = [float(row["train_success"]) for row in rows]
    return {
        "steps": [int(row["step"]) for row in rows],
        "raw": values,
        "evals": [(int(row["step"]), float(row["eval_success"])) for row in rows if row["eval_success"] is not None],
    }


def tick_positions(last_step: int) -> list[int]:
    candidates = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, last_step]
    return sorted({step for step in candidates if step <= last_step})


def main() -> None:
    global SNAPSHOT, RAW
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--label", default="2026-08-28 14:24 CST")
    cli = parser.parse_args()
    SNAPSHOT = cli.snapshot
    RAW = SNAPSHOT / "raw"
    base = load_base()
    action = load_run(base, "action_adv")
    prism = load_run(base, "prism")
    action["ma5"] = base.trailing(action["raw"], 5)
    action["ma10"] = base.trailing(action["raw"], 10)
    prism["ma5"] = base.trailing(prism["raw"], 5)
    prism["ma10"] = base.trailing(prism["raw"], 10)

    image = Image.new("RGB", (1600, 1810), base.LIGHT)
    draw = base.add_header(
        image,
        f"Live Shenzhen two-GPU GRPO methods — {cli.label}",
        "Raw training-rollout success with true trailing 5/10-step means; fixed-32 markers use the same reset set.",
    )
    base.line_panel(
        draw,
        (45, 145, 1555, 650),
        action["steps"],
        [("Action-Adv raw", action["raw"], "#D97706", 4), ("5-step mean", action["ma5"], "#0F766E", 5), ("10-step mean", action["ma10"], "#374151", 4)],
        0.65,
        1.005,
        f"GRPO-DVAC Action-Adv [0,2] — complete through Step {action['steps'][-1]}",
        x_ticks=tick_positions(action["steps"][-1]),
    )
    base.line_panel(
        draw,
        (45, 690, 1555, 1195),
        prism["steps"],
        [("Prism raw", prism["raw"], "#D97706", 4), ("5-step mean", prism["ma5"], "#0F766E", 5), ("10-step mean", prism["ma10"], "#374151", 4)],
        0.65,
        1.005,
        f"Prism-style DVAC-Rank-RLOO — complete through Step {prism['steps'][-1]}",
        x_ticks=tick_positions(prism["steps"][-1]),
    )

    maximum = max(action["steps"][-1], prism["steps"][-1])
    fixed_steps = list(range(1, maximum + 1))
    action_map = dict(action["evals"])
    prism_map = dict(prism["evals"])
    action_fixed = [action_map.get(step) for step in fixed_steps]
    prism_fixed = [prism_map.get(step) for step in fixed_steps]
    plot = base.line_panel(
        draw,
        (45, 1235, 1555, 1765),
        fixed_steps,
        [("Action-Adv fixed-32", action_fixed, "#D97706", 4), ("Prism fixed-32", prism_fixed, "#0F766E", 4)],
        0.65,
        1.005,
        "Fixed-32 evaluation every 5 steps",
        x_ticks=tick_positions(maximum),
    )
    base.draw_eval_markers(draw, plot, fixed_steps, 0.65, 1.005, action["evals"], "#D97706", "square")
    base.draw_eval_markers(draw, plot, fixed_steps, 0.65, 1.005, prism["evals"], "#0F766E", "triangle")
    figure = SNAPSHOT / "01_action_adv_prism_success_live.png"
    image.save(figure, optimize=True)

    rows = []
    for step in range(1, maximum + 1):
        row = {"step": step}
        for label, data in (("action_adv", action), ("prism", prism)):
            if step in data["steps"]:
                index = data["steps"].index(step)
                row[f"{label}_raw"] = data["raw"][index]
                row[f"{label}_ma5"] = data["ma5"][index]
                row[f"{label}_ma10"] = data["ma10"][index]
            else:
                row[f"{label}_raw"] = row[f"{label}_ma5"] = row[f"{label}_ma10"] = None
            row[f"{label}_fixed32"] = dict(data["evals"]).get(step)
        rows.append(row)
    with (SNAPSHOT / "success_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "snapshot": cli.label,
        "action_adv": {
            "complete_step": action["steps"][-1],
            "raw": action["raw"][-1],
            "ma5": statistics.mean(action["raw"][-5:]),
            "ma10": statistics.mean(action["raw"][-10:]) if len(action["raw"]) >= 10 else None,
            "fixed32": action["evals"],
        },
        "prism": {
            "complete_step": prism["steps"][-1],
            "raw": prism["raw"][-1],
            "ma5": statistics.mean(prism["raw"][-5:]),
            "ma10": statistics.mean(prism["raw"][-10:]),
            "fixed32": prism["evals"],
        },
    }
    (SNAPSHOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(figure), "summary": summary}))


if __name__ == "__main__":
    main()
