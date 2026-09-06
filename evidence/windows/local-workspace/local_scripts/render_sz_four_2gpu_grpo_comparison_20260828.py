from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
OUT = ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "four-2gpu-grpo-comparison-live-20260828-1944"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
RUNS = {
    "control": {
        "name": "Control",
        "color": "#111827",
        "path": ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "grpo-control-2gpu-stopped-step96-20260828" / "raw" / "runtime" / "driver.log",
    },
    "st_dvac": {
        "name": "ST-DVAC",
        "color": "#D97706",
        "path": ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence" / "shenzhen_grpo_dvac_w0to2_2gpu_stopped_step52_light_evidence_20260827" / "runtime" / "driver.log",
    },
    "prism": {
        "name": "Prism",
        "color": "#0F766E",
        "path": ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "action-adv-prism-live-20260828-1944" / "raw" / "prism" / "runtime" / "driver.log",
    },
    "action_adv": {
        "name": "Action-Adv",
        "color": "#C026D3",
        "path": ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "action-adv-prism-live-20260828-1944" / "raw" / "action_adv" / "runtime" / "driver.log",
    },
}


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("plot helper unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def aligned(data: dict[str, dict], key: str, maximum: int) -> list[tuple[str, list[float | None], str, int]]:
    output = []
    for run in RUNS:
        mapping = dict(zip(data[run]["steps"], data[run][key]))
        output.append((RUNS[run]["name"], [mapping.get(step) for step in range(0, maximum + 1)], RUNS[run]["color"], 4))
    return output


def paired_mean_delta(data: dict[str, dict], method: str, through: int) -> float:
    control = data["control"]["raw"][:through]
    candidate = data[method]["raw"][:through]
    return (statistics.mean(candidate) - statistics.mean(control)) * 100


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_base()
    data: dict[str, dict] = {}
    for run, meta in RUNS.items():
        rows = base.parse_grpo(meta["path"])
        raw = [float(row["train_success"]) for row in rows]
        data[run] = {
            "steps": [int(row["step"]) for row in rows],
            "raw": raw,
            "ma5": base.trailing(raw, 5),
            "ma10": base.trailing(raw, 10),
            "evals": [(int(row["step"]), float(row["eval_success"])) for row in rows if row["eval_success"] is not None],
        }

    maximum = max(item["steps"][-1] for item in data.values())
    image = Image.new("RGB", (1700, 2250), base.LIGHT)
    draw = base.add_header(
        image,
        "Shenzhen matched two-GPU GRPO-family comparison — 19:44 CST",
        "Same 64x4/G8/B1024 training budget. Colors: black=Control, orange=ST-DVAC, teal=Prism, magenta=Action-Adv.",
    )
    x_all = list(range(0, maximum + 1))
    ticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 96]
    base.line_panel(draw, (45, 145, 1655, 630), x_all, aligned(data, "raw", maximum), 0.65, 1.005, "Raw training-rollout success", x_ticks=ticks)
    base.line_panel(draw, (45, 670, 1655, 1155), x_all, aligned(data, "ma5", maximum), 0.70, 1.005, "True trailing 5-step mean", x_ticks=ticks)
    base.line_panel(draw, (45, 1195, 1655, 1680), x_all, aligned(data, "ma10", maximum), 0.72, 1.005, "True trailing 10-step mean (Action-Adv begins at g10)", x_ticks=ticks)

    eval_x = sorted({0, *range(5, maximum + 1, 5), maximum})
    eval_series = []
    for run in RUNS:
        mapping = dict(data[run]["evals"])
        eval_series.append((RUNS[run]["name"], [mapping.get(step) for step in eval_x], RUNS[run]["color"], 4))
    base.line_panel(draw, (45, 1720, 1655, 2205), eval_x, eval_series, 0.65, 1.005, "Fixed-32 evaluation every 5 steps", x_ticks=ticks)
    figure = OUT / "01_four_2gpu_grpo_raw_ma5_ma10.png"
    image.save(figure, optimize=True)

    rows_out = []
    for step in x_all:
        row = {"step": step}
        for run in RUNS:
            for key in ("raw", "ma5", "ma10"):
                mapping = dict(zip(data[run]["steps"], data[run][key]))
                row[f"{run}_{key}"] = mapping.get(step)
            row[f"{run}_fixed32"] = dict(data[run]["evals"]).get(step)
        rows_out.append(row)
    with (OUT / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader(); writer.writerows(rows_out)

    summary = {"snapshot": "2026-08-28 19:44 CST", "runs": {}, "paired_train_mean_delta_vs_control_pp": {}}
    for run in RUNS:
        raw = data[run]["raw"]
        evals = data[run]["evals"]
        summary["runs"][run] = {
            "complete_step": data[run]["steps"][-1],
            "raw_pct": raw[-1] * 100,
            "ma5_pct": statistics.mean(raw[-5:]) * 100 if len(raw) >= 5 else None,
            "ma10_pct": statistics.mean(raw[-10:]) * 100 if len(raw) >= 10 else None,
            "fixed32_successes": sum(round(value * 32) for _, value in evals),
            "fixed32_episodes": 32 * len(evals),
        }
        if run != "control":
            through = data[run]["steps"][-1]
            summary["paired_train_mean_delta_vs_control_pp"][run] = paired_mean_delta(data, run, through)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(figure), "summary": summary}))


if __name__ == "__main__":
    main()
