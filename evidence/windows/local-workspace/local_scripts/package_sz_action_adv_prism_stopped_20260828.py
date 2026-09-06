"""Render and package two small post-stop evidence bundles from downloaded files."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import statistics
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
DEFAULT_SNAPSHOT = (
    ROOT
    / "docs"
    / "rlinf-shenzhen-grpo-dvac-action-adv"
    / "evidence"
    / "action-adv-prism-stopped-20260828"
)
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
RUNS = {
    "action_adv": {
        "title": "GRPO-DVAC Action-Adv [0,2]",
        "name": "shenzhen_grpo_dvac_action_adv_w0to2_stopped_light_evidence_20260828",
        "remote_run": (
            "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/"
            "dvac-action-adv-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v1"
        ),
        "gpus": (4, 5),
        "caveat": (
            "This stopped v1 used mean reduction over H=50 action losses. Its pre-clip gradient "
            "scale was about 45.6x below matched Control. Action-level ratio/clip telemetry also "
            "used a Bx1 denominator for BxH values; those old ratio/clip fields are retained only "
            "as raw evidence and are not plotted as comparable metrics."
        ),
    },
    "prism": {
        "title": "Prism-style DVAC-Rank-RLOO v2",
        "name": "shenzhen_prism_dvac_rank_rloo_v2_stopped_light_evidence_20260828",
        "remote_run": (
            "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/"
            "prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2"
        ),
        "gpus": (6, 7),
        "caveat": (
            "This is the local-shard v2 run. Checkpoint shards remain on the server and are "
            "intentionally excluded from this light evidence bundle."
        ),
    },
}


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_success(base, title: str, rows: list[dict], output: Path) -> None:
    steps = [float(row["step"]) for row in rows]
    raw = [float(row["train_success"]) for row in rows]
    ma5 = base.trailing(raw, 5)
    ma10 = base.trailing(raw, 10)
    evals = [
        (int(row["step"]), float(row["eval_success"]))
        for row in rows
        if row["eval_success"] is not None
    ]
    image = Image.new("RGB", (1500, 1260), base.LIGHT)
    draw = base.add_header(
        image,
        f"{title}: stopped-run success",
        "Raw training-rollout success, true trailing 5/10-step means, and fixed-32 evaluation.",
    )
    base.line_panel(
        draw,
        (45, 145, 1455, 690),
        steps,
        [
            ("raw success", raw, "#D97706", 4),
            ("5-step mean", ma5, "#0F766E", 5),
            ("10-step mean", ma10, "#374151", 4),
        ],
        min(0.60, min(raw) - 0.02),
        1.005,
        "Training rollout success",
    )
    if evals:
        eval_steps = [float(step) for step, _ in evals]
        eval_values = [value for _, value in evals]
        low = min(0.70, min(eval_values) - 0.02)
        plot = base.line_panel(
            draw,
            (45, 730, 1455, 1220),
            eval_steps,
            [("fixed-32 success", eval_values, "#0F766E", 4)],
            low,
            1.005,
            "Fixed-32 evaluation every 5 steps",
        )
        base.draw_eval_markers(draw, plot, eval_steps, low, 1.005, evals, "#D97706", "square")
    image.save(output, optimize=True)


def render_training_dynamics(base, title: str, rows: list[dict], output: Path) -> None:
    steps = [float(row["step"]) for row in rows]
    grad = [float(row["grad_norm"]) for row in rows]
    step_time = [float(row["step_time_s"]) / 60 for row in rows]
    image = Image.new("RGB", (1500, 1080), base.LIGHT)
    draw = base.add_header(
        image,
        f"{title}: optimizer scale and wall time",
        "Pre-clip gradient norm and completed outer-step duration; no action-level ratio metric is reinterpreted.",
    )
    base.line_panel(
        draw,
        (45, 145, 1455, 575),
        steps,
        [("pre-clip grad norm", grad, "#D97706", 4)],
        0,
        max(1.0, max(grad) * 1.08),
        "Gradient norm",
    )
    base.line_panel(
        draw,
        (45, 620, 1455, 1040),
        steps,
        [("minutes / outer step", step_time, "#0F766E", 4)],
        0,
        max(1.0, max(step_time) * 1.08),
        "Step time",
    )
    image.save(output, optimize=True)


def render_resources(base, title: str, source: Path, gpus: tuple[int, int], output: Path) -> dict[str, float | int]:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("host_mem_available_kib")
            and all(row.get(f"gpu{gpu}_used_mib") for gpu in gpus)
        ]
    if not rows:
        raise RuntimeError(f"resource telemetry is empty: {source}")
    xs = list(range(len(rows)))
    host = [float(row["host_mem_available_kib"]) / 1024 / 1024 for row in rows]
    gpu_values = {
        gpu: [float(row[f"gpu{gpu}_used_mib"]) / 1024 for row in rows]
        for gpu in gpus
    }
    image = Image.new("RGB", (1500, 1080), base.LIGHT)
    draw = base.add_header(
        image,
        f"{title}: resource trajectory",
        "One-minute observer samples; GPU series are allocated GiB and host series is available RAM.",
    )
    base.line_panel(
        draw,
        (45, 145, 1455, 575),
        xs,
        [
            (f"GPU{gpus[0]}", gpu_values[gpus[0]], "#D97706", 4),
            (f"GPU{gpus[1]}", gpu_values[gpus[1]], "#0F766E", 4),
        ],
        0,
        max(80.0, max(gpu_values[gpus[0]] + gpu_values[gpus[1]]) * 1.05),
        "GPU memory",
    )
    base.line_panel(
        draw,
        (45, 620, 1455, 1040),
        xs,
        [("Host available GiB", host, "#374151", 4)],
        0,
        max(host) * 1.05,
        "Host available memory",
    )
    image.save(output, optimize=True)
    return {
        "observer_rows": len(rows),
        f"gpu{gpus[0]}_peak_gib": max(gpu_values[gpus[0]]),
        f"gpu{gpus[1]}_peak_gib": max(gpu_values[gpus[1]]),
        "host_available_min_gib": min(host),
        "host_available_final_gib": host[-1],
    }


def build_one(base, snapshot: Path, label: str, spec: dict[str, object]) -> dict[str, object]:
    raw = snapshot / "raw" / label
    name = str(spec["name"])
    output = snapshot / name
    zip_path = ROOT / "exports" / f"{name}.zip"
    for directory in (output / "runtime", output / "tensorboard", output / "derived", output / "figures"):
        directory.mkdir(parents=True, exist_ok=False)
    for section in ("runtime", "tensorboard"):
        for source in sorted((raw / section).glob("*")):
            if source.is_file():
                shutil.copy2(source, output / section / source.name)

    rows = base.parse_grpo(raw / "runtime" / "driver.log")
    if not rows:
        raise RuntimeError(f"no complete outer step found for {label}")
    with (output / "derived" / "core_step_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evals = [
        {
            "step": int(row["step"]),
            "fixed32_success_rate": float(row["eval_success"]),
            "successes": round(float(row["eval_success"]) * 32),
        }
        for row in rows
        if row["eval_success"] is not None
    ]
    if evals:
        with (output / "derived" / "fixed32_eval_points.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(evals[0]))
            writer.writeheader()
            writer.writerows(evals)

    title = str(spec["title"])
    render_success(base, title, rows, output / "figures" / "success_raw_ma5_ma10_fixed32.png")
    render_training_dynamics(base, title, rows, output / "figures" / "grad_norm_and_step_time.png")
    resources = render_resources(
        base,
        title,
        raw / "runtime" / "resource.csv",
        tuple(spec["gpus"]),
        output / "figures" / "resources_gpu_host.png",
    )
    success = [float(row["train_success"]) for row in rows]
    summary = {
        "method": title,
        "source_run": spec["remote_run"],
        "last_complete_step": int(rows[-1]["step"]),
        "termination": spec.get(
            "termination",
            "user-authorized exact stop to replace this run with the next experiment",
        ),
        "train_success_mean_pct": statistics.mean(success) * 100,
        "train_success_latest_pct": success[-1] * 100,
        "train_success_ma5_pct": statistics.mean(success[-5:]) * 100,
        "train_success_ma10_pct": statistics.mean(success[-10:]) * 100,
        "fixed32_successes": sum(int(row["successes"]) for row in evals),
        "fixed32_episodes": 32 * len(evals),
        "fixed32_points": evals,
        "latest_core_metrics": rows[-1],
        "resources": resources,
        "known_caveat": spec["caveat"],
        "excluded": ["checkpoints", "videos", "RoboTwin bulk data", "Ray session logs", "per-action tensors"],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        f"# {title} 轻量收尾包\n\n"
        f"最后完整 outer step：`{summary['last_complete_step']}`。\n\n"
        "包含 driver/resource 日志、resolved 与小型运行合同、TensorBoard event/config、逐步核心指标、"
        "fixed-32 评估点和三张图。\n\n"
        f"注意：{spec['caveat']}\n\n"
        "不含 checkpoint、视频、RoboTwin 大数据、Ray 全量日志或逐动作 tensor。\n",
        encoding="utf-8",
    )
    manifest = [
        {"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*"))
        if path.is_file()
    ]
    (output / "file_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{name}/{path.relative_to(output)}")
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
    return {
        "label": label,
        "zip": str(zip_path),
        "bytes": zip_path.stat().st_size,
        "files": len(manifest) + 1,
        "last_complete_step": summary["last_complete_step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    cli = parser.parse_args()
    for spec in RUNS.values():
        output = cli.snapshot / str(spec["name"])
        zip_path = ROOT / "exports" / f"{spec['name']}.zip"
        if output.exists() or zip_path.exists():
            raise FileExistsError(f"refusing to overwrite {output} or {zip_path}")
    (ROOT / "exports").mkdir(parents=True, exist_ok=True)
    base = load_base()
    results = [build_one(base, cli.snapshot, label, spec) for label, spec in RUNS.items()]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
