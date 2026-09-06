from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import statistics
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv" / "evidence" / "grpo-control-2gpu-stopped-step96-20260828"
RAW = EVIDENCE / "raw"
NAME = "shenzhen_grpo_control_2gpu_stopped_step96_light_evidence_20260828"
OUT = EVIDENCE / NAME
ZIP_PATH = ROOT / "exports" / f"{NAME}.zip"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_success(base, rows: list[dict], output: Path) -> None:
    steps = [float(row["step"]) for row in rows]
    raw = [float(row["train_success"]) for row in rows]
    ma5 = base.trailing(raw, 5)
    ma10 = base.trailing(raw, 10)
    evals = [(int(row["step"]), row["eval_success"]) for row in rows if row["eval_success"] is not None]
    image = Image.new("RGB", (1500, 1260), base.LIGHT)
    draw = base.add_header(
        image,
        "Two-GPU clean GRPO Control stopped after Step 96",
        "Training rollout success, 5/10-step moving means, and fixed-32 evaluation. Orange and deep teal are intentionally distinct.",
    )
    base.line_panel(
        draw,
        (45, 145, 1455, 690),
        steps,
        [("raw success", raw, "#D97706", 4), ("5-step mean", ma5, "#0F766E", 5), ("10-step mean", ma10, "#374151", 4)],
        min(0.65, min(raw) - 0.02),
        1.005,
        "Training rollout success",
    )
    eval_values = [float(value) for _, value in evals]
    plot = base.line_panel(
        draw,
        (45, 730, 1455, 1220),
        [float(step) for step, _ in evals],
        [("fixed-32 success", eval_values, "#0F766E", 4)],
        min(0.75, min(eval_values) - 0.02),
        1.005,
        "Fixed-32 evaluation every 5 steps",
    )
    base.draw_eval_markers(draw, plot, [float(step) for step, _ in evals], min(0.75, min(eval_values) - 0.02), 1.005, evals, "#D97706", "square")
    image.save(output, optimize=True)


def render_optimizer(base, rows: list[dict], output: Path) -> None:
    steps = [float(row["step"]) for row in rows]
    specs = [
        ("Approximate KL", "approx_kl", "#0F766E"),
        ("Clip fraction", "clip_fraction", "#D97706"),
        ("Gradient norm", "grad_norm", "#374151"),
    ]
    image = Image.new("RGB", (1500, 1570), base.LIGHT)
    draw = base.add_header(image, "Two-GPU clean GRPO Control: optimizer signals", "All 96 completed outer steps; raw finite values from the driver metric tables.")
    for index, (title, key, color) in enumerate(specs):
        values = [float(row[key]) for row in rows]
        low = min(0.0, min(values) * 1.05)
        high = max(values) * 1.08 if max(values) > 0 else 1.0
        base.line_panel(draw, (45, 140 + index * 475, 1455, 575 + index * 475), steps, [(title, values, color, 4)], low, high, title)
    image.save(output, optimize=True)


def render_resources(base, raw_path: Path, output: Path) -> dict[str, float | int]:
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("host_mem_available_kib")]
    xs = list(range(len(rows)))
    host = [float(row["host_mem_available_kib"]) / 1024 / 1024 for row in rows]
    gpu4 = [float(row["gpu4_used_mib"]) / 1024 for row in rows]
    gpu5 = [float(row["gpu5_used_mib"]) / 1024 for row in rows]
    image = Image.new("RGB", (1500, 1080), base.LIGHT)
    draw = base.add_header(image, "Two-GPU clean GRPO Control: resource trajectory", "One-minute observer samples; GPU memory is allocated GiB and host series is available RAM.")
    base.line_panel(draw, (45, 145, 1455, 575), xs, [("GPU4", gpu4, "#D97706", 4), ("GPU5", gpu5, "#0F766E", 4)], 0, max(80.0, max(gpu4 + gpu5) * 1.05), "GPU memory")
    base.line_panel(draw, (45, 620, 1455, 1040), xs, [("Host available GiB", host, "#374151", 4)], 0, max(host) * 1.05, "Host available memory")
    image.save(output, optimize=True)
    return {
        "observer_rows": len(rows),
        "gpu4_peak_gib": max(gpu4),
        "gpu5_peak_gib": max(gpu5),
        "host_available_min_gib": min(host),
        "host_available_final_gib": host[-1],
    }


def main() -> None:
    if OUT.exists() or ZIP_PATH.exists():
        raise FileExistsError(f"refusing to overwrite {OUT} or {ZIP_PATH}")
    for directory in (OUT / "runtime", OUT / "tensorboard", OUT / "derived", OUT / "figures"):
        directory.mkdir(parents=True, exist_ok=True)
    for source in (RAW / "runtime").glob("*"):
        if source.is_file():
            shutil.copy2(source, OUT / "runtime" / source.name)
    for source in (RAW / "tensorboard").glob("*"):
        if source.is_file():
            shutil.copy2(source, OUT / "tensorboard" / source.name)

    base = load_base()
    rows = base.parse_grpo(RAW / "runtime" / "driver.log")
    if int(rows[-1]["step"]) != 96:
        raise RuntimeError(f"expected Step96, got {rows[-1]['step']}")
    with (OUT / "derived" / "core_step_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evals = [{"step": int(row["step"]), "fixed32_success_rate": float(row["eval_success"]), "successes": round(float(row["eval_success"]) * 32)} for row in rows if row["eval_success"] is not None]
    with (OUT / "derived" / "fixed32_eval_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evals[0]))
        writer.writeheader()
        writer.writerows(evals)

    render_success(base, rows, OUT / "figures" / "success_raw_ma5_ma10_fixed32.png")
    render_optimizer(base, rows, OUT / "figures" / "optimizer_metrics.png")
    resources = render_resources(base, RAW / "runtime" / "resource.csv", OUT / "figures" / "resources_gpu_host.png")
    success = [float(row["train_success"]) for row in rows]
    summary = {
        "source_run": "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-control-formal100-2gpu64x4-b1024-fixed32-eval5-phys45-v2",
        "last_complete_step": 96,
        "termination": "user-authorized exact stop to replace Control with GRPO-DVAC Action-Adv [0,2]",
        "train_success_mean_pct": statistics.mean(success) * 100,
        "train_success_latest_pct": success[-1] * 100,
        "train_success_ma5_pct": statistics.mean(success[-5:]) * 100,
        "train_success_ma10_pct": statistics.mean(success[-10:]) * 100,
        "fixed32_successes": sum(int(row["successes"]) for row in evals),
        "fixed32_episodes": 32 * len(evals),
        "fixed32_points": evals,
        "latest_core_metrics": rows[-1],
        "resources": resources,
        "excluded": ["checkpoints", "videos", "RoboTwin bulk data", "Ray session logs", "per-action tensors"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 深圳两卡 clean GRPO Control Step 96 轻量收尾包\n\n"
        "Control 按用户授权在完整 Step 96 后精确停止，以原 GPU4/5 启动 GRPO-DVAC Action-Adv。\n\n"
        "包含 driver/resource 日志、resolved/contract/parity/command、TensorBoard event/config、逐步核心指标、fixed32评估点与三张图。"
        "不含 checkpoint、视频、RoboTwin 大数据、Ray 全量日志或逐动作 tensor。\n",
        encoding="utf-8",
    )
    manifest = [{"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size} for path in sorted(OUT.rglob("*")) if path.is_file()]
    (OUT / "file_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"{NAME}/{path.relative_to(OUT)}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    print(json.dumps({"zip": str(ZIP_PATH), "bytes": ZIP_PATH.stat().st_size, "files": len(manifest) + 1, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
