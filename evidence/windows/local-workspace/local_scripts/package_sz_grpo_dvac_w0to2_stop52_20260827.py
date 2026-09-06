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
SNAPSHOT = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence" / "dual-2gpu-comparison-live-20260827"
RAW = SNAPSHOT / "raw" / "dvac_w0to2"
NAME = "shenzhen_grpo_dvac_w0to2_2gpu_stopped_step52_light_evidence_20260827"
OUT = SNAPSHOT.parent / NAME
ZIP_PATH = ROOT / "exports" / f"{NAME}.zip"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"


def load_base():
    spec = importlib.util.spec_from_file_location("plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_resource(rows: list[dict[str, str]], output: Path) -> dict[str, float]:
    base = load_base()
    valid = [row for row in rows if row.get("timestamp") and row.get("host_mem_available_kib")]
    xs = list(range(len(valid)))
    host = [float(row["host_mem_available_kib"]) / 1024 / 1024 for row in valid]
    gpu6 = [float(row["gpu6_used_mib"]) / 1024 for row in valid if row.get("gpu6_used_mib")]
    gpu7 = [float(row["gpu7_used_mib"]) / 1024 for row in valid if row.get("gpu7_used_mib")]
    common = min(len(xs), len(gpu6), len(gpu7))
    xs, gpu6, gpu7, host = xs[:common], gpu6[:common], gpu7[:common], host[:common]
    image = Image.new("RGB", (1440, 1040), base.LIGHT)
    draw = base.add_header(
        image,
        "Stopped Shenzhen GRPO-DVAC [0,2]: resource trajectory",
        "One-minute observer telemetry; host memory is available RAM, GPU values are allocated memory.",
    )
    base.line_panel(
        draw,
        (45, 150, 1395, 560),
        xs,
        [("GPU6 used GiB", gpu6, "#D97706", 5), ("GPU7 used GiB", gpu7, "#0F766E", 5)],
        0,
        max(80.0, max(gpu6 + gpu7) * 1.05),
        "GPU memory",
    )
    base.line_panel(
        draw,
        (45, 600, 1395, 1010),
        xs,
        [("Host available GiB", host, "#0F766E", 5)],
        0,
        max(host) * 1.05,
        "Host available memory",
    )
    image.save(output, optimize=True)
    return {
        "observer_rows": len(valid),
        "gpu6_peak_gib": max(gpu6),
        "gpu7_peak_gib": max(gpu7),
        "host_available_min_gib": min(host),
        "host_available_final_gib": host[-1],
    }


def main() -> None:
    if OUT.exists() or ZIP_PATH.exists():
        raise FileExistsError(f"refusing to overwrite {OUT} or {ZIP_PATH}")
    (OUT / "runtime").mkdir(parents=True)
    (OUT / "tensorboard").mkdir()
    (OUT / "figures").mkdir()
    (OUT / "derived").mkdir()

    for source in sorted((RAW / "runtime").glob("*")):
        if source.is_file():
            shutil.copy2(source, OUT / "runtime" / source.name)
    for source in sorted((RAW / "tensorboard").glob("*")):
        if source.is_file():
            shutil.copy2(source, OUT / "tensorboard" / source.name)

    for source, name in (
        (SNAPSHOT / "01_current_2gpu_control_vs_dvac_raw_roll5_roll10.png", "success_raw_roll5_roll10_fixed32.png"),
        (SNAPSHOT / "current_2gpu_success_curves.csv", "paired_success_through_stop.csv"),
        (SNAPSHOT / "comparison_summary.json", "paired_comparison_summary.json"),
    ):
        target_dir = OUT / ("figures" if source.suffix == ".png" else "derived")
        shutil.copy2(source, target_dir / name)

    plot_base = load_base()
    core_rows = plot_base.parse_grpo(RAW / "runtime" / "driver.log")
    with (OUT / "derived" / "core_step_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(core_rows[0]))
        writer.writeheader()
        writer.writerows(core_rows)

    with (RAW / "runtime" / "resource.csv").open(newline="", encoding="utf-8") as handle:
        resource_rows = list(csv.DictReader(handle))
    resource_summary = render_resource(resource_rows, OUT / "figures" / "resource_gpu_host.png")

    paired = list(csv.DictReader((SNAPSHOT / "current_2gpu_success_curves.csv").open(newline="", encoding="utf-8")))
    stop_marker = (RAW / "runtime" / "stopped_by_user_for_prism.txt").read_text(encoding="utf-8").splitlines()
    summary = {
        "source_run": "/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/dvac-global-z-w0to2-formal100-2gpu64x4-b1024-fixed32-eval5-phys67-v2",
        "last_complete_step": 52,
        "termination": "user-authorized exact stop to replace the run with Prism-style DVAC-Rank-RLOO",
        "stop_marker": stop_marker,
        "train_success_mean_pct": statistics.mean(float(row["dvac_raw"]) for row in paired) * 100,
        "train_success_latest_pct": float(paired[-1]["dvac_raw"]) * 100,
        "train_success_roll5_pct": float(paired[-1]["dvac_roll5"]) * 100,
        "train_success_roll10_pct": float(paired[-1]["dvac_roll10"]) * 100,
        "fixed32_successes": 296,
        "fixed32_episodes": 320,
        "core_metric_rows": len(core_rows),
        "latest_core_metrics": core_rows[-1],
        "resource": resource_summary,
        "excluded": ["checkpoints", "videos", "RoboTwin bulk data", "Ray session logs", "per-action tensors"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# 深圳两卡 GRPO-DVAC `[0,2]` Step 52 轻量收尾包\n\n"
        "该正式实验按用户授权在完整 Step 52 后停止，以释放 GPU6/7 给 Prism-style DVAC-Rank-RLOO。\n\n"
        "包含完整 driver/resource 日志、resolved/contract/parity/command、TensorBoard event/config与展开CSV、"
        "Control-DVAC成功率对照图和资源图。排除checkpoint、视频、RoboTwin大数据、Ray全量日志和逐动作tensor。\n",
        encoding="utf-8",
    )

    manifest = []
    for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
        manifest.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size})
    (OUT / "file_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(p for p in OUT.rglob("*") if p.is_file()):
            archive.write(path, arcname=f"{NAME}/{path.relative_to(OUT)}")
    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure at {bad}")
    print(json.dumps({"zip": str(ZIP_PATH), "bytes": ZIP_PATH.stat().st_size, "files": len(manifest) + 1}))


if __name__ == "__main__":
    main()
