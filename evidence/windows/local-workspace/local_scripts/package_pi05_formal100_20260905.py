"""Build a small, offline closeout bundle from an authenticated read-only snapshot.

This script only reads/writes local evidence. It never connects to the server.
The input must establish all 100 completed steps, exit 0 and checkpoint 100.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import zipfile

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/rlinf-shenzhen-multitask-pi05/evidence"
DEFAULT_INPUT = EVIDENCE / "pi05_formal100_closeout_20260905.json"
DEFAULT_OUTPUT = EVIDENCE / "formal100-summary-20260905"
TEAL, ORANGE, INK, MUTED = "#087f8c", "#c65d08", "#162b3d", "#64748b"
TZ = dt.timezone(dt.timedelta(hours=8))


def write_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def cst(timestamp):
    return dt.datetime.fromtimestamp(float(timestamp), TZ).isoformat()


def clean_scalars(raw):
    result = {}
    for tag, events in raw.items():
        points = {}
        for point in events:
            step, value = int(point["step"]), float(point["value"])
            if 1 <= step <= 100:
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite metric: {tag} step {step}")
                points[step] = {"step": step, "value": value,
                                "wall_time": float(point.get("wall_time", 0))}
        if points:
            result[tag] = [points[step] for step in sorted(points)]
    return result


def render_png(path, title, subtitle, panels, footer):
    """Draw simple quantitative line charts with the installed Pillow runtime."""
    width, panel_height = 1200, 430
    height = 170 + len(panels)*panel_height + 80
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_file = "C:/Windows/Fonts/msyh.ttc"
    font = lambda size: ImageFont.truetype(font_file, size)
    draw.text((64, 30), title, fill=INK, font=font(35))
    draw.text((64, 87), subtitle, fill=MUTED, font=font(22))
    for index, spec in enumerate(panels):
        y = 158 + index*panel_height
        left, right, top, bottom = 106, 1140, y+91, y+panel_height-77
        draw.text((64, y), spec["title"]+(f"（{spec['unit']}）" if spec.get("unit") else ""), fill=INK, font=font(28))
        lx = left
        for i, series in enumerate(spec["series"]):
            color = series["color"]
            draw.line((lx, y+57, lx+38, y+57), fill=color, width=4)
            draw.text((lx+48, y+40), series["name"], fill=MUTED, font=font(21))
            lx += int(draw.textlength(series["name"], font=font(21))) + 90
        values = [point[1] for series in spec["series"] for point in series["points"]]
        ymin, ymax = spec.get("ymin", min(0, min(values))), spec.get("ymax", max(values)*1.12)
        if ymax <= ymin:
            ymax = ymin+1
        xmax = spec["xmax"]
        fx = lambda x: left+x/xmax*(right-left)
        fy = lambda v: bottom-(v-ymin)/(ymax-ymin)*(bottom-top)
        def fmt(v):
            return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}" if abs(v) >= .01 else "0" if v == 0 else f"{v:.1e}"
        for i in range(5):
            value = ymin+(ymax-ymin)*i/4
            py = fy(value)
            draw.line((left, py, right, py), fill="#e2e8f0", width=2)
            draw.text((left-15, py), fmt(value), fill=MUTED, font=font(21), anchor="rm")
        for i in range(6):
            value = xmax*i/5
            draw.text((fx(value), bottom+27), fmt(value), fill=MUTED, font=font(21), anchor="mm")
        draw.text(((left+right)/2, bottom+61), spec["xlabel"], fill=MUTED, font=font(22), anchor="mm")
        for i, series in enumerate(spec["series"]):
            points = [(fx(x), fy(v)) for x, v in series["points"]]
            color = series["color"]
            raw_train = spec["id"] == "success" and i == 0
            if raw_train:
                color = "#add6da"
            if series.get("dash"):
                for j in range(0, len(points)-1, 16):
                    segment = points[j:j+9]
                    if len(segment) > 1:
                        draw.line(segment, fill=color, width=3)
            else:
                draw.line(points, fill=color, width=2 if raw_train else 4)
            if spec["id"] == "success" and i == 2:
                for px, py in points:
                    draw.ellipse((px-6, py-6, px+6, py+6), fill="white", outline=color, width=3)
    draw.text((64, height-49), footer, fill=MUTED, font=font(21))
    image.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", action="store_true", help="Render clearly labelled incomplete preview PNGs only; no ZIP")
    args = parser.parse_args()
    d = json.loads(args.input.read_text(encoding="utf-8-sig"))
    scalars = clean_scalars(d["scalars"])
    train = scalars["env/success_once"]
    evaluation = scalars["eval/success_once"]
    if args.preview:
        preview = EVIDENCE / "formal100-preview-20260905"
        preview.mkdir(exist_ok=True)
        latest = train[-1]["step"]
        success = {"id": "success", "title": "成功率 · 预览", "unit": "%", "xlabel": "训练轮数", "xmax": 100,
                   "ymin": 0, "ymax": 100, "series": [
                       {"name": "训练 / 256条", "color": TEAL, "points": [[p["step"], p["value"]*100] for p in train]},
                       {"name": "训练 MA10", "color": TEAL, "points": [[train[i]["step"], statistics.mean(p["value"] for p in train[i-9:i+1])*100] for i in range(9, len(train))]},
                       {"name": "固定评估 / 32条", "color": ORANGE, "points": [[p["step"], p["value"]*100] for p in evaluation]}]}
        render_png(preview / "01_success_preview.png", f"π0.5 · 未结束预览 · 完整{latest}/100轮", str(d.get("time")), [success], "这是渲染预览；未确认100轮/exit0，不生成最终产物包。")
        panels = [{"id": tag, "title": label, "unit": "", "xlabel": "训练轮数", "xmax": 100,
                   "series": [{"name": label, "color": color, "points": [[p["step"], p["value"]] for p in scalars[tag]]}]}
                  for tag, label, color in [("train/actor/approx_kl", "近似 KL", TEAL), ("train/actor/grad_norm", "梯度范数", ORANGE)]]
        render_png(preview / "02_optimization_preview.png", f"π0.5 · 未结束预览 · 完整{latest}/100轮", str(d.get("time")), panels, "原始日志统计口径；预览文件不属于最终ZIP。")
        print(json.dumps({"preview": str(preview), "completed_steps": latest, "zip_created": False}, ensure_ascii=False))
        return
    state = d.get("state", {})
    checkpoint_files = d.get("checkpoint_files", {}).get("100", [])
    if str(state.get("exit_code.txt", state.get("exit_code"))) != "0":
        raise SystemExit("Closeout pending: no verified exit 0; no final package created.")
    if [p["step"] for p in train] != list(range(1, 101)):
        raise SystemExit("Closeout pending: train metrics do not cover steps 1-100.")
    if [p["step"] for p in evaluation] != list(range(5, 101, 5)):
        raise SystemExit("Closeout pending: fixed evaluation metrics incomplete.")
    if not checkpoint_files or not all(int(p["bytes"]) > 0 for p in checkpoint_files):
        raise SystemExit("Closeout pending: checkpoint 100 inventory absent/empty.")
    out = args.output.resolve()
    out.relative_to(EVIDENCE.resolve())
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "BUILD_LEDGER.md"
    write_text(ledger, "# 轻量产物构建记录\n\n"
               f"- 输入：`{args.input.name}`；采集时间：{d.get('time')}。\n"
               "- 构建前确认：训练1—100齐全、20次fixed32评估齐全、exit=0、step100文件清单非空。\n"
               "- 仅本地文档、CSV、图像和ZIP生成；不连接服务器、不执行模型或恢复训练。\n")

    source_files = {}
    resources = d.get("resource", d.get("resources", []))
    if not isinstance(resources, list):
        resources = []
    for name, content in d.get("files", {}).items():
        safe = PurePosixPath(name.replace("\\", "/"))
        if safe.is_absolute() or ".." in safe.parts:
            raise ValueError(f"Unsafe local relative path: {name}")
        if not isinstance(content, str):
            raise ValueError(f"Expected text file: {name}")
        if safe.suffix.lower() not in {".txt", ".sh", ".yaml", ".yml", ".json", ".csv", ".log", ".md"}:
            raise ValueError(f"Non-text artifact rejected: {name}")
        if len(content.encode("utf-8")) > 2_000_000:
            raise ValueError(f"Text artifact too large for brief package: {name}")
        if "resource" in safe.name.lower() and safe.suffix == ".csv":
            resources = list(csv.DictReader(io.StringIO(content)))
            continue
        source_files[str(safe)] = content
        write_text(out / "source" / Path(*safe.parts), content)
    if not any("resolved" in name.lower() for name in source_files):
        raise ValueError("Missing resolved configuration")
    if not any("command" in name.lower() for name in source_files):
        raise ValueError("Missing exact launch command")
    if not resources:
        raise ValueError("Missing sampled resource history")

    long_rows = [{"tag": tag, **point, "time_cst": cst(point["wall_time"]) if point["wall_time"] else ""}
                 for tag, points in sorted(scalars.items()) for point in points]
    write_csv(out / "metrics/all_scalars.csv", ["tag", "step", "value", "wall_time", "time_cst"], long_rows)
    for kind, include in [("training", lambda tag: not tag.startswith("eval/")),
                          ("evaluation", lambda tag: tag.startswith("eval/"))]:
        tags = [tag for tag in sorted(scalars) if include(tag)]
        wide = {}
        for tag in tags:
            for point in scalars[tag]:
                wide.setdefault(point["step"], {"step": point["step"]})[tag] = point["value"]
        for row in wide.values():
            success_tag = "env/success_once" if kind == "training" else "eval/success_once"
            n = 256 if kind == "training" else 32
            if success_tag in row:
                row["success_count"] = round(row[success_tag] * n)
                row["episode_count"] = n
        write_csv(out / f"metrics/{kind}.csv", ["step", "success_count", "episode_count"] + tags,
                  [wide[step] for step in sorted(wide)])
    resource_fields = list(dict.fromkeys(key for row in resources for key in row if key is not None))
    resources = [{key: value for key, value in row.items() if key is not None} for row in resources]
    write_csv(out / "metrics/resources.csv", resource_fields, resources)
    write_csv(out / "checkpoint100_inventory.csv", ["path", "bytes", "mtime"], checkpoint_files)
    write_text(out / "logs/errors.json", json.dumps(d.get("errors", {}), ensure_ascii=False, indent=2) + "\n")
    tail = d.get("driver_tail", [])
    write_text(out / "logs/driver_tail.log", tail if isinstance(tail, str) else "\n".join(tail) + "\n")
    write_text(out / "logs/completion_state.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")

    first10 = statistics.mean(p["value"] for p in train[:10])
    last10 = statistics.mean(p["value"] for p in train[-10:])
    best = max(evaluation, key=lambda p: p["value"])
    final = evaluation[-1]
    summary = {"snapshot_time": d.get("time"), "run": d["run"], "completed_steps": 100,
               "exit_code": 0, "train_final_success": train[-1]["value"],
               "train_first10_mean": first10, "train_last10_mean": last10,
               "eval_final": final, "eval_best_first_occurrence": best,
               "train_episodes": 25600, "optimizer_calls": 200, "fixed_eval_episodes": 640,
               "checkpoint_generations": 10, "scalar_tags": len(scalars),
               "resource_samples": len(resources), "source_json_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    write_text(out / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    chart_specs = []

    def metric_series(tag, label, color=TEAL, multiplier=1):
        return {"name": label, "color": color,
                "points": [[p["step"], p["value"] * multiplier] for p in scalars.get(tag, [])]}

    ma = [[train[i]["step"], statistics.mean(p["value"] for p in train[i-9:i+1]) * 100]
          for i in range(9, len(train))]
    success_series = [metric_series("env/success_once", "训练 / 256条", TEAL, 100),
                      {"name": "训练 MA10", "color": TEAL, "points": ma},
                      metric_series("eval/success_once", "固定评估 / 32条", ORANGE, 100)]
    chart_specs.append({"id": "success", "title": "成功率", "unit": "%", "xlabel": "训练轮数",
                        "xmax": 100, "ymin": 0, "ymax": 100, "series": success_series})
    render_png(out / "01_success.png", "π0.5 · move_pillbottle_pad · 100轮收尾",
               f"最终评估 {round(final['value']*32)}/32；最佳 step{best['step']}：{round(best['value']*32)}/32；训练 MA10 {last10:.2%}",
               [chart_specs[0]], f"训练首10轮 {first10:.2%} → 末10轮 {last10:.2%}；无同协议 step0 评估。")

    optimization = [("train/actor/approx_kl", "近似 KL", TEAL),
                    ("train/actor/grad_norm", "梯度范数", ORANGE)]
    missing = [tag for tag, _, _ in optimization if tag not in scalars]
    if missing:
        raise ValueError(f"Missing key optimizer metrics: {missing}")
    for tag, label, color in optimization:
        series = metric_series(tag, label, color)
        chart_specs.append({"id": tag, "title": label, "unit": "", "xlabel": "训练轮数",
                            "xmax": 100, "series": [series]})
    render_png(out / "02_optimization.png", "π0.5 · 优化指标", "训练日志原始统计口径 · 已完成1—100轮",
               chart_specs[1:3], "完整 ratio / clip / loss / LR 等见 CSV；单个统计量不等同学习效果。")
    for tag in ["train/actor/clip_fraction", "train/actor/ratio", "train/actor/lr", "time/step"]:
        if tag in scalars:
            chart_specs.append({"id": tag, "title": tag, "unit": "", "xlabel": "训练轮数",
                                "xmax": 100, "series": [metric_series(tag, tag)]})

    parsed_resources = []
    for row in resources:
        try:
            timestamp = dt.datetime.fromisoformat(row["timestamp"])
            parsed_resources.append((timestamp, row))
        except (KeyError, ValueError, TypeError):
            pass
    if not parsed_resources:
        raise ValueError("Resource history has no readable timestamps")
    t0 = parsed_resources[0][0]

    def resource_series(key, label, color, divisor):
        points = []
        for timestamp, row in parsed_resources:
            try:
                value = float(row[key]) / divisor
                if math.isfinite(value):
                    points.append([(timestamp-t0).total_seconds()/3600, value])
            except (KeyError, ValueError, TypeError):
                continue
        if not points:
            raise ValueError(f"No valid resource samples: {key}")
        return {"name": label, "color": color, "points": points}

    gpu_series = [resource_series("gpu4_used_mib", "GPU 4", TEAL, 1024),
                  resource_series("gpu5_used_mib", "GPU 5", ORANGE, 1024)]
    ram_series = [resource_series("host_mem_available_kib", "整机可用内存", TEAL, 2**30)]
    hours = (parsed_resources[-1][0]-t0).total_seconds()/3600
    for title, unit, series_list in [("两卡显存", "GiB", gpu_series),
                                    ("整机可用内存", "TiB", ram_series)]:
        chart_specs.append({"id": title, "title": title, "unit": unit, "xlabel": "运行小时",
                            "xmax": hours, "series": series_list})
    # Minute samples oscillate heavily by phase. A labelled 10-sample moving
    # mean keeps the small static figure readable; CSV and HTML keep raw data.
    gpu_png = {**chart_specs[-2], "title": "两卡显存 · 10次采样滑动均值", "series": []}
    for i, series in enumerate(gpu_series):
        points = series["points"]
        gpu_png["series"].append({**series, "dash": bool(i), "points": [
            [point[0], statistics.mean(v[1] for v in points[max(0,j-9):j+1])]
            for j, point in enumerate(points)]})
    render_png(out / "03_resources.png", "π0.5 · 资源记录", f"GPU4/5 · {len(resources)}次周期采样 · 整机内存包含其他任务影响",
               [gpu_png, chart_specs[-1]], "显存原始曲线与采样峰值请查离线HTML/CSV；周期采样未必捕获瞬时峰值。")

    template = Path(__file__).with_name("pi05_formal100_dashboard_template.html").read_text(encoding="utf-8")
    dashboard_data = {"summary": summary, "charts": chart_specs}
    write_text(out / "dashboard.html", template.replace("__DASHBOARD_DATA__", json.dumps(dashboard_data, ensure_ascii=False).replace("</", "<\\/")))
    config_names = [name for name in source_files if "resolved" in name.lower()]
    command_names = [name for name in source_files if "command" in name.lower()]
    readme = f"""# π0.5：100轮轻量实验产物

任务：`move_pillbottle_pad`；算法：GRPO；采集：{d.get('time')}。

- **已自然结束100轮，exit=0**；step100 checkpoint清单已确认，权重留在服务器。
- 最终训练：**{round(train[-1]['value']*256)}/256 = {train[-1]['value']:.2%}**；末10轮均值 **{last10:.2%}**。
- 固定评估：step100 **{round(final['value']*32)}/32 = {final['value']:.2%}**；最佳首次在step{best['step']}，**{round(best['value']*32)}/32 = {best['value']:.2%}**。
- 首10轮→末10轮训练均值：{first10:.2%} → {last10:.2%}。固定32条评估样本较少，仍有波动；没有同协议step0，不将首轮当SFT基线。
- 本阶段预算：25,600条训练轨迹、200次optimizer调用、640条fixed评估、10代checkpoint。
- 配置：64环境×4轮采样，G8，GB1024/MB32/update2，H50/C50/M10，noise0.5，horizon200；GPU4/5。

## 怎么看

1. 双击 **[dashboard.html](dashboard.html)**：离线交互图，可切换指标、查看具体数据，不需要网络。
2. 手机直接看 **[成功率](01_success.png)**、**[优化](02_optimization.png)**、**[资源](03_resources.png)** 三张PNG。
3. **[training.csv](metrics/training.csv)**：完整1—100轮；**[evaluation.csv](metrics/evaluation.csv)**：20次固定评估；**[all_scalars.csv](metrics/all_scalars.csv)**：全部{len(scalars)}项TensorBoard scalar导出；**[resources.csv](metrics/resources.csv)**：{len(resources)}次资源采样。
4. **[完整配置](source/{config_names[0]})**、**[原始启动命令](source/{command_names[0]})**；`source/`附源码锁等原始文本。
5. `logs/`保存关键driver尾日志、完成状态和错误计数；**[checkpoint清单](checkpoint100_inventory.csv)**仅列step100路径、大小、时间。

图中横轴为已完成训练轮数，不是optimizer调用数。训练每轮256条，固定评估每次32条；真实曲线从step1开始，不虚构step0。资源是运行期间的采样记录，显存峰值可能落在采样间隔内；整机内存包含其他任务影响。

这是 **1—100轮不可变收尾包**；随后接续到200的运行不混入本包。未包含模型/checkpoint本体、视频、数据集、完整Ray日志或TensorBoard二进制。

服务器原始运行目录：

```text
{d['run']}
```
"""
    write_text(out / "README.md", readme)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write("- 已导出全部scalar、train/eval宽表、原始资源CSV、完成状态、checkpoint清单和关键日志。\n"
                     "- 已生成3张PNG和无需网络的HTML；图像人工视检由执行代理完成后记录。\n")
    files = [p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"]
    manifest = {"input_sha256": summary["source_json_sha256"], "files": [
        {"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]}
    write_text(out / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    archive = out.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=out.name + "/" + path.relative_to(out).as_posix())
    if archive.stat().st_size > 5_000_000:
        raise ValueError(f"Archive exceeds 5 MB: {archive.stat().st_size}")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "files": len(manifest["files"])+1, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
