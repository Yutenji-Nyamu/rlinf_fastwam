from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CONTROL_LOG = ROOT / "tmp/rlt_control_formal480_v2_metrics.log"
METHOD_LOG = ROOT / "tmp/rlt_success_bc_formal480_v2_metrics.log"
RESOURCE_CSV = ROOT / "tmp/rlt_success_bc_dual_formal480_v2_resources.csv"
OUT = (
    ROOT
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence"
    / "rlt_success_bc_formal_live_g111_20260826"
)

TEAL = "#00796B"
ORANGE = "#E66101"
BLUE = "#3B82F6"
GRAY = "#667085"
LIGHT_GRID = "#D9E1EA"
TEXT = "#172033"
BG = "#F7F9FC"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size=size)


def metric_blocks(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(r"Global Step:\s*(\d+)/480", text))
    blocks = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((int(match.group(1)), text[match.start() : end]))
    return blocks


def section(block: str, start: str, end: str | None) -> str:
    begin = block.find(start)
    if begin < 0:
        return ""
    finish = block.find(end, begin + len(start)) if end else len(block)
    if finish < 0:
        finish = len(block)
    return block[begin:finish]


def first_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_metrics(path: Path) -> list[dict[str, float]]:
    rows = []
    for step, block in metric_blocks(path):
        env = section(block, " Environment ", " Evaluation ")
        if not env:
            env = section(block, " Environment ", " Replay Buffer ")
        evaluation = section(block, " Evaluation ", " Replay Buffer ")
        train_success = first_float(r"success_once=([-+0-9.eE]+)", env)
        if train_success is None:
            continue
        row: dict[str, float] = {"step": float(step), "train_success": train_success}
        eval_success = first_float(r"success_once=([-+0-9.eE]+)", evaluation)
        if eval_success is not None:
            row["eval_success"] = eval_success
        for name in ("weight_p05", "weight_mean", "weight_p95", "weight_ess_ratio"):
            value = first_float(rf"actor/rlt_dvac/{name}=([-+0-9.eE]+)", block)
            if value is not None:
                row[name] = value
        rows.append(row)
    return rows


def trailing(values: list[float], window: int) -> list[float]:
    return [sum(values[max(0, i - window + 1) : i + 1]) / min(window, i + 1) for i in range(len(values))]


def parse_resources(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "time": datetime.fromisoformat(row["timestamp"]),
                        "ram": float(row["memory_current"]) / 2**30,
                        "gpu0": float(row["gpu0_mib"]) / 1024,
                        "gpu1": float(row["gpu1_mib"]) / 1024,
                    }
                )
            except (KeyError, ValueError):
                continue
    start = rows[0]["time"]
    for row in rows:
        row["hours"] = (row["time"] - start).total_seconds() / 3600
    return rows


def xy(x: float, y: float, box: tuple[int, int, int, int], xr: tuple[float, float], yr: tuple[float, float]) -> tuple[int, int]:
    left, top, right, bottom = box
    px = left + (x - xr[0]) / (xr[1] - xr[0]) * (right - left)
    py = bottom - (y - yr[0]) / (yr[1] - yr[0]) * (bottom - top)
    return round(px), round(py)


def axes(draw: ImageDraw.ImageDraw, box, xr, yr, xticks, yticks, xlabel, ylabel):
    left, top, right, bottom = box
    for value in yticks:
        _, py = xy(xr[0], value, box, xr, yr)
        draw.line((left, py, right, py), fill=LIGHT_GRID, width=1)
        draw.text((left - 12, py), f"{value:g}", fill=GRAY, font=font(18), anchor="rm")
    for value in xticks:
        px, _ = xy(value, yr[0], box, xr, yr)
        draw.line((px, top, px, bottom), fill="#EEF2F6", width=1)
        draw.text((px, bottom + 9), f"{value:g}", fill=GRAY, font=font(18), anchor="ma")
    draw.line((left, top, left, bottom), fill=TEXT, width=2)
    draw.line((left, bottom, right, bottom), fill=TEXT, width=2)
    draw.text(((left + right) // 2, bottom + 42), xlabel, fill=GRAY, font=font(18), anchor="ma")
    draw.text((left, top - 12), ylabel, fill=GRAY, font=font(17), anchor="ls")


def line(draw, xs, ys, box, xr, yr, color, width=3):
    points = [xy(x, y, box, xr, yr) for x, y in zip(xs, ys)]
    if len(points) >= 2:
        draw.line(points, fill=color, width=width, joint="curve")


def marker(draw, point, color, shape="circle", radius=6):
    x, y = point
    if shape == "square":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=2)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=WHITE, width=2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    control = parse_metrics(CONTROL_LOG)
    method = parse_metrics(METHOD_LOG)
    resources = parse_resources(RESOURCE_CSV)
    common = min(int(control[-1]["step"]), int(method[-1]["step"]))
    control = [row for row in control if row["step"] <= common]
    method = [row for row in method if row["step"] <= common]

    cs = [100 * row["train_success"] for row in control]
    ms = [100 * row["train_success"] for row in method]
    c5, m5 = trailing(cs, 5), trailing(ms, 5)
    c10, m10 = trailing(cs, 10), trailing(ms, 10)
    steps = [row["step"] for row in control]

    summary = {
        "common_complete_step": common,
        "control": {
            "latest_train_success_pct": cs[-1],
            "mean_train_success_pct": sum(cs) / len(cs),
            "latest5_train_success_pct": sum(cs[-5:]) / 5,
            "latest10_train_success_pct": sum(cs[-10:]) / 10,
            "fixed_eval_pct": {str(int(r["step"])): 100 * r["eval_success"] for r in control if "eval_success" in r},
        },
        "method": {
            "latest_train_success_pct": ms[-1],
            "mean_train_success_pct": sum(ms) / len(ms),
            "latest5_train_success_pct": sum(ms[-5:]) / 5,
            "latest10_train_success_pct": sum(ms[-10:]) / 10,
            "fixed_eval_pct": {str(int(r["step"])): 100 * r["eval_success"] for r in method if "eval_success" in r},
        },
        "method_latest_weights": {
            key: method[-1].get(key) for key in ("weight_p05", "weight_mean", "weight_p95", "weight_ess_ratio")
        },
        "resources": {
            "current_ram_gib": resources[-1]["ram"],
            "peak_ram_gib": max(r["ram"] for r in resources),
            "current_gpu0_gib": resources[-1]["gpu0"],
            "current_gpu1_gib": resources[-1]["gpu1"],
            "peak_gpu0_gib": max(r["gpu0"] for r in resources),
            "peak_gpu1_gib": max(r["gpu1"] for r in resources),
        },
    }
    summary["delta_method_minus_control"] = {
        "mean_train_success_pp": summary["method"]["mean_train_success_pct"] - summary["control"]["mean_train_success_pct"],
        "latest5_train_success_pp": summary["method"]["latest5_train_success_pct"] - summary["control"]["latest5_train_success_pct"],
        "latest10_train_success_pp": summary["method"]["latest10_train_success_pct"] - summary["control"]["latest10_train_success_pct"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "curves.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "control_raw_pct", "control_ma5_pct", "control_ma10_pct", "method_raw_pct", "method_ma5_pct", "method_ma10_pct"])
        writer.writerows(zip(steps, cs, c5, c10, ms, m5, m10))

    image = Image.new("RGB", (1600, 1130), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 38), f"Single-GPU RLT control vs success-episode DVAC-BC — through Step {common}", fill=TEXT, font=font(34, True))
    draw.text((70, 87), "Training rollout success is noisy (8 episodes/step); fixed eval uses 20 episodes every 25 steps.", fill=GRAY, font=font(20))

    # Panel A: success.
    draw.rounded_rectangle((45, 130, 1555, 590), radius=18, fill=WHITE, outline="#D7DFEA", width=2)
    draw.text((80, 152), "A  Success: raw + trailing 5-step mean; fixed-20 eval markers", fill=TEXT, font=font(25, True))
    box = (125, 205, 1515, 520)
    axes(draw, box, (1, common), (0, 100), [1, 25, 50, 75, 100, common], [0, 25, 50, 75, 100], "Global step", "Success (%)")
    line(draw, steps, cs, box, (1, common), (0, 100), "#A7D3CE", 2)
    line(draw, steps, ms, box, (1, common), (0, 100), "#F5C59B", 2)
    line(draw, steps, c5, box, (1, common), (0, 100), TEAL, 5)
    line(draw, steps, m5, box, (1, common), (0, 100), ORANGE, 5)
    for row in control:
        if "eval_success" in row:
            marker(draw, xy(row["step"], 100 * row["eval_success"], box, (1, common), (0, 100)), TEAL, "circle", 7)
    for row in method:
        if "eval_success" in row:
            marker(draw, xy(row["step"], 100 * row["eval_success"], box, (1, common), (0, 100)), ORANGE, "square", 7)
    draw.line((1060, 167, 1100, 167), fill=TEAL, width=5)
    draw.text((1110, 167), "Control MA5 / eval ○", fill=TEAL, font=font(18), anchor="lm")
    draw.line((1300, 167, 1340, 167), fill=ORANGE, width=5)
    draw.text((1350, 167), "DVAC-BC MA5 / eval □", fill=ORANGE, font=font(18), anchor="lm")
    note = f"Latest MA5: control {c5[-1]:.1f}% | DVAC-BC {m5[-1]:.1f}% | delta {m5[-1]-c5[-1]:+.1f} pp"
    draw.text((125, 552), note, fill=TEXT, font=font(20, True))

    # Panel B: method weights.
    draw.rounded_rectangle((45, 615, 980, 1075), radius=18, fill=WHITE, outline="#D7DFEA", width=2)
    draw.text((80, 638), "B  DVAC-BC per-action weight distribution", fill=TEXT, font=font(25, True))
    weight_rows = [r for r in method if all(k in r for k in ("weight_p05", "weight_mean", "weight_p95", "weight_ess_ratio"))]
    wx = [r["step"] for r in weight_rows]
    wp05 = [r["weight_p05"] for r in weight_rows]
    wmean = [r["weight_mean"] for r in weight_rows]
    wp95 = [r["weight_p95"] for r in weight_rows]
    wess = [r["weight_ess_ratio"] for r in weight_rows]
    wbox = (125, 700, 940, 1005)
    axes(draw, wbox, (min(wx), max(wx)), (0.6, 1.4), [min(wx), 75, 100, max(wx)], [0.6, 0.8, 1.0, 1.2, 1.4], "Global step", "Weight / ESS")
    band = [xy(x, y, wbox, (min(wx), max(wx)), (0.6, 1.4)) for x, y in zip(wx, wp05)]
    band += [xy(x, y, wbox, (min(wx), max(wx)), (0.6, 1.4)) for x, y in reversed(list(zip(wx, wp95)))]
    draw.polygon(band, fill="#FBE1CA")
    line(draw, wx, wp05, wbox, (min(wx), max(wx)), (0.6, 1.4), ORANGE, 2)
    line(draw, wx, wp95, wbox, (min(wx), max(wx)), (0.6, 1.4), ORANGE, 2)
    line(draw, wx, wmean, wbox, (min(wx), max(wx)), (0.6, 1.4), TEXT, 3)
    line(draw, wx, wess, wbox, (min(wx), max(wx)), (0.6, 1.4), TEAL, 3)
    draw.text((125, 1035), f"Latest p05 / mean / p95 = {wp05[-1]:.3f} / {wmean[-1]:.3f} / {wp95[-1]:.3f}; ESS = {wess[-1]:.3f}", fill=TEXT, font=font(19, True))

    # Panel C: resources.
    draw.rounded_rectangle((1005, 615, 1555, 1075), radius=18, fill=WHITE, outline="#D7DFEA", width=2)
    draw.text((1040, 638), "C  Shared host resources", fill=TEXT, font=font(25, True))
    rbox = (1085, 700, 1515, 1005)
    xmax = resources[-1]["hours"]
    axes(draw, rbox, (0, xmax), (0, 240), [0, round(xmax / 2, 1), round(xmax, 1)], [0, 60, 120, 180, 240], "Hours", "RAM (GiB)")
    stride = max(1, len(resources) // 900)
    rr = resources[::stride]
    smooth_window = max(1, 60 // stride)
    smooth_ram = trailing([r["ram"] for r in rr], smooth_window)
    smooth_gpu0 = trailing([r["gpu0"] for r in rr], smooth_window)
    smooth_gpu1 = trailing([r["gpu1"] for r in rr], smooth_window)
    line(draw, [r["hours"] for r in rr], smooth_ram, rbox, (0, xmax), (0, 240), GRAY, 4)
    # GPU memory uses a right-side 0--40 GiB scale.
    line(draw, [r["hours"] for r in rr], [value * 6 for value in smooth_gpu0], rbox, (0, xmax), (0, 240), TEAL, 3)
    line(draw, [r["hours"] for r in rr], [value * 6 for value in smooth_gpu1], rbox, (0, xmax), (0, 240), ORANGE, 3)
    draw.text((1515, 680), "GPU scale 0–40 GiB", fill=GRAY, font=font(16), anchor="ra")
    draw.text((1085, 1035), f"Current RAM {resources[-1]['ram']:.1f} GiB; peaks RAM/GPU0/GPU1 {max(r['ram'] for r in resources):.1f}/{max(r['gpu0'] for r in resources):.1f}/{max(r['gpu1'] for r in resources):.1f} GiB", fill=TEXT, font=font(16, True))

    image.save(OUT / "RLT_CONTROL_VS_SUCCESS_BC_G111.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
