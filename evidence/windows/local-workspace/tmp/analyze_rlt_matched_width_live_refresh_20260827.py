from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/rlt_success_bc_matched_width_live_refresh_20260827"
CONTROL = OUT / "raw/control_metrics.log"
METHOD = OUT / "raw/method_metrics.log"


def font(size: int, bold: bool = False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def extract(text: str, pattern: str):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    markers = list(re.finditer(r"Global Step:\s*(\d+)/480", text))
    rows = []
    for index, marker in enumerate(markers):
        block = text[marker.start() : markers[index + 1].start() if index + 1 < len(markers) else len(text)]
        start = block.find(" Environment ")
        if start < 0:
            continue
        eval_start = block.find(" Evaluation ", start)
        replay_start = block.find(" Replay Buffer ", start)
        stop = eval_start if eval_start >= 0 else replay_start
        env = block[start : stop if stop >= 0 else len(block)]
        success = extract(env, r"success_once=([-+0-9.eE]+)")
        if success is None:
            continue
        row = {"step": int(marker.group(1)), "raw": success * 100.0, "eval": None}
        if eval_start >= 0:
            eval_block = block[eval_start : replay_start if replay_start >= 0 else len(block)]
            value = extract(eval_block, r"success_once=([-+0-9.eE]+)")
            if value is not None:
                row["eval"] = value * 100.0
        rows.append(row)
    by_step = {row["step"]: row for row in rows}
    rows = [by_step[step] for step in sorted(by_step)]
    for index, row in enumerate(rows):
        for window in (5, 10):
            values = [item["raw"] for item in rows[max(0, index - window + 1) : index + 1]]
            row[f"ma{window}"] = sum(values) / len(values)
    return rows


def stats(rows, end):
    values = [row["raw"] for row in rows if row["step"] <= end]
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "last5": sum(values[-5:]) / len(values[-5:]),
        "last10": sum(values[-10:]) / len(values[-10:]),
    }


def render(control, method, common_step):
    width, height = 1800, 720
    bg, ink, grid = "#F7F9FC", "#172033", "#D9E1EA"
    colors = {"control": "#00796B", "method": "#E66101"}
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    draw.text((55, 35), f"RLT control vs success-episode DVAC-BC (common Step {common_step})", fill=ink, font=font(36, True))
    draw.text((55, 84), "Training-rollout success; raw and trailing means. Fixed evaluation is reported separately.", fill="#52627A", font=font(20))
    draw.line((1280, 94, 1320, 94), fill=colors["control"], width=5)
    draw.text((1330, 94), "Control", fill="#475569", font=font(18), anchor="lm")
    draw.line((1450, 94, 1490, 94), fill=colors["method"], width=5)
    draw.text((1500, 94), "DVAC-BC", fill="#475569", font=font(18), anchor="lm")
    boxes = [(95, 170, 575, 640), (660, 170, 1140, 640), (1225, 170, 1705, 640)]
    specs = [("Per-step success", "raw"), ("Trailing 5-step mean", "ma5"), ("Trailing 10-step mean", "ma10")]

    def point(step, value, box):
        left, top, right, bottom = box
        x = left + (step - 1) / max(1, common_step - 1) * (right - left)
        y = bottom - value / 100.0 * (bottom - top)
        return round(x), round(y)

    for box, (title, key) in zip(boxes, specs):
        left, top, right, bottom = box
        draw.rounded_rectangle((left - 45, top - 45, right + 30, bottom + 35), radius=15, fill="#FFFFFF", outline="#D7E0EA", width=2)
        draw.text((left - 20, top - 34), title, fill=ink, font=font(24, True))
        for tick in (0, 25, 50, 75, 100):
            y = point(1, tick, box)[1]
            draw.line((left, y, right, y), fill=grid, width=1)
            draw.text((left - 10, y), str(tick), fill="#64748B", font=font(15), anchor="rm")
        for tick in range(100, common_step + 1, 100):
            x = point(tick, 0, box)[0]
            draw.text((x, bottom + 12), str(tick), fill="#64748B", font=font(15), anchor="ma")
        for name, rows in (("control", control), ("method", method)):
            pts = [point(row["step"], row[key], box) for row in rows if row["step"] <= common_step]
            if len(pts) > 1:
                draw.line(pts, fill=colors[name], width=4)
    path = OUT / f"RLT_CONTROL_VS_DVAC_BC_SUCCESS_THROUGH_G{common_step}.png"
    image.save(path)
    return path


def main():
    control = parse(CONTROL)
    method = parse(METHOD)
    common_step = min(control[-1]["step"], method[-1]["step"])
    by_control = {row["step"]: row for row in control}
    by_method = {row["step"]: row for row in method}
    with (OUT / "success_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["step", "control_raw", "control_ma5", "control_ma10", "method_raw", "method_ma5", "method_ma10"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in range(1, max(control[-1]["step"], method[-1]["step"]) + 1):
            c, m = by_control.get(step), by_method.get(step)
            writer.writerow({
                "step": step,
                "control_raw": "" if c is None else c["raw"],
                "control_ma5": "" if c is None else c["ma5"],
                "control_ma10": "" if c is None else c["ma10"],
                "method_raw": "" if m is None else m["raw"],
                "method_ma5": "" if m is None else m["ma5"],
                "method_ma10": "" if m is None else m["ma10"],
            })
    plot = render(control, method, common_step)
    summary = {
        "control_latest_step": control[-1]["step"],
        "method_latest_step": method[-1]["step"],
        "common_step": common_step,
        "through_common_step": {
            "control": stats(control, common_step),
            "method": stats(method, common_step),
        },
        "latest_fixed20": {
            "control": [[row["step"], row["eval"]] for row in control if row["eval"] is not None][-4:],
            "method": [[row["step"], row["eval"]] for row in method if row["eval"] is not None][-4:],
        },
        "plot": plot.name,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
