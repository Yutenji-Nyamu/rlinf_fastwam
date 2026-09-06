"""Render a compact operational dashboard for current Sidney and Fast-WAM jobs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


COLORS = {"sidney_pill": "#007C78", "fastwam": "#E76F00"}
LABELS = {"sidney_pill": "Sidney pi0.5 · pill", "fastwam": "Fast-WAM · stapler"}
BG, PANEL, GRID, TEXT, MUTED = "#F6F8FB", "#FFFFFF", "#DCE3EC", "#172033", "#667085"


def font(size: int, bold: bool = False):
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / ("seguisb.ttf" if bold else "segoeui.ttf")), size)


F_TITLE, F_PANEL, F_LEGEND, F_TEXT, F_SMALL = font(34, True), font(23, True), font(18, True), font(20), font(16)


def xy(points):
    return [(float(p["step"]), float(p["value"])) for p in points]


def rolling(points, window):
    values = [float(p["value"]) for p in points]
    return [(float(points[i]["step"]), sum(values[max(0, i-window+1):i+1]) / min(window, i+1)) for i in range(len(points))]


def draw_panel(draw, box, title, series, xmax, ylim=(0.0, 1.0), xlabel="completed global step", ylabel="success"):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#D5DCE7", width=2)
    draw.text((x0+28, y0+18), title, fill=TEXT, font=F_PANEL)
    left, top, right, bottom = x0+105, y0+74, x1-32, y1-54
    ymin, ymax = ylim
    xmin = 1.0
    def px(x): return left + (x-xmin)/max(1.0, xmax-xmin)*(right-left)
    def py(y): return bottom - (y-ymin)/max(1e-9, ymax-ymin)*(bottom-top)
    for i in range(6):
        y = ymin + (ymax-ymin)*i/5
        yy = py(y)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        draw.text((left-14, yy), f"{y:.0%}" if ymax <= 1.01 else f"{y:.1f}", fill=MUTED, font=F_SMALL, anchor="rm")
    for i in range(6):
        x = xmin + (xmax-xmin)*i/5
        xx = px(x)
        draw.line((xx, top, xx, bottom), fill="#EEF1F5", width=1)
        draw.text((xx, bottom+10), f"{x:.0f}", fill=MUTED, font=F_SMALL, anchor="ma")
    draw.line((left, bottom, right, bottom), fill="#7B8798", width=2)
    draw.line((left, top, left, bottom), fill="#7B8798", width=2)
    draw.text(((left+right)/2, y1-17), xlabel, fill=MUTED, font=F_SMALL, anchor="mm")
    draw.text((x0+20, (top+bottom)/2), ylabel, fill=MUTED, font=F_SMALL, anchor="mm")
    lx = x0 + 620
    for item in series:
        color = item["color"]
        draw.line((lx, y0+34, lx+38, y0+34), fill=color, width=5)
        draw.text((lx+47, y0+34), item["label"], fill=color, font=F_LEGEND, anchor="lm")
        lx += 350
        pts = [(px(x), py(y)) for x, y in item["points"] if math.isfinite(y)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=5, joint="curve")
        for xx, yy in pts:
            r = item.get("radius", 4)
            draw.ellipse((xx-r, yy-r, xx+r, yy+r), fill=color)


def resource_series(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return [], [], [], {}
    t0 = datetime.fromisoformat(rows[0]["timestamp"])
    mem_cols = [k for k in rows[0] if k.endswith("_used_mib")]
    util_cols = [k for k in rows[0] if k.endswith("_util_pct")]
    mem, util, ram = [], [], []
    for row in rows:
        if not row.get("timestamp"):
            continue
        hour = (datetime.fromisoformat(row["timestamp"]) - t0).total_seconds()/3600
        ms = [float(row[k]) for k in mem_cols if row.get(k)]
        us = [float(row[k]) for k in util_cols if row.get(k)]
        if ms: mem.append((hour, sum(ms)/len(ms)/1024))
        if us: util.append((hour, sum(us)/len(us)))
        if row.get("host_mem_available_kib"): ram.append((hour, float(row["host_mem_available_kib"])/1024**3))
    stats = {
        "peak_gpu_gib_per_card": max((v for _, v in mem), default=None),
        "median_gpu_util_pct": median([v for _, v in util]) if util else None,
        "latest_host_available_tib": ram[-1][1] if ram else None,
    }
    return mem, util, ram, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    args = parser.parse_args()
    src = args.src
    out = src / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((src / "metrics.json").read_text(encoding="utf-8"))
    xmax = max(p["step"] for run in data.values() for p in run["series"].get("env/success_once", []))

    image = Image.new("RGB", (1600, 1570), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 28), "Current RL training snapshot", fill=TEXT, font=F_TITLE)
    pill_note = "Sidney pill is still in its first rollout, so it has no completed-step metric yet. " if not data["sidney_pill"]["series"].get("env/success_once") else ""
    draw.text((48, 80), pill_note + "Operational comparison only: different models and tasks.", fill=MUTED, font=F_TEXT)
    boxes = [(38,125,1562,465), (38,488,1562,828), (38,851,1562,1191), (38,1214,1562,1548)]
    for box, title, window in zip(boxes[:3], ("Per-step rollout success", "Trailing 5-step mean", "Trailing 10-step mean"), (1,5,10)):
        series=[]
        for key in ("sidney_pill", "fastwam"):
            points=data[key]["series"].get("env/success_once", [])
            series.append({"label":LABELS[key],"color":COLORS[key],"points":xy(points) if window==1 else rolling(points,window)})
        draw_panel(draw,box,title,series,xmax)
    eval_series=[]
    for key in ("sidney_pill", "fastwam"):
        eval_series.append({"label":LABELS[key]+" fixed-32","color":COLORS[key],"points":xy(data[key]["series"].get("eval/success_once", [])),"radius":7})
    draw_panel(draw,boxes[3],"Fixed-32 evaluation",eval_series,xmax)
    image.save(out / "01_current_success_ma5_ma10_fixed32.png", quality=95)

    summary={}
    for key,run in data.items():
        train=run["series"].get("env/success_once",[])
        evals=run["series"].get("eval/success_once",[])
        times=[float(x["value"]) for x in run["series"].get("time/step",[])]
        _,_,_,stats=resource_series(src/"raw"/key/"resource.csv")
        summary[key]={
            "completed_step": train[-1]["step"] if train else None,
            "latest_success": train[-1]["value"] if train else None,
            "ma5": rolling(train,5)[-1][1] if train else None,
            "ma10": rolling(train,10)[-1][1] if train else None,
            "latest_eval": evals[-1] if evals else None,
            "median_last5_step_seconds": median(times[-5:]) if times else None,
            **stats,
        }
    (src/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
