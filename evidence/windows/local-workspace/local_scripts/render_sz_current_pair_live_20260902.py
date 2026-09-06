"""Render a compact live dashboard without external plotting dependencies."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"C:\Users\86136\Documents\rl")
SRC = ROOT / "docs" / "rlinf-shenzhen-experiment-expansion" / "evidence" / "current-pair-live-20260902-1552"
OUT = SRC / "figures"
COLORS = {"pi05_control": "#00796B", "fastwam_control": "#E76F00"}
LABELS = {"pi05_control": "pi0.5 GRPO", "fastwam_control": "Fast-WAM GRPO"}
BG = "#F6F8FB"
PANEL = "#FFFFFF"
GRID = "#DCE3EC"
TEXT = "#172033"
MUTED = "#667085"


def font(size: int, bold: bool = False):
    name = "seguisb.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size)


F12 = font(22)
F10 = font(18)
F9 = font(16)
F_TITLE = font(34, True)
F_PANEL = font(23, True)
F_LEGEND = font(19, True)


def rolling(points: list[dict], window: int) -> list[tuple[float, float]]:
    vals = [float(p["value"]) for p in points]
    return [
        (float(points[i]["step"]), sum(vals[max(0, i - window + 1): i + 1]) / min(window, i + 1))
        for i in range(len(points))
    ]


def xy(points: list[dict]) -> list[tuple[float, float]]:
    return [(float(p["step"]), float(p["value"])) for p in points]


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def fmt(v: float) -> str:
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str,
          series: list[dict], xlim: tuple[float, float], ylim: tuple[float, float],
          ylabel: str, failure_x: float | None = None,
          xlabel: str = "completed global step") -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=PANEL, outline="#D5DCE7", width=2)
    draw.text((x0 + 28, y0 + 18), title, fill=TEXT, font=F_PANEL)
    left, top, right, bottom = x0 + 102, y0 + 76, x1 - 30, y1 - 58
    xmin, xmax = xlim
    ymin, ymax = ylim

    def px(x: float) -> float:
        return left + (x - xmin) / max(1e-9, xmax - xmin) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - ymin) / max(1e-9, ymax - ymin) * (bottom - top)

    for y in nice_ticks(ymin, ymax):
        yy = py(y)
        draw.line((left, yy, right, yy), fill=GRID, width=2)
        label = f"{y:.0%}" if ymax <= 1.01 and ymin >= 0 else fmt(y)
        draw.text((left - 16, yy), label, fill=MUTED, font=F9, anchor="rm")
    for x in nice_ticks(xmin, xmax, 5):
        xx = px(x)
        draw.line((xx, top, xx, bottom), fill="#EEF1F5", width=1)
        draw.text((xx, bottom + 11), f"{x:.0f}", fill=MUTED, font=F9, anchor="ma")
    draw.line((left, bottom, right, bottom), fill="#7B8798", width=2)
    draw.line((left, top, left, bottom), fill="#7B8798", width=2)
    draw.text((x0 + 18, (top + bottom) / 2), ylabel, fill=MUTED, font=F9, anchor="mm")
    draw.text(((left + right) / 2, y1 - 18), xlabel, fill=MUTED, font=F9, anchor="mm")

    legend_x = x0 + 650
    for s in series:
        draw.line((legend_x, y0 + 34, legend_x + 42, y0 + 34), fill=s["color"], width=s.get("width", 5))
        draw.text((legend_x + 51, y0 + 34), s["label"], fill=s["color"], font=F_LEGEND, anchor="lm")
        legend_x += 330
        pts = [(px(a), py(b)) for a, b in s["points"] if xmin <= a <= xmax and math.isfinite(b)]
        if len(pts) >= 2:
            draw.line(pts, fill=s["color"], width=s.get("width", 5), joint="curve")
        for xx, yy in pts:
            r = s.get("radius", 4)
            draw.ellipse((xx-r, yy-r, xx+r, yy+r), fill=s["color"])

    if failure_x is not None and xmin <= failure_x <= xmax:
        xx = px(failure_x)
        for yy in range(int(top), int(bottom), 18):
            draw.line((xx, yy, xx, min(yy + 10, bottom)), fill="#B42318", width=2)
        draw.text((xx + 6, top + 5), "Fast-WAM failed in eval reset", fill="#B42318", font=F9)


def save_success(data: dict) -> None:
    img = Image.new("RGB", (1600, 1580), BG)
    d = ImageDraw.Draw(img)
    d.text((50, 30), "Current training success snapshot — 2026-09-02 15:52 CST", fill=TEXT, font=F_TITLE)
    d.text((50, 82), "Distinct colors: dark teal = pi0.5; orange = Fast-WAM. Fast-WAM ended during Step-15 evaluation.", fill=MUTED, font=F12)
    xmax = max(p["step"] for x in data.values() for p in x["series"]["env/success_once"])
    layouts = [(40, 130, 1560, 470), (40, 495, 1560, 835), (40, 860, 1560, 1200), (40, 1225, 1560, 1545)]
    specs = [
        ("Per-step training rollout success", 1),
        ("5-step moving mean", 5),
        ("10-step moving mean", 10),
    ]
    for box, (title, win) in zip(layouts[:3], specs):
        ss = []
        for key in ("pi05_control", "fastwam_control"):
            pts = data[key]["series"]["env/success_once"]
            ss.append({"label": LABELS[key], "color": COLORS[key], "points": xy(pts) if win == 1 else rolling(pts, win)})
        panel(d, box, title, ss, (1, xmax), (0.15, 1.0), "success", 15)
    ss = []
    for key in ("pi05_control", "fastwam_control"):
        ss.append({"label": LABELS[key] + " fixed-32", "color": COLORS[key], "points": xy(data[key]["series"]["eval/success_once"]), "radius": 7})
    panel(d, layouts[3], "Fixed-32 evaluation", ss, (1, xmax), (0.15, 1.0), "success", 15)
    img.save(OUT / "01_success_rates.png", quality=95)


def save_optimization(data: dict) -> None:
    tags = [
        ("train/actor/approx_kl", "Approximate KL", "KL"),
        ("train/actor/clip_fraction", "PPO clip fraction", "fraction"),
        ("train/actor/grad_norm", "Actor gradient norm", "norm"),
    ]
    img = Image.new("RGB", (1600, 1180), BG)
    d = ImageDraw.Draw(img)
    d.text((50, 30), "Optimization signals", fill=TEXT, font=F_TITLE)
    d.text((50, 82), "All values are finite; Fast-WAM stopped in evaluation, not in the optimizer.", fill=MUTED, font=F12)
    xmax = max(p["step"] for x in data.values() for p in x["series"]["env/success_once"])
    for i, (tag, title, ylabel) in enumerate(tags):
        ss, vals = [], []
        for key in ("pi05_control", "fastwam_control"):
            pts = xy(data[key]["series"][tag])
            vals.extend(v for _, v in pts)
            ss.append({"label": LABELS[key], "color": COLORS[key], "points": pts})
        lo = min(0.0, min(vals))
        hi = max(vals) * 1.12 if max(vals) > 0 else 1.0
        panel(d, (40, 130 + i*345, 1560, 450 + i*345), title, ss, (1, xmax), (lo, hi), ylabel, 15)
    img.save(OUT / "02_optimization_signals.png", quality=95)


def read_resource(label: str) -> list[dict]:
    path = SRC / "raw" / label / "resource.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    start = datetime.fromisoformat(rows[0]["timestamp"])
    out = []
    for row in rows:
        item = dict(row)
        item["hours"] = (datetime.fromisoformat(row["timestamp"]) - start).total_seconds() / 3600
        out.append(item)
    return out


def save_resources() -> None:
    resources = {k: read_resource(k) for k in ("pi05_control", "fastwam_control")}
    img = Image.new("RGB", (1600, 1180), BG)
    d = ImageDraw.Draw(img)
    d.text((50, 30), "Training resource telemetry", fill=TEXT, font=F_TITLE)
    d.text((50, 82), "Elapsed hours are normalized to each job's own launch; gaps indicate missing fields after a process ended.", fill=MUTED, font=F12)
    specs = []
    for key, rows in resources.items():
        gpu_cols = sorted(k for k in rows[0] if k.endswith("_used_mib"))
        util_cols = sorted(k for k in rows[0] if k.endswith("_util_pct"))
        mem = []
        util = []
        ram = []
        for r in rows:
            mvals = [float(r[c]) for c in gpu_cols if r.get(c) not in (None, "")]
            uvals = [float(r[c]) for c in util_cols if r.get(c) not in (None, "")]
            if mvals:
                mem.append((r["hours"], sum(mvals) / len(mvals) / 1024))
            if uvals:
                util.append((r["hours"], sum(uvals) / len(uvals)))
            if r.get("host_mem_available_kib") not in (None, ""):
                ram.append((r["hours"], float(r["host_mem_available_kib"]) / 1024 / 1024 / 1024))
        specs.append((key, mem, util, ram))

    xmax = max(max(p[0] for p in x[1]) for x in specs)
    mem_series = [{"label": LABELS[k], "color": COLORS[k], "points": m} for k, m, _, _ in specs]
    util_series = [{"label": LABELS[k], "color": COLORS[k], "points": u} for k, _, u, _ in specs]
    ram_series = [{"label": LABELS[k] + " window", "color": COLORS[k], "points": r} for k, _, _, r in specs]
    panel(d, (40, 130, 1560, 450), "Mean GPU memory per allocated card", mem_series, (0, xmax), (0, 80), "GiB/card", xlabel="elapsed hours")
    panel(d, (40, 475, 1560, 795), "Mean instantaneous GPU utilization", util_series, (0, xmax), (0, 100), "%", xlabel="elapsed hours")
    panel(d, (40, 820, 1560, 1140), "Host available memory during each job", ram_series, (0, xmax), (0, 2.0), "TiB", xlabel="elapsed hours")
    img.save(OUT / "03_resource_telemetry.png", quality=95)


def make_interactive(data: dict) -> None:
    payload = {}
    for key in data:
        train = data[key]["series"]["env/success_once"]
        payload[key] = {
            "label": LABELS[key], "color": COLORS[key], "raw": xy(train),
            "ma5": rolling(train, 5), "ma10": rolling(train, 10),
            "eval": xy(data[key]["series"]["eval/success_once"]),
        }
    html = r'''<!doctype html><meta charset="utf-8"><title>SZ live training dashboard</title>
<style>body{font-family:Segoe UI,Arial;background:#f6f8fb;color:#172033;margin:24px}.card{background:white;border:1px solid #d5dce7;border-radius:14px;padding:16px;margin:16px 0}button{margin-right:8px;padding:8px 14px}.tip{position:fixed;background:#172033;color:white;padding:7px;border-radius:6px;display:none;pointer-events:none}svg{width:100%;height:430px}</style>
<h1>Current training success — 2026-09-02 15:52 CST</h1><div><button onclick="render('raw')">Per-step</button><button onclick="render('ma5')">MA5</button><button onclick="render('ma10')">MA10</button><button onclick="render('eval')">Fixed-32</button></div><div class="card"><svg id="chart" viewBox="0 0 1200 430"></svg></div><div id="tip" class="tip"></div>
<script>const D=__DATA__;const svg=document.getElementById('chart'),tip=document.getElementById('tip');function E(n,a){let e=document.createElementNS('http://www.w3.org/2000/svg',n);for(const k in a)e.setAttribute(k,a[k]);return e}function render(mode){svg.innerHTML='';let all=Object.values(D).flatMap(s=>s[mode]);let xmax=Math.max(...all.map(p=>p[0])),ymin=.15,ymax=1;let X=x=>80+(x-1)/(Math.max(2,xmax)-1)*1080,Y=y=>370-(y-ymin)/(ymax-ymin)*310;for(let i=0;i<=5;i++){let y=ymin+(ymax-ymin)*i/5,l=E('line',{x1:80,y1:Y(y),x2:1160,y2:Y(y),stroke:'#dce3ec'});svg.append(l);let t=E('text',{x:68,y:Y(y)+5,'text-anchor':'end',fill:'#667085'});t.textContent=Math.round(y*100)+'%';svg.append(t)}for(const k in D){let s=D[k],pts=s[mode];let p=E('polyline',{points:pts.map(v=>X(v[0])+','+Y(v[1])).join(' '),fill:'none',stroke:s.color,'stroke-width':5});svg.append(p);for(const v of pts){let c=E('circle',{cx:X(v[0]),cy:Y(v[1]),r:6,fill:s.color});c.onmousemove=e=>{tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px';tip.textContent=s.label+' · Step '+v[0]+' · '+(v[1]*100).toFixed(2)+'%'};c.onmouseout=()=>tip.style.display='none';svg.append(c)}}let x=E('text',{x:620,y:415,'text-anchor':'middle',fill:'#667085'});x.textContent='completed global step';svg.append(x)}render('raw')</script>'''.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    (OUT / "interactive_success.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = json.loads((SRC / "metrics.json").read_text(encoding="utf-8"))
    save_success(data)
    save_optimization(data)
    save_resources()
    make_interactive(data)
    summary = {}
    for key, run in data.items():
        train = run["series"]["env/success_once"]
        times = [x["value"] for x in run["series"]["time/step"]]
        summary[key] = {
            "completed_step": train[-1]["step"],
            "latest_success": train[-1]["value"],
            "ma5": rolling(train, 5)[-1][1],
            "ma10": rolling(train, 10)[-1][1],
            "latest_eval": run["series"]["eval/success_once"][-1],
            "latest_kl": run["series"]["train/actor/approx_kl"][-1]["value"],
            "latest_clip": run["series"]["train/actor/clip_fraction"][-1]["value"],
            "latest_grad": run["series"]["train/actor/grad_norm"][-1]["value"],
            "median_last10_step_seconds": median(times[-10:]),
        }
    (SRC / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
