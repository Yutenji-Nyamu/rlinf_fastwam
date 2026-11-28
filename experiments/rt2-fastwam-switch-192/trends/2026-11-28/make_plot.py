"""Rebuild the standalone FastWAM success-rate figure from its per-round counts."""
import base64
import csv
import json
import math
import statistics
from pathlib import Path

from reportlab.graphics import renderSVG
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parent
MODES = (("raw", 0, 0), ("light", 11, 3), ("medium", 21, 6), ("wide", 41, 12))
COLORS = {"raw": "#82939d", "mean": "#00736d", "gauss": "#cf5d29", "base": "#785798"}


def smooth(points, width, sigma):
    means, gaussians = [], []
    for step, _ in points:
        near = [y for x, y in points if abs(step - x) <= width // 2]
        weighted = [(math.exp(-0.5 * ((step - x) / sigma) ** 2), y)
                    for x, y in points if abs(step - x) <= 3 * sigma]
        means.append((step, statistics.mean(near)))
        gaussians.append((step, sum(w * y for w, y in weighted) / sum(w for w, _ in weighted)))
    return means, gaussians


def chart(points, is_eval, width, sigma, baseline, x_max):
    drawing = Drawing(900, 153)
    plot = LinePlot()
    plot.x, plot.y, plot.width, plot.height = 45, 22, 839, 121
    line_data = []
    styles = []
    if is_eval:
        line_data.append([(0, baseline), (x_max, baseline)])
        styles.append(("base", 1.25, [3, 4], False))
    line_data.append(points)
    styles.append(("raw", 1.1 if is_eval else 0.95, None, is_eval))
    if width:
        line_data.extend(smooth(points, width, sigma))
        styles += [("mean", 2.55, None, False), ("gauss", 2.05, [6, 3], False)]
    plot.data = line_data
    for index, (name, weight, dash, with_marker) in enumerate(styles):
        line = plot.lines[index]
        line.strokeColor = HexColor(COLORS[name])
        line.strokeWidth = weight
        if dash:
            line.strokeDashArray = dash
        if with_marker:
            marker = makeMarker("FilledCircle")
            marker.size = 4.3
            marker.fillColor = marker.strokeColor = HexColor(COLORS[name])
            line.symbol = marker
    x_axis, y_axis = plot.xValueAxis, plot.yValueAxis
    x_axis.valueMin, x_axis.valueMax = 0, x_max
    x_axis.valueSteps = list(range(0, x_max + 1, 25))
    y_axis.valueMin, y_axis.valueMax = 0, 80
    y_axis.valueSteps = [0, 20, 40, 60, 80]
    y_axis.labelTextFormat = "%d%%"
    for axis in (x_axis, y_axis):
        axis.strokeColor = HexColor("#a6b4bc")
        axis.strokeWidth = 0.65
        axis.labels.fontName = "Helvetica"
        axis.labels.fontSize = 11
        axis.labels.fillColor = HexColor("#536672")
    y_axis.visibleGrid = True
    y_axis.gridStrokeColor = HexColor("#dce4e8")
    y_axis.gridStrokeWidth = 0.65
    drawing.add(plot)
    svg = renderSVG.drawToString(drawing)
    if isinstance(svg, str):
        svg = svg.encode()
    return "data:image/svg+xml;base64," + base64.b64encode(svg).decode()


def main():
    raw = {"train": [], "eval": [], "baseline": []}
    with (ROOT / "points.csv").open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            raw[row["series"]].append([int(row[k]) for k in ("round", "successes", "episodes")])
    for values in raw.values():
        values.sort()
    assert len(raw["baseline"]) == 1
    base = raw["baseline"][0]
    baseline = 100 * base[1] / base[2]
    last = raw["train"][-1][0]
    x_max = 25 * math.ceil(last / 25)
    images = {mode: {} for mode, _, _ in MODES}
    latest = {}
    for kind in ("train", "eval"):
        points = [(step, 100 * wins / total) for step, wins, total in raw[kind]]
        assert max(y for _, y in points) <= 80
        for mode, width, sigma in MODES:
            images[mode][kind] = chart(points, kind == "eval", width, sigma, baseline, x_max)
        step, wins, total = raw[kind][-1]
        latest[kind] = f"第 {step} 轮　{wins}/{total} · {100*wins/total:.1f}%"
    rows = []
    for a, b in ((1, 50), (51, 100), (101, last)):
        cells = []
        for kind in ("train", "eval"):
            group = [p for p in raw[kind] if a <= p[0] <= b]
            cells.append(f"{100 * sum(p[1] for p in group)/sum(p[2] for p in group):.1f}%")
        rows.append(f"<tr><td>第 {a}–{b} 轮</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>")
    value = HTML
    for key, repl in {
        "IMAGES": json.dumps(images, separators=(",", ":")),
        "POINTS": json.dumps(raw, separators=(",", ":")),
        "TRAIN_LAST": latest["train"], "EVAL_LAST": latest["eval"],
        "TABLE": "".join(rows), "XMAX": str(x_max),
        "CSV": base64.b64encode((ROOT / "points.csv").read_bytes()).decode(),
    }.items():
        value = value.replace("@@" + key + "@@", repl)
    (ROOT / "fastwam-trend.html").write_text(value, encoding="utf-8")


HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FastWAM GRPO · 拨开关 | 训练与评估</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f6;color:#192c36;font:16px/1.45 "Microsoft YaHei","PingFang SC",Arial,sans-serif}
main{max-width:1020px;margin:20px auto;padding:0 16px 30px}#figure,article{background:#fff;padding:22px 30px;border:1px solid #d6e0e3;border-radius:10px}
h1,h2,p{margin:0}h1{font-size:29px;letter-spacing:-.045em;font-weight:680}h2{font-size:17px}header p,.note,caption{color:#516773}.subtitle{margin:3px 0 15px;font-size:14px}
nav{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:13px}nav small{margin-right:4px;color:#536671}
button{cursor:pointer;border:1px solid #cad4d9;border-radius:5px;background:#fff;padding:5px 13px;font:inherit;font-size:13px;color:#29404a}button[aria-pressed=true]{background:#155d59;border-color:#155d59;color:#fff}button:focus-visible,a:focus-visible{outline:2px solid #28699e;outline-offset:3px}
.legend{display:flex;gap:10px 22px;flex-wrap:wrap;font-size:13px;color:#374f59;margin-bottom:14px}.item{display:inline-flex;align-items:center;gap:7px}.key{width:25px;height:0;border-top:2px solid #82939d}.key.m{border-color:#00736d;border-width:3px}.key.g{border-color:#cf5d29;border-top-style:dashed}.key.b{border-color:#785798;border-top-style:dotted}
.panel+ .panel{margin-top:11px}.panel-head{display:flex;justify-content:space-between;align-items:baseline;gap:6px;flex-wrap:wrap}.readout{font:13px/1.5 "Microsoft YaHei",Arial,sans-serif;color:#475f6b;font-variant-numeric:tabular-nums}.plot{display:block;width:100%;height:auto;aspect-ratio:900/153;touch-action:pan-y}
.axis-title{text-align:center;color:#566872;font-size:12px;margin:0}.note{font-size:12px;margin-top:12px}.onlymobile{display:none}
article{margin-top:14px}article table{border-collapse:collapse;margin:10px 0 8px;width:min(100%,530px);font-size:14px;text-align:left;font-variant-numeric:tabular-nums}
th,td{padding:5px 15px 5px 0;border-bottom:1px solid #e3e8ea}th{color:#546d77;font-weight:500}a{color:#006b65}.actions{margin-top:8px}
@media(max-width:600px){main{padding:0 6px 12px;margin:6px auto}#figure,article{padding:15px 10px;border-radius:7px}h1{font-size:23px}h2{font-size:15px}.legend{gap:4px 12px;font-size:11px}.readout{font-size:11px}.plot{min-height:106px}.onlymobile{display:inline}.subtitle{font-size:12px}nav{gap:4px}button{padding:4px 9px}.panel+.panel{margin-top:10px}}
</style></head><body><main>
<section id="figure" aria-label="FastWAM 成功率图">
<header><h1>FastWAM GRPO · 拨开关</h1><p class="subtitle">RoboTwin 2.0 · 物理卡 4、7 · 2026-11-28 11:47 快照</p></header>
<nav aria-label="平滑窗口"><small>平滑宽度</small><button data-mode="raw" aria-pressed="false">仅原始点</button><button data-mode="light" aria-pressed="false">11 轮</button><button data-mode="medium" aria-pressed="true">21 轮</button><button data-mode="wide" aria-pressed="false">41 轮</button></nav>
<div class="legend"><span class="item"><i class="key"></i>原始成功率</span><span class="item" id="mean"><i class="key m"></i><span></span></span><span class="item" id="gauss"><i class="key g"></i><span></span></span><span class="item"><i class="key b"></i>原模型独评 13/32（40.6%）</span></div>
<section class="panel"><div class="panel-head"><h2>训练采样 · 每轮 256 局</h2><output class="readout" id="train-out">@@TRAIN_LAST@@</output></div><img id="train" class="plot" width="900" height="153" alt="训练采样逐轮成功率"></section>
<section class="panel"><div class="panel-head"><h2>独立评估 · 每 5 轮评 32 局</h2><output class="readout" id="eval-out">@@EVAL_LAST@@</output></div><img id="eval" class="plot" width="900" height="153" alt="独立评估成功率和原模型基线"></section>
<p class="axis-title">训练轮次</p><p class="note">灰线保留原始波动。平滑按前后相邻的训练轮数计算，曲线首尾仅使用已有观测。恢复后重复的轮次取实际接续版本。</p>
</section>
<article><h2>分段成功率</h2><table><thead><tr><th>训练轮次</th><th>训练采样</th><th>独立评估</th></tr></thead><tbody>@@TABLE@@</tbody></table>
<p class="note">分段值用该段全部成功局数除以总局数。原模型基线单列。鼠标移到图上或用手指点选，可查看原始局数。</p>
<p class="note actions"><a download="fastwam-turn-switch-rounds.csv" href="data:text/csv;base64,@@CSV@@">下载逐轮原始数据（CSV）</a></p></article></main>
<script>const images=@@IMAGES@@, points=@@POINTS@@, cap=@@XMAX@@;
const modes={raw:[0,0], light:[11,3], medium:[21,6], wide:[41,12]};
function setMode(m) {
 document.querySelectorAll('button[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===m)));
 const [w,s]=modes[m]; const mean=document.getElementById('mean'),gauss=document.getElementById('gauss');
 mean.style.display=gauss.style.display=w?'inline-flex':'none';
 mean.lastElementChild.textContent=w+' 轮滑动平均'; gauss.lastElementChild.textContent='高斯平滑（σ='+s+' 轮）';
 for(const kind of ['train','eval']) document.getElementById(kind).src=images[m][kind];
}
function show(kind,p) {document.getElementById(kind+'-out').textContent='第 '+p[0]+' 轮　'+p[1]+'/'+p[2]+' · '+(100*p[1]/p[2]).toFixed(1)+'%'}
for(const kind of ['train','eval']){
 const img=document.getElementById(kind);const rows=points[kind];
 function inspect(e){const a=img.getBoundingClientRect(),step=cap*((e.clientX-a.left)/a.width*900-45)/839;const nearest=rows.reduce((p,q)=>Math.abs(q[0]-step)<Math.abs(p[0]-step)?q:p);show(kind,nearest)}
 img.addEventListener('pointermove',inspect);img.addEventListener('click',inspect);img.addEventListener('pointerleave',()=>show(kind,rows[rows.length-1]));
}
document.querySelectorAll('button[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));setMode('medium');
</script></body></html>"""

if __name__ == "__main__":
    main()

