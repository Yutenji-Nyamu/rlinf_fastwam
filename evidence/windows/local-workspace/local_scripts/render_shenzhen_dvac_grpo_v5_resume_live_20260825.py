from __future__ import annotations

import csv
import html
import importlib.util
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-pi0-ppo-rlt" / "evidence"
V4 = EVIDENCE / "dvac-grpo-v4-live-step31-20260825"
V5 = Path(os.environ.get(
    "DVAC_EVIDENCE_DIR",
    EVIDENCE / "dvac-grpo-v5-live-step40-20260825",
))
BASELINE_CSV = EVIDENCE / "grpo_v2_final_step52_20260823" / "grpo_console_scalars_step1_52.csv"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"


def load_base():
    spec = importlib.util.spec_from_file_location("grpo_plot_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_base()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_grpo_any(path: Path) -> list[dict[str, float | int | None]]:
    text = B.ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(re.finditer(r"Global Step:\s*(\d+)/100", text))
    rows: list[dict[str, float | int | None]] = []
    for idx, match in enumerate(matches):
        step = int(match.group(1))
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[match.start() : end]
        success = [float(value) for value in re.findall(rf"(?<![\w/])success_once=({B.NUMBER})", segment)]
        elapsed_match = re.search(r"Elapsed:\s*([0-9:]+)", segment)
        if not success or elapsed_match is None:
            raise ValueError(f"step {step}: incomplete table")
        row: dict[str, float | int | None] = {
            "step": step,
            "elapsed_s": B.parse_elapsed(elapsed_match.group(1)),
            "train_success": success[0],
            "eval_success": success[1] if len(success) > 1 else None,
            "approx_kl": B.first_number(segment, "actor/approx_kl"),
            "clip_fraction": B.first_number(segment, "actor/clip_fraction"),
            "grad_norm": B.first_number(segment, "actor/grad_norm"),
            "ratio_abs": B.first_number(segment, "actor/ratio_abs"),
            "policy_loss_abs": B.first_number(segment, "actor/policy_loss_abs"),
            "step_time_s": B.first_number(segment, "step"),
            "generate_rollouts_s": B.first_number(segment, "generate_rollouts"),
            "actor_training_s": B.first_number(segment, "actor_training"),
        }
        rows.append(row)
    steps = [int(row["step"]) for row in rows]
    if not steps or steps != list(range(min(steps), max(steps) + 1)):
        raise ValueError(f"non-consecutive GRPO steps: {steps}")
    for row in rows:
        for key, value in row.items():
            if key == "eval_success" and value is None:
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite {key} at step {row['step']}")
    return rows


def console_metric(path: Path, step: int, key: str) -> float:
    text = B.ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    matches = list(re.finditer(r"Global Step:\s*(\d+)/100", text))
    for index, match in enumerate(matches):
        if int(match.group(1)) != step:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return B.first_number(text[match.start() : end], key)
    raise ValueError(f"missing Global Step {step} for {key}")


def scalar_map(tb: dict, tag: str) -> dict[int, float]:
    return {int(item["step"]) + 1: float(item["value"]) for item in tb.get(tag, [])}


def stitch_scalar(v4_tb: dict, v5_tb: dict, tag: str, complete: int) -> list[float]:
    first = scalar_map(v4_tb, tag)
    second = scalar_map(v5_tb, tag)
    merged = {step: value for step, value in first.items() if step <= 30}
    merged.update({step: value for step, value in second.items() if 31 <= step <= complete})
    expected = list(range(1, complete + 1))
    if sorted(merged) != expected:
        raise ValueError(f"{tag}: expected global steps {expected}, got {sorted(merged)}")
    return [merged[step] for step in expected]


def vertical_cut(draw, plot, x_min: float, x_max: float, cut: float, label: str, color: str = B.GRAY) -> None:
    left, top, right, bottom = plot
    x = left + (cut - x_min) / (x_max - x_min) * (right - left)
    for y in range(int(top), int(bottom), 18):
        draw.line((x, y, x, min(y + 9, bottom)), fill=color, width=2)
    if label:
        draw.text((x + 8, top + 44), label, fill=color, font=B.F_SMALL)


def label_eval(draw, plot, step: float, value: float, text: str, color: str, dx: int, dy: int) -> None:
    left, top, right, bottom = plot
    x = left + (step - 1.0) / 51.0 * (right - left)
    y = bottom - (value - 0.84) / 0.16 * (bottom - top)
    draw.text((x + dx, y + dy), text, fill=color, font=B.F_SMALL)


def render_success(actual: list[dict], baseline: list[dict], evals: dict[int, float]) -> Path:
    dvac_color = "#E4572E"
    grpo_color = "#00796B"
    grpo_light = "#80CBC4"
    complete = int(actual[-1]["step"])
    xs = [float(row["step"]) for row in baseline]
    current = [float(row["train_success"]) for row in actual]
    base = [float(row["train_success"]) for row in baseline]
    current_pad = current + [None] * (len(base) - len(current))
    current_ma = B.trailing(current) + [None] * (len(base) - len(current))
    baseline_ma = B.trailing(base)
    base_paired = [value if index < complete else None for index, value in enumerate(base)]
    base_paired_ma = [value if index < complete else None for index, value in enumerate(baseline_ma)]
    base_future = [None if index < complete - 1 else value for index, value in enumerate(base)]
    base_future_ma = [None if index < complete - 1 else value for index, value in enumerate(baseline_ma)]
    baseline_evals = [(int(row["step"]), float(row["eval_success"])) for row in baseline if row["eval_success"]]

    out = V5 / f"01_dvac_resume_vs_grpo_success_step1_{complete}.png"
    image = Image.new("RGB", (1440, 1840), B.LIGHT)
    draw = B.add_header(
        image,
        f"GRPO-DVAC resumed trajectory through Step {complete}",
        "Orange = actual DVAC branch; deep teal = strict-matched GRPO. Dashed orange/gray = resume/current cuts.",
    )
    raw_plot = B.line_panel(
        draw,
        (45, 145, 1395, 630),
        xs,
        [
            ("DVAC raw", current_pad, dvac_color, 4),
            ("GRPO raw (paired)", base_paired, grpo_color, 4),
            ("GRPO baseline-only", base_future, grpo_light, 3),
        ],
        0.65,
        1.0,
        "Per-step training-rollout success",
    )
    ma_plot = B.line_panel(
        draw,
        (45, 655, 1395, 1140),
        xs,
        [
            ("DVAC 5-step mean", current_ma, dvac_color, 7),
            ("GRPO 5-step mean (paired)", base_paired_ma, grpo_color, 7),
            ("GRPO baseline-only", base_future_ma, grpo_light, 4),
        ],
        0.65,
        1.0,
        "Five-step moving mean",
    )
    eval_plot = B.line_panel(draw, (45, 1165, 1395, 1685), xs, [], 0.84, 1.0, "Fixed-64 evaluation")
    for plot in (raw_plot, ma_plot, eval_plot):
        vertical_cut(draw, plot, 1.0, 52.0, 30.5, "", dvac_color)
        vertical_cut(draw, plot, 1.0, 52.0, complete + 0.5, "", B.GRAY)

    current_markers = [(step - 0.25, value) for step, value in sorted(evals.items())]
    baseline_markers = [(step + 0.25, value) for step, value in baseline_evals]
    B.draw_eval_markers(draw, eval_plot, xs, 0.84, 1.0, current_markers, dvac_color, "square")
    B.draw_eval_markers(draw, eval_plot, xs, 0.84, 1.0, baseline_markers, grpo_color, "triangle")
    legend_y = eval_plot[1] + 14
    draw.rectangle((eval_plot[0] + 12, legend_y, eval_plot[0] + 30, legend_y + 18), fill=dvac_color)
    draw.text((eval_plot[0] + 40, legend_y - 2), "orange square = DVAC", fill=dvac_color, font=B.F_SMALL)
    lx = eval_plot[0] + 255
    draw.polygon(((lx + 9, legend_y), (lx, legend_y + 18), (lx + 18, legend_y + 18)), fill=grpo_color)
    draw.text((lx + 28, legend_y - 2), "teal triangle = matched GRPO", fill=grpo_color, font=B.F_SMALL)
    base_eval_map = dict(baseline_evals)
    for step, value in sorted(evals.items()):
        base_value = base_eval_map[step]
        label_eval(draw, eval_plot, step, max(value, base_value), f"D {round(value * 64)} | G {round(base_value * 64)}", B.INK, -50, -32)
    for step, value in baseline_evals:
        if step > complete:
            label_eval(draw, eval_plot, step + 0.25, value, f"G {round(value * 64)}/64", grpo_color, 10, 8)

    paired = base[:complete]
    fixed_dvac = sum(round(value * 64) for value in evals.values())
    fixed_base = sum(round(base_eval_map[step] * 64) for step in evals)
    card = (
        f"Paired Steps 1-{complete}: mean DVAC-GRPO = {(statistics.mean(current) - statistics.mean(paired)) * 100:+.2f} pp; "
        f"latest-5 = {(statistics.mean(current[-5:]) - statistics.mean(paired[-5:])) * 100:+.2f} pp.  "
        f"Fixed through Step {max(evals)} = {fixed_dvac}/{64 * len(evals)} vs {fixed_base}/{64 * len(evals)}."
    )
    draw.rounded_rectangle((55, 1710, 1385, 1805), radius=18, fill=B.WHITE, outline=B.GRID, width=2)
    draw.text((78, 1738), card, fill=B.INK, font=B.F_SMALL)
    image.save(out, optimize=True)
    return out


def render_optimization(actual: list[dict], series: dict[str, list[float]]) -> Path:
    complete = int(actual[-1]["step"])
    steps = [float(row["step"]) for row in actual]
    mean = series["train/actor/dvac_weight_mean"]
    sq = series["train/actor/dvac_weight_sq_mean"]
    std = [math.sqrt(max(0.0, second - first * first)) for first, second in zip(mean, sq)]
    out = V5 / f"02_dvac_resume_optimization_and_weights_step1_{complete}.png"
    image = Image.new("RGB", (1440, 2040), B.LIGHT)
    draw = B.add_header(
        image,
        f"GRPO-DVAC optimization and weights through Step {complete}",
        "The resume boundary is explicit; all plotted optimization scalars remain finite.",
    )
    panels = [
        (
            "Approximate KL and PPO clip fraction",
            [("KL", series["train/actor/approx_kl"], B.PURPLE, 5), ("Clip fraction", series["train/actor/clip_fraction"], B.GRPO, 5)],
            0.0,
            0.13,
        ),
        ("Pre-clip gradient norm", [("Gradient norm", series["train/actor/grad_norm"], B.GREEN, 5)], 0.0, 55.0),
        (
            "DVAC weight balance",
            [
                ("Weight mean", mean, B.PURPLE, 5),
                ("ESS fraction", series["train/actor/dvac_weight_ess_fraction"], B.GREEN, 5),
                ("Weight std", std, B.GRPO, 5),
            ],
            0.0,
            1.15,
        ),
        (
            "DVAC z clipping",
            [
                ("Upper clip fraction", series["train/actor/dvac_z_high_clip_fraction"], B.RED, 5),
                ("Lower clip fraction", series["train/actor/dvac_z_low_clip_fraction"], B.PPO, 5),
            ],
            0.0,
            0.07,
        ),
    ]
    top, height, gap = 145, 430, 25
    for index, (title, lines, low, high) in enumerate(panels):
        plot = B.line_panel(
            draw,
            (45, top + index * (height + gap), 1395, top + index * (height + gap) + height),
            steps,
            lines,
            low,
            high,
            title,
        )
        vertical_cut(draw, plot, 1.0, float(complete), 30.5, "resume", B.PURPLE)
    image.save(out, optimize=True)
    return out


def moving_optional(values: list[float], window: int) -> list[float | None]:
    return [statistics.mean(values[index - window + 1 : index + 1]) if index >= window - 1 else None for index in range(len(values))]


def read_resources(path: Path) -> list[dict]:
    rows = []
    for raw in read_csv(path):
        memory = [float(raw[f"gpu{gpu}_used_mib"]) / 1024 for gpu in range(4, 8)]
        utilization = [float(raw[f"gpu{gpu}_util_pct"]) for gpu in range(4, 8)]
        rows.append(
            {
                "timestamp": datetime.fromisoformat(raw["timestamp"]),
                "host_available_gib": float(raw["host_mem_available_kib"]) / 1024**2,
                "gpu_mean_gib": statistics.mean(memory),
                "gpu_max_gib": max(memory),
                "gpu_mean_util": statistics.mean(utilization),
                "gpu_max_util": max(utilization),
            }
        )
    return rows


def render_resources(resources: list[dict], complete: int) -> Path:
    start = resources[0]["timestamp"]
    hours = [(row["timestamp"] - start).total_seconds() / 3600 for row in resources]
    host = [float(row["host_available_gib"]) for row in resources]
    gpu_mean = [float(row["gpu_mean_gib"]) for row in resources]
    gpu_max = [float(row["gpu_max_gib"]) for row in resources]
    util = [float(row["gpu_mean_util"]) for row in resources]
    last = hours[-1]
    out = V5 / f"03_dvac_resume_resource_since_restart_step{complete}.png"
    image = Image.new("RGB", (1440, 1790), B.LIGHT)
    draw = B.add_header(
        image,
        f"GRPO-DVAC v5 resources since resume (through Step {complete})",
        "One-minute samples; 100 GiB is the approximate Ray 95% whole-node memory boundary.",
    )
    panels = [
        ("Host available memory", [("Available RAM", host, B.GREEN, 5), ("Approx. Ray boundary", [100.0] * len(host), B.RED, 3)], 0.0, 2050.0),
        ("Physical GPUs 4-7 memory", [("Mean card", gpu_mean, B.GRPO, 5), ("Max card", gpu_max, B.RED, 5)], 0.0, 80.0),
        ("Physical GPUs 4-7 utilization", [("Raw mean", util, B.GRPO_LIGHT, 2), ("15-minute mean", moving_optional(util, 15), B.GRPO, 6)], 0.0, 100.0),
    ]
    ticks = [0.0, last / 4, last / 2, 3 * last / 4, last]
    top, height, gap = 145, 460, 25
    for index, (title, lines, low, high) in enumerate(panels):
        B.line_panel(
            draw,
            (45, top + index * (height + gap), 1395, top + index * (height + gap) + height),
            hours,
            lines,
            low,
            high,
            title,
            x_label="Hours since v5 resume launch",
            x_ticks=ticks,
            x_tick_format=lambda value: f"{value:.1f}",
        )
    image.save(out, optimize=True)
    return out


def render_dashboard(summary: dict, success_actual: list[float], success_base: list[float], series: dict[str, list[float]], resources: list[dict]) -> Path:
    complete = summary["capture"]["complete_step"]
    latest_eval_step = max(summary["success"]["fixed64_dvac"])
    latest_eval_dvac = round(summary["success"]["fixed64_dvac"][latest_eval_step] * 64)
    latest_eval_base = round(summary["success"]["fixed64_baseline_paired"][latest_eval_step] * 64)
    start = resources[0]["timestamp"]
    hours = [(row["timestamp"] - start).total_seconds() / 3600 for row in resources]
    payload = {
        "success": {
            "x": list(range(1, complete + 1)),
            "dvac": success_actual,
            "grpo": success_base[:complete],
        },
        "optimization": {
            "x": list(range(1, complete + 1)),
            "kl": series["train/actor/approx_kl"],
            "clip": series["train/actor/clip_fraction"],
            "ess": series["train/actor/dvac_weight_ess_fraction"],
        },
        "resources": {
            "x": hours,
            "ram": [row["host_available_gib"] for row in resources],
            "gpu": [row["gpu_max_gib"] for row in resources],
        },
    }
    out = V5 / f"dvac_grpo_resume_live_dashboard_step{complete}.html"
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GRPO-DVAC Step {complete}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f5f7fb;color:#152033}}
main{{max-width:1180px;margin:auto;padding:28px}} h1{{margin:0 0 6px}} .sub{{color:#667085}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}}
.card,.chart{{background:white;border:1px solid #dfe5ef;border-radius:14px;padding:16px;box-shadow:0 2px 8px #1020400b}}
.big{{font-size:25px;font-weight:700;margin-top:6px}} .charts{{display:grid;gap:16px}} canvas{{width:100%;height:310px}}
.note{{font-size:13px;color:#667085;margin-top:8px}} code{{background:#eef1f6;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>GRPO-DVAC 续跑实时看板：完整 Step {complete}</h1>
<div class="sub">实际轨迹 = v4 Step 1–30 + v5 从 global_step_30 恢复；鼠标移动到曲线上可看精确值。</div>
<div class="cards">
<div class="card">最新 train success<div class="big">{summary['success']['dvac_latest']*100:.2f}%</div></div>
<div class="card">近 5 步相对原 GRPO<div class="big">{summary['success']['dvac_minus_baseline_latest5']*100:+.2f} pp</div></div>
<div class="card">Step {latest_eval_step} fixed-64<div class="big">{latest_eval_dvac}/64 vs {latest_eval_base}/64</div></div>
<div class="card">当前可用主存<div class="big">{summary['resources']['host_available_latest_gib']:.0f} GiB</div></div>
</div>
<div class="charts">
<section class="chart"><h2>训练成功率：DVAC vs 严格同预算 GRPO</h2><canvas id="success"></canvas><div class="note">紫色 DVAC，蓝色 GRPO；虚线为 Step 30 恢复边界。</div></section>
<section class="chart"><h2>优化与权重稳定性</h2><canvas id="optimization"></canvas><div class="note">KL、clip fraction、DVAC effective sample size fraction；三者共用纵轴仅用于看趋势。</div></section>
<section class="chart"><h2>v5 恢复后的资源</h2><canvas id="resources"></canvas><div class="note">绿色为主机可用 RAM（GiB），红色为四卡中单卡最大显存（GiB）；双纵轴。</div></section>
</div>
<script>
const D={json.dumps(payload, separators=(',', ':'))};
const colors={{purple:'#7C3AED',blue:'#2563EB',green:'#16803A',red:'#DC2626',gray:'#667085',grid:'#E5E7EB'}};
function chart(id,x,lines,opt={{}}){{
 const c=document.getElementById(id),ctx=c.getContext('2d'); let hover=-1;
 function resize(){{const r=c.getBoundingClientRect(); c.width=r.width*devicePixelRatio;c.height=r.height*devicePixelRatio;ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);draw();}}
 function draw(){{const W=c.clientWidth,H=c.clientHeight,L=62,R=42,T=18,B=44;ctx.clearRect(0,0,W,H);let ys=lines.flatMap(s=>s.v).filter(Number.isFinite);let lo=opt.lo??Math.min(...ys),hi=opt.hi??Math.max(...ys);if(hi===lo)hi=lo+1;
  ctx.strokeStyle=colors.grid;ctx.fillStyle=colors.gray;ctx.font='12px system-ui';for(let i=0;i<6;i++){{let y=T+(H-T-B)*i/5;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(W-R,y);ctx.stroke();let v=hi-(hi-lo)*i/5;ctx.fillText(v.toFixed(opt.dec??2),6,y+4);}}
  const px=i=>L+(W-L-R)*i/Math.max(1,x.length-1),py=v=>T+(H-T-B)*(hi-v)/(hi-lo);
  if(opt.resume){{let i=x.findIndex(v=>v>=30.5);let xx=L+(W-L-R)*(29.5)/Math.max(1,x.length-1);ctx.setLineDash([5,5]);ctx.strokeStyle=colors.purple;ctx.beginPath();ctx.moveTo(xx,T);ctx.lineTo(xx,H-B);ctx.stroke();ctx.setLineDash([]);}}
  lines.forEach(s=>{{ctx.strokeStyle=s.c;ctx.lineWidth=2.6;ctx.beginPath();s.v.forEach((v,i)=>{{let xx=px(i),yy=py(v);i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy)}});ctx.stroke();}});
  ctx.fillStyle=colors.gray;ctx.fillText(String(x[0].toFixed?x[0].toFixed(opt.xdec??0):x[0]),L,H-14);let last=String(x[x.length-1].toFixed?x[x.length-1].toFixed(opt.xdec??0):x[x.length-1]);ctx.fillText(last,W-R-ctx.measureText(last).width,H-14);
  let lx=L;lines.forEach(s=>{{ctx.fillStyle=s.c;ctx.fillRect(lx,T,14,3);ctx.fillStyle=colors.gray;ctx.fillText(s.n,lx+19,T+5);lx+=ctx.measureText(s.n).width+48;}});
  if(hover>=0){{let xx=px(hover);ctx.strokeStyle='#111827';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(xx,T);ctx.lineTo(xx,H-B);ctx.stroke();let txt='x='+Number(x[hover]).toFixed(opt.xdec??0)+'  '+lines.map(s=>s.n+'='+Number(s.v[hover]).toFixed(3)).join('  ');ctx.font='12px system-ui';let tw=ctx.measureText(txt).width+16;let tx=Math.min(Math.max(L,xx-tw/2),W-R-tw);ctx.fillStyle='#111827';ctx.fillRect(tx,H-B-29,tw,23);ctx.fillStyle='white';ctx.fillText(txt,tx+8,H-B-13);}}
 }}
 c.addEventListener('mousemove',e=>{{const r=c.getBoundingClientRect(),ratio=(e.clientX-r.left-62)/(r.width-104);hover=Math.max(0,Math.min(x.length-1,Math.round(ratio*(x.length-1))));draw();}});c.addEventListener('mouseleave',()=>{{hover=-1;draw();}});window.addEventListener('resize',resize);resize();
}}
chart('success',D.success.x,[{{n:'DVAC',v:D.success.dvac,c:colors.purple}},{{n:'GRPO',v:D.success.grpo,c:colors.blue}}],{{lo:.65,hi:1,dec:2,resume:true}});
chart('optimization',D.optimization.x,[{{n:'KL',v:D.optimization.kl,c:colors.purple}},{{n:'clip',v:D.optimization.clip,c:colors.blue}},{{n:'ESS',v:D.optimization.ess,c:colors.green}}],{{lo:0,hi:1,dec:2,resume:true}});
chart('resources',D.resources.x,[{{n:'RAM available',v:D.resources.ram,c:colors.green}},{{n:'GPU max GiB',v:D.resources.gpu,c:colors.red}}],{{lo:0,hi:2050,dec:0,xdec:1}});
</script></main></body></html>"""
    out.write_text(content, encoding="utf-8")
    return out


def main() -> None:
    v4_all = parse_grpo_any(V4 / "driver.log")
    v5_rows = parse_grpo_any(V5 / "driver.log")
    actual = [row for row in v4_all if int(row["step"]) <= 30] + [row for row in v5_rows if int(row["step"]) >= 31]
    steps = [int(row["step"]) for row in actual]
    complete = steps[-1]
    if steps != list(range(1, complete + 1)):
        raise ValueError(f"actual resume branch is not consecutive: {steps}")
    baseline = read_csv(BASELINE_CSV)
    v4_tb = json.loads((V4 / "tensorboard_scalars.json").read_text(encoding="utf-8"))
    v5_tb = json.loads((V5 / "tensorboard_scalars_snapshot.json").read_text(encoding="utf-8"))
    tags = [
        "train/actor/approx_kl",
        "train/actor/clip_fraction",
        "train/actor/grad_norm",
        "train/actor/dvac_weight_mean",
        "train/actor/dvac_weight_sq_mean",
        "train/actor/dvac_weight_ess_fraction",
        "train/actor/dvac_z_high_clip_fraction",
        "train/actor/dvac_z_low_clip_fraction",
    ]
    series = {tag: stitch_scalar(v4_tb, v5_tb, tag, complete) for tag in tags}
    v5_latest_only = {
        key: console_metric(V5 / "driver.log", complete, f"actor/{key}")
        for key in ("dvac_current_mean", "dvac_current_std", "dvac_history_mean", "dvac_history_std")
    }
    evals = {step: value for step, value in scalar_map(v4_tb, "eval/success_once").items() if step <= 30}
    evals.update({step: value for step, value in scalar_map(v5_tb, "eval/success_once").items() if step >= 31})
    resources = read_resources(V5 / "resource.csv")

    success_actual = [float(row["train_success"]) for row in actual]
    success_base = [float(row["train_success"]) for row in baseline]
    weight_mean = series["train/actor/dvac_weight_mean"]
    weight_sq = series["train/actor/dvac_weight_sq_mean"]
    weight_std = [math.sqrt(max(0.0, sq - mean * mean)) for mean, sq in zip(weight_mean, weight_sq)]
    host = [float(row["host_available_gib"]) for row in resources]
    gpu_max = [float(row["gpu_max_gib"]) for row in resources]
    gpu_mean = [float(row["gpu_mean_gib"]) for row in resources]
    base_eval_map = {int(row["step"]): float(row["eval_success"]) for row in baseline if row["eval_success"]}

    figures = [
        render_success(actual, baseline, evals),
        render_optimization(actual, series),
        render_resources(resources, complete),
    ]
    B.write_rows(V5 / f"dvac_actual_branch_console_step1_{complete}.csv", actual)
    summary = {
        "capture": {
            "complete_step": complete,
            "resume_from_step": 30,
            "next_incomplete_step": complete + 1,
            "resource_rows": len(resources),
            "resource_last_timestamp": resources[-1]["timestamp"].isoformat(),
        },
        "success": {
            "dvac_latest": success_actual[-1],
            "baseline_latest_paired": success_base[complete - 1],
            "dvac_mean_step1_current": statistics.mean(success_actual),
            "baseline_mean_step1_current": statistics.mean(success_base[:complete]),
            "dvac_minus_baseline_mean": statistics.mean(success_actual) - statistics.mean(success_base[:complete]),
            "dvac_trailing5": statistics.mean(success_actual[-5:]),
            "baseline_trailing5": statistics.mean(success_base[complete - 5 : complete]),
            "dvac_minus_baseline_latest5": statistics.mean(success_actual[-5:]) - statistics.mean(success_base[complete - 5 : complete]),
            "fixed64_dvac": evals,
            "fixed64_baseline_paired": {step: base_eval_map[step] for step in evals},
            "fixed64_cumulative_dvac": sum(round(value * 64) for value in evals.values()),
            "fixed64_cumulative_baseline": sum(round(base_eval_map[step] * 64) for step in evals),
        },
        "optimization_latest": {
            "approx_kl": series["train/actor/approx_kl"][-1],
            "clip_fraction": series["train/actor/clip_fraction"][-1],
            "grad_norm": series["train/actor/grad_norm"][-1],
            "policy_loss_abs": float(actual[-1]["policy_loss_abs"]),
        },
        "dvac_latest": {
            "current_mean": v5_latest_only["dvac_current_mean"],
            "history_mean": v5_latest_only["dvac_history_mean"],
            "current_std": v5_latest_only["dvac_current_std"],
            "history_std": v5_latest_only["dvac_history_std"],
            "weight_mean": weight_mean[-1],
            "weight_std": weight_std[-1],
            "ess_fraction": series["train/actor/dvac_weight_ess_fraction"][-1],
            "z_high_clip_fraction": series["train/actor/dvac_z_high_clip_fraction"][-1],
            "z_low_clip_fraction": series["train/actor/dvac_z_low_clip_fraction"][-1],
        },
        "timing": {
            "latest_step_s": float(actual[-1]["step_time_s"]),
            "v5_median_step_s": statistics.median(float(row["step_time_s"]) for row in v5_rows),
            "v5_latest5_mean_step_s": statistics.mean(float(row["step_time_s"]) for row in v5_rows[-5:]),
            "latest_rollout_s": float(actual[-1]["generate_rollouts_s"]),
            "latest_actor_training_s": float(actual[-1]["actor_training_s"]),
        },
        "resources": {
            "host_available_launch_gib": host[0],
            "host_available_min_gib": min(host),
            "host_available_latest_gib": host[-1],
            "gpu_card_peak_gib": max(gpu_max),
            "gpu_card_latest_max_gib": gpu_max[-1],
            "gpu_card_latest_mean_gib": gpu_mean[-1],
        },
        "figures": [str(path) for path in figures],
    }
    dashboard = render_dashboard(summary, success_actual, success_base, series, resources)
    summary["dashboard"] = str(dashboard)
    (V5 / f"summary_step{complete}.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    readme = f"""# GRPO-DVAC weights [0,2] closeout through complete Step {complete}

- Actual branch: v4 Step 1--30 plus v5 resumed Step 31--{complete}; discarded v4 Step 31--33 is intentionally excluded.
- Capture: the user-requested stop occurred during incomplete Step {complete + 1}; only complete Step 1--{complete} is analyzed.
- Inputs: small driver log, TensorBoard scalar JSON, resolved YAML, and one-minute resource CSV only.
- Outputs: three phone-readable PNGs, one self-contained interactive HTML dashboard, stitched console CSV, and summary JSON.
- No checkpoint, video, Ray session log, or ZIP is included.

## Key values

- Train success Step {complete}: {success_actual[-1] * 100:.2f}%.
- Paired mean DVAC-GRPO through Step {complete}: {(statistics.mean(success_actual) - statistics.mean(success_base[:complete])) * 100:+.2f} percentage points.
- Paired trailing-5 difference: {(statistics.mean(success_actual[-5:]) - statistics.mean(success_base[complete - 5:complete])) * 100:+.2f} percentage points.
- Fixed-64 through Step {max(evals)}: {sum(round(value * 64) for value in evals.values())}/{64 * len(evals)} versus {sum(round(base_eval_map[step] * 64) for step in evals)}/{64 * len(evals)}.
- Latest KL/clip/grad: {series['train/actor/approx_kl'][-1]:.4f} / {series['train/actor/clip_fraction'][-1]:.4f} / {series['train/actor/grad_norm'][-1]:.3f}.
- Latest DVAC ESS/weight mean: {series['train/actor/dvac_weight_ess_fraction'][-1]:.3f} / {weight_mean[-1]:.3f}.
- Latest host available / max card memory: {host[-1]:.1f} GiB / {gpu_max[-1]:.1f} GiB.
"""
    (V5 / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
