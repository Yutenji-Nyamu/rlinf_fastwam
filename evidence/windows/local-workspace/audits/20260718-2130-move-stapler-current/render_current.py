from __future__ import annotations

import csv
import datetime as dt
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
MOBILE = ROOT / "mobile-figures"
VIS = Path(
    r"C:\Users\86136\.codex\visualizations\2026\07\16"
    r"\019f69a4-b29a-77f1-8ee5-76ebd7e4aa49"
)
MOBILE.mkdir(parents=True, exist_ok=True)
VIS.mkdir(parents=True, exist_ok=True)


def parse_metrics() -> list[dict[str, float | int]]:
    text = (ROOT / "run_embodiment.log").read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?=│ Global Step:\s+\d+/100)", text)
    patterns = {
        "step": r"Global Step:\s+(\d+)/100",
        "success": r"success_once=([-+0-9.eE]+)",
        "kl": r"actor/approx_kl=([-+0-9.eE]+)",
        "clip": r"actor/clip_fraction=([-+0-9.eE]+)",
        "ratio_abs": r"actor/ratio_abs=([-+0-9.eE]+)",
        "grad": r"actor/grad_norm=([-+0-9.eE]+)",
        "loss_abs": r"actor/policy_loss_abs=([-+0-9.eE]+)",
        "step_time": r"Step Time:\s+([0-9.]+)s",
    }
    rows: list[dict[str, float | int]] = []
    for block in blocks:
        row: dict[str, float | int] = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                row[key] = int(match.group(1)) if key == "step" else float(match.group(1))
        if len(row) == len(patterns):
            rows.append(row)
    if not rows:
        raise RuntimeError("no complete metric blocks")
    return rows


def parse_resources() -> list[dict[str, float | dt.datetime]]:
    rows: list[dict[str, float | dt.datetime]] = []
    with (ROOT / "resources-downsampled.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "timestamp": dt.datetime.strptime(raw["timestamp"], "%Y-%m-%d %H:%M:%S"),
                        "ram": float(raw["cgroup_ram_pct"]),
                        "gpu0": float(raw["gpu0_memory_mb"]) / 1024,
                        "gpu1": float(raw["gpu1_memory_mb"]) / 1024,
                        "env": float(raw["env_rss_mb"]) / 1024,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        raise RuntimeError("no resource rows")
    return rows


def parse_peak() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / "peak.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def rolling(values: list[float], width: int) -> list[float]:
    return [sum(values[max(0, i - width + 1) : i + 1]) / len(values[max(0, i - width + 1) : i + 1]) for i in range(len(values))]


metrics = parse_metrics()
resources = parse_resources()
peak = parse_peak()
steps = [int(r["step"]) for r in metrics]
success = [float(r["success"]) * 100 for r in metrics]
success_ma = rolling(success, 5)
start = resources[0]["timestamp"]
hours = [(r["timestamp"] - start).total_seconds() / 3600 for r in resources]  # type: ignore[operator]
checkpoints = [s for s in (10, 20, 30) if s <= steps[-1]]

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 160})


def checkpoint_lines(ax: plt.Axes) -> None:
    for step in checkpoints:
        ax.axvline(step, color="#777777", linewidth=0.9, linestyle=":")
        ax.text(step, ax.get_ylim()[1], f" ckpt {step}", va="top", fontsize=8)


# Success: full measured history, axis begins at zero without inventing a step-0 value.
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(steps, success, marker="o", markersize=3.5, linewidth=1.2, alpha=0.72, label="training rollout")
ax.plot(steps, success_ma, linewidth=2.4, label="5-step trailing mean")
ax.set_xlim(0, steps[-1])
ax.set_ylim(0, 70)
ax.set_xlabel("RL step (first measured point: step 1)")
ax.set_ylabel("Success (%)")
ax.set_title(f"Move Stapler Pad: training-rollout success through step {steps[-1]}")
checkpoint_lines(ax)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(MOBILE / f"success-full-history-step{steps[-1]}.png", bbox_inches="tight")
plt.close(fig)

# Optimizer diagnostics.
fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True)
axes[0].plot(steps, [float(r["kl"]) for r in metrics], label="approx KL")
axes[0].plot(steps, [float(r["clip"]) for r in metrics], label="clip fraction")
axes[0].plot(steps, [float(r["ratio_abs"]) for r in metrics], label="ratio abs")
axes[0].axhline(0, color="#777777", linewidth=0.8)
axes[0].set_ylabel("Probability diagnostics")
axes[0].legend(loc="upper right", ncol=3)
axes[1].plot(steps, [float(r["grad"]) for r in metrics], label="grad norm")
loss_axis = axes[1].twinx()
loss_axis.plot(steps, [float(r["loss_abs"]) for r in metrics], color="#d97706", label="|policy loss|")
axes[1].set_ylabel("Grad norm")
loss_axis.set_ylabel("|Policy loss|")
axes[1].set_xlabel("RL step (first measured point: step 1)")
lines = axes[1].get_lines() + loss_axis.get_lines()
axes[1].legend(lines, [line.get_label() for line in lines], loc="upper right")
for ax in axes:
    ax.set_xlim(0, steps[-1])
    checkpoint_lines(ax)
fig.suptitle(f"GRPO diagnostics through step {steps[-1]}")
fig.tight_layout()
fig.savefig(MOBILE / f"grpo-full-history-step{steps[-1]}.png", bbox_inches="tight")
plt.close(fig)

# Resources: full run.
fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True)
axes[0].plot(hours, [float(r["ram"]) for r in resources], linewidth=1.2)
axes[0].axhline(100, color="#777777", linewidth=0.9, linestyle=":")
axes[0].axhline(95, color="#999999", linewidth=0.8, linestyle="--")
axes[0].set_ylim(40, 101)
axes[0].set_ylabel("cgroup RAM (%)")
axes[1].plot(hours, [float(r["gpu0"]) for r in resources], label="GPU0")
axes[1].plot(hours, [float(r["gpu1"]) for r in resources], label="GPU1")
axes[1].plot(hours, [float(r["env"]) for r in resources], label="Env RSS", linewidth=1)
axes[1].axhline(80, color="#777777", linewidth=0.9, linestyle=":")
axes[1].set_ylim(0, 82)
axes[1].set_ylabel("Memory (GiB)")
axes[1].set_xlabel("Elapsed wall time (hours)")
axes[1].legend(loc="upper right", ncol=3)
fig.suptitle("Resources from launch through the current snapshot")
fig.tight_layout()
fig.savefig(MOBILE / f"resources-full-history-step{steps[-1]}.png", bbox_inches="tight")
plt.close(fig)

payload = {
    "steps": steps,
    "success": [round(v, 4) for v in success],
    "successMa": [round(v, 4) for v in success_ma],
    "checkpointSuccess": [{"x": s, "y": round(success[steps.index(s)], 4)} for s in checkpoints],
    "kl": [float(r["kl"]) for r in metrics],
    "clip": [float(r["clip"]) for r in metrics],
    "ratioAbs": [float(r["ratio_abs"]) for r in metrics],
    "grad": [float(r["grad"]) for r in metrics],
    "lossAbs": [float(r["loss_abs"]) for r in metrics],
    "hours": [round(v, 3) for v in hours],
    "ram": [round(float(r["ram"]), 3) for r in resources],
    "gpu0": [round(float(r["gpu0"]), 3) for r in resources],
    "gpu1": [round(float(r["gpu1"]), 3) for r in resources],
    "env": [round(float(r["env"]), 3) for r in resources],
}

last5 = sum(success[-5:]) / len(success[-5:])
current_ram = float(resources[-1]["ram"])
peak_ram = float(peak["peak_ram_mb"]) / float(peak["cgroup_limit_mb"]) * 100
analysis = {
    "latest_step": steps[-1],
    "latest_success_pct": success[-1],
    "last5_success_pct": last5,
    "first10_success_pct": sum(success[:10]) / 10,
    "last10_success_pct": sum(success[-10:]) / 10,
    "current_ram_pct": current_ram,
    "peak_ram_pct": peak_ram,
    "peak_gpu0_gib": float(peak["peak_gpu0_mb"]) / 1024,
    "peak_gpu1_gib": float(peak["peak_gpu1_mb"]) / 1024,
    "process_alive": peak.get("process_alive"),
}
(ROOT / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

html = f'''<div id="fastwamMoveStaplerStep34">
  <div class="viz-grid">
    <div class="card viz-stat"><span class="text-muted">完成进度</span><span class="viz-stat-value">{steps[-1]} / 100</span><span class="text-small text-muted">21:30 CST；下一步运行中</span></div>
    <div class="card viz-stat"><span class="text-muted">最近 5 步成功率</span><span class="viz-stat-value">{last5:.2f}%</span><span class="text-small text-muted">训练 rollout，非固定 seed eval</span></div>
    <div class="card viz-stat"><span class="text-muted">RAM 当前 / 峰值</span><span class="viz-stat-value">{current_ram:.1f}% / {peak_ram:.1f}%</span><span class="text-small text-muted">OOM / OOM kill 均为 0</span></div>
  </div>
  <section><h3>成功率：完整已完成 step</h3><div class="fw-chart"><canvas data-chart="success" aria-label="step 1 到 {steps[-1]} 的训练 rollout 成功率和五步移动均值"></canvas></div></section>
  <section><h3>GRPO 更新指标</h3><div class="fw-chart"><canvas data-chart="optimizer" aria-label="step 1 到 {steps[-1]} 的 KL、clip、ratio abs、梯度范数与策略损失"></canvas></div></section>
  <section><h3>资源：从启动到当前</h3><div class="fw-chart"><canvas data-chart="resources" aria-label="完整运行期间的 cgroup RAM、两卡显存与环境进程内存"></canvas></div></section>
</div>
<style>
  #fastwamMoveStaplerStep34 {{ display:grid; gap:16px; color:var(--foreground); }}
  #fastwamMoveStaplerStep34 section {{ min-width:0; }}
  #fastwamMoveStaplerStep34 h3 {{ margin:0 0 6px; }}
  #fastwamMoveStaplerStep34 .fw-chart {{ position:relative; width:100%; min-width:0; height:260px; }}
  @media (max-width:480px) {{ #fastwamMoveStaplerStep34 .fw-chart {{ height:230px; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('fastwamMoveStaplerStep34');
  const d = {json.dumps(payload, ensure_ascii=False, separators=(",", ":"))};
  const css = getComputedStyle(document.documentElement);
  const series = n => css.getPropertyValue(`--viz-series-${{n}}`).trim();
  const fg = css.getPropertyValue('--foreground').trim();
  const muted = css.getPropertyValue('--muted-foreground').trim();
  const border = css.getPropertyValue('--border').trim();
  const points = (xs, ys) => xs.map((x, i) => ({{x:x, y:ys[i]}}));
  const common = {{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{labels:{{color:fg}}}},tooltip:{{enabled:true}}}}}};
  const xStep = {{type:'linear',min:0,max:{steps[-1]},ticks:{{color:muted}},grid:{{color:border}},title:{{display:true,text:'RL step',color:muted}}}};
  new Chart(root.querySelector('[data-chart="success"]'), {{type:'line',data:{{datasets:[
    {{label:'training rollout',data:points(d.steps,d.success),borderColor:series(1),backgroundColor:series(1),pointRadius:3,tension:.12}},
    {{label:'5-step mean',data:points(d.steps,d.successMa),borderColor:series(2),backgroundColor:series(2),pointRadius:0,borderDash:[5,4],tension:.2}},
    {{label:'checkpoint',data:d.checkpointSuccess,borderColor:series(3),backgroundColor:series(3),showLine:false,pointRadius:6,pointStyle:'triangle'}}
  ]}},options:{{...common,scales:{{x:xStep,y:{{min:0,max:70,ticks:{{color:muted}},grid:{{color:border}},title:{{display:true,text:'Success (%)',color:muted}}}}}}}}}});
  new Chart(root.querySelector('[data-chart="optimizer"]'), {{type:'line',data:{{datasets:[
    {{label:'approx KL',data:points(d.steps,d.kl),borderColor:series(1),backgroundColor:series(1),pointRadius:2,yAxisID:'y'}},
    {{label:'clip fraction',data:points(d.steps,d.clip),borderColor:series(2),backgroundColor:series(2),pointRadius:2,yAxisID:'y'}},
    {{label:'ratio abs',data:points(d.steps,d.ratioAbs),borderColor:series(3),backgroundColor:series(3),pointRadius:2,yAxisID:'y'}},
    {{label:'grad norm',data:points(d.steps,d.grad),borderColor:series(4),backgroundColor:series(4),pointRadius:1,yAxisID:'y1'}},
    {{label:'|policy loss|',data:points(d.steps,d.lossAbs),borderColor:series(5),backgroundColor:series(5),pointRadius:1,yAxisID:'y2'}}
  ]}},options:{{...common,scales:{{x:xStep,y:{{position:'left',ticks:{{color:muted}},grid:{{color:border}},title:{{display:true,text:'Probability diagnostics',color:muted}}}},y1:{{position:'right',ticks:{{color:muted}},grid:{{drawOnChartArea:false}},title:{{display:true,text:'Grad norm',color:muted}}}},y2:{{display:false,min:0,max:1}}}}}}}});
  new Chart(root.querySelector('[data-chart="resources"]'), {{type:'line',data:{{datasets:[
    {{label:'cgroup RAM %',data:points(d.hours,d.ram),borderColor:series(1),backgroundColor:series(1),pointRadius:0,yAxisID:'y'}},
    {{label:'GPU0 GiB',data:points(d.hours,d.gpu0),borderColor:series(2),backgroundColor:series(2),pointRadius:0,yAxisID:'y1'}},
    {{label:'GPU1 GiB',data:points(d.hours,d.gpu1),borderColor:series(3),backgroundColor:series(3),pointRadius:0,yAxisID:'y1'}},
    {{label:'Env RSS GiB',data:points(d.hours,d.env),borderColor:series(4),backgroundColor:series(4),pointRadius:0,yAxisID:'y1'}}
  ]}},options:{{...common,scales:{{x:{{type:'linear',min:0,ticks:{{color:muted}},grid:{{color:border}},title:{{display:true,text:'Elapsed hours',color:muted}}}},y:{{position:'left',min:40,max:100,ticks:{{color:muted}},grid:{{color:border}},title:{{display:true,text:'RAM (%)',color:muted}}}},y1:{{position:'right',min:0,max:82,ticks:{{color:muted}},grid:{{drawOnChartArea:false}},title:{{display:true,text:'Memory (GiB)',color:muted}}}}}}}}}});
}})();
</script>
'''
(VIS / f"fastwam-move-stapler-step{steps[-1]}.html").write_text(html, encoding="utf-8")

print(json.dumps(analysis, ensure_ascii=False, indent=2))
print(VIS / f"fastwam-move-stapler-step{steps[-1]}.html")
for path in sorted(MOBILE.glob("*.png")):
    print(path)
