from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
ANALYSIS = json.loads((ROOT / "analysis.json").read_text(encoding="utf-8"))
METRICS = ANALYSIS["metrics"]
VIS_DIR = Path(
    r"C:\Users\86136\.codex\visualizations\2026\07\16"
    r"\019f69a4-b29a-77f1-8ee5-76ebd7e4aa49"
)
MOBILE_DIR = ROOT / "mobile-figures"
VIS_DIR.mkdir(parents=True, exist_ok=True)
MOBILE_DIR.mkdir(parents=True, exist_ok=True)


def rolling(values: list[float], width: int = 5) -> list[float]:
    return [
        sum(values[max(0, i - width + 1) : i + 1])
        / len(values[max(0, i - width + 1) : i + 1])
        for i in range(len(values))
    ]


def load_resources() -> list[dict[str, float | dt.datetime]]:
    rows: list[dict[str, float | dt.datetime]] = []
    with (ROOT / "resources.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "timestamp": dt.datetime.strptime(
                            raw["timestamp"], "%Y-%m-%d %H:%M:%S"
                        ),
                        "ram_pct": float(raw["cgroup_ram_pct"]),
                        "gpu0_gib": float(raw["gpu0_memory_mb"]) / 1024,
                        "gpu1_gib": float(raw["gpu1_memory_mb"]) / 1024,
                        "env_gib": float(raw["env_rss_mb"]) / 1024,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


RESOURCES = load_resources()
STEPS = [int(row["step"]) for row in METRICS]
SUCCESS = [float(row["success_once"]) for row in METRICS]
SUCCESS_MA = rolling(SUCCESS)


def checkpoint_lines(ax: plt.Axes) -> None:
    for step in (10, 20):
        if step <= STEPS[-1]:
            ax.axvline(step, color="#888888", linewidth=1, linestyle=":")
            ax.text(step, ax.get_ylim()[1], f" ckpt {step}", va="top", fontsize=9)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 150,
    }
)

# Mobile success plot.
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(STEPS, [v * 100 for v in SUCCESS], marker="o", linewidth=1.5, label="step")
ax.plot(STEPS, [v * 100 for v in SUCCESS_MA], linewidth=2.4, label="5-step mean")
ax.set_xlim(0, STEPS[-1])
ax.set_ylim(90, 100.5)
ax.set_xlabel("RL step (axis starts at 0; first measurement is step 1)")
ax.set_ylabel("Training rollout success_once (%)")
ax.set_title("Fast-WAM GRPO training success — full completed history")
checkpoint_lines(ax)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(MOBILE_DIR / "success-full-history-step23.png", bbox_inches="tight")
plt.close(fig)

# Mobile optimizer plot.
fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True)
axes[0].plot(STEPS, [float(r.get("actor/approx_kl", 0)) for r in METRICS], label="approx KL")
axes[0].plot(STEPS, [float(r.get("actor/clip_fraction", 0)) for r in METRICS], label="clip fraction")
axes[0].plot(STEPS, [float(r.get("actor/ratio_abs", 0)) for r in METRICS], label="ratio abs")
axes[0].axhline(0, color="#888888", linewidth=0.8)
axes[0].set_ylabel("Probability diagnostics")
axes[0].legend(loc="upper left", ncol=3)
axes[1].plot(STEPS, [float(r.get("actor/grad_norm", 0)) for r in METRICS], label="grad norm")
axes[1].plot(STEPS, [float(r.get("actor/policy_loss_abs", 0)) * 100 for r in METRICS], label="100 × |policy loss|")
axes[1].set_ylabel("Update magnitude")
axes[1].set_xlabel("RL step (axis starts at 0; first measurement is step 1)")
axes[1].legend(loc="upper left")
for ax in axes:
    ax.set_xlim(0, STEPS[-1])
    checkpoint_lines(ax)
fig.suptitle("Fast-WAM GRPO optimizer diagnostics — full completed history")
fig.tight_layout()
fig.savefig(MOBILE_DIR / "grpo-full-history-step23.png", bbox_inches="tight")
plt.close(fig)

# Mobile resource plot.
start = RESOURCES[0]["timestamp"]
hours = [
    (row["timestamp"] - start).total_seconds() / 3600  # type: ignore[operator]
    for row in RESOURCES
]
fig, axes = plt.subplots(2, 1, figsize=(8, 7.2), sharex=True)
axes[0].plot(hours, [float(r["ram_pct"]) for r in RESOURCES], linewidth=1.1)
axes[0].axhline(100, color="#888888", linewidth=1, linestyle=":")
axes[0].set_ylim(50, 101)
axes[0].set_ylabel("cgroup RAM (%)")
axes[1].plot(hours, [float(r["gpu0_gib"]) for r in RESOURCES], label="GPU0")
axes[1].plot(hours, [float(r["gpu1_gib"]) for r in RESOURCES], label="GPU1")
axes[1].plot(hours, [float(r["env_gib"]) for r in RESOURCES], label="Env RSS", linewidth=1)
axes[1].axhline(80, color="#888888", linewidth=1, linestyle=":")
axes[1].set_ylim(0, 82)
axes[1].set_ylabel("Memory (GiB)")
axes[1].set_xlabel("Elapsed wall time (hours)")
axes[1].legend(loc="upper right", ncol=3)
fig.suptitle("Fast-WAM GRPO resources — full run from launch to 08:43 CST")
fig.tight_layout()
fig.savefig(MOBILE_DIR / "resources-full-history-step23.png", bbox_inches="tight")
plt.close(fig)

# Downsample resource data for the inline view.
stride = max(1, math.ceil(len(RESOURCES) / 480))
sampled = RESOURCES[::stride]
if sampled[-1] is not RESOURCES[-1]:
    sampled.append(RESOURCES[-1])
sampled_hours = [
    round((row["timestamp"] - start).total_seconds() / 3600, 3)  # type: ignore[operator]
    for row in sampled
]

payload = {
    "steps": STEPS,
    "success": [round(v * 100, 4) for v in SUCCESS],
    "successMa": [round(v * 100, 4) for v in SUCCESS_MA],
    "kl": [float(r.get("actor/approx_kl", 0)) for r in METRICS],
    "clip": [float(r.get("actor/clip_fraction", 0)) for r in METRICS],
    "ratioAbs": [float(r.get("actor/ratio_abs", 0)) for r in METRICS],
    "grad": [float(r.get("actor/grad_norm", 0)) for r in METRICS],
    "hours": sampled_hours,
    "ramPct": [round(float(r["ram_pct"]), 3) for r in sampled],
    "gpu0": [round(float(r["gpu0_gib"]), 3) for r in sampled],
    "gpu1": [round(float(r["gpu1_gib"]), 3) for r in sampled],
    "env": [round(float(r["env_gib"]), 3) for r in sampled],
}

last5 = sum(SUCCESS[-5:]) / len(SUCCESS[-5:]) * 100
peak_ram = float(ANALYSIS["peak"]["peak_ram_mb"]) / 245760 * 100
html = f'''<div id="fastwam-training-snapshot">
  <div class="viz-grid">
    <div class="card viz-stat"><span class="text-muted">完成进度</span><span class="viz-stat-value">23 / 100</span><span class="text-small text-muted">08:43 CST；step 24 rollout 中</span></div>
    <div class="card viz-stat"><span class="text-muted">最近 5 步成功率</span><span class="viz-stat-value">{last5:.2f}%</span><span class="text-small text-muted">训练 rollout；非固定种子 eval</span></div>
    <div class="card viz-stat"><span class="text-muted">cgroup RAM 峰值</span><span class="viz-stat-value">{peak_ram:.2f}%</span><span class="text-small text-muted">无 OOM / OOM kill</span></div>
  </div>
  <section><h3>训练 rollout 成功率</h3><div class="fw-chart"><canvas data-chart="success" aria-label="step 1 到 23 的训练 rollout success_once 与五步均值"></canvas></div></section>
  <section><h3>GRPO 更新强度</h3><div class="fw-chart"><canvas data-chart="grpo" aria-label="step 1 到 23 的 KL、clip、ratio abs 与梯度范数"></canvas></div></section>
  <section><h3>完整运行资源</h3><div class="fw-chart"><canvas data-chart="resources" aria-label="启动到 08:43 的 cgroup RAM、两卡显存和 Env RSS"></canvas></div></section>
</div>
<style>
  #fastwam-training-snapshot {{ display:grid; gap:16px; color:var(--foreground); }}
  #fastwam-training-snapshot section {{ min-width:0; }}
  #fastwam-training-snapshot h3 {{ margin:0 0 6px; }}
  #fastwam-training-snapshot .fw-chart {{ position:relative; width:100%; min-width:0; height:260px; }}
  @media (max-width:480px) {{ #fastwam-training-snapshot .fw-chart {{ height:230px; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('fastwam-training-snapshot');
  const d = {json.dumps(payload, ensure_ascii=False, separators=(",", ":"))};
  const css = getComputedStyle(document.documentElement);
  const color = n => css.getPropertyValue(`--viz-series-${{n}}`).trim();
  const fg = css.getPropertyValue('--foreground').trim();
  const muted = css.getPropertyValue('--muted-foreground').trim();
  const border = css.getPropertyValue('--border').trim();
  const base = {{responsive:true, maintainAspectRatio:false, interaction:{{mode:'index',intersect:false}}, plugins:{{legend:{{labels:{{color:fg}}}},tooltip:{{enabled:true}}}}, scales:{{x:{{type:'linear',min:0,ticks:{{color:muted}},grid:{{color:border}},title:{{display:true,text:'RL step',color:muted}}}},y:{{ticks:{{color:muted}},grid:{{color:border}}}}}}}};
  const points = (xs, ys) => xs.map((x,i)=>({{x,y:ys[i]}}));
  new Chart(root.querySelector('[data-chart="success"]'), {{type:'line',data:{{datasets:[{{label:'success_once',data:points(d.steps,d.success),borderColor:color(1),backgroundColor:color(1),pointRadius:3,tension:.15}},{{label:'5-step mean',data:points(d.steps,d.successMa),borderColor:color(2),backgroundColor:color(2),pointRadius:0,borderDash:[5,4],tension:.2}}]}},options:{{...base,scales:{{...base.scales,y:{{...base.scales.y,min:90,max:100.5,title:{{display:true,text:'Success (%)',color:muted}}}}}}}}}});
  new Chart(root.querySelector('[data-chart="grpo"]'), {{type:'line',data:{{datasets:[{{label:'approx KL',data:points(d.steps,d.kl),borderColor:color(1),backgroundColor:color(1),yAxisID:'y'}},{{label:'clip fraction',data:points(d.steps,d.clip),borderColor:color(2),backgroundColor:color(2),yAxisID:'y'}},{{label:'ratio abs',data:points(d.steps,d.ratioAbs),borderColor:color(3),backgroundColor:color(3),yAxisID:'y'}},{{label:'grad norm',data:points(d.steps,d.grad),borderColor:color(4),backgroundColor:color(4),yAxisID:'y1'}}]}},options:{{...base,scales:{{...base.scales,y:{{...base.scales.y,position:'left',title:{{display:true,text:'Probability diagnostics',color:muted}}}},y1:{{position:'right',ticks:{{color:muted}},grid:{{drawOnChartArea:false}},title:{{display:true,text:'Grad norm',color:muted}}}}}}}}}});
  new Chart(root.querySelector('[data-chart="resources"]'), {{type:'line',data:{{datasets:[{{label:'cgroup RAM %',data:points(d.hours,d.ramPct),borderColor:color(1),backgroundColor:color(1),pointRadius:0,yAxisID:'y'}},{{label:'GPU0 GiB',data:points(d.hours,d.gpu0),borderColor:color(2),backgroundColor:color(2),pointRadius:0,yAxisID:'y1'}},{{label:'GPU1 GiB',data:points(d.hours,d.gpu1),borderColor:color(3),backgroundColor:color(3),pointRadius:0,yAxisID:'y1'}},{{label:'Env RSS GiB',data:points(d.hours,d.env),borderColor:color(4),backgroundColor:color(4),pointRadius:0,yAxisID:'y1'}}]}},options:{{...base,scales:{{x:{{...base.scales.x,min:0,title:{{display:true,text:'Elapsed hours',color:muted}}}},y:{{position:'left',min:50,max:100,ticks:{{color:muted}},grid:{{color:border}},title:{{display:true,text:'RAM (%)',color:muted}}}},y1:{{position:'right',min:0,max:82,ticks:{{color:muted}},grid:{{drawOnChartArea:false}},title:{{display:true,text:'Memory (GiB)',color:muted}}}}}}}}}});
}})();
</script>
'''
(VIS_DIR / "fastwam-training-current.html").write_text(html, encoding="utf-8")
print(VIS_DIR / "fastwam-training-current.html")
for path in sorted(MOBILE_DIR.glob("*.png")):
    print(path)
