from __future__ import annotations

import csv
import importlib.util
import json
import statistics
from pathlib import Path

from PIL import Image


ROOT = Path(r"C:\Users\86136\Documents\rl")
TOPIC = ROOT / "docs" / "rlinf-shenzhen-grpo-dvac-action-adv"
OUT = TOPIC / "evidence" / "eight-2gpu-grpo-separated-figures-live-20260831"
BASE_SCRIPT = ROOT / "local_scripts" / "render_shenzhen_grpo_vs_ppo_20260823.py"
STYLE_SCRIPT = ROOT / "local_scripts" / "render_sz_six_2gpu_grpo_comparison_20260829.py"

# All entries below are Shenzhen two-GPU formal runs with the same scientific
# budget: 64 train env x 4 rollout epochs, G8, B1024/MB32/update2,
# fixed32/eval5. Failed reruns, smokes, four-GPU and AutoDL runs are excluded.
RUNS = {
    "control": {
        "name": "GRPO Control",
        "color": "#111827",
        "pattern": None,
        "dash": "solid",
        "marker": "circle",
        "symbol": "circle",
        "path": TOPIC
        / "evidence/grpo-control-2gpu-stopped-step96-20260828/raw/runtime/driver.log",
    },
    "st_02": {
        "name": "ST-DVAC [0,2]",
        "color": "#E69F00",
        "pattern": (18, 8),
        "dash": "dash",
        "marker": "square",
        "symbol": "square",
        "path": ROOT
        / "docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/shenzhen_grpo_dvac_w0to2_2gpu_stopped_step52_light_evidence_20260827/runtime/driver.log",
    },
    "prism": {
        "name": "Prism-RLOO",
        "color": "#009E73",
        "pattern": (7, 6),
        "dash": "dot",
        "marker": "triangle",
        "symbol": "triangle-up",
        "path": TOPIC
        / "evidence/action-adv-prism-stopped-20260828/raw/prism/runtime/driver.log",
    },
    "action_old": {
        "name": "Action-Adv old (H-mean)",
        "color": "#6B7280",
        "pattern": (18, 6, 4, 6),
        "dash": "dashdot",
        "marker": "cross",
        "symbol": "x",
        "path": TOPIC
        / "evidence/action-adv-prism-stopped-20260828/raw/action_adv/runtime/driver.log",
    },
    "action_fix_02": {
        "name": "Action-Adv Fix [0,2]",
        "color": "#CC79A7",
        "pattern": None,
        "dash": "solid",
        "marker": "diamond",
        "symbol": "diamond",
        "path": TOPIC
        / "evidence/action-fix-st-half-stopped-20260830/raw/action_fix/runtime/driver.log",
    },
    "st_half": {
        "name": "ST-DVAC [0.5,1.5]",
        "color": "#0072B2",
        "pattern": (12, 7),
        "dash": "dash",
        "marker": "star",
        "symbol": "star",
        "path": TOPIC
        / "evidence/action-fix-st-half-stopped-20260830/raw/st_half/runtime/driver.log",
    },
    "action_mid": {
        "name": "Action-Adv Fix [0.5,1.5]",
        "color": "#D55E00",
        "pattern": (22, 6, 5, 6),
        "dash": "longdashdot",
        "marker": "triangle",
        "symbol": "triangle-down",
        "path": TOPIC
        / "evidence/current-action-mid-st-narrow-live-20260831-1043/raw/action_mid/runtime/driver.log",
    },
    "st_narrow": {
        "name": "ST-DVAC [0.8,1.2]",
        "color": "#00796B",
        "pattern": (5, 5),
        "dash": "dot",
        "marker": "square",
        "symbol": "cross",
        "path": TOPIC
        / "evidence/current-action-mid-st-narrow-live-20260831-1043/raw/st_narrow/runtime/driver.log",
    },
}

FIGURES = {
    "raw": ("01_raw_success.png", "Per-step training-rollout success", 0.64),
    "ma5": ("02_ma5_success.png", "Trailing 5-step mean success", 0.68),
    "ma10": ("03_ma10_success.png", "Trailing 10-step mean success", 0.70),
    "fixed": ("04_fixed32_success.png", "Fixed-32 evaluation success", 0.68),
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw_legend(draw, base, style, data: dict[str, dict]) -> None:
    x0, y0 = 64, 124
    col_w, row_h = 420, 42
    legend_font = base.font(17, True)
    for idx, (key, meta) in enumerate(RUNS.items()):
        row, col = divmod(idx, 4)
        x, y = x0 + col * col_w, y0 + row * row_h
        style.draw_patterned_line(
            draw,
            [(x, y + 12), (x + 49, y + 12)],
            fill=meta["color"],
            width=5,
            pattern=meta["pattern"],
        )
        style.draw_marker(draw, x + 25, y + 12, meta["color"], meta["marker"], 5)
        label = f"{meta['name']}  (S{data[key]['steps'][-1]})"
        draw.text((x + 60, y), label, fill=meta["color"], font=legend_font)


def write_interactive(data: dict[str, dict], maximum: int) -> None:
    payload = {
        key: {
            "name": RUNS[key]["name"],
            "color": RUNS[key]["color"],
            "dash": RUNS[key]["dash"],
            "symbol": RUNS[key]["symbol"],
            **values,
        }
        for key, values in data.items()
    }
    html = f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Eight matched two-GPU GRPO runs</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#f4f7fb;color:#142033;margin:24px}}
.note{{color:#64748b;margin-bottom:18px}} .chart{{height:620px;background:white;margin:18px 0;border-radius:14px}}</style>
</head><body><h1>Eight matched two-GPU GRPO formal runs</h1>
<div class=\"note\">Common budget: 64 train env x 4 rollout epochs, G8, B1024/MB32/update2, fixed32/eval5. Each series ends at its real final/latest step.</div>
<div id=\"raw\" class=\"chart\"></div><div id=\"ma5\" class=\"chart\"></div>
<div id=\"ma10\" class=\"chart\"></div><div id=\"fixed\" class=\"chart\"></div>
<script>
const runs={json.dumps(payload, ensure_ascii=False)};
const titles={{raw:'Per-step training-rollout success',ma5:'Trailing 5-step mean success',ma10:'Trailing 10-step mean success',fixed:'Fixed-32 evaluation success'}};
for (const metric of ['raw','ma5','ma10','fixed']) {{
  const traces=Object.values(runs).map(r=>({{x:r.steps,y:r[metric],name:r.name,mode:metric==='fixed'?'lines+markers':'lines',connectgaps:false,
    line:{{color:r.color,width:3,dash:r.dash}},marker:{{color:r.color,size:8,symbol:r.symbol}},hovertemplate:'Step %{{x}}<br>%{{y:.2%}}<extra>%{{fullData.name}}</extra>'}}));
  Plotly.newPlot(metric,traces,{{title:titles[metric],paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',hovermode:'x unified',
    legend:{{orientation:'h',y:1.14}},xaxis:{{title:'Completed global step',range:[1,{maximum}],gridcolor:'#d8e0ea'}},
    yaxis:{{title:'Success rate',tickformat:'.0%',range:[metric==='raw'?0.64:(metric==='ma10'?0.70:0.68),1.005],gridcolor:'#d8e0ea'}},margin:{{l:75,r:35,t:110,b:70}}}},{{responsive:true,displaylogo:false}});
}}
</script></body></html>"""
    (OUT / "interactive.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_module(BASE_SCRIPT, "plot_base_eight")
    style = load_module(STYLE_SCRIPT, "plot_style_eight")
    style.RUNS = RUNS

    data: dict[str, dict] = {}
    for key, meta in RUNS.items():
        if not meta["path"].is_file():
            raise FileNotFoundError(meta["path"])
        rows = base.parse_grpo(meta["path"])
        raw = [float(row["train_success"]) for row in rows]
        data[key] = {
            "steps": [int(row["step"]) for row in rows],
            "raw": raw,
            "ma5": base.trailing(raw, 5),
            "ma10": base.trailing(raw, 10),
            "fixed": [
                float(row["eval_success"]) if row["eval_success"] is not None else None
                for row in rows
            ],
        }

    maximum = max(values["steps"][-1] for values in data.values())
    for metric, (filename, title, y_min) in FIGURES.items():
        image = Image.new("RGB", (1760, 1000), base.LIGHT)
        draw = base.add_header(
            image,
            f"Eight matched two-GPU GRPO formal runs — {title}",
            "Same sampling/update/eval budget; short runs stop at their real endpoint (snapshot: 2026-08-31 10:43 CST).",
        )
        draw_legend(draw, base, style, data)
        style.panel(
            draw,
            base,
            (45, 220, 1715, 950),
            data,
            metric,
            title,
            y_min,
            1.005,
            maximum,
        )
        image.save(OUT / filename, optimize=True)

    rows_out: list[dict[str, float | int | None]] = []
    maps = {
        key: {
            metric: dict(zip(values["steps"], values[metric]))
            for metric in FIGURES
        }
        for key, values in data.items()
    }
    for step in range(1, maximum + 1):
        row: dict[str, float | int | None] = {"step": step}
        for key in RUNS:
            for metric in FIGURES:
                row[f"{key}_{metric}"] = maps[key][metric].get(step)
        rows_out.append(row)
    with (OUT / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)

    summary: dict[str, object] = {"maximum_step": maximum, "runs": {}}
    for key, values in data.items():
        raw = values["raw"]
        fixed = [value for value in values["fixed"] if value is not None]
        summary["runs"][key] = {
            "name": RUNS[key]["name"],
            "complete_step": values["steps"][-1],
            "latest_raw_pct": raw[-1] * 100,
            "latest_ma5_pct": statistics.mean(raw[-5:]) * 100,
            "latest_ma10_pct": statistics.mean(raw[-10:]) * 100,
            "fixed32_successes": sum(round(value * 32) for value in fixed),
            "fixed32_episodes": len(fixed) * 32,
            "source_log": str(RUNS[key]["path"].relative_to(ROOT)),
        }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_interactive(data, maximum)
    print(json.dumps({"out": str(OUT), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
