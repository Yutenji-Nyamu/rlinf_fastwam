"""Render a concise high-contrast Control-vs-Prism live dashboard as SVG."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(r"C:\Users\86136\Documents\rl")
EVIDENCE = ROOT / "docs" / "rlinf-shenzhen-prism-dvac-grpo" / "evidence" / "control-prism-localshard-live-20260828"
CONTROL, CONTROL_LIGHT, PRISM = "#00796B", "#76AFA7", "#E69F00"
INK, GRID, BG = "#183044", "#D9E2E8", "#F6F8FA"


def moving(values: list[float], window: int) -> list[float]:
    return [mean(values[max(0, i - window + 1): i + 1]) for i in range(len(values))]


def main() -> None:
    payload = json.loads((EVIDENCE / "series.json").read_text(encoding="utf-8"))
    series: dict[str, dict[str, list[float] | list[int]]] = {}
    for label in ("control", "prism"):
        train = payload[label]["train_success"]
        steps = [int(x["step"]) for x in train]
        raw = [float(x["value"]) * 100 for x in train]
        evals = payload[label]["eval_success"]
        series[label] = {
            "steps": steps, "raw": raw, "ma5": moving(raw, 5), "ma10": moving(raw, 10),
            "eval_steps": [int(x["step"]) for x in evals], "eval": [float(x["value"]) * 100 for x in evals],
        }
        with (EVIDENCE / f"{label}_success.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(["step", "raw_pct", "ma5_pct", "ma10_pct"])
            writer.writerows(zip(steps, raw, series[label]["ma5"], series[label]["ma10"]))

    c, p = series["control"], series["prism"]
    paired = min(c["steps"][-1], p["steps"][-1])
    c_by = {s: i for i, s in enumerate(c["steps"])}; p_by = {s: i for i, s in enumerate(p["steps"])}
    paired_rows = []
    for step in sorted(set(c_by) & set(p_by)):
        if step <= paired:
            ci, pi = c_by[step], p_by[step]
            paired_rows.append({"step": step, "raw_diff_pp": p["raw"][pi] - c["raw"][ci],
                                "ma5_diff_pp": p["ma5"][pi] - c["ma5"][ci],
                                "ma10_diff_pp": p["ma10"][pi] - c["ma10"][ci]})
    summary = {
        "control_complete_step": c["steps"][-1], "prism_complete_step": p["steps"][-1], "paired_through": paired,
        "control_latest": {k: round(c[k][-1], 3) for k in ("raw", "ma5", "ma10")},
        "prism_latest": {k: round(p[k][-1], 3) for k in ("raw", "ma5", "ma10")},
        "paired_mean_raw_diff_pp": round(mean(x["raw_diff_pp"] for x in paired_rows), 3),
        "paired_latest_diff_pp": {k: round(paired_rows[-1][k], 3) for k in ("raw_diff_pp", "ma5_diff_pp", "ma10_diff_pp")},
        "control_eval_successes": [round(x * 32 / 100) for x in c["eval"]],
        "prism_eval_successes": [round(x * 32 / 100) for x in p["eval"]],
    }
    (EVIDENCE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    width, height = 1500, 1900
    left, right, panel_h = 120, 1450, 330
    panel_tops = [180, 590, 1000, 1410]
    max_step = max(c["steps"][-1], p["steps"][-1])
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           f'<rect width="100%" height="100%" fill="{BG}"/>',
           f'<text x="70" y="65" font-family="Segoe UI,Arial" font-size="34" font-weight="700" fill="{INK}">SZ-H100 matched 2-GPU runs: Control Step {c["steps"][-1]} vs Prism Step {p["steps"][-1]}</text>',
           f'<text x="70" y="108" font-family="Segoe UI,Arial" font-size="20" fill="#52697B">Orange = Prism-DVAC local-shard v2; teal = GRPO Control; dotted cutoff = paired data through Step {paired}</text>']

    def xcoord(step: float) -> float:
        return left + (step - 1) / max(1, max_step - 1) * (right - left)

    def draw_panel(top: int, title: str, key: str, ymin: float, ymax: float, eval_panel: bool = False) -> None:
        plot_top, plot_bottom = top + 50, top + panel_h - 42
        def ycoord(value: float) -> float: return plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
        svg.append(f'<rect x="60" y="{top}" width="1400" height="{panel_h}" rx="18" fill="white" stroke="#D5DFE6"/>')
        svg.append(f'<text x="90" y="{top+36}" font-family="Segoe UI,Arial" font-size="24" font-weight="700" fill="{INK}">{title}</text>')
        ticks = [ymin + i * (ymax - ymin) / 4 for i in range(5)]
        for tick in ticks:
            y = ycoord(tick); svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}"/>')
            svg.append(f'<text x="{left-18}" y="{y+6:.1f}" text-anchor="end" font-family="Segoe UI,Arial" font-size="16" fill="#607789">{tick:.0f}</text>')
        for step in range(10, max_step + 1, 10):
            x = xcoord(step); svg.append(f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_bottom}" stroke="{GRID}" opacity=".55"/>')
            svg.append(f'<text x="{x:.1f}" y="{plot_bottom+25}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="15" fill="#607789">{step}</text>')
        cutoff_x = xcoord(paired); svg.append(f'<line x1="{cutoff_x:.1f}" y1="{plot_top}" x2="{cutoff_x:.1f}" y2="{plot_bottom}" stroke="#6C7780" stroke-dasharray="6 7" stroke-width="2"/>')
        if eval_panel:
            for label, color, marker in (("control", CONTROL, "circle"), ("prism", PRISM, "square")):
                s = series[label]; pts = [(xcoord(x), ycoord(y)) for x, y in zip(s["eval_steps"], s["eval"])]
                svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="4"/>')
                for (x, y), value in zip(pts, s["eval"]):
                    if marker == "circle": svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
                    else: svg.append(f'<rect x="{x-6:.1f}" y="{y-6:.1f}" width="12" height="12" fill="{color}"/>')
                    if label == "prism": svg.append(f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="15" font-weight="700" fill="{color}">{round(value*32/100)}/32</text>')
        else:
            cpts = [(xcoord(x), ycoord(y)) for x, y in zip(c["steps"], c[key])]
            solid = [(x,y) for (x,y),step in zip(cpts,c["steps"]) if step <= paired]
            tail = [(x,y) for (x,y),step in zip(cpts,c["steps"]) if step >= paired]
            svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in solid)}" fill="none" stroke="{CONTROL}" stroke-width="4"/>')
            svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in tail)}" fill="none" stroke="{CONTROL_LIGHT}" stroke-width="3" stroke-dasharray="9 7"/>')
            ppts = [(xcoord(x), ycoord(y)) for x, y in zip(p["steps"], p[key])]
            svg.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in ppts)}" fill="none" stroke="{PRISM}" stroke-width="4.5"/>')
            for x,y in ppts: svg.append(f'<rect x="{x-3.5:.1f}" y="{y-3.5:.1f}" width="7" height="7" fill="{PRISM}"/>')
        svg.append(f'<text x="{(left+right)/2:.1f}" y="{top+panel_h-8}" text-anchor="middle" font-family="Segoe UI,Arial" font-size="17" fill="#607789">Completed global step</text>')

    draw_panel(panel_tops[0], "Per-step training-rollout success", "raw", 60, 100)
    draw_panel(panel_tops[1], "Trailing 5-step moving mean", "ma5", 60, 100)
    draw_panel(panel_tops[2], "Trailing 10-step moving mean", "ma10", 60, 100)
    draw_panel(panel_tops[3], "Fixed-32 evaluation", "eval", 75, 100, True)
    svg += [f'<circle cx="1060" cy="145" r="7" fill="{CONTROL}"/><text x="1075" y="152" font-family="Segoe UI,Arial" font-size="17" fill="{INK}">GRPO Control</text>',
            f'<rect x="1240" y="138" width="14" height="14" fill="{PRISM}"/><text x="1265" y="152" font-family="Segoe UI,Arial" font-size="17" fill="{INK}">Prism-DVAC</text>',
            '</svg>']
    out = EVIDENCE / "01_control_vs_prism_success_live.svg"
    out.write_text("\n".join(svg), encoding="utf-8")
    print(json.dumps(summary, indent=2)); print(out)


if __name__ == "__main__":
    main()
