from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/rlt_dvac_vs_original_live_g350_20260825"
RAW = OUT / "raw"

ORIGINAL_LOGS = [
    ROOT / "exports/rlt_stage2_formal_8env250_high_info_20260730_v1/runtime/driver.log",
    ROOT / "exports/rlt_stage2_formal_resume250_to480_high_info_20260731_v1/driver.log",
]
DVAC_LOG = RAW / "dvac_metrics.log"

STEP_RE = re.compile(r"Global Step:\s*(\d+)/(\d+)")
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


def metric(section: str, name: str) -> float | None:
    match = re.search(rf"(?<![\w/]){re.escape(name)}=({NUMBER})", section, re.I)
    return float(match.group(1)) if match else None


def parse_complete_tables(path: Path) -> list[dict[str, float | int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = text.split("╭")
    rows: list[dict[str, float | int]] = []
    for block in blocks:
        if "Global Step:" not in block or "╰" not in block:
            continue
        step_match = STEP_RE.search(block)
        if not step_match:
            continue
        step, target = map(int, step_match.groups())
        env_part = block.split("Environment", 1)[1]
        if "Evaluation" in env_part:
            train_part, eval_tail = env_part.split("Evaluation", 1)
            eval_part = eval_tail.split("Replay Buffer", 1)[0]
        else:
            train_part, eval_part = env_part.split("Replay Buffer", 1)[0], ""
        time_match = re.search(r"Step Time:\s*([0-9.]+)s", block)
        row: dict[str, float | int] = {
            "cycle": step,
            "target": target,
            "train_success": metric(train_part, "success_once"),
            "train_episodes": int(metric(train_part, "num_trajectories") or 0),
            "step_time_s": float(time_match.group(1)) if time_match else np.nan,
            "eval_success": metric(eval_part, "success_once") if eval_part else np.nan,
            "eval_episodes": int(metric(eval_part, "num_trajectories") or 0),
            "actor_switch_rate": metric(block, "replay/actor_switch_rate"),
            "critic_updates": metric(block, "rlt/critic_updates_run"),
            "actor_updates": metric(block, "rlt/actor_updates_run"),
            "update_step": metric(block, "rlt/update_step"),
            "dvac_weight_mean": metric(block, "rlt_dvac/weight_mean"),
            "dvac_weight_p05": metric(block, "rlt_dvac/weight_p05"),
            "dvac_weight_p95": metric(block, "rlt_dvac/weight_p95"),
            "dvac_weight_ess": metric(block, "rlt_dvac/weight_ess_ratio"),
        }
        rows.append(row)
    # A process can repeat the most recent table during logging; the last complete
    # table for a cycle is authoritative.
    return list({int(row["cycle"]): row for row in rows}.values())


def load_run(paths: list[Path] | Path, method: str) -> pd.DataFrame:
    paths = paths if isinstance(paths, list) else [paths]
    rows: list[dict[str, float | int]] = []
    for path in paths:
        rows.extend(parse_complete_tables(path))
    frame = pd.DataFrame(rows).sort_values("cycle").drop_duplicates("cycle", keep="last")
    frame.insert(0, "method", method)
    frame["success_pct"] = frame["train_success"] * 100
    for window in (5, 10):
        frame[f"success_ma{window}_pct"] = frame["success_pct"].rolling(window, min_periods=1).mean()
    frame["eval_success_pct"] = frame["eval_success"] * 100
    return frame.reset_index(drop=True)


def validate(original: pd.DataFrame, dvac: pd.DataFrame) -> None:
    assert original.cycle.tolist() == list(range(1, 481)), (
        original.cycle.min(), original.cycle.max(), len(original)
    )
    assert dvac.cycle.tolist() == list(range(1, int(dvac.cycle.max()) + 1)), (
        dvac.cycle.min(), dvac.cycle.max(), len(dvac)
    )
    assert set(original.train_episodes.unique()) == {8}
    assert set(dvac.train_episodes.unique()) == {8}
    for frame in (original, dvac):
        evaluated = frame[frame.eval_episodes > 0]
        assert set(evaluated.eval_episodes.unique()) == {20}


def render_plot(original: pd.DataFrame, dvac: pd.DataFrame) -> Path:
    latest = int(dvac.cycle.max())
    original_shared = original[original.cycle <= latest]
    eval_original = original_shared[original_shared.eval_episodes > 0]
    eval_dvac = dvac[dvac.eval_episodes > 0]

    width, height = 1800, 1640
    left, right = 112, 55
    chart_w = width - left - right
    panel_h = 390
    panel_tops = [225, 705, 1185]

    def x(cycle: float) -> float:
        return left + (cycle - 1) / max(1, latest - 1) * chart_w

    def y(value: float, top: float) -> float:
        return top + panel_h - np.clip(value, 0, 100) / 100 * panel_h

    def points(cycles, values, top: float) -> str:
        return " ".join(
            f"{x(float(c)):.1f},{y(float(v), top):.1f}"
            for c, v in zip(cycles, values, strict=True)
            if pd.notna(v)
        )

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        '<style>text{font-family:Arial,Segoe UI,sans-serif;fill:#172033}.title{font-size:32px;font-weight:700}'
        '.subtitle{font-size:17px;fill:#52627a}.paneltitle{font-size:22px;font-weight:700}'
        '.axis{font-size:14px;fill:#53627a}.legend{font-size:15px;fill:#334155}.note{font-size:13px;fill:#64748b}</style>',
        f'<text x="55" y="58" class="title">Original RLT vs RLT teacher-DVAC [0,2] — complete cycles 1–{latest}</text>',
    ]
    last = dvac.iloc[-1]
    common_original = original_shared.train_success.mean() * 100
    common_dvac = dvac.train_success.mean() * 100
    subtitle = (
        f"DVAC latest: raw {last.success_pct:.1f}% · trailing-5 {last.success_ma5_pct:.1f}% · "
        f"trailing-10 {last.success_ma10_pct:.1f}% | shared-cycle mean: original {common_original:.1f}% vs DVAC {common_dvac:.1f}%"
    )
    svg.append(f'<text x="55" y="92" class="subtitle">{escape(subtitle)}</text>')
    svg.append('<text x="55" y="122" class="note">Training points use 8 on-policy episodes; fixed-ID evaluation uses 20 episodes. All curves include complete cycles only.</text>')

    panel_titles = [
        "A. Raw training-rollout success per cycle (8 episodes per point)",
        "B. Identical trailing 5-cycle and 10-cycle means",
        "C. Deterministic fixed-ID evaluation (20 episodes per checkpoint)",
    ]
    for top, title in zip(panel_tops, panel_titles, strict=True):
        svg.append(f'<rect x="45" y="{top-62}" width="1710" height="445" rx="16" fill="white" stroke="#dbe3ef"/>')
        svg.append(f'<text x="65" y="{top-27}" class="paneltitle">{escape(title)}</text>')
        for tick in (0, 25, 50, 75, 100):
            yy = y(tick, top)
            svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+chart_w}" y2="{yy:.1f}" stroke="#dbe3ef" stroke-width="1"/>')
            svg.append(f'<text x="{left-18}" y="{yy+5:.1f}" class="axis" text-anchor="end">{tick}</text>')
        svg.append(f'<text x="35" y="{top+panel_h/2:.1f}" class="axis" transform="rotate(-90 35 {top+panel_h/2:.1f})" text-anchor="middle">success (%)</text>')
        for cycle, label in ((1, "1"), (50, "50"), (100, "100"), (150, "150"), (200, "200"), (250, "250"), (300, "300"), (350, "350")):
            if cycle <= latest:
                xx = x(cycle)
                svg.append(f'<line x1="{xx:.1f}" y1="{top+panel_h}" x2="{xx:.1f}" y2="{top+panel_h+6}" stroke="#64748b"/>')
                svg.append(f'<text x="{xx:.1f}" y="{top+panel_h+24}" class="axis" text-anchor="middle">{label}</text>')
        for cycle, color, dash in ((136, "#8b5cf6", "3 5"), (250.5, "#64748b", "8 6")):
            if cycle <= latest:
                xx = x(cycle)
                svg.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+panel_h}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>')

    # Panel A lines and legend.
    svg.append(f'<polyline fill="none" stroke="#2563eb" stroke-width="2" opacity="0.60" points="{points(original_shared.cycle, original_shared.success_pct, panel_tops[0])}"/>')
    svg.append(f'<polyline fill="none" stroke="#ea580c" stroke-width="2" opacity="0.78" points="{points(dvac.cycle, dvac.success_pct, panel_tops[0])}"/>')
    svg += [
        '<line x1="1130" y1="188" x2="1170" y2="188" stroke="#2563eb" stroke-width="4"/><text x="1180" y="194" class="legend">original RLT</text>',
        '<line x1="1360" y1="188" x2="1400" y2="188" stroke="#ea580c" stroke-width="4"/><text x="1410" y="194" class="legend">RLT + teacher-DVAC [0,2]</text>',
    ]

    # Panel B lines and legend.
    top = panel_tops[1]
    for frame, col5, col10 in ((original_shared, "#1d4ed8", "#60a5fa"), (dvac, "#c2410c", "#fb923c")):
        svg.append(f'<polyline fill="none" stroke="{col5}" stroke-width="3" points="{points(frame.cycle, frame.success_ma5_pct, top)}"/>')
        svg.append(f'<polyline fill="none" stroke="{col10}" stroke-width="3" points="{points(frame.cycle, frame.success_ma10_pct, top)}"/>')
    legend2 = [("#1d4ed8", "original · trailing 5"), ("#60a5fa", "original · trailing 10"),
               ("#c2410c", "DVAC · trailing 5"), ("#fb923c", "DVAC · trailing 10")]
    for i, (color, label) in enumerate(legend2):
        xx = 870 + (i % 2) * 365
        yy = 668 + (i // 2) * 25
        svg.append(f'<line x1="{xx}" y1="{yy}" x2="{xx+36}" y2="{yy}" stroke="{color}" stroke-width="4"/><text x="{xx+46}" y="{yy+5}" class="legend">{label}</text>')

    # Panel C eval lines, points, and success-count labels.
    top = panel_tops[2]
    for frame, color, shape, offset in ((eval_original, "#2563eb", "circle", -11), (eval_dvac, "#ea580c", "diamond", 22)):
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points(frame.cycle, frame.eval_success_pct, top)}"/>')
        for _, row in frame.iterrows():
            xx, yy = x(row.cycle), y(row.eval_success_pct, top)
            if shape == "circle":
                svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="5" fill="white" stroke="{color}" stroke-width="3"/>')
            else:
                svg.append(f'<rect x="{xx-5:.1f}" y="{yy-5:.1f}" width="10" height="10" transform="rotate(45 {xx:.1f} {yy:.1f})" fill="white" stroke="{color}" stroke-width="3"/>')
            count = int(round(row.eval_success * 20))
            svg.append(f'<text x="{xx:.1f}" y="{yy+offset:.1f}" class="note" text-anchor="middle" style="fill:{color}">{count}/20</text>')
    svg += [
        '<line x1="1110" y1="1148" x2="1150" y2="1148" stroke="#2563eb" stroke-width="4"/><text x="1160" y="1154" class="legend">original RLT fixed20</text>',
        '<line x1="1375" y1="1148" x2="1415" y2="1148" stroke="#ea580c" stroke-width="4"/><text x="1425" y="1154" class="legend">DVAC [0,2] fixed20</text>',
        f'<text x="{left+chart_w/2:.1f}" y="1625" class="axis" text-anchor="middle">global cycle</text>',
        '<text x="55" y="1603" class="note">Purple dotted: online updates and DVAC apply begin (g136). Gray dashed: original run restarted after g250; the current DVAC run is one fresh 1–480 process.</text>',
        '</svg>',
    ]

    path = OUT / f"RLT_ORIGINAL_VS_DVAC_W0TO2_THROUGH_G{latest}.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original = load_run(ORIGINAL_LOGS, "original_rlt")
    dvac = load_run(DVAC_LOG, "rlt_teacher_dvac_w0to2")
    validate(original, dvac)

    combined = pd.concat([original, dvac], ignore_index=True)
    combined.to_csv(OUT / "training_and_eval_cycles.csv", index=False)
    plot_path = render_plot(original, dvac)

    latest = int(dvac.cycle.max())
    original_shared = original[original.cycle <= latest]
    current_evals = dvac[dvac.eval_episodes > 0]
    original_evals = original[(original.eval_episodes > 0) & (original.cycle <= latest)]
    summary = {
        "snapshot_latest_complete_cycle": latest,
        "train_episodes_per_cycle": 8,
        "rolling_windows": {"kind": "trailing", "sizes_cycles": [5, 10], "min_periods": 1},
        "eval_contract": {
            "episodes_per_eval": 20,
            "parallel_envs": 4,
            "epochs_per_env": 5,
            "completed_eval_points": current_evals.cycle.astype(int).tolist(),
            "completed_eval_count": int(len(current_evals)),
            "completed_eval_episodes": int(current_evals.eval_episodes.sum()),
        },
        "dvac_latest": {
            "cycle": latest,
            "raw_success_pct": float(dvac.iloc[-1].success_pct),
            "ma5_success_pct": float(dvac.iloc[-1].success_ma5_pct),
            "ma10_success_pct": float(dvac.iloc[-1].success_ma10_pct),
        },
        "shared_cycle_mean_success_pct": {
            "original_rlt": float(original_shared.train_success.mean() * 100),
            "rlt_teacher_dvac_w0to2": float(dvac.train_success.mean() * 100),
            "delta_percentage_points": float((dvac.train_success.mean() - original_shared.train_success.mean()) * 100),
        },
        "fixed20_shared_points": [
            {
                "cycle": int(cycle),
                "original_successes": int(round(float(original_evals.set_index("cycle").loc[cycle].eval_success) * 20)),
                "dvac_successes": int(round(float(current_evals.set_index("cycle").loc[cycle].eval_success) * 20)),
            }
            for cycle in sorted(set(original_evals.cycle).intersection(current_evals.cycle))
        ],
        "plot": plot_path.name,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
