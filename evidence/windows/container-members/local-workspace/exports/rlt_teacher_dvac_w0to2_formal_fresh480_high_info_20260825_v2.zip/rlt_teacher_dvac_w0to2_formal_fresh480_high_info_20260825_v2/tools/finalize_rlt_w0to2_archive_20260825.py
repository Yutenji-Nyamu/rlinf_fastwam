from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import sys
import zipfile
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tmp"))
import analyze_rlt_dvac_vs_original_20260825 as base  # noqa: E402


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?"


def find_one(root: Path, name: str, required: str = "") -> Path:
    matches = [p for p in root.rglob(name) if required in str(p)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name!r} containing {required!r}, got {matches}")
    return matches[0]


def parse_method_metrics(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, float | int]] = []
    names = (
        "weight_mean", "weight_median", "weight_p05", "weight_p95",
        "weight_ess_ratio", "top20_weight_mass", "downweighted_fraction",
        "upweighted_fraction", "min_weight_fraction", "max_weight_fraction",
        "actor_switch_rate", "query_weight_q_gap_corr", "replay_age_mean",
    )
    for block in text.split("╭"):
        step_match = base.STEP_RE.search(block)
        if not step_match or "╰" not in block:
            continue
        row: dict[str, float | int] = {"cycle": int(step_match.group(1))}
        for name in names:
            match = re.search(rf"rlt_dvac/{re.escape(name)}=({NUMBER})", block, re.I)
            row[name] = float(match.group(1)) if match else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("cycle").drop_duplicates("cycle", keep="last")
    # Rich's three-column terminal table truncates the numeric value of the
    # long downweighted_fraction key.  The saved weights have no exact w=1
    # entries, so the two directions are complementary for this run.
    if frame.downweighted_fraction.isna().all() and frame.upweighted_fraction.notna().any():
        frame["downweighted_fraction"] = 1.0 - frame["upweighted_fraction"]
    return frame


def finite_stats(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "min": None, "max": None, "last": None}
    return {
        "mean": float(values.mean()), "min": float(values.min()),
        "max": float(values.max()), "last": float(values.iloc[-1]),
    }


def line_points(xs: np.ndarray, ys: np.ndarray, left: float, top: float,
                width: float, height: float, ymin: float, ymax: float) -> str:
    valid = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[valid], ys[valid]
    if not len(xs):
        return ""
    xmin, xmax = float(xs.min()), float(xs.max())
    dx, dy = max(xmax - xmin, 1.0), max(ymax - ymin, 1e-9)
    return " ".join(
        f"{left + (x - xmin) / dx * width:.1f},{top + height - (y - ymin) / dy * height:.1f}"
        for x, y in zip(xs, ys, strict=True)
    )


def render_diagnostics(method: pd.DataFrame, resources: pd.DataFrame, out: Path) -> Path:
    width, height = 1800, 1180
    panels = [(85, 175), (930, 175), (85, 665), (930, 665)]
    pw, ph = 730, 355
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fc"/>',
        '<style>text{font-family:Arial,Segoe UI,sans-serif;fill:#172033}.title{font-size:34px;font-weight:700}'
        '.sub{font-size:18px;fill:#52627a}.pt{font-size:22px;font-weight:700}.axis{font-size:13px;fill:#64748b}'
        '.legend{font-size:14px;fill:#334155}</style>',
        '<text x="55" y="58" class="title">RLT teacher-DVAC [0,2] — final method and resource diagnostics</text>',
        '<text x="55" y="94" class="sub">Cycle-level method telemetry and record-only system monitoring; no checkpoint tensors are embedded.</text>',
    ]

    def panel(index: int, title: str, series: list[tuple[str, str, np.ndarray, float, float]], xs: np.ndarray) -> None:
        left, top = panels[index]
        svg.append(f'<rect x="{left-30}" y="{top-55}" width="{pw+70}" height="{ph+105}" rx="16" fill="white" stroke="#dbe3ef"/>')
        svg.append(f'<text x="{left}" y="{top-20}" class="pt">{escape(title)}</text>')
        for tick in range(5):
            yy = top + tick * ph / 4
            svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+pw}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        for idx, (label, color, ys, ymin, ymax) in enumerate(series):
            points = line_points(xs, ys, left, top, pw, ph, ymin, ymax)
            svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
            lx = left + 10 + (idx % 3) * 235
            ly = top + ph + 34 + (idx // 3) * 22
            svg.append(f'<line x1="{lx}" y1="{ly-5}" x2="{lx+30}" y2="{ly-5}" stroke="{color}" stroke-width="4"/><text x="{lx+39}" y="{ly}" class="legend">{escape(label)}</text>')

    mx = method.cycle.to_numpy(dtype=float)
    panel(0, "A. Per-action gradient weights", [
        ("p05", "#2563eb", method.weight_p05.to_numpy(float), 0, 2),
        ("mean", "#059669", method.weight_mean.to_numpy(float), 0, 2),
        ("p95", "#ea580c", method.weight_p95.to_numpy(float), 0, 2),
    ], mx)
    panel(1, "B. Credit concentration", [
        ("weight ESS", "#7c3aed", method.weight_ess_ratio.to_numpy(float), 0, 1),
        ("top-20% mass", "#db2777", method.top20_weight_mass.to_numpy(float), 0, 1),
    ], mx)
    panel(2, "C. Weight direction and boundary hits", [
        ("downweighted", "#2563eb", method.downweighted_fraction.to_numpy(float), 0, 1),
        ("upweighted", "#ea580c", method.upweighted_fraction.to_numpy(float), 0, 1),
        ("at 0", "#0891b2", method.min_weight_fraction.to_numpy(float), 0, 1),
        ("at 2", "#ca8a04", method.max_weight_fraction.to_numpy(float), 0, 1),
    ], mx)

    rx = (resources.unix_time - resources.unix_time.iloc[0]).to_numpy(float) / 3600.0
    ram_gib = resources.cgroup_current_bytes.to_numpy(float) / (1024 ** 3)
    gpu0 = resources.gpu0_used_mib.to_numpy(float) / 1024.0
    gpu1 = resources.gpu1_used_mib.to_numpy(float) / 1024.0
    ymax = max(240.0, float(np.nanmax(ram_gib)) * 1.05)
    panel(3, "D. Host RAM and GPU memory", [
        ("cgroup RAM GiB", "#dc2626", ram_gib, 0, ymax),
        ("GPU0 GiB", "#2563eb", gpu0, 0, ymax),
        ("GPU1 GiB", "#059669", gpu1, 0, ymax),
    ], rx)
    svg.append('<text x="900" y="1160" class="axis" text-anchor="middle">A–C x-axis: global cycle; D x-axis: hours since launch. Panel scales are stated in the CSV/summary.</text>')
    svg.append('</svg>')
    path = out / "RLT_DVAC_W0TO2_FINAL_DIAGNOSTICS.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    args = parser.parse_args()

    extracted = args.extracted_root.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise RuntimeError(f"Output already exists: {out}")
    out.mkdir(parents=True)
    raw_out = out / "raw_server_evidence"
    shutil.copytree(extracted, raw_out)

    dvac_log = find_one(extracted, "metrics.log", "rlt_teacher_dvac_w0to2")
    resources_path = find_one(extracted, "resources.csv", "rlt_teacher_dvac_w0to2")
    resolved_path = find_one(extracted, "resolved.yaml", "rlt_teacher_dvac_w0to2")
    exit_path = find_one(extracted, "exit_code.txt", "rlt_teacher_dvac_w0to2")
    finished_path = find_one(extracted, "finished_at.txt", "rlt_teacher_dvac_w0to2")
    if exit_path.read_text().strip() != "0":
        raise RuntimeError("Old run did not finish with exit code 0")

    original = base.load_run(base.ORIGINAL_LOGS, "original_rlt")
    dvac = base.load_run(dvac_log, "rlt_teacher_dvac_w0to2")
    base.validate(original, dvac)
    if int(dvac.cycle.max()) != 480:
        raise RuntimeError(f"Expected complete 480 cycles, got {dvac.cycle.max()}")
    method = parse_method_metrics(dvac_log)
    resources = pd.read_csv(resources_path)

    analysis_dir = out / "analysis"
    analysis_dir.mkdir()
    base.OUT = analysis_dir
    combined = pd.concat([original, dvac], ignore_index=True)
    combined.to_csv(analysis_dir / "training_and_eval_cycles.csv", index=False)
    method.to_csv(analysis_dir / "rlt_dvac_method_cycles.csv", index=False)
    base.render_plot(original, dvac)
    render_diagnostics(method, resources, analysis_dir)

    original_eval = original[original.eval_episodes > 0]
    dvac_eval = dvac[dvac.eval_episodes > 0]
    summary = {
        "status": "complete",
        "cycle": 480,
        "finished_at": finished_path.read_text().strip(),
        "resolved_config": str(resolved_path.relative_to(extracted)),
        "train": {
            "original_mean_pct": float(original.train_success.mean() * 100),
            "dvac_mean_pct": float(dvac.train_success.mean() * 100),
            "delta_pp": float((dvac.train_success.mean() - original.train_success.mean()) * 100),
            "original_last100_pct": float(original.tail(100).train_success.mean() * 100),
            "dvac_last100_pct": float(dvac.tail(100).train_success.mean() * 100),
            "dvac_final_raw_pct": float(dvac.iloc[-1].success_pct),
            "dvac_final_ma5_pct": float(dvac.iloc[-1].success_ma5_pct),
            "dvac_final_ma10_pct": float(dvac.iloc[-1].success_ma10_pct),
        },
        "fixed20_eval": {
            "points": int(len(dvac_eval)),
            "episodes": int(dvac_eval.eval_episodes.sum()),
            "original_final_pct": float(original_eval.iloc[-1].eval_success_pct),
            "dvac_final_pct": float(dvac_eval.iloc[-1].eval_success_pct),
            "original_mean_over_points_pct": float(original_eval.eval_success_pct.mean()),
            "dvac_mean_over_points_pct": float(dvac_eval.eval_success_pct.mean()),
        },
        "method": {name: finite_stats(method[name]) for name in (
            "weight_mean", "weight_p05", "weight_p95", "weight_ess_ratio",
            "top20_weight_mass", "downweighted_fraction", "upweighted_fraction",
            "min_weight_fraction", "max_weight_fraction", "actor_switch_rate",
        )},
        "resources": {
            "peak_cgroup_gib": float(resources.cgroup_current_bytes.max() / (1024 ** 3)),
            "peak_gpu0_gib": float(resources.gpu0_used_mib.max() / 1024),
            "peak_gpu1_gib": float(resources.gpu1_used_mib.max() / 1024),
            "max_psi_some_avg10": float(resources.psi_some_avg10.max()),
            "max_psi_full_avg10": float(resources.psi_full_avg10.max()),
            "final_event_high": int(resources.event_high.iloc[-1]),
            "final_event_max": int(resources.event_max.iloc[-1]),
            "final_event_oom": int(resources.event_oom.iloc[-1]),
            "final_event_oom_kill": int(resources.event_oom_kill.iloc[-1]),
        },
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme = f"""# RLT teacher-DVAC `[0,2]` fresh-480 high-information archive

- Status: complete, exit code 0, cycle 480; finished `{summary['finished_at']}`.
- Training rollout mean: original RLT `{summary['train']['original_mean_pct']:.2f}%`, DVAC `{summary['train']['dvac_mean_pct']:.2f}%`, delta `{summary['train']['delta_pp']:+.2f}` pp.
- Last-100 training rollout mean: original `{summary['train']['original_last100_pct']:.2f}%`, DVAC `{summary['train']['dvac_last100_pct']:.2f}%`.
- Final fixed20: original `{summary['fixed20_eval']['original_final_pct']:.1f}%`, DVAC `{summary['fixed20_eval']['dvac_final_pct']:.1f}%`.
- Peak resources: cgroup RAM `{summary['resources']['peak_cgroup_gib']:.2f}` GiB; GPU0/1 `{summary['resources']['peak_gpu0_gib']:.2f}/{summary['resources']['peak_gpu1_gib']:.2f}` GiB.

`analysis/` contains the final comparison plot, DVAC diagnostics, cycle tables and machine-readable summary. `raw_server_evidence/` keeps logs, resolved config, resource CSV, small telemetry traces and checkpoint manifests. Model/optimizer/replay bodies, videos and full Ray logs are intentionally excluded.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    tools = out / "tools"
    tools.mkdir()
    shutil.copy2(Path(__file__), tools / Path(__file__).name)
    shutil.copy2(ROOT / "tmp/analyze_rlt_dvac_vs_original_20260825.py", tools / "analyze_rlt_dvac_vs_original_20260825.py")

    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                zf.write(path, Path(out.name) / path.relative_to(out))
    print(json.dumps({"output": str(out), "zip": str(args.zip_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
