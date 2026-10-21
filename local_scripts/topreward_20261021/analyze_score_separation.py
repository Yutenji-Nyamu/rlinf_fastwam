"""Describe existing native scores as ranking signals; no model or GPU inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def comparison(positive, negative, rng, bootstrap=4000):
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    wins = (positive[:, None] > negative).astype(float)
    wins += 0.5 * (positive[:, None] == negative)
    p = rng.integers(len(positive), size=(bootstrap, len(positive)))
    n = rng.integers(len(negative), size=(bootstrap, len(negative)))
    sampled = wins[p[:, :, None], n[:, None, :]].mean(axis=(1, 2))
    return {
        "success_n": len(positive),
        "failure_n": len(negative),
        "success_mean": float(positive.mean()),
        "failure_mean": float(negative.mean()),
        "success_median": float(np.median(positive)),
        "failure_median": float(np.median(negative)),
        "success_q25_q75": np.quantile(positive, [0.25, 0.75]).tolist(),
        "failure_q25_q75": np.quantile(negative, [0.25, 0.75]).tolist(),
        "auc": float(wins.mean()),
        "auc_bootstrap95": np.quantile(sampled, [0.025, 0.975]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.scores.read_text().splitlines()]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["actual_seed"]].append(row)
    episodes = []
    for seed, values in grouped.items():
        values.sort(key=lambda row: row["chunk"])
        scores = np.asarray([row["probability"] for row in values])
        episodes.append({
            "seed": seed,
            "split": values[0]["split"],
            "success": bool(values[-1]["final_native_success"]),
            "scores": scores,
        })

    aggregations = {
        "first": lambda x: x[0],
        "last": lambda x: x[-1],
        "mean": np.mean,
        "max": np.max,
        "delta_last_first": lambda x: x[-1] - x[0],
        "sum": np.sum,
        "first3_mean": lambda x: np.mean(x[:3]),
        "chunk3": lambda x: x[2],
    }
    rng = np.random.default_rng(20261021)
    report = {
        "source": "native-scores.jsonl, existing 128 native episodes / 999 windows",
        "purpose": "Post-hoc descriptive ranking analysis, not threshold selection",
        "auc_definition": "P(success score > failure score), ties count 0.5",
        "uncertainty": "4000 stratified episode bootstrap resamples; descriptive 95% interval",
        "limitations": [
            "Distinct initial states; no same-state G8 candidate groups.",
            "Native success ends episodes early; failure episodes run to 12 chunks.",
            "Late-chunk means condition on episodes still active; no endpoint padding.",
            "First chunk is after actions, not a reset-state score.",
            "First3 comparisons have equal chunk counts; successful final chunks may contain fewer executed steps.",
            "All metrics use already inspected data; no claim of a new independent validation set.",
        ],
        "splits": {},
    }
    for split in ["calibration", "holdout", "all"]:
        part = [e for e in episodes if split == "all" or e["split"] == split]
        output = {"summaries": {}, "trajectory": []}
        for name, aggregate in aggregations.items():
            p = [float(aggregate(e["scores"])) for e in part if e["success"]]
            n = [float(aggregate(e["scores"])) for e in part if not e["success"]]
            output["summaries"][name] = comparison(p, n, rng)
        output["lengths"] = {
            str(truth): dict(sorted(Counter(len(e["scores"]) for e in part if e["success"] == truth).items()))
            for truth in [True, False]
        }
        output["success_last_below_previous"] = sum(
            e["scores"][-1] < e["scores"][-2] for e in part if e["success"]
        ).item()
        for chunk in range(max(len(e["scores"]) for e in part)):
            record = {"chunk_1based": chunk + 1}
            for truth, label in [(True, "success"), (False, "failure")]:
                values = [e["scores"][chunk] for e in part if e["success"] == truth and len(e["scores"]) > chunk]
                record[label] = {"n": len(values)}
                if values:
                    record[label].update(mean=float(np.mean(values)), q25_q75=np.quantile(values, [0.25, 0.75]).tolist())
            output["trajectory"].append(record)
        report["splits"][split] = output

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "score-separation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    part = [e for e in episodes if e["split"] == "holdout"]
    palette = [(True, "最终成功", "#167a68"), (False, "最终失败", "#cf6448")]
    ax = axes[0]
    for truth, label, color in palette:
        selected = [e for e in part if e["success"] == truth]
        for e in selected:
            ax.plot(np.arange(1, len(e["scores"]) + 1), e["scores"], color=color, alpha=0.12, lw=0.7)
        entries = report["splits"]["holdout"]["trajectory"]
        key = "success" if truth else "failure"
        entries = [r for r in entries if r[key]["n"]]
        x = [r["chunk_1based"] for r in entries]
        y = [r[key]["mean"] for r in entries]
        ax.plot(x, y, color=color, lw=2.7, marker="o", ms=4, label=f"{label}均值（32条）")
    ax.set(title="分数沿轨迹如何变化｜留出集", xlabel="已执行的动作块编号", ylabel="Qwen 原始 P(True)", xlim=(0.8, 12.2), ylim=(0, 1))
    ax.set_xticks(range(1, 13))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.17)
    ax.text(0.02, -0.24, "细线：单条轨迹。成功第1–3块均有32条，第4块剩14条；\n失败每块均有32条。成功结束后不补帧、不延长曲线。", transform=ax.transAxes, fontsize=8.7, va="top")

    ax = axes[1]
    for truth, label, color in palette:
        values = np.sort([e["scores"][-1] for e in part if e["success"] == truth])
        ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=color, lw=2.4, label=label)
        ax.plot(values, np.zeros_like(values) + (0.025 if truth else 0.055), "|", color=color, alpha=0.6)
    m = report["splits"]["holdout"]["summaries"]["last"]
    ax.set(title=f"每条轨迹的终点分数分布｜AUC={m['auc']:.2f}", xlabel="Qwen 原始 P(True)", ylabel="分数不高于横轴值的轨迹比例", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.17)
    ax.text(0.02, -0.24, "成功终点在第3或4块，失败终点在第12块；\n这个终点AUC不是相同执行长度、相同起点下的候选排序。", transform=ax.transAxes, fontsize=8.7, va="top")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.89, bottom=0.28, wspace=0.27)
    fig.savefig(args.output_dir / "score-separation.png", dpi=170)
    fig.savefig(args.output_dir / "score-separation.pdf")
    plt.close(fig)
    print(json.dumps({s: r["summaries"] for s, r in report["splits"].items() if s != "all"}, indent=2))


if __name__ == "__main__":
    main()
