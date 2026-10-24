# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Read completed batches and compare N1/N10 on matching actual episodes."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def summarize(output):
    output = Path(output)
    episodes, batches = defaultdict(dict), defaultdict(list)
    for path in sorted(output.glob("summary-raw-c*-n*-b*.json")):
        match = re.fullmatch(r"summary-raw-c(\d+)-n(\d+)-b(\d+)\.json", path.name)
        if not match:
            continue
        c, n, batch = map(int, match.groups())
        summary = json.loads(path.read_text())
        batches[c, n].append(summary)
        for task in summary["tasks"]:
            if not task["valid"]:
                continue
            rows = json.loads((Path(task["output"]) / "episodes.json").read_text())
            for row in rows:
                key = (
                    task["task"],
                    batch,
                    row["slot"],
                    row["requested"],
                    row["actual"],
                )
                episodes[c, n][key] = row
    points, paired = [], []
    for c, n in sorted(batches):
        rows = episodes[c, n]
        by_task = defaultdict(list)
        for key, row in rows.items():
            by_task[key[0]].append(bool(row["success"]))
        points.append(
            {
                "C": c,
                "N": n,
                "completed_batches": len(batches[c, n]),
                "episodes": len(rows),
                "successes": sum(r["success"] for r in rows.values()),
                "task_mean": sum(sum(v) / len(v) for v in by_task.values())
                / len(by_task)
                if by_task
                else None,
                "tasks": {
                    t: {"episodes": len(v), "successes": sum(v)}
                    for t, v in by_task.items()
                },
                "missing_task_batches": sum(s["missing_tasks"] for s in batches[c, n]),
                "elapsed_seconds": sum(s["elapsed_seconds"] for s in batches[c, n]),
            }
        )
    for c in (10, 20, 30, 40):
        a, b = episodes[c, 1], episodes[c, 10]
        common = a.keys() & b.keys()
        if not common:
            continue
        by_task = defaultdict(list)
        for key in common:
            by_task[key[0]].append(int(b[key]["success"]) - int(a[key]["success"]))
        paired.append(
            {
                "C": c,
                "paired_episodes": len(common),
                "N1_only": sum(a[k]["success"] and not b[k]["success"] for k in common),
                "N10_only": sum(
                    b[k]["success"] and not a[k]["success"] for k in common
                ),
                "task_mean_N10_minus_N1": sum(sum(v) / len(v) for v in by_task.values())
                / len(by_task),
                "tasks": {
                    t: {"paired": len(v), "delta": sum(v) / len(v)}
                    for t, v in by_task.items()
                },
            }
        )
    return {
        "points": points,
        "paired": paired,
        "pair_key": ["task", "batch", "slot", "requested_seed", "actual_seed"],
        "note": "Missing batches excluded and reported; task means give each valid task equal weight. Smoke excluded.",
    }


if __name__ == "__main__":
    result = summarize(sys.argv[1])
    print(json.dumps(result, indent=2))
