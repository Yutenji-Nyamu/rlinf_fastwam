# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build plot-ready counts, intervals and actual-seed paired differences."""

import argparse
import math
from pathlib import Path

from protocol import read, save, thresholds


def interval(k, n):
    p, z = k / n, 1.959963984540054
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [center - half, center + half]


def summarize(root, condition):
    root = Path(root)
    paths = sorted((root / "batches").glob(condition + "_b*/done.json"))
    expected = 2 if condition == "calibration" else 4
    assert len(paths) == expected
    episodes, decisions, timing = [], [], []
    for p in paths:
        episodes.extend(read(p.with_name("episodes.json")))
        decisions.extend(read(p.with_name("decisions.json")))
        timing.extend(read(p.with_name("timing.json")))
    n, k = len(episodes), sum(e["success"] for e in episodes)
    assert n == expected * 32
    actual = [e["actual"] for e in episodes]
    result = {
        "condition": condition,
        "takeover_mode": decisions[0].get("takeover_mode", "weights"),
        "fraction_definition": "selected decisions / active decisions; selected submitted actions / all submitted actions",
        "episodes": n,
        "successes": k,
        "success_rate": k / n,
        "wilson95": interval(k, n),
        "unique_actual_seeds": len(set(actual)),
        "native_seed_retries": sum(e["native_retry"] for e in episodes),
        "inference_seconds": sum(t["inference_seconds"] for t in timing),
        "env_seconds": sum(t["env_seconds"] for t in timing),
        "strong_fraction": sum(d["strong"] for d in decisions) / len(decisions),
        "submitted_strong_fraction": sum(
            (d["end"] - d["start"]) * d["strong"] for d in decisions
        )
        / sum(d["end"] - d["start"] for d in decisions),
        "source_batches": [str(p.parent) for p in paths],
    }
    eligible = [d for d in decisions if d.get("eligible", True)]
    result["eligible_decisions"] = len(eligible)
    result["eligible_strong_fraction"] = sum(d["strong"] for d in eligible) / len(eligible) if eligible else 0
    for label, subset in [("first_three", [d for d in decisions if d["query"] < 3]),
                          ("after_three", [d for d in decisions if d["query"] >= 3]),
                          ("first_two", [d for d in decisions if d["query"] < 2]),
                          ("later", [d for d in decisions if d["query"] >= 2])]:
        result[label] = {"decisions": len(subset), "strong": sum(d["strong"] for d in subset),
                         "fraction": sum(d["strong"] for d in subset) / len(subset) if subset else 0}
    config = read(paths[0].with_name("config.json"))
    baseline = config.get("comparison_baseline", "weak_n10_c50")
    result["execute_chunk"] = config["execute_chunk"]
    result["max_takeover_queries"] = config.get("max_takeover_queries")
    base = root / f"summaries/{baseline}-episodes.json"
    if base.exists() and condition not in ("calibration", baseline):
        w = {e["actual"]: e for e in read(base)}
        other = {e["actual"]: e for e in episodes}
        matched = sorted(set(w) & set(other))
        gain = sum(not w[s]["success"] and other[s]["success"] for s in matched)
        loss = sum(w[s]["success"] and not other[s]["success"] for s in matched)
        result["paired_with_weak"] = {
            "baseline": baseline,
            "matched": len(matched),
            "gained": gain,
            "lost": loss,
            "difference": (gain - loss) / len(matched) if matched else None,
            "all_128_matched": len(matched) == 128 and len(set(actual)) == 128,
        }
    save(root / f"summaries/{condition}.json", result)
    save(root / f"summaries/{condition}-episodes.json", episodes)
    if condition == "calibration":
        save(
            root / "threshold.json",
            thresholds([p.with_name("decisions.json") for p in paths]),
        )
    index = [
        read(p)
        for p in sorted((root / "summaries").glob("*.json"))
        if not p.name.endswith("-episodes.json")
    ]
    save(root / "summary.json", index)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("condition")
    a = p.parse_args()
    summarize(a.root, a.condition)
