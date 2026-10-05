"""Join RM scores with native-frame categories, excluding latched-history controls."""
import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(manifest, probe):
    samples = {r["sample_id"]: r for r in manifest["samples"]}
    scored = {r["sample_id"]: r for r in probe["rows"]}
    if len(samples) != len(manifest["samples"]) or len(scored) != len(probe["rows"]) or set(samples) != set(scored):
        raise ValueError("RM outputs do not uniquely match every selected native frame")
    if probe["samples_sha256"] != manifest["samples_sha256"]:
        raise ValueError("RM scored another NPZ")
    groups = {key: [] for key in ("first_success", "near_false", "post_success_latched")}
    for uid, sample in samples.items():
        score = scored[uid]["score"]
        if not math.isfinite(score) or not 0 <= score <= 1 or scored[uid]["native_label"] != sample["label"]:
            raise ValueError("RM output score or copied label is invalid")
        groups[sample["category"]].append(score)
    categories = {}
    for key, values in groups.items():
        categories[key] = dict(count=len(values), score_min=min(values) if values else None,
            score_median=statistics.median(values) if values else None, score_max=max(values) if values else None,
            scores_at_least_0_9=sum(x >= .9 for x in values))
    thresholds = {}
    for threshold in (.5, .9):
        thresholds[str(threshold)] = dict(first_success_count=len(groups["first_success"]),
            first_success_detected=sum(x >= threshold for x in groups["first_success"]),
            first_success_missed=sum(x < threshold for x in groups["first_success"]),
            near_false_count=len(groups["near_false"]),
            near_false_false_positive=sum(x >= threshold for x in groups["near_false"]),
            near_false_true_negative=sum(x < threshold for x in groups["near_false"]),
            excluded_post_success_latched=len(groups["post_success_latched"]))
    return dict(categories=categories, thresholds=thresholds,
        near_false_meaning=manifest["near_false_meaning"], first_success_caveat=manifest["first_success_caveat"],
        post_success_caveat=manifest["post_success_caveat"],
        conclusion_limit="Small native classifier diagnostic. No OpenDW-domain guarantee and no calibrated threshold fitting.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-json", required=True, type=Path)
    p.add_argument("--probe-result", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("A fresh summary output path is required")
    result = summarize(json.loads(args.samples_json.read_text()), json.loads(args.probe_result.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
