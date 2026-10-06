"""CPU validation of this run's native capture before task-reward training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


RETRY = re.compile(r"UnStableError|trying\s+(?:a\s+)?new\s+seed|"
                   r"trial_seed[^\r\n]*(?:retry|retri|trying|重试|\+=\s*1)", re.IGNORECASE)
LOG_SUFFIXES = {".out", ".err", ".log", ".txt"}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(capture_dir, plan_path, worker_log_dir):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    expected = plan["native_episodes"]
    planned_seeds = plan["selected_seed_ids"]
    horizon, chunk = plan["horizon"], plan["action_chunk"]
    errors, records = [], []
    initial_hashes = defaultdict(list)
    if (type(expected) is not int or expected <= 0 or len(planned_seeds) != expected
            or len(set(planned_seeds)) != expected
            or any(type(seed) is not int for seed in planned_seeds)):
        errors.append("Plan does not contain the expected count of unique integer requested seed IDs")
    if not capture_dir.is_dir():
        errors.append("Capture directory is missing")
    paths = sorted(capture_dir.glob("native-*.json"))
    if len(paths) != expected:
        errors.append(f"Expected {expected} completed capture JSON records, found {len(paths)}")
    for path in paths:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            uid = row["episode_uid"]
            if not isinstance(uid, str) or not uid or type(row["seed"]) is not int:
                raise ValueError("Missing episode UID or integer requested seed")
            if row["task_name"] != plan["task"] or row["capture_mode"] != "reward_native":
                raise ValueError("Task or native capture mode differs from the plan")
            if row["collection_action_steps"] != horizon or row["max_episode_steps"] != horizon:
                raise ValueError("Capture did not complete the planned action horizon")
            labels, steps = row["native_success"], row["action_steps"]
            if steps != list(range(0, horizon + 1, chunk)):
                raise ValueError("Observed action clock does not match the planned C32 boundaries")
            if (len(labels) != len(steps) or labels[0] is not None
                    or any(type(label) is not bool for label in labels[1:])):
                raise ValueError("Native labels must be unknown at reset then boolean at each observation")
            hits = [index for index, label in enumerate(labels) if label is True]
            first = hits[0] if hits else None
            if row["first_success_position"] != first or row["reference_success"] is not bool(hits):
                raise ValueError("First-success metadata disagrees with native labels")
            endpoint = first if first is not None else len(steps) - 1
            if row["terminal_action_steps"] != steps[endpoint]:
                raise ValueError("Recorded terminal action step differs from the native label endpoint")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
                if "native_frames" not in data:
                    raise ValueError("Missing original native_frames array")
                frames = data["native_frames"]
                if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
                        or frames.shape[0] != len(labels) or min(frames.shape[1:3]) <= 0):
                    raise ValueError("Native RGB array dimensions/dtype/length do not match labels")
                if list(frames.shape[1:]) != row["source_frame_shape"]:
                    raise ValueError("Native image shape differs from the recorded source shape")
                if hashlib.sha256(frames.tobytes()).hexdigest() != row["native_frames_sha256"]:
                    raise ValueError("Native frame array hash differs from capture receipt")
                initial = hashlib.sha256(frames[0].tobytes()).hexdigest()
            record = dict(episode_uid=uid, requested_seed=row["seed"], initial_frame_sha256=initial,
                observations=len(labels), native_success=bool(hits), json_path=str(path),
                json_sha256=file_sha(path))
            records.append(record)
            initial_hashes[initial].append(dict(episode_uid=uid, requested_seed=row["seed"]))
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    uids = [row["episode_uid"] for row in records]
    requested = [row["requested_seed"] for row in records]
    if len(uids) != len(set(uids)):
        errors.append("Episode UID is duplicated")
    if len(requested) != len(set(requested)):
        errors.append("Requested seed is duplicated")
    if set(requested) != set(planned_seeds):
        errors.append("Completed requested seed IDs differ from the frozen plan")
    duplicate_initials = {digest: rows for digest, rows in initial_hashes.items() if len(rows) > 1}
    if duplicate_initials:
        errors.append("Identical initial native images found across capture records")

    matched_lines, match_count, scanned_files = [], 0, 0
    if not worker_log_dir.is_dir():
        errors.append("This run's worker-log directory is missing; reset retry scan is unavailable")
    else:
        for path in sorted(worker_log_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in LOG_SUFFIXES:
                continue
            scanned_files += 1
            try:
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    for line_number, line in enumerate(stream, 1):
                        if RETRY.search(line):
                            match_count += 1
                            if len(matched_lines) < 20:
                                matched_lines.append(dict(path=str(path), line=line_number, text=line.rstrip()[:800]))
            except OSError as exc:
                errors.append(f"Cannot read worker log {path}: {exc}")
    if not scanned_files:
        errors.append("No current worker text logs were available for reset retry verification")
    if match_count:
        errors.append(f"Found {match_count} simulator unstable/reset-seed retry log matches")

    return dict(schema_version=1, passed=not errors, task=plan["task"],
        capture_dir=str(capture_dir), plan=str(plan_path), plan_sha256=file_sha(plan_path),
        worker_log_dir=str(worker_log_dir), expected_episodes=expected,
        complete_valid_records=len(records), unique_episode_uids=len(set(uids)),
        unique_requested_seeds=len(set(requested)), native_successes=sum(row["native_success"] for row in records),
        native_failures=sum(not row["native_success"] for row in records),
        requested_seed_ids_match_plan=set(requested) == set(planned_seeds),
        duplicate_initial_images=duplicate_initials, worker_text_files_scanned=scanned_files,
        reset_retry_match_count=match_count, reset_retry_matches=matched_lines,
        actual_seed_recorded=False,
        conclusion=("No duplicate initial images or simulator seed retries were found; requested IDs match the plan. Actual simulator seeds were not separately recorded."
                    if not errors else "Capture validation failed; do not use these records for training until the listed issue is resolved."),
        errors=errors, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--worker-log-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = validate(args.capture_dir, args.plan, args.worker_log_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("records", "reset_retry_matches", "duplicate_initial_images")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
