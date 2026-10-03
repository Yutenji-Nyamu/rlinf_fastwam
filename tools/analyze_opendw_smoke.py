#!/usr/bin/env python3
"""Read-only smoke evidence analysis; writes only requested JSON/Markdown reports.

  python -B tools/analyze_opendw_smoke.py OWNER_DIR --output-prefix /path/analysis
  # Optional installed dependency, never installed by this tool:
  ... --tensorboard

Owner resource phase windows assign service request envelopes to trials. Repeated
request_id values are never joined across envelopes. Missing/ambiguous evidence
stays unknown. Checkpoint markers and nonzero gradients do not establish learning.
Naive timestamps need explicit --naive-offset +08:00; otherwise they are unknown.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics


def unknown(reason):
    return {"status": "unknown", "reason": reason}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def timestamp(value, naive_offset=None):
    if number(value):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            if naive_offset is None:
                return None
            sign = -1 if naive_offset.startswith("-") else 1
            hours, minutes = map(int, naive_offset[1:].split(":"))
            parsed = parsed.replace(tzinfo=timezone(sign * timedelta(hours=hours, minutes=minutes)))
        return parsed.timestamp()
    except (ValueError, OverflowError):
        return None


def row_time(row, offset=None):
    return timestamp(row.get("timestamp_utc", row.get("time", row.get("wall_time"))), offset)


def read_json(path, issues):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        issues.append({"source": str(path), "issue": "missing_or_invalid_json"})
        return {}


def read_jsonl(path, issues):
    rows = []
    try:
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            for index, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        row["_source"], row["_line"] = str(path), index
                        rows.append(row)
                    else:
                        raise ValueError()
                except ValueError:
                    issues.append({"source": str(path), "line": index, "issue": "invalid_jsonl_record"})
    except OSError:
        issues.append({"source": str(path), "issue": "missing_jsonl"})
    return rows


def summary(values):
    values = sorted(float(value) for value in values if number(value))
    if not values:
        return unknown("no finite observations")
    def quantile(q):
        position = (len(values) - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return values[low] * (high - position) + values[high] * (position - low) if low != high else values[low]
    return {"status": "observed", "count": len(values), "min": values[0], "max": values[-1],
            "mean": statistics.fmean(values), "p50": quantile(0.5), "p95": quantile(0.95)}


def config_read(path, issues):
    if not path or not Path(path).is_file():
        return {}
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except ValueError:
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except Exception as error:
            issues.append({"source": str(path), "issue": "config_unavailable", "error_type": type(error).__name__})
            return {}


def trial_windows(owner, plan, resources, issues, offset):
    trials = []
    rows = plan.get("trials", [])
    if not rows:
        keys = sorted({row.get("phase") for row in resources if re.fullmatch(r"n\d+", str(row.get("phase", "")))})
        rows = [{"key": key} for key in keys]
    for row in rows:
        key = row["key"]
        directory = owner / key
        result = read_json(directory / "result.json", issues) if (directory / "result.json").exists() else {}
        finished = read_json(directory / "driver-finished.json", issues) if (directory / "driver-finished.json").exists() else {}
        job = read_json(directory / "ray-job.json", issues) if (directory / "ray-job.json").exists() else {}
        active = [row_time(r, offset) for r in resources if r.get("phase") == key]
        active = [t for t in active if t is not None]
        begin = row_time(job, offset)
        starts = active + ([begin] if begin is not None else [])
        end = row_time(finished, offset)
        ends = [row_time(r, offset) for r in resources if r.get("phase") == key + "_complete"]
        ends = [t for t in ends if t is not None]
        if end is not None:
            window_end, end_basis = end, "driver-finished receipt"
        elif ends:
            window_end, end_basis = min(ends), "owner complete-phase sample"
        elif active:
            window_end, end_basis = max(active), "last observed phase sample; running tail excluded"
        else:
            window_end, end_basis = None, "unknown"
        start = min(starts) if starts else None
        cfg = config_read(row.get("config"), issues)
        trials.append({"key": key, "num_envs": row.get("num_envs", result.get("num_envs")),
                       "directory": str(directory), "start": start, "end": window_end,
                       "window_basis": {"start": "earliest owner phase/job timestamp", "end": end_basis},
                       "seconds": result.get("seconds"), "exit_code": result.get("exit_code", finished.get("exit_code")),
                       "config": cfg, "config_source": row.get("config"), "requests": []})
    return trials


def request_envelopes(events, issues, offset):
    requests, active = [], {}
    # Events retain file/line order; each service logs requests under one lock.
    for event in events:
        kind = event.get("event")
        stream = (event.get("_source"), event.get("pid"))
        if kind == "request_started":
            if stream in active:
                active[stream]["envelope_issue"] = "superseded without completion"
            request = {"index": len(requests), "request_id": event.get("request_id"), "pid": event.get("pid"),
                       "source": event.get("_source"), "start_line": event.get("_line"),
                       "start": row_time(event, offset), "end": None, "batch": event.get("batch"),
                       "completed": False, "rows": [], "row_starts": [], "events": [event]}
            requests.append(request)
            active[stream] = request
            continue
        request = active.get(stream)
        if request is None:
            continue
        if kind in {"row_started", "row_completed", "request_completed"} and event.get("request_id") != request["request_id"]:
            request["envelope_issue"] = "request_id changed within request envelope"
            continue
        request["events"].append(event)
        if kind == "row_started":
            request["row_starts"].append(event)
        if kind == "row_completed":
            request["rows"].append(event)
        if kind in {"request_completed", "request_failed"}:
            request["end"] = row_time(event, offset)
            request["completed"] = kind == "request_completed"
            active.pop(stream)
    return requests


def assign_requests(trials, requests):
    unmatched = []
    for request in requests:
        start, end = request["start"], request["end"]
        candidates = [trial for trial in trials if start is not None and end is not None and
                      trial["start"] is not None and trial["end"] is not None and
                      trial["start"] <= start <= end <= trial["end"]]
        if len(candidates) == 1 and not request.get("envelope_issue"):
            candidates[0]["requests"].append(request)
        else:
            unmatched.append({"index": request["index"], "request_id": request["request_id"],
                              "batch": request["batch"], "start": start, "end": end,
                              "reason": request.get("envelope_issue", "missing timestamps/completion or ambiguous trial window")})
    return unmatched


def memory_summary(trial, resources, offset):
    service = [event for request in trial["requests"] for event in request["events"]]
    result = {"service_rss_sample_peak_bytes": max((e.get("VmRSS_bytes", 0) for e in service), default=0) or None,
              "service_pss_sample_peak_bytes": max((e.get("Pss_bytes", 0) for e in service), default=0) or None,
              "service_cuda_allocated_sample_peak_bytes": max((e.get("cuda_allocated_bytes", 0) for e in service), default=0) or None,
              "service_cuda_max_allocated_bytes_during_requests": max((e.get("cuda_max_allocated_bytes", 0) for e in service), default=0) or None,
              "service_cuda_reserved_sample_peak_bytes": max((e.get("cuda_reserved_bytes", 0) for e in service), default=0) or None}
    board, process_gpu, rss = [], [], []
    for event in service:
        parts = str(event.get("gpu_nvml", "")).split(",")
        if len(parts) == 5:
            try:
                board.append(float(parts[2]) * 1024**2)
            except ValueError:
                pass
    snapshots = [r for r in resources if r.get("phase") == trial["key"]]
    for row in snapshots:
        processes = row.get("processes", [])
        pids = {str(p.get("pid")) for p in processes}
        rss_values = [p["VmRSS_kib"] * 1024 for p in processes if number(p.get("VmRSS_kib"))]
        if rss_values:
            rss.append(sum(rss_values))
        values = []
        for line in str(row.get("compute_memory_csv", "")).splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 3 and fields[0] in pids:
                match = re.fullmatch(r"([0-9.]+)\s*(?:MiB)?", fields[2])
                if match:
                    values.append(float(match[1]) * 1024**2)
        if values:
            process_gpu.append(sum(values))
    result.update(gpu_board_used_sample_peak_bytes=max(board, default=None),
                  managed_compute_memory_sample_peak_bytes=max(process_gpu, default=None),
                  managed_process_rss_sum_sample_peak_bytes=max(rss, default=None),
                  resource_samples=len(snapshots),
                  limits="Sampled peaks can miss transients. Board usage includes other processes; RSS sums count shared pages repeatedly. PSS here is service only. Torch max is service since its onload reset, not actor or board.")
    return result


def action_summary(requests):
    records = [row.get("action_telemetry", {}) for request in requests for row in request["row_starts"]]
    result = {}
    for kind in ("actions", "state"):
        entries = [row.get("wm_normalization", {}).get(kind, {}) for row in records]
        entries = [entry for entry in entries if entry.get("available") is True and number(entry.get("abs_z_gt5_count")) and
                   isinstance(entry.get("abs_z_gt5_count_by_dim"), list) and len(entry["abs_z_gt5_count_by_dim"]) == 14 and
                   all(number(value) for value in entry["abs_z_gt5_count_by_dim"])]
        denominator_per_row = 32 * 14 if kind == "actions" else 14
        if not entries:
            result[kind] = unknown("no parsed-stat telemetry")
            continue
        result[kind] = {"status": "observed", "observed_rows": len(entries),
                        "abs_z_gt5_count": sum(entry["abs_z_gt5_count"] for entry in entries),
                        "abs_z_gt5_fraction": sum(entry["abs_z_gt5_count"] for entry in entries) / (len(entries) * denominator_per_row),
                        "abs_z_gt5_count_by_dim": [sum(entry["abs_z_gt5_count_by_dim"][dim] for entry in entries) for dim in range(14)]}
    result["joint_command_delta_p95_per_row"] = summary([row.get("adjacent_command_abs_delta", {}).get("joint_p95") for row in records])
    result["unit_note"] = "Command changes are per annotation step, not physical velocity. Unlogged rows are not assumed zero."
    return result


def retained_groups(trial):
    cfg = trial["config"]
    env = cfg.get("env", {}).get("train", {})
    algorithm = cfg.get("algorithm", {})
    contract = (env.get("max_episode_steps") == 32 and env.get("max_steps_per_rollout_epoch") == 32 and
                env.get("rollout_epoch") == 1 and env.get("use_rel_reward") is True and env.get("reward_coef") == 1.0 and
                algorithm.get("group_size") == 8 and algorithm.get("filter_rewards") is True and
                algorithm.get("rewards_lower_bound") == 0.1 and algorithm.get("rewards_upper_bound") == 0.9)
    if not contract:
        return unknown("config does not prove R1/L32/relative reward=1/G8/filter [0.1,0.9]")
    if len(trial["requests"]) != 1:
        return unknown("single complete first-C32 request not established")
    request = trial["requests"][0]
    rows, batch = request["rows"], request["batch"]
    if (not request["completed"] or not isinstance(batch, int) or batch != trial["num_envs"] or batch % 8 or
            len(rows) != batch or sorted(row.get("row", -1) for row in rows) != list(range(batch)) or
            any(not number(row.get("score_last")) for row in rows)):
        return unknown("request lacks unique complete G8 rows matched to trial N")
    scores = [row["score_last"] for row in sorted(rows, key=lambda row: row["row"])]
    groups = []
    for start in range(0, batch, 8):
        values = scores[start:start + 8]
        mean = statistics.fmean(values)
        boundary = min(abs(mean - 0.1), abs(mean - 0.9)) < 1e-6
        groups.append({"row_start": start, "row_end_exclusive": start + 8, "last_scores": values,
                       "mean_trajectory_return": mean, "retained": None if boundary else 0.1 <= mean <= 0.9,
                       "boundary_rounding_unknown": boundary,
                       "population_std": statistics.pstdev(values)})
    return {"status": "reconstructed", "group_size": 8, "inclusive_bounds": [0.1, 0.9], "groups": groups,
            "retained_groups": sum(group["retained"] is True for group in groups),
            "unknown_groups": sum(group["retained"] is None for group in groups),
            "basis": "One first C32: relative frame rewards telescope from previous score 0 to score_last; group rows are contiguous. Near-boundary float32 reductions stay unknown.",
            "actor_learning": "not established: filter reconstruction is not gradient/update evidence"}


METRIC_PATTERN = re.compile(r"(?P<tag>[A-Za-z][A-Za-z0-9_./-]*(?:loss|grad)[A-Za-z0-9_./-]*)[\"']?\s*[:=]\s*(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def metrics_and_checkpoints(trial, issues, tensorboard):
    directory = Path(trial["directory"])
    values, evidence = {}, []
    def collect(key, value, source):
        if ("grad" in key or "loss" in key) and number(value):
            values.setdefault(key, []).append(float(value))
            evidence.append({"key": key, "value": float(value), "source": source})
    def walk(row, prefix, source):
        if isinstance(row, dict):
            for key, value in row.items():
                if not key.startswith("_"):
                    walk(value, prefix + "/" + key if prefix else key, source)
        else:
            collect(prefix, row, source)
    for path in directory.rglob("*.jsonl") if directory.exists() else []:
        for row in read_jsonl(path, issues):
            walk(row.get("metrics", row), "", f"{path}:{row['_line']}")
    log = directory / "driver.log"
    if log.exists():
        with log.open(encoding="utf-8", errors="replace") as stream:
            for index, line in enumerate(stream, 1):
                for match in METRIC_PATTERN.finditer(line):
                    collect(match["tag"], float(match["value"]), f"{log}:{index}")
    tb_status = "disabled"
    if tensorboard:
        try:
            # TensorBoard may discover an installed TensorFlow backend. Keep
            # this analysis process CPU-only before its optional import.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            tb_status = "enabled"
            for path in directory.rglob("events.out.tfevents.*"):
                accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
                for tag in accumulator.Tags().get("scalars", []):
                    if "grad" in tag or "loss" in tag:
                        for value in accumulator.Scalars(tag):
                            collect(tag, value.value, f"{path}:step={value.step}")
        except Exception as error:
            tb_status = "unavailable:" + type(error).__name__
    cp_dirs = [path for path in directory.rglob("global_step_*") if path.is_dir() and "checkpoints" in path.parts] if directory.exists() else []
    checkpoints = []
    marker_names = {"complete.json", "checkpoint_complete.json", "checkpoint-complete.json", ".complete", "COMPLETED", "_SUCCESS"}
    for path in cp_dirs:
        markers = []
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.name in marker_names:
                claim = None
                if candidate.suffix == ".json":
                    marker = read_json(candidate, issues)
                    claim = marker.get("complete")
                    if claim is None and marker.get("status") in {"complete", "completed", "success", "failed"}:
                        claim = marker["status"] != "failed"
                markers.append({"path": str(candidate), "claims_complete": claim})
        checkpoints.append({"path": str(path), "completion_markers": markers,
                            "status": "explicit_marker_present" if markers else "unknown_no_explicit_completion_marker"})
    gradients = [value for key, series in values.items() if "grad_norm" in key for value in series]
    gradient_nonzero = any(value > 0 for value in gradients) if gradients else None
    return {"metrics": {key: summary(series) for key, series in values.items()}, "scalar_evidence": evidence,
            "tensorboard": tb_status, "nonzero_grad_norm_reported": gradient_nonzero,
            "metric_count_note": "Observations can repeat across backends; counts are not optimizer-step counts.",
            "effective_learning": "not established by logs or checkpoint presence", "checkpoints": checkpoints,
            "checkpoint_note": "Only explicit marker presence is reported; directories, save-start lines and exit code 0 do not prove checkpoint completeness."}


def analyze(owner, tensorboard=False, offset=None):
    owner, issues = Path(owner), []
    plan = read_json(owner / "owner-plan.json", issues)
    resources = read_jsonl(owner / "resources.jsonl", issues)
    events = []
    service_files = sorted(owner.rglob("service-events.jsonl"))
    for path in service_files:
        events.extend(read_jsonl(path, issues))
    if not service_files:
        issues.append({"issue": "no service-events.jsonl found"})
    trials = trial_windows(owner, plan, resources, issues, offset)
    requests = request_envelopes(events, issues, offset)
    unmatched = assign_requests(trials, requests)
    output = []
    for trial in trials:
        rows = [row for request in trial["requests"] for row in request["rows"]]
        output.append({"key": trial["key"], "num_envs": trial["num_envs"], "exit_code": trial["exit_code"],
                       "owner_trial_seconds": trial["seconds"], "attribution_window_epoch": [trial["start"], trial["end"]],
                       "attribution_basis": trial["window_basis"], "request_count": len(trial["requests"]),
                       "requests": [{key: request[key] for key in ("index", "request_id", "pid", "source", "start_line", "start", "end", "batch", "completed")} for request in trial["requests"]],
                       "wm_plus_reward_row_seconds": summary([row.get("seconds") for row in rows]),
                       "pure_wm_row_seconds": unknown("service row timer includes WM plus reward; no independent WM completion timestamp"),
                       "score_last": summary([row.get("score_last") for row in rows]),
                       "score_max": summary([row.get("score_max") for row in rows]),
                       "memory": memory_summary(trial, resources, offset),
                       "action_telemetry": action_summary(trial["requests"]),
                       "reward_filter": retained_groups(trial),
                       "training": metrics_and_checkpoints(trial, issues, tensorboard)})
    return {"analysis_kind": "opendw_smoke_v1", "owner": str(owner), "naive_time_offset_assumption": offset, "trials": output,
            "unassigned_requests": unmatched, "issues": issues,
            "limits": "This is evidence analysis, not native success evaluation. Missing metrics and ambiguous request attribution stay unknown."}


def markdown(report):
    def fmt(value, scale=1):
        return f"{value / scale:.2f}" if number(value) else "unknown"
    lines = ["# OpenDW smoke 简报", "", "仅汇总已有证据；奖励模型分数不等于原生成功率。", "",
             "|试验|总耗时s|WM+奖励每行均值s|我方计算显存采样峰值GiB|服务RSS/PSS峰值GiB|末分数均值|保留G8组|", "|---|---:|---:|---:|---:|---:|---|"]
    for trial in report["trials"]:
        memory, reward_filter = trial["memory"], trial["reward_filter"]
        groups = f"{reward_filter['retained_groups']}/{len(reward_filter['groups'])}，未知{reward_filter['unknown_groups']}" if reward_filter.get("status") == "reconstructed" else "unknown"
        lines.append(f"|{trial['key']}|{fmt(trial['owner_trial_seconds'])}|{fmt(trial['wm_plus_reward_row_seconds'].get('mean'))}|{fmt(memory['managed_compute_memory_sample_peak_bytes'], 1024**3)}|{fmt(memory['service_rss_sample_peak_bytes'], 1024**3)}/{fmt(memory['service_pss_sample_peak_bytes'], 1024**3)}|{fmt(trial['score_last'].get('mean'))}|{groups}|")
    lines.append("")
    for trial in report["trials"]:
        training = trial["training"]
        clipping = trial["action_telemetry"]["actions"].get("abs_z_gt5_fraction")
        metric_text = ", ".join(f"{key}={value['mean']:.6g}" for key, value in training["metrics"].items()) or "unknown"
        markers = sum(bool(cp["completion_markers"]) for cp in training["checkpoints"])
        lines.append(f"- {trial['key']}：动作归一化越界比例 {fmt(clipping)}；grad/loss：{metric_text}；显式CP标记 {markers} 个；不据此宣称有效学习。")
    lines.extend(["", f"未能明确归属的请求：{len(report['unassigned_requests'])}；输入问题：{len(report['issues'])}。详细原因见JSON。",
                  "纯WM耗时未单独记录；表内耗时含奖励推理。显存/RSS均有采样边界，RSS求和会重复计算共享页。", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--naive-offset", default=None, help="Explicit timezone for otherwise unassignable naive timestamps, e.g. +08:00")
    args = parser.parse_args()
    if args.naive_offset is not None and not re.fullmatch(r"[+-](?:0\d|1[0-4]):[0-5]\d", args.naive_offset):
        parser.error("--naive-offset must be ±HH:MM")
    report = analyze(args.owner, args.tensorboard, args.naive_offset)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    # Output paths are explicit; do not overwrite any source evidence file.
    sources = {str(args.owner / "owner-plan.json"), str(args.owner / "resources.jsonl")}
    if str(json_path) in sources or json_path.name in {"service-events.jsonl", "driver-finished.json", "result.json"}:
        parser.error("report path must not overwrite evidence")
    if json_path.exists():
        try:
            previous_kind = json.loads(json_path.read_text(encoding="utf-8")).get("analysis_kind")
        except (OSError, ValueError, AttributeError):
            previous_kind = None
        if previous_kind != "opendw_smoke_v1":
            parser.error("existing output is not an analyzer report; refusing to overwrite evidence")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "trials": len(report["trials"]),
                      "unassigned_requests": len(report["unassigned_requests"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
