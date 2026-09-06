from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[4]
EXTRACTED = HERE.parent / "extracted"

RESOURCE_CSV = next(EXTRACTED.rglob("snapshot_resources_g23.csv"))
PROCESS_TSV = next(EXTRACTED.rglob("snapshot_process_rss_g23.tsv"))
REFERENCE_RESOURCE_CSV = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step22_20260821/runtime/resources.csv"
)
REFERENCE_PROCESS_TSV = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step22_20260821/runtime/process_rss.tsv"
)
BASE_ANALYZER = (
    WORKSPACE
    / "docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/formal_live_step22_20260821/analysis/analyze_resources.py"
)
SUCCESSFUL_GRPO_ANALYSIS_JSON = WORKSPACE / "audits/20260717-084926-grpo-current/analysis.json"


def load_base_analyzer():
    spec = importlib.util.spec_from_file_location("idea2_resource_base", BASE_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_ANALYZER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RESOURCE_CSV = RESOURCE_CSV
    module.PROCESS_TSV = PROCESS_TSV
    module.JSON_OUT = HERE / "RESOURCE_SUMMARY.json"
    module.PNG_OUT = HERE / "RESOURCE_OVERVIEW_G23.png"
    # Confirmed by the paired live read-only audit at 2026-08-22T09:59:55+08:00.
    module.CGROUP_LIMIT_GIB = 240.0
    return module


def read_resources(path: Path, max_elapsed: float | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    numeric = [column for column in frame.columns if column != "timestamp"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if max_elapsed is not None:
        frame = frame[frame["elapsed_s"] <= max_elapsed]
    return frame.sort_values(["elapsed_s", "gpu_index"]).reset_index(drop=True)


def read_process(path: Path, classify_process, max_elapsed: float | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    for column in ["pid", "ppid", "rss_kib", "pcpu"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "pid", "rss_kib"]).copy()
    frame["pid"] = frame["pid"].astype(int)
    frame["rss_gib"] = frame["rss_kib"] / 1024**2
    frame["role"] = frame.apply(classify_process, axis=1)
    first = frame["timestamp"].min()
    frame["relative_elapsed_s"] = (frame["timestamp"] - first).dt.total_seconds()
    if max_elapsed is not None:
        frame = frame[frame["relative_elapsed_s"] <= max_elapsed]
    return frame.sort_values("timestamp").reset_index(drop=True)


def rounded(value: float) -> float:
    return round(float(value), 3)


def compact_resource_summary(resources: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"elapsed_seconds": int(resources["elapsed_s"].max()), "gpu": {}}
    for gpu_index, frame in resources.groupby("gpu_index", sort=True):
        frame = frame.sort_values("elapsed_s")
        result["gpu"][str(int(gpu_index))] = {
            "memory_peak_gib": rounded(frame["gpu_memory_used_mib"].max() / 1024),
            "memory_latest_gib": rounded(frame.iloc[-1]["gpu_memory_used_mib"] / 1024),
            "util_mean_pct": rounded(frame["gpu_util_pct"].mean()),
            "util_p90_pct": rounded(frame["gpu_util_pct"].quantile(0.90)),
        }
    time = resources.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s").copy()
    for source, dest in {
        "cgroup_current_bytes": "current_gib",
        "cgroup_anon_bytes": "anon_gib",
        "cgroup_file_bytes": "file_gib",
    }.items():
        time[dest] = time[source] / 1024**3
    result["cgroup"] = {
        "current_initial_gib": rounded(time.iloc[0]["current_gib"]),
        "current_peak_gib": rounded(time["current_gib"].max()),
        "current_latest_gib": rounded(time.iloc[-1]["current_gib"]),
        "anon_initial_gib": rounded(time.iloc[0]["anon_gib"]),
        "anon_peak_gib": rounded(time["anon_gib"].max()),
        "anon_latest_gib": rounded(time.iloc[-1]["anon_gib"]),
        "file_initial_gib": rounded(time.iloc[0]["file_gib"]),
        "file_peak_gib": rounded(time["file_gib"].max()),
        "file_latest_gib": rounded(time.iloc[-1]["file_gib"]),
        "event_latest": {
            key: int(time.iloc[-1][key])
            for key in ["event_low", "event_high", "event_max", "event_oom", "event_oom_kill"]
        },
    }
    result["availability"] = {
        "host_memory_min_gib": rounded(time["host_mem_available_kib"].min() / 1024**2),
        "host_memory_latest_gib": rounded(time.iloc[-1]["host_mem_available_kib"] / 1024**2),
        "disk_initial_gib": rounded(time.iloc[0]["disk_available_kib"] / 1024**2),
        "disk_latest_gib": rounded(time.iloc[-1]["disk_available_kib"] / 1024**2),
    }
    return result


def compact_process_summary(process: pd.DataFrame) -> dict[str, object]:
    role_result: dict[str, object] = {}
    for role in ["EnvWorker", "FSDPActor", "RolloutWorker", "Driver"]:
        role_frame = process[process["role"] == role]
        if role_frame.empty:
            continue
        aggregate = role_frame.groupby("timestamp", as_index=False)["rss_gib"].sum().sort_values("timestamp")
        per_pid = []
        for pid, pid_frame in role_frame.groupby("pid"):
            pid_frame = pid_frame.sort_values("timestamp")
            per_pid.append(
                {
                    "pid": int(pid),
                    "peak_rss_gib": rounded(pid_frame["rss_gib"].max()),
                    "latest_rss_gib": rounded(pid_frame.iloc[-1]["rss_gib"]),
                }
            )
        role_result[role] = {
            "aggregate_peak_rss_gib": rounded(aggregate["rss_gib"].max()),
            "aggregate_latest_rss_gib": rounded(aggregate.iloc[-1]["rss_gib"]),
            "per_pid": sorted(per_pid, key=lambda item: item["peak_rss_gib"], reverse=True),
        }
    return role_result


def main() -> None:
    analyzer = load_base_analyzer()
    summary, resources, cgroup, critical_workers = analyzer.load_and_summarize()
    current_elapsed = float(resources["elapsed_s"].max())

    current_process = read_process(PROCESS_TSV, analyzer.classify_process)
    reference_resources = read_resources(REFERENCE_RESOURCE_CSV, max_elapsed=current_elapsed)
    reference_process = read_process(
        REFERENCE_PROCESS_TSV,
        analyzer.classify_process,
        max_elapsed=current_elapsed,
    )

    summary["live_limit_probe"] = {
        "timestamp": "2026-08-22T09:59:55+08:00",
        "memory_current_bytes": 233_413_758_976,
        "memory_current_gib": rounded(233_413_758_976 / 1024**3),
        "memory_max_bytes": 257_698_037_760,
        "memory_max_gib": 240.0,
        "memory_swap_current_bytes": 0,
        "events": {"low": 0, "high": 0, "max": 0, "oom": 0, "oom_kill": 0},
        "disk": {"filesystem": "/dev/md0", "available_reported": "737G", "used_pct": 65},
    }
    summary["reference_limit_note"] = (
        "memory.max=257698037760 bytes (240 GiB) was live-verified at "
        "2026-08-22T09:59:55+08:00; it is not a field in the archived resources CSV."
    )
    summary["assessment"]["memory_limit_provenance"] = (
        "Live read-only cgroup probe at 2026-08-22T09:59:55+08:00 confirmed memory.max=240 GiB."
    )
    summary["historical_comparison_v1_same_elapsed"] = {
        "comparison_elapsed_seconds": int(current_elapsed),
        "current_r_only": {
            "resources": compact_resource_summary(resources),
            "process_rss": compact_process_summary(current_process),
        },
        "v1_global_zscore": {
            "resources": compact_resource_summary(reference_resources),
            "process_rss": compact_process_summary(reference_process),
        },
        "interpretation": (
            "Same elapsed-time engineering comparison only. RSS can double-count shared pages; "
            "cgroup current is the container-total authority."
        ),
    }
    successful_grpo = json.loads(SUCCESSFUL_GRPO_ANALYSIS_JSON.read_text(encoding="utf-8"))
    successful_peak = successful_grpo["peak"]
    summary["historical_successful_grpo_full_run_resources"] = {
        "scope": "complete 100-step historical engineering context, not same-elapsed comparison",
        "cgroup_peak_gib": rounded(successful_peak["peak_ram_mb"] / 1024),
        "gpu0_peak_gib": rounded(successful_peak["peak_gpu0_mb"] / 1024),
        "gpu1_peak_gib": rounded(successful_peak["peak_gpu1_mb"] / 1024),
        "envworker_aggregate_peak_rss_gib": rounded(successful_peak["peak_env_rss_mb"] / 1024),
        "actor_aggregate_peak_rss_gib": rounded(successful_peak["peak_actor_rss_mb"] / 1024),
        "rollout_aggregate_peak_rss_gib": rounded(successful_peak["peak_rollout_rss_mb"] / 1024),
        "cgroup_oom": int(successful_peak["cgroup_oom"]),
        "cgroup_oom_kill": int(successful_peak["cgroup_oom_kill"]),
    }

    analyzer.JSON_OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    analyzer.render_plot(summary, resources, cgroup, critical_workers)
    print(analyzer.JSON_OUT)
    print(analyzer.PNG_OUT)


if __name__ == "__main__":
    main()
