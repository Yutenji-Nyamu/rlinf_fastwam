#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
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

"""Sample host/cgroup, GPU, and RLinf worker resource usage for one run.

The monitor is intentionally read-only.  It writes a two-second (configurable)
CSV time series plus an atomically refreshed peak.txt beside the training log.
It stops after recording one final sample once the target driver PID exits.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


MIB = 1024 * 1024
CSV_FIELDS = (
    "timestamp",
    "cgroup_ram_mb",
    "cgroup_limit_mb",
    "cgroup_ram_pct",
    "shm_used_mb",
    "gpu0_memory_mb",
    "gpu0_util_pct",
    "gpu0_power_w",
    "gpu1_memory_mb",
    "gpu1_util_pct",
    "gpu1_power_w",
    "gpu_total_memory_mb",
    "env_rss_mb",
    "actor_rss_mb",
    "rollout_rss_mb",
    "driver_rss_mb",
    "ray_system_rss_mb",
    "top_pid",
    "top_rss_mb",
    "top_command",
    "cgroup_oom",
    "cgroup_oom_kill",
)


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    rss_mb: float
    command: str


@dataclass
class Peaks:
    ram_mb: int = 0
    ram_time: str = ""
    gpu0_mb: int = 0
    gpu1_mb: int = 0
    gpu_total_mb: int = 0
    gpu_time: str = ""
    env_rss_mb: float = 0.0
    actor_rss_mb: float = 0.0
    rollout_rss_mb: float = 0.0

    def update(self, row: dict[str, object]) -> None:
        timestamp = str(row["timestamp"])
        ram_mb = int(row["cgroup_ram_mb"])
        gpu0_mb = int(row["gpu0_memory_mb"])
        gpu1_mb = int(row["gpu1_memory_mb"])
        gpu_total_mb = int(row["gpu_total_memory_mb"])

        if ram_mb >= self.ram_mb:
            self.ram_mb = ram_mb
            self.ram_time = timestamp
        if (
            gpu0_mb > self.gpu0_mb
            or gpu1_mb > self.gpu1_mb
            or gpu_total_mb >= self.gpu_total_mb
        ):
            self.gpu0_mb = max(self.gpu0_mb, gpu0_mb)
            self.gpu1_mb = max(self.gpu1_mb, gpu1_mb)
            self.gpu_total_mb = max(self.gpu_total_mb, gpu_total_mb)
            self.gpu_time = timestamp
        self.env_rss_mb = max(self.env_rss_mb, float(row["env_rss_mb"]))
        self.actor_rss_mb = max(self.actor_rss_mb, float(row["actor_rss_mb"]))
        self.rollout_rss_mb = max(self.rollout_rss_mb, float(row["rollout_rss_mb"]))


class CgroupMemory:
    """Resolve memory counters for both cgroup v2 and legacy cgroup v1."""

    def __init__(self) -> None:
        self.current_path: Path | None = None
        self.limit_path: Path | None = None
        self.events_path: Path | None = None
        self.failcnt_path: Path | None = None
        self._resolve()

    def _resolve(self) -> None:
        cgroup_entries: list[tuple[str, str]] = []
        try:
            for line in Path("/proc/self/cgroup").read_text().splitlines():
                _, controllers, relative = line.split(":", 2)
                cgroup_entries.append((controllers, relative.lstrip("/")))
        except (OSError, ValueError):
            pass

        v2_rel = next(
            (rel for controllers, rel in cgroup_entries if not controllers), ""
        )
        v2_candidates = (Path("/sys/fs/cgroup") / v2_rel, Path("/sys/fs/cgroup"))
        for base in v2_candidates:
            current = base / "memory.current"
            if current.is_file():
                self.current_path = current
                self.limit_path = base / "memory.max"
                self.events_path = base / "memory.events"
                return

        v1_rel = next(
            (
                rel
                for controllers, rel in cgroup_entries
                if "memory" in controllers.split(",")
            ),
            "",
        )
        v1_candidates = (
            Path("/sys/fs/cgroup/memory") / v1_rel,
            Path("/sys/fs/cgroup") / v1_rel,
            Path("/sys/fs/cgroup/memory"),
        )
        for base in v1_candidates:
            current = base / "memory.usage_in_bytes"
            if current.is_file():
                self.current_path = current
                self.limit_path = base / "memory.limit_in_bytes"
                self.failcnt_path = base / "memory.failcnt"
                return

    @staticmethod
    def _read_int(path: Path | None, default: int = 0) -> int:
        if path is None:
            return default
        try:
            value = path.read_text().strip()
            return default if value == "max" else int(value)
        except (OSError, ValueError):
            return default

    @staticmethod
    def _host_total_bytes() -> int:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return 0

    def sample(self) -> tuple[int, int, int, int]:
        current = self._read_int(self.current_path)
        limit = self._read_int(self.limit_path)
        host_total = self._host_total_bytes()
        # cgroup v1 often exposes an effectively unlimited sentinel.
        if limit <= 0 or (host_total > 0 and limit > host_total * 100):
            limit = host_total

        oom = 0
        oom_kill = 0
        if self.events_path is not None:
            try:
                events = {
                    key: int(value)
                    for key, value in (
                        line.split(maxsplit=1)
                        for line in self.events_path.read_text().splitlines()
                    )
                }
                oom = events.get("oom", 0)
                oom_kill = events.get("oom_kill", 0)
            except (OSError, ValueError):
                pass
        elif self.failcnt_path is not None:
            # v1 has no portable cumulative oom_kill counter; failcnt is the
            # closest conservative signal and is reported as cgroup_oom.
            oom = self._read_int(self.failcnt_path)
        return current, limit, oom, oom_kill


def _read_start_time(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        tail = stat[stat.rfind(")") + 2 :].split()
        if not tail or tail[0] == "Z":
            return None
        return tail[19]
    except (OSError, IndexError):
        return None


def _process_alive(pid: int, expected_start_time: str | None) -> bool:
    if expected_start_time is None:
        return False
    start_time = _read_start_time(pid)
    if start_time is None:
        return False
    return start_time == expected_start_time


def _processes() -> list[ProcessInfo]:
    results: list[ProcessInfo] = []
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError):
        page_size = 4096

    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            stat = (proc_dir / "stat").read_text()
            tail = stat[stat.rfind(")") + 2 :].split()
            if not tail or tail[0] == "Z":
                continue
            rss_mb = int(tail[21]) * page_size / MIB
            cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
            command = " ".join(cmdline.split())
            if not command:
                command = stat[stat.find("(") + 1 : stat.rfind(")")]
            results.append(ProcessInfo(int(proc_dir.name), rss_mb, command))
        except (OSError, ValueError, IndexError):
            # Processes can disappear at any point during /proc traversal.
            continue
    return results


def _sum_matching(processes: Iterable[ProcessInfo], needles: tuple[str, ...]) -> float:
    return sum(
        process.rss_mb
        for process in processes
        if any(needle in process.command for needle in needles)
    )


def _gpu_sample() -> list[tuple[int, int, float]]:
    command = (
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return [(0, 0, 0.0), (0, 0, 0.0)]

    gpus: list[tuple[int, int, float]] = []
    for line in result.stdout.splitlines():
        try:
            memory, utilization, power = (part.strip() for part in line.split(","))
            gpus.append((int(float(memory)), int(float(utilization)), float(power)))
        except (ValueError, TypeError):
            gpus.append((0, 0, 0.0))
    return (gpus + [(0, 0, 0.0), (0, 0, 0.0)])[:2]


def _sample(target_pid: int, cgroup: CgroupMemory) -> dict[str, object]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_bytes, limit_bytes, oom, oom_kill = cgroup.sample()
    current_mb = round(current_bytes / MIB)
    limit_mb = round(limit_bytes / MIB)
    ram_pct = round(100 * current_bytes / limit_bytes, 2) if limit_bytes else 0.0
    try:
        shm_used_mb = round(shutil.disk_usage("/dev/shm").used / MIB)
    except OSError:
        shm_used_mb = 0

    (gpu0_mb, gpu0_util, gpu0_power), (gpu1_mb, gpu1_util, gpu1_power) = _gpu_sample()
    processes = _processes()
    process_by_pid = {process.pid: process for process in processes}
    top = max(
        processes,
        key=lambda process: process.rss_mb,
        default=ProcessInfo(0, 0.0, ""),
    )

    env_rss = _sum_matching(processes, ("EnvWorker",))
    actor_rss = _sum_matching(processes, ("EmbodiedFSDPActor",))
    rollout_rss = _sum_matching(processes, ("MultiStepRolloutWorker",))
    driver_rss = process_by_pid.get(target_pid, ProcessInfo(target_pid, 0.0, "")).rss_mb
    ray_system_rss = _sum_matching(
        processes,
        (
            "raylet",
            "gcs_server",
            "dashboard.py",
            "dashboard/agent.py",
            "log_monitor.py",
            "runtime_env/agent/main.py",
        ),
    )

    return {
        "timestamp": timestamp,
        "cgroup_ram_mb": current_mb,
        "cgroup_limit_mb": limit_mb,
        "cgroup_ram_pct": ram_pct,
        "shm_used_mb": shm_used_mb,
        "gpu0_memory_mb": gpu0_mb,
        "gpu0_util_pct": gpu0_util,
        "gpu0_power_w": round(gpu0_power, 2),
        "gpu1_memory_mb": gpu1_mb,
        "gpu1_util_pct": gpu1_util,
        "gpu1_power_w": round(gpu1_power, 2),
        "gpu_total_memory_mb": gpu0_mb + gpu1_mb,
        "env_rss_mb": round(env_rss, 1),
        "actor_rss_mb": round(actor_rss, 1),
        "rollout_rss_mb": round(rollout_rss, 1),
        "driver_rss_mb": round(driver_rss, 1),
        "ray_system_rss_mb": round(ray_system_rss, 1),
        "top_pid": top.pid,
        "top_rss_mb": round(top.rss_mb, 1),
        "top_command": top.command,
        "cgroup_oom": oom,
        "cgroup_oom_kill": oom_kill,
    }


def _write_peak(
    path: Path,
    target_pid: int,
    process_alive: bool,
    row: dict[str, object],
    peaks: Peaks,
    csv_path: Path,
) -> None:
    contents = f"""updated_at={row['timestamp']}
target_pid={target_pid}
process_alive={process_alive}

current_ram_mb={row['cgroup_ram_mb']}
cgroup_limit_mb={row['cgroup_limit_mb']}
current_ram_pct={row['cgroup_ram_pct']}
peak_ram_mb={peaks.ram_mb}
peak_ram_time={peaks.ram_time}

current_gpu0_mb={row['gpu0_memory_mb']}
current_gpu1_mb={row['gpu1_memory_mb']}
current_gpu_total_mb={row['gpu_total_memory_mb']}
peak_gpu0_mb={peaks.gpu0_mb}
peak_gpu1_mb={peaks.gpu1_mb}
peak_gpu_total_mb={peaks.gpu_total_mb}
peak_gpu_time={peaks.gpu_time}

current_env_rss_mb={row['env_rss_mb']}
current_actor_rss_mb={row['actor_rss_mb']}
current_rollout_rss_mb={row['rollout_rss_mb']}
peak_env_rss_mb={peaks.env_rss_mb:.1f}
peak_actor_rss_mb={peaks.actor_rss_mb:.1f}
peak_rollout_rss_mb={peaks.rollout_rss_mb:.1f}

current_shm_used_mb={row['shm_used_mb']}
top_pid={row['top_pid']}
top_rss_mb={row['top_rss_mb']}
top_command={row['top_command']}

cgroup_oom={row['cgroup_oom']}
cgroup_oom_kill={row['cgroup_oom_kill']}

csv={csv_path}
"""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor cgroup RAM, two GPUs, and RLinf worker RSS until a driver "
            "PID exits."
        )
    )
    parser.add_argument(
        "--pid", type=int, required=True, help="training driver PID to follow"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="resource artifact directory"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, help="sample interval in seconds"
    )
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (args.out_dir / "resources.csv").resolve()
    peak_path = args.out_dir / "peak.txt"
    cgroup = CgroupMemory()
    peaks = Peaks()
    target_start_time = _read_start_time(args.pid)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    samples = 0
    with csv_path.open("w", newline="", buffering=1) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        while True:
            row = _sample(args.pid, cgroup)
            alive = _process_alive(args.pid, target_start_time)
            peaks.update(row)
            writer.writerow(row)
            csv_file.flush()
            _write_peak(peak_path, args.pid, alive, row, peaks, csv_path)
            samples += 1

            # The first sample made after the PID disappears is the final one.
            if not alive or stop_requested:
                break
            time.sleep(args.interval)

    print(
        f"resource monitor stopped: pid={args.pid} samples={samples} csv={csv_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
