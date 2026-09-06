"""Optional action-progress video trace for the RLinf RoboTwin control loop.

The recorder is deliberately a side channel: callers decide whether a frame is
needed before asking the camera to render it.  It neither changes the action
chunk nor calls the planner, TOPP, ``scene.step()``, or ``_update_render()``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _as_int_set(values) -> set[int]:
    if values is None:
        return set()
    if isinstance(values, (int, np.integer)):
        return {int(values)}
    return {int(value) for value in values}


def is_trace_selected(config: dict[str, Any]) -> bool:
    """Return whether this deterministic worker/slot/episode is selected."""

    if not bool(config.get("enabled", False)):
        return False

    worker_index = int(config.get("worker_index", -1))
    env_slot = int(config.get("env_slot", -1))
    episode_index = int(config.get("episode_index_within_slot", -1))
    worker_indices = _as_int_set(config.get("worker_indices", [0]))
    env_slots = _as_int_set(config.get("env_slots", [0]))
    max_episodes = int(config.get("max_episodes_per_slot", 1))

    return (
        worker_index in worker_indices
        and env_slot in env_slots
        and 0 <= episode_index < max_episodes
    )


def claim_recording_index(config: dict[str, Any]) -> Optional[int]:
    """Claim one persistent recording slot across VectorEnv recreation.

    RLinf offloads and rebuilds RoboTwin environments after a runner step.  A
    small exclusive marker in the run output keeps ``max_episodes_per_slot`` a
    run-level budget rather than resetting it with every new ``SubEnv``.
    """

    output_dir = Path(os.path.expanduser(str(config["output_dir"])))
    claim_dir = output_dir / ".claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    worker_index = int(config["worker_index"])
    env_slot = int(config["env_slot"])
    max_episodes = int(config.get("max_episodes_per_slot", 1))
    for recording_index in range(max_episodes):
        claim_path = claim_dir / (
            f"worker_{worker_index:03d}_env_slot_{env_slot:03d}_"
            f"recording_{recording_index:04d}.claim"
        )
        try:
            descriptor = os.open(
                claim_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        return recording_index
    return None


def progress_to_action_mapping(progress: float, chunk_len: int) -> dict[str, float | int]:
    """Map synchronized arm progress to an approximate model action index."""

    chunk_len = max(1, int(chunk_len))
    progress = float(np.clip(progress, 0.0, 1.0))
    h_float = progress * float(chunk_len - 1)
    h_lo = int(math.floor(h_float))
    h_hi = min(chunk_len - 1, h_lo + 1)
    return {
        "progress": progress,
        "h_float": h_float,
        "h_lo": h_lo,
        "h_hi": h_hi,
        "h_fraction": h_float - h_lo,
    }


@dataclass(frozen=True)
class CaptureDecision:
    reason: str
    mapping: dict[str, float | int]


class ProgressBinSampler:
    """Take the first frame in approximate action bins plus the query end."""

    def __init__(self, chunk_len: int, max_frames: int):
        self.chunk_len = max(1, int(chunk_len))
        self.max_frames = max(1, int(max_frames))
        self._seen_bins: set[int] = set()
        self._frames = 0

    def select(
        self,
        progress: float,
        *,
        query_end: bool = False,
        success: bool = False,
    ) -> Optional[CaptureDecision]:
        mapping = progress_to_action_mapping(progress, self.chunk_len)

        # Reserve one slot for a true query-end/success frame.  For C50 and a
        # 50-frame cap this yields up to 49 progress frames plus one end frame.
        if query_end or success:
            if self._frames >= self.max_frames:
                return None
            self._frames += 1
            return CaptureDecision("success" if success else "query_end", mapping)

        if self._frames >= max(0, self.max_frames - 1):
            return None
        bin_index = min(self.chunk_len - 2, int(mapping["h_lo"]))
        if self.chunk_len == 1:
            bin_index = 0
        if bin_index in self._seen_bins:
            return None
        self._seen_bins.add(bin_index)
        self._frames += 1
        return CaptureDecision("progress", mapping)


class ControlTraceRecorder:
    """One MP4 plus one frame-index CSV for a selected simulator episode."""

    CSV_FIELDS = (
        "frame_idx",
        "worker_index",
        "env_slot",
        "episode_index_within_slot",
        "recording_index",
        "reset_id",
        "query_idx",
        "action_slot_start",
        "chunk_len",
        "control_idx",
        "physics_idx",
        "left_control_idx",
        "left_control_len",
        "right_control_idx",
        "right_control_len",
        "left_progress",
        "right_progress",
        "progress",
        "h_float",
        "h_lo",
        "h_hi",
        "h_fraction",
        "success_after",
        "first_success",
        "query_end",
        "capture_reason",
    )

    @classmethod
    def from_config(
        cls,
        config: Optional[dict[str, Any]],
        *,
        reset_id: int,
        task_name: str,
    ) -> Optional["ControlTraceRecorder"]:
        config = dict(config or {})
        if not is_trace_selected(config):
            return None
        recording_index = claim_recording_index(config)
        if recording_index is None:
            return None
        config["recording_index"] = recording_index
        return cls(config, reset_id=reset_id, task_name=task_name)

    def __init__(self, config: dict[str, Any], *, reset_id: int, task_name: str):
        self.config = dict(config)
        self.reset_id = int(reset_id)
        self.task_name = str(task_name)
        self.worker_index = int(config["worker_index"])
        self.env_slot = int(config["env_slot"])
        self.episode_index = int(config["episode_index_within_slot"])
        self.recording_index = int(config["recording_index"])
        self.camera = str(config.get("camera", "head_camera"))
        self.fps = int(config.get("fps", 10))
        self.crf = int(config.get("crf", 32))
        self.output_width = int(config.get("output_width", 160))
        self.output_height = int(config.get("output_height", 120))
        self.max_frames_per_episode = int(config.get("max_frames_per_episode", 200))
        self.max_frames_per_query = int(config.get("max_frames_per_query", 50))

        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.task_name)
        episode_dir = (
            Path(os.path.expanduser(str(config["output_dir"])))
            / safe_task
            / f"worker_{self.worker_index:03d}"
            / f"env_slot_{self.env_slot:03d}"
            / (
                f"recording_{self.recording_index:04d}_"
                f"episode_{self.episode_index:04d}_reset_{self.reset_id}"
            )
        )
        episode_dir.mkdir(parents=True, exist_ok=False)
        self.episode_dir = episode_dir
        self.video_path = episode_dir / f"{self.camera}.mp4"
        self.csv_path = episode_dir / "frames.csv"
        self.metadata_path = episode_dir / "metadata.json"

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv_writer.writeheader()
        self._encoder: Optional[subprocess.Popen] = None
        self._encoder_error: Optional[str] = None
        self._source_shape: Optional[tuple[int, int]] = None
        self._sampler: Optional[ProgressBinSampler] = None
        self._query_idx: Optional[int] = None
        self._query_action_slot_start = 0
        self._query_chunk_len = 0
        self._frame_idx = 0
        self._success_seen = False
        self._closed = False

    def start_query(self, *, query_idx: int, action_slot_start: int, chunk_len: int) -> None:
        self._query_idx = int(query_idx)
        self._query_action_slot_start = int(action_slot_start)
        self._query_chunk_len = int(chunk_len)
        remaining = max(0, self.max_frames_per_episode - self._frame_idx)
        self._sampler = ProgressBinSampler(
            chunk_len=chunk_len,
            max_frames=min(self.max_frames_per_query, max(1, remaining)),
        )

    def select_frame(
        self,
        *,
        control_idx: int,
        physics_idx: int,
        left_control_idx: int,
        left_control_len: int,
        right_control_idx: int,
        right_control_len: int,
        success_after: bool,
        query_end: bool,
    ) -> Optional[dict[str, Any]]:
        if (
            self._closed
            or self._sampler is None
            or self._query_idx is None
            or self._frame_idx >= self.max_frames_per_episode
        ):
            return None

        left_progress = min(1.0, float(left_control_idx) / max(1, left_control_len))
        right_progress = min(1.0, float(right_control_idx) / max(1, right_control_len))
        progress = min(left_progress, right_progress)
        decision = self._sampler.select(
            progress,
            query_end=bool(query_end),
            success=bool(success_after),
        )
        if decision is None:
            return None

        first_success = bool(success_after) and not self._success_seen
        self._success_seen = self._success_seen or bool(success_after)
        return {
            "frame_idx": self._frame_idx,
            "worker_index": self.worker_index,
            "env_slot": self.env_slot,
            "episode_index_within_slot": self.episode_index,
            "recording_index": self.recording_index,
            "reset_id": self.reset_id,
            "query_idx": self._query_idx,
            "action_slot_start": self._query_action_slot_start,
            "chunk_len": self._query_chunk_len,
            "control_idx": int(control_idx),
            "physics_idx": int(physics_idx),
            "left_control_idx": int(left_control_idx),
            "left_control_len": int(left_control_len),
            "right_control_idx": int(right_control_idx),
            "right_control_len": int(right_control_len),
            "left_progress": left_progress,
            "right_progress": right_progress,
            **decision.mapping,
            "success_after": int(bool(success_after)),
            "first_success": int(first_success),
            "query_end": int(bool(query_end)),
            "capture_reason": decision.reason,
        }

    def write_frame(self, frame: np.ndarray, row: dict[str, Any]) -> None:
        if self._closed:
            return
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame [H,W,3], got {frame.shape}")
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and frame.size and frame.max() <= 1.0:
                frame = frame * 255.0
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        if self._encoder is None and self._encoder_error is None:
            self._start_encoder(frame.shape[1], frame.shape[0])
        if self._encoder is not None:
            try:
                self._encoder.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as exc:
                self._encoder_error = f"{type(exc).__name__}: {exc}"
                self._close_encoder()

        self._csv_writer.writerow({field: row.get(field, "") for field in self.CSV_FIELDS})
        self._csv_file.flush()
        self._frame_idx += 1

    def _start_encoder(self, width: int, height: int) -> None:
        self._source_shape = (height, width)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vf",
            f"scale={self.output_width}:{self.output_height}:flags=bicubic",
            "-vcodec",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            str(self.video_path),
        ]
        try:
            self._encoder = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            self._encoder_error = f"{type(exc).__name__}: {exc}"

    def finish(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_encoder()
        self._csv_file.close()
        metadata = {
            "schema_version": 1,
            "task_name": self.task_name,
            "worker_index": self.worker_index,
            "env_slot": self.env_slot,
            "episode_index_within_slot": self.episode_index,
            "recording_index": self.recording_index,
            "reset_id": self.reset_id,
            "camera": self.camera,
            "frame_count": self._frame_idx,
            "first_success_recorded": self._success_seen,
            "finish_reason": str(reason),
            "source_height": None if self._source_shape is None else self._source_shape[0],
            "source_width": None if self._source_shape is None else self._source_shape[1],
            "output_width": self.output_width,
            "output_height": self.output_height,
            "fps": self.fps,
            "crf": self.crf,
            "encoder_error": self._encoder_error,
            "video_path": str(self.video_path),
            "frames_csv": str(self.csv_path),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _close_encoder(self) -> None:
        encoder = self._encoder
        self._encoder = None
        if encoder is None:
            return
        try:
            if encoder.stdin is not None:
                try:
                    encoder.stdin.close()
                except OSError:
                    pass
            return_code = encoder.wait(timeout=15)
            if return_code != 0 and self._encoder_error is None:
                error = b"" if encoder.stderr is None else encoder.stderr.read()
                self._encoder_error = error.decode("utf-8", errors="replace").strip()
        except subprocess.TimeoutExpired:
            encoder.terminate()
            try:
                encoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()
            self._encoder_error = self._encoder_error or "ffmpeg close timeout"

    def close(self) -> None:
        self.finish("environment_close")
