"""Offline analysis for Shenzhen pi0 and Fast-WAM DVAC observations.

The source telemetry is read-only.  All derived tables, figures, and optional
storyboards are written to a new output directory.  The two policies keep
their model-native raw scales: raw ``V``/``y`` is never pooled across models.

The common analysis signal is::

    V_L(q, h) = sum_d Var_i(z_i(q, h, d)),  i in the last L endpoints
    y(q, h)   = log(V_L(q, h) + 1e-12)

Within each (policy, checkpoint, task, physical run, cohort, L, h) group, using only
``success_before == false`` queries for the position baseline, the script
then computes::

    b_h = median_q y(q, h)
    s_h = max(1.4826 * MAD_q y(q, h), scale_floor)
    r   = y - b_h
    R   = r / s_h

The raw-unit and standardized four-way ledgers are both retained::

    mu       = mean_h b_h
    P_h      = b_h - mu
    S_raw(q) = mean_h r(q, h)
    I_raw    = r - S_raw

    S_std(q) = mean_h R(q, h)
    I_std    = R - S_std

The exact identities are ``y = b + r``,
``y = mu + P_h + S_raw + I_raw``, and ``R = S_std + I_std``.  In
particular, ``y = b + S_std + I_std`` is dimensionally wrong and is not
emitted by this tool.

pi0 input contract
------------------
``trace_rollout_rankNN.npz`` plus matching
``query_index_rollout_rankNN.csv`` and ``episode_index_env_rankNN.csv``.
Both one-rank consolidated and multiple rank-local shards are accepted.

Fast-WAM input contract
-----------------------
``queries.csv`` and ``episodes.csv`` plus one NPZ per query.  Fast-WAM
endpoints are reconstructed from the raw chain as
``z = x_chain[:-1] - (timesteps / 1000) * v_chain``.
The denominator is required from manifest ``action_num_train_timesteps``;
the explicit legacy key ``num_train_timesteps`` is accepted with a recorded
compatibility note.  An optional official seed-map CSV can fill missing
``source_seed`` values in memory; source payloads are never rewritten.

Fast-WAM action/video alignment is emitted only when the telemetry contains
fresh per-action frame ranges, and only for actually executed ``h < 24``.
pi0's official combined video is query-boundary evidence only, so this script
never invents a pi0 future-h-to-physics-frame mapping.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


EPS = 1e-12
COMMON_L = 3
DEFAULT_SCALE_FLOOR = 1e-6
PI0_TAIL_LENGTHS = (2, 3, 4)
FASTWAM_TAIL_LENGTHS = (3, 5)
FASTWAM_EXECUTED_HORIZON = 24
POSITION_BOOTSTRAPS = 500
OUTCOME_BOOTSTRAPS = 2000
RANDOM_SEED = 20260822

SUCCESS_COLOR = "#3569b7"
FAIL_COLOR = "#c94f4f"
NEUTRAL_COLOR = "#6b7280"

PHASE_COLUMNS = [
    "query_state_phase",
    "phase_coarse",
    "phase_task",
    "phase_source",
    "phase_confidence",
]

ACTION_COLUMNS = [
    "source_kind",
    "run_id",
    "policy",
    "checkpoint",
    "task",
    "cohort_id",
    "group_id",
    "episode_key",
    "episode_id",
    "reset_id",
    "source_seed",
    "success",
    "first_success_action_slot",
    "first_success_query",
    "query_key",
    "query_idx",
    "success_before",
    "success_after",
    "h",
    "action_slot",
    "video_frame_index",
    "terminal_success_action",
    "terminal_success_video_frame",
    "video_path",
    "L",
    "V",
    "y_ln",
    "b_position",
    "mad_scale",
    "r_raw",
    "R_std",
    "mu",
    "P_h",
    "S_raw",
    "I_raw",
    "S_std",
    "I_std",
    *PHASE_COLUMNS,
    "alignment_contract",
]

L_SENSITIVITY_COLUMNS = [
    "group_id",
    "policy",
    "checkpoint",
    "task",
    "L_left",
    "L_right",
    "queries",
    "query_rank_correlation",
]

OUTCOME_COLUMNS = [
    "group_id",
    "policy",
    "checkpoint",
    "task",
    "metric",
    "success_episodes",
    "failure_episodes",
    "success_mean",
    "failure_mean",
    "success_minus_failure",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "inference_limit",
]

PHASE_EPISODE_COLUMNS = [
    "group_id",
    "policy",
    "checkpoint",
    "task",
    "episode_key",
    "episode_id",
    "success",
    "phase_axis",
    "phase_coarse",
    "phase_task",
    "query_state_phase",
    "queries",
    "cells",
    "mean_S_raw",
    "mean_S_std",
    "mean_abs_R_std",
    "mean_abs_I_raw",
    "mean_abs_I_std",
    "mean_y_ln",
]

PHASE_SUMMARY_COLUMNS = [
    "group_id",
    "policy",
    "checkpoint",
    "task",
    "phase_axis",
    "phase_coarse",
    "phase_task",
    "query_state_phase",
    "metric",
    "episodes",
    "mean",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "inference_limit",
]

STORYBOARD_INDEX_COLUMNS = [
    "episode_key",
    "selection_reason",
    "status",
    "video_path",
    "storyboard_path",
]


@dataclass
class QueryTrace:
    """One query's standardized metadata and model-native endpoint chain."""

    metadata: dict[str, Any]
    z_endpoint: np.ndarray


def _natural(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _first(mapping: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def _slug(value: Any, limit: int = 64) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    return (text or "unknown")[:limit]


def _short_checkpoint(value: Any) -> str:
    text = str(value or "unknown").replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] or "unknown"


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean field {field!r}: {value!r}")


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
        return None
    return int(float(value))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _source_metadata(root: Path) -> dict[str, Any]:
    """Merge stable run metadata without treating a rank manifest as authority."""

    metadata: dict[str, Any] = {}
    rank_manifests = sorted(root.glob("manifest_rollout_rank*.json"))
    if rank_manifests:
        metadata.update(_read_json(rank_manifests[0]))
    metadata.update(_read_json(root / "run_manifest.json"))
    return metadata


def _infer_task(root: Path, metadata: dict[str, Any], frame: pd.DataFrame | None = None) -> str:
    if frame is not None and "task" in frame.columns:
        tasks = sorted({str(value) for value in frame["task"].dropna() if str(value)})
        if len(tasks) == 1:
            return tasks[0]
    value = _first(metadata, ("task", "task_name", "env_name"))
    if value:
        return str(value)
    lowered = root.as_posix().lower()
    if "adjust_bottle" in lowered:
        return "adjust_bottle"
    return "unknown-task"


def _checkpoint(metadata: dict[str, Any]) -> str:
    return str(
        _first(
            metadata,
            (
                "checkpoint_revision",
                "checkpoint",
                "checkpoint_path",
                "model_path",
                "pretrained_model_name_or_path",
                "model",
            ),
            "unknown-checkpoint",
        )
    )


def _run_id(root: Path, metadata: dict[str, Any]) -> str:
    return str(_first(metadata, ("run_id", "name"), root.name))


def _cohort_id(run_id: str, metadata: dict[str, Any]) -> str:
    """Resolve an explicit analysis cohort, defaulting to the physical run.

    The physical run remains part of ``group_id`` even when an explicit cohort
    is supplied.  This makes independently collected P1/fixed-16 and fixed-64
    packets separate baselines by default instead of silently pooling them.
    """

    return str(
        _first(metadata, ("analysis_cohort", "cohort_id", "cohort"), run_id)
    )


def _group_id(
    policy: str,
    checkpoint: str,
    task: str,
    run_id: str,
    cohort_id: str,
) -> str:
    digest = hashlib.sha256(checkpoint.encode("utf-8")).hexdigest()[:8]
    return (
        f"{_slug(policy)}__{_slug(task)}__run-{_slug(run_id, 32)}"
        f"__cohort-{_slug(cohort_id, 24)}"
        f"__{_slug(_short_checkpoint(checkpoint), 32)}-{digest}"
    )


def _dataframe(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """Return a stable-schema frame even when ``rows`` is empty."""

    return pd.DataFrame(rows, columns=list(columns))


def _preflight_optional_dependencies(*, skip_plots: bool, skip_storyboards: bool) -> None:
    """Fail before output creation when default rendering dependencies are absent."""

    required: list[tuple[str, str]] = []
    if not skip_plots:
        required.append(("matplotlib", "--skip-plots"))
    if not skip_storyboards:
        required.append(("cv2", "--skip-storyboards"))
    missing: list[tuple[str, str, str]] = []
    for module, bypass in required:
        try:
            importlib.import_module(module)
        except Exception as exc:  # binary wheels can exist but fail during import
            missing.append((module, bypass, f"{type(exc).__name__}: {exc}"))
    if missing:
        detail = ", ".join(
            f"{module} (or use {bypass}; import failed: {reason})"
            for module, bypass, reason in missing
        )
        raise RuntimeError(f"Missing optional analysis dependencies: {detail}")


def _load_official_seed_maps(paths: Sequence[Path]) -> pd.DataFrame | None:
    """Read external official accepted-seed maps without mutating telemetry."""

    if not paths:
        return None
    parts: list[pd.DataFrame] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        frame = pd.read_csv(resolved)
        required = {"run_id", "episode_id", "source_seed"}
        if not required.issubset(frame.columns):
            raise ValueError(
                f"Official seed map {resolved} needs {sorted(required)}; "
                f"got {list(frame.columns)}"
            )
        frame = frame.copy()
        frame["_seed_map_path"] = str(resolved)
        parts.append(frame)
    combined = pd.concat(parts, ignore_index=True)
    keys = ["run_id", "episode_id"]
    if combined.duplicated(keys).any():
        duplicate = combined.loc[combined.duplicated(keys, keep=False), keys].head()
        raise ValueError(f"Official seed maps duplicate {keys}:\n{duplicate}")
    return combined


def _merge_official_seed_map(
    episodes: pd.DataFrame,
    seed_map: pd.DataFrame | None,
    *,
    default_run_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing episode seeds from a sidecar and reject contradictions."""

    result = episodes.copy()
    if "run_id" not in result:
        result["run_id"] = default_run_id
    if "source_seed" not in result:
        result["source_seed"] = pd.NA
    record: dict[str, Any] = {
        "official_seed_map_used": False,
        "official_seed_map_matched_episodes": 0,
        "official_seed_map_paths": [],
    }
    if seed_map is None:
        result["source_seed_provenance"] = np.where(
            result["source_seed"].notna(), "payload", "missing"
        )
        return result, record

    subset = seed_map[seed_map["run_id"].astype(str).isin(result["run_id"].astype(str))]
    merged = result.merge(
        subset[["run_id", "episode_id", "source_seed", "_seed_map_path"]].rename(
            columns={"source_seed": "source_seed_sidecar"}
        ),
        on=["run_id", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    payload_numeric = pd.to_numeric(merged["source_seed"], errors="coerce")
    sidecar_numeric = pd.to_numeric(merged["source_seed_sidecar"], errors="coerce")
    conflict = payload_numeric.notna() & sidecar_numeric.notna() & (
        payload_numeric != sidecar_numeric
    )
    if conflict.any():
        columns = ["run_id", "episode_id", "source_seed", "source_seed_sidecar"]
        raise ValueError(
            "Official seed map conflicts with payload:\n"
            + merged.loc[conflict, columns].head().to_string(index=False)
        )
    filled = merged["source_seed"].isna() & merged["source_seed_sidecar"].notna()
    merged.loc[filled, "source_seed"] = merged.loc[filled, "source_seed_sidecar"]
    merged["source_seed_provenance"] = np.where(
        payload_numeric.notna(),
        "payload",
        np.where(sidecar_numeric.notna(), "official_seed_map", "missing"),
    )
    paths_used = sorted(
        str(value)
        for value in merged.loc[sidecar_numeric.notna(), "_seed_map_path"].dropna().unique()
    )
    record.update(
        {
            "official_seed_map_used": bool(sidecar_numeric.notna().any()),
            "official_seed_map_matched_episodes": int(sidecar_numeric.notna().sum()),
            "official_seed_map_paths": paths_used,
        }
    )
    return merged.drop(columns=["source_seed_sidecar", "_seed_map_path"]), record


def _resolve_action_num_train_timesteps(metadata: dict[str, Any]) -> tuple[int, str, list[str]]:
    """Resolve Fast-WAM's action scheduler denominator with explicit provenance."""

    preferred = metadata.get("action_num_train_timesteps")
    legacy = metadata.get("num_train_timesteps")
    notes: list[str] = []
    if preferred not in (None, ""):
        value = int(preferred)
        source = "run_manifest.action_num_train_timesteps"
        if legacy not in (None, "") and int(legacy) != value:
            raise ValueError(
                "Fast-WAM manifest action_num_train_timesteps conflicts with "
                "legacy num_train_timesteps"
            )
        if legacy not in (None, ""):
            notes.append("legacy num_train_timesteps agrees with preferred action key")
    elif legacy not in (None, ""):
        value = int(legacy)
        source = "run_manifest.num_train_timesteps (legacy compatibility)"
        notes.append(
            "preferred action_num_train_timesteps absent; accepted explicit legacy key"
        )
    else:
        raise ValueError(
            "Fast-WAM run_manifest.json must record action_num_train_timesteps "
            "(preferred) or explicit legacy num_train_timesteps"
        )
    if value <= 0:
        raise ValueError("action_num_train_timesteps must be positive")
    return value, source, notes


def _resolve_telemetry_root(path: Path, marker: str) -> Path:
    path = path.expanduser().resolve()
    if (path / marker).exists() or list(path.glob(marker)):
        return path
    candidates = sorted({candidate.parent for candidate in path.rglob(marker)})
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No {marker!r} found under {path}")
    joined = "\n  ".join(str(value) for value in candidates)
    raise ValueError(f"Multiple telemetry roots found under {path}; pass one explicitly:\n  {joined}")


def _resolve_relative(root: Path, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    candidate = Path(text)
    if candidate.is_absolute():
        return str(candidate)
    return str((root / candidate).resolve())


def _rank_from_name(path: Path) -> int:
    match = re.search(r"rank(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Cannot parse rank from {path.name}")
    return int(match.group(1))


def _choose_episode_join_columns(query: pd.DataFrame, episode: pd.DataFrame) -> list[str]:
    mandatory = [name for name in ("episode_idx", "reset_id") if name in query and name in episode]
    if not mandatory:
        raise ValueError("pi0 query/episode tables have no shared episode_idx/reset_id key")
    columns = list(mandatory)
    for extra in ("eval_epoch", "source_env_rank", "stage_id", "local_env_slot"):
        if not episode.duplicated(columns).any():
            break
        if extra in query and extra in episode:
            columns.append(extra)
    if episode.duplicated(columns).any():
        duplicated = episode.loc[episode.duplicated(columns, keep=False), columns].head()
        raise ValueError(f"pi0 episode join is not many-to-one on {columns}:\n{duplicated}")
    return columns


def load_pi0_source(source: Path) -> tuple[list[QueryTrace], pd.DataFrame, dict[str, Any]]:
    """Load one consolidated or multi-rank pi0 telemetry directory."""

    root = _resolve_telemetry_root(source, "trace_rollout_rank*.npz")
    metadata = _source_metadata(root)
    run_id = _run_id(root, metadata)
    policy = str(_first(metadata, ("policy", "model_family"), "pi0"))
    checkpoint = _checkpoint(metadata)
    task = _infer_task(root, metadata)
    cohort_id = _cohort_id(run_id, metadata)
    group_id = _group_id(policy, checkpoint, task, run_id, cohort_id)

    episode_paths = sorted(root.glob("episode_index_env_rank*.csv"))
    if not episode_paths:
        raise FileNotFoundError(f"No pi0 episode index shards under {root}")
    episodes = pd.concat([pd.read_csv(path) for path in episode_paths], ignore_index=True)
    if "success" not in episodes:
        raise ValueError("pi0 episode index lacks success")
    episodes["success"] = [
        _parse_bool(value, field="success") for value in episodes["success"]
    ]

    traces: list[QueryTrace] = []
    query_parts: list[pd.DataFrame] = []
    validation = {
        "source_kind": "pi0",
        "root": str(root),
        "run_id": run_id,
        "trace_shards": 0,
        "queries": 0,
        "all_finite": True,
        "max_abs_final_chain_vs_model_action": 0.0,
        "video_alignment": "query_boundary_only; no future-h-to-physics-frame mapping",
    }

    for trace_path in sorted(root.glob("trace_rollout_rank*.npz")):
        rank = _rank_from_name(trace_path)
        query_path = root / f"query_index_rollout_rank{rank:02d}.csv"
        if not query_path.is_file():
            alternatives = sorted(root.glob(f"query_index_rollout_rank*{rank}.csv"))
            if len(alternatives) != 1:
                raise FileNotFoundError(f"Missing query index for {trace_path}")
            query_path = alternatives[0]
        query = pd.read_csv(query_path)
        if "trace_row" not in query:
            raise ValueError(f"{query_path} lacks trace_row")
        if "success_before" not in query:
            raise ValueError(f"{query_path} lacks required success_before")
        query = query.sort_values("trace_row").reset_index(drop=True)
        if query["trace_row"].astype(int).tolist() != list(range(len(query))):
            raise ValueError(f"{query_path} trace_row is not exactly 0..N-1")

        with np.load(trace_path, allow_pickle=False) as npz:
            required = {"z_endpoint", "timesteps"}
            missing = required.difference(npz.files)
            if missing:
                raise ValueError(f"{trace_path} missing {sorted(missing)}")
            z_all = np.asarray(npz["z_endpoint"], dtype=np.float32)
            if z_all.ndim != 4 or z_all.shape[0] != len(query):
                raise ValueError(
                    f"{trace_path}: z_endpoint {z_all.shape} does not match {len(query)} rows"
                )
            if not np.isfinite(z_all).all():
                raise ValueError(f"Non-finite z_endpoint in {trace_path}")
            timesteps = np.asarray(npz["timesteps"], dtype=np.float32).reshape(-1)
            if timesteps.shape != (z_all.shape[1],) or not np.isfinite(timesteps).all():
                raise ValueError(f"Invalid pi0 timesteps {timesteps.shape} for {z_all.shape}")

            env_action = np.asarray(npz["env_action"]) if "env_action" in npz else None
            executed_length = int(env_action.shape[1]) if env_action is not None and env_action.ndim >= 3 else None
            if "x_chain" in npz and "final_model_action" in npz:
                x_chain = np.asarray(npz["x_chain"], dtype=np.float32)
                final = np.asarray(npz["final_model_action"], dtype=np.float32)
                if x_chain.shape[0] != len(query) or x_chain.shape[1] != z_all.shape[1] + 1:
                    raise ValueError(f"Invalid pi0 x_chain shape {x_chain.shape}")
                delta = float(np.max(np.abs(x_chain[:, -1] - final)))
                validation["max_abs_final_chain_vs_model_action"] = max(
                    float(validation["max_abs_final_chain_vs_model_action"]), delta
                )

            for local_row, row in query.iterrows():
                row_dict = {key: _natural(value) for key, value in row.to_dict().items()}
                row_dict["_rank"] = rank
                row_dict["_trace_path"] = str(trace_path)
                row_dict["_executed_length"] = executed_length
                row_dict["_timesteps"] = timesteps.tolist()
                row_dict["_z"] = z_all[local_row].copy()
                query_parts.append(pd.DataFrame([row_dict]))
        validation["trace_shards"] = int(validation["trace_shards"]) + 1

    query_all = pd.concat(query_parts, ignore_index=True)
    join_columns = _choose_episode_join_columns(query_all, episodes)
    episode_columns = list(join_columns)
    for column in (
        "episode_uid",
        "success",
        "success_at_end",
        "return",
        "reward",
        "action_slots",
        "final_query_idx",
        "final_action_slot",
        "first_success_action_slot",
        "first_success_query",
        "terminal_success_video_frame",
        "termination_reason",
    ):
        if column in episodes and column not in episode_columns:
            episode_columns.append(column)
    merged = query_all.merge(
        episodes[episode_columns], on=join_columns, how="left", validate="many_to_one"
    )
    if merged["success"].isna().any():
        raise ValueError("Some pi0 queries do not join to an episode outcome")

    seen_query_keys: set[str] = set()
    for _, row in merged.iterrows():
        row_dict = row.to_dict()
        query_uid = str(row_dict.get("query_uid") or f"row{len(traces):06d}")
        query_key = f"{run_id}::{query_uid}"
        if query_key in seen_query_keys:
            raise ValueError(f"Duplicate pi0 query key: {query_key}")
        seen_query_keys.add(query_key)
        episode_uid = str(
            row_dict.get("episode_uid")
            or f"ep{int(row_dict['episode_idx']):06d}_reset{row_dict.get('reset_id')}"
        )
        z = np.asarray(row_dict.pop("_z"), dtype=np.float32)
        executed_length = row_dict.pop("_executed_length")
        standard = {
            "source_kind": "pi0",
            "source_root": str(root),
            "run_id": run_id,
            "policy": policy,
            "checkpoint": checkpoint,
            "task": task,
            "cohort_id": cohort_id,
            "group_id": group_id,
            "query_key": query_key,
            "query_uid": query_uid,
            "episode_key": f"{run_id}::{episode_uid}",
            "episode_uid": episode_uid,
            "episode_id": _natural(row_dict.get("episode_idx")),
            "reset_id": _natural(row_dict.get("reset_id")),
            "source_seed": _natural(row_dict.get("reset_id")),
            "source_seed_provenance": "payload_reset_id",
            "query_idx": int(row_dict.get("query_idx", 0)),
            "action_slot_start": int(row_dict.get("action_slot_start", 0)),
            "planned_exec_length": executed_length,
            "executed_length": executed_length,
            "success_before": _parse_bool(
                row_dict["success_before"], field="success_before"
            ),
            "success_after": (
                _parse_bool(row_dict["success_after"], field="success_after")
                if row_dict.get("success_after") not in (None, "")
                else None
            ),
            "success": _parse_bool(row_dict["success"], field="success"),
            "first_success_action_slot": _optional_int(
                row_dict.get("first_success_action_slot")
            ),
            "first_success_query": _optional_int(row_dict.get("first_success_query")),
            "episode_total_action_slots": _optional_int(row_dict.get("action_slots")),
            "termination_reason": str(row_dict.get("termination_reason", "")),
            "trace_path": str(row_dict.pop("_trace_path")),
            "trace_rank": int(row_dict.pop("_rank")),
            "trace_row": int(row_dict.get("trace_row", 0)),
            "head_image_path": _resolve_relative(root, row_dict.get("head_image_relpath")),
            "left_image_path": _resolve_relative(root, row_dict.get("left_wrist_image_relpath")),
            "right_image_path": _resolve_relative(root, row_dict.get("right_wrist_image_relpath")),
            "video_path": _resolve_relative(root, row_dict.get("video_relpath")),
            "video_alignment": "query_boundary_only",
            "video_frame_start": None,
            "video_frame_end_exclusive": None,
            "terminal_success_video_frame": _optional_int(
                row_dict.get("terminal_success_video_frame")
            ),
            "timesteps": row_dict.pop("_timesteps"),
        }
        traces.append(QueryTrace(standard, z))

    validation["queries"] = len(traces)
    validation["episode_join_columns"] = join_columns
    validation["episodes"] = int(len(episodes))
    validation["success_episodes"] = int(episodes["success"].sum())
    return traces, episodes, validation


def _fastwam_episode_join_columns(query: pd.DataFrame, episode: pd.DataFrame) -> list[str]:
    columns = [name for name in ("run_id", "episode_id", "reset_id") if name in query and name in episode]
    if "episode_id" not in columns:
        raise ValueError("Fast-WAM tables lack episode_id")
    while episode.duplicated(columns).any():
        for extra in ("task", "policy"):
            if extra in query and extra in episode and extra not in columns:
                columns.append(extra)
                break
        else:
            raise ValueError(f"Fast-WAM episode rows duplicate join columns {columns}")
    return columns


def reconstruct_fastwam_endpoints(
    x_chain: np.ndarray,
    v_chain: np.ndarray,
    timesteps: np.ndarray,
    *,
    num_train_timesteps: int = 1000,
) -> np.ndarray:
    """Return z_i = x_i - (timestep_i / num_train_timesteps) * v_i."""

    x = np.asarray(x_chain, dtype=np.float32)
    v = np.asarray(v_chain, dtype=np.float32)
    t = np.asarray(timesteps, dtype=np.float32).reshape(-1)
    if x.ndim != 3 or v.ndim != 3:
        raise ValueError(f"Expected Fast-WAM x/v rank 3, got {x.shape} and {v.shape}")
    if x.shape[0] != v.shape[0] + 1 or x.shape[1:] != v.shape[1:]:
        raise ValueError(f"Expected x=[M+1,H,D], v=[M,H,D], got {x.shape}, {v.shape}")
    if t.shape != (v.shape[0],):
        raise ValueError(f"Expected {v.shape[0]} timesteps, got {t.shape}")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    z = x[:-1] - (t.reshape(-1, 1, 1) / float(num_train_timesteps)) * v
    if not np.isfinite(z).all():
        raise ValueError("Non-finite reconstructed Fast-WAM endpoints")
    return z


def load_fastwam_source(
    source: Path,
    *,
    official_seed_map: pd.DataFrame | None = None,
) -> tuple[list[QueryTrace], pd.DataFrame, dict[str, Any]]:
    root = _resolve_telemetry_root(source, "queries.csv")
    queries = pd.read_csv(root / "queries.csv")
    episodes = pd.read_csv(root / "episodes.csv")
    metadata = _source_metadata(root)
    default_run_id = _run_id(root, metadata)
    if "success_before" not in queries:
        raise ValueError(f"{root / 'queries.csv'} lacks required success_before")
    episodes, seed_map_record = _merge_official_seed_map(
        episodes, official_seed_map, default_run_id=default_run_id
    )
    checkpoint = _checkpoint(metadata)
    default_policy = str(_first(metadata, ("policy", "model_family"), "fastwam"))
    join_columns = _fastwam_episode_join_columns(queries, episodes)
    if "success" not in episodes:
        raise ValueError("Fast-WAM episodes.csv lacks success")
    episodes["success"] = [
        _parse_bool(value, field="success") for value in episodes["success"]
    ]
    episode_columns = list(join_columns)
    for column in (
        "source_seed",
        "success",
        "total_action_slots",
        "queries",
        "first_success_action_slot",
        "first_success_query",
        "source_seed_provenance",
        "video_path",
    ):
        if column in episodes and column not in episode_columns:
            episode_columns.append(column)
    merged = queries.merge(
        episodes[episode_columns], on=join_columns, how="left", validate="many_to_one"
    )
    if merged["success"].isna().any():
        raise ValueError("Some Fast-WAM queries do not join to an episode outcome")

    action_num_train_timesteps, action_timestep_source, compatibility_notes = (
        _resolve_action_num_train_timesteps(metadata)
    )
    traces: list[QueryTrace] = []
    seen_query_keys: set[str] = set()
    validation = {
        "source_kind": "fastwam",
        "root": str(root),
        "run_id": default_run_id,
        "trace_files": 0,
        "queries": 0,
        "episodes": int(len(episodes)),
        "success_episodes": int(episodes["success"].sum()),
        "all_finite": True,
        "x_next_matches_x_chain": True,
        "max_abs_final_chain_vs_model_action": 0.0,
        "action_num_train_timesteps": action_num_train_timesteps,
        "action_num_train_timesteps_source": action_timestep_source,
        "compatibility_notes": compatibility_notes,
        **seed_map_record,
        "video_alignment": "fresh pre-action frame; exact only when frame ranges are populated",
    }

    for _, row in merged.sort_values(["episode_id", "query_idx"]).iterrows():
        row_dict = {key: _natural(value) for key, value in row.to_dict().items()}
        run_id = str(row_dict.get("run_id") or default_run_id)
        policy = str(row_dict.get("policy") or default_policy)
        task = str(row_dict.get("task") or _infer_task(root, metadata, queries))
        cohort_id = _cohort_id(run_id, metadata)
        group_id = _group_id(policy, checkpoint, task, run_id, cohort_id)
        episode_id = int(row_dict["episode_id"])
        query_idx = int(row_dict["query_idx"])
        query_uid = str(row_dict.get("query_uid") or f"episode{episode_id:04d}_query{query_idx:04d}")
        query_key = f"{run_id}::{query_uid}"
        if query_key in seen_query_keys:
            raise ValueError(f"Duplicate Fast-WAM query key: {query_key}")
        seen_query_keys.add(query_key)

        trace_path = Path(str(row_dict["trace_path"]))
        if not trace_path.is_absolute():
            trace_path = root / trace_path
        if not trace_path.is_file():
            raise FileNotFoundError(f"Fast-WAM trace missing: {trace_path}")
        with np.load(trace_path, allow_pickle=False) as npz:
            missing = {"x_chain", "v_chain", "timesteps"}.difference(npz.files)
            if missing:
                raise ValueError(f"{trace_path} missing {sorted(missing)}")
            x_chain = np.asarray(npz["x_chain"], dtype=np.float32)
            v_chain = np.asarray(npz["v_chain"], dtype=np.float32)
            timesteps = np.asarray(npz["timesteps"], dtype=np.float32).reshape(-1)
            z = reconstruct_fastwam_endpoints(
                x_chain,
                v_chain,
                timesteps,
                num_train_timesteps=action_num_train_timesteps,
            )
            if "x_next" in npz and not np.array_equal(x_chain[1:], npz["x_next"]):
                raise ValueError(f"{trace_path}: x_next != x_chain[1:]")
            if "final_model_action" in npz:
                delta = float(np.max(np.abs(x_chain[-1] - npz["final_model_action"])))
                validation["max_abs_final_chain_vs_model_action"] = max(
                    float(validation["max_abs_final_chain_vs_model_action"]), delta
                )

        episode_uid = f"episode{episode_id:04d}_reset{row_dict.get('reset_id')}"
        frame_start = _optional_int(row_dict.get("video_frame_start"))
        frame_end = _optional_int(row_dict.get("video_frame_end_exclusive"))
        executed_length = int(row_dict.get("executed_length") or 0)
        if frame_start is not None or frame_end is not None:
            if frame_start is None or frame_end is None:
                raise ValueError(f"Partial Fast-WAM frame range in {query_key}")
            if frame_end - frame_start != executed_length:
                raise ValueError(
                    f"Fast-WAM frame range {frame_start}:{frame_end} != executed_length "
                    f"{executed_length} for {query_key}"
                )
        standard = {
            "source_kind": "fastwam",
            "source_root": str(root),
            "run_id": run_id,
            "policy": policy,
            "checkpoint": checkpoint,
            "task": task,
            "cohort_id": cohort_id,
            "group_id": group_id,
            "query_key": query_key,
            "query_uid": query_uid,
            "episode_key": f"{run_id}::{episode_uid}",
            "episode_uid": episode_uid,
            "episode_id": episode_id,
            "reset_id": row_dict.get("reset_id"),
            "source_seed": row_dict.get("source_seed"),
            "source_seed_provenance": row_dict.get(
                "source_seed_provenance", "missing"
            ),
            "query_idx": query_idx,
            "action_slot_start": int(row_dict.get("query_start_action_slot") or 0),
            "planned_exec_length": int(row_dict.get("planned_exec_length") or 0),
            "executed_length": executed_length,
            "success_before": _parse_bool(
                row_dict["success_before"], field="success_before"
            ),
            "success_after": _parse_bool(
                row_dict.get("success_after", False), field="success_after"
            ),
            "success": _parse_bool(row_dict["success"], field="success"),
            "first_success_action_slot": _optional_int(
                row_dict.get("first_success_action_slot")
            ),
            "first_success_query": _optional_int(row_dict.get("first_success_query")),
            "episode_total_action_slots": _optional_int(
                row_dict.get("total_action_slots")
            ),
            "termination_reason": "success" if _parse_bool(row_dict["success"], field="success") else "timeout",
            "trace_path": str(trace_path.resolve()),
            "trace_rank": None,
            "trace_row": None,
            "head_image_path": _resolve_relative(root, row_dict.get("head_image_path")),
            "left_image_path": _resolve_relative(root, row_dict.get("left_image_path")),
            "right_image_path": _resolve_relative(root, row_dict.get("right_image_path")),
            "video_path": str(row_dict.get("video_path") or ""),
            "video_alignment": (
                "fresh_pre_action_frame_per_executed_action"
                if frame_start is not None
                else "query_only; no action-frame mapping"
            ),
            "video_frame_start": frame_start,
            "video_frame_end_exclusive": frame_end,
            "terminal_success_video_frame": _optional_int(
                row_dict.get("terminal_success_video_frame")
            ),
            "timesteps": timesteps.tolist(),
            "instruction": str(row_dict.get("instruction") or ""),
        }
        traces.append(QueryTrace(standard, z))
        validation["trace_files"] = int(validation["trace_files"]) + 1

    validation["queries"] = len(traces)
    validation["episode_join_columns"] = join_columns
    return traces, episodes, validation


def _supported_tail_lengths(source_kind: str, denoising_steps: int) -> tuple[int, ...]:
    requested = PI0_TAIL_LENGTHS if source_kind == "pi0" else FASTWAM_TAIL_LENGTHS
    values = tuple(length for length in requested if length <= denoising_steps)
    if COMMON_L not in values:
        raise ValueError(
            f"{source_kind} trace has {denoising_steps} endpoints, fewer than common L={COMMON_L}"
        )
    return values


def _compute_v(z: np.ndarray, length: int) -> np.ndarray:
    if z.ndim != 3:
        raise ValueError(f"Expected z=[M,H,D], got {z.shape}")
    if not 2 <= length <= z.shape[0]:
        raise ValueError(f"Invalid tail L={length} for z shape {z.shape}")
    return np.var(z[-length:], axis=0, ddof=0).sum(axis=-1)


def derive_metrics(
    traces: Sequence[QueryTrace], *, scale_floor: float = DEFAULT_SCALE_FLOOR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not traces:
        raise ValueError("No query traces were loaded")
    if not math.isfinite(scale_floor) or scale_floor <= 0:
        raise ValueError("scale_floor must be a fixed positive finite number")

    query_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for trace in traces:
        z = np.asarray(trace.z_endpoint, dtype=np.float32)
        if z.ndim != 3 or not np.isfinite(z).all():
            raise ValueError(f"Invalid z_endpoint for {trace.metadata.get('query_key')}: {z.shape}")
        m, horizon_size, action_dim = z.shape
        lengths = _supported_tail_lengths(str(trace.metadata["source_kind"]), m)
        by_l = {length: _compute_v(z, length) for length in lengths}
        qrow = dict(trace.metadata)
        qrow.update(
            {
                "denoising_steps": m,
                "horizon": horizon_size,
                "active_action_dim": action_dim,
                "tail_lengths": ",".join(str(value) for value in lengths),
                "baseline_eligible": not bool(trace.metadata["success_before"]),
            }
        )
        for length, values in by_l.items():
            total = float(values.sum())
            qrow[f"V_total_L{length}"] = total
            qrow[f"log_V_total_L{length}"] = float(np.log(total + EPS))
        query_rows.append(qrow)

        common_metadata = {
            "source_kind": qrow["source_kind"],
            "source_root": qrow["source_root"],
            "run_id": qrow["run_id"],
            "policy": qrow["policy"],
            "checkpoint": qrow["checkpoint"],
            "task": qrow["task"],
            "cohort_id": qrow["cohort_id"],
            "group_id": qrow["group_id"],
            "query_key": qrow["query_key"],
            "query_uid": qrow["query_uid"],
            "episode_key": qrow["episode_key"],
            "episode_uid": qrow["episode_uid"],
            "episode_id": qrow["episode_id"],
            "reset_id": qrow["reset_id"],
            "source_seed": qrow["source_seed"],
            "source_seed_provenance": qrow["source_seed_provenance"],
            "query_idx": qrow["query_idx"],
            "action_slot_start": qrow["action_slot_start"],
            "executed_length": qrow["executed_length"],
            "success": qrow["success"],
            "outcome": "success" if bool(qrow["success"]) else "failure",
            "success_before": qrow["success_before"],
            "success_after": qrow["success_after"],
            "first_success_action_slot": qrow["first_success_action_slot"],
            "first_success_query": qrow["first_success_query"],
            "baseline_eligible": qrow["baseline_eligible"],
            "query_state_phase": "UNLABELED",
            "phase_coarse": "UNLABELED",
            "phase_task": "UNLABELED",
            "phase_source": "",
            "phase_confidence": "",
        }
        for length, values in by_l.items():
            for h in range(horizon_size):
                executed_limit = int(qrow["executed_length"] or 0)
                if qrow["source_kind"] == "fastwam":
                    executed_limit = min(executed_limit, FASTWAM_EXECUTED_HORIZON)
                executed_flag = h < executed_limit
                horizon_rows.append(
                    {
                        **common_metadata,
                        "L": int(length),
                        "h": h,
                        "V": float(values[h]),
                        "y_ln": float(np.log(values[h] + EPS)),
                        "executed_flag": bool(executed_flag),
                        "action_slot": (
                            int(qrow["action_slot_start"]) + h if executed_flag else None
                        ),
                    }
                )

    query = pd.DataFrame(query_rows)
    horizon = pd.DataFrame(horizon_rows)
    if query["query_key"].duplicated().any():
        raise ValueError("Global query_key is not unique")

    baseline = horizon[horizon["baseline_eligible"]].copy()
    expected_groups = horizon[["group_id", "L", "h"]].drop_duplicates()
    observed_groups = baseline[["group_id", "L", "h"]].drop_duplicates()
    missing = expected_groups.merge(
        observed_groups, on=["group_id", "L", "h"], how="left", indicator=True
    )
    missing = missing[missing["_merge"] == "left_only"]
    if not missing.empty:
        raise ValueError(
            "No baseline-eligible query for some policy/checkpoint/task/L/h cells:\n"
            + missing.head(20).to_string(index=False)
        )

    center = (
        baseline.groupby(["group_id", "L", "h"], sort=False)["y_ln"]
        .median()
        .rename("b_position")
    )
    baseline = baseline.join(center, on=["group_id", "L", "h"])
    baseline["abs_centered"] = (baseline["y_ln"] - baseline["b_position"]).abs()
    mad = (
        baseline.groupby(["group_id", "L", "h"], sort=False)["abs_centered"]
        .median()
        .rename("mad_h")
    )
    horizon = horizon.join(center, on=["group_id", "L", "h"]).join(
        mad, on=["group_id", "L", "h"]
    )
    horizon["s_h_unfloored"] = 1.4826 * horizon["mad_h"]
    horizon["scale_floor"] = float(scale_floor)
    horizon["scale_floored"] = horizon["s_h_unfloored"] < scale_floor
    horizon["mad_scale"] = np.maximum(horizon["s_h_unfloored"], scale_floor)
    horizon["r_raw"] = horizon["y_ln"] - horizon["b_position"]
    horizon["R_std"] = horizon["r_raw"] / horizon["mad_scale"]

    mu = (
        horizon[["group_id", "L", "h", "b_position"]]
        .drop_duplicates()
        .groupby(["group_id", "L"], sort=False)["b_position"]
        .mean()
        .rename("mu")
    )
    horizon = horizon.join(mu, on=["group_id", "L"])
    horizon["P_h"] = horizon["b_position"] - horizon["mu"]
    horizon["S_raw"] = horizon.groupby(["query_key", "L"], sort=False)[
        "r_raw"
    ].transform("mean")
    horizon["I_raw"] = horizon["r_raw"] - horizon["S_raw"]
    horizon["S_std"] = horizon.groupby(["query_key", "L"], sort=False)[
        "R_std"
    ].transform("mean")
    horizon["I_std"] = horizon["R_std"] - horizon["S_std"]

    identities = {
        "max_abs_y_minus_b_plus_r": float(
            np.max(np.abs(horizon["y_ln"] - (horizon["b_position"] + horizon["r_raw"])))
        ),
        "max_abs_y_minus_raw_four_way": float(
            np.max(
                np.abs(
                    horizon["y_ln"]
                    - (
                        horizon["mu"]
                        + horizon["P_h"]
                        + horizon["S_raw"]
                        + horizon["I_raw"]
                    )
                )
            )
        ),
        "max_abs_R_minus_Sstd_plus_Istd": float(
            np.max(
                np.abs(horizon["R_std"] - (horizon["S_std"] + horizon["I_std"]))
            )
        ),
    }
    if max(identities.values()) > 1e-10:
        raise AssertionError(f"DVAC decomposition identity failed: {identities}")

    common = horizon[horizon["L"] == COMMON_L]
    qderived = (
        common.groupby("query_key", sort=False)
        .agg(
            S_raw_L3=("S_raw", "first"),
            S_std_L3=("S_std", "first"),
            mean_y_L3=("y_ln", "mean"),
            median_y_L3=("y_ln", "median"),
            mean_abs_R_std_L3=("R_std", lambda value: float(np.mean(np.abs(value)))),
            mean_abs_I_raw_L3=("I_raw", lambda value: float(np.mean(np.abs(value)))),
            mean_abs_I_std_L3=("I_std", lambda value: float(np.mean(np.abs(value)))),
            max_abs_I_std_L3=("I_std", lambda value: float(np.max(np.abs(value)))),
            floored_h_L3=("scale_floored", "sum"),
        )
        .reset_index()
    )
    query = query.merge(qderived, on="query_key", how="left", validate="one_to_one")
    query.attrs["identities"] = identities
    return query, horizon


def apply_phase_annotations(horizon: pd.DataFrame, annotation_path: Path | None) -> pd.DataFrame:
    if annotation_path is None:
        return horizon
    annotations = pd.read_csv(annotation_path)
    required_value = {"phase_coarse", "phase_task", "phase_source"}
    if not required_value.issubset(annotations.columns):
        raise ValueError(
            f"Phase annotations need {sorted(required_value)}; got {list(annotations.columns)}"
        )
    if {"query_key", "h"}.issubset(annotations.columns):
        keys = ["query_key", "h"]
    elif {"run_id", "episode_id", "query_idx", "h"}.issubset(annotations.columns):
        keys = ["run_id", "episode_id", "query_idx", "h"]
    elif "query_key" in annotations.columns:
        keys = ["query_key"]
    elif {"run_id", "episode_id", "query_idx"}.issubset(annotations.columns):
        keys = ["run_id", "episode_id", "query_idx"]
    else:
        raise ValueError(
            "Phase annotations need query_key (optionally +h) or "
            "run_id+episode_id+query_idx (optionally +h)"
        )
    if annotations.duplicated(keys).any():
        raise ValueError(f"Duplicate phase annotation keys: {keys}")
    values = keys + ["phase_coarse", "phase_task", "phase_source"]
    if "query_state_phase" in annotations:
        values.append("query_state_phase")
    if "phase_confidence" in annotations:
        values.append("phase_confidence")
    base = horizon.drop(
        columns=PHASE_COLUMNS
    )
    merged = base.merge(annotations[values], on=keys, how="left", validate="many_to_one")
    merged["phase_coarse"] = merged["phase_coarse"].fillna("UNLABELED")
    merged["phase_task"] = merged["phase_task"].fillna("UNLABELED")
    merged["phase_source"] = merged["phase_source"].fillna("")
    if "query_state_phase" not in merged:
        merged["query_state_phase"] = "UNLABELED"
    else:
        merged["query_state_phase"] = merged["query_state_phase"].fillna("UNLABELED")
    if "phase_confidence" not in merged:
        merged["phase_confidence"] = ""
    else:
        merged["phase_confidence"] = merged["phase_confidence"].fillna("")
    unexecuted_fastwam = (merged["source_kind"] == "fastwam") & ~merged[
        "executed_flag"
    ].astype(bool)
    merged.loc[unexecuted_fastwam, ["phase_coarse", "phase_task"]] = "UNLABELED"
    merged.loc[unexecuted_fastwam, ["phase_source", "phase_confidence"]] = ""
    return merged


def _unique_phase(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values if str(value) not in {"", "UNLABELED"}})
    if not unique:
        return "UNLABELED"
    return unique[0] if len(unique) == 1 else "MIXED"


def attach_query_phases(query: pd.DataFrame, horizon: pd.DataFrame) -> pd.DataFrame:
    """Attach query-state labels and descriptive action-phase summaries to queries."""

    common = horizon[horizon["L"] == COMMON_L].copy()
    fast_unexecuted = (common["source_kind"] == "fastwam") & ~common[
        "executed_flag"
    ].astype(bool)
    action_common = common[~fast_unexecuted]
    query_phase = (
        common.groupby("query_key", sort=False)
        .agg(query_state_phase=("query_state_phase", _unique_phase))
        .reset_index()
    )
    action_phase = (
        action_common.groupby("query_key", sort=False)
        .agg(
            phase_coarse=("phase_coarse", _unique_phase),
            phase_task=("phase_task", _unique_phase),
            phase_source=("phase_source", _unique_phase),
            phase_confidence=("phase_confidence", _unique_phase),
        )
        .reset_index()
    )
    base = query.drop(columns=PHASE_COLUMNS, errors="ignore")
    merged = base.merge(query_phase, on="query_key", how="left", validate="one_to_one")
    merged = merged.merge(action_phase, on="query_key", how="left", validate="one_to_one")
    for column in ("query_state_phase", "phase_coarse", "phase_task"):
        merged[column] = merged[column].fillna("UNLABELED")
    for column in ("phase_source", "phase_confidence"):
        merged[column] = merged[column].replace("UNLABELED", "").fillna("")
    return merged


def build_episode_metrics(
    query: pd.DataFrame,
    horizon: pd.DataFrame,
    *,
    pre_success_only: bool = True,
) -> pd.DataFrame:
    selected_query = query[~query["success_before"].astype(bool)].copy() if pre_success_only else query.copy()
    selected_keys = set(selected_query["query_key"])
    common = horizon[
        (horizon["L"] == COMMON_L) & horizon["query_key"].isin(selected_keys)
    ]
    horizon_query = (
        common.groupby("query_key", sort=False)
        .agg(
            query_mean_abs_I_raw=("I_raw", lambda value: float(np.mean(np.abs(value)))),
            query_mean_abs_I_std=("I_std", lambda value: float(np.mean(np.abs(value)))),
            query_mean_abs_R_std=("R_std", lambda value: float(np.mean(np.abs(value)))),
        )
        .reset_index()
    )
    working = selected_query.merge(
        horizon_query, on="query_key", how="left", validate="one_to_one"
    )
    computed_end = working["action_slot_start"] + working["executed_length"].fillna(0)
    working["episode_total_action_slots_resolved"] = working[
        "episode_total_action_slots"
    ].fillna(computed_end)
    episode = (
        working.groupby("episode_key", sort=False)
        .agg(
            source_kind=("source_kind", "first"),
            source_root=("source_root", "first"),
            run_id=("run_id", "first"),
            policy=("policy", "first"),
            checkpoint=("checkpoint", "first"),
            task=("task", "first"),
            cohort_id=("cohort_id", "first"),
            group_id=("group_id", "first"),
            episode_uid=("episode_uid", "first"),
            episode_id=("episode_id", "first"),
            reset_id=("reset_id", "first"),
            source_seed=("source_seed", "first"),
            source_seed_provenance=("source_seed_provenance", "first"),
            success=("success", "first"),
            first_success_action_slot=("first_success_action_slot", "first"),
            first_success_query=("first_success_query", "first"),
            terminal_success_video_frame=("terminal_success_video_frame", "max"),
            any_success_after=(
                "success_after",
                lambda value: any(bool(v) for v in value if not pd.isna(v)),
            ),
            queries=("query_key", "size"),
            total_action_slots=("episode_total_action_slots_resolved", "max"),
            episode_mean_S_raw=("S_raw_L3", "mean"),
            episode_median_S_raw=("S_raw_L3", "median"),
            episode_mean_S_std=("S_std_L3", "mean"),
            episode_median_S_std=("S_std_L3", "median"),
            episode_min_S_std=("S_std_L3", "min"),
            episode_max_S_std=("S_std_L3", "max"),
            episode_mean_abs_I_raw=("query_mean_abs_I_raw", "mean"),
            episode_mean_abs_I_std=("query_mean_abs_I_std", "mean"),
            episode_mean_abs_R_std=("query_mean_abs_R_std", "mean"),
            video_path=("video_path", "first"),
        )
        .reset_index()
    )
    episode["query_scope"] = "pre_success" if pre_success_only else "all_queries_descriptive"
    episode["representative"] = False
    episode["selection_reason"] = ""
    for group_id, group in episode.groupby("group_id", sort=False):
        selected = select_representative_episodes(group)
        for episode_key, reason in selected.items():
            mask = (episode["group_id"] == group_id) & (episode["episode_key"] == episode_key)
            episode.loc[mask, "representative"] = True
            episode.loc[mask, "selection_reason"] = reason
    return episode


def _closest_to_median(group: pd.DataFrame) -> str:
    target = float(group["total_action_slots"].median())
    seed_numeric = pd.to_numeric(group["source_seed"], errors="coerce")
    ordered = group.assign(
        distance=(group["total_action_slots"] - target).abs(),
        seed_numeric_tie=seed_numeric.fillna(np.inf),
        seed_text_tie=group["source_seed"].fillna("").astype(str),
        episode_tie=group["episode_id"].fillna(0),
    ).sort_values(
        ["distance", "seed_numeric_tie", "seed_text_tie", "episode_tie", "episode_key"]
    )
    return str(ordered.iloc[0]["episode_key"])


def select_representative_episodes(group: pd.DataFrame) -> dict[str, str]:
    """Pre-registered representative: median action length within outcome."""

    group = group.sort_values("episode_key").copy()
    selected: dict[str, str] = {}
    for success, label in ((True, "success"), (False, "failure")):
        outcome = group[group["success"].astype(bool) == success]
        if not outcome.empty:
            selected[_closest_to_median(outcome)] = (
                f"typical_{label}_nearest_outcome_median_action_slots"
            )
    return selected


def build_action_metrics(query: pd.DataFrame, horizon: pd.DataFrame) -> pd.DataFrame:
    """Map Fast-WAM executed h to exact fresh-pre-action video frame indices."""

    fast = query[query["source_kind"] == "fastwam"].copy()
    rows: list[dict[str, Any]] = []
    common = horizon[horizon["L"] == COMMON_L]
    by_query = {key: frame for key, frame in common.groupby("query_key", sort=False)}
    for _, qrow in fast.iterrows():
        start = _optional_int(qrow.get("video_frame_start"))
        end = _optional_int(qrow.get("video_frame_end_exclusive"))
        executed = min(int(qrow.get("executed_length") or 0), FASTWAM_EXECUTED_HORIZON)
        if start is None or end is None:
            continue
        if end - start != int(qrow.get("executed_length") or 0):
            raise ValueError(f"Frame range mismatch for {qrow['query_key']}")
        hframe = by_query[str(qrow["query_key"])].set_index("h")
        first_success_slot = _optional_int(qrow.get("first_success_action_slot"))
        terminal_frame = _optional_int(qrow.get("terminal_success_video_frame"))
        for h in range(executed):
            metric = hframe.loc[h]
            action_slot = int(qrow["action_slot_start"]) + h
            terminal_success_action = (
                action_slot == first_success_slot
                if first_success_slot is not None
                else bool(qrow.get("success_after"))
                and terminal_frame is not None
                and h == executed - 1
            )
            rows.append(
                {
                    "source_kind": "fastwam",
                    "run_id": qrow["run_id"],
                    "policy": qrow["policy"],
                    "checkpoint": qrow["checkpoint"],
                    "task": qrow["task"],
                    "cohort_id": qrow["cohort_id"],
                    "group_id": qrow["group_id"],
                    "episode_key": qrow["episode_key"],
                    "episode_id": qrow["episode_id"],
                    "reset_id": qrow["reset_id"],
                    "source_seed": qrow["source_seed"],
                    "success": qrow["success"],
                    "first_success_action_slot": first_success_slot,
                    "first_success_query": qrow["first_success_query"],
                    "query_key": qrow["query_key"],
                    "query_idx": qrow["query_idx"],
                    "success_before": qrow["success_before"],
                    "success_after": qrow["success_after"],
                    "h": h,
                    "action_slot": action_slot,
                    "video_frame_index": start + h,
                    "terminal_success_action": bool(terminal_success_action),
                    "terminal_success_video_frame": terminal_frame,
                    "video_path": qrow["video_path"],
                    "L": COMMON_L,
                    "V": metric["V"],
                    "y_ln": metric["y_ln"],
                    "b_position": metric["b_position"],
                    "mad_scale": metric["mad_scale"],
                    "r_raw": metric["r_raw"],
                    "R_std": metric["R_std"],
                    "mu": metric["mu"],
                    "P_h": metric["P_h"],
                    "S_raw": metric["S_raw"],
                    "I_raw": metric["I_raw"],
                    "S_std": metric["S_std"],
                    "I_std": metric["I_std"],
                    "query_state_phase": metric["query_state_phase"],
                    "phase_coarse": metric["phase_coarse"],
                    "phase_task": metric["phase_task"],
                    "phase_source": metric["phase_source"],
                    "phase_confidence": metric["phase_confidence"],
                    "alignment_contract": "fresh pre-action frame; h<executed_length and h<24",
                }
            )
    return _dataframe(rows, ACTION_COLUMNS)


def _rank_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.rank(method="average").corr(right.rank(method="average")))


def build_l_sensitivity(query: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in query.groupby("group_id", sort=False):
        lengths = sorted(
            int(column.rsplit("L", 1)[1])
            for column in group.columns
            if column.startswith("V_total_L") and group[column].notna().any()
        )
        for left_index, left in enumerate(lengths):
            for right in lengths[left_index + 1 :]:
                mask = group[f"V_total_L{left}"].notna() & group[f"V_total_L{right}"].notna()
                rows.append(
                    {
                        "group_id": group_id,
                        "policy": group["policy"].iloc[0],
                        "checkpoint": group["checkpoint"].iloc[0],
                        "task": group["task"].iloc[0],
                        "L_left": left,
                        "L_right": right,
                        "queries": int(mask.sum()),
                        "query_rank_correlation": _rank_correlation(
                            group.loc[mask, f"V_total_L{left}"],
                            group.loc[mask, f"V_total_L{right}"],
                        ),
                    }
                )
    return _dataframe(rows, L_SENSITIVITY_COLUMNS)


def build_outcome_summary(episode: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_id, group in episode.groupby("group_id", sort=False):
        for metric in (
            "episode_mean_S_raw",
            "episode_mean_S_std",
            "episode_mean_abs_I_raw",
            "episode_mean_abs_I_std",
            "episode_mean_abs_R_std",
        ):
            success = group.loc[group["success"].astype(bool), metric].to_numpy(float)
            failure = group.loc[~group["success"].astype(bool), metric].to_numpy(float)
            row = {
                "group_id": group_id,
                "policy": group["policy"].iloc[0],
                "checkpoint": group["checkpoint"].iloc[0],
                "task": group["task"].iloc[0],
                "metric": metric,
                "success_episodes": len(success),
                "failure_episodes": len(failure),
                "success_mean": float(np.mean(success)) if len(success) else np.nan,
                "failure_mean": float(np.mean(failure)) if len(failure) else np.nan,
                "success_minus_failure": (
                    float(np.mean(success) - np.mean(failure))
                    if len(success) and len(failure)
                    else np.nan
                ),
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "inference_limit": "episode-level descriptive bootstrap; observational, not causal",
            }
            if len(success) >= 2 and len(failure) >= 2:
                rng = np.random.default_rng(_stable_seed(f"{group_id}:{metric}:{RANDOM_SEED}"))
                samples = np.empty(OUTCOME_BOOTSTRAPS, dtype=np.float64)
                for index in range(OUTCOME_BOOTSTRAPS):
                    samples[index] = (
                        rng.choice(success, len(success), replace=True).mean()
                        - rng.choice(failure, len(failure), replace=True).mean()
                    )
                row["bootstrap_ci_low"], row["bootstrap_ci_high"] = np.quantile(
                    samples, [0.025, 0.975]
                )
            rows.append(row)
    return _dataframe(rows, OUTCOME_COLUMNS)


def build_phase_episode_metrics(
    query: pd.DataFrame,
    horizon: pd.DataFrame,
    *,
    pre_success_only: bool = True,
) -> pd.DataFrame:
    """Aggregate phase associations within episode before any cross-episode summary."""

    allowed_keys = set(
        query.loc[
            ~query["success_before"].astype(bool) if pre_success_only else slice(None),
            "query_key",
        ]
    )
    common = horizon[
        (horizon["L"] == COMMON_L) & horizon["query_key"].isin(allowed_keys)
    ].copy()
    rows: list[dict[str, Any]] = []

    action_cells = common[
        (common["phase_coarse"] != "UNLABELED")
        & (common["phase_task"] != "UNLABELED")
        & ((common["source_kind"] != "fastwam") | common["executed_flag"].astype(bool))
    ]
    action_keys = [
        "group_id",
        "policy",
        "checkpoint",
        "task",
        "episode_key",
        "episode_id",
        "success",
        "phase_coarse",
        "phase_task",
    ]
    for values, frame in action_cells.groupby(action_keys, sort=False, dropna=False):
        record = dict(zip(action_keys, values))
        rows.append(
            {
                **record,
                "phase_axis": "action_phase",
                "query_state_phase": _unique_phase(frame["query_state_phase"]),
                "queries": int(frame["query_key"].nunique()),
                "cells": int(len(frame)),
                "mean_S_raw": float(frame["S_raw"].mean()),
                "mean_S_std": float(frame["S_std"].mean()),
                "mean_abs_R_std": float(frame["R_std"].abs().mean()),
                "mean_abs_I_raw": float(frame["I_raw"].abs().mean()),
                "mean_abs_I_std": float(frame["I_std"].abs().mean()),
                "mean_y_ln": float(frame["y_ln"].mean()),
            }
        )

    query_cells = (
        common.groupby("query_key", sort=False)
        .agg(
            group_id=("group_id", "first"),
            policy=("policy", "first"),
            checkpoint=("checkpoint", "first"),
            task=("task", "first"),
            episode_key=("episode_key", "first"),
            episode_id=("episode_id", "first"),
            success=("success", "first"),
            S_raw=("S_raw", "first"),
            S_std=("S_std", "first"),
            mean_abs_R_std=("R_std", lambda value: float(value.abs().mean())),
            mean_abs_I_raw=("I_raw", lambda value: float(value.abs().mean())),
            mean_abs_I_std=("I_std", lambda value: float(value.abs().mean())),
            mean_y_ln=("y_ln", "mean"),
        )
        .reset_index()
        .merge(
        query[["query_key", "query_state_phase"]],
        on="query_key",
        how="left",
        validate="one_to_one",
        )
    )
    query_cells = query_cells[
        query_cells["query_state_phase"].fillna("UNLABELED") != "UNLABELED"
    ]
    state_keys = [
        "group_id",
        "policy",
        "checkpoint",
        "task",
        "episode_key",
        "episode_id",
        "success",
        "query_state_phase",
    ]
    for values, frame in query_cells.groupby(state_keys, sort=False, dropna=False):
        record = dict(zip(state_keys, values))
        rows.append(
            {
                **record,
                "phase_axis": "query_state_phase",
                "phase_coarse": "UNLABELED",
                "phase_task": "UNLABELED",
                "queries": int(frame["query_key"].nunique()),
                "cells": int(len(frame)),
                "mean_S_raw": float(frame["S_raw"].mean()),
                "mean_S_std": float(frame["S_std"].mean()),
                "mean_abs_R_std": float(frame["mean_abs_R_std"].mean()),
                "mean_abs_I_raw": float(frame["mean_abs_I_raw"].mean()),
                "mean_abs_I_std": float(frame["mean_abs_I_std"].mean()),
                "mean_y_ln": float(frame["mean_y_ln"].mean()),
            }
        )
    return _dataframe(rows, PHASE_EPISODE_COLUMNS)


def build_phase_summary(phase_episode: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap phase means over episode aggregates, never over action cells."""

    rows: list[dict[str, Any]] = []
    if phase_episode.empty:
        return _dataframe(rows, PHASE_SUMMARY_COLUMNS)
    keys = [
        "group_id",
        "policy",
        "checkpoint",
        "task",
        "phase_axis",
        "phase_coarse",
        "phase_task",
        "query_state_phase",
    ]
    metrics = [
        "mean_S_raw",
        "mean_S_std",
        "mean_abs_R_std",
        "mean_abs_I_raw",
        "mean_abs_I_std",
        "mean_y_ln",
    ]
    for values, frame in phase_episode.groupby(keys, sort=False, dropna=False):
        labels = dict(zip(keys, values))
        for metric in metrics:
            samples = frame[metric].dropna().to_numpy(float)
            low = high = np.nan
            if len(samples) >= 2:
                rng = np.random.default_rng(
                    _stable_seed(
                        ":".join(str(labels[key]) for key in keys)
                        + f":{metric}:{RANDOM_SEED}"
                    )
                )
                boot = np.empty(OUTCOME_BOOTSTRAPS, dtype=np.float64)
                for index in range(OUTCOME_BOOTSTRAPS):
                    boot[index] = rng.choice(samples, len(samples), replace=True).mean()
                low, high = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    **labels,
                    "metric": metric,
                    "episodes": int(len(samples)),
                    "mean": float(np.mean(samples)) if len(samples) else np.nan,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "inference_limit": (
                        "episode-first descriptive bootstrap; phase labels independent; "
                        "observational, not causal"
                    ),
                }
            )
    return _dataframe(rows, PHASE_SUMMARY_COLUMNS)


def _bootstrap_position_ci(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    eligible = group[group["baseline_eligible"]]
    matrix = eligible.pivot(index="query_key", columns="h", values="y_ln")
    query_episode = eligible.drop_duplicates("query_key").set_index("query_key")["episode_key"]
    query_episode = query_episode.reindex(matrix.index)
    episodes = np.array(sorted(query_episode.unique()), dtype=object)
    episode_rows = {
        episode: np.flatnonzero(query_episode.to_numpy() == episode) for episode in episodes
    }
    rng = np.random.default_rng(_stable_seed(str(group["group_id"].iloc[0])))
    boot = np.empty((POSITION_BOOTSTRAPS, matrix.shape[1]), dtype=np.float64)
    values = matrix.to_numpy(float)
    for index in range(POSITION_BOOTSTRAPS):
        sampled = rng.choice(episodes, len(episodes), replace=True)
        rows = np.concatenate([episode_rows[value] for value in sampled])
        boot[index] = np.median(values[rows], axis=0)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return low, high


def _get_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting needs matplotlib. Re-run with --skip-plots for tables only, "
            "or use an environment that already provides matplotlib."
        ) from exc
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _group_title(group: pd.DataFrame) -> str:
    first = group.iloc[0]
    return (
        f"{first['policy']} | {first['task']} | run={first['run_id']} | "
        f"{_short_checkpoint(first['checkpoint'])}"
    )


def plot_group_figures(
    output: Path, query: pd.DataFrame, horizon: pd.DataFrame, episode: pd.DataFrame
) -> list[str]:
    plt = _get_plt()
    created: list[str] = []
    common_horizon = horizon[horizon["L"] == COMMON_L]
    for group_id, hgroup in common_horizon.groupby("group_id", sort=False):
        qgroup = query[query["group_id"] == group_id].sort_values(
            ["episode_key", "query_idx"]
        )
        egroup = episode[episode["group_id"] == group_id].sort_values("episode_key")
        group_dir = output / "figures" / _slug(group_id)
        group_dir.mkdir(parents=True, exist_ok=True)
        title = _group_title(qgroup)

        position = hgroup.groupby("h", sort=True).agg(
            b_position=("b_position", "first"), mad_scale=("mad_scale", "first")
        )
        low, high = _bootstrap_position_ci(hgroup)
        fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True, constrained_layout=True)
        axes[0].plot(position.index, position["b_position"], color="#4c78a8", label="b_h = median eligible y")
        axes[0].fill_between(position.index, low, high, color="#4c78a8", alpha=0.2, label="95% episode bootstrap CI")
        axes[0].set_ylabel("model-native y = log(V+eps)")
        axes[0].legend(loc="best")
        axes[0].set_title(f"Position profile (raw scale stays within this group)\n{title}")
        axes[1].plot(position.index, position["mad_scale"], color="#f28e2b", label="s_h = max(1.4826 MAD, floor)")
        axes[1].axhline(float(hgroup["scale_floor"].iloc[0]), color="black", linestyle="--", linewidth=1, label="fixed scale floor")
        axes[1].set_xlabel("future action position h")
        axes[1].set_ylabel("robust scale s_h")
        axes[1].legend(loc="best")
        path = group_dir / "01_position_profile.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

        order = qgroup["query_key"].tolist()
        raw = hgroup.pivot(index="query_key", columns="h", values="y_ln").reindex(order)
        raw_residual = hgroup.pivot(index="query_key", columns="h", values="r_raw").reindex(order)
        residual = hgroup.pivot(index="query_key", columns="h", values="R_std").reindex(order)
        fig, axes = plt.subplots(1, 3, figsize=(17.0, max(4.0, min(10.0, 2.8 + len(order) * 0.045))), constrained_layout=True)
        raw_values = raw.to_numpy(float)
        raw_lo, raw_hi = np.quantile(raw_values, [0.01, 0.99])
        image0 = axes[0].imshow(raw_values, aspect="auto", cmap="viridis", vmin=raw_lo, vmax=raw_hi)
        fig.colorbar(image0, ax=axes[0], fraction=0.04, label="raw y (group-native)")
        axes[0].set_title("Raw y(q,h)")
        rlim = max(float(np.quantile(np.abs(residual.to_numpy(float)), 0.98)), 1e-9)
        raw_r_lim = max(float(np.quantile(np.abs(raw_residual.to_numpy(float)), 0.98)), 1e-9)
        image1 = axes[1].imshow(raw_residual, aspect="auto", cmap="coolwarm", vmin=-raw_r_lim, vmax=raw_r_lim)
        fig.colorbar(image1, ax=axes[1], fraction=0.04, label="raw-log residual r")
        axes[1].set_title("Raw residual r=y-b")
        image2 = axes[2].imshow(residual, aspect="auto", cmap="coolwarm", vmin=-rlim, vmax=rlim)
        fig.colorbar(image2, ax=axes[2], fraction=0.04, label="standardized R")
        axes[2].set_title("Standardized residual R=r/s_h")
        for axis in axes:
            axis.set_xlabel("future h")
            axis.set_ylabel("queries, sorted by episode then query")
        fig.suptitle(f"Raw, raw-residual, and standardized-residual; baselines use pre-success queries\n{title}")
        path = group_dir / "02_raw_vs_residual_heatmap.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

        fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True, constrained_layout=True)
        representative_keys = set(egroup.loc[egroup["representative"], "episode_key"])
        for episode_key, timeline in qgroup.groupby("episode_key", sort=False):
            timeline = timeline.sort_values(["action_slot_start", "query_idx"])
            success = bool(timeline["success"].iloc[0])
            color = SUCCESS_COLOR if success else FAIL_COLOR
            is_rep = episode_key in representative_keys
            axes[0].plot(
                timeline["action_slot_start"],
                timeline["S_raw_L3"],
                color=color,
                alpha=0.9 if is_rep else 0.18,
                linewidth=2.0 if is_rep else 0.8,
                marker="o" if is_rep else None,
                markersize=3,
            )
            axes[1].plot(
                timeline["action_slot_start"],
                timeline["S_std_L3"],
                color=color,
                alpha=0.9 if is_rep else 0.18,
                linewidth=2.0 if is_rep else 0.8,
                marker="o" if is_rep else None,
                markersize=3,
            )
            if is_rep:
                success_slot = _optional_int(timeline["first_success_action_slot"].iloc[0])
                if success_slot is not None:
                    for axis in axes:
                        axis.axvline(
                            success_slot + 1,
                            color="#20732d",
                            linestyle="--",
                            linewidth=1.2,
                            alpha=0.9,
                        )
        for axis in axes:
            axis.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        axes[0].set_ylabel("S_raw = mean_h r")
        axes[1].set_ylabel("S_std = mean_h R")
        axes[1].set_xlabel("episode action slot at query start")
        axes[0].set_title(f"Raw-unit and standardized query-wide timelines (bold = representatives)\n{title}")
        axes[1].text(0.01, 0.01, "blue=successful episode; red=failed episode", transform=axes[1].transAxes, color=NEUTRAL_COLOR)
        path = group_dir / "03_S_timeline.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

        reps = egroup[egroup["representative"]].sort_values(
            ["success", "episode_key"], ascending=[False, True]
        )
        if not reps.empty:
            fig, axes = plt.subplots(
                len(reps),
                1,
                figsize=(10.2, max(3.0, 2.45 * len(reps))),
                squeeze=False,
                constrained_layout=True,
            )
            phase_names = sorted(
                set(
                    hgroup.loc[
                        (hgroup["phase_coarse"] != "UNLABELED")
                        & hgroup["executed_flag"].astype(bool),
                        "phase_coarse",
                    ].astype(str)
                )
            )
            cmap = plt.get_cmap("tab20")
            phase_colors = {
                name: cmap(index % 20) for index, name in enumerate(phase_names)
            }
            for axis, (_, erow) in zip(axes[:, 0], reps.iterrows()):
                timeline = qgroup[qgroup["episode_key"] == erow["episode_key"]].sort_values(
                    ["action_slot_start", "query_idx"]
                )
                cells = hgroup[
                    (hgroup["episode_key"] == erow["episode_key"])
                    & hgroup["executed_flag"].astype(bool)
                    & (hgroup["phase_coarse"] != "UNLABELED")
                ].sort_values("action_slot")
                for _, cell in cells.iterrows():
                    slot = _optional_int(cell["action_slot"])
                    if slot is None:
                        continue
                    name = str(cell["phase_coarse"])
                    axis.axvspan(
                        slot,
                        slot + 1,
                        color=phase_colors.get(name, NEUTRAL_COLOR),
                        alpha=0.10,
                        linewidth=0,
                    )
                axis.plot(
                    timeline["action_slot_start"],
                    timeline["S_std_L3"],
                    color=SUCCESS_COLOR if bool(erow["success"]) else FAIL_COLOR,
                    marker="o",
                    linewidth=1.8,
                    markersize=3.5,
                )
                success_slot = _optional_int(erow.get("first_success_action_slot"))
                if success_slot is not None:
                    axis.axvline(
                        success_slot + 1,
                        color="#20732d",
                        linestyle="--",
                        linewidth=1.4,
                        label="terminal success boundary",
                    )
                axis.axhline(0, color="black", linewidth=0.7, alpha=0.35)
                axis.set_ylabel("S_std")
                axis.set_title(
                    f"{erow['episode_uid']} | {erow['selection_reason']} | "
                    f"query-state={_unique_phase(timeline['query_state_phase'])}"
                )
            axes[-1, 0].set_xlabel("episode action slot; phase tint is independently annotated")
            fig.suptitle(f"Representative S timeline with action-phase background\n{title}")
            path = group_dir / "03b_representative_S_phase_timeline.png"
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            created.append(str(path))

        representative = egroup.loc[egroup["representative"], ["episode_key", "selection_reason"]]
        rep_queries = qgroup[qgroup["episode_key"].isin(representative["episode_key"])].copy()
        rep_order = rep_queries.sort_values(["episode_key", "query_idx"])["query_key"].tolist()
        imatrix = hgroup[hgroup["query_key"].isin(rep_order)].pivot(index="query_key", columns="h", values="I_std").reindex(rep_order)
        fig, axis = plt.subplots(figsize=(10.0, max(3.4, 1.8 + len(rep_order) * 0.28)), constrained_layout=True)
        ilimit = max(float(np.quantile(np.abs(imatrix.to_numpy(float)), 0.98)), 1e-9)
        image = axis.imshow(imatrix, aspect="auto", cmap="coolwarm", vmin=-ilimit, vmax=ilimit)
        fig.colorbar(image, ax=axis, fraction=0.04, label="I_std(q,h) = R-S_std")
        axis.set_xlabel("future h")
        axis.set_ylabel("representative query rows")
        if str(qgroup["source_kind"].iloc[0]) == "fastwam":
            axis.axvline(FASTWAM_EXECUTED_HORIZON - 0.5, color="black", linestyle="--", linewidth=1.1)
            axis.text(FASTWAM_EXECUTED_HORIZON + 0.2, -0.55, "unexecuted future tail", fontsize=8)
        axis.set_title(f"Within-query standardized pattern I_std; representatives use outcome-median duration\n{title}")
        path = group_dir / "04_I_representative_heatmap.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), constrained_layout=True)
        metrics = (
            ("episode_mean_S_std", "episode mean S_std"),
            ("episode_mean_abs_I_std", "episode mean |I_std|"),
            ("episode_mean_abs_R_std", "episode mean |R_std|"),
        )
        for axis, (column, label) in zip(axes, metrics):
            values = []
            labels = []
            colors = []
            for success, name, color in ((True, "success", SUCCESS_COLOR), (False, "failure", FAIL_COLOR)):
                subset = egroup.loc[egroup["success"].astype(bool) == success, column].to_numpy(float)
                if len(subset):
                    values.append(subset)
                    labels.append(f"{name}\nn={len(subset)}")
                    colors.append(color)
            boxes = axis.boxplot(values, labels=labels, patch_artist=True, widths=0.55)
            for patch, color in zip(boxes["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.45)
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=0.2)
        fig.suptitle(f"Episode-level outcome summaries (descriptive)\n{title}")
        path = group_dir / "05_outcome_summary.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

        lengths = sorted(
            int(column.rsplit("L", 1)[1])
            for column in qgroup.columns
            if column.startswith("log_V_total_L") and qgroup[column].notna().any()
        )
        fig, axis = plt.subplots(figsize=(7.8, 4.0), constrained_layout=True)
        data = [qgroup[f"log_V_total_L{length}"].dropna().to_numpy(float) for length in lengths]
        axis.boxplot(data, labels=[f"L={value}" for value in lengths], showfliers=False)
        rng = np.random.default_rng(_stable_seed(group_id))
        for xpos, values in enumerate(data, start=1):
            axis.scatter(xpos + rng.normal(0, 0.035, len(values)), values, s=10, alpha=0.25)
        axis.set_ylabel("log(sum_h V_L), model-native")
        axis.set_title(f"Tail-length sensitivity; compare ranks, not absolute L scales\n{title}")
        path = group_dir / "06_L_sensitivity.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))
    return created


def plot_cross_policy_standardized(output: Path, query: pd.DataFrame, horizon: pd.DataFrame) -> str:
    plt = _get_plt()
    common = horizon[horizon["L"] == COMMON_L]
    groups = list(query.groupby("group_id", sort=False))
    fig, axes = plt.subplots(len(groups), 3, figsize=(12.0, max(3.0, 2.65 * len(groups))), squeeze=False, constrained_layout=True)
    for row_index, (group_id, qgroup) in enumerate(groups):
        hgroup = common[common["group_id"] == group_id]
        values = (
            (hgroup["R_std"].to_numpy(float), "R_std cells"),
            (qgroup["S_std_L3"].to_numpy(float), "S_std queries"),
            (hgroup["I_std"].to_numpy(float), "I_std cells"),
        )
        for axis, (array, label) in zip(axes[row_index], values):
            axis.hist(array, bins=40, density=True, color="#4c78a8", alpha=0.65)
            axis.axvline(0, color="black", linewidth=0.8)
            axis.set_xlabel(label)
            axis.set_ylabel("density")
        axes[row_index, 0].set_title(_group_title(qgroup))
    fig.suptitle("Cross-policy standardized structure only (raw V/y excluded)")
    path = output / "figures" / "cross_policy_standardized_panel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _resolve_video(raw_path: Any, source_root: Path, video_roots: Sequence[Path]) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    raw = Path(text)
    direct = [raw, source_root / raw]
    for candidate in direct:
        if candidate.is_file():
            return candidate.resolve()
    basename = raw.name
    search_roots = [source_root, *video_roots]
    matches: list[Path] = []
    for root in search_roots:
        root = root.expanduser().resolve()
        if root.is_dir():
            matches.extend(root.rglob(basename))
    unique = sorted({value.resolve() for value in matches if value.is_file()})
    return unique[0] if len(unique) == 1 else None


def _read_video_frame(path: Path, frame_index: int) -> Image.Image:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Fast-WAM video storyboard extraction needs opencv-python") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    finally:
        capture.release()


def _evenly_spaced_rows(frame: pd.DataFrame, count: int = 6) -> pd.DataFrame:
    if len(frame) <= count:
        return frame
    indices = np.unique(np.linspace(0, len(frame) - 1, count).round().astype(int))
    return frame.iloc[indices]


def build_fastwam_storyboards(
    output: Path,
    action: pd.DataFrame,
    episode: pd.DataFrame,
    *,
    video_roots: Sequence[Path],
) -> tuple[list[str], pd.DataFrame]:
    storyboard_dir = output / "storyboards"
    storyboard_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    created: list[str] = []
    representatives = episode[
        (episode["source_kind"] == "fastwam") & episode["representative"]
    ].sort_values(["group_id", "episode_key"])
    for _, erow in representatives.iterrows():
        actions = action[action["episode_key"] == erow["episode_key"]].sort_values("action_slot")
        if actions.empty:
            index_rows.append(
                {
                    "episode_key": erow["episode_key"],
                    "selection_reason": erow["selection_reason"],
                    "status": "no_exact_action_frame_mapping",
                    "video_path": erow["video_path"],
                    "storyboard_path": "",
                }
            )
            continue
        video = _resolve_video(
            erow["video_path"], Path(str(erow["source_root"])), video_roots
        )
        if video is None:
            index_rows.append(
                {
                    "episode_key": erow["episode_key"],
                    "selection_reason": erow["selection_reason"],
                    "status": "video_not_uniquely_resolved",
                    "video_path": erow["video_path"],
                    "storyboard_path": "",
                }
            )
            continue
        selected = _evenly_spaced_rows(actions, 6)
        terminal_actions = actions[actions["terminal_success_action"].astype(bool)]
        if not terminal_actions.empty:
            selected = (
                pd.concat([selected, terminal_actions], ignore_index=False)
                .drop_duplicates(["query_key", "h"])
                .sort_values("action_slot")
            )
        tiles: list[Image.Image] = []
        try:
            for _, arow in selected.iterrows():
                frame = _read_video_frame(video, int(arow["video_frame_index"]))
                frame.thumbnail((360, 260), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (380, 370), "white")
                tile.paste(frame, ((380 - frame.width) // 2, 4))
                draw = ImageDraw.Draw(tile)
                lines = [
                    f"slot={int(arow['action_slot'])} frame={int(arow['video_frame_index'])}"
                    + (" [terminal action]" if bool(arow["terminal_success_action"]) else ""),
                    f"q={int(arow['query_idx'])} h={int(arow['h'])} phase={arow['phase_coarse']}/{arow['phase_task']}",
                    f"query-state={arow['query_state_phase']}",
                    f"S_std={arow['S_std']:+.2f} R_std={arow['R_std']:+.2f} I_std={arow['I_std']:+.2f}",
                    f"y_ln={arow['y_ln']:+.3f} S_raw={arow['S_raw']:+.3f}",
                ]
                y = 266
                for line in lines:
                    draw.text((8, y), line, fill="black")
                    y += 18
                tiles.append(tile)
            terminal_frame = _optional_int(erow.get("terminal_success_video_frame"))
            if terminal_frame is not None:
                frame = _read_video_frame(video, terminal_frame)
                frame.thumbnail((360, 260), Image.Resampling.LANCZOS)
                tile = Image.new("RGB", (380, 370), "white")
                tile.paste(frame, ((380 - frame.width) // 2, 4))
                draw = ImageDraw.Draw(tile)
                draw.text(
                    (8, 268),
                    f"frame={terminal_frame} [terminal success boundary]",
                    fill="#20732d",
                )
                draw.text(
                    (8, 288),
                    f"after action slot={_optional_int(erow.get('first_success_action_slot'))}",
                    fill="black",
                )
                draw.text(
                    (8, 308),
                    "post-action boundary frame; no future-h metric assigned",
                    fill=NEUTRAL_COLOR,
                )
                tiles.append(tile)
        except RuntimeError as exc:
            index_rows.append(
                {
                    "episode_key": erow["episode_key"],
                    "selection_reason": erow["selection_reason"],
                    "status": f"frame_extraction_failed: {exc}",
                    "video_path": str(video),
                    "storyboard_path": "",
                }
            )
            continue
        columns = min(3, len(tiles))
        rows = int(math.ceil(len(tiles) / columns))
        canvas = Image.new("RGB", (columns * 380, rows * 370 + 52), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8),
            f"{erow['policy']} | {erow['task']} | {erow['episode_uid']} | {erow['selection_reason']}",
            fill="black",
        )
        draw.text(
            (8, 28),
            "Frames are fresh pre-action images; metrics use only actually executed h<24. Phase is independent annotation or UNLABELED.",
            fill=NEUTRAL_COLOR,
        )
        for index, tile in enumerate(tiles):
            canvas.paste(tile, ((index % columns) * 380, 52 + (index // columns) * 370))
        path = storyboard_dir / f"{_slug(erow['group_id'], 44)}__{_slug(erow['episode_uid'], 32)}.png"
        canvas.save(path)
        created.append(str(path))
        index_rows.append(
            {
                "episode_key": erow["episode_key"],
                "selection_reason": erow["selection_reason"],
                "status": "created",
                "video_path": str(video),
                "storyboard_path": str(path),
            }
        )
    return created, _dataframe(index_rows, STORYBOARD_INDEX_COLUMNS)


def build_query_frame_strips(
    output: Path, query: pd.DataFrame, episode: pd.DataFrame
) -> list[str]:
    """Create query-level strips; safe for pi0 but explicitly not future-h alignment."""

    created: list[str] = []
    strip_dir = output / "query_frame_strips"
    strip_dir.mkdir(parents=True, exist_ok=True)
    representatives = episode[episode["representative"]].sort_values(
        ["group_id", "episode_key"]
    )
    for _, erow in representatives.iterrows():
        rows = query[query["episode_key"] == erow["episode_key"]].sort_values("query_idx")
        rows = _evenly_spaced_rows(rows, 8)
        tiles: list[Image.Image] = []
        for _, qrow in rows.iterrows():
            path = Path(str(qrow.get("head_image_path") or ""))
            if not path.is_file():
                continue
            image = Image.open(path).convert("RGB")
            image.thumbnail((300, 220), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (320, 270), "white")
            tile.paste(image, ((320 - image.width) // 2, 3))
            draw = ImageDraw.Draw(tile)
            draw.text(
                (7, 230),
                f"query={int(qrow['query_idx'])} slot={int(qrow['action_slot_start'])} S_std={qrow['S_std_L3']:+.2f}",
                fill="black",
            )
            draw.text((7, 248), "query boundary only", fill=NEUTRAL_COLOR)
            tiles.append(tile)
        if not tiles:
            continue
        canvas = Image.new("RGB", (len(tiles) * 320, 304), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (7, 6),
            f"{erow['policy']} | {erow['task']} | {erow['episode_uid']} | {erow['selection_reason']}",
            fill="black",
        )
        draw.text(
            (7, 24),
            "Query-input frames align to S_std(q); they do not identify an individual future h physics frame.",
            fill=NEUTRAL_COLOR,
        )
        for index, tile in enumerate(tiles):
            canvas.paste(tile, (index * 320, 34))
        path = strip_dir / f"{_slug(erow['group_id'], 44)}__{_slug(erow['episode_uid'], 32)}.png"
        canvas.save(path)
        created.append(str(path))
    return created


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_outputs(
    output: Path,
    query: pd.DataFrame,
    horizon: pd.DataFrame,
    episode: pd.DataFrame,
    episode_all_queries: pd.DataFrame,
    action: pd.DataFrame,
    l_sensitivity: pd.DataFrame,
    outcome: pd.DataFrame,
    outcome_all_queries: pd.DataFrame,
    phase_episode: pd.DataFrame,
    phase_summary: pd.DataFrame,
    phase_episode_all_queries: pd.DataFrame,
    phase_summary_all_queries: pd.DataFrame,
    source_inventory: list[dict[str, Any]],
    *,
    scale_floor: float,
    skip_plots: bool,
    skip_storyboards: bool,
    video_roots: Sequence[Path],
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output}")
    output.mkdir(parents=True)
    query.to_csv(output / "query_metrics.csv", index=False)
    horizon.to_csv(output / "query_horizon.csv", index=False)
    episode.to_csv(output / "episode_metrics.csv", index=False)
    episode_all_queries.to_csv(
        output / "episode_metrics_all_queries.csv", index=False
    )
    action.to_csv(output / "fastwam_action_frame_metrics.csv", index=False)
    l_sensitivity.to_csv(output / "L_sensitivity.csv", index=False)
    outcome.to_csv(output / "outcome_summary.csv", index=False)
    outcome_all_queries.to_csv(output / "outcome_summary_all_queries.csv", index=False)
    phase_episode.to_csv(output / "phase_episode_metrics.csv", index=False)
    phase_summary.to_csv(output / "phase_summary.csv", index=False)
    phase_episode_all_queries.to_csv(
        output / "phase_episode_metrics_all_queries.csv", index=False
    )
    phase_summary_all_queries.to_csv(
        output / "phase_summary_all_queries.csv", index=False
    )

    query_annotation_columns = [
        "run_id",
        "episode_id",
        "query_idx",
        "query_key",
        "action_slot_start",
        "query_state_phase",
        "phase_coarse",
        "phase_task",
        "phase_source",
        "phase_confidence",
    ]
    query_template = query[
        [
            "run_id",
            "episode_id",
            "query_idx",
            "query_key",
            "action_slot_start",
            "query_state_phase",
        ]
    ].copy()
    query_template["query_state_phase"] = ""
    query_template["phase_coarse"] = ""
    query_template["phase_task"] = ""
    query_template["phase_source"] = ""
    query_template["phase_confidence"] = ""
    query_template[query_annotation_columns].to_csv(
        output / "query_phase_annotation_template.csv", index=False
    )

    annotation_columns = [
        "run_id",
        "episode_id",
        "query_idx",
        "h",
        "action_slot",
        "video_frame_index",
        "query_state_phase",
        "phase_coarse",
        "phase_task",
        "phase_source",
        "phase_confidence",
    ]
    if action.empty:
        pd.DataFrame(columns=annotation_columns).to_csv(
            output / "action_phase_annotation_template.csv", index=False
        )
    else:
        template = action[annotation_columns].copy()
        template["query_state_phase"] = ""
        template["phase_coarse"] = ""
        template["phase_task"] = ""
        template["phase_source"] = ""
        template["phase_confidence"] = ""
        template.to_csv(output / "action_phase_annotation_template.csv", index=False)

    figures: list[str] = []
    if not skip_plots:
        figures.extend(plot_group_figures(output, query, horizon, episode))
        figures.append(plot_cross_policy_standardized(output, query, horizon))

    query_strips: list[str] = []
    storyboards: list[str] = []
    storyboard_index = _dataframe([], STORYBOARD_INDEX_COLUMNS)
    if not skip_storyboards:
        query_strips = build_query_frame_strips(output, query, episode)
        storyboards, storyboard_index = build_fastwam_storyboards(
            output, action, episode, video_roots=video_roots
        )
    storyboard_index.to_csv(output / "storyboard_index.csv", index=False)

    identity_y = float(
        np.max(
            np.abs(horizon["y_ln"] - (horizon["b_position"] + horizon["r_raw"]))
        )
    )
    identity_raw_four = float(
        np.max(
            np.abs(
                horizon["y_ln"]
                - (
                    horizon["mu"]
                    + horizon["P_h"]
                    + horizon["S_raw"]
                    + horizon["I_raw"]
                )
            )
        )
    )
    identity_std = float(
        np.max(np.abs(horizon["R_std"] - (horizon["S_std"] + horizon["I_std"])))
    )
    group_summaries = []
    for group_id, group in query.groupby("group_id", sort=False):
        episodes = episode[episode["group_id"] == group_id]
        group_summaries.append(
            {
                "group_id": group_id,
                "policy": str(group["policy"].iloc[0]),
                "checkpoint": str(group["checkpoint"].iloc[0]),
                "task": str(group["task"].iloc[0]),
                "run_id": str(group["run_id"].iloc[0]),
                "cohort_id": str(group["cohort_id"].iloc[0]),
                "queries": int(len(group)),
                "episodes": int(len(episodes)),
                "success_episodes": int(episodes["success"].astype(bool).sum()),
                "failure_episodes": int((~episodes["success"].astype(bool)).sum()),
                "horizon": int(group["horizon"].iloc[0]),
                "denoising_steps": int(group["denoising_steps"].iloc[0]),
                "tail_lengths": str(group["tail_lengths"].iloc[0]),
                "scale_floored_position_bins_all_L": int(
                    horizon.loc[
                        horizon["group_id"] == group_id,
                        ["L", "h", "scale_floored"],
                    ]
                    .drop_duplicates(["L", "h"])["scale_floored"]
                    .sum()
                ),
                "representatives": episodes.loc[
                    episodes["representative"], ["episode_uid", "selection_reason"]
                ].to_dict("records"),
            }
        )
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "sources": len(source_inventory),
            "groups": int(query["group_id"].nunique()),
            "queries": int(len(query)),
            "pre_success_queries": int((~query["success_before"].astype(bool)).sum()),
            "post_success_queries_descriptive_only": int(
                query["success_before"].astype(bool).sum()
            ),
            "episodes": int(len(episode)),
            "horizon_rows": int(len(horizon)),
            "fastwam_executed_action_frame_rows": int(len(action)),
        },
        "formula": {
            "V_L": "sum_d population_var(z[-L:,h,d])",
            "y": "natural_log(V_L + 1e-12)",
            "baseline": (
                "success_before=false queries only; separately within "
                "policy/checkpoint/task/run/cohort/L/h"
            ),
            "position": "b_position=median_eligible(y_ln)",
            "scale": "mad_scale=max(1.4826*MAD_eligible(y_ln),scale_floor)",
            "residual": "r_raw=y_ln-b_position; R_std=r_raw/mad_scale",
            "raw_four_way": "mu=mean_h(b); P_h=b-mu; S_raw=mean_h(r_raw); I_raw=r_raw-S_raw",
            "standardized_two_way": "S_std=mean_h(R_std); I_std=R_std-S_std",
            "identities": [
                "y_ln=b_position+r_raw",
                "y_ln=mu+P_h+S_raw+I_raw",
                "R_std=S_std+I_std",
            ],
            "invalid_identity": "y_ln=b_position+S_std+I_std is invalid because units differ",
            "common_L": COMMON_L,
            "pi0_L_sensitivity": list(PI0_TAIL_LENGTHS),
            "fastwam_L_sensitivity": list(FASTWAM_TAIL_LENGTHS),
            "scale_floor": float(scale_floor),
        },
        "identity_checks": {
            "max_abs_y_minus_b_plus_r": identity_y,
            "max_abs_y_minus_raw_four_way": identity_raw_four,
            "max_abs_Rstd_minus_Sstd_plus_Istd": identity_std,
        },
        "groups": group_summaries,
        "source_inventory": source_inventory,
        "summary_scopes": {
            "episode_metrics.csv": "main pre-success queries only",
            "outcome_summary.csv": "main pre-success episode aggregates",
            "phase_episode_metrics.csv": "main pre-success, episode-first",
            "episode_metrics_all_queries.csv": "all-query descriptive companion",
            "outcome_summary_all_queries.csv": "all-query descriptive companion",
            "phase_episode_metrics_all_queries.csv": "all-query descriptive companion",
        },
        "artifacts": {
            "figures": figures,
            "query_frame_strips": query_strips,
            "fastwam_storyboards": storyboards,
        },
        "interpretation_limits": [
            "Raw V/y remains model-native and is never compared directly across pi0 and Fast-WAM.",
            "R_std/S_std/I_std support standardized structure, rank, phase, and outcome association only.",
            "DVAC is an empirical denoising-stability proxy, not calibrated uncertainty or safety.",
            "Phase labels are independent annotations; missing labels remain UNLABELED.",
            "Fast-WAM video mapping uses only fresh per-action frame ranges and executed h<24.",
            "Fast-WAM h=24..31 is an unexecuted future tail and receives no action/video phase.",
            "pi0 official combined video is query-boundary only; no future-h physics frame is inferred.",
            "Representative episodes are selected by outcome-class median action duration with fixed tie-breaks, never by DVAC or visual appeal.",
            "Outcome summaries aggregate within episode before descriptive bootstrap; no causal claim is made.",
        ],
    }
    summary = _json_safe(summary)
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def analyze(
    *,
    pi0_sources: Sequence[Path],
    fastwam_sources: Sequence[Path],
    output: Path,
    scale_floor: float = DEFAULT_SCALE_FLOOR,
    phase_annotations: Path | None = None,
    official_seed_maps: Sequence[Path] = (),
    skip_plots: bool = False,
    skip_storyboards: bool = False,
    video_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    output_resolved = output.expanduser().resolve()
    if output_resolved.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {output_resolved}")
    _preflight_optional_dependencies(
        skip_plots=skip_plots, skip_storyboards=skip_storyboards
    )
    official_seed_map = _load_official_seed_maps(official_seed_maps)
    traces: list[QueryTrace] = []
    source_inventory: list[dict[str, Any]] = []
    for source in pi0_sources:
        loaded, _episodes, inventory = load_pi0_source(source)
        traces.extend(loaded)
        source_inventory.append(inventory)
    for source in fastwam_sources:
        loaded, _episodes, inventory = load_fastwam_source(
            source, official_seed_map=official_seed_map
        )
        traces.extend(loaded)
        source_inventory.append(inventory)
    query, horizon = derive_metrics(traces, scale_floor=scale_floor)
    horizon = apply_phase_annotations(horizon, phase_annotations)
    query = attach_query_phases(query, horizon)
    episode = build_episode_metrics(query, horizon, pre_success_only=True)
    episode_all_queries = build_episode_metrics(
        query, horizon, pre_success_only=False
    )
    action = build_action_metrics(query, horizon)
    l_sensitivity = build_l_sensitivity(query)
    outcome = build_outcome_summary(episode)
    outcome_all_queries = build_outcome_summary(episode_all_queries)
    phase_episode = build_phase_episode_metrics(
        query, horizon, pre_success_only=True
    )
    phase_summary = build_phase_summary(phase_episode)
    phase_episode_all_queries = build_phase_episode_metrics(
        query, horizon, pre_success_only=False
    )
    phase_summary_all_queries = build_phase_summary(phase_episode_all_queries)
    return write_outputs(
        output_resolved,
        query,
        horizon,
        episode,
        episode_all_queries,
        action,
        l_sensitivity,
        outcome,
        outcome_all_queries,
        phase_episode,
        phase_summary,
        phase_episode_all_queries,
        phase_summary_all_queries,
        source_inventory,
        scale_floor=scale_floor,
        skip_plots=skip_plots,
        skip_storyboards=skip_storyboards,
        video_roots=video_roots,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pi0 and Fast-WAM DVAC telemetry without changing source runs."
    )
    parser.add_argument("--pi0-source", type=Path, action="append", default=[])
    parser.add_argument("--fastwam-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale-floor", type=float, default=DEFAULT_SCALE_FLOOR)
    parser.add_argument("--phase-annotations", type=Path)
    parser.add_argument(
        "--official-seed-map",
        type=Path,
        action="append",
        default=[],
        help=(
            "Read-only CSV sidecar with run_id,episode_id,source_seed; may be repeated. "
            "Payload files are never rewritten."
        ),
    )
    parser.add_argument("--video-root", type=Path, action="append", default=[])
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-storyboards", action="store_true")
    args = parser.parse_args(argv)
    if not args.pi0_source and not args.fastwam_source:
        parser.error("at least one --pi0-source or --fastwam-source is required")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = analyze(
        pi0_sources=args.pi0_source,
        fastwam_sources=args.fastwam_source,
        output=args.output,
        scale_floor=args.scale_floor,
        phase_annotations=args.phase_annotations,
        official_seed_maps=args.official_seed_map,
        skip_plots=args.skip_plots,
        skip_storyboards=args.skip_storyboards,
        video_roots=args.video_root,
    )
    print(f"ANALYSIS_OUTPUT={args.output.expanduser().resolve()}")
    print(json.dumps(summary["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
