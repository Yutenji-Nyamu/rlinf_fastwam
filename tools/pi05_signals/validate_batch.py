"""CPU-only validation of persisted action signals, native masks, and videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from run_batch import (ACTION_DIM, BATCH, HORIZON, MAIN_STEPS, SCHEMA,
                       SIDE_STEPS, SIGNALS, compute_scores, save, save_npz)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_query(path, previous_end, previous_success, config):
    query = int(path.stem.split("_")[1])
    metadata = _json(path.with_suffix(".json"))
    _require(metadata["schema"] == SCHEMA and metadata["query"] == query, "query metadata/schema mismatch")
    _require(len(metadata["task_descriptions"]) == BATCH and
             all(isinstance(text, str) for text in metadata["task_descriptions"]), "invalid task text")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    layers = int(data["num_layers"])
    width = int(data["hidden_size"])
    shapes = {"x_chain": (BATCH, 11, HORIZON, ACTION_DIM),
        "z_endpoint": (BATCH, MAIN_STEPS, HORIZON, ACTION_DIM),
        "velocity": (BATCH, MAIN_STEPS, HORIZON, ACTION_DIM), "timesteps": (MAIN_STEPS,),
        "final_model_action": (BATCH, HORIZON, ACTION_DIM),
        "side_action_model": (BATCH, HORIZON, ACTION_DIM), "env_action": (BATCH, HORIZON, ACTION_DIM),
        "last_layer_hidden": (BATCH, MAIN_STEPS, HORIZON, width),
        "sr_last_hidden": (BATCH, HORIZON, width), "layer_norms": (BATCH, MAIN_STEPS, layers, HORIZON),
        "shift_first_mean": (BATCH, HORIZON, width), "shift_last_mean": (BATCH, HORIZON, width),
        "hook_counts": (MAIN_STEPS, layers), "submitted_mask": (BATCH, HORIZON),
        "executed_mask": (BATCH, HORIZON), "active": (BATCH,), "success_after": (BATCH,),
        "action_slot_start": (BATCH,), "action_slot_end": (BATCH,)}
    for name, shape in shapes.items():
        _require(name in data and data[name].shape == shape, f"{name} shape must be {shape}")
        _require(np.isfinite(data[name]).all(), f"nonfinite raw {name}")
    _require(layers >= 3 and width > 0, "invalid runtime expert dimensions")
    _require(int(data["num_steps"]) == MAIN_STEPS and int(data["action_horizon"]) == HORIZON,
             "observer dimensions mismatch")
    _require(int(data["norm_tail_steps"]) == 5 and int(data["norm_tail_layers"]) == 3,
             "Norm window metadata mismatch")
    _require(str(data["capture_branch"]) == "main", "observer captured a non-main branch")
    _require(np.array_equal(data["step_indices"], np.arange(MAIN_STEPS)), "wrong hook step order")
    _require(np.array_equal(data["layer_indices"], np.arange(layers)), "wrong hook layer order")
    _require(np.all(data["hook_counts"] == 1), "missing/duplicated/side-chain hooks")
    _require(data["last_layer_hidden"].dtype == np.float16, "hidden storage is not float16")
    for key in ("sr_last_hidden", "layer_norms", "shift_first_mean", "shift_last_mean"):
        _require(data[key].dtype == np.float32, f"{key} storage must be float32")
    _require(np.array_equal(data["last_layer_hidden"][:, -1], data["sr_last_hidden"].astype(np.float16)),
             "FP16 hidden storage is not the rounded FP32 SR input")
    full_noise = data["initial_noise_full"]
    _require(full_noise.ndim == 3 and full_noise.shape[:2] == (BATCH, HORIZON) and
             full_noise.shape[-1] >= ACTION_DIM and np.isfinite(full_noise).all(), "invalid full model noise")
    _require(np.array_equal(full_noise[..., :ACTION_DIM], data["x_chain"][:, 0]), "initial noise mismatch")
    _require(np.array_equal(data["x_chain"][:, -1], data["final_model_action"]), "final main endpoint mismatch")
    _require(int(data["side_num_steps"]) == SIDE_STEPS, "side step count must be 5")
    _require(np.all(np.diff(data["timesteps"]) < 0), "native noise-to-data schedule must decrease")
    _require(np.allclose(data["timesteps"], np.linspace(1, 0.1, MAIN_STEPS), rtol=0, atol=1e-6),
             "changed native M10 evaluation times")
    _require(data["side_timesteps"].shape == (SIDE_STEPS,) and
             np.allclose(data["side_timesteps"], np.linspace(1, 0.2, SIDE_STEPS), rtol=0, atol=1e-6),
             "changed side M5 evaluation grid")
    _require(float(data["side_final_time"]) == 0.0, "side endpoint is not t=0")
    for key in ("submitted_mask", "executed_mask", "active", "success_after"):
        _require(data[key].dtype == np.bool_, f"{key} must be a bool mask")
    _require(np.array_equal(data["action_slot_start"], previous_end), "action counters not continuous")
    delta = data["action_slot_end"] - data["action_slot_start"]
    _require(np.all(delta >= 0), "action counter decreased")
    expected_active = ~previous_success & (previous_end < config["step_limit"])
    _require(np.array_equal(data["active"], expected_active), "active mask changed native behavior")
    expected_submitted = expected_active[:, None] & (np.arange(HORIZON)[None, :] < delta[:, None])
    _require(np.array_equal(data["submitted_mask"], expected_submitted), "submitted mask differs from old DV50")
    _require(np.array_equal(data["executed_mask"], expected_submitted & ~data["success_after"][:, None]),
             "executed-prefix mask incorrectly claims terminal TOPP action alignment")
    recomputed = compute_scores(data)
    summaries = {}
    for name, result in recomputed.items():
        score, valid = data[name], data[name + "_valid"]
        _require(score.shape == valid.shape == (BATCH, HORIZON), f"{name} score/mask shape")
        _require(valid.dtype == np.bool_, f"{name} valid mask is not bool")
        _require(np.array_equal(valid, result.valid), f"{name} valid mask not reproducible")
        _require(np.all(np.isnan(score[~valid])) and np.isfinite(score[valid]).all(),
                 f"{name} NaN/valid convention violated")
        _require(np.allclose(score, result.score, rtol=1e-10, atol=1e-12, equal_nan=True),
                 f"{name} cannot be recomputed from stored arrays")
        _require(np.all(score[valid] >= 0), f"negative {name} score")
        undefined = int((~valid).sum())
        expected_missing = BATCH if name == "geoaac" else 0
        _require(metadata["invalid_score_counts"][name] == undefined, f"{name} invalid metadata mismatch")
        if name == "geoaac":
            _require(np.isnan(score[:, 0]).all() and not valid[:, 0].any(), "GeoAAC first action must be missing")
        summaries[name] = {"valid": int(valid.sum()), "invalid": undefined,
            "invalid_beyond_geoaac_first": max(0, undefined - expected_missing),
            "all_valid_values_zero": bool(np.all(score[valid] == 0)),
            "minimum": float(score[valid].min()) if valid.any() else None,
            "maximum": float(score[valid].max()) if valid.any() else None}
    with np.load(path.with_name(path.stem + "_obs.npz"), allow_pickle=False) as obs:
        main, wrist = obs["main_images"], obs["wrist_images"]
        _require(main.ndim == 4 and main.shape[0] == BATCH and main.shape[-1] == 3
                 and min(main.shape[1:3]) > 0 and main.dtype == np.uint8,
                 "replay main images must preserve native uint8 [16,H,W,3] with positive H/W")
        _require(wrist.ndim == 5 and wrist.shape[:2] == (BATCH, 2) and wrist.shape[-1] == 3
                 and min(wrist.shape[2:4]) > 0 and wrist.dtype == np.uint8,
                 "replay wrist images must preserve native uint8 [16,2,Hw,Ww,3] with positive Hw/Ww")
        _require(obs["states"].shape == (BATCH, ACTION_DIM) and np.isfinite(obs["states"]).all(),
                 "invalid replay observation states")
        _require(metadata["extra_view_images_none"] == ("extra_view_images" not in obs.files),
                 "extra camera None metadata mismatch")
    return {"query": query, "hooks": [MAIN_STEPS, layers], "signals": summaries}, data["action_slot_end"], previous_success | data["success_after"]


def validate_batch(config_or_directory, *, raise_on_failure=True):
    source = Path(config_or_directory)
    config = _json(source / "config.json") if source.is_dir() else _json(source)
    out = source if source.is_dir() else Path(config["output"])
    report = {"time": time.time(), "schema": SCHEMA, "output": str(out), "passed": False,
              "queries": [], "videos": [], "errors": [], "warnings": []}
    try:
        import cv2
        done, episodes, smoke, schema = (_json(out / name) for name in
            ("done.json", "episodes.json", "model-smoke.json", "schema.json"))
        _require(not (out / "error.json").exists(), "batch has error.json")
        _require(done["schema"] == schema["schema"] == SCHEMA, "batch schema mismatch")
        _require(done["episodes"] == len(episodes) == BATCH, "batch is not 16 complete episodes")
        _require(len({item["actual"] for item in episodes}) == BATCH, "native setup retries duplicated actual seeds")
        parity_keys = {"env_action", "x_chain", "z_endpoint", "timesteps", "final_model_action", "full_model_chains"}
        rng_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
        _require(set(smoke["main_exact_parity"]) == parity_keys and
                 all(value is True for value in smoke["main_exact_parity"].values()),
                 "synthetic main parity missing or failed")
        _require(set(smoke["rng_after_equal"]) == rng_keys and
                 all(value is True for value in smoke["rng_after_equal"].values()),
                 "synthetic RNG-after equality missing or failed")
        _require(smoke["hook_counts_all_one"] and smoke["side_steps"] == SIDE_STEPS and
                 smoke["side_schedule_final_time"] == smoke["side_reported_final_time"] == 0.0,
                 "synthetic hooks/side schedule failed")
        _require(schema["tail_n"] == 5 and schema["ugrow_steps"] == [10, 5], "signal settings mismatch")
        traces = sorted(path for path in out.glob("query_*.npz") if re.fullmatch(r"query_\d+", path.stem))
        _require(len(traces) == done["queries"] and bool(traces), "trace count does not match completion")
        _require([int(path.stem.split("_")[1]) for path in traces] == list(range(len(traces))), "query numbering has gaps")
        previous_end = np.zeros(BATCH, dtype=int)
        previous_success = np.zeros(BATCH, dtype=bool)
        for path in traces:
            row, previous_end, previous_success = validate_query(path, previous_end, previous_success, config)
            report["queries"].append(row)
            for name, values in row["signals"].items():
                if values["invalid_beyond_geoaac_first"]:
                    report["warnings"].append({"query": row["query"], "signal": name,
                        "message": "mathematically invalid values reproduced from raw inputs",
                        "count": values["invalid_beyond_geoaac_first"]})
        _require(np.all(previous_success | (previous_end >= config["step_limit"])), "incomplete native episodes")
        _require(done["successes"] == int(previous_success.sum()), "success count mismatch")
        for slot, episode in enumerate(episodes):
            _require(episode["slot"] == slot and episode["success"] == bool(previous_success[slot]) and
                     episode["action_slots_submitted"] == int(previous_end[slot]), "episode summary mismatch")
            cap = cv2.VideoCapture(str(out / episode["video"]))
            decoded = 0
            try:
                _require(cap.isOpened(), f"cannot open video {episode['video']}")
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    _require(frame is not None and frame.ndim == 3, "invalid decoded frame")
                    decoded += 1
            finally:
                cap.release()
            _require(decoded == episode["frames"] == len(traces) + 1 and decoded >= 2,
                     f"video {episode['video']} decoded {decoded}, expected {len(traces) + 1}")
            report["videos"].append({"slot": slot, "path": episode["video"], "decoded_frames": decoded})
        report["passed"] = True
        report["episodes"] = BATCH
        report["successes"] = done["successes"]
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    report["finished_time"] = time.time()
    save(out / "validation.json", report)
    if not report["passed"] and raise_on_failure:
        raise RuntimeError(f"Signal batch validation failed; see {out / 'validation.json'}")
    return report


def run_self_test(output):
    """Generate CPU-only evidence and verify acceptance plus corruption rejection.

    This exercises the file/video validator, not a model, simulator, or GPU.
    Use a fresh output directory; all artifacts explicitly say synthetic.
    """
    import cv2

    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    config = {"output": str(out), "step_limit": HORIZON, "synthetic": True}
    save(out / "config.json", config)
    save(out / "schema.json", {"schema": SCHEMA, "tail_n": 5, "ugrow_steps": [10, 5], "synthetic": True})
    save(out / "done.json", {"schema": SCHEMA, "episodes": BATCH, "queries": 1,
                            "successes": 0, "synthetic": True})
    save(out / "model-smoke.json", {"synthetic": True,
        "main_exact_parity": {key: True for key in ("env_action", "x_chain", "z_endpoint", "timesteps",
                                                   "final_model_action", "full_model_chains")},
        "rng_after_equal": {key: True for key in ("python", "numpy", "torch_cpu", "torch_cuda")},
        "hook_counts_all_one": True, "side_steps": SIDE_STEPS,
        "side_schedule_final_time": 0.0, "side_reported_final_time": 0.0})
    shape = (BATCH, MAIN_STEPS, HORIZON, ACTION_DIM)
    last_hidden = np.ones((BATCH, MAIN_STEPS, HORIZON, 2), dtype=np.float32)
    data = {
        "x_chain": np.broadcast_to(-np.linspace(0, 1, MAIN_STEPS + 1, dtype=np.float32)[None, :, None, None],
                                    (BATCH, MAIN_STEPS + 1, HORIZON, ACTION_DIM)).copy(),
        "z_endpoint": -np.ones(shape, dtype=np.float32), "velocity": np.ones(shape, dtype=np.float32),
        "timesteps": np.linspace(1, 0.1, MAIN_STEPS, dtype=np.float32),
        "final_model_action": -np.ones((BATCH, HORIZON, ACTION_DIM), dtype=np.float32),
        "side_action_model": -np.ones((BATCH, HORIZON, ACTION_DIM), dtype=np.float32),
        "env_action": -np.ones((BATCH, HORIZON, ACTION_DIM), dtype=np.float32),
        "initial_noise_full": np.zeros((BATCH, HORIZON, 32), dtype=np.float32),
        "side_num_steps": np.asarray(SIDE_STEPS, dtype=np.int64),
        "side_timesteps": np.linspace(1, 0.2, SIDE_STEPS, dtype=np.float32),
        "side_final_time": np.asarray(0, dtype=np.float32),
        "last_layer_hidden": last_hidden.astype(np.float16),
        "sr_last_hidden": last_hidden[:, -1].copy(),
        "layer_norms": np.full((BATCH, MAIN_STEPS, 3, HORIZON), np.sqrt(2), dtype=np.float32),
        "shift_first_mean": last_hidden[:, 0].copy(), "shift_last_mean": last_hidden[:, -1].copy(),
        "num_layers": np.asarray(3), "hidden_size": np.asarray(2), "num_steps": np.asarray(MAIN_STEPS),
        "action_horizon": np.asarray(HORIZON), "norm_tail_steps": np.asarray(5),
        "norm_tail_layers": np.asarray(3), "capture_branch": np.asarray("main"),
        "step_indices": np.arange(MAIN_STEPS), "layer_indices": np.arange(3),
        "hook_counts": np.ones((MAIN_STEPS, 3), dtype=np.int64),
        "submitted_mask": np.ones((BATCH, HORIZON), dtype=bool),
        "executed_mask": np.ones((BATCH, HORIZON), dtype=bool),
        "active": np.ones(BATCH, dtype=bool), "success_after": np.zeros(BATCH, dtype=bool),
        "action_slot_start": np.zeros(BATCH, dtype=int), "action_slot_end": np.full(BATCH, HORIZON, dtype=int),
    }
    for name, result in compute_scores(data).items():
        data[name], data[name + "_valid"] = result
    save_npz(out / "query_000.npz", **data)
    save_npz(out / "query_000_obs.npz", main_images=np.zeros((BATCH, 224, 224, 3), dtype=np.uint8),
             wrist_images=np.zeros((BATCH, 2, 224, 224, 3), dtype=np.uint8),
             states=np.zeros((BATCH, ACTION_DIM), dtype=np.float32))
    save(out / "query_000.json", {"schema": SCHEMA, "query": 0, "synthetic": True,
        "task_descriptions": ["synthetic validator fixture"] * BATCH,
        "extra_view_images_none": True,
        "invalid_score_counts": {name: int((~data[name + "_valid"]).sum()) for name in SIGNALS}})
    episodes = []
    for slot in range(BATCH):
        filename = f"episode_{slot:02d}.mp4"
        writer = cv2.VideoWriter(str(out / filename), cv2.VideoWriter_fourcc(*"mp4v"), 4, (224, 224))
        _require(writer.isOpened(), "CPU synthetic video writer failed")
        try:
            writer.write(np.zeros((224, 224, 3), dtype=np.uint8))
            writer.write(np.full((224, 224, 3), 64, dtype=np.uint8))
        finally:
            writer.release()
        episodes.append({"slot": slot, "actual": slot, "success": False,
                         "action_slots_submitted": HORIZON, "video": filename, "frames": 2})
    save(out / "episodes.json", episodes)
    initial = validate_batch(out)
    data["dv"][0, 0] += 1.0
    save_npz(out / "query_000.npz", **data)
    rejected = validate_batch(out, raise_on_failure=False)
    _require(not rejected["passed"] and any("cannot be recomputed" in item["message"]
             for item in rejected["errors"]), "validator failed to reject a corrupted DV score")
    save(out / "expected-rejection.json", rejected)
    data["dv"][0, 0] -= 1.0
    save_npz(out / "query_000.npz", **data)
    restored = validate_batch(out)
    result = {"synthetic": True, "initial_passed": initial["passed"],
              "corrupt_score_rejected": not rejected["passed"], "restored_passed": restored["passed"],
              "time": time.time(), "output": str(out)}
    save(out / "self-test.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_or_directory", nargs="?")
    parser.add_argument("--self-test", metavar="FRESH_OUTPUT_DIRECTORY")
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(run_self_test(arguments.self_test)), flush=True)
    elif arguments.config_or_directory:
        result = validate_batch(arguments.config_or_directory)
        print(json.dumps({"passed": result["passed"], "queries": len(result["queries"]),
                          "episodes": result["episodes"], "warnings": len(result["warnings"])}), flush=True)
    else:
        parser.error("supply a batch config/directory or --self-test FRESH_OUTPUT_DIRECTORY")
