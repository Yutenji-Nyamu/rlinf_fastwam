"""One-process, GPU4-only Rynn diagnosis; semantic No is a recorded result.

cases.json: {"cases":[{"id":str,"frames_key":str,"instruction":str,
 "description":str,"expected_reference":null|bool,
 "meta_mode":"current"|"official_default"|"official_example"}]}
samples.npz: each frames_key contains uint8[K,256,320,3], K=8 or64.
The fixed production processor/model source is imported from --service-module.
K64 runs only B1. No inference prompt or checkpoint is modified to obtain Yes.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np


ROBOT = "A dual-arm ALOHA robot with two grippers manipulating objects on a tabletop."
CAMERA = "A fixed head RGB camera observing both arms and the tabletop."
OFFICIAL_ROBOT = "a Franka single-arm robot"
OFFICIAL_CAMERA = "the main right camera"
GIB = 1024 ** 3


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def save_result(output, result):
    temp = output / "result.json.tmp"
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, output / "result.json")


def read_service(path):
    spec = importlib.util.spec_from_file_location("fixed_rynn_diagnostic_runtime", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def add_official_metadata_cases(cases):
    """Reuse the official pixels/instruction with README's exact metadata."""
    output = [dict(case) for case in cases]
    ids = {case["id"] for case in output}
    for case in cases:
        if case["id"] not in ("official_k8", "official_k64"):
            continue
        comparison_id = case["id"] + "_readme_meta"
        if comparison_id in ids:
            raise ValueError("Official metadata comparison ID already exists")
        output.append(dict(case, id=comparison_id, meta_mode="official_example",
            parent_case_id=case["id"],
            description=case.get("description", "") + " [README exact robot/camera metadata; same instruction/pixels]"))
        ids.add(comparison_id)
    return output


def validate_cases(cases, arrays):
    ids = set()
    for case in cases:
        if (not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids
                or not isinstance(case.get("instruction"), str) or not case["instruction"].strip()):
            raise ValueError("Cases require unique IDs and a nonempty instruction")
        ids.add(case["id"])
        if case.get("meta_mode", "current") not in ("current", "official_default", "official_example"):
            raise ValueError("Unsupported input meta_mode")
        expected = case.get("expected_reference")
        if expected is not None and not isinstance(expected, bool):
            raise ValueError("expected_reference must be null or bool")
        frames = arrays[case["frames_key"]]
        if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1:] != (256, 320, 3) or len(frames) not in (8, 64):
            raise ValueError("Each case must contain uint8[K,256,320,3], K=8 or64")


def prepare_case(runtime, case, frames, robot, camera):
    from PIL import Image
    mode = case.get("meta_mode", "current")
    if mode == "official_default":
        robot, camera = None, None
    elif mode == "official_example":
        robot, camera = OFFICIAL_ROBOT, OFFICIAL_CAMERA
    elif mode == "generic":
        robot = "A robot arm manipulating objects."
        camera = "A fixed external RGB camera observing the manipulation."
    started = time.monotonic()
    prepared = runtime.processor.process_episode(instruction=case["instruction"],
        images=[Image.fromarray(frame) for frame in frames],
        robot_description=robot, camera_description=camera)
    ids = prepared["input_ids"][0].tolist()
    grid = prepared["image_grid_thw"].flatten(0, 1).cpu().numpy()
    pixels = prepared["pixel_values"].flatten(0, 1).detach().cpu().numpy()
    if grid.shape != (len(frames), 3):
        raise RuntimeError("Processor did not preserve one grid entry for every frame")
    if sum(int(np.prod(row)) for row in grid) != len(pixels):
        raise RuntimeError("Processor pixel patch count disagrees with frame grids")
    merge = int(runtime.model.config.vision_config.spatial_merge_size)
    expected_images = sum(int(np.prod(row)) // (merge * merge) for row in grid)
    image_positions = [i for i, token in enumerate(ids) if token == runtime.model.config.image_token_id]
    if len(image_positions) != expected_images:
        raise RuntimeError("Image-token count does not account for every input frame")
    if ids.count(runtime.model.config.value_token_id) != 8 * len(frames):
        raise RuntimeError("Unexpected absolute prediction-slot token count")
    if ids.count(runtime.model.config.relative_value_token_id) != 8 * (len(frames) - 1):
        raise RuntimeError("Unexpected relative prediction-slot token count")
    pixel_hashes = []
    cursor = 0
    for row in grid:
        count = int(np.prod(row))
        pixel_hashes.append(sha_bytes(pixels[cursor:cursor + count].tobytes()))
        cursor += count
    last_image_tokens = int(np.prod(grid[-1])) // (merge * merge)
    frame_hashes = [sha_bytes(frame.tobytes()) for frame in frames]
    stats = {"frames": len(frames), "frame_shape": list(frames.shape[1:]),
        "raw_frame_sha256": frame_hashes, "distinct_input_frames": len(set(frame_hashes)),
        "processed_frame_patch_sha256": pixel_hashes,
        "distinct_processed_frames": len(set(pixel_hashes)), "image_grid_thw": grid.tolist(),
        "pixel_values_shape": list(prepared["pixel_values"].shape),
        "input_tokens": len(ids), "image_tokens": len(image_positions),
        "last_image_token_span": [image_positions[-last_image_tokens], image_positions[-1]],
        "value_tokens": ids.count(runtime.model.config.value_token_id),
        "relative_value_tokens": ids.count(runtime.model.config.relative_value_token_id),
        "prompt_suffix": runtime.processor.tokenizer.decode(ids[-180:], skip_special_tokens=False),
        "robot_description": robot, "camera_description": camera,
        "prepare_s": time.monotonic() - started}
    return prepared, stats


def cpu_preflight(args, module, cases, arrays, output, result):
    """Exercise the real processor/config only; never load weights or CUDA."""
    from types import SimpleNamespace
    import torch
    from transformers import AutoConfig, AutoProcessor
    result["mode"] = "cpu_preflight"
    result["model_loaded"] = False
    result["asset_fingerprint"] = module.validate_assets(args.model_path, args.manifest_path)
    runtime = SimpleNamespace(
        model=SimpleNamespace(config=AutoConfig.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True)),
        processor=AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True, local_files_only=True))
    for case in cases:
        try:
            _, stats = prepare_case(runtime, case, arrays[case["frames_key"]],
                                    args.robot_description, args.camera_description)
            result["cases"].append({"case": case, "status": "prepared", "preprocessing": stats})
            print(json.dumps({"case": case["id"], "status": "prepared",
                "frames": stats["frames"], "input_tokens": stats["input_tokens"],
                "pixel_values_shape": stats["pixel_values_shape"],
                "grid_entries": len(stats["image_grid_thw"]), "image_tokens": stats["image_tokens"],
                "value_tokens": stats["value_tokens"], "relative_value_tokens": stats["relative_value_tokens"],
                "last_image_token_span": stats["last_image_token_span"],
                "distinct_input_frames": stats["distinct_input_frames"],
                "distinct_processed_frames": stats["distinct_processed_frames"]}), flush=True)
        except ValueError as exc:
            if case.get("meta_mode") != "official_default" or "use_meta=True requires" not in str(exc):
                raise
            result["cases"].append({"case": case, "status": "expected_input_error", "error": str(exc)})
            print(json.dumps({"case": case["id"], "status": "expected_input_error", "error": str(exc)}), flush=True)
    result["cuda_initialized"] = bool(torch.cuda.is_initialized())
    if result["cuda_initialized"]:
        raise RuntimeError("CPU processor preflight unexpectedly initialized CUDA")
    result["engineering_passed"] = True
    result["finished_at_unix"] = time.time()
    save_result(output, result)
    print(json.dumps({"engineering_passed": True, "mode": "cpu_preflight",
                      "cases": len(result["cases"]), "cuda_initialized": False,
                      "result": str(output / "result.json")}), flush=True)


def run_generation(runtime, samples):
    rows, diagnostics = runtime._generate_batch(samples, 128)
    attempts = [[dict(row)] for row in rows]
    all_diags = [diagnostics]
    retry = [i for i, row in enumerate(rows) if row["parse_status"] in ("missing_success", "truncated")]
    if retry:
        retried, diag = runtime._generate_batch([samples[i] for i in retry], 256)
        diag["retry_rows"] = retry
        all_diags.append(diag)
        for index, row in zip(retry, retried):
            attempts[index].append(dict(row))
            rows[index] = row
    for row, history in zip(rows, attempts):
        row["attempts"] = history
    return rows, all_diags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-module", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(4,))
    parser.add_argument("--cases-json", required=True, type=Path)
    parser.add_argument("--samples-npz", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--output", type=Path, help="Exact result.json path (parent holds diagnostic output)")
    parser.add_argument("--robot-description", default=ROBOT)
    parser.add_argument("--camera-description", default=CAMERA)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--cpu-preflight", action="store_true", help="Validate real processor inputs without model weights or CUDA")
    args = parser.parse_args()
    if not args.cpu_preflight and (os.environ.get("CUDA_VISIBLE_DEVICES") != "4" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
        raise RuntimeError("Diagnostic requires owner-set physical GPU4 and PCI_BUS_ID order")
    if args.output is not None and args.output.name != "result.json":
        raise ValueError("--output must name result.json")
    output = args.output.parent if args.output is not None else args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise RuntimeError("Diagnostic output already exists; use a fresh owner output directory")
    cases_data = json.loads(args.cases_json.read_text(encoding="utf-8"))
    cases = add_official_metadata_cases(cases_data["cases"])
    with np.load(args.samples_npz, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    validate_cases(cases, arrays)
    module = read_service(args.service_module)
    result = {"schema_version": 1, "pid": os.getpid(), "physical_gpu": 4,
        "engineering_passed": False, "quality_passed": None,
        "quality_note": "Diagnostic comparisons; No does not fail execution and no reward-quality gate is relaxed.",
        "started_at_unix": time.time(), "cases": [], "throughput": [],
        "sources": {"service_module": str(args.service_module),
            "service_sha256": sha_bytes(args.service_module.read_bytes()),
            "diagnostic_sha256": sha_bytes(Path(__file__).read_bytes()),
            "cases_json": str(args.cases_json), "cases_sha256": sha_bytes(args.cases_json.read_bytes()),
            "samples_npz": str(args.samples_npz), "samples_sha256": sha_bytes(args.samples_npz.read_bytes())}}
    runtime, failed = None, None
    started = time.monotonic()
    save_result(output, result)
    if args.cpu_preflight:
        try:
            cpu_preflight(args, module, cases, arrays, output, result)
        except BaseException as exc:
            result["error"] = {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()}
            save_result(output, result)
            raise
        return
    try:
        runtime = module.RynnSuccessService(args.model_path, args.manifest_path, 4, batch_size=16,
            robot_description=args.robot_description, camera_description=args.camera_description)
        result["fingerprint"] = runtime.fingerprint
        prepared_by_id = {}
        runtime.onload()
        for case in cases:
            case_result = {"case": case, "status": "pending"}
            try:
                prepared, stats = prepare_case(runtime, case, arrays[case["frames_key"]],
                                               args.robot_description, args.camera_description)
            except ValueError as exc:
                if case.get("meta_mode") != "official_default" or "use_meta=True requires" not in str(exc):
                    raise
                case_result.update(status="input_error", error=str(exc))
                result["cases"].append(case_result)
                # A separately named comparison, never a silent replacement of the requested case.
                fallback = dict(case, id=case["id"] + "_generic_meta", meta_mode="generic",
                    parent_case_id=case["id"],
                    description=case.get("description", "") + " [separate generic-metadata comparison]")
                cases.append(fallback)
                save_result(output, result)
                print(json.dumps({"case": case["id"], "status": "input_error", "error": str(exc)}), flush=True)
                continue
            rows, diagnostics = run_generation(runtime, [prepared])
            case_result.update(status="completed", preprocessing=stats,
                               output=rows[0], batches=diagnostics)
            prepared_by_id[case["id"]] = prepared
            result["cases"].append(case_result)
            save_result(output, result)
            print(json.dumps({"case": case["id"], "frames": stats["frames"],
                "success": rows[0]["success"], "match": rows[0]["match"],
                "parse_status": rows[0]["parse_status"], "raw_text": rows[0]["raw_text"],
                "input_tokens": stats["input_tokens"],
                "peak_allocated_gib": max(x["peak_allocated_bytes"] for x in diagnostics) / GIB}), flush=True)

        if not args.skip_throughput:
            pair = ("initial0_k8", "expert0_k8")
            if not all(key in prepared_by_id for key in pair):
                result["throughput_skipped"] = "Required initial0_k8/expert0_k8 case IDs missing"
            elif len({prepared_by_id[key]["input_ids"].shape[1] for key in pair}) != 1:
                result["throughput_skipped"] = "Pair has different prompt lengths; no padding change in this diagnostic"
            else:
                runtime.torch.cuda.synchronize(runtime.device)
                runtime.torch.cuda.empty_cache()
                repeated = [prepared_by_id[key] for key in pair] * 8
                previous = None
                for batch_size in (4, 8, 16):
                    if previous is not None:
                        projected = 17.84 + 2 * max(0.0, previous["peak_allocated_gib"] - 17.84) + 4
                        if previous["peak_reserved_gib"] > 50 or projected > 50:
                            result["throughput_stopped"] = {"before_batch": batch_size,
                                "reason": "Projected footprint exceeds the future policy-coexistence budget",
                                "projected_gib": projected}
                            break
                    began = time.monotonic()
                    outputs, diagnostics = [], []
                    for offset in range(0, 16, batch_size):
                        rows, diags = run_generation(runtime, repeated[offset:offset + batch_size])
                        outputs.extend(rows)
                        diagnostics.extend(diags)
                    elapsed = time.monotonic() - began
                    previous = {"batch_size": batch_size, "samples": 16,
                        "repeated_case_ids": list(pair), "elapsed_s": elapsed,
                        "clips_per_s": 16 / elapsed, "actual_batch_sizes": [x["size"] for x in diagnostics],
                        "peak_allocated_gib": max(x["peak_allocated_bytes"] for x in diagnostics) / GIB,
                        "peak_reserved_gib": max(x["peak_reserved_bytes"] for x in diagnostics) / GIB,
                        "unknown": sum(x["success"] is None for x in outputs),
                        "outputs": outputs, "batches": diagnostics,
                        "quality_note": "Repeated inputs benchmark throughput only; no success accuracy claim."}
                    result["throughput"].append(previous)
                    save_result(output, result)
                    print(json.dumps({key: value for key, value in previous.items()
                                      if key not in ("outputs", "batches")}), flush=True)
        result["engineering_passed"] = True
    except BaseException as exc:
        failed = exc
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
    finally:
        if runtime is not None:
            try:
                result["offload"] = runtime.offload()
                if not result["offload"]["ok"] or not result["offload"]["is_offloaded"]:
                    raise RuntimeError("Rynn did not acknowledge synchronized CPU offload")
            except BaseException as exc:
                failed = failed or exc
                result["offload_error"] = f"{type(exc).__name__}: {exc}"
        result["engineering_passed"] = bool(result["engineering_passed"] and failed is None)
        result["elapsed_s"] = time.monotonic() - started
        result["finished_at_unix"] = time.time()
        save_result(output, result)
    print(json.dumps({"engineering_passed": result["engineering_passed"],
                      "quality_passed": None, "result": str(output / "result.json")}), flush=True)
    if failed is not None:
        raise RuntimeError("Diagnostic execution failed; see result.json") from failed


if __name__ == "__main__":
    main()
