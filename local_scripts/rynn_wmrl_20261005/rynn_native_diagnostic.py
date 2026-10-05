"""Four K8/B1 comparisons using the pinned official inference code.

Official video: aspect-preserving <=640 and the previous fixed 320x256.
Expert episodes 0/4: native HDF5 pixels, no channel permutation, <=640.
The official nested numeric forward and Analysis generate statements are
compiled directly from the hash-locked source, without modifying that file.
Semantic No is a result, not an execution failure. No production code changes.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np


OFFICIAL_SHA256 = "2590fd6c9597f47b3e82fadce453b295ef54d547ea4341e8a42d1b354f6c1670"
ROBOT = "A dual-arm ALOHA robot with two grippers manipulating objects on a tabletop."
CAMERA = "A fixed head RGB camera observing both arms and the tabletop."


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_result(path, result):
    temp = path.with_name("result.json.tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def official_blocks(path):
    """Take exact statements from fixed main(), not a second implementation."""
    source = path.read_bytes()
    if sha(source) != OFFICIAL_SHA256:
        raise RuntimeError("Official inference source differs from verified clean commit")
    tree = ast.parse(source, filename=str(path))
    main_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    numeric = [node for node in main_node.body if isinstance(node, ast.FunctionDef) and node.name == "run_batch"]
    def assigns(node, name):
        return any(isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name
                   for target in item.targets) for item in ast.walk(node))
    generation = [node for node in main_node.body if isinstance(node, ast.With) and assigns(node, "gen_out")]
    if len(numeric) != 1 or len(generation) != 1:
        raise RuntimeError("Pinned official inference blocks are not uniquely identified")
    return (compile(ast.Module(body=numeric, type_ignores=[]), str(path), "exec"),
            compile(ast.Module(body=generation, type_ignores=[]), str(path), "exec"))


def build_cases(official, case_manifest, controls, native_arrays, native_metadata, previous_arrays):
    from PIL import Image
    source = case_manifest["official_example"]
    if not source or not source.get("path"):
        raise ValueError("Official example path is required")
    video = Path(source["path"])
    frames = official.load_video_frames(str(video))
    indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
    selected = [frames[int(index)] for index in indices]
    low_array = previous_arrays["official_k8"]
    if low_array.shape != (8, 256, 320, 3) or low_array.dtype != np.uint8:
        raise ValueError("Previous official_k8 pixels must match the completed diagnostic")
    low = [Image.fromarray(frame) for frame in low_array]
    common = dict(source=str(video), source_total_frames=len(frames), frame_indices=indices.tolist(),
        source_sha256=sha_file(video),
        source_frame_size_wh=list(frames[0].size), instruction=source["instruction"],
        robot_description="a Franka single-arm robot", camera_description="the main right camera",
        expected_reference=None, pixel_contract="official imageio RGB decode")
    cases = [dict(common, id="official_native_k8", images=official.resize_frames(selected, 640),
                  resize="official aspect-preserving max_side640"),
             dict(common, id="official_320_k8", images=low,
                  resize="exact previous diagnostic official_k8 320x256 pixels",
                  pixel_contract="frozen diagnostic-v1 cv2 BGR-to-RGB decode and INTER_AREA resize")]
    for episode, item_index in ((0, 1), (4, 9)):
        item = controls["items"][item_index]
        if item["episode_uid"] != f"sanity/episode{episode}/expert_episode":
            raise ValueError("Expert reference metadata no longer matches the fixed episode order")
        path = Path(item["source"])
        case = next(case for case in case_manifest["cases"] if case["id"] == f"expert{episode}_k8")
        key = f"expert{episode}"
        metadata, array = native_metadata[key], native_arrays[key]
        if (metadata["source"] != str(path) or metadata["episode_uid"] != item["episode_uid"]
                or array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != 8 or array.shape[-1] != 3
                or sha(array.tobytes()) != metadata["array_sha256"]
                or [int(array.shape[2]), int(array.shape[1])] != metadata["source_frame_size_wh"]):
            raise ValueError("Pre-exported native pixels disagree with reference provenance")
        total = int(metadata["source_total_frames"])
        sampled = np.linspace(0, total - 1, 8, dtype=int)
        if sampled.tolist() != metadata["frame_indices"]:
            raise ValueError("Pre-exported native frame selection is not the official uniform prefix")
        images = [Image.fromarray(frame) for frame in array]
        cases.append(dict(id=f"expert{episode}_native_k8", source=str(path), source_total_frames=total,
            source_sha256=sha_file(path),
            frame_indices=sampled.tolist(), source_frame_size_wh=list(images[0].size),
            images=official.resize_frames(images, 640), instruction=case["instruction"],
            robot_description=ROBOT, camera_description=CAMERA, expected_reference=True,
            resize="official aspect-preserving max_side640",
            pixel_contract="same cv2 decode without channel permutation as existing native reset contract"))
    return cases


def prepare_case(runtime, case):
    prepared = runtime.processor.process_episode(instruction=case["instruction"], images=case["images"],
        robot_description=case["robot_description"], camera_description=case["camera_description"])
    ids = prepared["input_ids"]
    grid = prepared["image_grid_thw"].flatten(0, 1)
    pixels = prepared["pixel_values"].flatten(0, 1)
    count = len(case["images"])
    if grid.shape != (count, 3) or int(grid.prod(dim=-1).sum()) != pixels.shape[0]:
        raise RuntimeError("Native processor grid/patch count mismatch")
    expected_tokens = int(grid.prod(dim=-1).sum()) // runtime.model.config.vision_config.spatial_merge_size ** 2
    if int((ids == runtime.model.config.image_token_id).sum()) != expected_tokens:
        raise RuntimeError("Native processor image-token count mismatch")
    value_tokens = int((ids == runtime.model.config.value_token_id).sum())
    relative_tokens = int((ids == runtime.model.config.relative_value_token_id).sum())
    if (value_tokens != count * runtime.processor.value_token_repeat
            or relative_tokens != (count - 1) * runtime.processor.relative_value_token_repeat):
        raise RuntimeError("Native processor prediction-slot token count mismatch")
    stats = dict(input_frame_sizes_wh=[list(image.size) for image in case["images"]],
        input_frame_sha256=[sha(np.asarray(image).tobytes()) for image in case["images"]],
        processed_pixels_sha256=sha(pixels.detach().cpu().numpy().tobytes()),
        input_tokens=int(ids.shape[1]), image_grid_thw=grid.tolist(), image_tokens=expected_tokens,
        value_tokens=value_tokens, relative_value_tokens=relative_tokens,
        pixel_values_shape=list(prepared["pixel_values"].shape))
    return prepared, stats


def run_case(runtime, official, blocks, case):
    torch = runtime.torch
    prepared, stats = prepare_case(runtime, case)
    ids = prepared["input_ids"]
    namespace = dict(official.__dict__)
    namespace.update(model=runtime.model, device=runtime.device, torch=torch,
        args=SimpleNamespace(max_new_tokens=128), final_sample=prepared,
        input_ids=ids.to(runtime.device).long(), eos_token_id=runtime.eos_token_id)
    exec(blocks[0], namespace)
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.monotonic()
    value = namespace["run_batch"]([prepared])
    torch.cuda.synchronize(runtime.device)
    forward_seconds = time.monotonic() - started
    gen_started = time.monotonic()
    exec(blocks[1], namespace)
    torch.cuda.synchronize(runtime.device)
    generate_seconds = time.monotonic() - gen_started
    tokens = namespace["gen_out"][0, ids.shape[1]:].detach().cpu().tolist()
    text = runtime.processor.tokenizer.decode(tokens, skip_special_tokens=True)
    parsed = official.parse_analysis(text)
    result = {key: val for key, val in case.items() if key != "images"}
    result.update(status="completed", **stats,
        last_remaining_time_official=value[0], official_analysis=parsed, raw_text=text,
        success={"Yes": True, "No": False}.get(parsed["success"]),
        match={"Yes": True, "No": False}.get(parsed["match"]),
        generated_tokens=len(tokens), max_new_tokens=128,
        eos_seen=runtime.eos_token_id in tokens, numeric_forward_s=forward_seconds,
        generate_s=generate_seconds, elapsed_s=time.monotonic() - started, memory=runtime._memory())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-module", required=True, type=Path)
    parser.add_argument("--official-inference", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(4,))
    parser.add_argument("--cases-json", required=True, type=Path)
    parser.add_argument("--controls-json", required=True, type=Path)
    parser.add_argument("--native-frames-npz", required=True, type=Path)
    parser.add_argument("--native-frames-json", required=True, type=Path)
    parser.add_argument("--samples-npz", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    if args.cpu_preflight and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES empty")
    if not args.cpu_preflight and (os.environ.get("CUDA_VISIBLE_DEVICES") != "4" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
        raise RuntimeError("Owner must bind this process exclusively to physical GPU4")
    if args.output.name != "result.json" or args.output.exists():
        raise ValueError("A fresh result.json output path is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=1, pid=os.getpid(), physical_gpu=4, engineering_passed=False,
        mode="cpu_preflight" if args.cpu_preflight else "native_diagnostic",
        quality_passed=None, quality_note="Four diagnostic comparisons; native expert references are not new physics labels.",
        started_at_unix=time.time(), cases=[], sources={
            "official_inference": str(args.official_inference), "official_sha256": OFFICIAL_SHA256,
            "service_module": str(args.service_module), "service_sha256": sha(args.service_module.read_bytes()),
            "diagnostic_sha256": sha(Path(__file__).read_bytes()),
            "cases_sha256": sha(args.cases_json.read_bytes()), "controls_sha256": sha(args.controls_json.read_bytes()),
            "native_frames_sha256": sha_file(args.native_frames_npz),
            "native_metadata_sha256": sha_file(args.native_frames_json),
            "previous_samples_sha256": sha_file(args.samples_npz)})
    save_result(args.output, result)
    runtime, failed = None, None
    try:
        blocks = official_blocks(args.official_inference)
        sys.path.insert(0, str(args.official_inference.parent))
        official = load_module("pinned_rynn_official_native", args.official_inference)
        service = load_module("pinned_rynn_runtime_native", args.service_module)
        with np.load(args.native_frames_npz, allow_pickle=False) as source:
            native_arrays = {key: source[key].copy() for key in ("expert0", "expert4")}
        with np.load(args.samples_npz, allow_pickle=False) as source:
            previous_arrays = {"official_k8": source["official_k8"].copy()}
        cases = build_cases(official, json.loads(args.cases_json.read_text(encoding="utf-8")),
            json.loads(args.controls_json.read_text(encoding="utf-8")), native_arrays,
            json.loads(args.native_frames_json.read_text(encoding="utf-8")), previous_arrays)
        if len(cases) != 4:
            raise RuntimeError("Expected exactly four native diagnostic cases")
        result["sources"]["official_plot_utils_sha256"] = sha_file(args.official_inference.parent / "plot_utils.py")
        if args.cpu_preflight:
            import torch
            from transformers import AutoConfig, AutoProcessor
            result["asset_fingerprint"] = service.validate_assets(args.model_path, args.manifest_path)
            cpu_runtime = SimpleNamespace(model=SimpleNamespace(config=AutoConfig.from_pretrained(
                args.model_path, trust_remote_code=True, local_files_only=True)),
                processor=AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True))
            for case in cases:
                _, stats = prepare_case(cpu_runtime, case)
                record = {key: val for key, val in case.items() if key != "images"}
                record.update(status="prepared", **stats)
                result["cases"].append(record)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            result["model_loaded"] = False
            result["cuda_initialized"] = bool(torch.cuda.is_initialized())
            if result["cuda_initialized"]:
                raise RuntimeError("CPU preflight unexpectedly initialized CUDA")
            result["engineering_passed"] = True
            return
        runtime = service.RynnSuccessService(args.model_path, args.manifest_path, 4, batch_size=1,
            robot_description=ROBOT, camera_description=CAMERA)
        result["fingerprint"] = runtime.fingerprint
        runtime.onload()
        for case in cases:
            record = run_case(runtime, official, blocks, case)
            result["cases"].append(record)
            save_result(args.output, result)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        result["engineering_passed"] = True
    except BaseException as exc:
        failed = exc
        result["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if runtime is not None:
            try:
                result["offload"] = runtime.offload()
                if not result["offload"]["ok"] or not result["offload"]["is_offloaded"]:
                    raise RuntimeError("Native diagnostic did not acknowledge synchronized CPU offload")
            except BaseException as exc:
                failed = failed or exc
                result["offload_error"] = f"{type(exc).__name__}: {exc}"
        result["engineering_passed"] = bool(result["engineering_passed"] and failed is None)
        result["finished_at_unix"] = time.time()
        result["elapsed_s"] = result["finished_at_unix"] - result["started_at_unix"]
        save_result(args.output, result)
    print(json.dumps(dict(engineering_passed=result["engineering_passed"], quality_passed=None,
                         result=str(args.output))), flush=True)
    if failed is not None:
        raise RuntimeError("Native diagnostic failed; see result.json") from failed


if __name__ == "__main__":
    main()
