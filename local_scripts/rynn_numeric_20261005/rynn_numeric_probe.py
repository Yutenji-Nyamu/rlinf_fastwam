"""Offline K8 numeric-head probe. No policy, optimizer, Success generation or HTTP.

Input: a JSON manifest of per-case metadata and a non-pickle NPZ of uint8
[8,256,320,3] head-camera clips. CPU export is deliberately a separate process.
The fixed official run_batch AST owns the forward and last-slot extraction;
a transparent model wrapper records absolute/relative slots for diagnostics.
"""

import argparse
import ast
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
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
KINDS = {"expert_prefix", "repeat_initial", "repeat_final", "reversed", "return_to_initial", "native_eval"}


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


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def official_numeric(path):
    source = path.read_bytes()
    if sha(source) != OFFICIAL_SHA256:
        raise RuntimeError("Official inference source differs from the pinned revision")
    tree = ast.parse(source, filename=str(path))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    blocks = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "run_batch"]
    if len(blocks) != 1:
        raise RuntimeError("Pinned official numeric function is not unique")
    return compile(ast.Module(body=blocks, type_ignores=[]), str(path), "exec")


def load_cases(manifest_path, arrays_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if manifest.get("schema_version") != 1 or not 1 <= len(cases) <= 512:
        raise ValueError("Expected schema_version 1 and between 1 and 512 cases")
    with np.load(arrays_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    seen = set()
    for case in cases:
        required = {"id", "frames_key", "episode_uid", "kind", "instruction", "source", "label_origin", "array_sha256"}
        if required - set(case) or case["kind"] not in KINDS or case["id"] in seen:
            raise ValueError("Invalid or duplicate case metadata")
        seen.add(case["id"])
        if not all(isinstance(case[k], str) and case[k].strip() for k in required):
            raise ValueError("Required case strings must be nonempty")
        frames = arrays[case["frames_key"]]
        if frames.shape != (8, 256, 320, 3) or frames.dtype != np.uint8:
            raise ValueError("Expected exactly K8, uint8, 256x320 RGB-contract pixels")
        if sha(frames.tobytes()) != case["array_sha256"]:
            raise ValueError("Frame array hash mismatch")
        indices = case.get("frame_indices")
        if not isinstance(indices, list) or len(indices) != 8 or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Eight nonnegative source-frame indices are required")
        fraction = case.get("prefix_fraction")
        if case["kind"] == "expert_prefix" and (not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1):
            raise ValueError("Expert prefixes require a fraction in [0,1]")
        if case.get("reference_success") not in (True, False, None):
            raise ValueError("reference_success must be a nullable boolean")
        if case["kind"] in ("reversed", "return_to_initial") and case.get("reference_success") is not None:
            raise ValueError("Synthetic temporal controls must not carry physical success/failure labels")
    return cases, arrays


def prepare(runtime, case, frames):
    from PIL import Image
    sample = runtime.processor.process_episode(instruction=case["instruction"],
        images=[Image.fromarray(frame) for frame in frames],
        robot_description=case.get("robot_description", ROBOT),
        camera_description=case.get("camera_description", CAMERA))
    ids = sample["input_ids"]
    grid = sample["image_grid_thw"].flatten(0, 1)
    pixels = sample["pixel_values"].flatten(0, 1)
    config = runtime.model.config
    if tuple(grid.shape) != (8, 3) or int(grid.prod(dim=-1).sum()) != pixels.shape[0]:
        raise RuntimeError("Processor image grid/patch count mismatch")
    image_tokens = int(grid.prod(dim=-1).sum()) // config.vision_config.spatial_merge_size ** 2
    if int((ids == config.image_token_id).sum()) != image_tokens:
        raise RuntimeError("Processor image-token count mismatch")
    if int((ids == config.value_token_id).sum()) != 8 * runtime.processor.value_token_repeat:
        raise RuntimeError("Processor absolute-slot count mismatch")
    if int((ids == config.relative_value_token_id).sum()) != 7 * runtime.processor.relative_value_token_repeat:
        raise RuntimeError("Processor relative-slot count mismatch")
    stats = dict(input_tokens=int(ids.shape[1]), image_grid_thw=grid.tolist(),
        distinct_input_frames=len({sha(frame.tobytes()) for frame in frames}),
        processed_pixels_sha256=sha(pixels.detach().cpu().numpy().tobytes()))
    return sample, stats


class CaptureModel:
    """Keep official forward kwargs unchanged and capture detached head outputs."""
    def __init__(self, model):
        self.model = model
        self.captured = None

    def __call__(self, **kwargs):
        result = self.model(**kwargs)
        count = int(kwargs["input_ids"].shape[0])
        absolute = result.value.pred_value.detach().float().cpu()
        heads = int(self.model.config.num_value_heads)
        if absolute.numel() != heads * count * 8:
            raise RuntimeError("Unexpected absolute head dimensions")
        # Source _gather_by_token_id flattens row-major B*K; heads precede it.
        absolute = absolute.reshape(heads, count, 8).mean(dim=0)
        relative = getattr(getattr(result, "relative", None), "pred_value", None)
        if relative is not None:
            relative = relative.detach().float().cpu()
            if relative.numel() != count * 7:
                raise RuntimeError("Unexpected relative head dimensions")
            relative = relative.reshape(count, 7)
        self.captured = {"absolute_slots": absolute.tolist(),
            "relative_slots": None if relative is None else relative.tolist(),
            "absolute_source_shape": list(result.value.pred_value.shape)}
        return result


def forward(runtime, function, wrapper, samples):
    torch = runtime.torch
    torch.cuda.synchronize(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    start = time.monotonic()
    last = function(samples)
    torch.cuda.synchronize(runtime.device)
    seconds = time.monotonic() - start
    slots = wrapper.captured
    if len(last) != len(samples) or slots is None:
        raise RuntimeError("Official numeric output batch shape mismatch")
    rows = []
    for i, value in enumerate(last):
        if not math.isfinite(value) or not np.isfinite(slots["absolute_slots"][i]).all():
            raise RuntimeError("Non-finite absolute output")
        if abs(value - slots["absolute_slots"][i][-1]) > 1e-5 * max(1, abs(value)):
            raise RuntimeError("Official last-slot extraction differs from captured last slot")
        rel = None if slots["relative_slots"] is None else slots["relative_slots"][i]
        if rel is not None and not np.isfinite(rel).all():
            raise RuntimeError("Non-finite relative output")
        rows.append(dict(last_remaining_value=value, absolute_slots=slots["absolute_slots"][i],
                         relative_slots=rel, relative_last=None if rel is None else rel[-1]))
    return rows, dict(batch_size=len(samples), elapsed_s=seconds,
        clips_per_s=len(samples) / seconds, input_tokens=int(samples[0]["input_ids"].shape[1]),
        absolute_source_shape=slots["absolute_source_shape"], memory=runtime._memory())


def ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        output[order[start:end]] = (start + end - 1) / 2
        start = end
    return output


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    if len(rx) < 2 or np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["episode_uid"]].append(row)
    trends = []
    for uid, rows in grouped.items():
        prefixes = sorted((r for r in rows if r["kind"] == "expert_prefix"), key=lambda r: r["prefix_fraction"])
        if len(prefixes) < 2:
            continue
        values = [r["last_remaining_value"] for r in prefixes]
        by_kind = {r["kind"]: r for r in rows if r["kind"] != "expert_prefix"}
        item = dict(episode_uid=uid, fractions=[r["prefix_fraction"] for r in prefixes], values=values,
            rank_correlation_fraction_vs_remaining=spearman([r["prefix_fraction"] for r in prefixes], values),
            initial_minus_final=values[0] - values[-1],
            strictly_decreasing_adjacent=sum(b < a for a, b in zip(values, values[1:])), adjacent_pairs=len(values)-1,
            full_prefix_minus_repeated_final=None, reversed_minus_forward=None, return_minus_forward=None,
            return_minus_initial=None)
        if "repeat_final" in by_kind:
            item["full_prefix_minus_repeated_final"] = values[-1] - by_kind["repeat_final"]["last_remaining_value"]
        if "reversed" in by_kind:
            item["reversed_minus_forward"] = by_kind["reversed"]["last_remaining_value"] - values[-1]
        if "return_to_initial" in by_kind:
            item["return_minus_forward"] = by_kind["return_to_initial"]["last_remaining_value"] - values[-1]
            item["return_minus_initial"] = by_kind["return_to_initial"]["last_remaining_value"] - values[0]
        trends.append(item)
    labeled = [r for r in records if r["kind"] == "native_eval" and type(r.get("reference_success")) is bool
               and r.get("prefix_fraction", 1.0) == 1.0]
    good = [r["last_remaining_value"] for r in labeled if r["reference_success"]]
    bad = [r["last_remaining_value"] for r in labeled if not r["reference_success"]]
    auc = None
    if good and bad:
        auc = float(np.mean([float(g < b) + 0.5 * float(g == b) for g in good for b in bad]))
    negative_corr = [r["rank_correlation_fraction_vs_remaining"] for r in trends
                     if r["rank_correlation_fraction_vs_remaining"] is not None]
    return dict(expert_trends=trends, expert_episode_count=len(trends),
        endpoints_lower_count=sum(r["initial_minus_final"] > 0 for r in trends),
        median_initial_minus_final=None if not trends else float(np.median([r["initial_minus_final"] for r in trends])),
        median_spearman=None if not negative_corr else float(np.median(negative_corr)),
        native_eval=dict(success_count=len(good), failure_count=len(bad), auc_low_value_predicts_success=auc,
            success_values=good, failure_values=bad),
        caveat="Expert progress and synthetic temporal controls cannot establish physical failure detection; native labels require a simulator evaluation receipt. Raw value units are not calibrated seconds or success probabilities.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("service-module", "official-inference", "model-path", "manifest-path", "cases-json", "samples-npz", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(4,))
    parser.add_argument("--batch-size", type=int, choices=(4, 8, 16), default=16)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args()
    expected_cuda = "" if args.cpu_preflight else "4"
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_cuda:
        raise RuntimeError("CPU preflight must hide GPUs; live probe must bind physical GPU4")
    if not args.cpu_preflight and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Owner must set PCI_BUS_ID")
    if args.output.name != "result.json" or args.output.exists():
        raise ValueError("A fresh result.json path is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(schema_version=1, pid=os.getpid(), physical_gpu=4,
        mode="cpu_preflight" if args.cpu_preflight else "numeric_head_probe", engineering_passed=False,
        quality_passed=None, quality_note="Diagnostic only; not a gate authorizing a reward replacement or training.",
        started_at_unix=time.time(), cases=[], batches=[],
        sources={name: dict(path=str(getattr(args, name)), sha256=sha_file(getattr(args, name)))
                 for name in ("service_module", "official_inference", "manifest_path", "cases_json", "samples_npz")})
    result["sources"]["probe"] = dict(path=str(Path(__file__)), sha256=sha_file(__file__))
    save(args.output, result)
    runtime, failed = None, None
    try:
        block = official_numeric(args.official_inference)
        service = load_module("pinned_rynn_numeric_service", args.service_module)
        cases, arrays = load_cases(args.cases_json, args.samples_npz)
        if args.cpu_preflight:
            import torch
            from transformers import AutoConfig, AutoProcessor
            result["fingerprint"] = service.validate_assets(args.model_path, args.manifest_path)
            runtime_cpu = SimpleNamespace(model=SimpleNamespace(config=AutoConfig.from_pretrained(
                args.model_path, trust_remote_code=True, local_files_only=True)),
                processor=AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True))
        else:
            runtime = service.RynnSuccessService(str(args.model_path), str(args.manifest_path), 4,
                batch_size=args.batch_size, robot_description=ROBOT, camera_description=CAMERA)
            runtime_cpu = runtime
            result["fingerprint"] = runtime.fingerprint
        start = time.monotonic()
        prepared, statistics = [], []
        for case in cases:
            sample, stats = prepare(runtime_cpu, case, arrays[case["frames_key"]])
            prepared.append(sample)
            statistics.append(stats)
        result["prepare_s"] = time.monotonic() - start
        del arrays
        buckets = service.bucket_indices(prepared, args.batch_size)
        result["planned_actual_batch_sizes"] = [len(b) for b in buckets]
        if args.cpu_preflight:
            result["cases"] = [{**c, **s, "status": "prepared"} for c, s in zip(cases, statistics)]
            result["cuda_initialized"] = bool(torch.cuda.is_initialized())
            if result["cuda_initialized"]:
                raise RuntimeError("CPU preflight initialized CUDA")
            result["engineering_passed"] = True
            return
        runtime.onload()
        wrapper = CaptureModel(runtime.model)
        namespace = dict(model=wrapper, torch=runtime.torch, device=runtime.device)
        exec(block, namespace)
        function = namespace["run_batch"]
        # One warmup, then B1 baselines. Same source video/instruction gives exact-length B16.
        initial_index = next(i for i, c in enumerate(cases) if c["kind"] == "expert_prefix" and c["prefix_fraction"] == 0)
        end_index = next(i for i, c in enumerate(cases) if c["kind"] == "expert_prefix" and c["prefix_fraction"] == 1
                         and c["episode_uid"] == cases[initial_index]["episode_uid"])
        forward(runtime, function, wrapper, [prepared[initial_index]])
        singleton = {}
        for index in (initial_index, end_index):
            rows, timing = forward(runtime, function, wrapper, [prepared[index]])
            singleton[index] = rows[0]
            result["batches"].append({"phase": "singleton_reference", "case_ids": [cases[index]["id"]], **timing})
        parity_indices = [initial_index, end_index] * 8
        parity, timing = forward(runtime, function, wrapper, [prepared[i] for i in parity_indices])
        absolute_errors = [abs(r["last_remaining_value"] - singleton[i]["last_remaining_value"])
                           for i, r in zip(parity_indices, parity)]
        result["batch_parity"] = dict(batch_size=16, case_ids=[cases[i]["id"] for i in parity_indices],
            singleton_values=[singleton[i]["last_remaining_value"] for i in parity_indices],
            batch_values=[r["last_remaining_value"] for r in parity],
            max_absolute_difference=max(absolute_errors),
            max_relative_difference=max(e / max(1, abs(singleton[i]["last_remaining_value"]))
                                        for i, e in zip(parity_indices, absolute_errors)),
            same_sign_endpoint_delta=(singleton[initial_index]["last_remaining_value"] > singleton[end_index]["last_remaining_value"])
                == (parity[0]["last_remaining_value"] > parity[1]["last_remaining_value"]))
        result["batches"].append({"phase": "batch16_reference", **timing})
        ordered = [None] * len(cases)
        for indices in buckets:
            rows, timing = forward(runtime, function, wrapper, [prepared[i] for i in indices])
            result["batches"].append({"phase": "dataset", "case_ids": [cases[i]["id"] for i in indices], **timing})
            for index, row in zip(indices, rows):
                ordered[index] = {**cases[index], **statistics[index], **row, "status": "completed"}
            result["cases"] = [r for r in ordered if r is not None]
            save(args.output, result)
            print(json.dumps(dict(completed=len(result["cases"]), total=len(cases), **timing)), flush=True)
        result["summary"] = summarize(result["cases"])
        dataset_batches = [b for b in result["batches"] if b["phase"] == "dataset"]
        seconds = sum(b["elapsed_s"] for b in dataset_batches)
        result["throughput"] = dict(case_count=len(cases), numeric_forward_total_s=seconds,
            clips_per_s=len(cases)/seconds, actual_batch_sizes=[b["batch_size"] for b in dataset_batches],
            peak_allocated_bytes=max(b["memory"]["peak_allocated_bytes"] for b in result["batches"]),
            peak_reserved_bytes=max(b["memory"]["peak_reserved_bytes"] for b in result["batches"]),
            caveat="Forward includes CPU tensor transfer and output readback; excludes input preprocessing, model load, and WM generation.")
        result["engineering_passed"] = True
    except BaseException as exc:
        failed = exc
        result["error"] = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        if runtime is not None:
            try:
                result["offload"] = runtime.offload()
                if not result["offload"]["ok"] or not result["offload"]["is_offloaded"]:
                    raise RuntimeError("Numeric probe did not acknowledge synchronized CPU offload")
            except BaseException as exc:
                failed = failed or exc
                result["offload_error"] = f"{type(exc).__name__}: {exc}"
        result["engineering_passed"] = bool(result["engineering_passed"] and failed is None)
        result["finished_at_unix"] = time.time()
        result["elapsed_s"] = result["finished_at_unix"] - result["started_at_unix"]
        save(args.output, result)
    print(json.dumps(dict(engineering_passed=result["engineering_passed"], result=str(args.output))), flush=True)
    if failed is not None:
        raise RuntimeError("Numeric probe failed; inspect result.json") from failed


if __name__ == "__main__":
    main()
