"""Loopback OpenDW true-batch inference with unchanged per-environment protocol.

Example (the owner supplies the authorized single physical GPU):
  CUDA_VISIBLE_DEVICES=6 python -B opendw_service_batched.py \
    --bundle /path/DW05-Robotwin --reward-checkpoint /path/resnet_rm.pth \
    --t5-path /path/t5-base --physical-gpu 6 --port 18941 --output-dir /path/run \
    --execution-mode batched --wm-batch-size 16

No GPU model is resident until /onload or /infer. /offload replies only after
all DW05 and RM parameters/buffers are on CPU and CUDA work is synchronized.
The B1 reference is explicit; batch failures never trigger automatic B1 fallback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

import numpy as np
from PIL import Image
from opendw_action_telemetry import action_telemetry
from wm_batch import infer_joint_batch


OPENDW_COMMIT = "e33befa8005a1585e0140dbf464566e90bc79aa1"
WORLDARENA_COMMIT = "5978ce5c81e55b8c8358f4f5966a13ce385ff155"
MAX_BODY_BYTES = 128 * 1024 * 1024


def parse_gpu_identity(output, physical_gpu):
    """Read physical IDs from unmasked nvidia-smi inventory, without CUDA."""
    matches = []
    for line in output.splitlines():
        parts = [value.strip() for value in line.split(',')]
        if len(parts) == 3 and parts[0] == str(physical_gpu):
            matches.append(dict(physical_gpu=physical_gpu, gpu_uuid=parts[1], pci=parts[2]))
    if len(matches) != 1 or not matches[0]['gpu_uuid'].startswith('GPU-'):
        raise RuntimeError('nvidia-smi did not identify exactly the requested physical GPU')
    return matches[0]


def read_gpu_identity(physical_gpu):
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader,nounits'],
        text=True, timeout=15)
    return parse_gpu_identity(output, physical_gpu)


def verify_cuda_identity(expected):
    """Only on /onload, after owner borrowing: verify logical CUDA0 UUID/PCI."""
    import ctypes as C
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(expected['physical_gpu']):
        raise RuntimeError('Service CUDA visibility changed from its one physical GPU')
    if read_gpu_identity(expected['physical_gpu']) != expected:
        raise RuntimeError('Physical GPU identity changed since CPU service preparation')
    cuda = C.CDLL('libcuda.so.1')
    cuda.cuInit.argtypes = [C.c_uint]
    cuda.cuDeviceGetCount.argtypes = [C.POINTER(C.c_int)]
    cuda.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
    cuda.cuDeviceGetPCIBusId.argtypes = [C.c_char_p, C.c_int, C.c_int]
    get_uuid = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
    get_uuid.argtypes = [C.c_void_p, C.c_int]
    count, device, pci, uuid = C.c_int(), C.c_int(), C.create_string_buffer(32), (C.c_ubyte * 16)()
    if not (cuda.cuInit(0) == 0 and cuda.cuDeviceGetCount(C.byref(count)) == 0
            and count.value == 1 and cuda.cuDeviceGet(C.byref(device), 0) == 0
            and cuda.cuDeviceGetPCIBusId(pci, len(pci), device) == 0
            and get_uuid(C.byref(uuid), device) == 0):
        raise RuntimeError('Cannot verify the one visible CUDA device')
    raw = bytes(uuid).hex()
    actual_uuid = 'GPU-' + '-'.join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))
    actual_pci = pci.value.decode()
    if (actual_uuid.lower() != expected['gpu_uuid'].lower()
            or actual_pci.lower()[-10:] != expected['pci'].lower()[-10:]):
        raise RuntimeError(f'Logical CUDA0 differs from requested physical GPU: {actual_uuid}, {actual_pci}')
    return dict(**expected, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                logical_device_count=count.value, logical_cuda0_uuid=actual_uuid, logical_cuda0_pci=actual_pci)


def decode_request(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        required = {"images", "actions", "states", "seeds", "instructions"}
        if not required.issubset(source.files):
            raise ValueError(f"Request missing keys: {sorted(required - set(source.files))}")
        data = {key: source[key].copy() for key in source.files}
    images, actions, states = data["images"], data["actions"], data["states"]
    batch = len(images)
    if not 1 <= batch <= 256:
        raise ValueError("Batch must contain 1..256 logical environments")
    if images.shape != (batch, 384, 320, 3) or images.dtype != np.uint8:
        raise ValueError("images must be uint8[B,384,320,3]")
    if actions.shape != (batch, 32, 14) or states.shape != (batch, 14):
        raise ValueError("actions/states must be [B,32,14]/[B,14]")
    if actions.dtype != np.float32 or states.dtype != np.float32:
        raise ValueError("actions/states must be raw float32 values")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Nonfinite action/state")
    if np.any((actions[..., [6, 13]] < 0) | (actions[..., [6, 13]] > 1)):
        raise ValueError("Caller must clip gripper commands to [0,1]")
    seeds = data["seeds"]
    if seeds.shape != (batch,) or seeds.dtype.kind not in "iu" or np.any(seeds < 0):
        raise ValueError("seeds must be nonnegative integer[B]")
    instructions = data["instructions"]
    if instructions.shape != (batch,) or instructions.dtype.kind not in "US":
        raise ValueError("instructions must be Unicode or UTF8 strings[B]")
    return data


def encode_response(next_images, scores, timings=None, future_head_frames=None):
    result = io.BytesIO()
    values = {"next_images": np.asarray(next_images, dtype=np.uint8), "scores": np.asarray(scores, dtype=np.float32)}
    if timings is not None:
        values["timing_s"] = np.asarray(timings, dtype=np.float32)
    if future_head_frames is not None:
        values["future_head_frames"] = np.asarray(future_head_frames, dtype=np.uint8)
    np.savez(result, **values)
    return result.getvalue()


def row_identity(data, row):
    """Optional logging identity; missing/malformed metadata stays unknown."""
    result = {}
    for source, target in (("env_indices", "env_index"), ("reset_ids", "reset_id"),
                           ("global_env_indices", "global_env_index")):
        values = data.get(source)
        valid = (isinstance(values, np.ndarray) and
                 values.shape == (len(data["images"]),) and values.dtype.kind in "iu")
        result[target] = int(values[row]) if valid else None
    for key in ('env_process_index', 'env_process_count'):
        value = data.get(key)
        result[key] = (int(value) if isinstance(value, np.ndarray) and value.shape == ()
                       and value.dtype.kind in 'iu' else None)
    return result


def batch_ranges(size, batch_size):
    if not 1 <= size <= 256 or batch_size < 1:
        raise ValueError("Invalid logical/WM batch size")
    return [(start, min(start + batch_size, size)) for start in range(0, size, batch_size)]


def instructions_for(data, start, end):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in data["instructions"][start:end]]


def validate_videos(videos, batch):
    if len(videos) != batch:
        raise RuntimeError(f"WM returned {len(videos)} videos for batch {batch}")
    result = [[frame.convert("RGB") for frame in frames] for frames in videos]
    if any(len(frames) != 9 or any(frame.size != (320, 384) for frame in frames) for frames in result):
        raise RuntimeError("WM videos must preserve [B,9,384,320,3] order and dimensions")
    return result


def process_memory():
    result = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                result[key + "_bytes"] = int(value.split()[0]) * 1024
    except OSError:
        pass
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                result["Pss_bytes"] = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    return result


class ResourceLog:
    def __init__(self, output_dir, physical_gpu, interval=10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "service-events.jsonl"
        self.physical_gpu = physical_gpu
        self.interval = interval
        self.phase = "initializing"
        self.torch = None
        self._lock = threading.Lock()
        self._resource_lock = threading.Lock()
        self._resource_cache = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._monitor, name="resource-monitor", daemon=True)
        self._thread.start()

    def _cuda_snapshot(self):
        result = {}
        if self.torch is not None and self.torch.cuda.is_initialized():
            try:
                result.update(cuda_allocated_bytes=self.torch.cuda.memory_allocated(0),
                              cuda_reserved_bytes=self.torch.cuda.memory_reserved(0),
                              cuda_max_allocated_bytes=self.torch.cuda.max_memory_allocated(0),
                              cuda_max_reserved_bytes=self.torch.cuda.max_memory_reserved(0))
            except Exception as error:
                result["cuda_metrics_error"] = str(error)
        return result

    def _sample_resources(self):
        result = {"resource_timestamp_utc": datetime.now(timezone.utc).isoformat(), **process_memory()}
        try:
            query = subprocess.run(
                ["nvidia-smi", "-i", str(self.physical_gpu), "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            result["gpu_nvml"] = query.stdout.strip() if query.returncode == 0 else {"error": query.stderr.strip()}
            processes = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            result["self_gpu_processes"] = [line.strip() for line in processes.stdout.splitlines() if line.split(",", 1)[0].strip() == str(os.getpid())]
        except (OSError, subprocess.TimeoutExpired) as error:
            result["nvml_error"] = str(error)
        return result

    def snapshot(self, refresh_resources=False):
        # Only the background 10-second monitor (and the initial event) reads
        # smaps/nvidia-smi. Row evidence reuses this sample, with fresh cheap
        # CUDA allocator/peak counters, so B16 does not spawn 64 smi queries.
        with self._resource_lock:
            if refresh_resources or self._resource_cache is None:
                self._resource_cache = self._sample_resources()
            result = dict(self._resource_cache)
        result.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                      phase=self.phase, **self._cuda_snapshot())
        return result

    def event(self, event, refresh_resources=False, **fields):
        record = {**self.snapshot(refresh_resources), "event": event, **fields}
        encoded = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded + "\n")
            status_path = self.output_dir / "service-status.json"
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(encoded + "\n", encoding="utf-8")
            os.replace(temporary, status_path)
        print(encoded, flush=True)
        return record

    def _monitor(self):
        while not self._stop.wait(self.interval):
            self.event("heartbeat", refresh_resources=True)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=12)


def set_dw_device(model, device):
    """DW05 .to() moves weights but does not update its cached device field."""
    model.to(device)
    model.device = device
    for name in ("text_encoder", "vae"):
        component = getattr(model, name, None)
        # The official .to also moves these; set only an actual cached device,
        # never overwrite a read-only property inferred from parameters.
        if component is not None and "device" in vars(component):
            component.device = device
    return model


def remaining_cuda_tensors(*models):
    found = []
    seen = set()
    for prefix, model in enumerate(models):
        for name, value in list(model.named_parameters()) + list(model.named_buffers()):
            if id(value) in seen:
                continue
            seen.add(id(value))
            if value.device.type != "cpu":
                found.append(f"{prefix}:{name}:{value.device}")
    return found


def clear_dw_runtime_caches(model):
    # WanVideoVAE38 normally clears these on successful encode/decode. A raised
    # exception can leave feature tensors in ordinary Python lists; .to(cpu)
    # does not visit those lists, so explicitly clear the official cache API.
    vae_model = getattr(getattr(model, "vae", None), "model", None)
    clear_cache = getattr(vae_model, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


class Backend:
    def __init__(self, args, log):
        self.args, self.log = args, log
        self.gpu_identity = read_gpu_identity(args.physical_gpu)
        self.lock = threading.RLock()
        self.is_offloaded = True
        self.fatal_error = None
        self.requests_completed = 0
        self.rows_completed = 0
        self.samples_saved = 0
        self.batch_calls_completed = 0
        self.max_actual_wm_batch = 0
        log.phase = "loading_cpu"
        log.event("cpu_load_started")
        import torch
        from dexbotic.policy import dw05_policy
        from dexbotic.exp.dw05_exp import DIFFSYNTH_MODEL_BASE_PATH_ENV
        from opendw_reward import RoboTwinT5Reward

        self.torch = torch
        self.log.torch = torch
        torch.set_num_threads(args.cpu_threads)
        self.device = torch.device(args.device)
        source_root = Path(dw05_policy.__file__).resolve().parents[2]
        head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        if head != OPENDW_COMMIT:
            raise RuntimeError(f"Unexpected OpenDW source commit: {head}")
        self.pil_to_model_tensor = dw05_policy.pil_to_model_tensor
        # The official policy uses setdefault; this explicit service bundle
        # must override any unrelated inherited model-cache environment value.
        os.environ[DIFFSYNTH_MODEL_BASE_PATH_ENV] = str(args.bundle)
        config = dw05_policy.DW05RobotWinPolicyConfig(
            checkpoint_path=str(args.bundle / "model.pt"),
            norm_stats_path=str(args.bundle / "norm_stats.json"),
            model_base_path=str(args.bundle), device="cpu", mixed_precision="bf16",
            action_horizon=32, num_video_frames=9, num_inference_steps=10,
            normalization_mode="zscore_14d", action_condition_mode="absolute",
            image_layout="robotwin_resize", raw_state_dim=14, raw_action_dim=14,
            load_text_encoder=True, rand_device="cpu", tiled=False,
        )
        self.policy = dw05_policy.DW05RobotWinPolicy(config)
        if (self.policy.action_dim, self.policy.proprio_dim) != (14, 14):
            raise RuntimeError("This service requires the released Robotwin 14D bundle")
        self.policy.model.eval().requires_grad_(False)
        # load_checkpoint has completed; retaining its CPU state dictionary is
        # unnecessary for inference and can duplicate a full model in host RAM.
        self.policy.checkpoint_payload = None
        log.event("world_model_loaded_cpu", source_root=str(source_root), source_commit=head)
        self.reward = RoboTwinT5Reward(args.reward_checkpoint, args.t5_path)
        gc.collect()
        residual = remaining_cuda_tensors(self.policy.model, self.reward)
        if residual:
            raise RuntimeError(f"CPU initialization retained non-CPU tensors: {residual[:8]}")
        manifest = {
            "pid": os.getpid(), "opendw_commit": head, "worldarena_commit": WORLDARENA_COMMIT,
            "bundle": str(args.bundle), "reward_checkpoint": str(args.reward_checkpoint),
            "reward_checkpoint_bytes": args.reward_checkpoint.stat().st_size,
            "t5_path": str(args.t5_path), "norm_stats_sha256": hashlib.sha256((args.bundle / "norm_stats.json").read_bytes()).hexdigest(),
            "physical_gpu": args.physical_gpu, "device": str(self.device),
            "physical_gpu_identity": self.gpu_identity,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "action_horizon": 32, "num_video_frames": 9, "num_inference_steps": 10,
            "logical_batch_execution": args.execution_mode, "wm_batch_size": args.wm_batch_size,
            "batch_kernel_sha256": hashlib.sha256(Path(__file__).with_name("wm_batch.py").read_bytes()).hexdigest(),
            "automatic_batch_fallback": False, "per_row_seed": True,
            "history_protocol": "Current caller image/state per row; no cross-request shared history",
            "reward_batch": "actual WM batch times eight future head frames; same row-major order",
            "reward_strict_load": True,
            "reward_preprocess": "head RGB -> bilinear 224 -> ImageNet normalize",
        }
        (args.output_dir / "service-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.phase = "ready_offloaded"
        log.event("ready", is_offloaded=True)

    def health(self):
        return {"ok": self.fatal_error is None, "pid": os.getpid(), "phase": self.log.phase,
                "is_offloaded": self.is_offloaded, "fatal_error": self.fatal_error,
                "requests_completed": self.requests_completed, "rows_completed": self.rows_completed,
                "execution_mode": self.args.execution_mode, "wm_batch_size": self.args.wm_batch_size,
                "batch_calls_completed": self.batch_calls_completed, "max_actual_wm_batch": self.max_actual_wm_batch,
                "physical_gpu": self.args.physical_gpu, "physical_gpu_identity": self.gpu_identity}

    def onload(self):
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(f"Service requires owner recovery after: {self.fatal_error}")
            if not self.is_offloaded:
                return {"ok": True, "is_offloaded": False}
            self.log.phase = "onloading"
            self.log.event("onload_started")
            identity = verify_cuda_identity(self.gpu_identity)
            self.log.event('gpu_mapping_verified', **identity)
            set_dw_device(self.policy.model, self.device)
            self.policy.config.device = str(self.device)
            self.reward.to(self.device)
            self.torch.cuda.synchronize(self.device)
            self.is_offloaded = False
            self.torch.cuda.reset_peak_memory_stats(self.device)
            self.log.phase = "ready_loaded"
            self.log.event("onload_completed", is_offloaded=False)
            return {"ok": True, "is_offloaded": False}

    def offload(self):
        with self.lock:
            self.log.phase = "offloading"
            self.log.event("offload_started")
            if self.torch.cuda.is_initialized():
                self.torch.cuda.synchronize(self.device)
            clear_dw_runtime_caches(self.policy.model)
            set_dw_device(self.policy.model, self.torch.device("cpu"))
            self.policy.config.device = "cpu"
            self.reward.to("cpu")
            gc.collect()
            if self.torch.cuda.is_initialized():
                self.torch.cuda.synchronize(self.device)
                self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize(self.device)
            residual = remaining_cuda_tensors(self.policy.model, self.reward)
            if residual:
                raise RuntimeError(f"Offload left non-CPU parameters/buffers: {residual[:8]}")
            self.is_offloaded = True
            self.log.phase = "failed_offloaded" if self.fatal_error else "ready_offloaded"
            self.log.event("offload_completed", is_offloaded=True, residual_model_cuda_tensors=0)
            return {"ok": True, "is_offloaded": True}

    def _save_sample(self, frames, scores, data, row, instruction, seconds):
        if self.samples_saved >= self.args.evidence_samples:
            return
        destination = self.args.output_dir / "samples" / f"sample-{self.samples_saved:03d}"
        destination.mkdir(parents=True, exist_ok=False)
        frames[0].save(destination / "initial.png")
        frames[-1].save(destination / "predicted-final.png")
        Image.fromarray(data["images"][row]).save(destination / "input.png")
        frames[0].save(destination / "prediction.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)
        np.savez(destination / "inputs-and-scores.npz", actions=data["actions"][row], state=data["states"][row], scores=scores, seed=data["seeds"][row])
        metadata = {"instruction": instruction, "seed": int(data["seeds"][row]), "seconds": seconds,
                    "frame_action_indices": list(range(0, 33, 4)), "scores": scores.tolist(),
                    "request_id": str(data.get("request_id", "")), "row": row,
                    **row_identity(data, row)}
        (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.samples_saved += 1

    def _begin_phase(self, phase):
        self.log.phase = phase
        if self.torch.cuda.is_initialized():
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def _end_phase(self):
        if not self.torch.cuda.is_initialized():
            return {}
        self.torch.cuda.synchronize(self.device)
        return dict(cuda_max_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device),
                    cuda_max_reserved_bytes=self.torch.cuda.max_memory_reserved(self.device))

    def _batch_context(self, instructions):
        # Encode each distinct prompt once per WM batch. Proprio is deliberately
        # absent here: the batched kernel appends each row's own state token.
        encoded = {}
        for instruction in instructions:
            prompt = self.policy.format_prompt(instruction)
            if prompt not in encoded:
                encoded[prompt] = self.policy.model.encode_prompt(prompt)
        ordered = [encoded[self.policy.format_prompt(value)] for value in instructions]
        max_length = max(context.shape[1] for context, _ in ordered)
        contexts, masks = [], []
        for context, mask in ordered:
            if context.ndim != 3 or context.shape[0] != 1 or mask.shape != context.shape[:2]:
                raise RuntimeError("Prompt encoder must return [1,L,D] context and [1,L] mask")
            length = context.shape[1]
            if length < max_length:
                context = self.torch.cat((context, context.new_zeros((1, max_length - length, context.shape[2]))), dim=1)
                mask = self.torch.cat((mask, mask.new_zeros((1, max_length - length))), dim=1)
            contexts.append(context)
            masks.append(mask)
        return self.torch.cat(contexts, dim=0), self.torch.cat(masks, dim=0)

    def _predict_videos(self, data, start, end, instructions):
        images = self.torch.cat([
            self.pil_to_model_tensor(Image.fromarray(data["images"][row]), self.device, self.policy.model.torch_dtype)
            for row in range(start, end)], dim=0)
        actions = self.torch.stack([
            self.policy.normalize_action_condition(data["actions"][row], data["states"][row])
            for row in range(start, end)], dim=0)
        states = self.torch.cat([self.policy.normalize_state(data["states"][row]) for row in range(start, end)], dim=0)
        seeds = [int(value) for value in data["seeds"][start:end]]
        size = end - start
        if images.shape != (size, 3, 384, 320) or actions.shape != (size, 32, 14) or states.shape != (size, 14):
            raise RuntimeError("Normalized per-environment WM input shapes changed")
        common = dict(input_image=images, action=actions, proprio=states, num_video_frames=9,
                      action_horizon=32, num_inference_steps=10, sigma_shift=self.policy.config.sigma_shift,
                      rand_device="cpu", tiled=False)
        if self.args.execution_mode == "b1_reference":
            if size != 1:
                raise RuntimeError("B1 reference cannot accept a batched call")
            prediction = self.policy.model.infer_joint(prompt=self.policy.format_prompt(instructions[0]),
                seed=seeds[0], test_action_with_infer_action=False, **common)
            videos = [prediction["video"]]
            proof = {"kernel": "official_infer_joint", "batch_size": 1}
        else:
            context, context_mask = self._batch_context(instructions)
            prediction = infer_joint_batch(self.policy.model, context=context, context_mask=context_mask,
                                            seeds=seeds, prompt=None, **common)
            videos = prediction["video"]
            proof = {"kernel": "infer_joint_batch", "batch_size": prediction["batch_size"],
                     "denoiser_batch_sizes": prediction["denoiser_batch_sizes"],
                     "decoded_video_shape": prediction.get("decoded_video_shape"),
                     "action_shape": prediction.get("action_shape")}
            if proof["batch_size"] != size or proof["denoiser_batch_sizes"] != [size] * 10:
                raise RuntimeError("WM kernel did not preserve the requested true batch at all denoising steps")
        return validate_videos(videos, size), proof

    def _score_videos(self, videos, instructions):
        # Exact old RM image crop/preprocess, now one row-major B*8 call.
        heads = np.stack([np.asarray(frame, dtype=np.uint8)[:256] for frames in videos for frame in frames[1:]])
        labels = [instruction for instruction in instructions for _ in range(8)]
        rewards = self.reward.compute_reward(self.torch.from_numpy(heads), labels)
        scores = rewards.float().cpu().numpy().astype(np.float32, copy=True)
        if scores.shape != (len(videos) * 8,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise RuntimeError("Invalid reward probabilities or batch ordering shape")
        return scores.reshape(len(videos), 8)

    def infer(self, data):
        with self.lock:
            self.onload()
            next_images, scores_batch, timings, future_heads = [], [], [], []
            return_heads = bool(np.asarray(data.get("return_future_head_frames", False)).item())
            batch = len(data["images"])
            request_id = str(data.get("request_id", f"request-{self.requests_completed}"))
            request_started = time.monotonic()
            self.log.event("request_started", request_id=request_id, batch=batch,
                           execution_mode=self.args.execution_mode, wm_batch_size=self.args.wm_batch_size)
            for batch_index, (start, end) in enumerate(batch_ranges(batch, self.args.wm_batch_size)):
                size = end - start
                instructions = instructions_for(data, start, end)
                started = time.monotonic()
                with self.torch.inference_mode():
                    self._begin_phase("world_model_inference")
                    videos, kernel_proof = self._predict_videos(data, start, end, instructions)
                    wm_peak = self._end_phase()
                    wm_seconds = time.monotonic() - started
                    self._begin_phase("reward_inference")
                    reward_started = time.monotonic()
                    scores = self._score_videos(videos, instructions)
                    reward_peak = self._end_phase()
                    reward_seconds = time.monotonic() - reward_started
                seconds = time.monotonic() - started
                # Per-row timing is amortized batch compute, not B1 latency;
                # shared actual latency is recorded once in batch_completed.
                row_seconds = seconds / size
                for offset, row in enumerate(range(start, end)):
                    frames, score = videos[offset], scores[offset]
                    next_images.append(np.asarray(frames[-1], dtype=np.uint8).copy())
                    if return_heads:
                        future_heads.append(np.stack([np.asarray(frame, dtype=np.uint8)[:256].copy() for frame in frames[1:]]))
                    scores_batch.append(score)
                    timings.append(row_seconds)
                    self._save_sample(frames, score, data, row, instructions[offset], row_seconds)
                    self.rows_completed += 1
                    self.log.event("row_completed", request_id=request_id, row=row, batch_index=batch_index,
                        seed=int(data["seeds"][row]), seconds=row_seconds, timing_kind="amortized_batch_compute",
                        actual_wm_batch=size, score_min=float(score.min()), score_max=float(score.max()),
                        score_last=float(score[-1]), action_telemetry=action_telemetry(
                            data["actions"][row], data["states"][row], norm_stats=self.policy.norm_stats,
                            normalization_mode=self.policy.normalization_mode,
                            action_condition_mode=self.policy.action_condition_mode), **row_identity(data, row))
                self.batch_calls_completed += 1
                self.max_actual_wm_batch = max(self.max_actual_wm_batch, size)
                self.log.event("batch_completed", request_id=request_id, batch_index=batch_index,
                    row_start=start, row_end_exclusive=end, actual_wm_batch=size,
                    configured_wm_batch=self.args.wm_batch_size, execution_mode=self.args.execution_mode,
                    output_shapes={"videos": [size, 9, 384, 320, 3], "scores": [size, 8]},
                    outputs_finite=True, rows_completed=size,
                    seconds=seconds, world_model_seconds=wm_seconds, reward_seconds=reward_seconds,
                    wm_peak=wm_peak, reward_peak=reward_peak, kernel_proof=kernel_proof,
                    rows_per_second=size / max(seconds, 1e-9))
                del videos, frames, scores, score
            self.requests_completed += 1
            self.log.phase = "ready_loaded"
            self.log.event("request_completed", request_id=request_id, batch=batch,
                           seconds=time.monotonic() - request_started, compute_seconds=sum(timings))
            return encode_response(np.stack(next_images), np.stack(scores_batch), timings,
                                   np.stack(future_heads) if return_heads else None)


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OpenDWSmoke/1"

        def log_message(self, format, *args):
            pass

        def _reply(self, status, body, content_type="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._reply(200, backend.health())
            else:
                self._reply(404, {"ok": False, "error": "Unknown endpoint"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY_BYTES:
                    raise ValueError("Invalid request Content-Length")
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise ValueError("Incomplete request body")
                if self.path == "/infer":
                    data = decode_request(payload)
                    body = backend.infer(data)
                    self._reply(200, body, "application/octet-stream")
                elif self.path in {"/onload", "/offload"}:
                    if json.loads(payload) != {}:
                        raise ValueError("Control request must be an empty JSON object")
                    result = backend.onload() if self.path == "/onload" else backend.offload()
                    self._reply(200, result)
                else:
                    self._reply(404, {"ok": False, "error": "Unknown endpoint"})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                backend.log.event("request_rejected", error=str(error), path=self.path)
                self._reply(400, {"ok": False, "error": str(error)})
            except Exception as error:
                backend.fatal_error = type(error).__name__ + ": " + str(error)
                backend.log.phase = "failed"
                backend.log.event("request_failed", error=backend.fatal_error, traceback=traceback.format_exc(), path=self.path)
                self._reply(500, {"ok": False, "error": backend.fatal_error})
    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reward-checkpoint", type=Path, required=True)
    parser.add_argument("--t5-path", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-samples", type=int, default=2)
    parser.add_argument("--monitor-seconds", type=float, default=10)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--execution-mode", choices=("batched", "b1_reference"), default="batched")
    parser.add_argument("--wm-batch-size", type=int, choices=(1, 16), default=16)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        parser.error("CUDA_VISIBLE_DEVICES must name exactly the authorized physical GPU")
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local unprivileged TCP port")
    if args.evidence_samples < 0 or args.cpu_threads < 1 or args.monitor_seconds < 1:
        parser.error("Invalid evidence, CPU thread, or monitor limit")
    if (args.execution_mode, args.wm_batch_size) not in {("batched", 16), ("b1_reference", 1)}:
        parser.error("Use batched/16 or explicit b1_reference/1; automatic batch reduction is disabled")
    if (args.output_dir / "service-events.jsonl").exists():
        parser.error("Service output already exists; use a new run directory")
    for path in (args.bundle / "model.pt", args.bundle / "norm_stats.json",
                 args.bundle / "text_encoder/model.pth", args.bundle / "vae/model.pth",
                 args.bundle / "tokenizer/tokenizer_config.json", args.reward_checkpoint):
        if not path.is_file():
            parser.error(f"Missing asset: {path}")
    return args


def main():
    args = parse_args()
    log = ResourceLog(args.output_dir, args.physical_gpu, args.monitor_seconds)
    backend = None
    server = None
    try:
        backend = Backend(args, log)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backend))
        def stop(signum, frame):
            log.event("shutdown_requested", signal=signum)
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        log.event("listening", url=f"http://127.0.0.1:{args.port}")
        server.serve_forever(poll_interval=0.5)
    except BaseException as error:
        log.phase = "failed"
        log.event("fatal", error=type(error).__name__ + ": " + str(error), traceback=traceback.format_exc())
        raise
    finally:
        if server is not None:
            server.server_close()
        if backend is not None:
            try:
                backend.offload()
            except Exception as error:
                log.event("shutdown_offload_failed", error=str(error))
        log.event("stopped")
        log.close()


if __name__ == "__main__":
    main()
