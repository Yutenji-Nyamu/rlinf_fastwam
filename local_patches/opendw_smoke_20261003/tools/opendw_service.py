"""Loopback, B1-at-a-time OpenDW + WorldArena reward service for RLinf smoke.

Example (the owner supplies the authorized single physical GPU):
  CUDA_VISIBLE_DEVICES=4 python -B tools/opendw_service.py \
    --bundle /path/DW05-Robotwin --reward-checkpoint /path/resnet_rm.pth \
    --t5-path /path/t5-base --physical-gpu 4 --port 18941 --output-dir /path/run

No GPU model is resident until /onload or /infer. /offload replies only after
all DW05 and RM parameters/buffers are on CPU and CUDA work is synchronized.
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


OPENDW_COMMIT = "e33befa8005a1585e0140dbf464566e90bc79aa1"
WORLDARENA_COMMIT = "5978ce5c81e55b8c8358f4f5966a13ce385ff155"
MAX_BODY_BYTES = 128 * 1024 * 1024


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


def encode_response(next_images, scores, timings=None):
    result = io.BytesIO()
    values = {"next_images": np.asarray(next_images, dtype=np.uint8), "scores": np.asarray(scores, dtype=np.float32)}
    if timings is not None:
        values["timing_s"] = np.asarray(timings, dtype=np.float32)
    np.savez(result, **values)
    return result.getvalue()


def row_identity(data, row):
    """Optional logging identity; missing/malformed metadata stays unknown."""
    result = {}
    for source, target in (("env_indices", "env_index"), ("reset_ids", "reset_id")):
        values = data.get(source)
        valid = (isinstance(values, np.ndarray) and
                 values.shape == (len(data["images"]),) and values.dtype.kind in "iu")
        result[target] = int(values[row]) if valid else None
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
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._monitor, name="resource-monitor", daemon=True)
        self._thread.start()

    def snapshot(self):
        result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "phase": self.phase, **process_memory()}
        if self.torch is not None and self.torch.cuda.is_initialized():
            try:
                result.update(cuda_allocated_bytes=self.torch.cuda.memory_allocated(0),
                              cuda_reserved_bytes=self.torch.cuda.memory_reserved(0),
                              cuda_max_allocated_bytes=self.torch.cuda.max_memory_allocated(0),
                              cuda_max_reserved_bytes=self.torch.cuda.max_memory_reserved(0))
            except Exception as error:
                result["cuda_metrics_error"] = str(error)
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

    def event(self, event, **fields):
        record = {**self.snapshot(), "event": event, **fields}
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
            self.event("heartbeat")

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
        self.lock = threading.RLock()
        self.is_offloaded = True
        self.fatal_error = None
        self.requests_completed = 0
        self.rows_completed = 0
        self.samples_saved = 0
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
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "action_horizon": 32, "num_video_frames": 9, "num_inference_steps": 10,
            "logical_batch_execution": "B1 sequential", "reward_strict_load": True,
            "reward_preprocess": "head RGB -> bilinear 224 -> ImageNet normalize",
        }
        (args.output_dir / "service-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log.phase = "ready_offloaded"
        log.event("ready", is_offloaded=True)

    def health(self):
        return {"ok": self.fatal_error is None, "pid": os.getpid(), "phase": self.log.phase,
                "is_offloaded": self.is_offloaded, "fatal_error": self.fatal_error,
                "requests_completed": self.requests_completed, "rows_completed": self.rows_completed,
                "physical_gpu": self.args.physical_gpu}

    def onload(self):
        with self.lock:
            if self.fatal_error:
                raise RuntimeError(f"Service requires owner recovery after: {self.fatal_error}")
            if not self.is_offloaded:
                return {"ok": True, "is_offloaded": False}
            self.log.phase = "onloading"
            self.log.event("onload_started")
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

    def infer(self, data):
        with self.lock:
            self.onload()
            next_images, scores_batch, timings = [], [], []
            batch = len(data["images"])
            request_id = str(data.get("request_id", f"request-{self.requests_completed}"))
            self.log.event("request_started", request_id=request_id, batch=batch)
            for row in range(batch):
                self.log.phase = "world_model_inference"
                started = time.monotonic()
                instruction = data["instructions"][row]
                if isinstance(instruction, bytes):
                    instruction = instruction.decode("utf-8")
                instruction = str(instruction)
                identity = row_identity(data, row)
                self.log.event(
                    "row_started", request_id=request_id, row=row, seed=int(data["seeds"][row]),
                    action_telemetry=action_telemetry(
                        data["actions"][row], data["states"][row], norm_stats=self.policy.norm_stats,
                        normalization_mode=self.policy.normalization_mode,
                        action_condition_mode=self.policy.action_condition_mode,
                    ),
                    **identity,
                )
                with self.torch.inference_mode():
                    image = self.pil_to_model_tensor(Image.fromarray(data["images"][row]), self.device, self.policy.model.torch_dtype)
                    action = self.policy.normalize_action_condition(data["actions"][row], data["states"][row])
                    proprio = self.policy.normalize_state(data["states"][row])
                    prediction = self.policy.model.infer_joint(
                        prompt=self.policy.format_prompt(instruction), input_image=image,
                        num_video_frames=9, action_horizon=32, action=action, proprio=proprio,
                        num_inference_steps=10, sigma_shift=self.policy.config.sigma_shift,
                        seed=int(data["seeds"][row]), rand_device="cpu", tiled=False,
                        test_action_with_infer_action=False,
                    )
                    frames = [frame.convert("RGB") for frame in prediction["video"]]
                    if len(frames) != 9 or any(frame.size != (320, 384) for frame in frames):
                        raise RuntimeError(f"Unexpected predicted video shape: {[frame.size for frame in frames]}")
                    del prediction, image, action, proprio
                    self.torch.cuda.synchronize(self.device)
                    self.log.phase = "reward_inference"
                    heads = np.stack([np.asarray(frame, dtype=np.uint8)[:256] for frame in frames[1:]])
                    # Original RM preprocess resizes its input directly to 224;
                    # avoid a redundant resize through the policy's 256 pixels.
                    rewards = self.reward.compute_reward(self.torch.from_numpy(heads), [instruction] * 8)
                    score = rewards.float().cpu().numpy().astype(np.float32, copy=True)
                    del rewards
                if score.shape != (8,) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
                    raise RuntimeError("Invalid reward probabilities")
                seconds = time.monotonic() - started
                next_images.append(np.asarray(frames[-1], dtype=np.uint8).copy())
                scores_batch.append(score)
                timings.append(seconds)
                self._save_sample(frames, score, data, row, instruction, seconds)
                self.rows_completed += 1
                self.log.event("row_completed", request_id=request_id, row=row, seconds=seconds,
                               score_min=float(score.min()), score_max=float(score.max()), score_last=float(score[-1]),
                               **identity)
                del frames, heads
            self.requests_completed += 1
            self.log.phase = "ready_loaded"
            self.log.event("request_completed", request_id=request_id, batch=batch, seconds=sum(timings))
            return encode_response(np.stack(next_images), np.stack(scores_batch), timings)


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
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        parser.error("CUDA_VISIBLE_DEVICES must name exactly the authorized physical GPU")
    if not 1024 <= args.port <= 65535:
        parser.error("Use a local unprivileged TCP port")
    if args.evidence_samples < 0 or args.cpu_threads < 1 or args.monitor_seconds < 1:
        parser.error("Invalid evidence, CPU thread, or monitor limit")
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
