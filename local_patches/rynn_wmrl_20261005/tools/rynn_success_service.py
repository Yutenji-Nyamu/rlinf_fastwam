"""Pinned RynnValue Success batch service; initially CPU resident.

POST /infer: binary np.savez (allow_pickle=False), exactly these fields:
  frames: uint8[B,8,256,320,3], instructions: Unicode[B],
  episode_uids: Unicode[B], end_action_indices: integer[B],
  frame_action_indices: integer[B,8], request_id: Unicode scalar.
Optional rm_batch_size: positive integer scalar, <= service --batch-size cap.
Frames are one chronological head-camera prefix per row, including duplicates
for short prefixes; frame_action_indices[:, -1] == end_action_indices.
JSON response has ordered items [{episode_uid,end_action_idx,success: bool|null,
match: bool|null,parse_status,raw_text,attempts}], timings, memory, fingerprint.
Top-level success codes (1/0/-1) and identity arrays are also provided.

GET /health; POST /onload; POST /offload (empty bodies) return JSON. /infer
loads once and stays GPU resident until synchronized /offload. Busy is 409;
model/offload failures are 500 and never become a logical No. Service binds
only 127.0.0.1 and accepts only its single declared physical GPU in 4..7.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import traceback
import zipfile

import numpy as np

MODEL_ID = "Alibaba-DAMO-Academy/RynnValue-8B"
MODEL_REVISION = "8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e"
ATTENTION = "pred_slot_isolated_eager"
SERVICE_VERSION = "rynn-success-prefix8-v2-bf16"
MAX_BODY_BYTES = 256 * 1024 * 1024
MAX_ROWS = 128
LOCKED_FILES = {
    "config.json": "5bb10e3b8e29a3030cfb238cb9cadb4712871b10310550665378f2d0391493b9",
    "processor_config.json": "9d6a2984147074c6f81b0bca947446cfab0cf542cfe05cc8eb19a95f8e290d80",
    "preprocessor_config.json": "90a842b2fb4e3122e8a111c4f1e716d19a9b8f8b49b6c83de527989762fd1795",
    "value_heads.py": "767c0375d058211eedef07a4b0ce9f4a6f978dc823e1726be7a45020f0c7dcac",
    "value_tokenizer.py": "b42a4b4a1151227a46c7c467390d7d17dab99d6d5e276f771d18d44a8802a211",
    "attention_impl.py": "6254d3371324c5b9d0b2fde28bca55da803daa0919e839f2bbdcba7a6f2084cc",
    "modeling_rynn_value_lang.py": "ed56f23402cffc17645ea67a909a101c9f4d7c6128c174ef9450ac34464bab54",
    "processing_rynn_value_lang.py": "206e6c947e1ea2ec176a2e7c2a8d7b90ac2aa3c773f1e0505d698dd446b1c082",
    "conversations.py": "68440aeddf92e016a8be6c3ae5e87f6eac755d1980295979c1e9b679cfbffeb8",
}


class BusyError(RuntimeError):
    pass


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_assets(directory, manifest_path):
    directory, manifest_path = Path(directory), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Manifest does not identify the pinned RynnValue revision")
    records = manifest.get("files", manifest.get("shards", []))
    if isinstance(records, list):
        records = {r.get("path", r.get("name", r.get("rfilename"))): r for r in records}
    if not isinstance(records, dict):
        raise ValueError("Invalid shard manifest")
    hashes = {}
    for name, expected in LOCKED_FILES.items():
        actual = sha256_file(directory / name)
        if actual != expected:
            raise ValueError(f"Pinned source/config differs: {name}")
        hashes[name] = actual
    index_path = directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    if shards != [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]:
        raise ValueError("Expected the four official model shards")
    shard_receipts = []
    for name in shards:
        rec = records.get(name)
        if isinstance(rec, str):
            rec = {"sha256": rec}
        if not isinstance(rec, dict) or not any(rec.get(k) for k in ("sha256", "etag", "size", "size_bytes")):
            raise ValueError(f"Missing download provenance: {name}")
        size = (directory / name).stat().st_size
        expected_size = rec.get("size", rec.get("size_bytes"))
        if expected_size is not None and size != expected_size:
            raise ValueError(f"Shard size mismatch: {name}")
        shard_receipts.append({"path": name, "size": size, "download_provenance": rec})
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix in (".py", ".json", ".jinja", ".txt"):
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large metadata: {path.name}")
            hashes.setdefault(path.name, sha256_file(path))
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "manifest_sha256": sha256_file(manifest_path),
            "files_sha256": hashes, "shards": shard_receipts,
            "metadata_sha256": hashlib.sha256(canonical_json(hashes).encode()).hexdigest()}


def read_gpu_identity(physical_gpu):
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
        text=True, timeout=15)
    matches = []
    for line in output.splitlines():
        values = [x.strip() for x in line.split(",")]
        if len(values) == 3 and values[0] == str(physical_gpu):
            matches.append(dict(physical_gpu=physical_gpu, gpu_uuid=values[1], pci=values[2]))
    if len(matches) != 1 or not matches[0]["gpu_uuid"].startswith("GPU-"):
        raise RuntimeError("Cannot identify requested physical GPU")
    return matches[0]


def verify_cuda_identity(expected):
    import ctypes as C
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(expected["physical_gpu"]):
        raise RuntimeError("Expected one fixed physical GPU in CUDA_VISIBLE_DEVICES")
    if read_gpu_identity(expected["physical_gpu"]) != expected:
        raise RuntimeError("Physical GPU identity changed")
    cuda = C.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [C.c_uint]
    cuda.cuDeviceGetCount.argtypes = [C.POINTER(C.c_int)]
    cuda.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
    cuda.cuDeviceGetPCIBusId.argtypes = [C.c_char_p, C.c_int, C.c_int]
    get_uuid = getattr(cuda, "cuDeviceGetUuid_v2", cuda.cuDeviceGetUuid)
    get_uuid.argtypes = [C.c_void_p, C.c_int]
    count, device = C.c_int(), C.c_int()
    pci, uuid = C.create_string_buffer(32), (C.c_ubyte * 16)()
    if not (cuda.cuInit(0) == 0 and cuda.cuDeviceGetCount(C.byref(count)) == 0
            and count.value == 1 and cuda.cuDeviceGet(C.byref(device), 0) == 0
            and cuda.cuDeviceGetPCIBusId(pci, len(pci), device) == 0
            and get_uuid(C.byref(uuid), device) == 0):
        raise RuntimeError("Cannot verify one visible CUDA device")
    raw = bytes(uuid).hex()
    actual = "GPU-" + "-".join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))
    if (actual.lower() != expected["gpu_uuid"].lower()
            or pci.value.decode().lower()[-10:] != expected["pci"].lower()[-10:]):
        raise RuntimeError("Logical CUDA0 is not the declared physical GPU")
    return {**expected, "logical_device_count": 1, "logical_cuda0_uuid": actual}


def _strings(array, length, name, limit):
    if array.shape != (length,) or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a string[B] array")
    values = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in array]
    if any(not x.strip() or len(x) > limit for x in values):
        raise ValueError(f"{name} contains empty or excessive strings")
    return values


def validate_request(data):
    required = {"frames", "instructions", "episode_uids", "end_action_indices",
                "frame_action_indices", "request_id"}
    if required - set(data) or set(data) - required - {"rm_batch_size"}:
        raise ValueError(f"Expected exactly {sorted(required)}")
    frames = data["frames"]
    batch = len(frames) if frames.ndim else 0
    if not 1 <= batch <= MAX_ROWS or frames.shape != (batch, 8, 256, 320, 3) or frames.dtype != np.uint8:
        raise ValueError("frames must be uint8[B,8,256,320,3], 1<=B<=128")
    instructions = _strings(data["instructions"], batch, "instructions", 4096)
    uids = _strings(data["episode_uids"], batch, "episode_uids", 512)
    ends, indices = data["end_action_indices"], data["frame_action_indices"]
    if ends.shape != (batch,) or ends.dtype.kind not in "iu" or np.any(ends < 0):
        raise ValueError("end_action_indices must be nonnegative integer[B]")
    if (indices.shape != (batch, 8) or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices[:, 1:] < indices[:, :-1])
            or np.any(indices[:, -1] != ends)):
        raise ValueError("frame_action_indices must be chronological integer[B,8] ending at end_action_indices")
    if len(set(zip(uids, map(int, ends)))) != batch:
        raise ValueError("Duplicate episode_uid/end_action_idx identity")
    request_id = data["request_id"]
    if request_id.shape != () or request_id.dtype.kind not in "US":
        raise ValueError("request_id must be a string scalar")
    rid = request_id.item()
    rid = rid.decode("utf-8") if isinstance(rid, bytes) else str(rid)
    if not rid.strip() or len(rid) > 512:
        raise ValueError("Invalid request_id")
    result = {**data, "instructions": instructions, "episode_uids": uids, "request_id": rid}
    if "rm_batch_size" in data:
        override = data["rm_batch_size"]
        if override.shape != () or override.dtype.kind not in "iu" or not 1 <= int(override) <= MAX_ROWS:
            raise ValueError("rm_batch_size must be a positive integer scalar")
        result["rm_batch_size"] = int(override)
    return result


def decode_request(payload):
    if not 0 < len(payload) <= MAX_BODY_BYTES:
        raise ValueError("Invalid request length")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if len(entries) not in (6, 7) or sum(x.file_size for x in entries) > MAX_BODY_BYTES:
                raise ValueError("NPZ expanded size or fields exceed request limits")
        with np.load(io.BytesIO(payload), allow_pickle=False) as source:
            data = {key: source[key] for key in source.files}
        return validate_request(data)
    except (OSError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("Malformed NPZ request") from exc


FIELD_RE = re.compile(r"(?im)^\s*-?\s*(Success|Match)\s*:\s*([^\r\n]*)$")


def parse_analysis(text, truncated=False):
    fields = {"success": [], "match": []}
    invalid_success = False
    for name, value in FIELD_RE.findall(text):
        answer = re.fullmatch(r"(Yes|No)[.!]?\s*", value.strip(), re.IGNORECASE)
        if not answer:
            invalid_success |= name.lower() == "success"
            continue
        fields[name.lower()].append(answer[1].lower() == "yes")
    if invalid_success:
        return {"success": None, "match": None,
                "parse_status": "truncated" if truncated else "invalid_success_format"}
    if any(len(set(values)) > 1 for values in fields.values()):
        return {"success": None, "match": None, "parse_status": "conflicting_fields"}
    match = fields["match"][0] if fields["match"] else None
    success = fields["success"][0] if fields["success"] else None
    if success is True and match is False:
        return {"success": None, "match": match, "parse_status": "conflicting_match_success"}
    status = "ok" if success is not None else "truncated" if truncated else "missing_success"
    return {"success": success, "match": match, "parse_status": status}


def bucket_indices(prepared, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    buckets = OrderedDict()
    for row, sample in enumerate(prepared):
        ids = sample["input_ids"]
        if len(ids.shape) != 2 or ids.shape[0] != 1:
            raise ValueError("Processor must return one sample at a time")
        key = (int(ids.shape[1]), tuple(sample["image_grid_thw"].shape))
        buckets.setdefault(key, []).append(row)
    return [rows[start:start + batch_size] for rows in buckets.values()
            for start in range(0, len(rows), batch_size)]


def collate_prepared(samples, torch, device):
    if not samples or len({int(x["input_ids"].shape[1]) for x in samples}) != 1:
        raise ValueError("Only equal-length token sequences may share a generation batch")
    return {
        "input_ids": torch.cat([x["input_ids"] for x in samples], dim=0).to(device).long(),
        "attention_mask": torch.cat([x["attention_mask"] for x in samples], dim=0).to(device).long(),
        "pixel_values": torch.cat([x["pixel_values"].flatten(0, 1) for x in samples], dim=0).to(device),
        "image_grid_thw": torch.cat([x["image_grid_thw"].flatten(0, 1) for x in samples], dim=0).to(device).long(),
    }


def restore_order(size, batches):
    results = [None] * size
    for indices, values in batches:
        if len(indices) != len(values):
            raise RuntimeError("Generation result count differs from input count")
        for row, value in zip(indices, values):
            if row < 0 or row >= size or results[row] is not None:
                raise RuntimeError("Duplicate or invalid generated row")
            results[row] = value
    if any(x is None for x in results):
        raise RuntimeError("Missing generated row")
    return results


def prepare_bf16_cpu_model(model, torch):
    """Mirror official inference.py's explicit model.to(..., dtype=BF16).

BRO heads construct Linear layers with an explicit FP32 default. Passing
torch_dtype to from_pretrained alone did not convert these custom parameters
in the pinned runtime. Success generation still evaluates the value heads on
prefill, so a BF16 hidden state otherwise reaches a FP32 Linear layer.
Cast in CPU preparation; checkpoint files and the generation prompt stay fixed.
"""
    model = model.to(device="cpu", dtype=torch.bfloat16)
    bad = [name for name, parameter in model.named_parameters()
           if parameter.is_floating_point() and parameter.dtype != torch.bfloat16]
    if bad:
        raise RuntimeError(f"Model retained non-BF16 floating parameters: {bad[:8]}")
    return model


class RynnSuccessService:
    def __init__(self, model_path, manifest_path, physical_gpu, batch_size=8,
                 robot_description="a bimanual ALOHA robot",
                 camera_description="a fixed third-person camera", max_new_tokens=128,
                 retry_max_new_tokens=256, unknown_retries=1):
        if physical_gpu not in (4, 5, 6, 7) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
            raise ValueError("Service requires exactly its declared physical GPU 4..7")
        if not 1 <= batch_size <= MAX_ROWS or max_new_tokens < 1 or retry_max_new_tokens < max_new_tokens:
            raise ValueError("Invalid batch or generation limits")
        if unknown_retries not in (0, 1) or not robot_description.strip() or not camera_description.strip():
            raise ValueError("Invalid retry or observation metadata")
        self.gpu_identity = read_gpu_identity(physical_gpu)
        assets = validate_assets(model_path, manifest_path)
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.lock = threading.Lock()
        self.healthy, self.is_offloaded = True, True
        self.completed_requests, self.last_error = 0, None
        self.batch_size = batch_size
        self.max_new_tokens, self.retry_max_new_tokens = max_new_tokens, retry_max_new_tokens
        self.unknown_retries = unknown_retries
        self.robot_description, self.camera_description = robot_description, camera_description
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        config._attn_implementation = ATTENTION
        self.model = AutoModel.from_pretrained(
            model_path, config=config, torch_dtype=torch.bfloat16,
            trust_remote_code=True, local_files_only=True).eval()
        self.model = prepare_bf16_cpu_model(self.model, torch)
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        if (self.model.config._attn_implementation != ATTENTION
                or self.model.config.value_token_repeat != 8
                or self.processor.value_token_repeat != 8 or not self.processor.use_meta):
            raise RuntimeError("Loaded model differs from pinned inference contract")
        self._assert_cpu()
        self.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.fingerprint = {**assets, "service_version": SERVICE_VERSION,
            "service_sha256": sha256_file(__file__), "attention": ATTENTION,
            "dtype": "bfloat16", "prefix_frames": 8, "image_shape": [256, 320, 3],
            "dtype_policy": "official_explicit_full_model_bfloat16",
            "batch_size": batch_size, "max_new_tokens": max_new_tokens,
            "retry_max_new_tokens": retry_max_new_tokens, "unknown_retries": unknown_retries,
            "robot_description": robot_description, "camera_description": camera_description,
            "runtime_versions": {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "Pillow", "huggingface-hub")}}
        self.fingerprint["sha256"] = hashlib.sha256(canonical_json(self.fingerprint).encode()).hexdigest()
        self.last_memory = {"cuda_initialized": bool(torch.cuda.is_initialized())}

    def _assert_cpu(self):
        if any(x.device.type != "cpu" for x in self.model.parameters()) or any(
                x.device.type != "cpu" for x in self.model.buffers()):
            raise RuntimeError("Rynn model did not fully return to CPU")

    def _memory(self):
        cuda = self.torch.cuda
        if not cuda.is_initialized():
            return {"cuda_initialized": False, "allocated_bytes": 0, "reserved_bytes": 0}
        return {"cuda_initialized": True,
            "allocated_bytes": int(cuda.memory_allocated(self.device)),
            "reserved_bytes": int(cuda.memory_reserved(self.device)),
            "peak_allocated_bytes": int(cuda.max_memory_allocated(self.device)),
            "peak_reserved_bytes": int(cuda.max_memory_reserved(self.device))}

    def health(self):
        return {"ok": self.healthy, "pid": os.getpid(), "busy": self.lock.locked(),
            "is_offloaded": self.is_offloaded, "physical_gpu": self.gpu_identity["physical_gpu"],
            "gpu_identity": self.gpu_identity, "fingerprint": self.fingerprint,
            "completed_requests": self.completed_requests, "last_error": self.last_error,
            "memory": self.last_memory, "batch_size": self.batch_size}

    def _onload(self):
        if not self.healthy:
            raise RuntimeError("Service unhealthy; owner must replace this exact process")
        if not self.is_offloaded:
            return
        self.cuda_identity = verify_cuda_identity(self.gpu_identity)
        self.torch.cuda.set_device(self.device)
        self.is_offloaded = False
        self.model.to(self.device)
        self.torch.cuda.synchronize(self.device)
        self.last_memory = self._memory()

    def _offload(self):
        cuda = self.torch.cuda
        if cuda.is_initialized():
            cuda.synchronize(self.device)
        self.model.to("cpu")
        for module in self.model.modules():
            if isinstance(getattr(module, "rope_deltas", None), self.torch.Tensor):
                module.rope_deltas = None
        self._assert_cpu()
        gc.collect()
        if cuda.is_initialized():
            cuda.synchronize(self.device)
            cuda.empty_cache()
            cuda.synchronize(self.device)
        self.last_memory = self._memory()
        self.is_offloaded = True

    def _exclusive(self, function):
        if not self.lock.acquire(blocking=False):
            raise BusyError("Rynn service busy")
        try:
            return function()
        except Exception as exc:
            self.healthy = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            try:
                self._offload()
            except Exception as release_error:
                self.is_offloaded = False
                self.last_error += f"; offload failed: {release_error}"
            raise
        finally:
            self.lock.release()

    def onload(self):
        def perform():
            self._onload()
            return {**self.health(), "busy": False}
        return self._exclusive(perform)

    def offload(self):
        def perform():
            self._offload()
            return {**self.health(), "busy": False}
        return self._exclusive(perform)

    def _generate_batch(self, samples, max_tokens):
        torch = self.torch
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.monotonic()
        kwargs = collate_prepared(samples, torch, self.device)
        input_length = kwargs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(**kwargs, logits_to_keep=1,
                max_new_tokens=max_tokens, do_sample=False, num_beams=1,
                eos_token_id=self.eos_token_id, pad_token_id=self.eos_token_id, use_cache=True)
        if generated.ndim != 2 or generated.shape[0] != len(samples):
            raise RuntimeError("Unexpected generation batch dimensions")
        token_rows = generated[:, input_length:].detach().cpu().tolist()
        torch.cuda.synchronize(self.device)
        elapsed = time.monotonic() - started
        memory = self._memory()
        rows = []
        for tokens in token_rows:
            had_eos = self.eos_token_id in tokens
            if had_eos:
                tokens = tokens[:tokens.index(self.eos_token_id) + 1]
            text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            parsed = parse_analysis(text, truncated=not had_eos and len(tokens) >= max_tokens)
            rows.append({**parsed, "raw_text": text, "generated_tokens": len(tokens),
                         "max_new_tokens": max_tokens, "batch_elapsed_s": elapsed})
        del generated, kwargs
        return rows, {"size": len(samples), "input_tokens": int(input_length),
                      "max_new_tokens": max_tokens, "elapsed_s": elapsed, **memory}

    def infer(self, data):
        # Decode/validate errors are caller errors and do not poison the model.
        request = validate_request(data) if isinstance(data["instructions"], np.ndarray) else data
        effective_batch_size = request.get("rm_batch_size", self.batch_size)
        if not isinstance(effective_batch_size, int) or not 1 <= effective_batch_size <= self.batch_size:
            raise ValueError("Requested rm_batch_size exceeds the service batch-size cap")
        def perform():
            from PIL import Image
            started = time.monotonic()
            load_started = time.monotonic()
            self._onload()
            onload_s = time.monotonic() - load_started
            prepare_started = time.monotonic()
            prepared = []
            for row, instruction in enumerate(request["instructions"]):
                sample = self.processor.process_episode(instruction=instruction,
                    images=[Image.fromarray(frame) for frame in request["frames"][row]],
                    robot_description=self.robot_description, camera_description=self.camera_description)
                ids = sample["input_ids"]
                if (int((ids == self.model.config.value_token_id).sum()) != 8 * 8
                        or int((ids == self.model.config.relative_value_token_id).sum()) != 7 * 8):
                    raise RuntimeError("Unexpected prediction slot counts")
                prepared.append(sample)
            prepare_s = time.monotonic() - prepare_started
            batches, diagnostics = [], []
            for indices in bucket_indices(prepared, effective_batch_size):
                rows, diag = self._generate_batch([prepared[x] for x in indices], self.max_new_tokens)
                diag["row_indices"] = indices
                diagnostics.append(diag)
                for row in rows:
                    row["attempts"] = [{k: v for k, v in row.items() if k != "attempts"}]
                retry_indices = [i for i, row in enumerate(rows)
                    if row["parse_status"] in ("missing_success", "truncated")]
                if self.unknown_retries and retry_indices:
                    retried, retry_diag = self._generate_batch(
                        [prepared[indices[i]] for i in retry_indices], self.retry_max_new_tokens)
                    retry_diag["row_indices"] = [indices[i] for i in retry_indices]
                    retry_diag["retry"] = True
                    diagnostics.append(retry_diag)
                    for local_idx, retried_row in zip(retry_indices, retried):
                        attempts = rows[local_idx]["attempts"] + [dict(retried_row)]
                        rows[local_idx] = {**retried_row, "attempts": attempts}
                batches.append((indices, rows))
            ordered = restore_order(len(prepared), batches)
            items = [{**row, "episode_uid": request["episode_uids"][i],
                "end_action_idx": int(request["end_action_indices"][i]),
                "frame_action_indices": request["frame_action_indices"][i].tolist()}
                for i, row in enumerate(ordered)]
            self.completed_requests += 1
            self.last_memory = self._memory()
            self.last_memory["request_peak_allocated_bytes"] = max(
                x["peak_allocated_bytes"] for x in diagnostics)
            self.last_memory["request_peak_reserved_bytes"] = max(
                x["peak_reserved_bytes"] for x in diagnostics)
            return {"ok": True, "request_id": request["request_id"], "items": items,
                "success": [-1 if x["success"] is None else int(x["success"]) for x in items],
                "episode_uids": [x["episode_uid"] for x in items],
                "end_action_indices": [x["end_action_idx"] for x in items],
                "raw_text": [x["raw_text"] for x in items],
                "parse_status": [x["parse_status"] for x in items],
                "timings": {"elapsed_s": time.monotonic() - started,
                    "onload_s": onload_s, "prepare_s": prepare_s, "batches": diagnostics},
                "memory": self.last_memory, "fingerprint_sha256": self.fingerprint["sha256"],
                "pid": os.getpid(), "is_offloaded": self.is_offloaded,
                "rm_batch_size": effective_batch_size}
        return self._exclusive(perform)


def make_handler(service, log_path=None):
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            print(canonical_json({"time": time.time(), "http": format % args}), flush=True)

        def _reply(self, status, value):
            body = canonical_json(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self._reply(404, {"ok": False, "error": "Unknown endpoint"})
                return
            self._reply(200, service.health())

        def do_POST(self):
            try:
                if self.path not in ("/infer", "/onload", "/offload"):
                    self._reply(404, {"ok": False, "error": "Unknown endpoint"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > MAX_BODY_BYTES:
                    raise ValueError("Invalid Content-Length")
                if self.path == "/infer":
                    result = service.infer(decode_request(self.rfile.read(length)))
                else:
                    if length:
                        raise ValueError("Lifecycle endpoints require an empty body")
                    result = service.onload() if self.path == "/onload" else service.offload()
                if log_path and self.path == "/infer":
                    with Path(log_path).open("a", encoding="utf-8") as stream:
                        stream.write(canonical_json(result) + "\n")
                self._reply(200, result)
            except BusyError as exc:
                self._reply(409, {"ok": False, "error": str(exc)})
            except ValueError as exc:
                # Model failures also subclass ValueError; service health tells them apart.
                self._reply(400 if service.healthy else 500,
                            {"ok": False, "error": str(exc), "pid": os.getpid()})
            except Exception as exc:
                traceback.print_exc()
                self._reply(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "pid": os.getpid()})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(4, 5, 6, 7))
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--robot-description", required=True)
    parser.add_argument("--camera-description", required=True)
    parser.add_argument("--max-new-tokens", default=128, type=int)
    parser.add_argument("--retry-max-new-tokens", default=256, type=int)
    parser.add_argument("--unknown-retries", default=1, type=int, choices=(0, 1))
    parser.add_argument("--log-path")
    args = parser.parse_args()
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    service = RynnSuccessService(args.model_path, args.manifest_path, args.physical_gpu,
        batch_size=args.batch_size, robot_description=args.robot_description,
        camera_description=args.camera_description, max_new_tokens=args.max_new_tokens,
        retry_max_new_tokens=args.retry_max_new_tokens, unknown_retries=args.unknown_retries)
    print(canonical_json({"event": "ready", **service.health()}), flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service, args.log_path))
    server.daemon_threads = True
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.offload()


if __name__ == "__main__":
    main()
