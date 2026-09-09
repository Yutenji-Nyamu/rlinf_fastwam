"""Frozen RynnValue-8B numeric scorer, runnable without importing rlinf.

Only the locked author's model is loaded. Each request scores chronological
four-frame prefixes, returns their last absolute-value slots in seconds, and
offloads the model back to CPU before acknowledging completion.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gc
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.metadata
import io
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping


MODEL_ID = "Alibaba-DAMO-Academy/RynnValue-8B"
MODEL_REVISION = "8738c5e4ce4418ea0266e9fbeffae0eb9bbb230e"
SCORER_VERSION = "rynnvalue-8b-prefix4-raw-seconds-v1"
ATTENTION = "pred_slot_isolated_eager"
PREFIX_FRAMES = 4
MAX_IMAGE_SIDE = 640
MAX_REQUEST_BYTES = 128 * 1024 * 1024
MAX_FRAME_BYTES = 8 * 1024 * 1024
MAX_FRAMES = 2048
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_TOTAL_PIXELS = 128 * 1024 * 1024

# Small files inspected at MODEL_REVISION. Hashes cover the actual HF code,
# not the similar GitHub package or its README examples.
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


class ScorerBusyError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(raw: bytes | str) -> dict:
    def reject(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(raw, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def prefix_indices(end_index: int) -> list[int]:
    """Author's linspace(0, end, 4, dtype=int), including initial duplicates."""
    if isinstance(end_index, bool) or not isinstance(end_index, int) or end_index < 0:
        raise ValueError("end_index must be a nonnegative integer")
    return [0, end_index // 3, (2 * end_index) // 3, end_index]


def validate_request(request: Mapping[str, Any]) -> dict:
    if not isinstance(request, Mapping):
        raise ValueError("Request must be an object")
    required = {"instruction", "robot_description", "camera_description", "frames_png_b64"}
    allowed = required | {"episode_id"}
    if set(request) - allowed or required - set(request):
        raise ValueError("Expected instruction, robot_description, camera_description, frames_png_b64; optional episode_id")
    for name in ("instruction", "robot_description", "camera_description"):
        value = request[name]
        if not isinstance(value, str) or not value.strip() or len(value) > 4096:
            raise ValueError(f"{name} must be a nonempty string of at most 4096 characters")
    if "episode_id" in request and (not isinstance(request["episode_id"], str) or len(request["episode_id"]) > 256):
        raise ValueError("episode_id must be a string of at most 256 characters")
    frames = request["frames_png_b64"]
    if not isinstance(frames, list) or not 2 <= len(frames) <= MAX_FRAMES:
        raise ValueError(f"Expected 2..{MAX_FRAMES} frames (N+1 observations)")
    if any(not isinstance(frame, str) or not frame or len(frame) > 4 * ((MAX_FRAME_BYTES + 2) // 3) for frame in frames):
        raise ValueError("Each frame must be a bounded base64 PNG string")
    if sum(len(frame) for frame in frames) > MAX_REQUEST_BYTES:
        raise ValueError("Encoded frames exceed request size limit")
    return dict(request)


def decode_frames(encoded_frames: list[str]):
    from PIL import Image

    images = []
    total_pixels = 0
    for index, encoded in enumerate(encoded_frames):
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"Frame {index}: invalid base64") from exc
        if len(raw) > MAX_FRAME_BYTES or not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"Frame {index}: expected a bounded PNG")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                width, height = opened.size
                total_pixels += width * height
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS or total_pixels > MAX_TOTAL_PIXELS:
                    raise ValueError("Image pixel budget exceeded")
                if opened.format != "PNG" or opened.mode != "RGB" or getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("Expected a single RGB PNG, without palette/alpha/grayscale")
                opened.load()
                image = opened.copy()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Frame {index}: {exc}") from exc
        if max(image.size) > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / max(image.size)
            target = tuple(max(1, int(round(size * scale))) for size in image.size)
            image = image.resize(target, resample=Image.Resampling.BICUBIC)
        images.append(image)
    # The project uses one camera, with a fixed resolution within an episode.
    if len({image.size for image in images}) != 1:
        raise ValueError("All episode frames must have one consistent image size")
    return images


def extract_absolute_last_value(prediction, expected_slots: int = PREFIX_FRAMES) -> tuple[float, list[float]]:
    """Locked model: one head and flattened B*K slots; B is exactly one."""
    import torch

    if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != (1, expected_slots):
        raise ValueError(f"Expected absolute prediction [1,{expected_slots}], got {getattr(prediction, 'shape', None)}")
    values = prediction.detach().float().cpu()
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 512.0001).any():
        raise ValueError("Absolute values must be finite seconds in [0,512]")
    numbers = values[0].tolist()
    return float(numbers[-1]), numbers


def _manifest_files(manifest: dict) -> dict[str, dict]:
    files = manifest.get("files", manifest.get("shards", []))
    if isinstance(files, dict):
        return {name: (dict(item) if isinstance(item, dict) else {"sha256": item}) for name, item in files.items()}
    if isinstance(files, list):
        result = {}
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("Manifest files must contain objects")
            name = item.get("path", item.get("name", item.get("rfilename")))
            if not isinstance(name, str) or name in result:
                raise ValueError("Manifest file paths must be unique strings")
            result[name] = dict(item)
        return result
    raise ValueError("Manifest files/shards must be a list or mapping")


def validate_artifacts(model_directory: Path, manifest_path: Path) -> dict:
    manifest = strict_json(manifest_path.read_bytes())
    if manifest.get("model_id") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Manifest must identify the locked official RynnValue-8B revision")
    file_records = _manifest_files(manifest)
    hashes = {}
    for name, expected in LOCKED_FILES.items():
        path = model_directory / name
        if not path.is_file():
            raise ValueError(f"Missing locked model file: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Official model source/config hash mismatch: {name}")
        hashes[name] = actual
    index_path = model_directory / "model.safetensors.index.json"
    index = strict_json(index_path.read_bytes())
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if shard_names != [f"model-{number:05d}-of-00004.safetensors" for number in range(1, 5)]:
        raise ValueError("Expected the four official 8B safetensors shards")
    shards = []
    for name in shard_names:
        path = model_directory / name
        record = file_records.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise ValueError(f"Missing shard or download provenance: {name}")
        size = path.stat().st_size
        expected_size = record.get("size", record.get("size_bytes"))
        if expected_size is not None and size != expected_size:
            raise ValueError(f"Downloaded shard size mismatch: {name}")
        if not any(record.get(key) for key in ("sha256", "etag", "size", "size_bytes")):
            raise ValueError(f"Shard download provenance lacks hash/etag/size: {name}")
        shards.append({"path": name, "size": size, "download_provenance": record})
    hashes[index_path.name] = sha256_file(index_path)
    # Includes tokenizer files and additional imported Python modules; no weight scan.
    for path in sorted(model_directory.iterdir()):
        if path.is_file() and path.suffix in (".py", ".json", ".jinja", ".txt"):
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large metadata file: {path.name}")
            hashes.setdefault(path.name, sha256_file(path))
    source_files = {name: digest for name, digest in hashes.items() if name.endswith(".py")}
    processor_files = {name: digest for name, digest in hashes.items() if not name.endswith(".py") and not name.startswith("model.safetensors")}
    return {
        "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256": hashlib.sha256(canonical_json(source_files).encode()).hexdigest(),
        "processor_sha256": hashlib.sha256(canonical_json(processor_files).encode()).hexdigest(),
        "index_sha256": hashes[index_path.name], "files_sha256": hashes,
        "shards": shards,
    }


class RynnValueScorer:
    def __init__(self, model_path: str, manifest_path: str, device: str = "cuda:0"):
        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor

        if not device.startswith("cuda"):
            raise ValueError("Production scorer requires a CUDA device with CPU offload")
        if model_path == MODEL_ID:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
        directory = Path(model_path).resolve()
        if not directory.is_dir():
            raise ValueError("model_path must be the locked HF id or an existing local model directory")
        artifacts = validate_artifacts(directory, Path(manifest_path))
        self.torch = torch
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self._busy = False
        self._healthy = True
        self._completed_requests = 0
        self._last_error = None
        config = AutoConfig.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
        config._attn_implementation = ATTENTION
        self.model = AutoModel.from_pretrained(
            str(directory), config=config, torch_dtype=torch.bfloat16,
            trust_remote_code=True, local_files_only=True,
        ).eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
        if getattr(self.model.config, "_attn_implementation", None) != ATTENTION:
            raise ValueError("Model did not retain the required isolation attention")
        if (getattr(self.model.config, "num_value_heads", None) != 1
                or getattr(self.model.config, "value_token_repeat", None) != 8
                or getattr(self.processor, "value_token_repeat", None) != 8
                or not getattr(self.processor, "use_meta", False)):
            raise ValueError("Loaded model/processor differs from the audited value-slot contract")
        self.model.to(device="cpu", dtype=torch.bfloat16)
        self._assert_cpu_resident()
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow", "huggingface-hub")}
        fingerprint = {
            **artifacts, "scorer_version": SCORER_VERSION,
            "scorer_sha256": sha256_file(Path(__file__)), "runtime_versions": versions,
            "attention": ATTENTION, "dtype": "bfloat16", "prefix_frames": PREFIX_FRAMES,
            "max_image_side": MAX_IMAGE_SIDE, "resize": "PIL.BICUBIC.aspect_preserving",
            "batch_size": 1, "readout": "one_absolute_head_last_slot_raw_seconds",
            "language_generation": False, "relative_fusion": False,
            "normalization": "none", "residency": "cpu_between_requests",
        }
        fingerprint["fingerprint_sha256"] = hashlib.sha256(canonical_json(fingerprint).encode()).hexdigest()
        self.fingerprint = fingerprint

    def _assert_cpu_resident(self):
        if any(tensor.device.type != "cpu" for tensor in self.model.parameters()):
            raise RuntimeError("Model parameters did not return to CPU")
        if any(tensor.device.type != "cpu" for tensor in self.model.buffers()):
            raise RuntimeError("Model buffers did not return to CPU")

    def health(self) -> dict:
        return {"ok": self._healthy, "busy": self._busy,
                "residency": ("unknown_after_failed_offload" if not self._healthy else
                              "request_in_progress" if self._busy else "cpu"),
                "device": str(self.device), "completed_requests": self._completed_requests,
                "last_error": self._last_error, "fingerprint": self.fingerprint}

    def _infer_prefix(self, images, request) -> tuple[float, float | None]:
        torch = self.torch
        prepared = self.processor.process_episode(
            instruction=request["instruction"], images=images,
            robot_description=request["robot_description"], camera_description=request["camera_description"],
        )
        ids = prepared["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Expected one tokenized prefix per forward")
        value_token_id = self.model.config.value_token_id
        relative_token_id = self.model.config.relative_value_token_id
        if (int((ids == value_token_id).sum()) != 4 * 8
                or int((ids == relative_token_id).sum()) != 3 * 8):
            raise ValueError("Tokenized prefix has incorrect absolute/relative query slot counts")
        kwargs = {
            "input_ids": ids.to(self.device).long(),
            "attention_mask": prepared["attention_mask"].to(self.device).long(),
            "pixel_values": prepared["pixel_values"].flatten(0, 1).to(self.device),
            "image_grid_thw": prepared["image_grid_thw"].flatten(0, 1).to(self.device).long(),
            "use_cache": False, "logits_to_keep": 0,
        }
        with torch.inference_mode():
            output = self.model(**kwargs)
        value, _ = extract_absolute_last_value(output.value.pred_value)
        entropy = getattr(output.value, "entropy", None)
        last_entropy = None
        if entropy is not None:
            if tuple(entropy.shape) != (1, PREFIX_FRAMES) or not torch.isfinite(entropy).all():
                raise ValueError("Unexpected absolute-head entropy shape/value")
            last_entropy = float(entropy[0, -1].detach().float().cpu())
        if output.logits is not None:
            raise ValueError("Numeric-only forward unexpectedly produced LM logits")
        return value, last_entropy

    def _offload(self):
        self.model.to("cpu")
        # Qwen's plain cached rope_deltas tensor is not a registered buffer.
        for module in self.model.modules():
            value = getattr(module, "rope_deltas", None)
            if isinstance(value, self.torch.Tensor):
                module.rope_deltas = None
        self._assert_cpu_resident()
        gc.collect()
        self.torch.cuda.empty_cache()

    def score(self, raw_request: Mapping[str, Any]) -> dict:
        request = validate_request(raw_request)
        if not self._lock.acquire(blocking=False):
            raise ScorerBusyError("Scorer is processing another episode")
        self._busy = True
        started = time.monotonic()
        moved_to_gpu = False
        result = None
        try:
            if not self._healthy:
                raise RuntimeError("Scorer is unhealthy after a failed CPU offload; restart required")
            images = decode_frames(request["frames_png_b64"])
            self.torch.cuda.set_device(self.device)
            self.torch.cuda.reset_peak_memory_stats(self.device)
            moved_to_gpu = True  # also clean partial moves if the copy raises OOM
            self.model.to(self.device)
            values, entropies = [], []
            for end_index in range(len(images)):
                prefix = [images[index] for index in prefix_indices(end_index)]
                value, entropy = self._infer_prefix(prefix, request)
                values.append(value)
                entropies.append(entropy)
            peak = int(self.torch.cuda.max_memory_allocated(self.device))
            delta = [values[index] - values[index + 1] for index in range(len(values) - 1)]
            result = {
                "ok": True, "episode_id": request.get("episode_id"),
                "remaining_seconds": values, "delta_seconds": delta,
                "diagnostics": {
                    "frame_count": len(values), "prefix_frames": PREFIX_FRAMES,
                    "resized_width": images[0].width, "resized_height": images[0].height,
                    "first_seconds": values[0], "last_seconds": values[-1],
                    "minimum_seconds": min(values), "maximum_seconds": max(values),
                    "delta_mean_seconds": sum(delta) / len(delta),
                    "delta_positive_fraction": sum(value > 0 for value in delta) / len(delta),
                    "delta_negative_fraction": sum(value < 0 for value in delta) / len(delta),
                    "last_slot_entropy": entropies, "peak_allocated_bytes": peak,
                },
                "fingerprint": self.fingerprint,
            }
            self._last_error = None
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"[:2000]
            raise
        finally:
            try:
                if moved_to_gpu:
                    try:
                        self._offload()
                    except Exception as exc:
                        self._healthy = False
                        self._last_error = f"CPU offload failed: {exc}"[:2000]
                        raise
                if result is not None:
                    result["diagnostics"]["request_seconds"] = time.monotonic() - started
                    result["diagnostics"]["residency_after_request"] = "cpu"
                    result["diagnostics"]["allocated_bytes_after_offload"] = int(self.torch.cuda.memory_allocated(self.device))
                    self._completed_requests += 1
            finally:
                self._busy = False
                self._lock.release()
        return result


def serve(scorer: RynnValueScorer, port: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RynnValueLocal/1"

        def _send_json(self, code: int, value: dict):
            body = canonical_json(value).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self._send_json(404, {"ok": False, "error": "Unknown endpoint"})
                return
            health = scorer.health()
            self._send_json(200 if health["ok"] else 503, health)

        def do_POST(self):
            if self.path != "/score":
                self._send_json(404, {"ok": False, "error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("Invalid request Content-Length")
                self.connection.settimeout(120)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Incomplete request body")
                result = scorer.score(strict_json(raw))
                self._send_json(200, result)
            except ScorerBusyError as exc:
                self._send_json(409, {"ok": False, "error": str(exc)})
            except (ValueError, UnicodeError) as exc:
                self._send_json(400, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]})

        def log_message(self, format, *args):
            print(f"[rynnvalue-http] {format % args}", flush=True)

    # Loopback only. HTTP threads allow health reads during scoring; the model
    # lock admits just one request and rejects concurrent score work.
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(canonical_json({"event": "ready", "host": "127.0.0.1", "port": port,
                              "fingerprint_sha256": scorer.fingerprint["fingerprint_sha256"]}), flush=True)
        server.serve_forever(poll_interval=0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("serve", "score-request"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--model-path", required=True)
        subparser.add_argument("--manifest", required=True)
        subparser.add_argument("--device", default="cuda:0")
        if name == "serve":
            subparser.add_argument("--port", required=True, type=int)
        else:
            subparser.add_argument("--request-json", required=True)
            subparser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if args.command == "serve" and not 1 <= args.port <= 65535:
        parser.error("port must be in [1,65535]")
    scorer = RynnValueScorer(args.model_path, args.manifest, args.device)
    if args.command == "serve":
        serve(scorer, args.port)
    else:
        request_path = Path(args.request_json)
        if request_path.stat().st_size > MAX_REQUEST_BYTES:
            raise ValueError("Request file exceeds size limit")
        result = scorer.score(strict_json(request_path.read_bytes()))
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp-{os.getpid()}")
        temporary.write_text(canonical_json(result) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        print(canonical_json({"ok": True, "output_json": str(output), "frames": len(result["remaining_seconds"])}), flush=True)


if __name__ == "__main__":
    main()
