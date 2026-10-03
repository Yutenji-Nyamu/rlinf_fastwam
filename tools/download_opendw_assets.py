#!/usr/bin/env python3
"""Bounded HTTP Range download of the three fixed official OpenDW assets.

Linux, Python standard library only. Default: 8 total connections, 32 MiB/chunk.
Completed chunks have durable SHA256 receipts; final size/SHA256 must match the
official fixed-revision LFS metadata before any file becomes a bundle asset.

  python -u tools/download_opendw_assets.py --destination /path/DW05-Robotwin \
    --proxy http://127.0.0.1:7890 --connections 8

Existing files are hash-checked, never blindly accepted or overwritten. Reuse:
  --reuse 'vae/model.pth=/path/Wan2.2_VAE.pth'

Only an operator-confirmed, stopped, sequential HTTP writer's contiguous prefix
may be imported, explicitly (never pass an Xet .incomplete file):
  --ack-contiguous-http-prefix --http-prefix 'model.pt=/exact/http.partial'
The filename/size of a Hugging Face .incomplete file cannot establish provenance.
This tool never discovers/imports HF/Xet cache partials automatically. It never
prints request URLs, redirects, signed URLs, HTTP headers, or error bodies.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import threading
import time
import urllib.error
import urllib.request


REPO = "Dexmal/DW05-Robotwin"
REVISION = "6ab5f9e2636610cba440d08264663efe70c3f761"
MIB = 1024 * 1024


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str


# Source: https://huggingface.co/api/models/Dexmal/DW05-Robotwin/tree/
# 6ab5f9e2636610cba440d08264663efe70c3f761?recursive=true
# LFS oid is the complete file's SHA256, not the Git blob oid or Xet hash.
ASSETS = (
    Asset("model.pt", 12041813433, "4ea55d5fd73aceab9917886d51717b8bdac21854a77b98e12dfed8ee34164529"),
    Asset("text_encoder/model.pth", 11361920418, "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"),
    Asset("vae/model.pth", 2818839170, "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"),
)


class DownloadError(Exception):
    """Messages from this class contain only locally constructed diagnostics."""


class Stopped(DownloadError):
    pass


def digest_file(path, start=0, length=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining is None or remaining > 0:
            block = handle.read(MIB if remaining is None else min(MIB, remaining))
            if not block:
                if remaining:
                    raise DownloadError("file shorter than hash range")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Progress:
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()

    def __call__(self, event, **values):
        message = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
        line = json.dumps(message, ensure_ascii=False)
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            print(line, flush=True)


class BundleLock:
    """Kernel lock releases automatically on process exit, including SIGKILL."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.handle = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                if not self.handle.read(1):
                    self.handle.write(" ")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise DownloadError("another range downloader owns this destination") from None
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, *_):
        self.handle.close()


def valid_complete(path, asset):
    return path.is_file() and path.stat().st_size == asset.size and digest_file(path) == asset.sha256


def reuse_asset(source, target, asset, log):
    source = Path(source)
    log("reuse_hash_started", file=asset.name, bytes=asset.size)
    if not valid_complete(source, asset):
        log("reuse_rejected", file=asset.name, reason="size_or_sha256_mismatch")
        return False
    temporary = target.with_name(target.name + ".reuse.part")
    if temporary.exists():
        raise DownloadError("existing reuse temporary file requires explicit inspection")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, MIB)
        dst.flush()
        os.fsync(dst.fileno())
    if not valid_complete(temporary, asset):
        raise DownloadError("copied reuse file failed final size/SHA256")
    os.replace(temporary, target)
    log("reuse_verified", file=asset.name, bytes=asset.size, sha256=asset.sha256)
    return True


class RangeFile:
    def __init__(self, root, asset, chunk_bytes, log, stop, prefix=None, url=None):
        self.asset, self.log, self.stop = asset, log, stop
        self.chunk_bytes, self.lock = chunk_bytes, threading.Lock()
        self.target = Path(root) / asset.name
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.part = self.target.with_name(self.target.name + ".http-ranges.part")
        self.receipt = self.target.with_name(self.target.name + ".http-ranges.json")
        self.url = url or f"https://huggingface.co/{REPO}/resolve/{REVISION}/{asset.name}?download=true"
        self.header = {"version": 1, "repo": REPO, "revision": REVISION, "file": asset.name,
                       "size": asset.size, "sha256": asset.sha256, "chunk_bytes": chunk_bytes}
        if self.receipt.exists():
            self.state = json.loads(self.receipt.read_text())
            if any(self.state.get(k) != v for k, v in self.header.items()):
                raise DownloadError("range receipt metadata/chunk size mismatch")
            if not self.part.is_file() or self.part.stat().st_size != asset.size:
                raise DownloadError("range receipt has no matching-sized partial")
            self._validate_resume()
        else:
            if self.part.exists():
                raise DownloadError("range partial lacks receipt; refusing speculative resume")
            self.state = {**self.header, "prefix_end": 0, "prefix_sha256": None, "done": {}}
            with self.part.open("xb") as handle:
                handle.truncate(asset.size)
                handle.flush()
                os.fsync(handle.fileno())
            if prefix is not None:
                self._import_prefix(Path(prefix))
            atomic_json(self.receipt, self.state)

    def _import_prefix(self, source):
        if any("xet" in part.lower() for part in source.parts):
            raise DownloadError("Xet prefix paths are explicitly forbidden")
        if source.resolve() == self.part.resolve():
            raise DownloadError("prefix cannot refer to destination range partial")
        before = source.stat()
        length = before.st_size
        if not 0 < length <= self.asset.size:
            raise DownloadError("HTTP prefix size is outside file bounds")
        digest = hashlib.sha256()
        with source.open("rb") as src, self.part.open("r+b") as dst:
            remaining = length
            while remaining:
                block = src.read(min(MIB, remaining))
                if not block:
                    raise DownloadError("HTTP prefix shrank while copying")
                dst.write(block)
                digest.update(block)
                remaining -= len(block)
            dst.flush()
            os.fsync(dst.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise DownloadError("HTTP prefix writer is still active; stop it before import")
        self.state.update(prefix_end=length, prefix_sha256=digest.hexdigest(),
                          prefix_provenance="operator_confirmed_contiguous_http_only")
        for start, end in self.chunks():
            if end <= length:
                self.state["done"][str(start)] = digest_file(self.part, start, end - start)
        self.log("http_prefix_imported", file=self.asset.name, bytes=length, sha256=digest.hexdigest())

    def _validate_resume(self):
        prefix = int(self.state["prefix_end"])
        if not 0 <= prefix <= self.asset.size:
            raise DownloadError("invalid prefix range receipt")
        if prefix and digest_file(self.part, 0, prefix) != self.state["prefix_sha256"]:
            raise DownloadError("imported HTTP prefix failed resume hash")
        expected_starts = {str(start) for start, _ in self.chunks()}
        if not set(self.state["done"]).issubset(expected_starts):
            raise DownloadError("range receipt contains unknown chunk offset")
        for start, end in self.chunks():
            key = str(start)
            if key in self.state["done"] and digest_file(self.part, start, end - start) != self.state["done"][key]:
                del self.state["done"][key]
                self.log("resume_chunk_invalidated", file=self.asset.name, start=start, end=end)
        atomic_json(self.receipt, self.state)
        self.log("resume_validated", file=self.asset.name, completed_chunks=len(self.state["done"]))

    def chunks(self):
        return ((start, min(start + self.chunk_bytes, self.asset.size))
                for start in range(0, self.asset.size, self.chunk_bytes))

    def pending(self):
        return [(start, end) for start, end in self.chunks() if str(start) not in self.state["done"]]

    def remaining_bytes(self):
        with self.lock:
            return sum(end - max(start, self.state["prefix_end"]) for start, end in self.pending())

    def fetch(self, start, end, proxy, timeout, retries):
        begin = max(start, int(self.state["prefix_end"]))
        for attempt in range(retries + 1):
            if self.stop.is_set():
                raise Stopped("download interrupted")
            started = time.monotonic()
            try:
                handlers = [] if proxy is None else [urllib.request.ProxyHandler(
                    {} if proxy == "" else {"http": proxy, "https": proxy}
                )]
                opener = urllib.request.build_opener(*handlers)
                request = urllib.request.Request(self.url, headers={
                    "Range": f"bytes={begin}-{end - 1}", "Accept-Encoding": "identity",
                    "User-Agent": "opendw-fixed-assets-range/1",
                })
                with opener.open(request, timeout=timeout) as response:
                    expected_range = f"bytes {begin}-{end - 1}/{self.asset.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise DownloadError("server did not honor the exact byte Range")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise DownloadError("compressed byte Range is unsupported")
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None and int(content_length) != end - begin:
                        raise DownloadError("Range Content-Length mismatch")
                    with self.part.open("r+b", buffering=0) as handle:
                        handle.seek(begin)
                        remaining = end - begin
                        while remaining:
                            if self.stop.is_set():
                                raise Stopped("download interrupted")
                            block = response.read(min(MIB, remaining))
                            if not block:
                                raise DownloadError("Range response ended early")
                            handle.write(block)
                            remaining -= len(block)
                        if response.read(1):
                            raise DownloadError("Range response exceeded requested length")
                        os.fsync(handle.fileno())
                digest = digest_file(self.part, start, end - start)
                with self.lock:
                    self.state["done"][str(start)] = digest
                    atomic_json(self.receipt, self.state)
                self.log("chunk_complete", file=self.asset.name, start=start, end=end,
                         network_bytes=end - begin, seconds=round(time.monotonic() - started, 3), sha256=digest)
                return
            except Stopped:
                raise
            except Exception as error:
                # Never stringify urllib exceptions: they can contain signed URLs.
                reason = str(error) if isinstance(error, DownloadError) else type(error).__name__
                status = error.code if isinstance(error, urllib.error.HTTPError) else None
                if isinstance(error, urllib.error.HTTPError):
                    error.close()
                self.log("chunk_retry" if attempt < retries else "chunk_failed", file=self.asset.name,
                         start=start, end=end, attempt=attempt + 1, error=reason, http_status=status)
                if attempt >= retries:
                    raise DownloadError(f"exhausted retries for {self.asset.name} at byte {start}") from None
                if self.stop.wait(min(30, 2 ** attempt) + random.random()):
                    raise Stopped("download interrupted")

    def finish(self):
        if self.pending():
            raise DownloadError("cannot finalize incomplete chunks")
        self.log("final_hash_started", file=self.asset.name, bytes=self.asset.size)
        if not valid_complete(self.part, self.asset):
            raise DownloadError(f"final size/SHA256 mismatch for {self.asset.name}; partial retained")
        if self.target.exists():
            raise DownloadError("destination appeared during download; refusing overwrite")
        os.replace(self.part, self.target)
        self.state["complete"] = True
        atomic_json(self.receipt, self.state)
        self.log("asset_verified", file=self.asset.name, bytes=self.asset.size, sha256=self.asset.sha256)


def named_paths(entries):
    result = {}
    names = {asset.name for asset in ASSETS}
    for entry in entries:
        name, separator, path = entry.partition("=")
        if not separator or name not in names or not path or name in result:
            raise DownloadError("asset mapping must be a unique known FILE=/absolute/path")
        candidate = Path(path)
        if not candidate.is_absolute():
            raise DownloadError("reuse/prefix sources require absolute paths")
        result[name] = candidate
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--connections", type=int, choices=range(1, 13), default=8)
    parser.add_argument("--chunk-mib", type=int, choices=range(4, 129), default=32)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--proxy", default=None, help="HTTP proxy; omitted means existing environment proxy settings")
    parser.add_argument("--files", nargs="+", choices=[asset.name for asset in ASSETS], default=[asset.name for asset in ASSETS])
    parser.add_argument("--reuse", action="append", default=[])
    parser.add_argument("--http-prefix", action="append", default=[])
    parser.add_argument("--ack-contiguous-http-prefix", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300 or not 0 <= args.retries <= 10:
        parser.error("timeout must be 1..300 seconds and retries 0..10")
    if args.http_prefix and not args.ack_contiguous_http_prefix:
        parser.error("HTTP prefix import requires --ack-contiguous-http-prefix; Xet partials are forbidden")
    args.destination.mkdir(parents=True, exist_ok=True)
    log = Progress(args.destination / "http-download-events.jsonl")
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    try:
        reuse, prefixes = named_paths(args.reuse), named_paths(args.http_prefix)
        with BundleLock(args.destination / ".http-download.lock"):
            log("download_started", pid=os.getpid(), repo=REPO, revision=REVISION,
                connections=args.connections, chunk_bytes=args.chunk_mib * MIB, files=args.files)
            downloads = []
            for asset in ASSETS:
                if asset.name not in args.files:
                    continue
                if stop.is_set():
                    raise Stopped("download interrupted")
                target = args.destination / asset.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    log("existing_hash_started", file=asset.name, bytes=asset.size)
                    if not valid_complete(target, asset):
                        raise DownloadError(f"existing {asset.name} failed size/SHA256; preserved")
                    log("existing_verified", file=asset.name, bytes=asset.size, sha256=asset.sha256)
                    continue
                if asset.name in reuse and reuse_asset(reuse[asset.name], target, asset, log):
                    continue
                downloads.append(RangeFile(args.destination, asset, args.chunk_mib * MIB, log, stop,
                                           prefix=prefixes.get(asset.name)))
            # Round-robin submission prevents the largest file monopolizing the queue.
            jobs = []
            active = [(item, iter(item.pending())) for item in downloads]
            while active:
                next_active = []
                for item, queue in active:
                    segment = next(queue, None)
                    if segment is not None:
                        jobs.append((item, *segment))
                        next_active.append((item, queue))
                active = next_active
            initial_remaining = sum(item.remaining_bytes() for item in downloads)
            log("download_plan", pending_chunks=len(jobs), remaining_network_bytes=initial_remaining)
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=args.connections) as pool:
                pending = {pool.submit(item.fetch, start, end, args.proxy, args.timeout, args.retries)
                           for item, start, end in jobs}
                try:
                    while pending:
                        finished, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                        for future in finished:
                            future.result()
                        remaining_bytes = sum(item.remaining_bytes() for item in downloads)
                        elapsed = time.monotonic() - started
                        if not finished:
                            log("download_heartbeat", pending_chunks=len(pending), remaining_network_bytes=remaining_bytes,
                                completed_network_bytes=initial_remaining - remaining_bytes,
                                average_mib_s=round((initial_remaining - remaining_bytes) / MIB / max(elapsed, 0.001), 3))
                        if stop.is_set():
                            raise Stopped("download interrupted")
                except BaseException:
                    stop.set()
                    for future in pending:
                        future.cancel()
                    raise
            for item in downloads:
                item.finish()
            log("all_assets_verified", revision=REVISION, files=args.files)
            return 0
    except Stopped:
        log("download_stopped", resumable=True)
        return 130
    except Exception as error:
        reason = str(error) if isinstance(error, DownloadError) else type(error).__name__
        log("download_failed", error=reason, resumable=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
