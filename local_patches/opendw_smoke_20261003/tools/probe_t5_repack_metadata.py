#!/usr/bin/env python3
"""Read-only CPU feasibility probe for byte-identical T5 archive reconstruction.

Reads safetensors JSON, the stopped official download prefix, and bounded HTTP
ZIP metadata ranges. Imports neither torch nor safetensors, loads no tensors,
and never writes/replaces either model. The pickle reader permits only inert
storage/tensor metadata constructors; all other globals fail closed.
"""
import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import struct
import time
import urllib.error
import urllib.request
import zipfile


DTYPES = {
    "BFloat16Storage": ("BF16", 2), "HalfStorage": ("F16", 2),
    "FloatStorage": ("F32", 4), "DoubleStorage": ("F64", 8),
    "LongStorage": ("I64", 8), "IntStorage": ("I32", 4),
    "ShortStorage": ("I16", 2), "CharStorage": ("I8", 1),
    "ByteStorage": ("U8", 1), "BoolStorage": ("BOOL", 1),
}


def rebuild_tensor(storage, offset, shape, stride, *unused):
    return {"storage": storage, "offset": offset,
            "shape": list(shape), "stride": list(stride)}


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name in DTYPES:
            return ("storage_type", name)
        if module == "torch._utils" and name in {
                "_rebuild_tensor", "_rebuild_tensor_v2"}:
            return rebuild_tensor
        raise ValueError(f"Unapproved pickle global: {module}.{name}")

    def persistent_load(self, item):
        if not (isinstance(item, tuple) and len(item) == 5 and
                item[0] == "storage" and isinstance(item[1], tuple) and
                item[1][0] == "storage_type"):
            raise ValueError(f"Unrecognized persistent metadata: {item!r}")
        _, storage_type, key, device, numel = item
        dtype, itemsize = DTYPES[storage_type[1]]
        return {"key": str(key), "dtype": dtype, "itemsize": itemsize,
                "device": str(device), "numel": int(numel)}


class PrefixAndHTTP(io.RawIOBase):
    def __init__(self, prefix, url, total_size, proxy, cache_dir=None):
        self.local = open(prefix, "rb")
        self.local_size = os.fstat(self.local.fileno()).st_size
        self.url, self.total_size, self.pos = url, total_size, 0
        self.http_bytes, self.ranges = 0, []
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_identity = hashlib.sha256(f"{url}\n{total_size}".encode()).hexdigest()[:24]
        handlers = [urllib.request.ProxyHandler(
            {"http": proxy, "https": proxy} if proxy else {})]
        self.opener = urllib.request.build_opener(*handlers)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else (
            self.pos + offset if whence == 1 else self.total_size + offset)
        if target < 0:
            raise ValueError("Negative seek")
        self.pos = target
        return target

    def read(self, size=-1):
        size = self.total_size - self.pos if size < 0 else size
        size = min(size, self.total_size - self.pos)
        if size <= 0:
            return b""
        if size > 2 * 1024 * 1024:
            raise ValueError(f"Metadata read too large: {size}")
        start = self.pos
        if start + size <= self.local_size:
            self.local.seek(start)
            data = self.local.read(size)
            if len(data) != size:
                raise ValueError("Stopped prefix changed/truncated")
        else:
            if self.http_bytes + size > 8 * 1024 * 1024:
                raise ValueError("HTTP metadata budget exceeded")
            end = start + size - 1
            cache_path = (self.cache_dir / f"{self.cache_identity}-{start}-{end}.bin"
                          if self.cache_dir else None)
            data, cached = None, False
            if cache_path and cache_path.is_file() and cache_path.with_suffix(".json").is_file():
                try:
                    candidate = cache_path.read_bytes()
                    receipt = json.loads(cache_path.with_suffix(".json").read_text())
                except OSError as error:
                    raise RuntimeError("Metadata cache read failed") from error
                if (len(candidate) != size or receipt["sha256"] != hashlib.sha256(candidate).hexdigest()
                        or receipt["start"] != start or receipt["size"] != size):
                    raise RuntimeError("Metadata range cache validation failed")
                data, cached = candidate, True
            if data is None:
                for attempt in range(1, 5):
                    try:
                        req = urllib.request.Request(self.url, headers={
                            "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity",
                            "User-Agent": "T5MetadataProbe/2"})
                        with self.opener.open(req, timeout=30) as response:
                            expected = f"bytes {start}-{end}/{self.total_size}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected:
                                raise RuntimeError(f"HTTP did not honor exact metadata Range {start}-{end}")
                            data = response.read(size + 1)
                            if len(data) != size:
                                raise RuntimeError("HTTP metadata range length mismatch")
                        break
                    except (urllib.error.URLError, OSError) as error:
                        if attempt == 4:
                            # zipfile catches OSError and would misleadingly report BadZipFile.
                            raise RuntimeError(f"Metadata HTTP range {start}-{end} failed after 4 attempts: "
                                               f"{type(error).__name__}: {error}") from error
                        print(f"metadata_range_retry start={start} size={size} attempt={attempt} "
                              f"error={type(error).__name__}", flush=True)
                        time.sleep(min(2 ** (attempt - 1), 4))
                self.http_bytes += size
                if cache_path:
                    receipt = {"start": start, "size": size, "total_size": self.total_size,
                               "sha256": hashlib.sha256(data).hexdigest()}
                    temporary = cache_path.with_suffix(f".part-{os.getpid()}")
                    temporary.write_bytes(data)
                    cache_path.with_suffix(".json").write_text(json.dumps(receipt) + "\n")
                    os.replace(temporary, cache_path)
            self.ranges.append({"start": start, "size": size,
                                "cache_hit": cached,
                                "sha256": hashlib.sha256(data).hexdigest()})
        self.pos += size
        return data

    def close(self):
        self.local.close()
        super().close()


def contiguous_stride(shape):
    out, stride = [], 1
    for size in reversed(shape):
        out.append(stride)
        stride *= size
    return list(reversed(out))


def prefix_storage_start(prefix, info):
    """Only inspect a local header when it is already in the stopped prefix."""
    if info.header_offset + 30 > prefix.local_size:
        return None
    prefix.local.seek(info.header_offset)
    header = prefix.local.read(30)
    values = struct.unpack("<4s5H3I2H", header)
    if values[0] != b"PK\x03\x04" or values[3] != zipfile.ZIP_STORED:
        raise ValueError("Unexpected local ZIP header")
    return info.header_offset + 30 + values[-2] + values[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sf", "prefix", "url", "sha256", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--metadata-cache", help="Independent directory for bounded HTTP range bytes")
    args = parser.parse_args()
    if Path(args.out).resolve() in {Path(args.sf).resolve(), Path(args.prefix).resolve()}:
        raise ValueError("Report must not overwrite an input")
    with open(args.sf, "rb") as sf:
        header_len, = struct.unpack("<Q", sf.read(8))
        if not 1 <= header_len <= 2 * 1024 * 1024:
            raise ValueError("Unexpected safetensors header length")
        header_bytes = sf.read(header_len)
        sf_meta = json.loads(header_bytes)
        sf_data_base = 8 + header_len
        sf_size = os.fstat(sf.fileno()).st_size
        tensors = {key: value for key, value in sf_meta.items() if key != "__metadata__"}
        if sf_data_base + max(row["data_offsets"][1] for row in tensors.values()) != sf_size:
            raise ValueError("Safetensors file/header length mismatch")
        cache_dir = args.metadata_cache or str(Path(args.out).with_suffix(".ranges"))
        with PrefixAndHTTP(args.prefix, args.url, args.size, args.proxy, cache_dir) as source:
            prefix_initial_size = source.local_size
            with zipfile.ZipFile(source) as archive:
                entries = archive.infolist()
                pickles = [entry for entry in entries if entry.filename.endswith("/data.pkl")]
                if len(pickles) != 1 or pickles[0].file_size > 2 * 1024 * 1024:
                    raise ValueError("Expected one small data.pkl")
                data_pickle = archive.read(pickles[0])
                state = MetadataUnpickler(io.BytesIO(data_pickle)).load()
                if not isinstance(state, (dict, collections.OrderedDict)):
                    raise ValueError("Expected state dictionary")
                root = pickles[0].filename[:-len("data.pkl")]
                storage_entries = {entry.filename[len(root + "data/"):]: entry
                                   for entry in entries if entry.filename.startswith(root + "data/")}
                issues, rows, samples = [], [], []
                seen_storages = set()
                for key, value in state.items():
                    if not isinstance(value, dict) or "storage" not in value:
                        raise ValueError(f"Unexpected tensor metadata at {key}")
                    storage = value["storage"]
                    storage_key = storage["key"]
                    info, st = storage_entries.get(storage_key), tensors.get(key)
                    reasons = []
                    if st is None:
                        reasons.append("missing safetensors key")
                    else:
                        if st["dtype"] != storage["dtype"] or st["shape"] != value["shape"]:
                            reasons.append("dtype or shape differs")
                        if st["data_offsets"][1] - st["data_offsets"][0] != storage["numel"] * storage["itemsize"]:
                            reasons.append("storage length differs")
                    if value["offset"] != 0 or value["stride"] != contiguous_stride(value["shape"]):
                        reasons.append("view or noncontiguous tensor")
                    if math.prod(value["shape"]) != storage["numel"]:
                        reasons.append("tensor does not cover entire storage")
                    if storage_key in seen_storages:
                        reasons.append("shared storage")
                    seen_storages.add(storage_key)
                    if info is None or info.compress_type != zipfile.ZIP_STORED or info.file_size != storage["numel"] * storage["itemsize"]:
                        reasons.append("ZIP storage layout/size differs")
                    start = prefix_storage_start(source, info) if info else None
                    row = {"key": key, **value, "sf": st,
                           "zip": None if info is None else {"name": info.filename,
                           "header_offset": info.header_offset, "data_start_in_prefix": start,
                           "size": info.file_size, "crc32": f"{info.CRC:08x}",
                           "flags": info.flag_bits, "compression": info.compress_type},
                           "issues": reasons}
                    rows.append(row)
                    if reasons:
                        issues.append({"key": key, "reasons": reasons})
                    if not reasons and start is not None and len(samples) < 12:
                        available = min(info.file_size, source.local_size - start)
                        if available > 0:
                            n = min(256 * 1024, available)
                            positions = sorted({0, max(0, available // 2 - n // 2), available - n})
                            for offset in positions:
                                source.local.seek(start + offset)
                                original = source.local.read(n)
                                sf.seek(sf_data_base + st["data_offsets"][0] + offset)
                                existing = sf.read(n)
                                equal = original == existing
                                samples.append({"key": key, "storage_offset": offset,
                                                "size": n, "equal": equal,
                                                "official_sha256": hashlib.sha256(original).hexdigest(),
                                                "sf_sha256": hashlib.sha256(existing).hexdigest()})
                                if not equal:
                                    issues.append({"key": key, "reasons": ["prefix bytes differ"]})
                if set(state) != set(tensors):
                    issues.append({"key_set": {"sf_only": sorted(set(tensors) - set(state)),
                                                "pth_only": sorted(set(state) - set(tensors))}})
                if seen_storages != set(storage_entries):
                    issues.append({"unmapped_zip_storages": sorted(set(storage_entries) - seen_storages)})
                report = {
                    "status": "metadata_compatible_not_full_verification" if not issues else "incompatible",
                    "official_sha256_required": args.sha256,
                    "official_size": args.size, "safetensors_size": sf_size,
                    "safetensors_path": args.sf, "prefix_path": args.prefix,
                    "prefix_size": prefix_initial_size,
                    "sf_header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                    "data_pickle_sha256": hashlib.sha256(data_pickle).hexdigest(),
                    "tensor_count": len(state), "storage_count": len(storage_entries),
                    "zip_entry_count": len(entries),
                    "sf_data_base": sf_data_base,
                    "non_storage_bytes": args.size - sum(entry.file_size for entry in storage_entries.values()),
                    "http_bytes": source.http_bytes, "http_ranges": source.ranges,
                    "issues": issues, "prefix_samples": samples, "tensors": rows,
                    "zip_entries": [{"name": entry.filename, "offset": entry.header_offset,
                                     "size": entry.file_size, "compressed_size": entry.compress_size,
                                     "flags": entry.flag_bits, "compression": entry.compress_type,
                                     "crc32": f"{entry.CRC:08x}"} for entry in entries],
                    "central_directory_start": archive.start_dir,
                }
                if os.fstat(source.local.fileno()).st_size != prefix_initial_size:
                    raise ValueError("Download prefix changed during probe")
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps({key: report[key] for key in (
        "status", "tensor_count", "storage_count", "zip_entry_count",
        "non_storage_bytes", "http_bytes", "issues", "prefix_samples")}, indent=2))


if __name__ == "__main__":
    main()
