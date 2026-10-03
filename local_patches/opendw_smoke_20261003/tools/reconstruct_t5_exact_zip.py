#!/usr/bin/env python3
"""Rebuild this official T5 ZIP from identical local SF bytes, with exact SHA.

Only small original metadata ranges use HTTP (8 workers maximum). No torch or
model deserialization, no bundle replacement, and no changes to either source
or other download. The output must be a new independent path. verified.json is
written only after every storage CRC, full size, and read-back SHA match.
Requires adjacent probe_t5_repack_metadata.py for its bounded range reader.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
import zlib

from probe_t5_repack_metadata import PrefixAndHTTP


EXPECTED_SIZE = 11361920418
EXPECTED_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
EXPECTED_URL = ("https://huggingface.co/Dexmal/DW05-Robotwin/resolve/"
                "6ab5f9e2636610cba440d08264663efe70c3f761/text_encoder/model.pth")
CHUNK = 8 * 1024 * 1024


def note(phase, **fields):
    print(json.dumps({"phase": phase, **fields}, sort_keys=True), flush=True)


class MetadataBank:
    def __init__(self, prefix, url, size, proxy, cache_dir, workers):
        self.prefix, self.url, self.size = prefix, url, size
        self.proxy, self.cache_dir, self.workers = proxy, cache_dir, workers
        self.prefix_stat = Path(prefix).stat()
        self.prefix_file = open(prefix, "rb")
        self.segments = []
        self.http_bytes = 0
        self.fetch_records = []
        self.network_budget = 4 * 1024 * 1024
        self.cache_identity = hashlib.sha256(f"{url}\n{size}".encode()).hexdigest()[:24]
        cache_dir.mkdir(parents=True, exist_ok=True)
        for receipt_path in sorted(cache_dir.glob(self.cache_identity + "-*.json")):
            receipt = json.loads(receipt_path.read_text())
            binary = receipt_path.with_suffix(".bin")
            if not binary.is_file():
                continue
            start, count = int(receipt["start"]), int(receipt["size"])
            if not 0 <= start < size or not 0 < count <= 2 * 1024 * 1024 or start + count > size:
                raise RuntimeError("Metadata cache bounds invalid")
            data = binary.read_bytes()
            if len(data) != count or hashlib.sha256(data).hexdigest() != receipt["sha256"]:
                raise RuntimeError("Metadata cache SHA/length invalid")
            self.segments.append((start, start + count, data))
        if sum(len(row[2]) for row in self.segments) > 16 * 1024 * 1024:
            raise RuntimeError("Unexpectedly large metadata cache")

    def covered(self, start, size):
        end, position, pieces = start + size, start, []
        while position < end:
            if position < self.prefix_stat.st_size:
                stop = min(end, self.prefix_stat.st_size)
                self.prefix_file.seek(position)
                piece = self.prefix_file.read(stop - position)
                if len(piece) != stop - position:
                    raise RuntimeError("Stopped prefix truncated")
            else:
                choices = [row for row in self.segments if row[0] <= position < row[1]]
                if not choices:
                    return None
                lower, upper, data = max(choices, key=lambda row: row[1])
                stop = min(end, upper)
                piece = data[position - lower:stop - lower]
            pieces.append(piece)
            position = stop
        return b"".join(pieces)

    def fetch_many(self, requests, phase):
        needed = list(dict.fromkeys((start, count) for start, count in requests
                                    if self.covered(start, count) is None))
        if not needed:
            return
        if any(not 0 <= start < self.size or not 0 < count <= 2 * 1024 * 1024 or
               start + count > self.size for start, count in needed):
            raise RuntimeError("HTTP metadata bounds invalid")
        if self.http_bytes + sum(count for _, count in needed) > self.network_budget:
            raise RuntimeError("Total metadata HTTP budget exceeded")

        def fetch_one(item):
            start, count = item
            with PrefixAndHTTP(self.prefix, self.url, self.size, self.proxy, self.cache_dir) as reader:
                reader.seek(start)
                data = reader.read(count)
                return start, count, data, reader.http_bytes, reader.ranges

        started = time.monotonic()
        note(phase, ranges=len(needed), workers=self.workers,
             maximum_bytes=sum(count for _, count in needed))
        executor = ThreadPoolExecutor(max_workers=self.workers)
        futures = {executor.submit(fetch_one, item): item for item in needed}
        done = 0
        try:
            for future in as_completed(futures, timeout=900):
                start, count, data, transferred, records = future.result()
                self.segments.append((start, start + count, data))
                self.http_bytes += transferred
                self.fetch_records.extend(records)
                done += 1
                if done % 10 == 0 or done == len(needed):
                    note(phase, completed=done, total=len(needed),
                         http_bytes=self.http_bytes, elapsed=round(time.monotonic() - started, 1))
        finally:
            # Cancel pending requests on any error; active requests retain the
            # range reader's 4 x 30-second bounded retry policy.
            executor.shutdown(wait=True, cancel_futures=True)

    def unchanged(self):
        now = os.fstat(self.prefix_file.fileno())
        if (now.st_size, now.st_mtime_ns) != (self.prefix_stat.st_size, self.prefix_stat.st_mtime_ns):
            raise RuntimeError("Stopped official prefix changed")

    def close(self):
        self.prefix_file.close()


def sf_header(stream):
    stream.seek(0)
    count, = struct.unpack("<Q", stream.read(8))
    if not 1 <= count <= 2 * 1024 * 1024:
        raise RuntimeError("SF JSON header bounds invalid")
    data = stream.read(count)
    values = json.loads(data)
    return count + 8, data, {key: value for key, value in values.items() if key != "__metadata__"}


def local_storage_start(bank, row):
    info = row["zip"]
    offset = int(info["header_offset"])
    header = bank.covered(offset, 30)
    if header is None:
        raise RuntimeError("ZIP local header unavailable")
    signature, version, flags, compression, modtime, moddate, crc, compressed, size, namelen, extralen = struct.unpack("<4s5H3I2H", header)
    if signature != b"PK\x03\x04" or compression != 0 or flags & 1 or flags != info["flags"]:
        raise RuntimeError("ZIP local header flags/compression differ")
    if namelen + extralen > 4096:
        raise RuntimeError("Unexpected ZIP local header length")
    following = bank.covered(offset + 30, namelen + extralen)
    if following is None:
        bank.fetch_many([(offset + 30, namelen + extralen)], "extended_header")
        following = bank.covered(offset + 30, namelen + extralen)
    name = following[:namelen].decode("utf-8" if flags & 0x800 else "cp437")
    if name != info["name"]:
        raise RuntimeError("ZIP local filename differs from central directory")
    start = offset + 30 + namelen + extralen
    if info["data_start_in_prefix"] is not None and start != info["data_start_in_prefix"]:
        raise RuntimeError("ZIP local payload offset differs from metadata probe")
    if not 0 <= start < bank.size or start + info["size"] > bank.size:
        raise RuntimeError("ZIP storage payload outside official file")
    return start


def reconstruct(args):
    started = time.monotonic()
    report = json.loads(Path(args.report).read_text())
    if (report["status"] != "metadata_compatible_not_full_verification" or report["issues"] or
            report["official_size"] != EXPECTED_SIZE or report["official_sha256_required"] != EXPECTED_SHA256 or
            not report["prefix_samples"] or not all(row["equal"] for row in report["prefix_samples"])):
        raise RuntimeError("Report is not the approved compatible official T5 metadata")
    if args.url != EXPECTED_URL:
        raise RuntimeError("Only the fixed official T5 URL is accepted")
    sf_path = Path(args.sf or report["safetensors_path"])
    prefix_path = Path(args.prefix or report["prefix_path"])
    output = Path(args.out)
    if output.exists() or output.resolve() in {sf_path.resolve(), prefix_path.resolve(), Path(args.report).resolve()}:
        raise RuntimeError("Output must be a new independent file")
    output.parent.mkdir(parents=True, exist_ok=True)
    verified_path = output.parent / "verified.json"
    if verified_path.exists():
        raise RuntimeError("Output directory already has a verification receipt")
    if shutil.disk_usage(output.parent).free < EXPECTED_SIZE + 64 * 1024 * 1024:
        raise RuntimeError("Insufficient free disk space for independent output")
    cache = Path(args.metadata_cache or str(Path(args.report).with_suffix(".ranges")))
    bank = MetadataBank(str(prefix_path), args.url, EXPECTED_SIZE, args.proxy, cache, args.workers)
    try:
        if bank.prefix_stat.st_size != report["prefix_size"]:
            raise RuntimeError("Stopped prefix size differs from probe")
        with sf_path.open("rb") as sf:
            sf_stat = os.fstat(sf.fileno())
            data_base, header_bytes, tensors = sf_header(sf)
            if (sf_stat.st_size != report["safetensors_size"] or data_base != report["sf_data_base"] or
                    hashlib.sha256(header_bytes).hexdigest() != report["sf_header_sha256"]):
                raise RuntimeError("SF source differs from metadata probe")
            rows = report["tensors"]
            if len(rows) != report["storage_count"] or set(tensors) != {row["key"] for row in rows}:
                raise RuntimeError("Tensor/storage key sets differ")
            for row in rows:
                info, st = row["zip"], tensors[row["key"]]
                if (row["issues"] or st != row["sf"] or info["compression"] != 0 or
                        st["data_offsets"][1] - st["data_offsets"][0] != info["size"]):
                    raise RuntimeError("Tensor/storage metadata differs")
            windows = [(max(0, row["zip"]["header_offset"] - 32),
                        min(EXPECTED_SIZE, row["zip"]["header_offset"] + 512) -
                        max(0, row["zip"]["header_offset"] - 32)) for row in rows]
            bank.fetch_many(windows, "fetch_original_local_headers")
            placements = sorted([(local_storage_start(bank, row), row) for row in rows], key=lambda value: value[0])
            gaps, position = [], 0
            for start, row in placements:
                if start < position:
                    raise RuntimeError("Overlapping ZIP storages")
                gaps.append((position, start - position))
                position = start + row["zip"]["size"]
            gaps.append((position, EXPECTED_SIZE - position))
            if sum(size for _, size in gaps) != report["non_storage_bytes"]:
                raise RuntimeError("ZIP non-storage byte count differs")
            bank.fetch_many([(start, count) for start, count in gaps if count], "fetch_original_remaining_gaps")
            if any(count and bank.covered(start, count) is None for start, count in gaps):
                raise RuntimeError("Original metadata bytes do not completely cover every gap")
            bank.unchanged()
            stream_hash, written, verified_storages = hashlib.sha256(), 0, []
            note("write_independent_output", path=str(output), storages=len(placements), http_bytes=bank.http_bytes)
            with output.open("xb") as destination:
                def write(data):
                    nonlocal written
                    destination.write(data)
                    stream_hash.update(data)
                    written += len(data)

                for index, (start, row) in enumerate(placements):
                    gap_start, gap_size = gaps[index]
                    if written != gap_start:
                        raise RuntimeError("Output gap position differs")
                    if gap_size:
                        write(bank.covered(gap_start, gap_size))
                    if written != start:
                        raise RuntimeError("Output storage position differs")
                    sf.seek(data_base + row["sf"]["data_offsets"][0])
                    remaining, crc = row["zip"]["size"], 0
                    while remaining:
                        data = sf.read(min(CHUNK, remaining))
                        if not data:
                            raise RuntimeError("SF tensor bytes truncated")
                        write(data)
                        crc = zlib.crc32(data, crc)
                        remaining -= len(data)
                    if crc != int(row["zip"]["crc32"], 16):
                        raise RuntimeError(f"Official ZIP storage CRC differs: {row['key']}")
                    verified_storages.append({"key": row["key"], "crc32": f"{crc:08x}"})
                    if (index + 1) % 10 == 0 or index + 1 == len(placements):
                        note("write_independent_output", storages_done=index + 1,
                             storages_total=len(placements), bytes_written=written)
                final_start, final_size = gaps[-1]
                if written != final_start:
                    raise RuntimeError("Output tail position differs")
                if final_size:
                    write(bank.covered(final_start, final_size))
                destination.flush()
                os.fsync(destination.fileno())
            now = os.fstat(sf.fileno())
            if (now.st_size, now.st_mtime_ns) != (sf_stat.st_size, sf_stat.st_mtime_ns):
                raise RuntimeError("SF source changed during reconstruction")
            bank.unchanged()
            if written != EXPECTED_SIZE or output.stat().st_size != EXPECTED_SIZE or stream_hash.hexdigest() != EXPECTED_SHA256:
                raise RuntimeError(f"Reconstructed file size/SHA mismatch: size={written} sha256={stream_hash.hexdigest()}")
        note("verify_output_readback", size=EXPECTED_SIZE)
        readback_hash = hashlib.sha256()
        with output.open("rb") as stream:
            while data := stream.read(CHUNK):
                readback_hash.update(data)
        if readback_hash.hexdigest() != EXPECTED_SHA256 or output.stat().st_size != EXPECTED_SIZE:
            raise RuntimeError("Output read-back SHA/size mismatch")
        receipt = {"verified": True, "path": str(output), "size": EXPECTED_SIZE,
                   "sha256": EXPECTED_SHA256, "readback_sha256": readback_hash.hexdigest(),
                   "official_url": args.url, "source_sf": str(sf_path), "source_prefix": str(prefix_path),
                   "metadata_report": args.report, "storage_count": len(verified_storages),
                   "storage_crc32": verified_storages, "additional_http_bytes": bank.http_bytes,
                   "metadata_http_ranges": bank.fetch_records,
                   "elapsed_seconds": round(time.monotonic() - started, 3),
                   "bundle_replaced": False}
        with verified_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        note("verified", path=str(output), sha256=EXPECTED_SHA256,
             size=EXPECTED_SIZE, receipt=str(verified_path), elapsed_seconds=receipt["elapsed_seconds"])
    finally:
        bank.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sf")
    parser.add_argument("--prefix")
    parser.add_argument("--url", default=EXPECTED_URL)
    parser.add_argument("--proxy", default="http://127.0.0.1:7890")
    parser.add_argument("--metadata-cache")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be 1..8")
    try:
        reconstruct(args)
    except Exception as error:
        note("failed_unverified", error_type=type(error).__name__, error=str(error),
             output=args.out, verified_receipt_written=False)
        raise


if __name__ == "__main__":
    main()
