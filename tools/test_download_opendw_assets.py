"""Run on the server CPU: python -B tools/test_download_opendw_assets.py."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from download_opendw_assets import Asset, DownloadError, RangeFile


class RangeDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.data = bytes(range(251)) * 101
        self.asset = Asset("model.pt", len(self.data), hashlib.sha256(self.data).hexdigest())
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.events, self.requests = [], []
        self.stop = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                header = self.headers["Range"]
                begin, end = [int(value) for value in header[6:].split("-")]
                outer.requests.append((begin, end))
                data = outer.data[begin:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {begin}-{end}/{len(outer.data)}")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/asset"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary.cleanup()

    def log(self, event, **values):
        self.events.append((event, values))

    def make(self, **kwargs):
        return RangeFile(self.root / "bundle", self.asset, 4096, self.log, self.stop, url=self.url, **kwargs)

    def fetch_pending(self, item):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(item.fetch, begin, end, "", 5, 0) for begin, end in item.pending()]
            for future in futures:
                future.result()

    def test_http_prefix_and_parallel_ranges_finalize_exact_file(self):
        prefix = self.root / "known-http.partial"
        prefix.write_bytes(self.data[:6000])
        item = self.make(prefix=prefix)
        self.assertEqual(item.remaining_bytes(), len(self.data) - 6000)
        self.fetch_pending(item)
        self.assertTrue(all(begin >= 6000 for begin, _ in self.requests))
        item.finish()
        self.assertEqual(item.target.read_bytes(), self.data)
        self.assertEqual(prefix.read_bytes(), self.data[:6000])

    def test_resume_verifies_chunk_hash_and_redownloads_only_corrupt_or_missing(self):
        item = self.make()
        segments = list(item.chunks())
        item.fetch(*segments[0], "", 5, 0)
        item.fetch(*segments[1], "", 5, 0)
        with item.part.open("r+b") as handle:
            handle.seek(segments[1][0])
            handle.write(b"corrupt")
        self.requests.clear()
        resumed = self.make()
        self.assertNotIn(segments[0], resumed.pending())
        self.assertIn(segments[1], resumed.pending())
        self.fetch_pending(resumed)
        resumed.finish()
        self.assertTrue(all(begin >= 4096 for begin, _ in self.requests))
        self.assertEqual(resumed.target.read_bytes(), self.data)

    def test_xet_prefix_is_rejected_and_final_hash_blocks_bad_complete_file(self):
        prefix = self.root / "xet" / "anything.incomplete"
        prefix.parent.mkdir()
        prefix.write_bytes(self.data[:5000])
        with self.assertRaisesRegex(DownloadError, "Xet"):
            self.make(prefix=prefix)
        # Separate clean destination; deliberately wrong source body with valid
        # Range headers must never be promoted to a completed bundle asset.
        item = RangeFile(self.root / "bad-server-bundle", self.asset, 4096, self.log, self.stop, url=self.url)
        self.data = b"z" * len(self.data)
        self.fetch_pending(item)
        with self.assertRaisesRegex(DownloadError, "SHA256 mismatch"):
            item.finish()
        self.assertFalse(item.target.exists())
        self.assertTrue(item.part.exists())


if __name__ == "__main__":
    unittest.main()
