"""Archive project-owned lightweight evidence with explicit exclusions and redaction.

This script creates a new archive/staging tree only. Original sources are untouched.
"""
import argparse, collections, datetime, getpass, gzip, hashlib, io, json, os, re, stat, subprocess, tarfile, zipfile
from pathlib import Path, PurePosixPath

DATA = {'.pt', '.pth', '.distcp', '.safetensors', '.bin', '.npy', '.npz', '.pkl', '.pickle', '.h5', '.hdf5', '.parquet', '.arrow', '.db', '.sqlite', '.sqlite3'}
VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.gif'}
CACHE = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.ruff_cache', '.mypy_cache', '.venv', 'venv'}
TEXT = {'.md', '.txt', '.log', '.json', '.jsonl', '.yaml', '.yml', '.toml', '.csv', '.tsv', '.sh', '.py', '.patch', '.diff', '.html', '.xml', '.ini', '.cfg', '.rst', '.js', '.ts', '.css', '.ps1', '.sql', '.ipynb', '.c', '.cc', '.cpp', '.h', '.hpp', '.cu', '.cuh'}
SECRET_FILES = {'.env', 'id_rsa', 'id_ed25519', 'credentials', 'credentials.json', 'cookies.txt', 'auth.json', 'known_hosts', 'authorized_keys'}
TOKENS = [re.compile(r'\b(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{16,}'), re.compile(r'\bsk-[A-Za-z0-9_-]{20,}'),
          re.compile(r'(?i)(?:password|passwd|密码)\s*[=:：]\s*[\"\']?([^\s\"\'`,;{}<>]{4,})'),
          re.compile(r'(?i)https?://[^/\s:@]+:[^/\s@]+@')]

class Archive:
    def __init__(self, destination, secret=''):
        self.dest = Path(destination)
        self.dest.mkdir(parents=True, exist_ok=False)
        self.secret = secret
        self.rows = []
        self.counts = collections.Counter()
        self.bytes = collections.Counter()
        self.names = set()
    def omit(self, source, size, reason):
        self.rows.append({'source': str(source), 'bytes': size, 'disposition': reason})
        self.counts[reason] += 1; self.bytes[reason] += size
    def classify(self, name, size):
        p = PurePosixPath(name.replace('\\', '/'))
        if p.name.lower() in SECRET_FILES or p.suffix.lower() in {'.pem', '.key', '.p12', '.pfx'}:
            return 'credential_file_excluded'
        if p.name == '.git' or any(c in CACHE for c in p.parts):
            return 'vcs_or_cache_excluded'
        if p.suffix.lower() in DATA:
            if p.name in {'learner.pt', 'dvac.pt'} and size < 1024**2:
                return None
            return 'bulk_tensor_replay_dataset_excluded'
        if p.suffix.lower() in VIDEO:
            return 'video_excluded'
        if any(c.lower() in {'success_data', 'replay', 'replay_buffer', 'replay_data', 'raw_frames', 'frames', 'image_obs'} for c in p.parts):
            return 'bulk_raw_data_excluded'
        if size > 50*1024**2 and p.suffix.lower() not in TEXT:
            return 'large_binary_excluded'
        if p.suffix.lower() in {'.pyc', '.pyo', '.so', '.dll', '.exe', '.a', '.o', '.whl', '.pack', '.idx', '.bundle', '.7z', '.rar', '.zst'}:
            return 'compiled_dependency_excluded'
        return None
    def add(self, source, name, data):
        if not data:
            text = ''
        else:
            try: text = data.decode('utf-8', errors='replace' if PurePosixPath(name).suffix.lower() in TEXT else 'strict') if b'\0' not in data[:8192] else None
            except UnicodeDecodeError: text = None
        original_sha = hashlib.sha256(data).hexdigest(); redactions = 0
        if text is not None:
            if re.search(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----', text):
                return self.omit(source, len(data), 'private_key_content_excluded')
            if self.secret:
                redactions += text.count(self.secret); text = text.replace(self.secret, '[REDACTED]')
            for pattern in TOKENS:
                if 'password' in pattern.pattern:
                    def repl(m):
                        value = m.group(1)
                        # Preserve obvious placeholders / source variable names.
                        if value in {'None', 'null', 'False', 'True', 'os.environ', 'getpass.getpass', 'args.password', 'REDACTED'} or value.startswith(('$', '[', '*')):
                            return m.group(0)
                        return m.group(0).replace(value, '[REDACTED]')
                    old = text; text = pattern.sub(repl, text); redactions += int(old != text)
                else:
                    text, n = pattern.subn('[REDACTED]', text); redactions += n
            data = text.encode('utf-8')
        transform = 'redacted_copy' if redactions else 'unchanged'
        if len(data) >= 10*1024**2 and text is not None:
            data = gzip.compress(data, compresslevel=6, mtime=0)
            name += '.gz'; transform += '+lossless_gzip'
        if len(data) >= 50*1024**2:
            return self.omit(source, len(data), 'still_large_after_compression_excluded')
        relative = PurePosixPath(name.replace('\\', '/'))
        assert not relative.is_absolute() and '..' not in relative.parts and '.git' not in relative.parts
        if str(relative) in self.names:
            old = self.dest.joinpath(*relative.parts)
            if hashlib.sha256(old.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest():
                self.rows.append({'source': str(source), 'archived': str(relative), 'source_sha256': original_sha, 'bytes': len(data), 'disposition': 'identical_alias'})
                return
            relative = PurePosixPath(str(relative)+'.variant-'+hashlib.sha256(data).hexdigest()[:12])
        self.names.add(str(relative))
        dest = self.dest.joinpath(*relative.parts); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        self.rows.append({'source': str(source), 'archived': str(relative), 'source_sha256': original_sha,
                          'archived_sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'disposition': transform, 'redactions': redactions})
        self.counts['included'] += 1; self.bytes['included'] += len(data)
    def file(self, path, name):
        try:
            size = path.stat().st_size
            reason = self.classify(name, size)
            if reason: return self.omit(path, size, reason)
            low = path.name.lower()
            if low.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
                if size > 256*1024**2:
                    return self.omit(path, size, 'large_container_excluded')
                self.omit(path, size, 'container_replaced_by_filtered_members')
                if low.endswith('.zip'):
                    with zipfile.ZipFile(path) as z:
                        for item in z.infolist():
                            if item.is_dir(): continue
                            reason = self.classify(item.filename, item.file_size)
                            if item.file_size > 64*1024**2: reason = 'large_container_member_excluded'
                            if reason: self.omit(str(path)+'!'+item.filename, item.file_size, reason); continue
                            if '..' in PurePosixPath(item.filename).parts or item.filename.startswith('/'): continue
                            self.add(str(path)+'!'+item.filename, 'container-members/'+name+'/'+item.filename, z.read(item))
                else:
                    with tarfile.open(path, 'r:*') as z:
                        for item in z:
                            if not item.isfile(): continue
                            reason = self.classify(item.name, item.size)
                            if item.size > 64*1024**2: reason = 'large_container_member_excluded'
                            if reason: self.omit(str(path)+'!'+item.name, item.size, reason); continue
                            if '..' in PurePosixPath(item.name).parts or item.name.startswith('/'): continue
                            self.add(str(path)+'!'+item.name, 'container-members/'+name+'/'+item.name, z.extractfile(item).read())
                return
            if low.endswith('.gz'):
                if size > 50*1024**2: return self.omit(path, size, 'large_compressed_data_excluded')
                with gzip.open(path, 'rb') as z: data = z.read(512*1024**2+1)
                if len(data) > 512*1024**2: return self.omit(path, size, 'large_inflated_data_excluded')
                # A gzip-wrapped tensor is still excluded; do not expose it as a log.
                reason = self.classify(name[:-3], len(data))
                if reason: return self.omit(path, size, reason)
                return self.add(path, name[:-3], data)
            self.add(path, name, path.read_bytes())
        except (OSError, ValueError, zipfile.BadZipFile, tarfile.TarError) as exc:
            self.omit(path, 0, 'read_error:'+type(exc).__name__)
    def walk(self, root, prefix, skip=()):
        root = Path(root)
        for curr, dirs, files in os.walk(root, followlinks=False):
            here = Path(curr)
            dirs[:] = [d for d in dirs if d not in CACHE and not (here/d).is_symlink() and str(here/d) not in skip]
            for name in files:
                q = here/name
                if q.is_symlink() or not q.is_file(): continue
                self.file(q, prefix+'/'+q.relative_to(root).as_posix())
    def finish(self, extra=None):
        manifest = self.dest/'archive-manifest.jsonl.gz'
        with gzip.open(manifest, 'wt', encoding='utf-8') as f:
            for row in self.rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
        summary = {'time': datetime.datetime.now().astimezone().isoformat(), 'counts': dict(self.counts), 'bytes': dict(self.bytes), 'extra': extra}
        (self.dest/'archive-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--dest', required=True); p.add_argument('--tar', required=True)
    a = p.parse_args()
    secret = getpass.getpass('Known credential to redact (process only): ')
    arc = Archive(a.dest, secret)
    root = Path(a.root).resolve(); skip = [str(Path(a.dest).parent.resolve())]
    nested = []
    # Existing clones are snapshotted as ordinary source, without .git internals.
    arc.walk(root, 'local-workspace', skip=skip)
    arc.finish({'scope': 'Windows rl workspace, excluding caches, credentials, raw data and large binaries; container members filtered; originals unchanged'})
    assert not Path(a.tar).exists()
    with tarfile.open(a.tar, 'w:gz', compresslevel=3) as z:
        for q in Path(a.dest).rglob('*'):
            if q.is_file(): z.add(q, arcname=q.relative_to(a.dest).as_posix(), recursive=False)
    print('PACKET_BYTES='+str(Path(a.tar).stat().st_size), flush=True)
