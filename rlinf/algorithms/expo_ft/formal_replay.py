"""Physical-step replay for the bounded formal-scale EXPO smoke.

Q samples uniform same-episode real C10 starts; FM samples uniform successful
real H50 starts. Successful clean50 provenance authorizes an explicit offline
last-step reward=1 label: parquet itself contains no measured rewards/dones or
post-last-action observation. Only that terminal next observation is a labelled
absorbing placeholder (last real observation, continuation=0). Online final
observations and every executed action are required to be real.

Episodes remain disk-backed with immutable SHA/identity and per-frame manifests.
A byte-bounded CPU cache retains loaded episodes on demand. No image
augmentation, ratio cap, invented action tail or 256-chunk capacity is used.
"""
from __future__ import annotations

import bisect
from collections import Counter, OrderedDict
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import uuid

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


VERSION = 1
C = 10
H = 50
D = 14
GAMMA = .99
IMAGE_COLUMNS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
                 'observation.images.cam_right_wrist')


def _resident_bytes(value):
    """Count resident tensor/Arrow buffers plus Python objects once per episode."""
    seen, storages = set(), set()
    def size(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        overhead = sys.getsizeof(item)
        if torch.is_tensor(item):
            storage = item.untyped_storage()
            key = (storage.data_ptr(), storage.nbytes())
            extra = 0 if key in storages else storage.nbytes()
            storages.add(key)
            return overhead + extra
        if hasattr(item, 'get_total_buffer_size'):  # Arrow image table.
            return max(overhead, item.get_total_buffer_size())
        if isinstance(item, dict):
            return overhead + sum(size(k) + size(v) for k, v in item.items())
        if isinstance(item, (tuple, list)):
            return overhead + sum(size(v) for v in item)
        return overhead
    return size(value)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return copy.deepcopy(value)


def _digest(value):
    digest = hashlib.sha256()
    def feed(item):
        if torch.is_tensor(item):
            item = item.detach().cpu().contiguous()
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); feed(item[key])
        elif isinstance(item, (tuple, list)):
            for part in item:
                feed(part)
        else:
            digest.update(repr(item).encode())
    feed(value)
    return digest.hexdigest()


def _stat(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_uid]


def _checked(root, path):
    path = Path(path).absolute()
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError('Replay/source path escapes its owned root: ' + str(path))
    if path.exists() and path.stat().st_uid != os.getuid():
        raise ValueError('Replay/source file has a different owner: ' + str(path))
    return path


def _pin(path):
    before = _stat(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        if [os.fstat(stream.fileno()).st_dev, os.fstat(stream.fileno()).st_ino] != before[:2]:
            raise RuntimeError('Replay/source file replaced before hash')
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if _stat(path) != before:
        raise RuntimeError('Replay/source file changed during hash: ' + str(path))
    return dict(sha256=digest.hexdigest(), bytes=before[2], identity=before)


def _verify(path, pin, full=False):
    if _stat(path) != pin['identity']:
        raise RuntimeError('Pinned replay/source identity changed: ' + str(path))
    if full and _pin(path) != pin:
        raise RuntimeError('Pinned replay/source bytes changed: ' + str(path))


def _atomic_json(path, payload):
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(payload, stream, indent=2); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def _observation(obs):
    obs = _cpu(obs)
    main, wrist, state = obs['main_images'], obs['wrist_images'], obs['states']
    if not torch.is_tensor(main) or main.ndim != 4 or main.shape[0] != 1 or main.shape[-1] != 3:
        raise ValueError('Real main RGB must be [1,H,W,3]')
    if not torch.is_tensor(wrist) or wrist.ndim != 5 or wrist.shape[:2] != (1, 2) or wrist.shape[-1] != 3:
        raise ValueError('Real two-wrist RGB must be [1,2,H,W,3]')
    for image in (main, wrist):
        if image.dtype != torch.uint8:
            if not image.is_floating_point() or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Real RGB must be uint8 or finite float[0,1]')
    if not torch.is_tensor(state) or state.ndim != 2 or state.shape[0] != 1 or state.shape[1] != D or not torch.isfinite(state).all():
        raise ValueError('Real canonical joint state must be finite [1,14]')
    prompt = obs['task_descriptions']
    if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], str) or not prompt[0].strip():
        raise ValueError('Real task description is required')
    return obs


def _rgb224(image):
    """Aspect-preserving native resize+black-pad, never geometric augmentation."""
    if image.dtype != torch.uint8:
        image = (image.float() * 255).round().clamp(0, 255).to(torch.uint8)
    if tuple(image.shape[-3:-1]) == (224, 224):
        return image
    original_shape = image.shape
    height, width = original_shape[-3:-1]
    ratio = max(width / 224, height / 224)
    new_height, new_width = max(1, int(height / ratio)), max(1, int(width / ratio))
    pixels = image.reshape(-1, height, width, 3).permute(0, 3, 1, 2).float()
    pixels = F.interpolate(pixels, (new_height, new_width), mode='bilinear', align_corners=False)
    pad_h, pad_w = 224 - new_height, 224 - new_width
    pixels = F.pad(pixels, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    return pixels.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).reshape(*original_shape[:-3], 224, 224, 3)


def _batch_ready(obs):
    result = dict(obs)
    result['main_images'] = _rgb224(obs['main_images'])
    result['wrist_images'] = _rgb224(obs['wrist_images'])
    # Native extraction and the demo loader can differ only in whether this
    # unused optional OpenPI slot is explicit. Keep mixed FM/Q schemas equal.
    result.setdefault('extra_view_images', None)
    return result


def _stack(observations):
    keys = set(observations[0])
    if any(set(obs) != keys for obs in observations):
        raise ValueError('Native observation schema differs in sampled windows')
    result = {}
    for key in keys:
        values = [obs[key] for obs in observations]
        if torch.is_tensor(values[0]):
            result[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            result[key] = sum(values, [])
        elif all(item is None for item in values):
            result[key] = None
        else:
            raise ValueError('Unsupported native observation batch field: ' + key)
    return result


def _actions(value):
    if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
        result = torch.stack([item.detach().cpu().reshape(D) for item in value])
    else:
        result = torch.as_tensor(value).detach().cpu()
        if result.ndim == 3 and result.shape[0] == 1:
            result = result[0]
    result = result.float().contiguous()
    if result.ndim != 2 or result.shape[1] != D or len(result) == 0 or not torch.isfinite(result).all():
        raise ValueError('Real submitted canonical actions must be finite [T,14]')
    return result


def _flags(value, length):
    if isinstance(value, (bool, np.bool_)) or (torch.is_tensor(value) and value.ndim == 0):
        result = torch.zeros(length, dtype=torch.bool); result[-1] = bool(value)
        return result
    result = torch.as_tensor(value, dtype=torch.bool).reshape(-1).cpu()
    if len(result) != length:
        raise ValueError('Per-step done flags must match real action length')
    return result


class FormalReplay:
    """One fixed owned disk pool reused by fresh and new-process resume.

    Caller supplies a CPU-verified successful clean50 dataset. Its no-reward
    parquet gets *offline annotation*, not a claim of measured reward. The
    constructor's complete/revision/episode/frame checks and full source SHA
    pins preserve that reviewed provenance. There is no capacity eviction.
    """
    def __init__(self, root, demo_path, seed=42, cache_limit_bytes=64 * 1024**3):
        import pyarrow.parquet as pq
        self.root = Path(root).absolute(); self.demo_path = Path(demo_path).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _checked(self.root, self.root); _checked(self.demo_path, self.demo_path)
        self.uid = os.getuid(); self.seed = int(seed); self.rng = random.Random(seed)
        if type(cache_limit_bytes) is not int or cache_limit_bytes < 0:
            raise ValueError('Replay cache byte limit must be a nonnegative integer')
        self._cache = OrderedDict()
        self._cache_sizes = {}
        self.cache_limit_bytes = cache_limit_bytes
        self.cache_bytes = self.cache_hits = self.cache_misses = 0
        self.samples_q = self.samples_fm = 0
        self.last_sample = self.last_fm = None
        self.demo_entries = []; self._demo_data = {}
        prepared_path = _checked(self.demo_path, self.demo_path / 'prepared.json')
        tasks_path = _checked(self.demo_path, self.demo_path / 'meta/tasks.jsonl')
        prepared = json.loads(prepared_path.read_text())
        if prepared.get('complete') is not True or not prepared.get('dataset_revision'):
            raise ValueError('Missing reviewed successful-demo complete/revision authority')
        tasks = {}
        for line in tasks_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line); index = int(row['task_index'])
            if index in tasks or not isinstance(row['task'], str) or not row['task'].strip():
                raise ValueError('Task metadata must contain unique real descriptions')
            tasks[index] = row['task']
        paths = sorted(self.demo_path.glob('data/*/episode_*.parquet'))
        if not paths:
            raise ValueError('No successful clean50 parquet episodes')
        pins = {'prepared.json': _pin(prepared_path), 'meta/tasks.jsonl': _pin(tasks_path)}
        total_frames = 0
        for path in paths:
            _checked(self.demo_path, path)
            relative = path.relative_to(self.demo_path).as_posix(); pins[relative] = _pin(path)
            available = set(pq.ParquetFile(path).schema_arrow.names)
            required = set(IMAGE_COLUMNS) | {'observation.state', 'action', 'task_index'}
            if not required.issubset(available):
                raise ValueError('Missing real demo camera/state/action/task columns: ' + relative)
            table = pq.read_table(path, columns=['action', 'observation.state', 'task_index'])
            actions = _actions(table['action'].to_pylist()); count = len(actions)
            states = torch.as_tensor(table['observation.state'].to_pylist(), dtype=torch.float32)
            if states.shape != (count, D) or not torch.isfinite(states).all():
                raise ValueError('Demo canonical state schema differs: ' + relative)
            indices = [int(value) for value in table['task_index'].to_pylist()]
            if len(indices) != count or len(set(indices)) != 1 or indices[0] not in tasks:
                raise ValueError('Demo episode crosses/misses task descriptions: ' + relative)
            rewards = torch.zeros(count); rewards[-1] = 1.0
            term = torch.zeros(count, dtype=torch.bool); term[-1] = True
            trunc = torch.zeros(count, dtype=torch.bool)
            entry = dict(kind='demo', id=relative, path=relative, pin=pins[relative], frames=count,
                         q_count=max(0, count - C + 1), fm_count=max(0, count - H + 1), success=True,
                         task_index=indices[0], prompt=tasks[indices[0]],
                         reward_source='offline_last_step_success1_annotation_from_reviewed_success_demo',
                         terminal_next_obs_source='last_real_preaction_obs_absorbing_placeholder')
            self.demo_entries.append(entry)
            self._demo_data[relative] = dict(actions=actions, states=states, rewards=rewards,
                                             terminated=term, truncated=trunc)
            total_frames += count
        if 'episodes' in prepared and int(prepared['episodes']) != len(paths):
            raise ValueError('Prepared demo episode count differs from pinned parquet inventory')
        if 'frames' in prepared and int(prepared['frames']) != total_frames:
            raise ValueError('Prepared demo frame count differs from actual actions')
        self.contract = dict(version=VERSION, demo_path=str(self.demo_path.resolve()),
            source_pins=pins, dataset_revision=prepared['dataset_revision'], prepared=prepared,
            demo_episode_count=len(paths), demo_frame_count=total_frames,
            chunk=C, fm_horizon=H, canonical_dim=D, discount=GAMMA,
            sampling='uniform real physical-step starts across demo+online; no fixed source ratio',
            q_boundaries='complete same-episode C only; terminal-last retained; any timeout touching excluded',
            demo_labels='reviewed successful-source offline annotation: reward0 then last1/terminated; not parquet measured rewards',
            demo_terminal_next_obs='last real observation, absorbing placeholder only, continuation0',
            online_final_obs='required real post-final-action observation',
            image_batch_geometry='aspect-preserving bilinear resize+black-pad224; no augmentation; raw hashes retained')
        self.contract_sha256 = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()
        online = _checked(self.root, self.root / 'online'); online.mkdir(mode=0o700, exist_ok=True)
        index_path = _checked(self.root, self.root / 'index.json')
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index.get('version') != VERSION or index.get('contract_sha256') != self.contract_sha256:
                raise ValueError('Existing disk replay has a different source/config contract')
            self.root_id = index['root_id']; self.online_entries = index['online_entries']
            uuid.UUID(self.root_id)
            if not isinstance(self.online_entries, list):
                raise ValueError('Disk replay episode index is not a list')
            for entry in self.online_entries:
                self._verify_online(entry, full=True)
            if len({entry['id'] for entry in self.online_entries}) != len(self.online_entries):
                raise ValueError('Disk replay index repeats an episode artifact')
            if len({(type(entry['episode_id']), entry['episode_id']) for entry in self.online_entries}) != len(self.online_entries):
                raise ValueError('Disk replay index repeats an episode ID')
        else:
            self.root_id = str(uuid.uuid4()); self.online_entries = []
            self._write_index()
        self._root_identity = [self.root.stat().st_dev, self.root.stat().st_ino, self.uid]
        self._rebuild()

    def _write_index(self):
        _atomic_json(self.root / 'index.json', dict(version=VERSION, root_id=self.root_id,
                     contract_sha256=self.contract_sha256, online_entries=self.online_entries))

    def _rebuild(self):
        self.entries = self.demo_entries + self.online_entries
        self._q_ends = []; self._fm_ends = []
        q_total = fm_total = 0
        for entry in self.entries:
            q_total += entry['q_count']; fm_total += entry['fm_count']
            self._q_ends.append(q_total); self._fm_ends.append(fm_total)
        self.q_windows, self.fm_windows = q_total, fm_total
        if not self.q_windows or not self.fm_windows:
            raise ValueError('No full real C10 Q or successful H50 FM windows')

    def _verify_online(self, entry, full=False):
        count = entry['frames']
        if (entry.get('kind') != 'online' or type(count) is not int or count < 1 or
                type(entry.get('success')) is not bool or type(entry.get('terminal')) is not bool or
                type(entry.get('timeout')) is not bool or entry['terminal'] == entry['timeout'] or
                entry['success'] != entry['terminal'] or
                entry['q_count'] != max(0, count - C + 1) - int(entry['timeout'] and count >= C) or
                entry['fm_count'] != (max(0, count - H + 1) if entry['success'] else 0)):
            raise ValueError('Online replay eligibility/episode metadata differs')
        for name, pin_key in (('path', 'pin'), ('manifest_path', 'manifest_pin')):
            path = _checked(self.root, self.root / entry[name])
            _verify(path, entry[pin_key], full)
        manifest = json.loads((self.root / entry['manifest_path']).read_text())
        if (manifest.get('version') != VERSION or manifest.get('root_id') != self.root_id or
                manifest.get('episode_id') != entry['episode_id'] or manifest.get('frames') != count or
                manifest.get('success') != entry['success'] or manifest.get('terminal') != entry['terminal'] or
                manifest.get('timeout') != entry['timeout'] or len(manifest.get('frame_sha256', [])) != count):
            raise ValueError('Online per-frame manifest identity differs')

    def append_episode(self, frames, canonical_actions, rewards, terminated, truncated,
                       success, episode_id, final_obs):
        if isinstance(episode_id, bool) or not isinstance(episode_id, (int, str)) or str(episode_id) == '':
            raise ValueError('Online episode_id must be a unique integer/string')
        if any(entry['episode_id'] == episode_id for entry in self.online_entries):
            raise ValueError('Online episode already published; no implicit overwrite')
        actions = _actions(canonical_actions); count = len(actions)
        if len(frames) != count:
            raise ValueError('Exactly one real preaction frame per actual physical action is required')
        observations = [_observation(frame) for frame in frames]
        final = _observation(final_obs)
        reward = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1).cpu()
        term, trunc = _flags(terminated, count), _flags(truncated, count)
        if len(reward) != count or not torch.isfinite(reward).all():
            raise ValueError('Real per-step rewards must match actual actions')
        if (term[:-1] | trunc[:-1]).any() or not bool(term[-1] | trunc[-1]) or bool(term[-1] & trunc[-1]):
            raise ValueError('One complete episode with a single terminal or timeout boundary is required')
        if bool(success) != bool(term[-1]):
            raise ValueError('Native RoboTwin success/termination labels differ')
        if len({obs['task_descriptions'][0] for obs in observations + [final]}) != 1:
            raise ValueError('Online episode prompt changed within an episode')
        token = hashlib.sha256(json.dumps(episode_id).encode()).hexdigest()[:16]
        stem = f'episode-{len(self.online_entries):06d}-{token}'
        path = _checked(self.root, self.root / 'online' / (stem + '.pt'))
        manifest_path = _checked(self.root, self.root / 'online' / (stem + '.json'))
        if path.exists() or manifest_path.exists():
            raise ValueError('Uncommitted episode artifact exists; inspect before reusing its ID')
        frame_hashes = [_digest(obs) for obs in observations]; final_hash = _digest(final)
        payload = dict(version=VERSION, root_id=self.root_id, episode_id=episode_id,
                       observations=observations, final_obs=final, actions=actions, rewards=reward,
                       terminated=term, truncated=trunc, success=bool(success))
        temp = path.with_suffix('.pt.partial')
        if temp.exists():
            raise ValueError('Partial episode exists; inspect its exact artifact')
        with temp.open('xb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temp, 0o600); os.replace(temp, path)
        manifest = dict(version=VERSION, root_id=self.root_id, episode_id=episode_id,
                        frame_sha256=frame_hashes, final_obs_sha256=final_hash,
                        actions_sha256=_digest(actions), rewards_sha256=_digest(reward),
                        flags_sha256=_digest(dict(terminated=term, truncated=trunc)), frames=count,
                        success=bool(success), terminal=bool(term[-1]), timeout=bool(trunc[-1]),
                        final_obs_source='real post-final-action online observation')
        _atomic_json(manifest_path, manifest)
        q_count = max(0, count - C + 1) - int(bool(trunc[-1]) and count >= C)
        entry = dict(kind='online', id=stem, episode_id=episode_id,
                     path=path.relative_to(self.root).as_posix(), pin=_pin(path),
                     manifest_path=manifest_path.relative_to(self.root).as_posix(), manifest_pin=_pin(manifest_path),
                     frames=count, q_count=q_count, fm_count=max(0, count - H + 1) if success else 0,
                     success=bool(success), terminal=bool(term[-1]), timeout=bool(trunc[-1]),
                     reward_source='real online env.step rewards', terminal_next_obs_source='real online final_obs')
        self.online_entries.append(entry); self._write_index(); self._rebuild()
        return copy.deepcopy(entry)

    def _cached(self, entry):
        key = (entry['kind'], entry['id'])
        if key in self._cache:
            root = self.demo_path if entry['kind'] == 'demo' else self.root
            _verify(root / entry['path'], entry['pin'])
            if entry['kind'] != 'demo':
                _verify(self.root / entry['manifest_path'], entry['manifest_pin'])
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        if entry['kind'] == 'demo':
            import pyarrow.parquet as pq
            path = _checked(self.demo_path, self.demo_path / entry['path'])
            _verify(path, entry['pin'])
            value = pq.read_table(path, columns=list(IMAGE_COLUMNS))
        else:
            self._verify_online(entry)
            value = torch.load(self.root / entry['path'], map_location='cpu', weights_only=False)
            manifest = json.loads((self.root / entry['manifest_path']).read_text())
            if (value.get('version') != VERSION or value.get('root_id') != self.root_id or
                    value.get('episode_id') != entry['episode_id'] or value.get('success') != entry['success'] or
                    len(value.get('observations', [])) != entry['frames'] or
                    value['actions'].shape != (entry['frames'], D) or
                    value['rewards'].shape != (entry['frames'],) or
                    value['terminated'].shape != (entry['frames'],) or
                    value['truncated'].shape != (entry['frames'],) or
                    bool(value['terminated'][-1]) != entry['terminal'] or
                    bool(value['truncated'][-1]) != entry['timeout']):
                raise ValueError('Online torch episode schema/identity differs')
            if ([_digest(obs) for obs in value['observations']] != manifest['frame_sha256'] or
                    _digest(value['final_obs']) != manifest['final_obs_sha256'] or
                    _digest(value['actions']) != manifest['actions_sha256'] or
                    _digest(value['rewards']) != manifest['rewards_sha256'] or
                    _digest(dict(terminated=value['terminated'], truncated=value['truncated'])) != manifest['flags_sha256']):
                raise ValueError('Online episode does not match its real per-frame/action manifest')
        size = _resident_bytes(value)
        if size <= self.cache_limit_bytes:
            while self.cache_bytes + size > self.cache_limit_bytes:
                evicted, _ = self._cache.popitem(last=False)
                self.cache_bytes -= self._cache_sizes.pop(evicted)
            self._cache[key] = value
            self._cache_sizes[key] = size
            self.cache_bytes += size
        return value

    def cache_metrics(self):
        return dict(cache_hits=self.cache_hits, cache_misses=self.cache_misses,
                    cache_bytes=self.cache_bytes, cache_entries=len(self._cache),
                    cache_limit_bytes=self.cache_limit_bytes)

    def _demo_observation(self, entry, table, index):
        def decode(column):
            item = table[column][index].as_py()
            content = item.get('bytes') if isinstance(item, dict) else item
            if not isinstance(content, bytes) or not content:
                raise ValueError('Demo image must be real embedded PNG/JPEG bytes: ' + entry['path'])
            with Image.open(io.BytesIO(content)) as image:
                return torch.from_numpy(np.array(image.convert('RGB'))).unsqueeze(0)
        data = self._demo_data[entry['id']]
        return _observation(dict(main_images=decode(IMAGE_COLUMNS[0]),
            wrist_images=torch.stack([decode(IMAGE_COLUMNS[1]), decode(IMAGE_COLUMNS[2])], dim=1),
            states=data['states'][index:index+1], task_descriptions=[entry['prompt']]))

    def _draw(self, size, fm=False):
        if type(size) is not int or size < 1:
            raise ValueError('Replay batch size must be positive integer')
        ends = self._fm_ends if fm else self._q_ends; total = ends[-1]
        refs = []
        for _ in range(size):
            flat = self.rng.randrange(total); entry_index = bisect.bisect_right(ends, flat)
            start = flat - (ends[entry_index-1] if entry_index else 0)
            refs.append((entry_index, start))
        return refs

    def _windows(self, refs, horizon):
        # Group work by episode to avoid reopening parquet/torch per random row;
        # return each result at its sampled position, preserving owned RNG order.
        grouped = {}
        for position, (entry_index, start) in enumerate(refs):
            grouped.setdefault(entry_index, []).append((position, start))
        windows = [None] * len(refs)
        for entry_index, positions in grouped.items():
            entry = self.entries[entry_index]; cached = self._cached(entry)
            data = self._demo_data[entry['id']] if entry['kind'] == 'demo' else cached
            for position, start in positions:
                end = start + horizon
                if end > entry['frames'] or bool(data['truncated'][start:end].any()):
                    raise RuntimeError('An invalid short/timeout window entered the eligible pool')
                if bool(data['terminated'][start:end-1].any()):
                    raise RuntimeError('Window crosses a real episode terminal')
                terminal = bool(data['terminated'][end-1]); placeholder = False
                if entry['kind'] == 'demo':
                    current = self._demo_observation(entry, cached, start)
                    if end < entry['frames']:
                        nxt = self._demo_observation(entry, cached, end)
                    else:
                        if not terminal:
                            raise ValueError('Missing real demo next obs for a nonterminal window')
                        nxt = self._demo_observation(entry, cached, entry['frames'] - 1)
                        placeholder = True
                else:
                    current = cached['observations'][start]
                    nxt = cached['observations'][end] if end < entry['frames'] else cached['final_obs']
                windows[position] = dict(obs=_batch_ready(current), next_obs=_batch_ready(nxt),
                    actions=data['actions'][start:end], rewards=data['rewards'][start:end],
                    terminal=terminal, ref=dict(kind=entry['kind'], episode=entry['id'], start=start, end=end,
                        reward_source=entry['reward_source'], terminal_next_obs_placeholder=placeholder,
                        raw_obs_sha256=_digest(current), raw_next_obs_sha256=_digest(nxt)))
        return windows

    def sample(self, B, backend, device):
        windows = self._windows(self._draw(B), C)
        current = _stack([window['obs'] for window in windows])
        nxt = _stack([window['next_obs'] for window in windows])
        canonical = torch.stack([window['actions'] for window in windows])
        normalized = backend.encode_executed(current, canonical)[..., :D].detach().to(device)
        if normalized.shape != (B, C, D) or not torch.isfinite(normalized).all():
            raise ValueError('Native normalization must preserve actual full C10 actions')
        observations = backend.critic_observation(current)
        next_observations = backend.critic_observation(nxt); next_observations['env_obs'] = nxt
        powers = torch.tensor([GAMMA ** index for index in range(C)])
        rewards = torch.stack([(window['rewards'] * powers).sum() for window in windows]).to(device)
        self.samples_q += B
        self.last_sample = dict(source_counts=dict(Counter(window['ref']['kind'] for window in windows)),
            windows=[window['ref'] for window in windows], eligible_q_windows=self.q_windows,
            terminal_next_obs_placeholders=sum(window['ref']['terminal_next_obs_placeholder'] for window in windows),
            demo_reward_annotation='success-source offline reward0/last1; not parquet recorded rewards')
        return dict(obs=observations, next_obs=next_observations, actions=normalized,
            rewards=rewards, continuations=torch.tensor([0. if window['terminal'] else 1. for window in windows], device=device),
            executed_steps=torch.full((B,), C, dtype=torch.long, device=device),
            valids=torch.ones(B, device=device))

    def sample_fm(self, B):
        windows = self._windows(self._draw(B, fm=True), H)
        observations = [window['obs'] for window in windows]
        actions = torch.stack([window['actions'] for window in windows]).float()
        if actions.shape != (B, H, D) or not torch.isfinite(actions).all():
            raise ValueError('FM must contain exactly 50 real canonical actions')
        counts = dict(Counter(window['ref']['kind'] for window in windows))
        self.samples_fm += B
        self.last_fm = dict(source_counts=counts, windows=[window['ref'] for window in windows],
                            eligible_successful_h50_windows=self.fm_windows, fabricated_action_tails=0)
        return observations, actions, counts

    def state_dict(self):
        return dict(version=VERSION, root=str(self.root.resolve()), root_id=self.root_id,
            root_identity=list(self._root_identity), contract=copy.deepcopy(self.contract), contract_sha256=self.contract_sha256,
            online_entries=copy.deepcopy(self.online_entries), rng=self.rng.getstate(), seed=self.seed,
            samples_q=self.samples_q, samples_fm=self.samples_fm,
            q_windows=self.q_windows, fm_windows=self.fm_windows)

    def load_state_dict(self, state):
        if (state.get('version') != VERSION or state.get('root') != str(self.root.resolve()) or
                state.get('root_id') != self.root_id or state.get('root_identity') != self._root_identity or
                state.get('contract_sha256') != self.contract_sha256 or state.get('contract') != self.contract):
            raise ValueError('Replay root/demo/source/config identity differs; strict restore refused')
        if state['online_entries'] != self.online_entries:
            raise ValueError('Disk pool differs from checkpoint refs; inspect uncommitted/future episodes before restore')
        for entry in state['online_entries']:
            self._verify_online(entry, full=True)
        for relative, pin in self.contract['source_pins'].items():
            _verify(_checked(self.demo_path, self.demo_path / relative), pin)
        self._rebuild()
        if state['q_windows'] != self.q_windows or state['fm_windows'] != self.fm_windows:
            raise ValueError('Replay eligible physical-step window counts differ')
        self.rng.setstate(state['rng']); self.seed = state['seed']
        self.samples_q = int(state['samples_q']); self.samples_fm = int(state['samples_fm'])
        self._cache.clear(); self._cache_sizes.clear()
        self.cache_bytes = self.cache_hits = self.cache_misses = 0
        self.last_sample = self.last_fm = None
