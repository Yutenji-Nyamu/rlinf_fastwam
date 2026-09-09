"""Freeze one scale using historical training labels, without collecting or fitting."""
import collections
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-online-bc-attena-fk')
ST = Path('/data/chenyiteng/results/server-maintenance-20260909/bc-attena-fk')
sys.path.insert(0, str(ROOT))
from rlinf.algorithms.online_bc_attena_fk import Geometry, OnlineBCAttenaFK, motion_to_magnitude, motion_to_weights

assert os.getuid() == 1003
torch.set_num_threads(1)
asset = ROOT / 'examples/embodiment/config/bc_attena'
geometry = Geometry.from_file(asset / 'aloha_geometry.json')
run = Path('/home/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-pillbottle-bc4x1-b1024-u5-m10-seed42-eval8x4-gpu6-formal100-20260908-v1')
paths = list(run.glob('*/checkpoints/global_step_100/actor/online_bc/rank_0/success_replay.pt'))
assert len(paths) == 1
path = paths[0]
before = path.stat()
data = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
counts = collections.Counter(tuple(r['episode_id'].tolist()) for r in data['records'])
records = [r for r in data['records'] if counts[tuple(r['episode_id'].tolist())] <= 3]
numeric_sha = hashlib.sha256()
features = []
masks = []
for record in records:
    assert record['action_valid_mask'].all()
    for key in ('episode_id', 'action', 'observation/state', 'action_valid_mask'):
        numeric_sha.update(record[key].numpy().tobytes())
    prepared = geometry.prepare_record(record)
    features.append(prepared['attena_fk_motion'])
    masks.append(prepared['action_valid_mask'])
motion = torch.stack(features)
mask = torch.stack(masks)
u = motion_to_magnitude(motion, ell=0.1)
positive = u[u > 1e-8]
assert positive.numel() > 0
c = float(np.median(positive.numpy()))
calibration = {
    'schema_version': 1, 'geometry_id': geometry.geometry_id,
    'aggregation': 'sum_arm_l2', 'ell': 0.1, 'c_m': c,
    'positive_tolerance_m': 1e-8,
    'reference': {
        'source': str(path), 'source_bytes': before.st_size,
        'source_mtime_ns': before.st_mtime_ns,
        'selected_numeric_sha256': numeric_sha.hexdigest(),
        'source_code_head': '01d770db3988da7862454e97434d4ff08f726fa2',
        'max_success_chunks': 3, 'episodes': sum(n <= 3 for n in counts.values()),
        'queries': len(records), 'valid_positions': int(u.numel()),
        'positive_positions': int(positive.numel()),
        'estimator': 'median of strictly positive >1e-8m magnitudes from stored float32 FK features',
        'data_usage': 'Historical training preprocessing only; reference records do not enter new replay',
        'new_environment_episodes': 0, 'new_optimizer_steps': 0,
    },
}
method = OnlineBCAttenaFK(geometry, calibration)
w, metrics = motion_to_weights(motion, c, valid_mask=mask)
assert 0.5 < metrics['attena_fk/weight_mean'] < 2
assert metrics['attena_fk/weight_std'] > 0.01
after = path.stat()
assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
target = asset / 'clean4u5_sum_calibration.json'
with target.open('x') as stream:
    json.dump(calibration, stream, indent=2, allow_nan=False)
receipt = {
    'time': datetime.datetime.now().astimezone().isoformat(), 'passed': True,
    'calibration': calibration, 'weight_metrics': metrics,
    'motion_quantiles_m': np.quantile(u.numpy(), [0, .1, .5, .9, 1]).tolist(),
    'geometry_id': geometry.geometry_id, 'method_state': method.method_state(),
    'calibration_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'read_only_reference_unchanged': True,
}
(ST / 'calibration-receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
