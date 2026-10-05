"""Score one small labeled frame batch with the published click_bell classifier.

Inputs are CPU NPZ arrays images uint8[N,H,W,3], instructions Unicode[N],
labels int[N] (0/1, or -1 unknown) and optional sample_ids Unicode[N]. Labels
must come from native evidence, not another RM. No training or service starts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('samples', 'checkpoint', 't5-path', 'reward-module-dir', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--physical-gpu', type=int, choices=range(4, 8), required=True)
    p.add_argument('--batch-size', type=int, choices=(4, 8, 16), default=16)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(a.physical_gpu):
        raise RuntimeError('Caller must bind exactly the authorized physical GPU')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Caller must preserve the existing physical GPU mapping')
    if a.output.exists():
        raise FileExistsError(a.output)
    import numpy as np
    import torch
    sys.path.insert(0, str(a.reward_module_dir))
    from opendw_reward import RoboTwinT5Reward
    with np.load(a.samples, allow_pickle=False) as data:
        images, instructions, labels = [data[k].copy() for k in ('images', 'instructions', 'labels')]
        sample_ids = data['sample_ids'].tolist() if 'sample_ids' in data.files else [str(i) for i in range(len(images))]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError('Require uint8 main camera RGB frames')
    if len(images) > 128 or len(images) == 0 or instructions.shape != (len(images),) or labels.shape != (len(images),) or len(sample_ids) != len(images):
        raise ValueError('Require 1..128 matched frames, instructions, labels and IDs')
    if instructions.dtype.kind not in 'US' or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Invalid instruction or native labels')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Require exactly one visible GPU')
    started = time.monotonic()
    model = RoboTwinT5Reward(a.checkpoint, a.t5_path).to('cuda:0')
    torch.cuda.synchronize()
    loaded = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    scores = []
    with torch.inference_mode():
        for begin in range(0, len(images), a.batch_size):
            values = model.compute_reward(torch.from_numpy(images[begin:begin+a.batch_size]), instructions[begin:begin+a.batch_size].tolist())
            scores.extend(values.detach().float().cpu().tolist())
    torch.cuda.synchronize()
    elapsed = time.monotonic() - loaded
    scores = np.asarray(scores)
    if scores.shape != labels.shape or not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise RuntimeError('Classifier returned invalid probabilities')
    report = {'schema': 1, 'checkpoint': str(a.checkpoint), 'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              'samples': str(a.samples), 'samples_sha256': hashlib.sha256(a.samples.read_bytes()).hexdigest(),
              'strict_load': True, 'parameter_count': sum(v.numel() for v in model.parameters()),
              'physical_gpu': a.physical_gpu, 'batch_size': a.batch_size, 'count': len(images), 'load_seconds': loaded-started,
              'inference_seconds': elapsed, 'frames_per_second': len(images)/elapsed,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'thresholds': {}, 'rows': []}
    for threshold in (0.5, 0.9):
        pred = scores >= threshold
        report['thresholds'][str(threshold)] = {
            'true_positive': int(np.sum(pred & (labels == 1))), 'false_negative': int(np.sum(~pred & (labels == 1))),
            'false_positive': int(np.sum(pred & (labels == 0))), 'true_negative': int(np.sum(~pred & (labels == 0))),
            'unknown_label_count': int(np.sum(labels == -1)),
        }
    for i, value in enumerate(scores):
        report['rows'].append({'sample_id': sample_ids[i], 'native_label': int(labels[i]), 'score': float(value), 'success_at_0_9': bool(value >= .9)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
