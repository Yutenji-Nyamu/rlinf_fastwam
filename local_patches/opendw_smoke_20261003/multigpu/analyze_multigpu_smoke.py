#!/usr/bin/env python3
"""Read-only N64/G8/R8 OpenDW two-service smoke report (CPU only).

Writes new JSON/Markdown reports outside the owner evidence directory. Does not
connect to Ray/services, change training, install dependencies, or use CUDA.
TensorBoard and full rank-0/rank-1 checkpoint scans are explicit optional reads.
Missing, unfinished, ambiguous, or changed evidence remains unknown.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
from xml.sax.saxutils import escape


KIND = 'opendw_multigpu_smoke_analysis_v1'
SUCCESS = struct.unpack('f', struct.pack('f', 0.9))[0]
PLACEMENT = {'actor': '4,5', 'env': '6,7', 'rollout': '4,5'}


def unknown(reason, **extra):
    return dict(status='unknown', reason=reason, **extra)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summary(values):
    values = sorted(float(v) for v in values if number(v))
    if not values:
        return unknown('No finite observations')
    def percentile(p):
        index = (len(values) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)
    return dict(status='observed', count=len(values), min=values[0], max=values[-1],
                mean=statistics.fmean(values), p50=percentile(.5), p95=percentile(.95))


def timestamp(value, offset=None):
    if number(value):
        return float(value)
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            if offset is None:
                return None
            hours, minutes = map(int, offset[1:].split(':'))
            dt = dt.replace(tzinfo=timezone(timedelta(minutes=(hours * 60 + minutes) * (-1 if offset[0] == '-' else 1))))
        return dt.timestamp()
    except (ValueError, AttributeError, OverflowError):
        return None


def row_time(row, offset=None):
    return timestamp(row.get('timestamp_utc', row.get('time', row.get('wall_time'))), offset)


def read_json(path, issues):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        issues.append(dict(source=str(path), issue=type(exc).__name__))
        return {}


def read_jsonl(path, issues):
    rows = []
    try:
        with Path(path).open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('Expected object')
                    rows.append(dict(row, _line=line_no, _source=str(path)))
                except ValueError:
                    issues.append(dict(source=str(path), line=line_no, issue='invalid_jsonl'))
    except OSError as exc:
        issues.append(dict(source=str(path), issue=type(exc).__name__))
    return rows


def config_read(path, issues):
    try:
        text = Path(path).read_text(encoding='utf-8')
        try:
            return json.loads(text)
        except ValueError:
            import yaml
            return yaml.safe_load(text)
    except Exception as exc:
        issues.append(dict(source=str(path), issue='config_' + type(exc).__name__))
        return {}


def request_envelopes(events, offset=None):
    """The service lock serializes requests; IDs repeat across trials."""
    requests, active = [], None
    for event in events:
        kind = event.get('event')
        if kind == 'request_started':
            if active is not None:
                active['problem'] = 'Superseded before completion'
            active = dict(request_id=event.get('request_id'), pid=event.get('pid'),
                          start=row_time(event, offset), end=None, batch=event.get('batch'),
                          source=event.get('_source'), start_line=event.get('_line'),
                          completed=False, rows=[], row_starts=[], events=[event])
            requests.append(active)
        elif active is not None:
            active['events'].append(event)
            if kind in ('row_started', 'row_completed', 'request_completed'):
                if event.get('request_id') != active['request_id'] or event.get('pid') != active['pid']:
                    active['problem'] = 'Request identity changed inside envelope'
            if kind == 'row_started':
                active['row_starts'].append(event)
            elif kind == 'row_completed':
                active['rows'].append(event)
            elif kind in ('request_completed', 'request_failed'):
                active['end'] = row_time(event, offset)
                active['completed'] = kind == 'request_completed'
                active = None
    return requests


def reviewed_contract(cfg):
    e, a, r = cfg.get('env', {}).get('train', {}), cfg.get('algorithm', {}), cfg.get('runner', {})
    length = e.get('max_episode_steps')
    return (e.get('total_num_envs') == 64 and e.get('rollout_epoch') == 8
            and e.get('chunk') == 32 and e.get('frame_stride') == 4
            and length in (32, 384) and e.get('max_steps_per_rollout_epoch') == length
            and e.get('env_type') == 'opendw_robotwin' and e.get('group_size') == a.get('group_size') == 8
            and e.get('auto_reset') is False and e.get('ignore_terminations') is False
            and e.get('use_rel_reward') is True and e.get('reward_coef') == 1.0
            and e.get('success_reward_threshold') == .9 and a.get('filter_rewards') is True
            and a.get('rewards_lower_bound') == .1 and a.get('rewards_upper_bound') == .9
            and a.get('adv_type') == 'grpo' and a.get('reward_type') == 'chunk_level'
            and a.get('update_epoch') == 2 and cfg.get('cluster', {}).get('component_placement') == PLACEMENT
            and cfg.get('rollout', {}).get('pipeline_stage_num') == 1
            and r.get('max_steps') == r.get('max_epochs') == 1
            and r.get('resume_dir') is None and r.get('ckpt_path') is None)


def reconstruct(cfg, service_requests):
    """Infer R boundaries only after all 32 local trajectories terminate.

    Call counters continue across resets. An epoch with all-success before L has
    no further service calls; it still ends conclusively here. Never use reset_id
    alone as a trajectory key, since reset IDs may repeat within/across epochs.
    """
    if not reviewed_contract(cfg):
        return unknown('Not the reviewed fresh N64/G8/R8/C32/two-env-rank contract')
    if sorted(service_requests) != [0, 1]:
        return unknown('Two independently attributed env ranks are required')
    e = cfg['env']['train']
    max_chunks, groups, terminal_scores, success_count = e['max_episode_steps'] // 32, [], [], 0
    epoch_summaries = []
    for rank in (0, 1):
        requests = service_requests[rank]
        cursor = 0
        for epoch in range(8):
            tracks = [dict(reset_id=None, last=0., chunks=0, success=False) for _ in range(32)]
            alive = set(range(32))
            for chunk in range(1, max_chunks + 1):
                if not alive:
                    break
                if cursor >= len(requests):
                    return unknown('Incomplete rollout evidence', rank=rank, epoch=epoch, chunk=chunk,
                                   complete_rank_epochs=len(epoch_summaries))
                request = requests[cursor]
                match = re.fullmatch(r'seed(-?\d+)-call(\d+)', str(request.get('request_id')))
                if (not match or int(match[1]) != e.get('seed', 0) + rank or int(match[2]) != cursor
                        or not request.get('completed') or request.get('problem')):
                    return unknown('Noncontiguous/mismatched request envelope', rank=rank, call=cursor)
                cursor += 1
                rows = request['rows']
                if request.get('batch') != len(alive) or len(rows) != len(alive):
                    return unknown('Request does not cover all active envs', rank=rank, epoch=epoch, chunk=chunk)
                if sorted(row.get('row', -1) for row in rows) != list(range(len(alive))):
                    return unknown('Duplicate or missing request row index')
                seen, ended = set(), set()
                for row in rows:
                    index, reset = row.get('env_index'), row.get('reset_id')
                    if (not isinstance(index, int) or index not in alive or index in seen
                            or not isinstance(reset, int) or reset < 0
                            or row.get('env_process_index') != rank or row.get('env_process_count') != 2
                            or row.get('global_env_index') != rank * 32 + index):
                        return unknown('Missing/inconsistent rank, env, global-env, or reset identity')
                    seen.add(index)
                    track = tracks[index]
                    if track['reset_id'] is None:
                        track['reset_id'] = reset
                    if track['reset_id'] != reset:
                        return unknown('Reset changed inside one trajectory', rank=rank, epoch=epoch, env=index)
                    last, maximum = row.get('score_last'), row.get('score_max')
                    if not all(number(v) and 0 <= v <= 1 for v in (last, maximum)) or last > maximum + 1e-7:
                        return unknown('Missing/invalid row scores')
                    hit = maximum >= SUCCESS
                    track.update(last=float(last), chunks=chunk, success=hit)
                    if hit or chunk == max_chunks:
                        ended.add(index)
                if seen != alive:
                    return unknown('Active env set incomplete')
                alive -= ended
            if alive:
                return unknown('Epoch ended with unfinished trajectories')
            success_count += sum(t['success'] for t in tracks)
            terminal_scores.extend(t['last'] for t in tracks)
            epoch_summaries.append(dict(env_rank=rank, rollout_epoch=epoch,
                                        terminal_score=summary(t['last'] for t in tracks),
                                        success_trajectories=sum(t['success'] for t in tracks)))
            for start in range(0, 32, 8):
                group = tracks[start:start + 8]
                if len({t['reset_id'] for t in group}) != 1:
                    return unknown('G8 members have different reset states', rank=rank, epoch=epoch, start=start)
                returns = [t['last'] for t in group]
                mean, deviation = statistics.fmean(returns), statistics.stdev(returns)
                retained = None if min(abs(mean - .1), abs(mean - .9)) < 1e-6 else .1 <= mean <= .9
                chunks = sum(t['chunks'] for t in group)
                groups.append(dict(env_rank=rank, rollout_epoch=epoch, local_group=start // 8,
                                   global_env_start=rank * 32 + start, reset_id=group[0]['reset_id'],
                                   returns=returns, mean_return=mean, sample_std=deviation, retained=retained,
                                   nonzero_advantage=bool(retained and deviation > 0),
                                   valid_chunks_before_filter=chunks,
                                   valid_chunks_after_filter=(chunks if retained else 0) if retained is not None else None))
        if cursor != len(requests):
            return unknown('Extra requests after all R8 epochs', rank=rank, used=cursor, observed=len(requests))
    unknown_groups = sum(g['retained'] is None for g in groups)
    valid_chunks = None if unknown_groups else sum(g['valid_chunks_after_filter'] for g in groups)
    return dict(status='reconstructed', trajectories=512, groups=groups, epochs=epoch_summaries,
                retained_groups=sum(g['retained'] is True for g in groups), unknown_groups=unknown_groups,
                nonzero_advantage_groups=sum(g['nonzero_advantage'] for g in groups),
                valid_chunks_before_filter=sum(g['valid_chunks_before_filter'] for g in groups),
                valid_chunks_after_filter=valid_chunks, maximum_chunks=512 * max_chunks,
                valid_chunk_fraction=None if valid_chunks is None else valid_chunks / (512 * max_chunks),
                reward_model_success_trajectories=success_count, terminal_score=summary(terminal_scores),
                unique_group_reset_ids=len({g['reset_id'] for g in groups}),
                basis='Relative frame rewards telescope to last valid score; threshold-any-frame ends at chunk boundary. G8 grouping stays inside rank and epoch. Filter rounding within 1e-6 is unknown; reconstruction is not actual actor-mask telemetry.')


def action_summary(requests):
    records = [row.get('action_telemetry', {}) for req in requests for row in req['row_starts']]
    result = {}
    for kind, denominator in (('actions', 32 * 14), ('state', 14)):
        entries = [row.get('wm_normalization', {}).get(kind, {}) for row in records]
        entries = [x for x in entries if x.get('available') is True and number(x.get('abs_z_gt5_count'))
                   and isinstance(x.get('abs_z_gt5_count_by_dim'), list) and len(x['abs_z_gt5_count_by_dim']) == 14
                   and all(number(v) for v in x['abs_z_gt5_count_by_dim'])]
        count = sum(x['abs_z_gt5_count'] for x in entries)
        result[kind] = (dict(status='observed', rows=len(entries), abs_z_gt5_count=count,
                             abs_z_gt5_fraction=count / (len(entries) * denominator),
                             abs_z_gt5_count_by_dim=[sum(x['abs_z_gt5_count_by_dim'][i] for x in entries) for i in range(14)])
                        if entries else unknown('No valid parsed-stat telemetry'))
    result['joint_command_delta_p95_per_row'] = summary(x.get('adjacent_command_abs_delta', {}).get('joint_p95') for x in records)
    return result


def resources_summary(resources, events):
    peaks, observed, outside, rss = {}, Counter(), [], []
    for row in resources:
        owned = {p.get('pid') for p in row.get('processes', [])}
        rss_values = [p['VmRSS_kib'] * 1024 for p in row.get('processes', []) if number(p.get('VmRSS_kib'))]
        if rss_values:
            rss.append(sum(rss_values))
        contexts = row.get('gpu_processes', [])
        uuid_to_gpu = {p.get('uuid'): p.get('gpu') for p in contexts}
        totals = {}
        for line in str(row.get('compute_memory_csv', '')).splitlines():
            pieces = [p.strip() for p in line.split(',')]
            if len(pieces) != 3 or not pieces[0].isdigit() or int(pieces[0]) not in owned:
                continue
            gpu = uuid_to_gpu.get(pieces[1])
            match = re.fullmatch(r'([0-9.]+)\s*(?:MiB)?', pieces[2])
            if isinstance(gpu, int) and match:
                totals[gpu] = totals.get(gpu, 0) + float(match[1]) * 1024**2
        for gpu, value in totals.items():
            observed[gpu] += 1
            peaks[gpu] = max(peaks.get(gpu, 0), value)
        outside.extend(dict(gpu=p.get('gpu'), pid=p.get('pid'), type=p.get('type')) for p in contexts
                       if p.get('pid') in owned and p.get('gpu') not in (4, 5, 6, 7))
    service_peaks = {}
    for key, rows in events.items():
        service_peaks[key] = {field: max((r[field] for r in rows if number(r.get(field))), default=None)
                              for field in ('VmRSS_bytes', 'Pss_bytes', 'cuda_allocated_bytes', 'cuda_reserved_bytes',
                                            'cuda_max_allocated_bytes', 'cuda_max_reserved_bytes')}
        boards = []
        for row in rows:
            pieces = str(row.get('gpu_nvml', '')).split(',')
            if len(pieces) == 5:
                try:
                    boards.append(float(pieces[2]) * 1024**2)
                except ValueError:
                    pass
        service_peaks[key]['gpu_board_used_sample_peak_bytes'] = max(boards, default=None)
    return dict(managed_compute_peak_bytes_by_gpu={str(g): peaks.get(g) for g in (4, 5, 6, 7)},
                compute_sample_count_by_gpu={str(g): observed[g] for g in (4, 5, 6, 7)},
                managed_rss_sum_peak_bytes=max(rss, default=None), service_peaks=service_peaks,
                owned_outside_gpu4_7_contexts=outside, owner_sample_count=len(resources),
                limits='Sampled peaks can miss transients. Compute memory excludes graphics; outside-context check includes both. Board memory includes all users. RSS sums double-count shared pages; service PSS does not. Torch max values are since most recent service onload reset.')


def tensorboard_metrics(directory, enabled):
    if not enabled:
        return unknown('TensorBoard read not requested')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.util.tensor_util import make_ndarray
        records, errors, sources = [], [], []
        for path in sorted(Path(directory).rglob('events.out.tfevents.*')):
            before = path.stat()
            accumulator = EventAccumulator(str(path), size_guidance={'scalars': 0, 'tensors': 0}).Reload()
            tags = accumulator.Tags()
            for kind in ('scalars', 'tensors'):
                for tag in tags.get(kind, []):
                    if not any(s in tag.lower() for s in ('advantage', 'grad_norm', 'loss', 'mask', 'learning_rate', '/lr')):
                        continue
                    series = accumulator.Scalars(tag) if kind == 'scalars' else accumulator.Tensors(tag)
                    for point in series:
                        value = point.value if kind == 'scalars' else make_ndarray(point.tensor_proto)
                        if kind == 'tensors':
                            if value.size != 1:
                                continue
                            value = value.item()
                        if number(value):
                            records.append(dict(tag=tag, step=point.step, value=float(value)))
                        else:
                            errors.append(dict(tag=tag, step=point.step, issue='nonfinite'))
            digest, after = sha(path), path.stat()
            sources.append(dict(path=str(path), sha256=digest,
                                stable=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)))
        grads = [r['value'] for r in records if r['tag'].endswith('actor/grad_norm')]
        advs = [r['value'] for r in records if re.search(r'advantages?_(max|min|std)$', r['tag'])]
        return dict(status='observed' if records else 'unknown', sources=sources, errors=errors, records=records,
                    nonzero_grad=any(v > 0 for v in grads) if grads else None,
                    nonzero_advantage=any(abs(v) > 0 for v in advs) if advs else None,
                    finite=not errors if records else None,
                    stable=bool(sources) and all(s['stable'] for s in sources),
                    mask_tags=sorted({r['tag'] for r in records if 'mask' in r['tag'].lower()}),
                    limits='Metric records can repeat across backends; counts are not optimizer steps. Missing valid-mask tag is unknown, not zero.')
    except Exception as exc:
        return unknown('TensorBoard read failed: ' + type(exc).__name__ + ': ' + str(exc))


def tensor_summary(value, torch):
    stack, count, finite, nonzero = [value], 0, True, False
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, torch.Tensor):
            if item.device.type != 'cpu':
                raise ValueError('Non-CPU tensor')
            count += 1
            for piece in item.detach().reshape(-1).split(1024 * 1024):
                finite = finite and bool(torch.isfinite(piece).all())
                nonzero = nonzero or bool(torch.count_nonzero(piece))
    return dict(tensors=count, finite=finite, nonzero=nonzero)


def checkpoint_evidence(experiment, enabled):
    if not enabled:
        return unknown('Full two-rank CPU checkpoint scan not requested')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    rows = []
    try:
        import torch
        import numpy as np
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA already initialized')
        torch.set_num_threads(2)
        # NumPy 1.x may expose a partial numpy._core compatibility package.
        # Import the actual module instead of assuming its attribute is loaded.
        import importlib
        multiarray = importlib.import_module('numpy.core.multiarray')
        safe = [(multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
                (multiarray._reconstruct, 'numpy._core.multiarray._reconstruct'),
                np.ndarray, np.dtype, type(np.dtype('uint32'))]
        root = Path(experiment) / 'checkpoints/global_step_1/actor/local_shard_checkpoint'
        expected = [root / f'checkpoint_rank_{rank}.pt' for rank in (0, 1)]
        if sorted(root.glob('checkpoint_rank_*.pt')) != expected:
            raise ValueError('Expected exactly the two configured actor rank shards')
        for rank, path in enumerate(expected):
            before = path.stat()
            with torch.serialization.safe_globals(safe):
                data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if not isinstance(data, dict) or not {'model', 'optimizers'} <= set(data):
                raise ValueError('Unexpected checkpoint schema')
            model, optimizer = tensor_summary(data['model'], torch), tensor_summary(data['optimizers'], torch)
            optimizers = data['optimizers'] if isinstance(data['optimizers'], (list, tuple)) else [data['optimizers']]
            steps, active = [], False
            for opt in optimizers:
                if not isinstance(opt, dict) or not isinstance(opt.get('state'), dict) or not opt.get('param_groups'):
                    raise ValueError('Missing optimizer state/groups')
                for state in opt['state'].values():
                    step = state['step'].item() if isinstance(state.get('step'), torch.Tensor) else state.get('step')
                    if not number(step):
                        raise ValueError('Invalid Adam step')
                    steps.append(float(step))
                for group in opt['param_groups']:
                    lr = group.get('lr')
                    lr = lr.item() if isinstance(lr, torch.Tensor) else lr
                    if not number(lr) or lr < 0:
                        raise ValueError('Invalid optimizer learning rate')
                    if lr > 0:
                        for parameter in group['params']:
                            state = opt['state'].get(parameter, {})
                            m1, m2 = tensor_summary(state.get('exp_avg'), torch), tensor_summary(state.get('exp_avg_sq'), torch)
                            active |= m1['nonzero'] and m2['nonzero']
            digest, after = sha(path), path.stat()
            rows.append(dict(rank=rank, path=str(path), bytes=after.st_size, sha256=digest, model=model,
                             optimizer=optimizer, positive_lr_nonzero_adam_moments=active,
                             adam_step_min=min(steps, default=None), adam_step_max=max(steps, default=None),
                             stable=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)))
            del data, optimizers
        verified = all(r['model']['tensors'] and r['model']['finite'] and r['optimizer']['finite']
                       and r['positive_lr_nonzero_adam_moments'] and (r['adam_step_min'] or 0) > 0 and r['stable'] for r in rows)
        return dict(status='checked', ranks=rows, finite_nonzero_update_state=verified,
                    cuda_initialized=torch.cuda.is_initialized())
    except Exception as exc:
        return unknown('CPU checkpoint read failed: ' + type(exc).__name__ + ': ' + str(exc), ranks=rows)


def decide(binding, reconstruction, tb, cp):
    if not binding or reconstruction.get('status') != 'reconstructed' or reconstruction.get('unknown_groups'):
        return 'unknown'
    if reconstruction['retained_groups'] == 0:
        return 'no_valid_group'
    if reconstruction['nonzero_advantage_groups'] == 0:
        return 'no_advantage_signal'
    if (tb.get('finite') is True and tb.get('stable') is True and tb.get('nonzero_grad') is True
            and tb.get('nonzero_advantage') is True):
        if cp.get('finite_nonzero_update_state') is True and cp.get('cuda_initialized') is False:
            return 'effective_update_verified'
        return 'nonzero_training_signal_observed_checkpoint_unverified'
    return 'unknown'


def analyze(owner, tensorboard=False, checkpoint_scan=False, offset=None):
    owner, issues = Path(owner).resolve(), []
    plan = read_json(owner / 'owner-plan.json', issues)
    resources = read_jsonl(owner / 'resources.jsonl', issues)
    final = read_json(owner / 'final.json', issues) if (owner / 'final.json').is_file() else None
    events_by_service, requests_by_service, input_files = {}, {}, [owner / 'owner-plan.json', owner / 'resources.jsonl']
    services = plan.get('services', [])
    for service in services:
        key = service['key']
        files = sorted((owner / 'services' / key).rglob('service-events.jsonl'))
        if len(files) != 1:
            issues.append(dict(service=key, issue='expected_one_events_file', found=len(files)))
            events_by_service[key], requests_by_service[key] = [], []
            continue
        input_files.extend(files)
        events_by_service[key] = read_jsonl(files[0], issues)
        requests_by_service[key] = request_envelopes(events_by_service[key], offset)
    trials = []
    for row in plan.get('trials', []):
        key, directory = row['key'], owner / row['key']
        cfg = config_read(row.get('config'), issues)
        phase = [r for r in resources if r.get('phase') == key]
        job = read_json(directory / 'ray-job.json', issues) if (directory / 'ray-job.json').exists() else {}
        finished = read_json(directory / 'driver-finished.json', issues) if (directory / 'driver-finished.json').exists() else {}
        result = read_json(directory / 'result.json', issues) if (directory / 'result.json').exists() else {}
        starts = [row_time(r, offset) for r in phase] + [row_time(job, offset)]
        starts = [t for t in starts if t is not None]
        end = row_time(finished, offset)
        if end is None:
            end = max(starts, default=None)
        trials.append(dict(key=key, directory=str(directory), config=cfg, config_source=row.get('config'),
                           config_sha256=row.get('config_sha256'), start=min(starts, default=None), end=end,
                           result=result, finished=finished, phase=phase, requests={}, service_events={}))
    unmatched = []
    for rank, service in enumerate(services):
        key = service['key']
        for request in requests_by_service[key]:
            matches = [t for t in trials if all(v is not None for v in (t['start'], t['end'], request['start'], request['end']))
                       and t['start'] <= request['start'] <= request['end'] <= t['end']]
            if len(matches) != 1 or not request['completed'] or request.get('problem'):
                possible = [t['key'] for t in trials if request['start'] is None or t['start'] is None or t['end'] is None
                            or t['start'] <= request['start'] <= t['end']]
                unmatched.append(dict(service=key, request_id=request['request_id'], start_line=request['start_line'],
                                      possible_trials=possible,
                                      reason=request.get('problem', 'No unique complete time attribution')))
            else:
                matches[0]['requests'].setdefault(rank, []).append(request)
        for trial in trials:
            trial['service_events'][key] = [r for r in events_by_service[key] if trial['start'] is not None and trial['end'] is not None
                                           and row_time(r, offset) is not None and trial['start'] <= row_time(r, offset) <= trial['end']]
    outputs = []
    for trial in trials:
        cfg, key = trial['config'], trial['key']
        reconstruction = reconstruct(cfg, trial['requests'])
        service_stats = []
        for rank, service in enumerate(services):
            requests = trial['requests'].get(rank, [])
            rows = [r for req in requests for r in req['rows']]
            service_stats.append(dict(key=service['key'], physical_gpu=service.get('physical_gpu'), env_rank=rank,
                                      request_count=len(requests), row_count=len(rows),
                                      wm_plus_reward_row_seconds=summary(r.get('seconds') for r in rows),
                                      score_last_per_chunk=summary(r.get('score_last') for r in rows),
                                      score_max_per_chunk=summary(r.get('score_max') for r in rows),
                                      reset_row_histogram=dict(Counter(str(r.get('reset_id')) for r in rows)),
                                      actions=action_summary(requests)))
        log = cfg.get('runner', {}).get('logger', {})
        experiment = Path(log.get('log_path', '/missing')) / log.get('experiment_name', '')
        log_bound = experiment.resolve().is_relative_to(Path(trial['directory']).resolve())
        # This RLinf version stores events below log_path/tensorboard/all,
        # while checkpoints live below log_path/experiment_name/checkpoints.
        metric_root = Path(log.get('log_path', '/missing')) / 'tensorboard'
        metrics_bound = metric_root.resolve().is_relative_to(Path(trial['directory']).resolve())
        tb = tensorboard_metrics(metric_root, tensorboard) if log_bound and metrics_bound else unknown('Logger path outside trial')
        cp = checkpoint_evidence(experiment, checkpoint_scan) if log_bound else unknown('Logger path outside trial')
        config_path = Path(trial['config_source']) if trial['config_source'] else None
        binding = (plan.get('mode') == 'multigpu_smoke' and reviewed_contract(cfg) and log_bound
                   and Path(plan.get('owner_dir', '/missing')).resolve() == owner
                   and [s.get('physical_gpu') for s in services] == [6, 7]
                   and config_path is not None and config_path.is_file() and sha(config_path) == trial['config_sha256']
                   and trial['result'].get('exit_code') == trial['finished'].get('exit_code') == 0
                   and not issues and not any(key in r['possible_trials'] for r in unmatched))
        outputs.append(dict(key=key, exit_code=trial['result'].get('exit_code', trial['finished'].get('exit_code')),
                            trial_seconds=trial['result'].get('seconds'), time_window=[trial['start'], trial['end']],
                            window_note='Running trials exclude tail beyond latest owner resource sample.',
                            binding_verified=binding, services=service_stats, reconstruction=reconstruction,
                            resources=resources_summary(trial['phase'], trial['service_events']),
                            tensorboard=tb, checkpoint=cp, learning_status=decide(binding, reconstruction, tb, cp)))
        if config_path is not None:
            input_files.append(config_path)
    if (owner / 'final.json').exists():
        input_files.append(owner / 'final.json')
    return dict(analysis_kind=KIND, owner=str(owner), naive_offset_assumption=offset, trials=outputs,
                owner_final=final, unassigned_requests=unmatched, issues=issues,
                source_sha256={str(p): sha(p) for p in input_files if p.is_file()},
                limits='Reward-model scores are not native success. Effective-update status requires retained varying-return G8 groups, finite nonzero advantage/grad logs, and complete finite two-rank optimizer checkpoint evidence. Exit0, weight decay or checkpoint presence alone never passes. Pure WM timing is not separately recorded.')


def markdown(report):
    def fmt(value, scale=1):
        return f'{value / scale:.3g}' if number(value) else 'unknown'
    lines = ['# OpenDW N64/R8 双服务 smoke', '', '奖励模型判分不等于原生任务成功率。', '',
             '|试验|耗时s|有效G8组|有效chunk比例|训练证据|', '|---|---:|---:|---:|---|']
    for trial in report['trials']:
        rec = trial['reconstruction']
        groups = f"{rec['retained_groups']}/64" if rec.get('status') == 'reconstructed' else 'unknown'
        lines.append(f"|{trial['key']}|{fmt(trial['trial_seconds'])}|{groups}|{fmt(rec.get('valid_chunk_fraction'))}|{trial['learning_status']}|")
    for trial in report['trials']:
        lines.extend(['', f"## {trial['key']}", '', '|服务/卡|完成行|WM+RM均值/p95秒|末分数均值|动作z裁剪比例|', '|---|---:|---:|---:|---:|'])
        for s in trial['services']:
            timing, actions = s['wm_plus_reward_row_seconds'], s['actions']['actions']
            lines.append(f"|{s['key']}/{s['physical_gpu']}|{s['row_count']}|{fmt(timing.get('mean'))}/{fmt(timing.get('p95'))}|{fmt(s['score_last_per_chunk'].get('mean'))}|{fmt(actions.get('abs_z_gt5_fraction'))}|")
        peaks = trial['resources']['managed_compute_peak_bytes_by_gpu']
        lines.extend(['', '- 我方计算显存采样峰值：' + '；'.join(f'GPU{k} {fmt(v, 1024**3)} GiB' for k, v in peaks.items()) + '。',
                      f"- 有效学习判定：{trial['learning_status']}；配置与完成回执绑定：{trial['binding_verified']}。"])
        if trial['reconstruction']['status'] == 'unknown':
            lines.append('- 重建不足：' + trial['reconstruction']['reason'] + '。')
    lines.extend(['', f"未归属请求 {len(report['unassigned_requests'])}，输入问题 {len(report['issues'])}；细节见 JSON。",
                  '显存为采样峰值；图形上下文另列 JSON。RSS求和重复计算共享页。训练信号与真实环境效果分开判断。', ''])
    return '\n'.join(lines)


def svg(report):
    """Small dependency-free sampled-memory/latency chart; unknown is explicit."""
    trials = report['trials']
    width, height = 960, 110 + 210 * len(trials)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#f8fafc"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#0f172a} .small{font-size:12px;fill:#475569}</style>',
           '<text x="24" y="32" font-size="21" font-weight="bold">OpenDW N64 / G8 / R8 smoke</text>',
           '<text x="24" y="56" class="small">Sampled owned compute memory; WM row latency includes reward inference. Unknown means unavailable.</text>']
    peak_values = [v / 1024**3 for t in trials for v in t['resources']['managed_compute_peak_bytes_by_gpu'].values() if number(v)]
    latency_values = [s['wm_plus_reward_row_seconds'].get('p95') for t in trials for s in t['services']]
    max_peak = max(peak_values + [1.]) * 1.1
    max_latency = max([v for v in latency_values if number(v)] + [1.]) * 1.1
    for index, trial in enumerate(trials):
        y = 91 + index * 210
        out.append(f'<text x="24" y="{y}" font-size="17" font-weight="bold">{escape(trial["key"])}</text>')
        out.append(f'<text x="24" y="{y+20}" class="small">{escape(trial["learning_status"])}</text>')
        for i, (gpu, value) in enumerate(trial['resources']['managed_compute_peak_bytes_by_gpu'].items()):
            yy = y + 50 + 28 * i
            known = number(value)
            label = f'{value / 1024**3:.2f} GiB' if known else 'unknown'
            bar_width = 245 * (value / 1024**3) / max_peak if known else 0
            out.extend([f'<text x="24" y="{yy+13}" class="small">GPU {escape(gpu)}</text>',
                        f'<rect x="82" y="{yy}" width="245" height="18" rx="3" fill="#e2e8f0"/>',
                        f'<rect x="82" y="{yy}" width="{bar_width:.2f}" height="18" rx="3" fill="#0284c7"/>',
                        f'<text x="340" y="{yy+13}" class="small">{label}</text>'])
        for i, service in enumerate(trial['services']):
            timing = service['wm_plus_reward_row_seconds']
            yy = y + 50 + 52 * i
            p95, mean = timing.get('p95'), timing.get('mean')
            label = f'mean {mean:.2f}s / p95 {p95:.2f}s' if number(mean) and number(p95) else 'unknown'
            bar_width = 170 * p95 / max_latency if number(p95) else 0
            out.extend([f'<text x="470" y="{yy+13}" class="small">{escape(service["key"])}</text>',
                        f'<rect x="535" y="{yy}" width="170" height="18" rx="3" fill="#e2e8f0"/>',
                        f'<rect x="535" y="{yy}" width="{bar_width:.2f}" height="18" rx="3" fill="#059669"/>',
                        f'<text x="720" y="{yy+13}" class="small">{label}</text>',
                        f'<text x="535" y="{yy+36}" class="small">{service["row_count"]} completed rows</text>'])
    out.extend([f'<text x="24" y="{height-20}" class="small">Resource peaks may miss transients. This chart does not measure native-task success.</text>', '</svg>'])
    return '\n'.join(out) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner', type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--checkpoint-scan', action='store_true', help='CPU mmap and full finite/hash scan of both actor shards; potentially minutes')
    parser.add_argument('--naive-offset', help='Explicit ±HH:MM for naive owner timestamps')
    args = parser.parse_args()
    if args.naive_offset and not re.fullmatch(r'[+-](?:0\d|1[0-4]):[0-5]\d', args.naive_offset):
        parser.error('Invalid --naive-offset')
    paths = [args.output_prefix.with_suffix(ext).resolve() for ext in ('.json', '.md', '.svg')]
    if any(p.is_relative_to(args.owner.resolve()) or p.exists() for p in paths):
        parser.error('Use new report paths outside the owner evidence directory')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    report = analyze(args.owner, args.tensorboard, args.checkpoint_scan, args.naive_offset)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for path, text in zip(paths, (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', markdown(report), svg(report))):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(json.dumps(dict(json=str(paths[0]), markdown=str(paths[1]), svg=str(paths[2]),
                          trials={t['key']: t['learning_status'] for t in report['trials']})))


if __name__ == '__main__':
    main()
