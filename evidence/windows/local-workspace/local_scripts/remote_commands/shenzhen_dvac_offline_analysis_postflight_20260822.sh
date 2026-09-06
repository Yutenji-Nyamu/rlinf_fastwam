#!/usr/bin/env bash
set -euo pipefail

OUTPUT=/data/chenyiteng/results/dvac-observation/analysis/sz-dvac-pi0-fixed64-fastwam-adjust-p1-v1
PYTHON=/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/bin/python

test -d "$OUTPUT"
printf 'OUTPUT_TOTAL\n'
du -sb "$OUTPUT"
printf 'OUTPUT_FILE_COUNT=%s\n' "$(find "$OUTPUT" -type f | wc -l)"
printf 'EXTENSION_COUNTS_AND_BYTES\n'
find "$OUTPUT" -type f -printf '%f\t%s\n' | "$PYTHON" -c '
import collections, pathlib, sys
rows = [line.rstrip("\n").split("\t") for line in sys.stdin if line.strip()]
stats = collections.defaultdict(lambda: [0, 0])
for name, size in rows:
    suffix = pathlib.Path(name).suffix.lower() or "[none]"
    stats[suffix][0] += 1
    stats[suffix][1] += int(size)
for suffix in sorted(stats):
    print(f"{suffix}\tfiles={stats[suffix][0]}\tbytes={stats[suffix][1]}")
'

printf 'TOP_LEVEL_FILES\n'
find "$OUTPUT" -maxdepth 1 -type f -printf '%f\t%s\n' | sort

"$PYTHON" - "$OUTPUT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

root = Path(sys.argv[1])
summary = json.loads((root / 'analysis_summary.json').read_text(encoding='utf-8'))
print('SUMMARY_COUNTS=' + json.dumps(summary['counts'], sort_keys=True))
print('IDENTITY_CHECKS=' + json.dumps(summary['identity_checks'], sort_keys=True))
assert summary['counts']['groups'] == 2
assert max(abs(float(value)) for value in summary['identity_checks'].values()) < 1e-10
for group in summary['groups']:
    print('GROUP=' + json.dumps({
        key: group[key]
        for key in ('group_id', 'policy', 'task', 'run_id', 'cohort_id', 'queries', 'episodes', 'success_episodes', 'failure_episodes')
    }, sort_keys=True))
for source in summary['source_inventory']:
    keep = {
        key: source.get(key)
        for key in (
            'source_kind', 'run_id', 'queries', 'episodes', 'success_episodes',
            'action_num_train_timesteps', 'action_num_train_timesteps_source',
            'official_seed_map_used', 'max_abs_final_chain_vs_model_action'
        )
    }
    print('SOURCE=' + json.dumps(keep, sort_keys=True))

csv_names = sorted(path.name for path in root.glob('*.csv'))
for name in csv_names:
    path = root / name
    frame = pd.read_csv(path)
    print(f'CSV={name}\trows={len(frame)}\tcolumns={len(frame.columns)}\tbytes={path.stat().st_size}')

query = pd.read_csv(root / 'query_metrics.csv')
horizon = pd.read_csv(root / 'query_horizon.csv')
episode = pd.read_csv(root / 'episode_metrics.csv')
episode_all = pd.read_csv(root / 'episode_metrics_all_queries.csv')
action = pd.read_csv(root / 'fastwam_action_frame_metrics.csv')
assert set(query['run_id']) == {
    'pi0-adjust_bottle-fixed64-800baf80-v1',
    'fastwam-adjust_bottle-p1-16ep-c63dc9b5-v2',
}
assert query['group_id'].nunique() == 2
assert int((~query['success_before'].astype(bool)).sum()) == int(episode['queries'].sum()) == 294
assert int(episode_all['queries'].sum()) == len(query) == 336
assert int(action['h'].max()) < 24
assert int(action['terminal_success_action'].astype(bool).sum()) == 16
assert set(horizon.loc[(horizon['source_kind'] == 'fastwam') & ~horizon['executed_flag'].astype(bool), 'phase_coarse']) == {'UNLABELED'}
print('QUERY_BY_SOURCE=' + json.dumps(query.groupby('source_kind').size().astype(int).to_dict(), sort_keys=True))
print('POST_SUCCESS_BY_SOURCE=' + json.dumps(query.groupby('source_kind')['success_before'].sum().astype(int).to_dict(), sort_keys=True))
print('ACTION_CONTRACT=' + json.dumps({
    'rows': len(action),
    'max_h': int(action['h'].max()),
    'terminal_success_actions': int(action['terminal_success_action'].astype(bool).sum()),
}, sort_keys=True))

storyboard_index = pd.read_csv(root / 'storyboard_index.csv')
print('STORYBOARD_STATUS=' + json.dumps(storyboard_index['status'].value_counts().astype(int).to_dict(), sort_keys=True))
pngs = sorted(root.rglob('*.png'))
storyboards = sorted((root / 'storyboards').glob('*.png'))
query_strips = sorted((root / 'query_frame_strips').glob('*.png'))
assert pngs and storyboards and query_strips
total_pixels = 0
for path in pngs:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        total_pixels += image.width * image.height
print(f'PNG_VERIFY_OK files={len(pngs)} storyboards={len(storyboards)} query_strips={len(query_strips)} total_pixels={total_pixels}')
print('POSTFLIGHT_ASSERTIONS_OK')
PY
