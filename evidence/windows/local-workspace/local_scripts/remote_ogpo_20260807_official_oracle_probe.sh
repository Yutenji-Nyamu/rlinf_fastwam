#!/usr/bin/env bash
set -u

/usr/bin/python3 - <<'PY'
from urllib.request import Request, urlopen

commit = '0b3be413cde766a41257c6b19c0c2b06393a557f'
base = f'https://raw.githubusercontent.com/simchowitzlabpublic/OGPO_public/{commit}/'
targets = {
    'ogpo/agents/modules/pg_helper.py': [(115, 420)],
    'ogpo/agents/ogpo.py': [(350, 475), (1810, 1870), (2070, 2160), (2220, 2290)],
    'ogpo/agents/modules/q_helper.py': [(1, 70), (320, 360)],
    'ogpo/utils/datasets.py': [(170, 210)],
    'ogpo/configs/algos/common.yaml': None,
    'ogpo/configs/algos/ogpo.yaml': None,
    'scripts/ogpo/square_image_paligemma.sh': None,
}

for path, ranges in targets.items():
    print(f'=== {path} ===')
    try:
        request = Request(base + path, headers={'User-Agent': 'Codex-readonly-source-audit'})
        with urlopen(request, timeout=30) as response:
            lines = response.read().decode('utf-8').splitlines()
    except Exception as exc:
        print(f'FETCH_ERROR {type(exc).__name__}: {exc}')
        continue
    selected = ranges or [(1, len(lines))]
    for start, end in selected:
        for number in range(start, min(end, len(lines)) + 1):
            print(f'{number:4}: {lines[number - 1]}')
PY
